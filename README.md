# ML DRL Scheduler — a PPO-learned MU-MIMO downlink scheduler

This repository contains a single-cell downlink MU-MIMO scheduling simulator (32 BS antennas, 1-Rx UEs, 8 RBGs x 4 spatial layers per 0.5 ms slot, TR 38.901 UMi channels via Sionna, RZF precoding computed **only** from fed-back, quantized, possibly stale CSI) and a PPO agent trained to schedule on it. The central finding, established over four progressively harder training campaigns (Run1–Run4): **a single learned policy matches the regime-optimal tuned heuristic in every fixed regime, and decisively beats *all* fixed heuristics — including a per-seed oracle that picks the best heuristic after the fact — whenever conditions are mixed** (mixed load, mixed speed, mixed arrival rate). Fixed heuristics each fail somewhere; the learned policy performs regime selection internally (e.g. modulating MU-MIMO depth with load and CSI reliability), and this skill transfers zero-shot to environments it never trained on (multi-packet queues, perfect CSI).

## Headline results

All comparisons are paired (same seeds), on held-out seeds, against the strongest fixed heuristic in each environment (naming: `gate+metric`, e.g. `SUS+CQI`). Full protocol and per-run details: [docs/RUNS.md](docs/RUNS.md).

| # | Result | Environment | Margin |
|---|--------|-------------|--------|
| 1 | Run2 HardMain: PPO vs best fixed heuristic | K=16, 30 km/h, stale quantized CSI (p_csi = 0.2) | **+50 %** (and +3.6 % over a genie-CSI baseline) |
| 2 | Run3 MixedLoad_L2 vs **per-seed oracle envelope** (per-seed best of the two strongest heuristics, SU+CQI and SUS+CQI@0.8) | n_active ~ U[16, 32] per episode | **+6.35 %** (n = 20, significant) |
| 3 | Run3 MixedSpeed_L2b vs per-seed oracle envelope | UE speed ~ U(5, 30) km/h per episode | **+6.88 %** (n = 20; ~+9–10 % vs best realizable hybrid switch) |
| 4 | Run3 DeadlineScarcity vs SU+CQI | K=16, deadlines [2, 6] slots | **+2.62 %** (n = 60, CI [+82, +258]) |
| 5 | Run4 QueueMixedArrival vs SUS+CQI | per-UE FIFO queues, p_arrival ~ U(0.15, 0.40) per episode | **+21.4 %** (best queue-era result) |
| 6 | Run4 transfer: Run3 policy zero-shot on the queue env (queue-blind observations) | multi-packet queue env | **+11.4 %**, 10/10 seeds |
| 7 | Genie zero-shot: Run3 policy under **perfect CSI** (never trained on it) | pmi_mode=genie, p_csi = 1.0, 20-seed paired | **+8.1 %** vs SUS+CQI; **156 W – 4 L / 160** vs all fixed heuristics |
| 8 | Regime lock-in of fixed-condition training | K16-trained policy evaluated at K32 | **−17 %** — mixed-condition training is what creates adaptivity |
| 9 | OOD robustness: QueueMixedArrival policy at fixed arrival rates it never trained on | trained on p ~ U(0.15, 0.40); evaluated at fixed p = 0.10 … 0.50, 8 paired seeds each | tie (−0.4 %) at OOD-low p = 0.10; **+44.2 %** at p = 0.45 and **+135.7 %** at p = 0.50 (vs per-seed best baseline; % inflated by small denominators — absolute 4236 vs 2774) — no overfitting cliff |

Single-regime results are honest: at the easiest operating point (Run3 Uniform10) PPO merely **ties** the best tuned heuristic (+0.36 %, n = 40), and in the forced-MU K=32 scarcity regime it wins small but real (**+1.71 %**, n = 40, vs SUS+CQI@0.8). The wins concentrate exactly where no fixed rule can follow the conditions.

Status note (2026-07-10): `QueueHighLoad`, `QueueMixedArrival`, `GenieL2b` and `GenieFineTune` were still training on the status date; their margins quote the banked `best.pt` so far. Per-run statuses: [docs/RUNS.md](docs/RUNS.md).

## Repository map

### Simulation & learning code (repo root)

| File | Role |
|------|------|
| `config.py` | Single `Config` dataclass: every system/PPO parameter, plus presets (`debug`/`main`/`hard_*`/`hetero`/`queue`) |
| `channel.py` | TR 38.901 UMi channel generation via `sionna.phy` (TensorFlow); per-episode caching, exact velocity rescaling |
| `codebook.py` | PMI codebooks: simplified Type-II-like 56-bit sparse codebook (default) and random unit-norm codebook |
| `csi.py` | CSI feedback: quantized direction + SU-SE CQI, Bernoulli feedback opportunities (`p_csi`), staleness buffer |
| `phy.py` | RZF precoding from reconstructed ĥ (`rzf_precoder`), per-RBG SINR (`compute_slot_sinr`), Shannon SE, noise calibration |
| `transmission.py` | Per-RBG transmission units (CBG-like), ideal-IR mutual-information accumulation, HARQ retx with preemption |
| `traffic.py` | Bernoulli arrivals with deadlines; Run4 per-UE FIFO queue (size 8); `queue_size=1` is bit-exact Run3 legacy |
| `env.py` | `SchedulerEnv`: slot dynamics, observation construction, reward; action = [8 x 4] allocation matrix |
| `baselines.py` | Fixed heuristic grid: {SUS-gated MU, SU} x {CQI, Deadline-PF, PF, Random, MaxWeight, EDF} — schedulers never see the true channel |
| `metrics.py` | Evaluation loop + per-episode metrics (reward, throughput, SINR, completion, deadline miss, retx drop, JFI over **active** users); the full standard-9 set — adding total failure and MU depth — is assembled by `env_episode_metrics` in `train_phase2.py` and the analysis scripts |
| `policy.py` | `ActorCritic`: shared per-(UE, RBG) encoder, autoregressive layer-major actor over 32 sub-actions, critic (v2 adds structured value features) |
| `ppo.py` | Rollout collection, GAE, clipped PPO with return normalization, separate actor/critic grad clipping, KL early-stop guard v2 |
| `run_phase1.py` | Phase-1 sanity driver: env/PHY checks + baseline comparison, no RL |
| `train_phase2.py` | Training entry point: modes, resume, warm start (`--init_from`), patience harvest, atomic checkpointing |
| `eval_phase2.py` | Evaluate a checkpoint against all baselines (paired) |

### Documentation

- **[docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md)** — how the simulation is constructed, layer by layer, with code references.
- **[docs/RUNS.md](docs/RUNS.md)** — every training run: configuration, purpose, result, lesson.
- `PHASE1_SUMMARY.md`, `PHASE2_SUMMARY.md` — original design/spec documents for the env and the PPO agent.
- `STANDARDS_AUDIT_20260707.md` — 3GPP-conformance audit; read this for the honest list of idealizations.

### Run artifacts (`Run1/` … `Run4/`)

Each campaign folder holds one subfolder per training run (e.g. `Run3/MixedSpeed_L2b/`, `Run4/QueueMixedArrival/`) containing `config.json` (exact configuration), `ckpt/` (`best.pt` = banked best checkpoint, `latest.pt`), `csv_logs/` (per-update training + eval metrics), `tb_logs/` (TensorBoard), and a `*_run.log` (under `Run1/logs/` and `Run2/logs/` for Run1–2, e.g. `Run2/logs/Run2_HardMain_run.log`; inside the run folder for Run3, e.g. `Run3/MixedSpeed_L2b/MixedSpeed_L2b_run.log`; at the campaign level for Run4, e.g. `Run4/QueueMain_run.log`). Additionally:

- `Run1/`, `Run2/` — `README.md` per campaign; Run2 includes `HardDebug_confirm` (pre-flight) and `code_backup/`.
- `Run3/` — `CHANGES_vs_Run2.md` inside each run folder, `BASELINE_AUDIT_2026-07-02.md` (official 2x4 baseline grid + SUS-threshold tuning), `_analysis_20260702/`, `_watchdog.sh`, per-run `_wrap_*.sh` auto-resume wrappers.
- `Run4/` — `RUN4_CHANGES_vs_Run3.md` (queue model, obs +3/UE, 2x6 baseline grid, SUS re-sweep to 0.75), `_analysis/` (`csi_fidelity.py`, genie zero-shot, paired-CSI and OOD-arrival CSVs), `_calib_20260706/` (queue-era calibration CSVs: `sus_threshold_sweep.csv` — the SUS 0.75 re-sweep — and `l2b_zeroshot_queue.csv` — the +11.4% zero-shot probe; the p_arrival ρ-recalibration measurements are recorded in `RUN4_CHANGES_vs_Run3.md`), `_init_from_L2b/`, `_wrap_*.sh`.

## Quickstart

Dependencies (the channel stack is TensorFlow-based, the agent is PyTorch; the channel code uses the `sionna.phy` namespace introduced in Sionna 1.0 — developed against Sionna 1.2.x):

```bash
pip install numpy tensorflow "sionna>=1.0" torch tensorboard
```

Sanity-check the environment, PHY chain, and baselines (no training, CPU-friendly):

```bash
python run_phase1.py
```

Train (each `--mode` selects a preset from `config.py`; CLI flags override it):

```bash
# Run2 operating point: K=16, 30 km/h, heavy load, stale CSI
python train_phase2.py --mode hard_main --run_name my_hardmain

# Run3-style mixed regimes: heterogeneous mobility, K=32, critic v2
python train_phase2.py --mode hetero --critic_v2 --num_ue 32 \
    --n_active_min 16 --n_active_max 32 --ue_speed_min 5 --ue_speed_max 30 \
    --entropy_coef 0.02 --target_kl 0.02

# Run4 queue era: per-UE FIFO queues, per-episode random arrival rate
python train_phase2.py --mode queue --critic_v2 \
    --p_arrival_min 0.15 --p_arrival_max 0.40

# Perfect-CSI (genie) world; warm-start the actor from a previous run
python train_phase2.py --mode hetero --critic_v2 --num_ue 32 \
    --n_active_min 16 --n_active_max 32 --ue_speed_min 5 --ue_speed_max 30 \
    --entropy_coef 0.02 --target_kl 0.02 --pmi_mode genie --p_csi 1.0 \
    --init_from Run3/MixedSpeed_L2b/ckpt/best.pt
```

Long runs: use the auto-resume wrappers (`Run4/_wrap_*.sh`) — they relaunch `train_phase2.py --resume <ckpt>` after any interruption; checkpoint writes are atomic (`atomic_save` in `train_phase2.py`), and a SIGTERM mid-update saves the last *boundary* snapshot, never half-stepped weights.

Evaluate a checkpoint against all baselines:

```bash
python eval_phase2.py --checkpoint runs/<run>/ckpt/best.pt --mode main
```

Note that `eval_phase2.py` only rebuilds the `debug` and `main` presets — `--mode main` is the Run1 (easy) operating point; it cannot reproduce the Run2 hard point and cannot load Run3/Run4 checkpoints (no critic-v2/queue support). The canonical post-hoc evaluations live elsewhere: **Run2** — `Run2/scripts/final_eval.py` (12 held-out seeds at the hard point); **Run3** — the analysis scripts in `Run3/_analysis_20260702/`: `eval_full9_jfiactive.py <RunName>` (official 2×4 grid, 20 seeds) and `final20_envelope.py <RunName>` (oracle-envelope/hybrid verdicts), both of which read the run's `config.json` and set the tuned SUS threshold 0.8 internally; **Run4** — the training loop's own paired 3-seed eval (`Run4/<Run>/csv_logs/eval_metrics.csv`) plus the recorded result CSVs in `Run4/_analysis/` and `Run4/_calib_20260706/` (the genie/OOD probe scripts were ad hoc and not preserved; regenerate by adapting the Run3 scripts to the run's `config.json`).

Post-hoc analyses live in `Run4/_analysis/` — e.g. `csi_fidelity.py` decomposes the CSI error (result: quantization ~92–93 % of the direction error, staleness ~7–8 %; SU beamforming loss 1.39 dB).

## Reproducibility notes

- **Paired protocol.** Every PPO-vs-baseline comparison runs all schedulers on the *same* episode seeds (same channels, same arrivals) and reports paired differences with pre-declared sample sizes.
- **Held-out seeds.** Evaluation uses seeds ≥ 10000, disjoint from training seeds.
- **Deterministic evaluation.** The PPO policy acts greedily (argmax) at eval time.
- **JFI convention.** Jain's fairness index is computed over **active** users only (`metrics.py`).
- **Baselines are tuned, not strawmen.** The SUS orthogonality threshold is re-swept at each operating point (0.8 at the Run3 load, 0.75 in the queue era — see `Run3/BASELINE_AUDIT_2026-07-02.md`, `Run4/RUN4_CHANGES_vs_Run3.md`); using the untuned default costs the baseline up to ~13 %.
- **No genie leakage.** No scheduler — learned or heuristic — ever observes the true channel in its decision path (`baselines.py`, `env.py`); precoding uses only fed-back quantized CSI, except in explicitly labelled `pmi_mode=genie` experiments.
- **Known idealizations.** The standards audit found no substantive 3GPP violations but flags real simplifications — chiefly uncapped Shannon spectral efficiency (24.3 % of SU attempts exceed the 7.4063 b/s/Hz MCS cap vs 0.21 % for MU). These idealizations cut in the baselines' favour, so the reported PPO margins are lower bounds. See `STANDARDS_AUDIT_20260707.md`.

## Where to go next

- How the simulator works, slot by slot: **[docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md)**
- What each of the ~15 training runs did and found: **[docs/RUNS.md](docs/RUNS.md)**
- What is and is not 3GPP-faithful: **`STANDARDS_AUDIT_20260707.md`**
