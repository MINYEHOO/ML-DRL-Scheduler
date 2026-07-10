# Experiment History — Run 1 → Run 4 and the Genie Studies

**Status date: 2026-07-10.** Runs marked *training* were still running on that date.

This document is the complete, honest record of every training run in this
repository: what was changed relative to the previous run, why, what happened
(including the failures), and what was learned. The simulation itself — channel,
CSI feedback, PHY abstraction, traffic, MDP — is described in the companion
simulation document; here we only restate configuration deltas. Code references
use the form `file.py: symbol` so every claim can be checked against source.

**Conventions used throughout**

- **Baseline naming**: spatial gate + selection metric joined by `+`.
  `SUS+CQI` = SUS-orthogonality-gated MU with CQI-greedy metric
  (`baselines.py: SUSCQI`); `SU+CQI` = one-UE-per-RBG restriction of the same
  metric (`baselines.py: _su_restrict`). The gate set is {SUS-gated MU, SU},
  the metric set is {CQI, Deadline-PF, PF, Random} (Run3, 2×4 grid), extended
  in Run4 with {MaxWeight, EDF} (2×6 grid).
- **Standard metric order** for comparison tables (fixed by the repo owner):
  reward, throughput, SINR, completion, deadline miss, retx drop, total
  failure, MU depth, JFI. JFI is computed over **active** users only
  (`metrics.py`; definition fixed 2026-07-06, see §3.2).
- **Evaluation protocol**: paired same-seed episodes on held-out seeds
  (10000+), deterministic (argmax) PPO, pre-declared sample sizes for final
  verdicts. "MU depth" = per-UE realized pairing depth: for each scheduled
  UE, the average co-scheduled UE-count (itself included) of the RBG-slots
  it occupies, averaged over all scheduled UEs (1.0 = pure SU, 4.0 = always
  full 4-layer MU; `env.py:236-244` paired_depth_sum_per_ue,
  `train_phase2.py: env_episode_metrics`). Note this UE-weighted mean reads
  slightly higher than a plain per-used-RBG average when depths are mixed.
- **Reproduction map (which artifact regenerates which number)**: Run2 +50%
  table (§2.3) = `Run2/scripts/final_eval.py` (12 held-out seeds 10000–10011
  at the hard point). Run3 single-regime verdicts (§3.3) =
  `Run3/_analysis_20260702/stats20*.csv` (seed batches 10000–10019 /
  10020–10039 / 10040–10059 pooled to the pre-declared n). Run3
  official-grid sweep + JFI band (§3.2/§3.4) = `eval_full9_jfiactive.py
  <RunName>` → `official9full_<RunName>.csv`. Run3 envelope/hybrid verdicts
  (§3.3) = `final20_envelope.py <RunName>` → `final20_<RunName>.csv` (both
  scripts read the run's `config.json` and set SUS@0.8 internally). Run4
  queue margins (§4.3) = each run's `Run4/<Run>/csv_logs/eval_metrics.csv`:
  quoted best = maximum PPO `reward` row; baseline reference = mean of that
  file's rows for the strongest baseline (deterministic on eval seeds
  10000–10002). Transfer/genie/OOD (§5–§6, §4.4) = recorded CSVs
  `Run4/_calib_20260706/l2b_zeroshot_queue.csv`,
  `Run4/_analysis/genie_zeroshot_l2b_20seed.csv`, `l2b_paired_csi.csv`,
  `ood_p_arrival.csv` (generating scripts were ad hoc and not preserved;
  rebuild from the run `config.json` + `baselines.all_baselines` +
  `train_phase2.PPOScheduler` following the Run3 script pattern). All CIs
  are 95% paired CIs on per-seed differences.
- **Honesty note on absolute numbers**: the standards audit
  (`STANDARDS_AUDIT_20260707.md`) found no substantive 3GPP violations but
  one significant idealization — uncapped Shannon spectral efficiency
  (24.3% of SU transmission attempts exceed the 7.4063 b/s/Hz NR cap, vs
  0.21% for MU). This and the other audited idealizations are
  conservative-direction (they favor the baselines), so the PPO margins
  reported below are **lower bounds** under the audit's analysis.

---

## 0. Master summary

| Run | When | Config delta (vs previous) | Purpose / hypothesis | Headline result |
|---|---|---|---|---|
| **Run1 FirstFullRun** | 2026-06-12 → stopped @1199 | First real training: K=16, 3 km/h, p_arr 0.2, deadline U[5,30], p_csi 0.2, value_coef 0.5 | Can the Phase-2 PPO harness learn at all? | **FAILED** — PPO 4046 ≈ SUS+PF 4117; two root causes diagnosed (no headroom; value grad ~34,000× policy grad) |
| **Run2 HardMain** | 2026-06-15 → 06-21 (2000 upd, ~5.6 d) | Hard point: 30 km/h, p_arr 0.4, dl U[3,12]; optimizer fixes (return norm, separate clip, value_coef 0.25) | Same architecture at an operating point with headroom | **PPO 6438 = +50% vs best MU heuristic (4292), +3.6% vs genie ref (6217)**; mechanism = learned near-SU (depth 1.12) |
| **Run3** (5 runs) | late June → 2026-07-06 | K=32 era: p_csi 0.6, entropy 0.02, axes: speed / deadline / K / mixed load / mixed speed | Where exactly does the learned advantage come from? | Single-regime: tie / +2.62% / +1.71%. **Mixed regimes: beats the per-seed oracle envelope by +6.35% / +6.88%** |
| **Run4 queue family** (5 runs) | 2026-07-06 → (2 still training; 2 verdict-banked runs' auto-resume wrappers were still cycling on 2026-07-10, see §4.3) | One-packet traffic → per-UE FIFO queues (size 8), p_arr 0.22, 2×6 baseline grid | Does the advantage survive when queueing/fairness levers exist for the heuristics? | **+16.8% to +21.4% vs strongest tuned baseline** in every queue environment |
| **Transfer studies** | 2026-07-06+ | L2b policy on queue env, warm vs fresh, entropy A/B | Do Run3 skills transfer? | Queue-blind zero-shot **+11.4%** (10/10 seeds); warm-start exceeds that ceiling at update 9; fresh first touches it at update 159 and holds above it only from ~update 399 |
| **Genie studies** | 2026-07-08+ | Perfect CSI (`codebook.py: GenieCodebook`, p_csi 1.0), exact L2b env otherwise | Is the policy a stale-CSI trick or a real scheduling policy? | L2b **zero-shot +8.1%** vs SUS+CQI (19/20), 156W–4L vs all 8 fixed heuristics; depth decisions CSI-invariant |
| **OOD probe** | 2026-07-10 | QueueMixedArrival best.pt at fixed p ∈ {0.10, 0.22, 0.33, 0.45, 0.50}, 8 paired seeds each | Does the mixed-arrival policy overfit p ~ U(0.15, 0.40)? | Tie (−0.4 %) at OOD-low; +10.2 % / +20.1 % in-dist; **+44.2 % / +135.7 % OOD-high** — no overfitting cliff, upward extrapolation robust |

---

## 1. Run 1 — FirstFullRun (failure and diagnosis)

**Folder**: `Run1/` (README, checkpoints, diagnostic figures,
`Run1/scripts/numeric_scout.py`).

**Setup.** First end-to-end training of the Phase-2 PPO stack
(`policy.py` shared-encoder actor/critic, `ppo.py` GAE + clipped PPO,
`train_phase2.py` loop) at the *default* operating point
(`config.py: phase2_main_config`): K=16 UEs, 3 km/h, p_arrival 0.2,
deadline U[5,30] slots, p_csi 0.2, value_coef 0.5, one global
grad-norm clip. Planned 2000 updates on an RTX 4090 (~240 s/update).

**What happened.** Nothing. PPO eval reward plateaued at baseline level from
update ~50 onward (PPO ~4046 vs best heuristic SUS+PF 4117; best
plateau-level eval 4162 at update 639 — the on-disk `best.pt` actually
banked a one-off 4458 spike at update 1199, judged noise). The run was
manually stopped at update 1199 to avoid wasting the remaining ~2 days.

**Diagnosis** (measured with `Run1/scripts/numeric_scout.py`, two independent
root causes):

1. **No headroom at the operating point.** At 3 km/h / light load / loose
   deadlines, *all* schedulers land within ~1.4% total reward of each other.
   Resources exceed demand, and the channel is so slowly varying that CSI
   staleness is irrelevant — there is literally nothing for a learned policy
   to win.
2. **Optimizer defect: un-normalized value targets.** Returns of magnitude
   ~383 fed to an MSE value loss produced a value gradient ≈ 9237 vs a policy
   gradient ≈ 0.27 — a factor of **~34,000×**. The single global
   `clip_grad_norm_` budget was dominated by the value term; the critic never
   learned (held-out explained variance ~0.06 in the numeric_scout recompute;
   the training-log EV column is merely noisy around ~0.1 with no upward
   trend) and the shared encoder was
   dragged along by the value loss. (An earlier hypothesis — "the clip freezes
   the actor" — was tested and *rejected*: Adam cancels a constant clip scale,
   and the actor did move 4–7% per update, just aimlessly. The real casualty
   was the critic + shared encoder.)

**Fixes carried into Run2** (all still in the code today):
return/value-target normalization with inverse transform at read-out
(`policy.py`: return-normalizer buffers, `update_return_normalizer()`),
separate actor/critic gradient clipping (`ppo.py`), value_coef 0.5 → 0.25
(`config.py: phase2_hard_*` presets).

**Lessons.** (a) Choose an operating point with demonstrated headroom before
training; (b) when learning stalls, measure gradient scales per loss term —
a 4-order-of-magnitude imbalance is invisible in the loss curves.

---

## 2. Run 2 — HardMain (the +50% result)

**Folder**: `Run2/` (`Run2/README.md`, `Run2/Run2_HardMain/`,
`Run2/HardDebug_confirm/`, probe scripts and figures under `Run2/scripts/`,
`Run2/figures/`).

### 2.1 Choosing the hard operating point (read-only probes)

Before retraining, three config-only probes located the headroom
(`Run2/figures/{headroom,oracle,speed}_probe.png`):

- **Headroom probe**: raising load and tightening deadlines
  (p_arr 0.4, dl U[3,12]) widened the all-scheduler reward spread from ~1.3%
  (the probe's independent re-measurement of the §1 easy point, where the
  Run1 diagnosis had measured ~1.4%) to ~8.9% — necessary but not sufficient
  (it only proves "better than Random" is possible).
- **Oracle probe** (3 km/h): fresh-CSI ≈ stale-CSI performance (gap ~0) —
  no room *above* the heuristics at walking speed.
- **Speed sweep** (decisive): the fresh-vs-stale oracle gap opens to **+23%
  at 30 km/h and +34% at 60 km/h**. Staleness is the real lever: heuristics
  ignore CSI Age, a learned policy can exploit it. Increasing the deadline
  urgency weight `eta_d` was counter-productive, so `eta_d=1` was kept.
  **30 km/h adopted.**

### 2.2 Confirmation run before committing 5 GPU-days

`HardDebug_confirm` (K=16, 300 slots, 150 updates, ~2.5 h) verified the
optimizer fixes: grad-norm ~9000 → O(10), explained variance negative →
positive, and PPO eval broke above every baseline at update ~39 (+9% over
the best baseline), finishing the 150-update run **+20–27%** above it.
Only then was the real run launched.

### 2.3 Main run and result

`train_phase2.py --mode hard_main` (`config.py: phase2_hard_main_config`):
K=16, 1000-slot episodes, 30 km/h, p_arr 0.4, dl U[3,12], p_csi 0.2.
2000 updates in 480,436 s (~5.6 days, 240 s/update) on GPU 3 (RTX 4090).
Effective convergence at update ~479 (best.pt = 479); the remaining ~1500
updates bought nothing (final 6276 < best 6438) — the first argument for
early stopping / patience harvesting, later made standard.

Final evaluation, 12 held-out seeds (standard metric order where available;
from `Run2/README.md` §6):

| Scheduler | Reward | Thr (Mbps) | SINR (dB) | Compl. | Miss | Retx-drop | Total fail | Depth |
|---|---|---|---|---|---|---|---|---|
| Best MU heuristic (CQI-greedy) | 4292 | 44 | 6.7 | 0.743 | 0.087 | 0.167 | 0.254 | 1.63 |
| SU+CQI (hand-coded SU baseline) | 6241 | – | 22.2 | 0.843 | 0.152 | 0.003 | – | 1.00 |
| Genie reference (perfect CSI) | 6217 | 54 | 9.4 | 0.869 | 0.121 | 0.008 | – | 2.95 |
| **PPO (stale CSI)** | **6438** | **57** | 20.6 | 0.854 | 0.140 | 0.004 | 0.144 | 1.12 |

**PPO = +50% vs the best MU heuristic and +3.6% vs the genie reference**,
12/12 seed wins; PPO's worst seed (5613) beat the heuristics' best seed
(5510), with the lowest per-seed variance (9.4% vs 14–22%).

### 2.4 Mechanism: the SU discovery (and the honest framing)

At 30 km/h with p_csi 0.2, the BS's RZF nulls (`phy.py: rzf_precoder()`,
computed from fed-back `h_hat`, evaluated against `h_true` in
`phy.py: compute_slot_sinr()`) point at *old* channel directions, so MU
pairing fails: all five MU heuristics run depth ~1.6 and eat a 17% retx-drop
rate — even SUS+PF, because its orthogonality test uses the same stale
directions. **PPO learned near-SU scheduling (depth 1.12, SINR 20.6 dB,
transmission failures ≈ 0)**, converting the failure mode from
"transmit-and-fail" (heuristics: retx 0.167) to "transmit only when sure"
(PPO: miss 0.140); total failure 0.144 vs 0.254.

The honest decomposition (`Run2/scripts/su_baseline.py`): a hand-coded SU+CQI
baseline gets 6241 (+45% over MU-CQI 4292); PPO adds +3.2% on top of it via
occasional safe MU (depth 1.12) and better UE selection. So most of the +50%
is *the SU strategy itself* — the research value is that PPO **discovered**
this non-obvious robust strategy from reward alone, while every
literature heuristic in the pool (CQI/PF/Deadline-PF/SUS+PF) is MU and failed.

### 2.5 AdaptCheck and the p_csi sweep — is it adaptation or memorization?

Two follow-ups (run dirs `runs/AdaptCheck_pcsi10`, `runs/sweep_pcsi*`):

- **AdaptCheck**: retraining at p_csi = 1.0 produced **depth 2.26** — with
  reliable CSI, PPO moves to MU. The SU policy is *conditional on CSI
  reliability*, not memorized.
- **p_csi sweep**: the PPO margin over the strongest heuristic shrinks as
  feedback becomes fresher: **+30.6% at p_csi 0.2 → +14.2% at p_csi 1.0**
  (intermediate points +26.2% / +28.3% / +26.9% at p_csi 0.4 / 0.6 / 0.8 —
  a clear downward trend from stalest to freshest, though not strictly
  monotonic) — consistent with the staleness-exploitation account.

**Known limitations logged at the time** (`Run2/README.md` §8): weak critic
(EV ~0.06 ceiling), no fairness advantage (Jain 0.675 ≈ heuristics), the
genie reference is myopic-greedy and thus not a true upper bound, and only a
single operating point had been validated. These four items shaped Run3.

---

## 3. Run 3 — the K=32 era: physics, tuned baselines, mixed regimes

**Folder**: `Run3/` — five runs (`Uniform10_Ent002`, `DeadlineScarcity`,
`ScarcityK32`, `MixedLoad_L2`, `MixedSpeed_L2` → `MixedSpeed_L2b`), each with
a `CHANGES_vs_Run2.md`; the baseline audit `Run3/BASELINE_AUDIT_2026-07-02.md`;
raw analysis under `Run3/_analysis_20260702/`.

Common deltas vs Run2 for all Run3 runs: p_csi 0.2 → **0.6** (moderate
staleness), entropy coef 0.01 → **0.02**, per-run axis changes below. New
mechanisms added to the simulator for the mixed runs: per-episode active-user
draw `n_active ~ U[n_min, n_max]` (`config.py`/`env.py`/`traffic.py`,
2026-07-01) and per-UE continuous speed draw `U(v_min, v_max)` implemented as
an exact velocity rescale of the TR 38.901 trajectories (`channel.py`;
exactness confirmed by the standards audit).

### 3.1 The physics deep-dive: quantization, not staleness

Run2's story ("stale CSI breaks MU") turned out to be incomplete. Offline
probes (`Run3/_analysis_20260702/mu_physics_probe.py`, depth-cap sweeps)
established:

- **SU beats MU at *every* p_csi**, including p_csi = 1.0 (fresh feedback).
- A **depth-cap sweep** on the heuristics shows throughput *monotonically
  decreasing* with allowed MU depth under imperfect CSI — near-SU is the
  physically optimal operating mode here, so PPO's learned depth ~1.1–1.3 is
  correct, not a training artifact.
- The bottleneck is **codebook quantization**, not staleness: the
  Type-II-like 56-bit PMI (`codebook.py: Type2SparseCodebook`, 4 atoms ×
  (9+3+2) bits) achieves direction correlation only ~0.86–0.88, and RZF
  nulls computed from a ~0.87-accurate direction leak too much interference —
  the classic Jindal limited-feedback saturation effect. (Quantified
  precisely later by the Run4 CSI-fidelity probe, §7: quantization ≈ 92–93%
  of the direction error, staleness ≈ 7–8% at Run3/Run4 speeds and
  p_csi 0.6.)

This reframing matters: the SU-vs-MU decision is governed by *CSI fidelity*
(quantization floor + staleness) and *load*, and both are things a learned
policy can condition on.

### 3.2 The official baseline grid — and why baselines must be tuned

The repo owner fixed the official comparison set as the 2×4 grid
{SUS-gated MU | SU} × {CQI, Deadline-PF, PF, Random} (`baselines.py`).
The audit (`Run3/BASELINE_AUDIT_2026-07-02.md`) then made the baselines as
strong as possible before any PPO claim:

- **SUS threshold must be tuned per operating point.** Sweeps at the Run3
  load found 0.8 ≫ the code default 0.5: e.g. MixedLoad SUS+CQI 8904 (@0.5)
  → 9006 (@0.8), and SUS+PF 7170 → 8110 (~+13–14%); ScarcityK32 paired
  40-seed: (0.8 − 0.5) = +158.4, CI [+114, +203], 36/40 wins. 0.8 filters
  only bad pairs while keeping depth (~3.96); 0.9 starts costing DoF. All
  Run3 verdicts use **SUS+CQI@0.8** (or SU+CQI where that is stronger).
- **Regime champions**: K16 runs → SU+CQI; K32 runs → SUS+CQI@0.8. A 3-seed
  screen of the remaining SUS-gated candidates @0.8 (SUS+PF and SUS+vPF,
  across all 5 Run3 configs; `Run3/_analysis_20260702/sus_family_08_screen.out`)
  left the champion set unchanged (all 10 measurements −10.0% to −21.5% vs
  the champion; the other SUS-gated metrics were later confirmed weaker in
  the 20-seed official-grid evaluation, §3.4).
- **vPF (a user-proposed PF variant) was honestly retired**: +7.4% at the
  loose threshold 0.5, but −0.3 to −2.2% at the tuned 0.8, with no Jain
  gain → excluded from main comparisons (code kept for Run4 re-testing).
- **JFI definition fixed (2026-07-06)**: Jain's index is computed over
  *active* UEs only; the old all-K version compressed the metric by
  n_active/K and was re-computed everywhere. All other 8 metrics audited
  clean. With the corrected JFI, PPO lands just below the SUS-family band
  (0.686–0.699 vs SUS+CQI 0.731–0.733, i.e. −0.03 to −0.045) and well above
  SU+CQI, which is exposed as the least fair (0.586–0.589).

### 3.3 The five runs

| Run | Axis changed (only variable vs its sibling) | Final verdict (paired, pre-declared n) |
|---|---|---|
| `Uniform10_Ent002` | K16, 10 km/h, p_csi 0.6 — "is near-SU physics or under-exploration?" | **TIE** vs SU+CQI: +0.36%, CI [−24, +74], 24/40 (n=40) |
| `DeadlineScarcity` | deadline U[3,12] → **U[2,6]** | **WIN** vs SU+CQI: **+2.62%**, CI [+82, +258], 38/60 (n=60, pre-declared) |
| `ScarcityK32` | K 16 → **32** (all active) | **WIN** vs SUS+CQI@0.8: **+1.71%**, CI [+35, +322], 26/40 (n=40) |
| `MixedLoad_L2` | K=32, **n_active ~ U[16,32] per episode**, 15 km/h | **Beats per-seed oracle envelope +6.35%**, CI [+338, +803], 17/20 (n=20) |
| `MixedSpeed_L2` → `L2b` | + **per-UE speeds U(5,30) km/h per episode** | **Beats per-seed oracle envelope +6.88%**, CI [+410, +824], 16/20 (n=20) |

**Uniform10_Ent002** settled the Run2 debate: with fresher CSI (p_csi 0.6,
10 km/h) and doubled exploration entropy, PPO *still* converged to near-SU
(depth ~1.03) — the physical-optimum explanation wins (a 0.05-entropy
sibling, "Ent005", diverged and was abandoned; 0.02 became the Run3
standard). Statistically this run is also the protocol cautionary tale: at
n=20 the margin over SU+CQI was "+1.08%, significant"; doubling to n=40
collapsed it to +0.36%, CI [−24, +74] → **tie**. Verdict: in the
uncongested K16 regime, SU+CQI *is* the ceiling and PPO reaches it.

**DeadlineScarcity** changed exactly one knob — deadlines U[2,6] — to make
urgency triage matter. PPO beats SU+CQI by +2.62% (n=60, pre-declared after
the n=20 → n=40 margin wobbled: +199 significant → +97 borderline → +170
significant; the n was declared *before* the final batch to avoid optional
stopping). Against SUS+CQI@0.8 the margin is +25.9% (60/60).

**ScarcityK32** changed exactly one knob — K=32, everyone active — creating
~4:1 demand:supply. The training trace contains the single clearest piece of
adaptation evidence in the project: **PPO's MU depth moved from 0.98 (update
9) to 4.00 (update 19)** — the same architecture that chose SU at K16 chose
full MU at K32. The final margin over the tuned SUS+CQI@0.8 is small but
real: +1.71% (n=40; an initial 3-seed "structural tie" verdict was corrected
by the 20→40-seed evaluation — small-n verdicts on either side are
unreliable). Interpretation: when the regime forces "always MU-4", a fixed
heuristic already plays the optimal spatial mode, and only a ~1.7% deadline-
triage residue is left for learning.

**MixedLoad_L2** is where the thesis lands. With n_active redrawn per
episode from U[16,32], no fixed heuristic can be right in every episode
(K16-optimal = SU+CQI, K32-optimal = SUS+CQI@0.8). Final 20-seed paired
verdict: PPO 9560 vs SUS+CQI@0.8 8550 (**+11.81%, 20/20**), vs the strongest
*realizable* hybrid switch (backlog-count threshold, best T=14) 8768
(+9.03%, 18/20), and — the defense experiment that pre-empts the "adaptation
is just one if-statement" reviewer objection — vs the **per-episode oracle
envelope** (the better of SU+CQI / SUS+CQI@0.8 chosen per seed by an oracle)
8989: **+6.35%, CI [+338, +803], 17/20**. The advantage does not come from
mode *selection* but from within-episode behavior (triage + selective
pairing); a realizable switch is nearly useless because episode n_active is
not directly observable.

**MixedSpeed_L2 → L2b: the collapse and the recipe.** MixedSpeed added
per-UE CSI-reliability heterogeneity on top of mixed load. The first attempt
(L2) crossed the baseline bar at update 29 (9179 > 8939) and then **collapsed
under two destructive updates** (update 39: minibatch KL 0.054 / 43% clip
fraction; update 52: 0.066 / 47%) — PPO's ratio clipping demonstrably did
not prevent aggregate policy drift — recovering only to 8350 after 110 more
updates. The post-mortem also confirmed the critic was input-starved: online
EV ~0.04, and an offline probe showed the old 6-scalar global feature vector
regresses held-out returns at R² −0.99 while a rich structured feature set
reaches **+0.52**. L2 was stopped at ~172 and kept (unmodified) as the
collapse exhibit and warm-start donor.

**L2b** re-ran the *identical environment* with three recipe changes
(all opt-in flags; flag-off byte-identity verified before launch):

1. **KL early-stop guard** (`ppo.py`, `--target_kl 0.02`): on minibatch
   k3-KL > 1.5× target, discard the step. The v1 guard (abort remaining
   epochs entirely) fired on 79% of updates and starved the critic; **v2**
   instead freezes actor + shared encoder (grad=None) and continues
   value-only learning — KL is an actor quantity, the critic has no reason
   to stop.
2. **Critic v2** (`policy.py`, `--critic_v2`): +27 structured value-input
   features (deadline histograms, backlog totals, retx occupancy, CQI/Age
   aggregates, slot phase; 134 → 161 dims), motivated by the offline probe.
3. **Warm-start** (`--init_from`): actor-side 18 tensors from L2's best.pt
   (update 29); value head, return normalizer and optimizer fresh.

Outcome: no collapse; healthy plateau (entropy held 0.5–0.8, guard kept KL
at 0.015–0.03), **best 10030 at update 859**, manually harvested at ~1146.
Final 20-seed paired verdict: 9581 vs oracle envelope 8964 → **+6.88%**,
vs best hybrid +10.15% (19/20), vs SUS+CQI@0.8 +12.85% (20/20) — an almost
exact replication of MixedLoad's +6.35% on an independent mixing axis.
Residual open problem: critic v2's *online* EV stayed ~0.05, far below its
offline ceiling of 0.52 (moving policy + episode-scale variance) — carried
to Run4 as an open item. (MixedLoad_L2, which trained without the guard, was
itself later degraded by entropy exhaustion — 0.84 → 0.09 across repeated KL
spikes, eval decaying to ~8000 — and was stopped at ~1057; its verdict above
uses its safely banked best.pt from update 219. Both endings argue for
`best.pt` banking + patience harvesting.)

### 3.4 The official-grid clean sweep, and cross-eval regime lock-in

Against the full official 2×4 grid at both mixed runs (20 seeds, paired;
`Run3/_analysis_20260702/official9_*.csv`): **PPO significantly beats all 8
fixed heuristics in all 16 comparisons** (every CI excludes 0). Minimum
margin: vs SUS+CQI +11.8% / +12.8%; vs SU+CQI +18.1% / +18.8%; the other six
+20% to +208%. Within-seed clean sweep of all 8 at once: 17/20 and 16/20 —
SU+CQI is the winner in every missed seed; all but one are low-load (16–19
active) episodes (the exception, MixedSpeed seed 10000 with 24 UEs active,
is a near-tie: SU+CQI 8121 vs PPO 8082, −0.5%).
Side observation: SUS+Deadline-PF ≈ SUS+PF (deadline weighting is useless
under the SUS gate) while SU+Deadline-PF > SU+PF — metric effects interact
with the spatial mode.

**Cross-evaluation (regime lock-in).** Policies trained at a fixed K are
locked into their regime: the K16 policy evaluated at K32 loses ~17%.
Mixed-regime training is what creates the level-2 adaptivity (one policy
spanning SU↔MU); it is not free with scale, it comes from the training
distribution.

**Run3 thesis (as carried into the paper):** in a *single* regime, PPO ties
or slightly beats the regime-optimal tuned heuristic (tie / +2.62% / +1.71%);
in *mixed* regimes no fixed heuristic can follow, and PPO wins big — above
even the oracle envelope (+6.35% / +6.88%) and ~+9–10% above any realizable
switch.

---

## 4. Run 4 — the multi-packet queue family

**Folder**: `Run4/` — design doc `Run4/RUN4_CHANGES_vs_Run3.md`, calibration
raw data `Run4/_calib_20260706/`, analysis `Run4/_analysis/`, five queue runs
plus two genie runs (§6), all driven by auto-resume wrappers
(`Run4/_wrap_*.sh`).

### 4.1 Why queues, and the five design decisions

Run3's baseline audit had identified a structural fact of the one-packet
traffic model (`traffic.py` pre-Run4): arrivals only reach *idle* UEs, so a
weak user's demand is self-blocking and fairness/queueing heuristics (PF,
MaxWeight) have no lever to act on. Run4 removes exactly that confound:
**per-UE FIFO queues** (FIFO deque capped at queue_size = 8; arrivals to a
full queue are counted as buffer-overflow drops; `traffic.py: TrafficModel`),
Bernoulli arrivals every slot for every active UE, deadlines that tick down
*while queued* (expiry anywhere in the queue = deadline miss; `env.py`),
buffer-overflow drops penalized like misses. Five user-ratified design
decisions (RUN4_CHANGES §2): (1) near-saturation load ρ ≈ 0.8 with the Run3
n_active mixing kept; (2) reward function unchanged — one axis at a time;
(3) finite buffer of 8; (4) in-queue deadline consumption (required for EDF
to mean anything); (5) PHY/HARQ rules untouched, so all differences
attribute to the traffic layer. `queue_size=1` reproduces Run3
**bit-exactly** (RNG-exact legacy arrival path; verified byte-identical CSVs).

Observation and networks: +3 per-UE features (queue_len, queue_bits,
next_deadline; `env.py: get_observation()`), encoder 70 → 73
(`policy.py`, gated on queue mode), critic v2 +6 queue-pressure features
(161 → 167). Baselines extended to the 2×6 grid with
**MaxWeight** (queue_bits × CQI; `baselines.py: MaxWeight/SUSMaxWeight`) and
**EDF** (min(HOL, next-packet) deadline — deliberately given exactly the
information the PPO observation exposes; `baselines.py: EDF/SUSEDF`).

### 4.2 Calibration and adversarial verification

- **p_arrival recalibrated 0.4 → 0.22** by measuring effective load at the
  queue operating point: ρ 0.58–0.78 across n_active 17–32, mean delay
  2.5–3.3 slots (p95 5–8), ~0.7 packets/UE queued, zero buffer overflow
  (deadline expiry clears queues before the buffer cap binds) — the intended
  near-saturation regime.
- **SUS threshold re-swept at the queue operating point → 0.75**
  (`Run4/_calib_20260706/sus_threshold_sweep.csv`; 7 thresholds × 2 metrics ×
  10 seeds, paired). 0.70–0.80 is a statistically flat optimum (0.75 argmax;
  0.75 vs 0.8 on the SUS+CQI arm: +24, CI [−24, +72]; pooled over both
  metrics: +29, CI [−13, +72]); the grid default 0.5 loses ~125 and 0.9
  loses ~150 (both vs 0.8). Consistent with Run3's 0.8 (also on the plateau); with standing
  backlog, slightly looser gating (depth 3.85 → 3.90) pays.
- **40-agent adversarial verification** of the queue implementation (10
  reviewers × cross-validation lenses): **core algorithms clean** — RNG-exact
  queue-dynamics A/B over 2000 slots, packet conservation, overflow stress,
  independent reward recomputation all passed, zero rejections. Confirmed
  findings were doc/code mismatches and ops robustness, fixed with approval:
  EDF upgraded to min(HOL, next); JFI active-only enforced in training/eval
  logs; resume robustness (patience counter persisted in checkpoints;
  `--init_from` refuses queue_size mismatches; patience arming only after
  ≥100 updates); and a **boundary-snapshot interrupt save** — a SIGTERM
  arriving *mid*-`ppo_update` used to save half-stepped weights under the
  previous update's label (latent since Run3; consistency preserved but
  reproducibility broken). The fix clones a frozen update-boundary snapshot
  (weights + optimizer + RNG); verified by injecting SIGTERM mid-update and
  confirming the resumed trajectory matches an uninterrupted run row-for-row.

### 4.3 The queue runs

All margins are vs the **strongest baseline in that environment** (3-seed
run-time eval, seeds 10000–10002). Two runs (`QueueHighLoad`,
`QueueMixedArrival`) were still training on 2026-07-10. Two more —
`QueueMain` and `QueueFineTuneEnt02` — have banked verdicts, but their
auto-resume wrappers were observed still cycling (briefly relaunching the
runs) on 2026-07-10, so their CSV logs may append rows after the evals cited
here; the banked `best.pt` and the numbers below are unaffected.

| Run | Config | Best (update) | Margin | Status 2026-07-10 |
|---|---|---|---|---|
| `QueueMain` | p_arr fixed 0.22, fresh nets | **5732** @489 | **+17.8%** vs SUS+CQI 4865 | verdict banked (plateaued, last eval 5444 @749); wrapper still cycling on 2026-07-10 |
| `QueueHighLoad` | p_arr fixed 0.33, fresh | **5556** @559+ | **+16.8%** | **training** |
| `QueueMixedArrival` | p_arr ~ U(0.15, 0.40) per episode, fresh | **6020** @519+ | **+21.4%** (best queue result) | **training** |
| `QueueFineTune` | warm from L2b (`--init_from Run4/_init_from_L2b/l2b_padded_init.pt` — an offline zero-padded 70→73-encoder copy of L2b best.pt; `--init_from` refuses the raw L2b checkpoint outright via its `queue_size` guard (1 vs 8); a shape mismatch alone would be silently left at fresh initialization, hence the offline padding), entropy 0.01 | **5796** @219 | **+19.1%** | patience-10 harvested (final eval @339) |
| `QueueFineTuneEnt02` | same warm start, entropy 0.02 | **5832** @279 | **+19.9%** | verdict banked (A/B arm); wrapper still cycling on 2026-07-10 |

**QueueMain** answers the headline question: yes, the learned advantage
survives — and grows — when the heuristics finally have queue levers
(MaxWeight, EDF) to pull. +17.8% over the strongest tuned baseline at the
calibrated operating point.

**QueueHighLoad** (p_arr 0.33) probes deeper saturation; at that load the
CQI- and MaxWeight-metric baselines split by ~4–17% at high n_active
(load finally differentiates the heuristic metrics), and PPO holds +16.8%.

**QueueMixedArrival** draws p_arrival per episode from U(0.15, 0.40) — this
*decorrelates load from n_active*, so the queue observations become the only
reliable load signal; it is the queue-era analogue of Run3's mixed regimes
and produced the best queue result, +21.4%. It is also the **premature-
harvest cautionary tale**: patience-15 auto-harvested it at update 399 with
best 5525; an accidental resurrection (the run was restarted by recovery
tooling) let it train on, and it broke through to 6020 — patience 15 was too
short for this env. The harvest heuristic trades GPU-days against exactly
this risk; treat harvested plateaus in mixing-heavy environments with
suspicion.

**QueueFineTune / QueueFineTuneEnt02** are the warm-start and entropy A/B
arms, discussed in §5.

### 4.4 OOD generalization probe — fixed arrival rates (2026-07-10)

Trained on p_arrival ~ U(0.15, 0.40), does the QueueMixedArrival policy
overfit its training distribution? The probe
(`Run4/_analysis/ood_p_arrival.csv`) evaluates its `best.pt` at five **fixed**
arrival rates — two inside the training range (0.22, 0.33), three outside
(0.10, 0.45, 0.50) — against the strongest fixed baselines (SUS+CQI, SUS+MW,
SU+CQI; SUS gate at the queue-calibrated 0.75), 8 paired seeds (10000–10007)
per point. Margin = mean per-seed paired % difference vs the **per-seed best**
fixed baseline (at heavy load the baseline denominators shrink, inflating
percentages — absolute rewards quoted where that bites):

| p_arrival | Region | PPO margin | Seed wins |
|---|---|---|---|
| 0.10 | OOD (low) | **−0.4 % — tie** (nothing to win at light load; SU+CQI is the regime optimum) | 5/8 |
| 0.22 | in-distribution (QueueMain point) | **+10.2 %** | 8/8 |
| 0.33 | in-distribution (QueueHighLoad point) | **+20.1 %** | 8/8 |
| 0.45 | OOD (high) | **+44.2 %** | 8/8 |
| 0.50 | OOD (far) | **+135.7 %** (per-seed % inflated by small baseline denominators — quote the absolute too: PPO 4236 vs 2774) | 8/8 |

**Conclusion: no overfitting cliff at the training-distribution boundary.**
Upward extrapolation is robust — as load rises past the training range the
baselines collapse faster than PPO (at p = 0.50 SU+CQI's mean reward is
negative) — and downward degrades gracefully to parity: at p = 0.10 the
system is lightly loaded and there is nothing to win, the Run1/Uniform10
headroom lesson reappearing at evaluation time.

---

## 5. Transfer studies (Run3 → Run4)

- **Zero-shot, queue-blind**: the Run3 MixedSpeed_L2b policy run unmodified
  on the queue environment (its three queue observation columns zero-padded)
  scores **5375 vs SUS+CQI 4823 = +11.4%, 10/10 seeds**
  (`Run4/_calib_20260706/l2b_zeroshot_queue.csv`). Run3's skills — CSI-aware
  SU/MU selection, deadline triage — transfer almost fully to an environment
  the policy has literally never seen and cannot fully observe.
- **Warm vs fresh**: the warm-started arms exceeded the 5375 zero-shot
  ceiling **at update 9**, their first eval (5671); freshly trained
  `QueueMain` first touched that level at update ~159 (a brief 5398 spike in
  the 3-seed run-eval), held it durably only from ~update 399, and did not
  match the warm arms' update-9 score until ~update 489 (its best, 5732).
  Final heights: warm 5796/5832 vs fresh 5732 — warm wins on
  sample efficiency decisively and on final height slightly (one seed per
  arm; treat the height ordering as suggestive, not proven).
- **Entropy A/B** (only difference between the two warm arms):
  0.02 → 5832 vs 0.01 → 5796, i.e. **+0.6% — no meaningful difference** at
  this scale. Run4's default of 0.01 is retained.

---

## 6. Genie / perfect-CSI studies

Setup: `pmi_mode=genie` (`codebook.py: GenieCodebook`, BS uses
`h_hat == h_true`) plus p_csi = 1.0, in the otherwise exact MixedSpeed_L2b
environment. Purpose: separate "the policy exploits CSI imperfection" from
"the policy is a good scheduler".

**Zero-shot, 20 seeds, paired** (`Run4/_analysis/genie_zeroshot_l2b_20seed.csv`):
the *unchanged* L2b policy (trained entirely under imperfect CSI) scores
**10095 average, +8.1% vs SUS+CQI (CI [+505, +1016], 19/20)** and **+326 vs
the per-seed best of all 8 fixed heuristics (CI [+134, +519], 16/20)**.
Per-baseline win/loss over the 8 × 20 = 160 paired comparisons —
**156W–4L**:

| Baseline (mean reward) | PPO wins | Losses (all at load extremes, −2.7% to −6.0%) |
|---|---|---|
| SUS+CQI (9334) | 19/20 | 1 (highest-load seed) |
| SU+CQI (8595) | 17/20 | 3 (low-load seeds incl. n_active=16) |
| SUS+Deadline-PF (8280), SUS+PF (8236), SUS+Random (6925), SU+Deadline-PF (7282), SU+PF (6603), SU+Random (3965) | 20/20 each | — |

The only losses are to the regime-matched CQI heuristic at load extremes —
i.e. exactly where a fixed heuristic happens to be the regime optimum.

**Depth invariance**: MU depth is **load-driven** (1.31 at n_active 16 →
2.52 at n_active 32; `Run4/_analysis/genie_zeroshot_l2b_20seed.csv`) and
**nearly CSI-invariant** (paired mean 1.92 imperfect → 1.90 perfect;
`Run4/_analysis/l2b_paired_csi.csv`, 10 paired seeds, same policy under
imperfect vs perfect CSI). Perfect CSI adds **execution
quality, not different decisions**: same policy, +4.7% reward
(9709 → 10167). This is the cleanest evidence that the policy's decision
structure is a scheduling policy, not a stale-CSI artifact.

**Training in the genie world** (both as of 2026-07-10):

- `GenieL2b` (fresh training under perfect CSI): best **~9939 @439** vs
  SUS+CQI 9893 → **tie so far after 700+ updates** — *training*.
- `GenieFineTune` (warm from L2b): best **10780 @169 = +9.1%** — *training*.

Provisional ranking in the perfect-CSI world:
**warm-start > zero-shot L2b > fresh-trained ≈ SUS+CQI.** The skills learned
under imperfect CSI are not merely transferable to perfect CSI — so far they
appear *hard to re-learn from scratch* there, plausibly because imperfect
CSI provides a richer failure signal during training.

---

## 7. CSI-fidelity probe (why the physics story is quantitative)

`Run4/_analysis/csi_fidelity.py` measures the end-to-end fed-back channel
estimate (`phy.py: reconstruct_h_hat()` from Type-II PMI + CQI, vs `h_true`)
at the Run3/Run4 operating point:

| Quantity | Value |
|---|---|
| Direction cosine ρ (mean) | **0.924** (ρ² = 0.854) |
| Magnitude ratio | 0.930 |
| Phase-aligned NMSE | 0.145 |
| Error share: quantization vs staleness | **~92–93% vs ~7–8%** (direction error / NMSE; in SU beamforming dB cost, ~98% vs ~2%) |
| SU beamforming cost of imperfect CSI | **1.39 dB** (~0.10 b/s/Hz) |

This closes the loop on §3.1: at p_csi 0.6 and 5–30 km/h the codebook
quantization floor dominates staleness by roughly an order of magnitude
(~11–14:1 in direction-error/NMSE terms; ~57:1 in SU beamforming dB cost),
which is why depth decisions
are CSI-fidelity- and load-driven rather than Age-driven in the Run3/Run4
regime (contrast Run2's 30 km/h / p_csi 0.2 point, where staleness was the
engineered lever).

---

## 8. Key lessons

**Statistics protocol (evolved across Run3, now standard):**

1. **Paired same-seed evaluation** on held-out seeds (10000+), deterministic
   policy, absolute + percentage margins with CIs and win counts.
2. **Pre-declare n before the final batch** (DeadlineScarcity declared n=60
   after n=20/n=40 wobbled) — avoids optional stopping.
3. **Double n before believing small margins.** Uniform10's "+1.08%,
   significant at n=20" evaporated to +0.36% at n=40; ScarcityK32's 3-seed
   "tie" reversed to a +1.71% win at n=40. Small-n verdicts are unreliable
   in *both* directions.
4. **Tune the baselines first** (SUS threshold per operating point; add SU
   restrictions and queue-aware heuristics) — a win over an untuned baseline
   is not a result.
5. **Run the defense experiments yourself** (oracle envelope, realizable
   hybrid switch) before a reviewer asks.

**Training-recipe lessons:**

6. Operating points must have measured headroom (Run1); probes are cheap,
   5-day runs are not.
7. Diagnose optimizer *scales*, not just losses (Run1's 34,000× ratio).
8. Ratio clipping alone does not prevent policy collapse — use a KL guard,
   and make it value-only (v2) so the critic keeps learning (L2 → L2b).
9. Critic quality is often an input-representation problem (R² −0.99 → +0.52
   from features alone); online EV can still lag the offline ceiling.
10. Bank `best.pt` (top-k) — every Run3 verdict was rescued from a banked
    best after later-run degradation or collapse.
11. Patience auto-harvest saves compute but can harvest prematurely
    (QueueMixedArrival: harvested at 5525, resurrected, broke through to
    6020) — size patience to the environment's mixing difficulty.
12. Warm-starting beats fresh training on sample efficiency by more than an
    order of magnitude (above the zero-shot ceiling at update 9; fresh needed
    ~159 updates to first touch it, ~399 to durably exceed it, and ~489 to
    match the warm arms' update-9 score) and did not cost final height.

**Ops lessons (containerized, restart-prone GPU environment):**

13. The real safety net is **checkpoints + auto-resume wrappers**
    (`Run4/_wrap_*.sh`, `Run3/_watchdog.sh`), not process persistence.
14. Interrupt saves must snapshot at **update boundaries** (weights +
    optimizer + RNG); mid-update SIGTERM saves silently break
    reproducibility (latent since Run3, found by adversarial verification,
    fixed and empirically verified in Run4).
15. **Any stopped run must be removed from every recovery list in the same
    breath** — the 2026-07-06 resurrection incident (a bashrc recovery hook
    revived three finished Run3 runs, appending 454 junk CSV rows and
    overwriting three latest.pt; all best.pt and official result CSVs were
    unharmed) — and, ironically, the same failure mode un-harvested
    QueueMixedArrival to its best result. Automation cuts both ways;
    make it explicit.
16. Keep a **bit-exact legacy path** when extending the simulator
    (queue_size=1 == Run3): it converts "we changed the environment" from an
    assumption into a verified byte-identity test.
