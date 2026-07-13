# System Model — How the Simulation Is Constructed

This document describes the complete simulation pipeline of the ML DRL Scheduler:
a single-cell downlink MU-MIMO system in which a PPO-learned scheduler (and a
grid of tuned heuristic baselines) allocates users to frequency/spatial resources
under **quantized, randomly-delayed CSI feedback** and **deadline-constrained
traffic**. Every physical-layer statement below names the source file and
function so it can be verified against the code. Idealizations are declared
explicitly in §13, following the 3GPP standards audit
(`STANDARDS_AUDIT_20260707.md`).

The simulator is 15 Python files; the physics lives in seven of them:

| File | Role |
|---|---|
| `config.py` | Single `Config` dataclass; all parameters and run presets |
| `channel.py` | TR 38.901 UMi true channel via Sionna (`ChannelGenerator`) |
| `codebook.py` | Type-II-like 56-bit PMI codebook (OMP over DFT atoms) + genie |
| `csi.py` | True-CSI generation, Bernoulli feedback buffer, Age |
| `phy.py` | RZF precoding, SINR, PHY abstraction (Shannon MI), noise calibration |
| `traffic.py` | Bernoulli packet arrivals, per-UE FIFO queues, deadlines |
| `transmission.py` | Transmission units, ideal-IR MI accumulation, HARQ/retx |
| `env.py` | Gym-style environment tying it all together (dynamics only) |
| `baselines.py` | 2×4 / 2×6 heuristic scheduler grid |
| `policy.py` / `ppo.py` / `train_phase2.py` | PPO actor-critic, update rule, training loop |
| `metrics.py` | Episode metrics and evaluation loop |
| `run_phase1.py` / `eval_phase2.py` | Phase-1 baseline runner / standalone checkpoint-vs-baselines evaluator (same held-out seed-10000+ protocol) |

---

## 1. One Slot, End to End

One environment step = one NR slot (0.5 ms at 30 kHz SCS,
`config.py: Config.slot_duration`). The per-slot pipeline, in execution order
(`env.py: _prepare_slot()` then `env.py: step()` → `_finish_slot()`):

```
slot t
│  PREPARE  (env.py: _prepare_slot)
│  1. True channel h_true[t]  [K, 8 RBG, 32 ant]          channel.py: get_channel()
│  2. UE-side CSI: OMP-quantize h_true → PMI, CQI          codebook.py: quantize(),
│     (precomputed once per episode, deterministic)        csi.py: precompute_episode_csi()
│  3. Bernoulli(p_csi) per-UE feedback → CSI buffer;       csi.py: CSIFeedbackBuffer.step()
│     Age[u]=0 on report else Age[u]+=1
│  4. Pending retransmissions compacted into fixed          transmission.py: compact_pending()
│     positions (RBG-pinned, front layers)
│  5. Bernoulli(p_arrival) packet arrivals → FIFO queues   traffic.py: generate_arrivals()
│  6. BS channel estimate h_hat from FED-BACK PMI+CQI       phy.py: reconstruct_h_hat()
│
│  DECIDE   (outside the env — PPO actor or a baseline)
│  7. Scheduler reads the observation (feedback CSI +      env.py: get_observation()
│     traffic state ONLY; never h_true) and emits an
│     8×4 allocation matrix                                 policy.py: decode() /
│                                                           baselines.py: schedule()
│
│  FINISH   (env.py: _finish_slot)
│  8. Validate allocation, create transmission units,      env.py: _sanitize_and_create()
│     commit B_tx bits per unit
│  9. Per-RBG RZF precoder from h_hat; per-stream SINR     phy.py: rzf_precoder(),
│     evaluated on h_true (stale/quantized CSI → mismatch)  compute_slot_sinr()
│ 10. Mutual-information accumulation → ACK / NACK /       transmission.py: process_slot()
│     retx-exhausted drop (ideal IR HARQ)
│ 11. Packet completion; deadline tick for ALL queued      env.py: _finish_slot,
│     packets; deadline-miss / overflow removal             traffic.py: decrement_deadlines()
│ 12. Reward = urgency-weighted useful MI + completions    env.py: _finish_slot
│     − misses/drops;  EWMA throughput update
└─ slot t+1
```

Key information-structure invariant (verified in the standards audit): **the
scheduler and the precoder consume only fed-back PMI/CQI/Age. The true channel
is used exclusively for physics evaluation** (CQI generation, SINR, ACK/NACK).
No genie CSI enters any decision path in any run, PPO or baseline
(`channel.py` module docstring; audit §4 "정보 구조") — the sole, deliberate
exception being the Run4 genie upper-bound experiments, where
`pmi_mode='genie'` + `p_csi=1.0` make the *feedback itself* perfect while
still flowing through the same buffer (§4.1).

---

## 2. System Parameters at a Glance

Canonical operating point = the Run4 preset (`config.py: phase4_queue_config()`).
Earlier runs used the same physics with different traffic/mobility points
(historical column; see the run docs for details).

| Parameter | Canonical (Run4) | Historical (Run1→Run3) | Source |
|---|---|---|---|
| BS antennas | 32 (4×4 dual cross-pol panel) | same | `config.py:26-30` |
| UE antennas | 1 (rank-1 per UE, by physics) | same | `config.py:27` |
| Carrier / SCS | 3.5 GHz, 30 kHz (μ=1) | same | `config.py:49-50` |
| Bandwidth | 8 RBG × 8 RB × 12 SC = 64 PRB = 23.04 MHz | same | `config.py: rbg_bandwidth` |
| Slot | 0.5 ms; episode 1000 slots = 0.5 s | same | `config.py: slot_duration` |
| REs per RBG | 8·12·14 = 1344 (`eta_data`=1.0) | same | `config.py: n_re_rbg` |
| Scheduling grid | 8 RBG × 4 layers = 32 positions/slot | same | `config.py: num_positions` |
| K (UEs) | 32, active n ~ U{16..32}/episode | 16 fixed (Run1/2, Run3 Uniform10/DeadlineScarcity), 32 (Run3 ScarcityK32 onward) | `phase4_queue_config` |
| UE speed | per-UE U(5, 30) km/h per episode | 3 (Run1), 30 (Run2), 10/15 fixed → U(5,30) (Run3 stages; mix {5,10,15,30} only in the deleted HeteroV1) | `channel.py:119-131` |
| CSI codebook | Type-II-like, 56 bit/(UE,RBG) | same in all runs (Phase-2 environment default since 2026-05-21; random_unit_norm was the Phase-1 baseline) | `codebook.py` |
| CSI feedback prob. p_csi | 0.6 | 0.2 (Run1/2), 0.6 (Run3), sweep 0.2–1.0 | `config.py:81` |
| Traffic | Bernoulli p_arrival=0.22/UE/slot, FIFO queue size 8 | one-packet model, p_arr 0.2→0.4 | `traffic.py` |
| Packet size / deadline | U[4000,12000] bit / U[3,12] slots | deadlines U[5,30] (Run1) | `config.py:93-96` |
| Power | P_total=8 (normalized) → P_r=1 per RBG, equal split per stream | same | `config.py:114`, `phy.py:105` |
| Operating SNR | σ² calibrated per episode: median SU beamformed SNR = 10 dB | same | `phy.py: sigma2_from_gain()` |
| HARQ | max_retx=4 → 5 attempts/unit, ideal IR | same | `config.py:121`, `transmission.py` |
| SUS threshold (baselines) | 0.75 (re-swept at queue point) | 0.8 (Run3), 0.5 (grid default) | `phase4_queue_config`, ground truth |

---

## 3. Channel: TR 38.901 UMi via Sionna

`channel.py: ChannelGenerator` wraps Sionna's system-level TR 38.901 UMi
street-canyon model (`sionna.phy.channel.tr38901.UMi`).

- **Antenna geometry** (`channel.py:56-69`): BS = `PanelArray` with 4 rows × 4
  columns × dual cross-polarization = **32 fully-digital ports**, 38.901 element
  pattern; UE = single vertically-polarized omni element. 32 ports equals the
  NR CSI-RS maximum (audit §4).
- **Topology** (`channel.py:137-149`): one `gen_single_sector_topology` drop per
  episode — pathloss, LOS/NLOS state, and shadow fading enabled
  (`config.py:58-59`), all UEs outdoor (`indoor_probability=0.0`,
  `config.py:60`).
- **Per-UE speeds** (`channel.py:119-147`): three modes — scalar
  `ue_speed_kmh`, a shuffled discrete mix `ue_speed_mix` (Run3 hetero:
  {5,10,15,30} km/h), or continuous per-UE `U(ue_speed_min, ue_speed_max)`
  (Run3 L2b / Run4: U(5,30) km/h). Implementation trick: the drop is generated
  at the maximum speed, then each UE's velocity **vector is rescaled** to its
  assigned magnitude (direction kept). Because speed enters TR 38.901 only
  through the Doppler phase (§7.5 step 11), this is statistically **exact**,
  not an approximation (audit §2).
- **Time evolution** (`channel.py:152-162`): the whole episode's CIR is drawn
  in one call `umi(num_slots, 1/slot_duration)` — consecutive slots are
  Doppler-correlated at a 2 kHz sampling rate. The channel is **block-fading
  per slot** (one realization per slot; f_D·T_slot < 0.05 at 30 km/h, audit §4).
- **Frequency granularity** (`channel.py:81, 159`): `cir_to_ofdm_channel` is
  evaluated at **one frequency per RBG** (RBG-center sample, 2.88 MHz spacing)
  → the channel is frequency-flat within an RBG (declared idealization, §13).
  Result: `h_true[slot, K, 8, 32]` complex128.

The true channel is internal to the simulator; `env.py: get_observation()`
never exposes it.

## 4. CSI Acquisition at the UE

### 4.1 Type-II-like 56-bit codebook (`codebook.py: Type2SparseCodebook`)

A simplified Type-II-like sparse atomic feedback model — explicitly **not**
exact 3GPP Rel-15 Type II (class docstring, `codebook.py:126-137`):

- **Dictionary** (`_build_dictionary()`): 2D DFT beams over the 4×4 per-pol
  grid, oversampled (O1,O2)=(4,4) → 256 beams per polarization; each beam is
  embedded into one polarization block ([b;0] or [0;b]) → **512 unit-norm atoms**
  in C³². Adjacent oversampled atoms are ~90 % correlated (test T6 comment).
- **UE encoding** (`_quantize_batch()`): per (UE, RBG), **OMP selects L=4
  atoms** with least-squares re-fit at each step; the LS coefficients are then
  quantized — amplitudes normalized to the strongest atom and rounded to
  **8 uniform levels (3 bits)**, phases to **QPSK (2 bits)**.
- **Payload** (`config.py: type2_payload_bits`): 4 atoms × (⌈log₂ 512⌉ = 9 atom
  bits + 3 amp + 2 phase) = **56 bits per (UE, RBG)**, i.e. 448 bits per full
  8-RBG report. The report re-selects beams/amplitudes per RBG (Rel-15 varies
  only co-phase per subband — declared as beyond-Rel-15 freedom, offset by
  half the combining coefficients; audit §2/§3).
- L=4, QPSK co-phase, and the (4,4) dual-pol 32-port grid are exact legal
  Rel-15 parameter values (audit §4).

Two alternative codebooks share the same `quantize/reconstruct` interface
(`codebook.py: make_codebook()`): `random_unit_norm` (256 random codewords,
Phase-1 baseline) and **`genie`** (`GenieCodebook`) which returns the exact
unit-norm channel direction — combined with `p_csi=1.0` the BS then holds
`h_hat == h_true` every slot, the perfect-CSI upper bound used in the Run4
genie experiments.

### 4.2 CQI (`csi.py: generate_true_csi()` / `precompute_episode_csi()`)

CQI is the **continuous** single-user spectral efficiency along the *quantized*
direction:

```
SNR_SU = P_r · |h_trueᴴ d̂|² / σ²,    CQI = log₂(1 + SNR_SU)      (csi.py:40-43)
```

There is no 4-bit CQI table and no MCS quantization — a declared idealization
(§13.3, top of the table). Because OMP is deterministic and σ² is episode-level,
the whole episode's true CSI is precomputed once per episode seed
(`csi.py: precompute_episode_csi()`); this is bit-identical to per-slot
computation and cannot leak future information (the actor only ever sees the
feedback buffer).

### 4.3 Feedback process and Age (`csi.py: CSIFeedbackBuffer`)

- Per-slot, per-UE i.i.d. **Bernoulli(p_csi)** feedback (`step()`,
  `csi.py:157-167`): on a report, UE u's direction/CQI for **all 8 RBGs**
  update at once and `Age[u] = 0`; otherwise the stale values persist and
  `Age[u] += 1`. (8-PRB subband CSI in a single report is a legal NR
  configuration; audit §2.)
- At episode start every UE reports once (`reset()`, forced initial feedback —
  episodes begin from a converged CSI state; `csi.py:150-155`).
- The buffer's `direction_fb [K,8,32]`, `cqi_fb [K,8]`, `age [K]` are the
  **only CSI anywhere downstream** — precoder and all schedulers included.

Measured fidelity at the Run4 operating point (p_csi=0.6, U(5,30) km/h;
`Run4/_analysis/csi_fidelity.py`): direction cosine ρ mean **0.924**
(ρ² 0.854), magnitude ratio 0.930, phase-aligned NMSE 0.145. Decomposition:
**quantization contributes ~92–93 % of the direction error, staleness
~7–8 %** — the codebook, not the feedback delay, is the CSI bottleneck (the
central Run3 physics finding). SU beamforming cost of imperfect CSI: 1.39 dB (~0.10 b/s/Hz).

## 5. Channel Reconstruction at the BS

`phy.py: reconstruct_h_hat()` rebuilds a full complex channel estimate from
nothing but the feedback:

```
ĥ = g · d̂,   g = √(σ² · (2^CQI_fb − 1) / P_r)                    (phy.py:63-74)
```

i.e. the magnitude is chosen so the SU SNR of ĥ along the reported direction
exactly reproduces the fed-back SNR (round-trip verified in
`codebook.py` test T13). When CSI is stale, ĥ is a stale reconstruction — the
env recomputes ĥ each slot from the current buffer (`env.py:181-183`).

## 6. Precoding and Power

`phy.py: rzf_precoder()` — per-RBG regularized zero-forcing over the m
co-scheduled UEs of that RBG:

```
V = Ĥᴴ (Ĥ Ĥᴴ + α I)⁻¹,  α = σ²   (config.py: rzf_alpha_mode="noise",
                                    env.py:230 passes alpha=noise_var)
columns renormalized to unit norm, then scaled by √(P_r / m)      (phy.py:98-105)
```

so the **total RBG power is exactly P_r = 1** regardless of depth m
(numerically verified to ≤4.4e-16 in the audit). SINR is then evaluated with
the **true** channel (`phy.py: _rbg_sinr()`, `compute_slot_sinr()`): desired
power = |h_true,uᴴ w_u|², interference = Σ_{v≠u} |h_true,uᴴ w_v|², over
interference + σ². Precoding mismatch from quantized/stale CSI is therefore
the physical failure mechanism of MU pairing.

**The orthogonality / desired-gain tradeoff.** Equal power split means adding
a stream to an RBG costs every existing stream a factor m/(m−1) of transmit
power, while imperfect ĥ leaves residual inter-stream interference that RZF
cannot null (it nulls the *estimated* channels, not the true ones). A layer is
worth scheduling only if its MI exceeds what the power split plus residual
interference takes from the others. Under the 56-bit codebook (ρ² ≈ 0.85) this
tradeoff resolves near SU — the Run3 depth-cap sweep showed heuristic
throughput decreasing monotonically with MU depth, i.e. PPO's learned depth
~1.3 at K=16 is physics, not a training artifact (ground truth). Note the
conservative direction: equal split with no waterfilling is a deliberately
pessimistic gNB choice **for MU**, so MU/PPO gains are understated on this
axis (audit §5).

Noise calibration (`phy.py: sigma2_from_gain()`, `calibrate_noise()`): σ² is
re-fit **per episode (drop)** so the median SU beamformed SNR over
(slot, UE, RBG) equals `target_snr_db = 10 dB`. This is a surrogate for an
interference-limited UMi geometry (RAN1 large-scale-calibration medians), not
a physical link budget — absolute dBm claims are forbidden (audit §2/§3).
Per-UE near-far spread (~46 dB pathloss dynamic range) is preserved.

**Causality caveat (audit rounds 9–10).** Because the median is taken over
the WHOLE episode, σ² — and therefore every CQI value, the RZF α, and
post-RZF predictions from slot 0 on — is a function of the episode's entire
channel realization, including its future. Strict per-slot causality
("observations up to t depend only on channels up to t") is intentionally
traded for a fixed operating point: this is a **non-causal episode-level
benchmark normalization to equal median SNR per world, not a
deployment-causal PHY**. It is one scalar per seed, shared identically by
every scheduler and by PPO training and evaluation alike, so no within-world
comparison or paired CI depends on per-slot causality. (A frozen σ² derived
from long-term large-scale statistics on an independent calibration set is a
legitimate alternative design for a future generation.)

## 7. PHY Abstraction and HARQ

No per-RE link simulation. The abstraction (`phy.py: mi_bits()`,
`transmission.py: process_slot()`):

- **Transmission unit** (`transmission.py: TransmissionUnit`): one
  (packet, UE, RBG) with a fixed target payload
  `B_tx = min(uncommitted backlog, eta·1344·beta·CQI_fb)` bits, committed at
  creation (`env.py:391-401`, `phy.py: predict_b_tx()`; `beta_rate=1.0`,
  `b_tx_epsilon=1` bit floor, sub-epsilon residual tails are swallowed into
  the unit so a packet can always complete, `env.py:395-400`).
- **Ideal IR accumulation**: each slot a unit is on air it accumulates
  `useful = min(1344·log₂(1+SINR_true), B_tx − I_acc)`; ACK is deterministic
  when `I_acc ≥ B_tx` (`transmission.py:204-214`). Shannon SE is **uncapped**
  (no 7.4063 b/s/Hz NR limit) — the top-priority declared idealization; it
  structurally favors SU operation (24.3 % of SU attempts exceed the cap vs
  0.21 % of MU attempts), so the PPO-vs-SU-baseline gap is conservative
  (audit §3, ground truth).
- **Retransmission**: a NACKed unit stays **pinned to its RBG** with B_tx
  unchanged (TB-semantics: TS 38.214 §5.1.3.2 analog) and is compacted to the
  front layers at the next slot start
  (`transmission.py: compact_pending()`); those positions are force-fixed and
  the scheduler cannot override them (`env.py:162-171`). A unit gets
  `1 + max_retx = 5` attempts; exhaustion drops the **whole parent packet**
  including sibling units in other RBGs (`env.py:249-258`). If more than
  L_max=4 pending units collide in one RBG, the earliest-deadline four are
  kept and the rest become retx-overflow packet drops
  (`transmission.py:134-162`).
- **Standards mapping** (audit §2): a unit is a **CBG-like per-RBG
  abstraction** (independent A/N per unit; ≤8 units/UE mirrors
  maxCodeBlockGroupsPerTransportBlock=8; one TB/MCS spanning a CBG group is
  **not** modeled); a UE's
  multi-RBG rank-1 allocation ≡ one PDSCH with RA Type 0 + PRG subband
  precoding; RBG pinning is a standard-permitted gNB restriction adopted to
  keep the DRL action space stationary. HARQ RTT is 1 slot with error-free
  A/N (idealization; favors the retx-heavy MU baselines → PPO margins are
  lower bounds).

### 7.1 Link adaptation and its known bias

`B_tx` is sized from the fed-back **full-power SU CQI**
(`phy.py: predict_b_tx()`) even when m > 1 streams share the RBG power (each
stream then gets P_r/m plus residual inter-stream interference). Under
ideal-IR HARQ this makes depth ≥ 2 **first transmissions structurally NACK**:
even with perfect, mutually orthogonal CSI, the required IR rounds at the
10 dB operating point are **1.34 / 1.64 / 1.91 for m = 2/3/4**. Measured
consequences: ~60 % of the MU heuristics' scheduled positions are pinned on
retransmissions, and a retx-drop failure channel (2.6–4.5 %) opens that SU
strategies never face. This models a gNB with **no MU-aware backoff and no
OLLA** — a deliberate abstraction, applied identically to every scheduler
(PPO and all baselines) — and it tilts the landscape against deep MU. For
the ablation, the flag `cfg.mu_aware_la` (added 2026-07-10; default False =
historical behavior) makes the env de-rate B_tx by the planned stream count:
SE_m = log₂(1 + (2^CQI − 1)/m).

Ablation outcome (10 paired seeds, K=32 mixed point; see
`Run4/_analysis/la_ablation_{type2,genie}.csv` and docs/RUNS.md §3.1): the
de-rate closes the retx-drop failure channel (MU retx-drop → ~0) but does
**not** remove the first-NACK itself — ~93–95% of cap-limited m≥2 first
transmissions still NACK, because SNR/m ignores the RZF projection loss
(0.973/0.947/0.921 for m=2/3/4) and residual interference (probe
2026-07-12). **The artifact is also double-edged** — its retransmission
pinning also served as an implicit serve-to-completion mechanism, so
removing it raises deadline misses and the net MU-vs-SU heuristic balance
barely moves (+4.5% → +4.1%). The first-order MU limiter in this simulator
is codebook quantization, not the LA artifact; conclusions about MU value
remain conditional on this LA abstraction either way. The complete
treatment — B_tx from the predicted post-RZF SINR of the final RBG group,
with depth-wise backoff β_m calibrated scheduler-independently to a 90%
first-ACK target — is `la_mode="post_rzf"` (2026-07-13; `la_planner.py`,
Gates 1–3 in `Run4/_analysis/scripts/audit_probes/gate23_post_rzf.py`).

## 8. Traffic and QoS

`traffic.py: TrafficModel` — Bernoulli arrivals with hard per-packet deadlines.

- **Packets** (`traffic.py: Packet`): size ~ U[4000, 12000] bits, deadline ~
  U[deadline_min, deadline_max] slots (Run4: U[3,12]); each packet tracks
  committed vs ACKed bits so it can be split across several transmission units.
- **Run3 legacy mode** (`queue_size=1`): one head-of-line packet per UE; an
  idle UE draws Bernoulli(p_arrival), a busy UE draws nothing
  (self-throttling). This path is **RNG-exact** — Run4 code reproduces Run3
  byte-identically at queue_size=1 (verified; `RUN4_CHANGES_vs_Run3.md`).
- **Run4 queue mode** (`queue_size=8`): **every active UE draws
  Bernoulli(p_arrival) every slot**; an arrival landing on a full 8-deep FIFO
  is a **buffer-overflow drop**, penalized in the reward exactly like a
  deadline miss (`env.py:173-178, 303-308`). **Deadlines tick for all queued
  packets** (`traffic.py: decrement_deadlines()`); a packet expiring while
  waiting in the queue is a deadline miss (`pop_expired_queued()`), HOL expiry
  additionally purges its in-flight units and can cascade if the promoted
  successor is already expired (`env.py:284-298` while-loop). Only the HOL
  packet is schedulable; `packets[u]` remains the HOL view so the entire
  scheduler interface is unchanged.
- p_arrival was recalibrated for queue mode 0.4 → **0.22** so nominal ρ ≈ 0.8
  at mean load n_active=24 against a reference service capacity
  C_ref ≈ 52 kbit/slot (n.b. kbit per 0.5 ms slot, not Mbps;
  `phase4_queue_config` docstring) (per-episode ρ spans ~0.53 at n=16 to
  ~1.07 at n=32); measured effective ρ 0.58–0.78, buffer overflow 0 at calibration —
  deadline expiry cleans queues before the buffer binds
  (`config.py: phase4_queue_config` docstring, `RUN4_CHANGES_vs_Run3.md`).
- Level-2 mixed regimes (all drawn from dedicated RNG streams so other draws
  are untouched; `env.py:106-124`, speeds `channel.py:119-131`): per-episode
  `n_active ~ U{16..32}`
  (mixed load), per-UE speed U(5,30) (mixed mobility), and per-episode
  `p_arrival ~ U(0.15,0.40)` (QueueMixedArrival — decorrelates offered load
  from n_active so the queue observations become the only load signal).
- Hard deadlines map to delay-critical GBR **PDB accounting** / PDCP
  discardTimer; max_retx + parent-drop is RLC-UM-like (no ARQ recovery) —
  appropriate for deadline traffic (audit §2/§5).

## 9. Scheduler Interface

### 9.1 Observation (`env.py: get_observation()`)

Everything a scheduler (PPO or baseline) may see:

| Field | Shape | Meaning |
|---|---|---|
| `direction_fb` | [K,8,32] ℂ | fed-back unit-norm directions (stale allowed) |
| `cqi_fb` | [K,8] | fed-back continuous CQI |
| `age` | [K] | slots since UE's last report |
| `deadline`, `backlog`, `uncommitted` | [K] | HOL packet state |
| `avg_throughput` | [K] | EWMA of ACKed bits, window t_c=100 (`env.py:313-315`) |
| `active` | [K] bool | UE has a HOL packet |
| `queue_len`, `queue_bits`, `next_deadline` | [K] | Run4 queue state (`traffic.py:167-180`) |
| `fixed_allocation/mask/unit_map`, `initial_S_r` | [8,4] (`initial_S_r`: list of 8 per-RBG UE-id sets) | retx-pinned positions this slot |

PPO's per-(UE,RBG) encoder input (`policy.py: build_encoder_input()`):
Re/Im of direction (64) + CQI/8 + min(Age,50)/50 + backlog/B_norm +
deadline/deadline_max + avg_thr/B_norm + active flag = **70 dims**; queue mode
appends queue_len/8, queue_bits/(8·B_norm), next_deadline/deadline_max →
**73 dims** (`policy.py:72-81`). B_norm = 8000 = mean packet size.

**Formally a POMDP.** In-flight HARQ state — per-unit accumulated mutual
information and attempt counts — is not part of the observation, so identical
observations can precede different ACK/drop outcomes; the decision process is
formally a POMDP rather than an MDP. All baselines act on the same
information (strict parity), so no scheduler is advantaged by the hidden
state.

### 9.2 Action space: 8×4 autoregressive (`policy.py: decode()`)

One slot's action is an allocation matrix [8 RBG × 4 layers] with entries in
{0 = no-user, 1..K = UE}. The actor decodes the 32 positions **sequentially in
layer-major order** (layer 0 across all RBGs, then layer 1, …), each position
a masked categorical over K+1 choices whose logits come from
`ScoreNet` (per-UE: 64-d embedding + 7 in-slot scalars incl. OrthoScore of the
candidate vs the RBG's already-selected set, remaining commit budget,
predicted B_tx; `policy.py:389-423`) and `NoUserHead` (per-position). In-slot
state (selected sets S_r, per-UE budget `temp_uncommit`, in-slot RBG count) is
threaded through the sequence — the same budget logic the env applies, so
decode-time masks equal env-time validity.

Masks and structural limits (`policy.py: _position_logits_and_mask()`,
`env.py: _sanitize_and_create()`):

- **Validity**: a UE is selectable iff it has an active packet, uncommitted
  backlog > ε, and predicted B_tx ≥ ε = 1 bit.
- **Per-UE rank-1 physics**: a UE may appear **at most once per RBG** (dup
  filter) — with a single Rx antenna a UE can receive one stream per RBG; it
  may however be allocated on several RBGs in the same slot (multi-RBG rank-1
  PDSCH ≡ RA Type 0, audit §2). Up to L_max=4 different UEs share one RBG.
- **RBG closure**: choosing no-user closes the RBG — all later layers of that
  RBG are auto-empty, enforced defensively by the env. No-user is invalid on
  an RBG's first stream while candidates exist, and forced when none do.
- **Retx-fixed positions** are skipped by the actor and force-overwritten by
  the env regardless of the action.

Baselines build the same [8,4] matrix by layer-major greedy selection through
identical helpers (`env.py: position_candidates()`, `orthoscore_all()`,
`estimate_btx()`) — strict information parity with PPO.

### 9.3 Critic (`policy.py: build_value_input()`)

Mean-pool + max-pool of the 64-d embeddings + 6 global scalars = 134 dims;
**critic v2** (`ppo_critic_v2`, `build_value_feats_v2()`) appends 27 structured
features (5-bin deadline histogram with backlog mass, backlog totals,
retx-grid occupancy histogram, CQI/Age aggregates, episode phase) → 161, plus
6 queue-pressure features in queue mode → 167. Offline probe: held-out
value R² −0.99 → +0.52 with v2 (comment at `policy.py:96-99`).

## 10. Reward (`env.py: _finish_slot()`)

Two-time-scale, per slot:

```
r_t = λ_s · Σ_u (1 + η_D/(deadline_u+1)) · useful_u / B_norm     [short-term MI]
    + λ_c · n_completed                                           [completion]
    − λ_m · n_failed                                              [failure]
```

with λ_s = λ_c = 1, **λ_m = 2**, η_D = 1, B_norm = 8000
(`config.py:123-128`). `useful_u` is the true delivered MI this slot,
deadline-urgency-weighted with the deadline snapshot taken **before** the tick.
`n_failed` counts deadline misses (HOL and in-queue) + retx-exhaustion drops +
retx-compaction overflow drops + (queue mode) buffer-overflow rejects — all
packet-failure modes are penalized identically (`env.py:303-311`).

Comparison tables follow the standing 9-metric convention (reward, throughput,
SINR, completion, deadline miss, retx drop, total failure, MU depth, JFI;
user rule 2026-07-06). The underlying per-episode records come from
`train_phase2.py: env_episode_metrics()` (logs mu_depth and each failure
component; total failure = their sum, derived in analysis) and
`metrics.py: run_episode()`; **JFI is computed over active UEs only** in both.

Throughput definition: `throughput_mbps` counts **link-layer ACKed bits**.
Bits of packets that are later dropped (deadline miss / retx exhaustion)
remain counted — measured at ~1–2 % of the total; there is no rollback, by
design. Application-level goodput is tracked by `completion_rate`.

## 11. Baseline Schedulers (`baselines.py`)

Official comparison grid (user-defined, 2026-07-02 restructure;
`baselines.py: all_baselines()`): a **2×4 factorial**
{spatial mode} × {metric}, extended to **2×6** in queue mode. Names join gate
and metric with '+':

| | CQI | Deadline-PF | PF | Random | MW (queue) | EDF (queue) |
|---|---|---|---|---|---|---|
| **SUS-gated MU** | SUS+CQI | SUS+Deadline-PF | SUS+PF | SUS+Random | SUS+MW | SUS+EDF |
| **SU** (1 UE/RBG) | SU+CQI | SU+Deadline-PF | SU+PF | SU+Random | SU+MW | SU+EDF |

- **SUS gate** (`baselines.py: SUSPF._select()`): first stream = best metric;
  later streams restricted to UEs with
  `OrthoScore(u|S_r) = 1 − max_{v∈S_r}|d̂_uᴴ d̂_v|² ≥ threshold`
  (`env.py: orthoscore_all()`), else the RBG closes. The threshold is
  **tuned, not defaulted**: 0.8 at the Run3 operating point (+13 % for SUS+PF
  vs the 0.5 default; MixedLoad 3-seed 7170→8110,
  `Run3/BASELINE_AUDIT_2026-07-02.md`), re-swept at the Run4 queue point →
  **0.75** (flat optimum 0.70–0.80, statistically tied; the 0.5 default loses
  ~153 reward vs the 0.75 argmax, ~125 vs the 0.8 plateau edge — mean paired
  over 2 schedulers × 10 seeds) (`Run4/_calib_20260706/`, ground truth).
  Blind-MU variants (no gate) and the in-slot virtual-PF variant `SUS+vPF`
  exist in code but are excluded from the official grid — blind MU as a
  strawman no deployed scheduler uses, vPF because its in-slot spread helps
  only at loose thresholds (`Run3/BASELINE_AUDIT_2026-07-02.md`).
- **Metrics**: CQI-greedy; PF = CQI / max(EWMA thr, 1); Deadline-PF = PF ×
  (1 + η_D/(deadline+1)); MaxWeight = (queue_bits/B_norm) × CQI (the
  throughput-optimal queueing-theory reference); EDF = earliest of
  min(HOL deadline, next-packet deadline) — exactly the two deadlines the
  observation exposes (info parity with PPO; `baselines.py:199-233`).
- All baselines use only the observation — never h_true — and thread the same
  per-UE commit budget the env applies (`baselines.py: Scheduler.schedule()`).

## 12. PPO Training Recipe (`ppo.py`, `train_phase2.py`)

Core: clipped PPO with a **macro action** per slot — the ratio is
`exp(Σ new logp − Σ old logp)` over the ≤32 sub-decisions, log-ratio clamped
to ±20 against float32 overflow (`ppo.py:190-195`). One episode (1000 slots)
per update; 4 epochs × minibatch 256; γ=0.99, GAE λ=0.95, clip 0.2, lr 3e-4,
Adam; advantage normalization per rollout; **GAE bootstraps through the
episode time limit with V(s_T)** (truncation, not termination;
`ppo.py: compute_gae()`).

Stability upgrades, validated across Run1→Run4 (all flags in `config.py`):

1. **Return normalization** (`policy.py:314-353`, `ppo.py:205-208`): the value
   head predicts in normalized return space (EMA μ/σ buffers updated once per
   rollout); raw returns are O(hundreds) and previously made the value gradient
   ~34,000× the policy gradient, starving the shared encoder (Run1 diagnosis).
2. **Separate actor/critic gradient clipping** (`ppo.py:149-271`): the shared
   encoder is clipped with the actor group, `value_head` alone in the critic
   group (both at 0.5), with `value_coef` lowered to 0.25 in the hard,
   hetero, and queue presets (every operating point since Run2).
3. **KL early-stop guard v2** (`ppo.py:237-256`; target 0.02, enabled in Run4
   preset): when a minibatch k3-KL exceeds 1.5× target, the **actor and shared
   encoder freeze for the update's remaining passes while the value head keeps
   training** (v1 halted everything and starved the critic exactly when its
   warm-up noise caused the drift). Clip alone provably did not bound aggregate
   drift (MixedSpeed_L2 collapse: KL 0.054/0.066 at clip-frac 0.43/0.47).
4. **Critic v2** structured value features (§9.3).
5. **Warm start** (`train_phase2.py: --init_from`): fresh run loading all
   matching-shape tensors except `value_head` and the return normalizer.
   The Run4 queue fine-tunes were warm-started from an offline zero-padded
   L2b checkpoint (70→73 encoder input, `Run4/_init_from_L2b/`); the genie
   fine-tune keeps the 70-input encoder and warm-starts from the raw L2b
   `best.pt`. `--init_from` itself performs no padding: it refuses
   `queue_size` mismatches outright, while shape-mismatched tensors are
   silently left at their fresh initialization (listed in the startup
   `fresh:` printout).
6. **Patience auto-harvest** (`--patience_evals`, default 15 in hetero/queue
   modes): early stop after 15 no-improvement eval rounds, **armed only after
   update ≥100** (`--patience_min_updates`; the near-random first eval
   otherwise sets the watermark and the critic warm-up dip eats the budget).
   Ground-truth caveat: patience 15 can still be too short
   (QueueMixedArrival broke through only after an accidental resurrection).
7. **best.pt banking + held-out eval**: every 10 updates, 3 deterministic
   (argmax) episodes on seeds 10000+ — a range training never visits
   (`train_phase2.py: run_eval()`, assertion at line 339); the best-eval
   checkpoint is banked atomically. Entropy coef 0.02 (Run3 runs) vs 0.01:
   measured A/B in Run4 = +0.6 %, no meaningful difference.
8. **Interrupt-safe checkpointing** (`train_phase2.py: ckpt_payload()`,
   `atomic_save()`): SIGTERM/Ctrl-C saves the last *update-boundary* snapshot
   (mid-update weights would silently diverge on resume — verified both ways);
   RNG states and patience counters are checkpointed so auto-resume wrappers
   (`Run4/_wrap_*.sh`) reproduce the uninterrupted trajectory.

Reproducibility: each episode is driven by two seeds — `cfg.seed +
episode_idx` for the topology drop and whole-episode channel/CSI precompute
(`env.py:86`), and `cfg.seed + 7919·episode_idx` for the main env RNG
(CSI-feedback Bernoulli + traffic arrivals; `env.py:83`). Every side draw has
its own generator instance (speeds: episode seed +777, n_active: `cfg.seed +
4441·episode_idx`, baseline Random: `cfg.seed + 100003/100004`, minibatch
shuffle: `cfg.seed + 31337 + update`), so no stream ever consumes another
stream's draws; seed *values*, however, are not strictly disjoint across
streams — the mixed-arrival draw reuses the main episode seed `cfg.seed +
7919·episode_idx` exactly rather than a disjoint offset (`env.py:120`), and
the offsets can coincide at isolated points (episode 0's n_active seed
`+4441·0` equals its main-RNG seed `+7919·0`; update 339's shuffle seed
`+31337+339` equals episode 4's main-RNG seed `+7919·4`) — correlations with
no effect on the paired PPO-vs-baseline comparisons, which see identical env
randomness on both sides; the whole-episode
channel/CSI precompute is LRU-cached (16 entries) so eval seed revisits are
bit-identical (`channel.py:88-95`, `env.py:46-51`). All PPO-vs-baseline
comparisons are paired on the same seeds.

## 13. What the Model Deliberately Is Not

Honesty section, condensed from `STANDARDS_AUDIT_20260707.md` (a 12-agent
audit against 3GPP TS 38.211/213/214/101-1, TR 38.901 et al.). Headline: **no
substantive PHY-procedure violations**; two framing-level items must be worded
as declared abstractions; the remaining deviations are explicit idealizations,
most of which are *conservative for the paper's PPO claims* because they help
the retransmission-reliant MU baselines.

### 13.1 Framing items (must be declared, then not violations)

| Item | Simulation | Required framing |
|---|---|---|
| 1-Rx UE at 3.5 GHz | 1 Rx antenna (`config.py:27`) on a 3.5 GHz carrier — band n77/n78 mandates ≥4 Rx (TS 38.101-1 §7.3.2); no exception category fits | "Generic 3.5 GHz FR1-like carrier (no NR band claimed) + deliberately worst-case 1-Rx UEs; extra Rx branches would help all schedulers symmetrically" |
| All-DL carrier + per-slot feedback | Every RE is DL data (`eta_data=1`) yet A/N + CSI arrive every slot; 3.5 GHz is TDD-only | "Only DL is modeled; HARQ-ACK/CSI delivered over an ideal error-free out-of-band (FDD-like) feedback plane" |

### 13.2 Standards-equivalent mechanisms

| Simulation mechanism | NR equivalent (clause) |
|---|---|
| Multi-RBG rank-1 allocation per UE (`env.py:386-401`) | One PDSCH, RA Type 0 bitmap + PRG subband precoding (TS 38.214 §5.1.2.2.1, §5.1.2.3) |
| Per-(UE,RBG) retx unit, ≤8/UE, independent A/N (`transmission.py`) | CBG-like per-RBG abstraction (cf. CBG-based HARQ, maxCBG/TB = 8, TS 38.214 §5.1.7; one TB/MCS per CBG group is **not** modeled); RBG pinning = permitted gNB restriction |
| One report updates all 8 RBGs (`csi.py:157-167`) | 8-PRB CSI subband, single report covers all subbands (TS 38.214 Tab. 5.2.1.4-2) |
| B_tx fixed across retx (`transmission.py:41`) | TBS invariance on retransmission (TS 38.214 §5.1.3.2) |
| Hard per-packet deadlines, in-queue expiry (`traffic.py`) | Delay-critical GBR PDB accounting / PDCP discardTimer (TS 23.501 §5.7.3.4, TS 38.323) |
| Velocity-rescale per-UE speeds (`channel.py:134-147`) | Exact under TR 38.901 §7.5 (speed enters Doppler phase only) |
| σ² calibrated to 10 dB median SU SNR (`phy.py:29-57`) | Surrogate for interference-limited UMi geometry (TR 38.901 §7.8 calibration medians) — never quote dBm |
| Expired HOL packet's pending retx units are cancelled (`env.py:284-298`) | Permitted gNB discretion — NR DL HARQ is asynchronous & adaptive, retransmissions may simply not be scheduled (TS 38.214 §5.1, TS 38.321) |

### 13.3 Idealizations (priority order) and who each one favors

| Idealization | Where in code | Favors | Consequence for claims |
|---|---|---|---|
| **Uncapped Shannon SE + continuous CQI** (no 7.4063 b/s/Hz cap, no CQI/MCS tables) | `phy.py:152-167`, `csi.py:42` | **SU operation** (SU baselines *and* PPO's SU mode): 24.3 % of SU attempts exceed the cap vs 0.21 % of MU | Top of the limitations list; SU-vs-MU gap partly cap-inflated; PPO-vs-SU-baseline gap conservative. Defend with a capped-SE ablation |
| Ideal IR HARQ (lossless MI accumulation, deterministic ACK) | `transmission.py:204-219` | Mildly favors retx-reliant **MU** strategies | Absolute completion inflated; upper bound on LDPC/RV combining |
| **SU-CQI link adaptation without MU backoff / OLLA** (B_tx from full-power SU CQI even at depth m > 1) | `phy.py: predict_b_tx()`, `env.py:391-401` | **SU-leaning strategies** (depth ≥ 2 first transmissions structurally NACK under ideal IR; §7.1) | Depth-cap / p_csi / genie conclusions are conditional on this abstraction; ablation flag `cfg.mu_aware_la` (2026-07-10) |
| 1-slot HARQ RTT, error-free A/N (real: ≥4–8 slots) | `env.py:162`, `transmission.py:215-219` | Retx-heavy **MU baselines**, tight-deadline feasibility | **PPO margins are lower bounds**; deadlines live on a compressed timescale |
| Zero-latency CSI (min Age 0; real ≥4–5 slots) | `env.py:147-183` | All schedulers' MU/RZF null quality | Stale-CSI degradation reported is a lower bound → conservative for the Age-aware PPO |
| eta_data = 1.0 (no DM-RS/PDCCH/CSI-RS overhead) | `config.py:109` | Uniform ~8–25 % absolute inflation | Comparisons overhead-invariant (load calibrated on same capacity) |
| 100 % DL duty (no TDD pattern) | slot loop | Neutral (~1.3× absolute inflation) | Absolute throughput/latency not deployment-representative |
| Frequency-flat channel per RBG (1 sample / 2.88 MHz) | `channel.py:81,159` | **MU** (no intra-RBG precoder mismatch) | Flag; UMi coherence BW ≈ RBG width |
| Single cell, static white interference | `phy.py:116` | **MU** (RZF controls every interferer) | Interference dynamics out of scope |
| Per-RBG independent B_tx; new+retx units coexist per slot | `env.py:391-401` | **MU** at NACK-prone operating points | Maps to optional multi-PDSCH UE capability (≤5 needed vs ≤7 allowed) |
| Deterministic ACK (no BLER curve / OLLA) | `transmission.py:47-49` | Slight SU tilt; completion inflated | Failures arise only from CSI error + MU interference — the studied mechanism, isolated |
| Bernoulli p_csi feedback, free & error-free UL | `csi.py:157-167` | ~Neutral-to-conservative (40 % loss rate is far more pessimistic than standard reliability) | p_csi ablations cover sensitivity |
| Per-subband full PMI re-selection (beyond Rel-15) | `codebook.py` | All schedulers' RZF accuracy | Offset by ~3× payload; applied uniformly |
| No per-antenna power constraint (sum power only) | `phy.py:103-105` | Neutral (peak/avg ≤2.7 dB) | One-sentence flag |
| σ² per-episode re-fit; non-physical link budget | `phy.py:29-57` | Neutral across schedulers; marginal PPO training stability | Never quote absolute power |
| σ² calibrated **per CSI-world** to a 10 dB median SU SNR | `phy.py: sigma2_from_gain()` | Neutral within a world | **Cross-world ABSOLUTE reward comparisons are invalid** (σ² differs by ~0.5–0.7 dB between the quantized and genie worlds); within-world comparisons unaffected |
| Equal per-stream power (no waterfilling) | `phy.py:105` | **Conservative for MU/PPO** (asset) | MU/PPO gains are understated on this axis |
| All-outdoor UEs (indoor_probability=0, no O2I) | `config.py:60` | ~Neutral; removes worst-case deep-O2I UEs, mildly friendly to MU pairing | Scenario simplification vs the TR 38.901 evaluation assumption of 80 % indoor; one-sentence flag |
| Ideal control plane (PDCCH/DCI capacity unmodeled) | slot loop, `env.py:352-404` | Neutral across schedulers; slight absolute-capacity inflation at high K | Up to 32 allocations + multi-PDSCH DCIs cost 0 REs; CORESET / blind-decode / CCE limits (TS 38.213 §10.1) out of scope |
| Ideal UE receiver (no DM-RS estimation loss; interference treated as Gaussian MI) | `phy.py:108-116` | ~Neutral (real UEs lose ~0.5–1 dB; the Gaussian-interference assumption is slightly pessimistic) | Standard system-level-simulation practice; assumption-list line only |
| t=0 full CSI; same-slot arrival scheduling | `csi.py:150-155`, `env.py:148-178` | Negligible (<1 % transient) | Reproducibility note only |

### 13.4 Compliant (no caveat needed)

Numerology μ=1 / 0.5 ms / 14 sym / 12 SC (TS 38.211); 64-PRB active BWP with
RBG size P=8 = Config 2 exactly (TS 38.214 Tab. 5.1.2.2.1-1); ≤4 layers/RBG ≤
orthogonal DM-RS ports (TS 38.211 §7.4.1.1); ≤8 HARQ processes/UE (limit 16);
Type-II parameters L=4 / QPSK / (4,4) 32-port grid all legal Rel-15 values
(TS 38.214 §5.2.2.2.3) with measured direction correlation in the
limited-feedback literature range; TR 38.901 UMi Table 7.2-1 defaults with
drop-based procedure and valid block fading; exact per-RBG power accounting;
and the information structure itself — identical fed-back observations for
PPO and every baseline, true channel used for physics only. Finally, the
scheduling algorithms themselves (SUS gate, PF/CQI/MW/EDF metrics, the PPO
policy) and the RZF precoder with its CQI-based `ĥ` reconstruction are
implementation-specific in NR — PDSCH precoding is DM-RS-transparent
(TS 38.211 §7.3.1.4) and user selection is a gNB design freedom — so they
require no standards defense (audit §5).

---

*Cross-references: run-by-run configurations and results are in the Run
documentation (Run1–Run4); the authoritative final numbers cited here
(fidelity probe, cap-exceedance rates, threshold sweeps) come from
`Run4/_analysis/`, `Run4/_calib_20260706/`, `RUN4_CHANGES_vs_Run3.md`,
`Run3/BASELINE_AUDIT_2026-07-02.md`, and `STANDARDS_AUDIT_20260707.md`.*
