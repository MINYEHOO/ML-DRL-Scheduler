# Code changes from the external-audit era (2026-07-10 → 2026-07-13)

Scope: **code** changes that came out of publishing the repo and the seven
rounds of external (GPT) audit + probe-based adjudication. Pure
documentation edits are excluded on purpose — those corrected *wording that
invited misreading*, not simulator behavior (adjudication records:
`_analysis/gpt_audit_adjudication.md`, `_analysis/gpt_audit_round2_adjudication.md`,
`docs/RUNS.md` §4.5). Every entry below changed what the code *does* or what
it *records*.

Ground rule that held throughout: the legacy default configuration stays
**bit-exact** (CSV-identical debug runs against the pre-change code after
every edit block — 7 blocks total), because two live trainings auto-resume
from disk with whatever code is checked out.

---

## 1. `mu_aware_la` — SNR/m link-adaptation flag (round 1–2, C-cluster)

**Problem found by audit:** B_tx for a new unit was sized from the
full-power SU CQI even when m > 1 streams share the RBG power, making
depth ≥ 2 first transmissions structurally NACK (required IR rounds
1.34/1.64/1.91 for m = 2/3/4 at 10 dB).

**Change:** `config.mu_aware_la` (default False = historical), env-side
de-rate at unit creation: `SE_m = log2(1 + (2^CQI − 1)/m)` with `m_planned`
counted by a pre-pass over the allocation; `--mu_aware_la` CLI +
resume-warn-list entry. Used by the `QueueMixedFairLA` run (GPU5).

**Status after later rounds:** kept as the `la_mode="snr_m"` ablation tier.
A probe showed it closes the retx-drop channel but ~93–95% of cap-limited
m ≥ 2 first transmissions still NACK (projection loss + residual
interference ignored) — which motivated §3 below.

## 2. LA/decode-order configuration axes (rounds 3–5 design review)

`config.py`: `la_mode ∈ {legacy, snr_m, post_rzf}` (empty string derives
from `mu_aware_la` for backward compatibility), `decode_order ∈
{layer_major, rbg_major}`, and `validate_la()` which **forbids
post_rzf + layer_major** (under layer-major the final RBG group is unknown
until all layers are decoded, so exact budget accounting would need
retroactive refunds — the very mismatch the redesign removes). Env
validates at construction; `train_phase2` validates at CLI parse.

## 3. `la_planner.py` — shared post-RZF link-adaptation planner (NEW module)

The core of the redesign. One implementation, used **identically** by the
env, every baseline, and the PPO decoder:

- `predict_group_link_adaptation()`: RZF W from h_hat (same
  `rzf_precoder`, same α = σ², same power split as the actual
  transmission) → predicted post-RZF SINR of the FINAL group → B_tx cap.
  In the genie world prediction ≡ realization (Gate 1: 21,593 new units,
  depth 1–4, first-ACK 100.00%).
- `SlotAllocationPlanner.close_rbg()`: RBG-major group closure —
  ε-dropped members shrink the group via a fixed-point loop, fixed retx
  members shape W but keep their historical B_tx (immutable), budgets
  debit exactly once per RBG. Exports (`planned_btx_map`) feed Gate 2.
- **Canonical member order** inside `close_rbg` (fixed members sorted):
  per-user SINR is permutation-invariant mathematically but only to
  ~1e-12 in floating point; canonicalization made scheduler-side and
  env-side predictions bit-identical (Gate 2 max |Δ| = 6.4e-12 → 0.0).
- Depth-wise backoff selection (see §7).

## 4. `env.py` — post_rzf unit creation + planner-consistent observation

- `_sanitize_and_create_post_rzf()`: RBG-major scan of the allocation with
  spec-§5 closure semantics, group finalized per RBG through the shared
  planner; `last_actual_btx` / `last_env_planner` recorded for Gate 2.
- `get_observation()` now includes `noise_var`, so a scheduler can rebuild
  `h_hat_slot` bit-identically (`reconstruct_h_hat` is a pure function of
  obs fields + noise_var) — this is what lets the PPO decoder run the same
  planner from the stored trajectory during replay.
- HARQ/LA instrumentation counters in `_finish_slot` (pure counters, no
  reward/RNG effect): new units, first-attempt ACKs, attempts per ACKed
  unit, per-depth unit/ACK counts, pinned positions, completed payload
  bits.

## 5. `baselines.py` — rbg_major traversal with exact budget threading

`Scheduler._schedule_rbg_major()`: same traversal/closure as the env scan.
Under post_rzf the per-UE budget is the planner's `_remaining` (debited at
closure → threaded budget ≡ env actual commit); under legacy it keeps the
per-pick `estimate_btx` debit — that combination is the **order-effect
control** (legacy + rbg_major), which showed SU baselines bit-identical to
layer-major (traversal order cannot matter at depth 1) and SUS baselines
−2~−3%. `SUSPFVirtual` got `_slot_init`/`_after_close` hooks so its
virtual-PF bump uses finalized B_tx.

## 6. `policy.py` — one decode/replay code path (`_rbg_major_pass`)

Rollout decode and PPO-update replay share a single rbg-major
implementation (teacher-forcing is a parameter, not a second code path), so
rollout/replay can only diverge through the weights. Budget features/masks
come from the planner's float64 remaining; SU-CQI `pred_btx` stays as a
*feature only* (debits happen at closure with final values). Gate 3:
max |Δlogp| = 0.0 over 11,891 decisions, budget traces bit-equal. Because a
shared path can hide shared bugs, an **independent closed-form reference**
(`_analysis/scripts/audit_probes/independent_reference_test.py`: orthogonal
groups SINR_i = g_i²P/(mσ²), hand-traced ε-drop/swallow-tail/debit) guards
the math core, alongside Gate 1's physical oracle.

## 7. Depth-wise backoff β_m (rounds 6–7) + global-β kept as ablation

- `config.la_beta` (scalar) — global-β **ablation** mode. Round-6 audit
  proved (analytically: for m = 1 the predicted post-RZF SINR *equals* the
  fed-back SNR by `reconstruct_h_hat`'s definition) that a global
  β = 0.6469 cuts SU rate by exactly 35% → the SU+CQI collapse to 1246.
- `config.la_beta_by_depth` — official mode: β_m = (0.9815, 0.7306,
  0.6466, 0.5922), the per-depth 10th percentile of MI_actual/cap_pred
  from a **scheduler-independent** calibration (36 episodes advanced with
  empty allocations, uniform random groups; same β_m for every scheduler;
  genie = 1.0). Holdout validation on 12 disjoint episodes, β_m frozen:
  m1 88.8%, m2–m4 93.5–94.8%, with **episode-cluster bootstrap** CIs
  (member-level binomial CIs overstate precision ~3×). Script:
  `_analysis/scripts/audit_probes/beta_m_calib_holdout.py`.
- `--la_beta` / `--la_beta_by_depth` CLI; both in the resume warn-list.

## 8. `train_phase2.py` — instrumentation columns + the empty-bin lesson

post_rzf runs (only — legacy/live CSV headers untouched) log
`first_ack_rate`, `attempts_per_acked`, `pinned_fraction`, `goodput_mbps`
and per-depth triples `acks_m*/units_m*/first_ack_m*`.

**Bug found by audit round 6:** my first per-depth rate used
`fa/max(ub, 1)` and was averaged over episodes — episodes with an *empty*
depth bin contributed phantom 0%s, producing the spurious
"SUS m1 first-ACK 36%" (true conditional rate on n = 179 units: 99.4%).
Fix: empty bins are **NaN**, raw numerator/denominator columns are always
stored, and multi-episode aggregation must pool Σacks/Σunits (never average
per-episode rates).

## 9. Reproducibility hardening (round 6–7)

- `train_phase2._git_state()`: every run stamps `git_hash` +
  `git_dirty_py` (tracked `*.py` only — run outputs are tracked too and are
  always dirty during live runs) into `config.json` and every checkpoint.
- A **fresh** run refuses to start with uncommitted `.py` changes
  (`--allow_dirty` escape for throwaway experiments); resume is never
  blocked at this layer (auto-resume wrappers must survive unrelated
  edits).
- `Run4/_wrap_queuepostrzf.sh` (the pilot's wrapper) closes the resume
  hole: every launch **and** resume verifies (a) HEAD's root `*.py` tree
  identical to the pinned redesign commit `efcfac6`, (b) clean root-`*.py`
  worktree, (c) the checkpoint's stamped commit has an identical root-py
  tree. Mismatch → idle HOLD loop (tmux session stays alive, so the
  watchdog does not zombie-churn); `ALLOW_HASH_MISMATCH=1` is the explicit
  override. Docs/analysis commits don't trip the pin; any training-code
  drift does.

## 10. Verification artifacts (all in `_analysis/scripts/audit_probes/`)

`gate23_post_rzf.py` (Gates 1–3 + retx immutability),
`independent_reference_test.py`, `beta_m_calib_holdout.py` (+ cluster
bootstrap), `control_lm_rm.py` (order-effect control),
`new_la_baselines.py` (4-row world decomposition; results in
`_analysis/new_la_baselines{,_betam}.csv`).

| gate | result |
|---|---|
| G1 genie first-ACK (new units, depth 1–4) | 100.00% (21,593 units) |
| G2 planned vs actual commit | max \|Δ\| = 0.0 (86,986 units, keyset mismatches 0) |
| G3 rollout vs replay | max \|Δlogp\| = 0.0 (11,891 decisions), budget traces bit-equal |
| retx immutability | 0 B_tx changes, 0 RBG pin moves |
| legacy bit-exactness | CSV-identical through all 7 edit blocks |

First consumer: `Run4/QueuePostRZF` (pilot, GPU0, seed 2024, launched
2026-07-13, code pinned to `efcfac6` via commit `22bda54`).
