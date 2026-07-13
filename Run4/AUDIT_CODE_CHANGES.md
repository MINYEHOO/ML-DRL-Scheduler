# Code changes from the external-audit era (2026-07-10 → 2026-07-13)

Scope: **code** changes that came out of publishing the repo and the eight
rounds of external (GPT) audit + probe-based adjudication. Pure
documentation edits are excluded on purpose — those corrected *wording that
invited misreading*, not simulator behavior (adjudication records:
`_analysis/gpt_audit_adjudication.md`, `_analysis/gpt_audit_round2_adjudication.md`,
`docs/RUNS.md` §4.5). Every entry below changed what the code *does* or what
it *records*.

Ground rule that held throughout: the legacy default configuration stays
**bit-exact** — CSV-identical 3-update debug runs (`--mode debug --seed 7`)
against the pre-change reference built from `Run3/code_backup/` (the frozen
pre-redesign tree), re-checked after every edit block (8 blocks total) —
because two live trainings auto-resume from disk with whatever code is
checked out.

---

## 1. `mu_aware_la` — SNR/m link-adaptation flag (round 1–2, C-cluster)

**Problem found by audit:** B_tx for a new unit was sized from the
full-power SU CQI even when m > 1 streams share the RBG power, making
**cap-limited** depth ≥ 2 first transmissions structurally NACK
(backlog-capped units send less than the cap and can still first-ACK). The
required accumulated-MI-to-B_tx ratio is 1.34/1.64/1.91 for m = 2/3/4 at
10 dB — i.e. the first slot delivers only 1/1.34… of the target, not a
statement about integer HARQ round counts.

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
implementation (teacher-forcing is a parameter, not a second code path).
Budget features/masks come from the planner's float64 remaining; SU-CQI
`pred_btx` stays as a *feature only* (debits happen at closure with final
values). Gate 3: under identical obs/config/weights and the tested
execution conditions (CPU, deterministic threading), decode and replay
agreed exactly — max |Δlogp| = 0.0 over 11,820 decisions, budget traces
bit-equal. Because a shared path can hide shared bugs, an **independent
closed-form reference**
(`_analysis/scripts/audit_probes/independent_reference_test.py`: orthogonal
groups SINR_i = g_i²P/(mσ²), hand-traced ε-drop/swallow-tail/debit) guards
the math core, alongside Gate 1 — which is itself a *simulator PHY-chain
consistency gate* (prediction vs realization inside the same PHY
implementation), not an external oracle for the RZF math.

## 7. Depth-wise backoff β_m (rounds 6–7) + global-β kept as ablation

- `config.la_beta` (scalar) — global-β **ablation** mode. Round-6 audit
  proved (analytically: for m = 1 the predicted post-RZF SINR *equals* the
  fed-back SNR by `reconstruct_h_hat`'s definition) that a global
  β = 0.6469 cuts the **cap-limited m = 1 B_tx ceiling** by exactly
  35.31% (backlog-limited units and ε-tail swallowing are not scaled by β)
  → the SU+CQI collapse to 1246.
- `config.la_beta_by_depth` — official mode: β_m = (0.9815, 0.7306,
  0.6466, 0.5922), the per-depth 10th percentile of MI_actual/cap_pred
  from a **scheduler-independent** calibration (36 episodes advanced with
  empty allocations, uniform random groups; same β_m for every scheduler;
  genie = 1.0). Correct characterization: *independently calibrated
  depth-dependent margins targeting approximately 10% first-transmission
  BLER* — not exactly-10%-everywhere. Holdout on 12 disjoint episodes,
  β_m frozen: m1 88.8%, m2–m4 93.5–94.8% pooled, with **episode-cluster
  bootstrap** CIs (member-level binomial CIs overstate precision ~1.2×
  for m = 1 and ~3.3–4.1× for m = 2–4, because samples within an episode
  share its channel mixture). Script (self-reproducing end-to-end):
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

## 9. Reproducibility hardening (rounds 6–8)

- `train_phase2._git_state()`: every run stamps `git_hash` +
  `git_dirty_py` into `config.json` and every checkpoint. `git_dirty_py`
  covers **tracked-modified and untracked** `*.py` anywhere in the repo
  (`git status --porcelain -- '*.py'`); run outputs are excluded because
  they are tracked non-py files, always dirty during live runs. On any git
  error the stamp **fails open** (`"no-git"/"unknown"`, dirty = False) —
  stamping must never break training, so the wrapper pin below is the
  enforcing layer.
- A **fresh** run refuses to start with uncommitted `.py` changes
  (`--allow_dirty` escape for throwaway experiments); resume is never
  blocked at this layer (auto-resume wrappers must survive unrelated
  edits).
- `Run4/_wrap_queuepostrzf.sh` (the official run's wrapper) closes the
  resume hole: every launch **and** resume verifies (a) HEAD's
  **root-level `*.py` tree** identical to the pinned baseline commit,
  (b) clean root-`*.py` worktree, (c) the checkpoint's stamped commit has
  an identical root-py tree. Scope is exactly that: the flat root-level
  Python modules are the entire runtime import surface today, but the pin
  does NOT cover the wrapper itself, CLI arguments, dependencies, or any
  future sub-directory modules. Mismatch → idle HOLD loop that keeps
  touching the run's `.wd_marker` every 30 s so the hang-watchdog
  (`Run3/_watchdog.sh`, HANG_MIN-based kill+relaunch) does not churn the
  HOLD — verified by an integration test with `HANG_MIN=1 CHECK_SEC=10`:
  the new HOLD survives untouched while a no-marker control variant gets
  kill/relaunch-churned. `ALLOW_HASH_MISMATCH=1` is the explicit override;
  it cannot be injected into a running shell — apply it by relaunching the
  session deliberately.
- Commit roles: `efcfac6` = the post-RZF redesign (root-py baseline of the
  first pilot attempt); `22bda54` = round-7 ops/docs (pin + cluster CI, no
  root-py change); the round-8 fix commit = the **current** root-py
  baseline that `QueuePostRZF` is pinned to (snr_m routing fix — see §11).

## 10. Verification artifacts (all in `_analysis/scripts/audit_probes/`)

`gate1_genie_first_ack.py` (Gate 1 full: depth bins asserted non-empty,
full-cap/backlog-cap separated), `gate23_post_rzf.py` (Gates 2–3 + retx
immutability + a depth-1 Gate-1 supplement), `independent_reference_test.py`,
`beta_m_calib_holdout.py` (calibration → holdout → Wilson + cluster
bootstrap, end-to-end in one run), `control_lm_rm.py` (order-effect
control), `new_la_baselines.py` (BOTH post-RZF worlds; per-baseline
summaries with pooled per-depth rates + `*_raw.csv` world×scheduler×seed
rows). Superseded outputs are kept, clearly labeled, in
`_analysis/superseded/` (the global-β summary whose per-depth columns
carried the empty-bin artifact).

| gate | result |
|---|---|
| G1 genie first-ACK (new units, depth 1–4 all non-empty) | 100.00% (21,593 units) |
| G2 planned vs actual commit | max \|Δ\| = 0.0 (86,981 units, keyset mismatches 0) |
| G3 rollout vs replay (same obs/config/weights) | max \|Δlogp\| = 0.0 (11,820 decisions), budget traces bit-equal |
| retx immutability | 0 B_tx changes, 0 RBG pin moves |
| legacy bit-exactness | CSV-identical through all 8 edit blocks (reference: `Run3/code_backup/`) |

## 11. Round 8 — `snr_m` routing fix + verification-artifact consolidation

External review of THIS ledger (round 8) found one real dormant bug and
several stale checked-in artifacts; all fixed:

- **Bug:** `env.py`'s SNR/m branch still gated on the raw `mu_aware_la`
  flag, so `la_mode="snr_m"` alone silently ran legacy, and
  `la_mode="legacy"` + `mu_aware_la=True` silently de-rated. Fix: gate on
  `resolved_la_mode() == "snr_m"`. Four-combination regression (3-update
  debug runs, CSV-compared): `("", False)` ≡ legacy reference;
  `("legacy", True)` ≡ legacy reference; `("snr_m", False)` ≡
  `("", True)`; and `("", True)` ≠ legacy on env/ppo metrics (the de-rate
  demonstrably routes). No live or completed run ever used the broken
  combinations (`QueueMixedFairLA` sets `mu_aware_la=True`, post_rzf runs
  set `la_mode="post_rzf"` — both behavior-identical across the fix).
- Verification scripts re-checked-in as final versions and re-run from the
  repo (previous copies were pre-final snapshots; results unchanged).
- HOLD/watchdog interaction fixed + integration-tested (see §9).

First consumer: `Run4/QueuePostRZF` (GPU0, seed 2024, restarted fresh
2026-07-13 on the round-8 commit after the snr_m fix — the earlier
same-day attempt on `efcfac6`/`22bda54` was archived, not resumed, so the
official run's entire history lives on one immutable code baseline).
