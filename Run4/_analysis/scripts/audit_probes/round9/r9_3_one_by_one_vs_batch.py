"""Round-10b maximality check v2: batch epsilon-drop vs EXHAUSTIVE oracle.

v2 corrections (audit round 10b): (1) env-side closures only (v1 hooked the
planner class-wide and double-counted scheduler+env, reporting 482 events =
2x ~241 unique); (2) v1's "same" only compared CARDINALITY (one-by-one did
not keep MORE); v2 compares against an exhaustive subset oracle -- with
l_max = 4 there are at most 2^4-1 = 15 non-empty new-member subsets per
closure, so full enumeration is cheap and definitive. For every env-side
closure where the batch pass dropped at least one member, the oracle finds
all feasible subsets (S is feasible iff, on the group fixed+S, every u in S
has min(remaining_u, cap_u) >= epsilon) and reports:
  - whether the batch kept-set is a maximum-cardinality feasible subset
  - whether any feasible subset beats it on cardinality (true maximality gap)
  - whether the batch B_tx values match a recomputation on its own set
  - oracle best = (max cardinality, then max total bits)

Run from the repo root:
  PYTHONPATH=. python3 Run4/_analysis/scripts/audit_probes/round9/r9_3_one_by_one_vs_batch.py
"""
import itertools

import numpy as np

from config import phase4_queue_config
from env import SchedulerEnv
from baselines import SUSCQI, CQIGreedy, Random
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation

cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                          la_beta_by_depth=(0.9815, 0.7306, 0.6466, 0.5922),
                          p_arrival_min=0.15, p_arrival_max=0.40)
env = SchedulerEnv(cfg)

CMP = dict(drop_events=0, batch_is_max_cardinality=0, oracle_beats_batch=0,
           batch_btx_mismatch=0, oracle_extra_members=0,
           oracle_extra_bits=0.0)
orig_close = SlotAllocationPlanner.close_rbg
IN_ENV = [False]
_orig_create = SchedulerEnv._sanitize_and_create_post_rzf


def _create_hook(self, allocation):
    IN_ENV[0] = True
    try:
        return _orig_create(self, allocation)
    finally:
        IN_ENV[0] = False


SchedulerEnv._sanitize_and_create_post_rzf = _create_hook


def subset_eval(planner, r, fixed, subset, remaining0):
    """caps/b for group fixed+subset with pre-closure budgets; None if any
    subset member falls below epsilon. Mirrors the batch rule EXACTLY,
    including the sub-epsilon tail swallow (b = remaining when the leftover
    tail would be < epsilon) -- without it, kept-and-swallowed members
    show phantom B_tx mismatches."""
    grp = fixed + list(subset)
    h = planner.h_hat_slot[grp, r, :]
    _, _, caps = predict_group_link_adaptation(
        h, planner.noise_var, planner.alpha, cfg.p_rbg, cfg)
    bs = {}
    for i, u in enumerate(grp):
        if u in fixed:
            continue
        b = min(remaining0[u], float(caps[i]))
        if b < cfg.b_tx_epsilon:
            return None
        if remaining0[u] - b < cfg.b_tx_epsilon:
            b = remaining0[u]               # swallow rule (la_planner.py)
        bs[u] = b
    return bs


def hooked(self, r, fixed_ues, new_ues):
    pre_remaining = self._remaining.copy()
    kept, btx = orig_close(self, r, fixed_ues, new_ues)
    if IN_ENV[0]:
        fixed = sorted(set(int(u) for u in fixed_ues))
        cand = list(dict.fromkeys(u for u in new_ues
                                  if pre_remaining[u] > 0.0))
        if len(kept) < len(cand):           # batch dropped someone
            CMP["drop_events"] += 1
            best = (0, 0.0)
            for k in range(len(cand), 0, -1):
                found = False
                for S in itertools.combinations(cand, k):
                    bs = subset_eval(self, r, fixed, S, pre_remaining)
                    if bs is not None:
                        found = True
                        tot = sum(bs.values())
                        if (k, tot) > best:
                            best = (k, tot)
                if found:
                    break                   # max cardinality found
            if len(kept) >= best[0]:
                CMP["batch_is_max_cardinality"] += 1
            else:
                CMP["oracle_beats_batch"] += 1
                CMP["oracle_extra_members"] += best[0] - len(kept)
                CMP["oracle_extra_bits"] += best[1] - sum(btx.values())
            if kept:                        # batch B_tx self-consistency
                bs = subset_eval(self, r, fixed, tuple(kept), pre_remaining)
                if bs is None or any(abs(bs[u] - btx[u]) > 1e-9
                                     for u in kept):
                    CMP["batch_btx_mismatch"] += 1
    return kept, btx


if __name__ == "__main__":
    SlotAllocationPlanner.close_rbg = hooked
    for sch in (SUSCQI(), CQIGreedy(), Random(seed=1)):
        for s in range(10000, 10008):
            env.reset(episode_idx=s)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
    SlotAllocationPlanner.close_rbg = orig_close
    print("env-side drop events            :", CMP["drop_events"])
    print("batch == max-cardinality subset :",
          CMP["batch_is_max_cardinality"])
    print("oracle beats batch (cardinality):", CMP["oracle_beats_batch"],
          f"(extra members {CMP['oracle_extra_members']}, "
          f"extra bits {CMP['oracle_extra_bits']:.3f})")
    print("batch B_tx recompute mismatches :", CMP["batch_btx_mismatch"])
    assert CMP["batch_btx_mismatch"] == 0, "batch B_tx not self-consistent"
