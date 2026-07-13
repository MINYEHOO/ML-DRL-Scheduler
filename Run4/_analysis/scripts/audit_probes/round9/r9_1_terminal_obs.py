"""Probe: freshness of the observation returned at done=True.

Runs a short debug episode with the CQIGreedy baseline, snapshots the
second-to-last obs (s_{T-1}, prepared) and the final obs (returned at done),
and compares:
  1. final obs fixed_mask.sum()  vs  actual pending retx units in txmgr
     (what a real compaction at slot T would pin)
  2. final obs CSI age vs the previous obs age (stale = identical object state)
  3. which obs fields changed between s_{T-1} and the final obs
  4. whether _prepare_slot(T) / true s_T is even constructible (index bounds)
Repeats over several episodes for the fixed_mask mismatch statistic.
"""
import numpy as np
from config import debug_config
from env import SchedulerEnv
from baselines import CQIGreedy

cfg = debug_config()
env = SchedulerEnv(cfg)
sched = CQIGreedy()

mismatches = []
for ep in range(5):
    obs = env.reset(episode_idx=ep)
    done = False
    prev_obs = None
    while not done:
        prev_obs = obs
        alloc = sched.schedule(env)
        obs, r, done, info = env.step(alloc)

    final_obs = obs
    n_pending_true = len(env.txmgr.units)          # actual pending retx after last slot
    n_mask = int(final_obs["fixed_mask"].sum())     # stale mask from prev compaction
    # what a compaction NOW would pin (without mutating env state further than
    # compact_pending does -- it may overflow-drop; run on a snapshot count only)
    mismatches.append((ep, n_mask, n_pending_true))
    if ep == 0:
        print(f"episode_len={cfg.episode_len}, final obs slot={final_obs['slot']}")
        # field-by-field: which parts changed from s_{T-1} obs to final obs
        same, diff = [], []
        for k in final_obs:
            a, b = prev_obs[k], final_obs[k]
            if k == "initial_S_r":
                eq = all(sa == sb for sa, sb in zip(a, b))
            elif isinstance(a, np.ndarray):
                eq = np.array_equal(a, b)
            else:
                eq = a == b
            (same if eq else diff).append(k)
        print("UNCHANGED vs s_(T-1):", sorted(same))
        print("CHANGED   vs s_(T-1):", sorted(diff))
        print(f"age identical to s_(T-1) obs: "
              f"{np.array_equal(prev_obs['age'], final_obs['age'])}, "
              f"age min/max = {final_obs['age'].min()}/{final_obs['age'].max()}")
        # is a true s_T constructible?
        try:
            env.channel.get_channel(cfg.episode_len)
            print("get_channel(T): OK (unexpected)")
        except Exception as e:
            print(f"get_channel(T) raises: {type(e).__name__}: {e}")
        try:
            env._episode_csi.slot(cfg.episode_len)
            print("episode_csi.slot(T): OK (no bounds check?)")
        except Exception as e:
            print(f"episode_csi.slot(T) raises: {type(e).__name__}: {e}")

print("\nfixed_mask.sum (stale, in final obs)  vs  pending retx units (truth):")
for ep, nm, nt in mismatches:
    print(f"  ep{ep}: mask={nm:3d}  pending_units={nt:3d}  "
          f"{'MISMATCH' if nm != nt else 'equal'}")
