"""Probe 3: staleness error V(mixed final obs) - V(corrected s_T approx)
across several episodes (CQIGreedy rollout for speed, trained critic eval)."""
import dataclasses
import numpy as np
import torch

from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import CQIGreedy

ck = torch.load("Run3/MixedSpeed_L2b/ckpt/best.pt", map_location="cpu")
fields = {f.name for f in dataclasses.fields(Config)}
kw = {k: v for k, v in ck["cfg"].items() if k in fields}
kw["debug"] = True; kw["episode_len_debug"] = 200
cfg = Config(**kw)
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg); ac.load_state_dict(ck["model"]); ac.eval()
sched = CQIGreedy()

errs = []
with torch.no_grad():
    for ep in range(6):
        obs = env.reset(episode_idx=20000 + ep)
        done = False
        while not done:
            obs, r, done, info = env.step(sched.schedule(env))
        lv_mixed = float(ac.state_value(obs))
        res = env.txmgr.compact_pending(env.traffic)
        fo = dict(obs)
        fo["fixed_allocation"] = res["fixed_allocation"]
        fo["fixed_mask"] = res["fixed_mask"]
        fo["fixed_unit_map"] = res["fixed_unit_map"]
        fo["initial_S_r"] = res["initial_S_r"]
        fo["occupied"] = res["fixed_allocation"].copy()
        fo["age"] = obs["age"] + 1.0
        lv_corr = float(ac.state_value(fo))
        errs.append(lv_mixed - lv_corr)
        print(f"ep{ep}: stale_mask={int(obs['fixed_mask'].sum()):3d} "
              f"corr_mask={int(res['fixed_mask'].sum()):3d}  "
              f"V_mixed={lv_mixed:8.3f}  V_corr={lv_corr:8.3f}  "
              f"err={lv_mixed-lv_corr:+7.3f}")
errs = np.array(errs)
print(f"\nstaleness err: mean={errs.mean():+.3f}  std={errs.std():.3f}  "
      f"|max|={np.abs(errs).max():.3f}")
