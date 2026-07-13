"""Claim 3 probe C: on REAL drop events (Random baseline, Run4 point), does
one-at-a-time removal keep more members than the current all-at-once pass?"""
import json
import numpy as np

import la_planner
from config import Config
from env import SchedulerEnv
from baselines import Random

raw = json.load(open("/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF/config.json"))
raw.pop("git_hash", None); raw.pop("git_dirty_py", None)
raw["debug"] = True; raw["episode_len_debug"] = 200
cfg = Config(**raw)

orig = la_planner.SlotAllocationPlanner.close_rbg
CMP = dict(drop_events=0, same=0, one_by_one_kept_more=0, extra_members=0,
           extra_bits=0.0)

def one_by_one(self, r, fixed, new0, rem):
    new = list(new0)
    while True:
        group = fixed + new
        if not group:
            return [], {}
        h = self.h_hat_slot[group, r, :]
        _, _, caps = la_planner.predict_group_link_adaptation(
            h, self.noise_var, self.alpha, self.cfg.p_rbg, self.cfg)
        b = {u: min(rem[u], float(caps[i])) for i, u in enumerate(group)
             if u not in fixed}
        bad = {u: v for u, v in b.items() if v < self.cfg.b_tx_epsilon}
        if not bad:
            return list(new), b
        worst = min(bad, key=bad.get)
        new = [u for u in new if u != worst]

def wrapped(self, r, fixed_ues, new_ues):
    fixed = sorted(set(int(u) for u in fixed_ues))
    eligible = [u for u in new_ues if self._remaining[u] > 0.0]
    rem = self._remaining.copy()
    kept, btx = orig(self, r, fixed_ues, new_ues)
    if eligible and len(kept) < len(eligible):
        CMP["drop_events"] += 1
        k2, b2 = one_by_one(self, r, fixed, eligible, rem)
        if len(k2) > len(kept):
            CMP["one_by_one_kept_more"] += 1
            CMP["extra_members"] += len(k2) - len(kept)
            CMP["extra_bits"] += sum(b2[u] for u in k2 if u not in kept)
        else:
            CMP["same"] += 1
    return kept, btx

la_planner.SlotAllocationPlanner.close_rbg = wrapped

sched = Random(seed=1)
for ep in range(3):
    env = SchedulerEnv(cfg)
    env.reset(episode_idx=10000 + ep)
    for t in range(cfg.episode_len_debug):
        env.step(sched.schedule(env))
print(CMP)
