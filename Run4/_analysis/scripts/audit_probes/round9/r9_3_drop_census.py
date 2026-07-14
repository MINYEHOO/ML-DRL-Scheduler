"""Claim 3 probe B: empirical epsilon-drop / all-drop / ghost-pick rates at
the Run4 operating point, for CQI-greedy (blind), Random (blind) and
SUS-CQI (ortho-gated) baselines. Counts on the ENV-side planner (the one
that decides actual unit creation)."""
import json
import numpy as np

import la_planner
from config import Config
from env import SchedulerEnv
from baselines import CQIGreedy, Random, SUSCQI

raw = json.load(open("Run4/QueuePostRZF/config.json"))
raw.pop("git_hash", None); raw.pop("git_dirty_py", None)
raw["debug"] = True; raw["episode_len_debug"] = 200
cfg = Config(**raw)

STATS = dict(env=dict(closes=0, new_members=0, dropped=0, all_drop=0,
                      all_drop_feasible_singleton=0, drop_iters_gt1=0))
IN_ENV = {"flag": False}

orig = la_planner.SlotAllocationPlanner.close_rbg

def wrapped(self, r, fixed_ues, new_ues):
    eligible = [u for u in new_ues if self._remaining[u] > 0.0]
    kept, btx = orig(self, r, fixed_ues, new_ues)
    if IN_ENV["flag"] and eligible:
        s = STATS["env"]
        s["closes"] += 1
        s["new_members"] += len(eligible)
        dropped = [u for u in eligible if u not in kept]
        s["dropped"] += len(dropped)
        if not kept:
            s["all_drop"] += 1
            # feasibility of a singleton among the dropped?
            for u in dropped:
                _, _, c1 = la_planner.predict_group_link_adaptation(
                    self.h_hat_slot[u:u+1, r, :], self.noise_var,
                    self.alpha, self.cfg.p_rbg, self.cfg)
                if min(self._remaining[u], float(c1[0])) >= self.cfg.b_tx_epsilon:
                    s["all_drop_feasible_singleton"] += 1
                    break
    return kept, btx

la_planner.SlotAllocationPlanner.close_rbg = wrapped

for sched in (CQIGreedy(), Random(seed=1), SUSCQI()):
    for k in STATS["env"]:
        STATS["env"][k] = 0
    total_picks = 0
    n_ep, T = 3, cfg.episode_len_debug
    for ep in range(n_ep):
        env = SchedulerEnv(cfg)
        env.reset(episode_idx=10000 + ep)
        for t in range(T):
            alloc = sched.schedule(env)
            total_picks += int((alloc > 0).sum())
            IN_ENV["flag"] = True
            env.step(alloc)
            IN_ENV["flag"] = False
    s = STATS["env"]
    print(f"{sched.name:12s} slots={n_ep*T} closes(new>0)={s['closes']} "
          f"new={s['new_members']} dropped={s['dropped']} "
          f"(ghost-pick rate {s['dropped']/max(s['new_members'],1):.4%}) "
          f"all-drop RBGs={s['all_drop']} "
          f"all-drop-with-feasible-singleton={s['all_drop_feasible_singleton']}")
