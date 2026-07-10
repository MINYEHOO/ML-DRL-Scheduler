"""SU-only baseline variants (RBG-restricted to 1 user) — keeps the MU baselines,
adds SU-CQI / SU-PF / SU-Deadline-PF. Answers: is PPO's win just 'go SU', or
'smart SU'? Full metrics + MU-depth on the SAME 12 seeds. READ-ONLY, CPU."""
import numpy as np
from config import phase2_hard_main_config
from env import SchedulerEnv
from baselines import CQIGreedy, PF, DeadlinePF
from metrics import jains_index

SEEDS = list(range(10000, 10012))


def make_su(BaseCls):
    class SU(BaseCls):
        name = "SU-" + BaseCls.name
        def _select(self, env, obs, r, l, cand, sel_ue, closed):
            if len(sel_ue) > 0:          # RBG already has a user -> SU: add no more
                closed[r] = True
                return None
            pick = super()._select(env, obs, r, l, cand, sel_ue, closed)
            if pick is not None:
                closed[r] = True         # 1 user, then close the RBG (SU-MIMO)
            return pick
    return SU


def full(env, sch, seeds):
    R = {k: [] for k in ("reward", "comp", "miss_dl", "retx", "sinr", "jain", "depth", "pos")}
    for s in seeds:
        env.reset(episode_idx=s); done = False; dep = []; pos = []
        while not done:
            a = sch.schedule(env); occ = (a > 0); pos.append(int(occ.sum()))
            for r in range(env.cfg.num_rbg):
                n = int(occ[r].sum())
                if n > 0: dep.append(n)
            _, _, done, _ = env.step(a)
        e = env.ep; arr = max(e["n_arrivals"], 1)
        et = env.cfg.episode_len * env.cfg.slot_duration
        R["reward"].append(e["reward"]); R["comp"].append(e["n_comp"]/arr)
        R["miss_dl"].append(e["n_miss_deadline"]/arr); R["retx"].append(e["n_retx_drop"]/arr)
        R["sinr"].append(10*np.log10(e["sinr_sum"]/e["sinr_count"]) if e["sinr_count"] else float("nan"))
        R["jain"].append(jains_index(env.cum_acked_bits))
        R["depth"].append(np.mean(dep)); R["pos"].append(np.mean(pos))
    return {k: float(np.mean(v)) for k, v in R.items()}


cfg = phase2_hard_main_config()
env = SchedulerEnv(cfg)
print("%-16s %8s %6s %7s %8s %7s %6s %7s %6s" %
      ("scheduler", "reward", "comp", "miss_dl", "retx_dr", "SINRdB", "jain", "UE/RBG", "pos"))
print("-"*82)
for BaseCls in (CQIGreedy, PF, DeadlinePF):
    m = full(env, make_su(BaseCls)(), SEEDS)
    print("%-16s %8.1f %6.3f %7.3f %8.4f %7.2f %6.3f %7.2f %6.1f" %
          ("SU-"+BaseCls.name, m["reward"], m["comp"], m["miss_dl"], m["retx"],
           m["sinr"], m["jain"], m["depth"], m["pos"]))
