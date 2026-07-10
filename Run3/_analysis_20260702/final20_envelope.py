"""FINAL 20-seed hybrid/envelope evaluation for a mixed-regime run (argv[1]).

The paper-grade replacement of the 3-seed defense experiment: on seeds
10000-10019, evaluate pure SU-CQI, pure SUS-CQI@0.8, HybridSwitch(T in
{10,14,18,22}), and PPO best.pt (deterministic). Seed-major for channel-cache
reuse. Reports paired PPO-vs-envelope and PPO-vs-SUS CIs. Per-episode rows
stream to final20_<run>.csv.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv as csvmod          # noqa: E402
import json                   # noqa: E402
import time                   # noqa: E402

import numpy as np            # noqa: E402
import torch                  # noqa: E402

torch.set_num_threads(6)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                              # noqa: E402
from env import SchedulerEnv                           # noqa: E402
from policy import ActorCritic                         # noqa: E402
from baselines import SUCQI, SUSCQI                    # noqa: E402
from train_phase2 import env_episode_metrics, PPOScheduler   # noqa: E402

RUN = sys.argv[1]              # MixedLoad_L2 | MixedSpeed_L2b
SEEDS = list(range(10000, 10020))
T_SWEEP = (10, 14, 18, 22)


class HybridSwitch:
    def __init__(self, T):
        self.T = T
        self.name = f"Hybrid<={T}"
        self.su = SUCQI()
        self.sus = SUSCQI()

    def schedule(self, env):
        n = int(env.get_observation()["active"].sum())
        return (self.su if n <= self.T else self.sus).schedule(env)


base = f"/home/MYH/ML_DRL_Scheduler/Run3/{RUN}"
cj = json.load(open(f"{base}/config.json"))
fields = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in cj.items() if k in Config.__dataclass_fields__}
cfg = Config(**fields)
cfg.sus_ortho_threshold = 0.8
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ck = torch.load(f"{base}/ckpt/best.pt", map_location="cpu")
ac.load_state_dict(ck["model"])
ac.eval()
print(f"[{RUN}] best.pt update {ck['update']} (3-seed eval "
      f"{ck.get('eval_reward'):.1f}), critic_v2={cfg.ppo_critic_v2}", flush=True)

scheds = ([("SU-CQI", SUCQI()), ("SUS-CQI@0.8", SUSCQI())]
          + [(h.name, h) for h in (HybridSwitch(T) for T in T_SWEEP)]
          + [("PPO-best", PPOScheduler(ac, deterministic=True))])
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"final20_{RUN}.csv"), "w", newline="")
w = csvmod.writer(out)
w.writerow(["scheduler", "seed", "reward", "thr_mbps", "comp", "drop", "depth"])

per = {name: {} for name, _ in scheds}
t0 = time.time()
for seed in SEEDS:
    for name, sched in scheds:
        env.reset(seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        m = env_episode_metrics(env, cfg)
        w.writerow([name, seed, m["reward"], m["throughput_mbps"],
                    m["completion_rate"], m["retx_drop_rate"], m["mu_depth"]])
        out.flush()
        per[name][seed] = m["reward"]
        print(f"[{RUN}] seed {seed} {name:12s} {m['reward']:8.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== {RUN}: FINAL 20-seed summary ===")
for name, _ in scheds:
    r = np.array([per[name][s] for s in SEEDS])
    print(f"  {name:14s} {r.mean():8.1f} +- {r.std(ddof=1):6.1f}")
envl = np.array([max(per["SU-CQI"][s], per["SUS-CQI@0.8"][s]) for s in SEEDS])
ppo = np.array([per["PPO-best"][s] for s in SEEDS])
best_h = max(((n, np.mean([per[n][s] for s in SEEDS]))
              for n, _ in scheds if n.startswith("Hybrid")), key=lambda x: x[1])
print(f"  {'oracle-envelope':14s} {envl.mean():8.1f}  | best hybrid: "
      f"{best_h[0]} ({best_h[1]:.1f})")
for tag, ref in (("SUS-CQI@0.8", np.array([per["SUS-CQI@0.8"][s] for s in SEEDS])),
                 ("best-hybrid", np.array([per[best_h[0]][s] for s in SEEDS])),
                 ("oracle-envelope", envl)):
    d = ppo - ref
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    v = "유의한 승" if lo > 0 else ("유의한 패" if hi < 0 else "동률")
    print(f"  PPO - {tag}: {d.mean():+8.1f} CI[{lo:+.0f},{hi:+.0f}] "
          f"{int((d > 0).sum())}/{len(d)}승 ({d.mean()/ref.mean():+.2%}) -> {v}")
print(f"[{RUN}] done in {time.time()-t0:.0f}s")
