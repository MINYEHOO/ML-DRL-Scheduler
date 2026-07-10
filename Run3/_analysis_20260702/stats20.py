"""20-seed statistical reinforcement for a STOPPED run (argv[1]).

PPO best.pt (deterministic) vs the run's true top baselines on held-out seeds
10000-10019, SEED-MAJOR loop (all schedulers share one seed's cached channel
before moving on -- the env LRU holds 16 entries so scheduler-major would
regenerate). Paired per-seed differences -> mean, 95% CI, win count.
Per-episode rows stream to stats20_<run>.csv so partial progress is usable.
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
from baselines import SUCQI, SUSCQI, CQIGreedy         # noqa: E402
from train_phase2 import env_episode_metrics, PPOScheduler   # noqa: E402

RUN = sys.argv[1]
BASELINES = {
    "Uniform10_Ent002": [("SU-CQI", SUCQI(), None),
                         ("SUS-CQI@0.8", SUSCQI(), 0.8)],
    "DeadlineScarcity": [("SU-CQI", SUCQI(), None),
                         ("SUS-CQI@0.8", SUSCQI(), 0.8)],
    "ScarcityK32": [("SUS-CQI@0.8", SUSCQI(), 0.8),
                    ("CQI-greedy", CQIGreedy(), None),
                    ("SU-CQI", SUCQI(), None)],
}[RUN]
SEEDS = list(range(10000, 10020))

base = f"/home/MYH/ML_DRL_Scheduler/Run3/{RUN}"
cj = json.load(open(f"{base}/config.json"))
fields = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in cj.items() if k in Config.__dataclass_fields__}
cfg = Config(**fields)
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ck = torch.load(f"{base}/ckpt/best.pt", map_location="cpu")
ac.load_state_dict(ck["model"])
ac.eval()
print(f"[{RUN}] best.pt update {ck['update']} eval {ck.get('eval_reward'):.1f} "
      f"K={cfg.num_ue}", flush=True)

scheds = [("PPO-best", PPOScheduler(ac, deterministic=True), None)] + BASELINES
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"stats20_{RUN}.csv")
out = open(out_path, "w", newline="")
w = csvmod.writer(out)
w.writerow(["scheduler", "seed", "reward", "thr_mbps", "comp", "drop"])

res = {name: [] for name, _, _ in scheds}
t0 = time.time()
for seed in SEEDS:
    for name, sched, th in scheds:
        cfg.sus_ortho_threshold = th if th is not None else 0.5
        env.reset(seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        m = env_episode_metrics(env, cfg)
        w.writerow([name, seed, m["reward"], m["throughput_mbps"],
                    m["completion_rate"], m["retx_drop_rate"]])
        out.flush()
        res[name].append(m["reward"])
        print(f"[{RUN}] seed {seed} {name:12s} {m['reward']:8.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== {RUN}: {len(SEEDS)}-seed summary ===")
for name, r in res.items():
    r = np.array(r)
    print(f"  {name:14s} mean {r.mean():8.1f} +- {r.std(ddof=1):6.1f}")
ppo = np.array(res["PPO-best"])
for name, r in res.items():
    if name == "PPO-best":
        continue
    d = ppo - np.array(r)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  PPO - {name}: {d.mean():+8.1f} (95% CI +-{1.96*se:.1f} paired)  "
          f"wins {int((d > 0).sum())}/{len(d)}  rel {d.mean()/np.mean(r):+.2%}")
print(f"[{RUN}] done in {time.time()-t0:.0f}s -> {out_path}")
