"""Re-evaluation with CORRECTED JFI = Jain over ACTIVE UEs only (user directive
2026-07-06). All other metrics reproduce the deterministic episodes exactly;
jain_allk kept as an audit column. argv[1] = run name."""
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
from metrics import jains_index                        # noqa: E402
from baselines import (SUSCQI, SUSDeadlinePF, SUSPF, SUSRandom,   # noqa: E402
                       SUCQI, SUDeadlinePF, SUPF, SURandom)
from train_phase2 import env_episode_metrics, PPOScheduler   # noqa: E402

RUN = sys.argv[1]
SEEDS = list(range(10000, 10020))
base = f"/home/MYH/ML_DRL_Scheduler/Run3/{RUN}"
cj = json.load(open(f"{base}/config.json"))
fields = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in cj.items() if k in Config.__dataclass_fields__}
cfg = Config(**fields)
cfg.sus_ortho_threshold = 0.8          # SUS family at the swept optimum
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ck = torch.load(f"{base}/ckpt/best.pt", map_location="cpu")
ac.load_state_dict(ck["model"])
ac.eval()
print(f"[{RUN}] best.pt update {ck['update']}", flush=True)

SCHEDS = [("SUS+CQI", SUSCQI()), ("SUS+DPF", SUSDeadlinePF()),
          ("SUS+PF", SUSPF()), ("SUS+Rnd", SUSRandom(seed=cfg.seed + 100_003)),
          ("SU+CQI", SUCQI()), ("SU+DPF", SUDeadlinePF()),
          ("SU+PF", SUPF()), ("SU+Rnd", SURandom(seed=cfg.seed + 100_004)),
          ("PPO", PPOScheduler(ac, deterministic=True))]

out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"official9full_{RUN}.csv"), "w", newline="")
w = csvmod.writer(out)
w.writerow(["scheduler", "seed", "reward", "thr_mbps", "sinr_db", "comp",
            "miss", "retx_drop", "total_fail", "depth", "jain", "jain_allk"])
t0 = time.time()
for seed in SEEDS:
    for name, sched in SCHEDS:
        env.reset(seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        m = env_episode_metrics(env, cfg)
        n_act = int(env.traffic.n_active)
        jain_active = jains_index(env.ep["acked_per_ue"][:n_act])
        arr = max(m["n_arrivals"], 1)
        total = (m["deadline_miss_rate"] + m["retx_drop_rate"]
                 + m["retx_overflow_drop"] / arr)
        w.writerow([name, seed, m["reward"], m["throughput_mbps"],
                    m["mean_sinr_db"], m["completion_rate"],
                    m["deadline_miss_rate"], m["retx_drop_rate"], total,
                    m["mu_depth"], jain_active, m["jain"]])
        out.flush()
        print(f"[{RUN}] seed {seed} {name:8s} rew {m['reward']:8.1f} "
              f"jain_act {jain_active:.3f} ({time.time()-t0:.0f}s)", flush=True)
print("done")
