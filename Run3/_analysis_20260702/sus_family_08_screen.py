"""Screen ALL SUS-family baselines at threshold 0.8 on every Run3 config
(seeds 10000-10002, matching the logged bars). Escalation rule: anything
within 3% of that run's champion gets a full-seed statistical test."""
import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json                   # noqa: E402
import sys                    # noqa: E402
import time                   # noqa: E402

import numpy as np            # noqa: E402

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                              # noqa: E402
from env import SchedulerEnv                           # noqa: E402
from baselines import SUSPF, SUSPFVirtual              # noqa: E402
from train_phase2 import env_episode_metrics           # noqa: E402

# run -> (champion name, champion mean on seeds 10000-2) for distance check
CHAMPS = {
    "Uniform10_Ent002": ("SU-CQI", 7325.2),
    "DeadlineScarcity": ("SU-CQI", 6677.9),
    "ScarcityK32": ("SUS-CQI@0.8", 10178.9),
    "MixedLoad_L2": ("PPO/SUS-CQI@0.8", 9006.4),
    "MixedSpeed_L2": ("PPO/SUS-CQI@0.8", 8938.9),
}
SEEDS = [10000, 10001, 10002]
t0 = time.time()
for run, (champ_name, champ) in CHAMPS.items():
    cj = json.load(open(f"/home/MYH/ML_DRL_Scheduler/Run3/{run}/config.json"))
    fields = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in cj.items() if k in Config.__dataclass_fields__}
    cfg = Config(**fields)
    cfg.sus_ortho_threshold = 0.8
    env = SchedulerEnv(cfg)
    print(f"\n### {run} (champion {champ_name} = {champ:.1f})", flush=True)
    for sched in (SUSPF(), SUSPFVirtual()):
        ms = []
        for seed in SEEDS:
            env.reset(seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sched.schedule(env))
            ms.append(env_episode_metrics(env, cfg))
        rew = np.mean([m["reward"] for m in ms])
        gap = rew / champ - 1.0
        flag = "  << ESCALATE (within 3%)" if gap > -0.03 else ""
        print(f"  {sched.name+'@0.8':16s} rew {rew:8.1f}  "
              f"thr {np.mean([m['throughput_mbps'] for m in ms]):6.2f}  "
              f"comp {np.mean([m['completion_rate'] for m in ms]):.3f}  "
              f"depth {np.mean([m['mu_depth'] for m in ms]):.2f}  "
              f"vs champ {gap:+.1%}{flag}  ({time.time()-t0:.0f}s)", flush=True)
