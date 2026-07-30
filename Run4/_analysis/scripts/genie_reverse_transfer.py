"""Reverse transfer: genie-trained policies back in the imperfect world.

User question (2026-08-01): the GenieS40HL pair improved in the perfect-CSI
world (FT 8014 / Fresh 7502 vs SUS 7473). If the genie-fine-tuned policy
still performs in the ORIGINAL imperfect world, that is a signal for the
"train on hindsight-reconstructed past traffic, deploy on future" idea.
Hypothesis to test: genie training erodes the stale-CSI discounting
(depth moderation) that made the original HighLoad policy win.

Protocol: original QueuePostRZF_S40HighLoad world (p_csi 0.6, type2 56-bit,
beta_m S40), held-out seeds 10000-10019 -- same seeds as the final20 eval,
so PPO-HL (5352) and SUS+CQI@0.7 (4668) per-seed anchors come from
queue_s40highload_final20.csv without re-running.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch, csv
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/genie_reverse_transfer.csv"
SEEDS = list(range(10000, 10020))
raw = json.load(open("Run4/QueuePostRZF_S40HighLoad/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw["la_beta_by_depth"])
raw["ue_speed_mix"] = ()
cfg = Config(**raw)
env = SchedulerEnv(cfg)

POLICIES = [
    ("GenieFT@329",   "Run4/GenieS40HL_FineTune/ckpt/best.pt"),
    ("GenieFresh@409", "Run4/GenieS40HL_Fresh/ckpt/best.pt"),
]
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["policy", "seed", "reward", "goodput_mbps", "deadline_miss_rate",
            "mu_depth"])
for name, path in POLICIES:
    ck = torch.load(path, map_location="cpu")
    ac = ActorCritic(cfg)
    ac.load_state_dict(ck["model"], strict=True)
    ac.eval()
    sch = PPOScheduler(ac, deterministic=True)
    rs = []
    for seed in SEEDS:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        rs.append(m)
        w.writerow([name, seed, round(m["reward"], 1),
                    round(m.get("goodput_mbps", float("nan")), 2),
                    round(m["deadline_miss_rate"], 4),
                    round(m["mu_depth"], 3)])
        print(f"{name} seed {seed}: rew {m['reward']:7.0f} depth "
              f"{m['mu_depth']:.2f}", flush=True)
    print(f"== {name}: mean reward {np.mean([m['reward'] for m in rs]):.0f} "
          f"good {np.mean([m['goodput_mbps'] for m in rs]):.1f} "
          f"miss {np.mean([m['deadline_miss_rate'] for m in rs]):.3f} "
          f"depth {np.mean([m['mu_depth'] for m in rs]):.2f}", flush=True)
print("anchors (same seeds, final20 csv): PPO-HL 5352 / SUS+CQI@0.7 4668")
print("saved ->", OUT)
