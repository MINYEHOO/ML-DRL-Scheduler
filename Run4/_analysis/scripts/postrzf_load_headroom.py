"""Load-headroom probe for the post-RZF world (no training).

Question (owner, 2026-07-16): with retx-drop closed, is the current op point
(p ~ U(0.15,0.40)) leaving enough room ABOVE the strongest baseline for
scheduling intelligence to matter? Method: evaluate the baseline grid and
the FROZEN current-best PPO (QueuePostRZF best.pt, 5122@39) at FIXED
p_arrival in {0.30, 0.40, 0.50}, 4 paired seeds. Rising baseline spread and
rising frozen-PPO deficit/surplus with load indicate how much headroom a
higher-load official world would offer (cf. the legacy OOD probe: +44~136%
at p 0.45/0.50).
"""
import csv, json
import numpy as np, torch

from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler
from baselines import all_baselines

SEEDS = [10000, 10001, 10002, 10003]
LOADS = [0.30, 0.40, 0.50]
OUT = "Run4/_analysis/postrzf_load_headroom.csv"

raw = json.load(open("Run4/QueuePostRZF/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw["la_beta_by_depth"])
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())

ck = torch.load("Run4/QueuePostRZF/ckpt/best.pt", map_location="cpu")
print("frozen policy: best.pt update", ck["update"])

w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["p_arrival", "scheduler", "seed", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate",
            "mu_depth"])
for p in LOADS:
    r = dict(raw)
    r["p_arrival"] = p
    r["p_arrival_min"] = 0.0
    r["p_arrival_max"] = 0.0          # 고정 부하 (mixed-arrival 비활성)
    cfg = Config(**r)
    ac = ActorCritic(cfg); ac.load_state_dict(ck["model"]); ac.eval()
    scheds = all_baselines(cfg) + [PPOScheduler(ac, deterministic=True)]
    names = [getattr(s, "name", "PPO-frozen") for s in scheds]
    for sch, name in zip(scheds, names):
        for s in SEEDS:
            env = SchedulerEnv(cfg)
            env.reset(episode_idx=s)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            em = env_episode_metrics(env, cfg)
            w.writerow([p, name, s] + [round(float(em[k]), 4) for k in
                       ("reward", "throughput_mbps", "goodput_mbps",
                        "completion_rate", "deadline_miss_rate", "mu_depth")])
    print(f"p={p} done", flush=True)
print("saved", OUT)
