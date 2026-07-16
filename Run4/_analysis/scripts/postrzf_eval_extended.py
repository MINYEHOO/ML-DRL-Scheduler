"""Pin-safe extended evaluator for the live post-RZF arms.

The in-training eval CSV has only 6 columns (pin forbids widening it
mid-run). This script loads each arm's CURRENT latest.pt and re-runs the
same deterministic eval episodes (seeds 10000-10002), logging the FULL
metric set (deadline_miss_rate, goodput_mbps, ...) to
Run4/_analysis/postrzf_eval_extended.csv. Baseline reference rows (SUS+CQI)
are appended once. Run on CPU; safe to invoke any time.
"""
import csv, json, os
import numpy as np, torch

from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler
from baselines import SUSCQI

RUNS = ["QueuePostRZF", "QueuePostRZF_Ent02", "QueuePostRZF_Ent03"]
SEEDS = [10000, 10001, 10002]
OUT = "Run4/_analysis/postrzf_eval_extended.csv"
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mean_sinr_db", "mu_depth",
        "jain", "first_ack_rate", "attempts_per_acked"]

def make_cfg(run):
    raw = json.load(open(f"Run4/{run}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    return Config(**raw)

new_file = not os.path.exists(OUT)
w = csv.writer(open(OUT, "a", newline=""))
if new_file:
    w.writerow(["run", "ckpt_update", "seed"] + KEYS)

cfg0 = make_cfg(RUNS[0])
env = SchedulerEnv(cfg0)
if new_file:                                   # baseline 참조행 1회
    sch = SUSCQI()
    for s in SEEDS:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        em = env_episode_metrics(env, cfg0)
        w.writerow(["SUS+CQI", "-", s] + [round(float(em[k]), 4) for k in KEYS])
    print("baseline rows written")

for run in RUNS:
    cfg = make_cfg(run)
    ck = torch.load(f"Run4/{run}/ckpt/latest.pt", map_location="cpu")
    ac = ActorCritic(cfg); ac.load_state_dict(ck["model"]); ac.eval()
    sch = PPOScheduler(ac, deterministic=True)
    for s in SEEDS:
        env2 = SchedulerEnv(cfg)
        env2.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env2.step(sch.schedule(env2))
        em = env_episode_metrics(env2, cfg)
        w.writerow([run, ck["update"], s] + [round(float(em[k]), 4) for k in KEYS])
    print(f"{run} @upd {ck['update']} done")
