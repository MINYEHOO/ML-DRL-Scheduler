"""Blind-holdout (21000-21099) evaluation of the frozen baseline set.

Implementations and all fixed choices were frozen at commit 2bbd76c BEFORE any
21000-band episode was generated (development used only the burned band
20000-20019). Primaries fixed in advance: SUS-RPS (traffic-aware),
PF-Greedy-SDS (literature), SUS+CQI-Feasible (ours); the rest is secondary.

Also records per-slot scheduler decision latency percentiles (p50/p95/p99 ms,
perf_counter around schedule() only), as requested for the runtime table.

argv: seed_start seed_end(excl) tag
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "4"
import csv, json, time
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
os.chdir("/home/MYH/ML_DRL_Scheduler")
from config import Config
from env import SchedulerEnv
from baselines import (all_baselines, SUSRPS, SURPS, SUSCQIFeasible,
                       SUSDeadlinePFFeasible, SUSCQIDepth3, PFGreedySDS,
                       CQIGreedySDS)
from train_phase2 import env_episode_metrics

S0, S1, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
OUT = f"Run4/_analysis/holdout21_baselines_{TAG}.csv"
raw = json.load(open("Run4/QueuePostRZF_S40HL_CQI4/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw["la_beta_by_depth"])
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
raw["sus_ortho_threshold"] = 0.75          # T*, pilot-tuned long ago
cfg = Config(**raw)
assert cfg.seed == 2024 and cfg.cqi_mode == "nr4bit"

sus = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
scheds = [("SUS+CQI", sus)] + [(c().name, c()) for c in (
    SUSRPS, SURPS, SUSCQIFeasible, SUSDeadlinePFFeasible, SUSCQIDepth3,
    PFGreedySDS, CQIGreedySDS)]
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]
env = SchedulerEnv(cfg)
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["seed", "sched"] + KEYS + ["lat_p50_ms", "lat_p95_ms", "lat_p99_ms"])
t0 = time.time()
for i, seed in enumerate(range(S0, S1)):
    for name, sch in scheds:
        env.reset(episode_idx=seed)
        lat = []
        done = False
        while not done:
            t1 = time.perf_counter()
            a = sch.schedule(env)
            lat.append(1e3 * (time.perf_counter() - t1))
            _, _, done, _ = env.step(a)
        m = env_episode_metrics(env, cfg)
        w.writerow([seed, name] + [round(float(m[k]), 4) for k in KEYS]
                   + [round(float(np.percentile(lat, q)), 3)
                      for q in (50, 95, 99)])
    f.flush()
    print(f"[{TAG}] seed {seed} done [{i+1}/{S1-S0}, "
          f"{(time.time()-t0)/60:.1f}m]", flush=True)
print(f"[{TAG}] done -> {OUT}", flush=True)
