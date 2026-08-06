"""Re-evaluate SUS+CQI at the fresh-pilot (40k-seed) T* on the independent
test set 30000-30039 (audit fix #1/#2 follow-through, 2026-08-06).

Usage: python3 ood_reeval_sus_tstar.py "P055:0.75,V60max:0.75,D26:0.80" tag
Runs SUS+CQI@thr for each listed world on the full n=40 eval seeds and
writes rows compatible with the paper-grid CSVs. Baselines only (CPU-safe).
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, csv
from config import Config
from env import SchedulerEnv
from baselines import all_baselines
from train_phase2 import env_episode_metrics

os.chdir("/home/MYH/ML_DRL_Scheduler")
PAIRS = [p.split(":") for p in sys.argv[1].split(",")]
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"
OUT = f"Run4/_analysis/OOD/results/ood_reeval_sus_{TAG}.csv"
SEEDS = list(range(30000, 30040))
BASE = "Run4/QueuePostRZF_S40HL_CQI4"

WORLDS = {
    "P055":   dict(p_arrival_min=0.55, p_arrival_max=0.55),
    "P010":   dict(p_arrival_min=0.10, p_arrival_max=0.10),
    "V60max": dict(ue_speed_max=60.0),
    "CSI02":  dict(p_csi=0.2),
    "D26":    dict(deadline_min=2, deadline_max=6),
    "STORM2": dict(p_arrival_min=0.55, p_arrival_max=0.55,
                   ue_speed_max=60.0, p_csi=0.2),
}

def make_cfg(overrides, thr):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw.update(overrides)
    raw["sus_ortho_threshold"] = thr
    return Config(**raw)

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["config", "sched", "seed", "n_active", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])
for world, thr_s in PAIRS:
    thr = float(thr_s)
    t0 = time.time()
    print(f"=== [{world}] SUS+CQI@{thr:.2f} on n=40 eval seeds ===", flush=True)
    cfg = make_cfg(WORLDS[world], thr)
    env = SchedulerEnv(cfg)
    sch = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
    rs = []
    for i, seed in enumerate(SEEDS):
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        rs.append(m["reward"])
        w.writerow([world, f"SUS+CQI@{thr:.2f}", seed, int(env.traffic.n_active),
                    round(m["reward"], 1), round(m["throughput_mbps"], 2),
                    round(m.get("goodput_mbps", float("nan")), 2),
                    round(m["completion_rate"], 4),
                    round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        if (i + 1) % 10 == 0:
            f.flush()
    print(f"  mean {np.mean(rs):.1f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
print("saved ->", OUT)
f.close()
