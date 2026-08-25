"""OOD evaluation of the frozen new baselines (SUS-RPS / SUS+CQI-Feasible /
SUS+CQI@m=3) on the SAME protocol as the paper's PPO OOD grid.

Worlds and T* are copied verbatim from the existing OOD scripts (paper grid:
ood_grid_extend100.py; K-scale: ood_userscale_ext100.py, T* from the 40k
pilot) so the rows are directly comparable with ood_all_n100.csv. Seeds
30000-30099 -- the SAME episodes the PPO rows used. NOTE these seeds are not
blind (the reviewer's point stands); this is a comparable-protocol evaluation,
not a blind one, and is reported as such.

The three baselines carry ZERO world-specific tuning: RPS has no tunables
(W=8, p_u=1), the feasibility filter has none, and m=3 was fixed on the
ID dev band. Each world reuses the SUS gate's per-world T* selected long ago
for SUS+CQI -- no new threshold sweep for the new baselines.

argv: world_csv (comma list or 'all')  tag
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
from baselines import SUSRPS, SUSCQIFeasible, SUSCQIDepth3
from train_phase2 import env_episode_metrics

BASE = "Run4/QueuePostRZF_S40HL_CQI4"
SEEDS = list(range(30000, 30100))
WORLDS = {  # name -> (overrides, T*)   [verbatim from the existing OOD scripts]
    "P055":   (dict(p_arrival_min=0.55, p_arrival_max=0.55), 0.75),
    "P010":   (dict(p_arrival_min=0.10, p_arrival_max=0.10), 0.80),
    "V60max": (dict(ue_speed_max=60.0), 0.75),
    "CSI02":  (dict(p_csi=0.2), 0.70),
    "D26":    (dict(deadline_min=2, deadline_max=6), 0.80),
    "STORM2": (dict(p_arrival_min=0.55, p_arrival_max=0.55,
                    ue_speed_max=60.0, p_csi=0.2), 0.75),
    "K8":     (dict(num_ue=8, n_active_min=8, n_active_max=8), 0.80),
    "K48":    (dict(num_ue=48, n_active_min=48, n_active_max=48), 0.75),
    "K60":    (dict(num_ue=60, n_active_min=60, n_active_max=60), 0.80),
}
names = list(WORLDS) if sys.argv[1] == "all" else sys.argv[1].split(",")
TAG = sys.argv[2]
OUT = f"Run4/_analysis/ood_newbaselines_{TAG}.csv"

def make_cfg(ov, thr):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw.update(ov)
    raw["sus_ortho_threshold"] = thr
    cfg = Config(**raw)
    assert cfg.cqi_mode == "nr4bit"
    return cfg

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "mu_depth", "jain"]
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["world", "sched", "seed"] + KEYS)
for world in names:
    ov, thr = WORLDS[world]
    cfg = make_cfg(ov, thr)
    env = SchedulerEnv(cfg)
    scheds = [(c().name, c()) for c in (SUSRPS, SUSCQIFeasible, SUSCQIDepth3)]
    t0 = time.time()
    print(f"=== [{TAG}] {world} T*={thr:.2f} ===", flush=True)
    for i, seed in enumerate(SEEDS):
        for nm, sch in scheds:
            env.reset(episode_idx=seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            w.writerow([world, nm, seed]
                       + [round(float(m[k]), 4) for k in KEYS])
        f.flush()
        if (i + 1) % 10 == 0:
            print(f"[{TAG}] {world} {i+1}/100 [{(time.time()-t0)/60:.1f}m]",
                  flush=True)
    print(f"[{TAG}] {world} done [{(time.time()-t0)/60:.1f}m]", flush=True)
print(f"[{TAG}] all done -> {OUT}", flush=True)
