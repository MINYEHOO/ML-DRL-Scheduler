"""SUS+MW dedicated threshold sweep on the 40k pilot seeds (follow-up to
the champion-verification FLAGs, 2026-08-06).

The 12-baseline pilot found SUS+MW within 10% of the SUS+CQI champion in
P055/V60max/CSI02/D26 -- but MW was only evaluated at the CQI-chosen T*.
To claim "champion verified", MW deserves its own threshold sweep: if its
own-best still trails SUS+CQI's, the champion identity is rigorous.
Worlds x thresholds {0.60..0.80} x pilot seeds 40000-40007, CPU-safe.
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
OUT = "Run4/_analysis/OOD/results/ood_pilot40k_musweep.csv"
PILOT = list(range(40000, 40008))
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]
BASE = "Run4/QueuePostRZF_S40HL_CQI4"

# world -> (overrides, SUS+CQI champion mean at its own T*, from the pilot)
WORLDS = {
    "P055":   (dict(p_arrival_min=0.55, p_arrival_max=0.55), 3827.0),
    "V60max": (dict(ue_speed_max=60.0), 4354.0),
    "CSI02":  (dict(p_csi=0.2), 3731.0),
    "D26":    (dict(deadline_min=2, deadline_max=6), 4206.0),
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
w.writerow(["world", "sched", "seed", "reward"])
for world, (ov, champ) in WORLDS.items():
    t0 = time.time()
    by = {t: [] for t in SUS_THRS}
    cfg = make_cfg(ov, SUS_THRS[0])
    env = SchedulerEnv(cfg)
    for seed in PILOT:
        for thr in SUS_THRS:
            cfg.sus_ortho_threshold = thr
            sch = [s for s in all_baselines(cfg) if s.name == "SUS+MW"][0]
            env.reset(episode_idx=seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            by[thr].append(m["reward"])
            w.writerow([world, f"SUS+MW@{thr:.2f}", seed, round(m["reward"], 1)])
        f.flush()
    best = max(SUS_THRS, key=lambda t: np.mean(by[t]))
    mw = np.mean(by[best])
    print(f"[{world}] SUS+MW sweep: " + "  ".join(f"{t:.2f}:{np.mean(by[t]):.0f}" for t in SUS_THRS)
          + f"\n  own-best {mw:.0f}@{best:.2f} vs SUS+CQI champion {champ:.0f} -> "
          + ("⚠️ MW WINS -- n=40 재평가 필요" if mw > champ else f"CQI 챔피언 유지 (격차 {champ-mw:+.0f})")
          + f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
print("saved ->", OUT)
f.close()
