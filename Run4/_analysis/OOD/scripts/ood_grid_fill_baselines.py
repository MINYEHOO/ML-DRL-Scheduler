"""Fill the paper OOD grid with the remaining display baselines (n=40).

The main grid only evaluated the two CQI baselines (enough for the
champion verdict, not enough for the paper table). The confirmed display
set is {SUS, SU} x {CQI, PF, Random}; this adds the four missing rows
(SUS+PF, SUS+Random, SU+PF, SU+Random) on the SAME test seeds
30000-30039, in the SAME six worlds, at each world's fresh-pilot T*
(SUS family only -- SU variants have no threshold).

argv[1]=comma world subset, argv[2]=tag (CPU-safe, 3-way split).
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
SEL = sys.argv[1].split(",") if len(sys.argv) > 1 else None
TAG = sys.argv[2] if len(sys.argv) > 2 else "all"
OUT = f"Run4/_analysis/OOD/results/ood_grid_fill_{TAG}.csv"
SEEDS = list(range(30000, 30040))
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
WANT = ["SUS+PF", "SUS+Random", "SU+PF", "SU+Random"]

# world -> (overrides, fresh-pilot T* for the SUS family)
WORLDS = {
    "P055":   (dict(p_arrival_min=0.55, p_arrival_max=0.55), 0.75),
    "P010":   (dict(p_arrival_min=0.10, p_arrival_max=0.10), 0.80),
    "V60max": (dict(ue_speed_max=60.0), 0.75),
    "CSI02":  (dict(p_csi=0.2), 0.70),
    "D26":    (dict(deadline_min=2, deadline_max=6), 0.80),
    "STORM2": (dict(p_arrival_min=0.55, p_arrival_max=0.55,
                    ue_speed_max=60.0, p_csi=0.2), 0.75),
}
names = SEL if SEL else list(WORLDS)
print(f"fill worker TAG={TAG}, worlds: {names}", flush=True)

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

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["config", "sched", "seed", "n_active", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])
for world in names:
    ov, thr = WORLDS[world]
    t0 = time.time()
    print(f"\n=== [{world}] SUS thr {thr:.2f}, n=40 ===", flush=True)
    cfg = make_cfg(ov, thr)
    env = SchedulerEnv(cfg)
    scheds = [s for s in all_baselines(cfg) if s.name in WANT]
    assert len(scheds) == 4, [s.name for s in scheds]
    acc = {s.name: [] for s in scheds}
    for i, seed in enumerate(SEEDS):          # seed-outer: channel cache hot
        for sch in scheds:
            env.reset(episode_idx=seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            acc[sch.name].append(m["reward"])
            w.writerow([world, sch.name, seed, int(env.traffic.n_active),
                        round(m["reward"], 1), round(m["throughput_mbps"], 2),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["completion_rate"], 4),
                        round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        if (i + 1) % 10 == 0:
            f.flush()
            print(f"  ..{i+1}/40 ({(time.time()-t0)/60:.1f} min)", flush=True)
    for n in WANT:
        print(f"  {n:11s} {np.mean(acc[n]):8.1f}", flush=True)
print("\nsaved ->", OUT)
f.close()
