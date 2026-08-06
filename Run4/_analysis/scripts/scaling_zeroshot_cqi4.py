"""B3 size-axis zero-shot, CQI4-main edition (2026-08-06).

The original B3 (scaling_zeroshot_probe.py) used the Ent02 continuous
policy; with CQI4 as the paper's main policy this re-runs the K axis for
QueuePostRZF_S40HL_CQI4 best.pt@409 in the nr4bit world. Per-K worlds:
num_ue=K, n_active ~ U{K/2..K}; all other knobs = the CQI4 training
world; beta_m stays at the K=32 calibration (zero-shot premise).
Protocol: 8 seeds 30000-30007 (mini-table claim: "does not lose"),
per-K SUS+CQI threshold sweep + SU+CQI anchor.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch, csv
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/scaling_zeroshot_cqi4.csv"
SEEDS = list(range(30000, 30008))
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
CK = torch.load(f"{BASE}/ckpt/best.pt", map_location="cpu")
KS = (16, 24, 32, 48)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"policy device: {DEV}; CQI4 best.pt@{CK['update']}", flush=True)

def make_cfg(K):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw["num_ue"] = K
    raw["n_active_min"] = K // 2
    raw["n_active_max"] = K
    cfg = Config(**raw)
    assert cfg.cqi_mode == "nr4bit"
    return cfg

def episode(env, cfg, sched, seed):
    env.reset(episode_idx=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(sched.schedule(env))
    return env_episode_metrics(env, cfg)

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["K", "sched", "seed", "reward", "goodput_mbps",
            "deadline_miss_rate", "mu_depth"])

summary = []
for K in KS:
    t0 = time.time()
    print(f"\n=== [K={K}] n_active U({K // 2},{K}) ===", flush=True)
    cfg = make_cfg(K)
    env = SchedulerEnv(cfg)
    ac = ActorCritic(cfg)
    ac.load_state_dict(CK["model"], strict=True)   # K-independent params
    ac.eval()
    ppo = PPOScheduler(ac.to(DEV), deterministic=True)

    res = {}
    for seed in SEEDS:                    # seed-outer: channel gen once/seed
        for thr in SUS_THRS:
            cfg.sus_ortho_threshold = thr
            sch = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
            m = episode(env, cfg, sch, seed)
            res.setdefault(f"SUS+CQI@{thr:.2f}", {})[seed] = m["reward"]
            w.writerow([K, f"SUS+CQI@{thr:.2f}", seed, round(m["reward"], 1),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        for label, sch in (("SU+CQI", [s for s in all_baselines(cfg) if s.name == "SU+CQI"][0]),
                           ("PPO", ppo)):
            m = episode(env, cfg, sch, seed)
            res.setdefault(label, {})[seed] = m["reward"]
            w.writerow([K, label, seed, round(m["reward"], 1),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        f.flush()
    bname = max((k for k in res if k != "PPO"),
                key=lambda k: np.mean(list(res[k].values())))
    bb = res[bname]
    d = [res["PPO"][s] - bb[s] for s in SEEDS]
    ppo_mean = np.mean(list(res["PPO"].values()))
    depth = np.nan
    print(f"  PPO {ppo_mean:7.0f} vs {bname} {np.mean(list(bb.values())):7.0f}"
          f"  마진 {(ppo_mean - np.mean(list(bb.values()))) / abs(np.mean(list(bb.values()))) * 100:+.1f}%"
          f"  승 {sum(x > 0 for x in d)}/8", flush=True)
    summary.append((K, ppo_mean, np.mean(list(bb.values())), bname,
                    sum(x > 0 for x in d)))
    print(f"  [K={K}] {(time.time() - t0) / 60:.1f} min", flush=True)

print("\n=== K-scaling 요약 (CQI4 best@409 동결, nr4bit, seeds 30000-30007) ===")
for K, r, b, bn, wins in summary:
    print(f"  K={K:2d}: PPO {r:6.0f} vs {bn:13s} {b:6.0f}  "
          f"마진 {(r - b) / abs(b) * 100:+6.1f}%  승 {wins}/8")
print("saved ->", OUT)
f.close()
