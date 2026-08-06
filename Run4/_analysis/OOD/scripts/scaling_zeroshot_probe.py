"""B3: size-axis zero-shot generalization probe (user approved 2026-07-23).

Question: the per-UE-encoder + pointer actor has no K-dependent weights --
does the K=32-trained Ent02 policy work, WITHOUT retraining, in worlds
with a different UE-set size? K=16/24 shrink the observation set below
anything seen in training (training's n_active=16 still had 16 idle UEs
present in obs); K=48 extrapolates above the training range entirely.

Worlds: Ent02 world config with num_ue=K and n_active ~ U{K/2 .. K}
(proportional to the original 16..32 of 32). Everything else fixed,
including beta_m (gNB config stays -- that IS the zero-shot premise).
Schedulers: PPO (zero-shot), SUS+CQI@0.7, SU+CQI. 8 seeds each.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler
import csv

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/scaling_zeroshot.csv"
SEEDS = list(range(10000, 10008))
K_LIST = (16, 24, 32, 48)          # 32 = in-distribution reference
CK = torch.load("Run4/QueuePostRZF_Ent02/ckpt/best.pt", map_location="cpu")

def make_cfg(K):
    raw = json.load(open("Run4/QueuePostRZF_Ent02/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw["num_ue"] = K
    raw["n_active_min"] = K // 2
    raw["n_active_max"] = K
    cfg = Config(**raw)
    cfg.sus_ortho_threshold = 0.7
    return cfg

w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["K", "sched", "seed", "reward", "goodput_mbps", "mu_depth",
            "deadline_miss_rate", "n_active"])
for K in K_LIST:
    cfg = make_cfg(K)
    # step 0: architecture compatibility -- strict weight load at this K
    ac = ActorCritic(cfg)
    ac.load_state_dict(CK["model"], strict=True)   # fails loudly if K leaks into shapes
    ac.eval()
    print(f"K={K}: strict load OK (파라미터가 K-독립임을 확인)", flush=True)
    env = SchedulerEnv(cfg)
    scheds = [("PPO-zs", PPOScheduler(ac, deterministic=True))] + \
             [(s.name, s) for s in all_baselines(cfg)
              if s.name in ("SUS+CQI", "SU+CQI")]
    agg = {}
    for name, sch in scheds:
        ms = []
        for seed in SEEDS:
            env.reset(episode_idx=seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            ms.append(m)
            w.writerow([K, name, seed, round(m["reward"], 1),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["mu_depth"], 3),
                        round(m["deadline_miss_rate"], 4),
                        int(m["n_active"])])
        agg[name] = np.mean([x["reward"] for x in ms])
        print(f"  K={K} {name:8s} reward {agg[name]:8.1f}  "
              f"depth {np.mean([x['mu_depth'] for x in ms]):.2f}  "
              f"miss {np.mean([x['deadline_miss_rate'] for x in ms]):.3f}", flush=True)
    bb = max(v for k, v in agg.items() if k != "PPO-zs")
    print(f"  K={K}: PPO-zs vs 최강 baseline = {(agg['PPO-zs']-bb)/abs(bb)*100:+.1f}%", flush=True)
print("saved ->", OUT)
