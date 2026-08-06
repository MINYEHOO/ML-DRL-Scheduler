"""Load-response curve: frozen Ent02 policy across a p_arrival sweep.

Step-1 (cheap, CPU, no training) characterization for the "does the
learned advantage grow with congestion?" question. Take the RETIRED,
frozen Ent02 best.pt (trained at p_arrival U(0.15,0.40)) and evaluate it
across worlds that differ ONLY in p_arrival_max (min kept at 0.15, per
user). At each load, baselines re-run on the SAME seeds, so the margin
cancels the structural failure floor. Anchors: max=0.40 = Ent02's own
training load; max=0.50 = HighLoad's training load.

Caveat baked in: SUS threshold FIXED at 0.7 (Ent02-world swept optimum);
a per-load threshold re-sweep is a refinement, not done here. "best
baseline" per load picks max(SUS+CQI@0.7, SU+CQI) -- the SU/MU crossover
is itself informative. This is the FROZEN-policy (robustness) curve; the
trained-per-load (adaptation) points come from S40/HighLoad separately.
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
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/load_response.csv"
SEEDS = list(range(10000, 10005))                       # 5 seeds
LOADS = [0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65]      # p_arrival_max (min=0.15)
CK = torch.load("Run4/QueuePostRZF_Ent02/ckpt/best.pt", map_location="cpu")

def make_cfg(pmax):
    raw = json.load(open("Run4/QueuePostRZF_Ent02/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw["p_arrival_min"] = 0.15
    raw["p_arrival_max"] = pmax
    cfg = Config(**raw)
    cfg.sus_ortho_threshold = 0.7
    return cfg

w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["p_arrival_max", "mean_load", "sched", "seed", "reward",
            "goodput_mbps", "deadline_miss_rate", "mu_depth"])
print(f"{'p_max':>6s} {'평균부하':>7s} | {'PPO':>6s} {'최강base':>8s} {'(누구)':>8s} "
      f"{'마진%':>6s} | {'PPOdepth':>8s} {'PPOmiss':>7s}", flush=True)
summary = []
for pmax in LOADS:
    cfg = make_cfg(pmax)
    ac = ActorCritic(cfg)
    ac.load_state_dict(CK["model"], strict=True)
    ac.eval()
    env = SchedulerEnv(cfg)
    scheds = [("PPO", PPOScheduler(ac, deterministic=True))] + \
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
            w.writerow([pmax, round((0.15 + pmax) / 2, 3), name, seed,
                        round(m["reward"], 1),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["deadline_miss_rate"], 4),
                        round(m["mu_depth"], 3)])
        agg[name] = {k: np.mean([x[k] for x in ms])
                     for k in ("reward", "mu_depth", "deadline_miss_rate")}
    bases = {n: v["reward"] for n, v in agg.items() if n != "PPO"}
    bname = max(bases, key=bases.get)
    bb, ppo = bases[bname], agg["PPO"]["reward"]
    marg = (ppo - bb) / bb * 100
    summary.append((pmax, (0.15 + pmax) / 2, ppo, bb, bname, marg,
                    agg["PPO"]["mu_depth"], agg["PPO"]["deadline_miss_rate"]))
    print(f"{pmax:6.2f} {(0.15+pmax)/2:7.3f} | {ppo:6.0f} {bb:8.0f} {bname:>8s} "
          f"{marg:+6.1f} | {agg['PPO']['mu_depth']:8.2f} {agg['PPO']['deadline_miss_rate']:7.3f}",
          flush=True)

print("\n=== 부하-반응 요약 (마진 = PPO vs 그 부하의 최강 baseline) ===")
for pmax, ml, ppo, bb, bn, marg, d, miss in summary:
    print(f"  p_max {pmax:.2f} (평균 {ml:.3f}): 마진 {marg:+.1f}%  (최강={bn})  depth {d:.2f}  miss {miss:.3f}")
peak = max(summary, key=lambda x: x[5])
print(f"\n마진 정점: p_max {peak[0]:.2f} ({peak[5]:+.1f}%). "
      f"Ent02 훈련부하=0.40, HighLoad 훈련부하=0.50 위치 참고.")
print("saved ->", OUT)
