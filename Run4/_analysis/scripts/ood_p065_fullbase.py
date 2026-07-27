"""P065 supplement: ALL 12 baselines at fixed p=0.65 (8 seeds 30000-30007).

Why: in the OOD probe both SUS+CQI and SU+CQI went negative at p=0.65,
so the standing rule fires -- verify no other baseline family (PF/DPF/
MW/Rnd) quietly wins the saturation regime before we claim PPO's +285 is
the only scheduler above water. Same world/seeds as the probe's P065.
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
from baselines import all_baselines

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/ood_p065_fullbase.csv"
SEEDS = list(range(30000, 30008))
raw = json.load(open("Run4/QueuePostRZF_S40HighLoad/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw["la_beta_by_depth"])
raw["ue_speed_mix"] = ()
raw["p_arrival_min"] = 0.65
raw["p_arrival_max"] = 0.65
cfg = Config(**raw)
cfg.sus_ortho_threshold = 0.70          # probe-swept optimum at P065
env = SchedulerEnv(cfg)
from train_phase2 import env_episode_metrics

w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["sched", "seed", "reward", "goodput_mbps", "deadline_miss_rate", "mu_depth"])
means = {}
for sch in all_baselines(cfg):
    rs = []
    for seed in SEEDS:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        rs.append(m["reward"])
        w.writerow([sch.name, seed, round(m["reward"], 1),
                    round(m.get("goodput_mbps", float("nan")), 2),
                    round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
    means[sch.name] = np.mean(rs)
    print(f"{sch.name:12s} {means[sch.name]:8.0f}", flush=True)
print("\nPPO-HL (probe): +285  |  최강 baseline:",
      max(means, key=means.get), round(max(means.values())), flush=True)
print("saved ->", OUT)
