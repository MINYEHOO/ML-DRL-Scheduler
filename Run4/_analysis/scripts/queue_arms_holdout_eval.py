"""Held-out evaluation of the three QueuePostRZF entropy arms' best.pt.

Run-eval numbers (3 seeds) said Ent02 5743 vs SUS+CQI 5108 (+12.4%).
Before quoting that as the headline, evaluate all three arm bests on the
8 held-out seeds 10000-10007 -- the exact seeds and world of
new_la_baselines_betam(_raw).csv (verified identical env config; only
ppo_entropy_coef differs, which is training-only). Reuses that grid for
baselines; evaluates only the PPO policies here.
"""
import os, sys, csv, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/queue_arms_holdout.csv"
SEEDS = list(range(10000, 10008))
ARMS = [("PPO-Ent02", "Run4/QueuePostRZF_Ent02"),
        ("PPO-0.01",  "Run4/QueuePostRZF"),
        ("PPO-Ent03", "Run4/QueuePostRZF_Ent03")]

raw = json.load(open("Run4/QueuePostRZF_Ent02/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.resolved_la_mode() == "post_rzf" and cfg.la_beta_by_depth
env = SchedulerEnv(cfg)

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain",
        "first_ack_rate", "attempts_per_acked"]
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["baseline", "seed"] + KEYS)
for name, run in ARMS:
    ac = ActorCritic(cfg)
    ck = torch.load(f"{run}/ckpt/best.pt", map_location="cpu")
    ac.load_state_dict(ck["model"], strict=True)
    ac.eval()
    sch = PPOScheduler(ac, deterministic=True)
    print(f"{name}: best.pt update {ck['update']} run-eval {ck.get('eval_reward', float('nan')):.1f}", flush=True)
    ms = []
    for s in SEEDS:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        em = env_episode_metrics(env, cfg)
        ms.append(em)
        w.writerow([name, s] + [round(float(em[k]), 4) for k in KEYS])
        print(f"  seed {s}: rew {em['reward']:.0f}", flush=True)
    print(f"{name}: held-out mean {np.mean([m['reward'] for m in ms]):.1f}  "
          f"depth {np.mean([m['mu_depth'] for m in ms]):.2f}  "
          f"miss {np.mean([m['deadline_miss_rate'] for m in ms]):.3f}", flush=True)

# paired comparison vs the existing betam baseline grid
base = {}
for r in csv.DictReader(open("Run4/_analysis/new_la_baselines_betam_raw.csv")):
    base.setdefault(r["baseline"], {})[int(r["seed"])] = float(r["reward"])
ours = {}
for r in csv.DictReader(open(OUT)):
    ours.setdefault(r["baseline"], {})[int(r["seed"])] = float(r["reward"])
best_base_name = max(base, key=lambda n: np.mean(list(base[n].values())))
bb = base[best_base_name]
print(f"\nbest baseline on held-out: {best_base_name} mean {np.mean(list(bb.values())):.1f}")
for name, _ in ARMS:
    d = [ours[name][s] - bb[s] for s in SEEDS]
    wins = sum(x > 0 for x in d)
    print(f"{name}: mean {np.mean(list(ours[name].values())):7.1f}  "
          f"vs {best_base_name}: {np.mean(d):+.0f} ({np.mean(d)/np.mean(list(bb.values()))*100:+.1f}%), "
          f"win {wins}/8, paired-std {np.std(d, ddof=1):.0f}", flush=True)
print("saved ->", OUT)
