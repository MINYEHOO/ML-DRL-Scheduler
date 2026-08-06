"""OOD probe: does the mixed-arrival policy extrapolate OUTSIDE its
training distribution p ~ U(0.15, 0.40)?

QueueMixedArrival best.pt evaluated at FIXED p_arrival points:
  in-range   : 0.22, 0.33   (interior references)
  OOD        : 0.10, 0.45, 0.50  (below / above / far above the box)
vs SUS+CQI / SUS+MW / SU+CQI, paired (same seeds -> same channel/traffic
stream per seed; one env instance so the per-seed channel cache is reused
across p values -- p only enters the Bernoulli arrival draw).
n_active stays mixed 16-32 per episode exactly as in training.
"""
import os, sys, csv
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "10"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(10)
from config import phase4_queue_config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import SUSCQI, SUSMaxWeight, SUCQI
from train_phase2 import env_episode_metrics, PPOScheduler

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/ood_p_arrival.csv"
SEEDS = list(range(10000, 10008))
PS = [0.10, 0.22, 0.33, 0.45, 0.50]

cfg = phase4_queue_config()          # queue 8, thr 0.75, n_active 16-32 mixed
assert cfg.p_arrival_min == 0.0      # per-episode draw disabled -> fixed p
env = SchedulerEnv(cfg)

ac = ActorCritic(cfg)
ck = torch.load("/home/MYH/ML_DRL_Scheduler/Run4/QueueMixedArrival/ckpt/best.pt",
                map_location="cpu")
ac.load_state_dict(ck["model"])
ac.eval()
print(f"MixedArrival best.pt: update {ck['update']}, eval {ck.get('eval_reward'):.1f}",
      flush=True)

scheds = [("PPO-Mixed", PPOScheduler(ac, deterministic=True)),
          ("SUS+CQI", SUSCQI()), ("SUS+MW", SUSMaxWeight()), ("SU+CQI", SUCQI())]
FIELDS = ["p_arrival", "seed", "n_active", "sched", "reward", "throughput_mbps",
          "completion_rate", "deadline_miss_rate", "buffer_overflow_rate",
          "mu_depth", "mean_delay", "mean_qlen", "jain"]
f = open(OUT, "w", newline=""); w = csv.writer(f); w.writerow(FIELDS); f.flush()

for seed in SEEDS:                    # seed-major: channel cache reused across p
    for P in PS:
        cfg.p_arrival = P             # only the arrival Bernoulli changes
        for name, sch in scheds:
            env.reset(episode_idx=seed)
            for _ in range(cfg.episode_len):
                env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            w.writerow([P, seed, env.traffic.n_active, name,
                        round(m["reward"], 1), round(m["throughput_mbps"], 2),
                        round(m["completion_rate"], 4),
                        round(m["deadline_miss_rate"], 4),
                        round(m["buffer_overflow_rate"], 5),
                        round(m["mu_depth"], 3), round(m["mean_delay"], 3),
                        round(m["mean_qlen"], 3), round(m["jain"], 4)])
            f.flush()
        print(f"seed {seed} p={P} done", flush=True)
f.close()

# summary: PPO margin vs best baseline per p
import statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(OUT)))
by = defaultdict(lambda: defaultdict(dict))
for r in rows:
    by[float(r["p_arrival"])][r["seed"]][r["sched"]] = float(r["reward"])
print("\n=== PPO-Mixed margin vs best fixed baseline, per p_arrival ===")
print(f"{'p':>5s} {'구분':>8s} | {'PPO':>7s} {'bestBase':>9s} {'margin':>8s} {'승':>5s}")
for P in PS:
    d = by[P]; margins = []; wins = 0
    for sd in d:
        ppo = d[sd]["PPO-Mixed"]
        bb = max(v for k, v in d[sd].items() if k != "PPO-Mixed")
        margins.append((ppo - bb) / abs(bb) * 100 if bb else 0)
        wins += ppo > bb
    tag = "IN" if 0.15 <= P <= 0.40 else "OOD"
    pm = st.mean(d[sd]["PPO-Mixed"] for sd in d)
    bm = st.mean(max(v for k, v in d[sd].items() if k != "PPO-Mixed") for sd in d)
    print(f"{P:5.2f} {tag:>8s} | {pm:7.0f} {bm:9.0f} {st.mean(margins):+7.1f}% {wins:>3d}/8")
print("saved ->", OUT)
