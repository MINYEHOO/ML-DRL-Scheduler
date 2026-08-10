"""FINAL paper evaluation: reserved seeds 20000-20049 (n=50), CQI4 main.

This consumes the untouched final-evaluation band reserved since 2026-07.
Protocol identical to final40 (2026-08-07):
  * world = the CQI4 training world (nr4bit, post-RZF + beta_m(CQI4),
    speed U(5,40), p_arrival U(0.15,0.50), p_csi 0.6, K=32 U(16,32));
  * policy = best.pt@409, frozen;
  * SUS threshold T* = 0.75, selected on pilot seeds 40000-40007
    (queue_s40hl_cqi4_final40.out sweep; pilot-separated -- NO tuning
    decision touches any 20000-band seed);
  * schedulers: PPO + the 8 grid baselines (paper main figure uses
    PPO + {SUS,SU}x{CQI,PF,DPF}; Random variants kept for the appendix).
argv[1]=seed_start argv[2]=seed_end(exclusive) argv[3]=tag -- the 50
seeds are split 10-per-GPU across five workers.
"""
import os, sys, csv, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

os.chdir("/home/MYH/ML_DRL_Scheduler")
RUN = "Run4/QueuePostRZF_S40HL_CQI4"
S0, S1, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SEEDS = list(range(S0, S1))
OUT = f"Run4/_analysis/queue_s40hl_cqi4_final50_{TAG}.csv"
TSTAR = 0.75   # from the 40k pilot sweep (final40); same world, reused

raw = json.load(open(f"{RUN}/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
raw["sus_ortho_threshold"] = TSTAR
cfg = Config(**raw)
assert cfg.cqi_mode == "nr4bit" and cfg.la_beta_by_depth

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
ac = ActorCritic(cfg)
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
SHORT = {"SUS+Deadline-PF": "SUS+DPF", "SUS+Random": "SUS+Rnd",
         "SU+Deadline-PF": "SU+DPF", "SU+Random": "SU+Rnd"}
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
scheds = [("PPO", PPOScheduler(ac.to(DEV), deterministic=True))] + \
         [(SHORT.get(s.name, s.name), s) for s in all_baselines(cfg)
          if SHORT.get(s.name, s.name) in ORDER]
assert sorted(n for n, _ in scheds) == sorted(ORDER)
print(f"worker {TAG}: seeds {S0}-{S1-1}, device {DEV}, T*={TSTAR}, "
      f"best.pt@{ck['update']}", flush=True)

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]
env = SchedulerEnv(cfg)
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["seed", "sched"] + KEYS)
t0 = time.time()
for i, seed in enumerate(SEEDS):
    rewards = {}
    for name, sch in scheds:            # seed-outer: channel generated once
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        rewards[name] = m["reward"]
        w.writerow([seed, name] + [round(float(m[k]), 4) for k in KEYS])
    f.flush()
    marg = rewards["PPO"] - max(v for n, v in rewards.items() if n != "PPO")
    print(f"seed {seed}: PPO {rewards['PPO']:7.0f} margin {marg:+7.0f}"
          f"   [{i+1}/{len(SEEDS)}, {(time.time()-t0)/60:.1f} min]", flush=True)
f.close()
print(f"worker {TAG} done -> {OUT}", flush=True)
