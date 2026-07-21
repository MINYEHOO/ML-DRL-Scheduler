"""Resource-JFI probe (user question 2026-07-21): Jain's index over per-UE
SCHEDULED-SLOT counts (access share) vs the standard delivered-bits JFI,
Ent02 world, 5 seeds, 9 figure schedulers. Prediction being tested: with
SUS access abundance (32 positions >= n_active), resource-JFI compresses
to ~1.0 for the MU family and differentiates only the SU family."""
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
from train_phase2 import PPOScheduler
from metrics import jains_index

os.chdir("/home/MYH/ML_DRL_Scheduler")
raw = json.load(open("Run4/QueuePostRZF_Ent02/config.json"))
for k in ("git_hash", "git_dirty_py"): raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
cfg.sus_ortho_threshold = 0.7
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ac.load_state_dict(torch.load("Run4/QueuePostRZF_Ent02/ckpt/best.pt", map_location="cpu")["model"], strict=True)
ac.eval()
SHORT = {"SUS+Deadline-PF": "SUS+DPF", "SUS+Random": "SUS+Rnd",
         "SU+Deadline-PF": "SU+DPF", "SU+Random": "SU+Rnd"}
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
scheds = [("PPO", PPOScheduler(ac, deterministic=True))] + \
         [(SHORT.get(s.name, s.name), s) for s in all_baselines(cfg)
          if SHORT.get(s.name, s.name) in ORDER]
res = {n: ([], []) for n in ORDER}
for seed in range(10000, 10005):
    for name, sch in scheds:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        na = env.traffic.n_active
        slots = np.asarray(env.ep["sched_slots_per_ue"], float)[:na]
        res[name][0].append(jains_index(slots))
        res[name][1].append(jains_index(env.cum_acked_bits[:na]))
    print(f"seed {seed} done", flush=True)
print(f"\n{'scheduler':10s} {'자원-JFI':>9s} {'배달-JFI':>9s}")
for n in ORDER:
    print(f"{n:10s} {np.mean(res[n][0]):9.3f} {np.mean(res[n][1]):9.3f}")
