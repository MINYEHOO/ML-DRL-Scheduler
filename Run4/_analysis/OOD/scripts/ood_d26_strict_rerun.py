"""D26 strict zero-shot rerun (audit fix #3, user-approved 2026-08-06).

The original D26 run changed deadline_max 12->6, which also rescaled the
policy's observation normalization (deadline/deadline_max) -- frozen
weights, but NOT frozen preprocessing. This rerun decouples the two:
  * env cfg   : deadline ~ U[2,6]   (world generation)
  * policy cfg: deadline_max = 12   (training-time preprocessing, frozen)
The split needs no code change: build_encoder_input reads deadline_max
from the ActorCritic's OWN cfg (policy.py:68,80), while the environment
draws deadlines from its env cfg. All other knobs identical; baselines
are unaffected (they never use the normalized deadline).

PPO row only, n=40 eval seeds 30000-30039. Output rows labelled
'PPO-strict' alongside the original grid CSVs.
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
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/OOD/results/ood_d26_strict.csv"
SEEDS = list(range(30000, 30040))
BASE = "Run4/QueuePostRZF_S40HL_CQI4"

def make_cfg(**overrides):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw.update(overrides)
    cfg = Config(**raw)
    assert cfg.cqi_mode == "nr4bit"
    return cfg

cfg_env = make_cfg(deadline_min=2, deadline_max=6)   # world: U[2,6]
cfg_pol = make_cfg()                                 # preprocessing: /12 (training)
assert cfg_pol.deadline_max == 12

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{BASE}/ckpt/best.pt", map_location="cpu")
ac = ActorCritic(cfg_pol)
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
ppo = PPOScheduler(ac.to(DEV), deterministic=True)
print(f"device {DEV}; env deadline U[{cfg_env.deadline_min},{cfg_env.deadline_max}], "
      f"policy norm /{cfg_pol.deadline_max}; best.pt@{ck['update']}", flush=True)

env = SchedulerEnv(cfg_env)
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["config", "sched", "seed", "n_active", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])
rs = []
t0 = time.time()
for i, seed in enumerate(SEEDS):
    env.reset(episode_idx=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(ppo.schedule(env))
    m = env_episode_metrics(env, cfg_env)
    rs.append(m)
    w.writerow(["D26", "PPO-strict", seed, int(env.traffic.n_active),
                round(m["reward"], 1), round(m["throughput_mbps"], 2),
                round(m.get("goodput_mbps", float("nan")), 2),
                round(m["completion_rate"], 4),
                round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
    if (i + 1) % 10 == 0:
        f.flush()
        print(f"  ..{i+1}/40 ({(time.time()-t0)/60:.1f} min)", flush=True)
print(f"PPO-strict D26: mean {np.mean([m['reward'] for m in rs]):.1f}  "
      f"depth {np.mean([m['mu_depth'] for m in rs]):.2f}  "
      f"miss {np.mean([m['deadline_miss_rate'] for m in rs]):.3f}", flush=True)
print("saved ->", OUT)
f.close()
