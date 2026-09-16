"""Evaluate ONE frozen PPO policy across the nine OOD worlds.

Same worlds, same per-world T*, same seeds (30000-30099) as the paper's PPO
OOD grid, so rows drop straight into ood_all_n100.csv comparisons.

Built for the NARROW experiment: NARROW trains on a COLLAPSED world (speed 20,
p_arrival 0.30, K_act 24 -- the wide world's means, so difficulty is held and
only DIVERSITY is removed). If a narrow-trained policy still holds up across
these nine shifts, the robustness is a property of learning; if it breaks while
the wide-trained policy holds, the robustness came from domain randomisation.
Either answer is publishable -- the second one bounds the paper's claim.

The policy's own decode flags (force_full_rank / no_user_scale) are carried
over. The train/eval world mismatch is expected here and is NOT guarded.

argv: run_dir  world_csv|all  tag
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "4"
import csv, json, time
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
os.chdir("/home/MYH/ML_DRL_Scheduler")
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler

BASE = "Run4/QueuePostRZF_S40HL_CQI4"      # defines every OOD world
POL_RUN, SEL, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = list(range(30000, 30100))
WORLDS = {   # verbatim from ood_grid_extend100.py / ood_userscale_ext100.py
    "P055":   (dict(p_arrival_min=0.55, p_arrival_max=0.55), 0.75),
    "P010":   (dict(p_arrival_min=0.10, p_arrival_max=0.10), 0.80),
    "V60max": (dict(ue_speed_max=60.0), 0.75),
    "CSI02":  (dict(p_csi=0.2), 0.70),
    "D26":    (dict(deadline_min=2, deadline_max=6), 0.80),
    "STORM2": (dict(p_arrival_min=0.55, p_arrival_max=0.55,
                    ue_speed_max=60.0, p_csi=0.2), 0.75),
    "K8":     (dict(num_ue=8, n_active_min=8, n_active_max=8), 0.80),
    "K48":    (dict(num_ue=48, n_active_min=48, n_active_max=48), 0.75),
    "K60":    (dict(num_ue=60, n_active_min=60, n_active_max=60), 0.80),
}
names = list(WORLDS) if SEL == "all" else SEL.split(",")
pol_cfg = json.load(open(f"{POL_RUN}/config.json"))
ck = torch.load(f"{POL_RUN}/ckpt/best.pt", map_location="cpu")

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
    # the policy's own action-space flags travel with the weights
    cfg.ppo_force_full_rank = bool(pol_cfg.get("ppo_force_full_rank", False))
    cfg.ppo_no_user_scale = float(pol_cfg.get("ppo_no_user_scale", 1.0))
    return cfg

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "mu_depth", "jain"]
OUT = f"Run4/_analysis/ood_policy_{TAG}.csv"
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["world", "sched", "seed"] + KEYS)
print(f"[{TAG}] {POL_RUN} best.pt@{ck['update']} (trained seed "
      f"{pol_cfg['seed']}, speed {pol_cfg['ue_speed_min']}-"
      f"{pol_cfg['ue_speed_max']}, p_arr {pol_cfg['p_arrival_min']}-"
      f"{pol_cfg['p_arrival_max']}, K_act {pol_cfg['n_active_min']}-"
      f"{pol_cfg['n_active_max']})  worlds {names}  dev {DEV}", flush=True)
for world in names:
    ov, thr = WORLDS[world]
    cfg = make_cfg(ov, thr)
    ac = ActorCritic(cfg)
    ac.load_state_dict(ck["model"], strict=True)
    ac.eval()
    sch = PPOScheduler(ac.to(DEV), deterministic=True)
    env = SchedulerEnv(cfg)
    t0 = time.time()
    for i, seed in enumerate(SEEDS):
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        w.writerow([world, "PPO", seed] + [round(float(m[k]), 4) for k in KEYS])
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"[{TAG}] {world} {i+1}/100 [{(time.time()-t0)/60:.1f}m]",
                  flush=True)
    print(f"[{TAG}] {world} done [{(time.time()-t0)/60:.1f}m]", flush=True)
print(f"[{TAG}] all done -> {OUT}", flush=True)
