"""Same L2b policy, SAME seeds, two CSI worlds -- paired per-seed comparison.

Isolates the effect of perfect CSI by holding EVERYTHING else fixed (policy,
seed => same n_active, same speeds, same channel realization, same traffic).
Only pmi_mode/p_csi differ:
  imperfect = type2_sparse_56bit + p_csi 0.6   (the L2b training world)
  perfect   = genie              + p_csi 1.0   (h_hat == h_true)
Reports per-seed n_active and MU depth side by side, so "does perfect CSI make
the policy pair more" is answered WITHIN each load level, not across a blend.
"""
import os, sys, csv
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
from config import phase2_hetero_config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/l2b_paired_csi.csv"
SEEDS = list(range(10000, 10010))

def make(pmi, pcsi):
    return phase2_hetero_config(pmi_mode=pmi, p_csi=pcsi, num_ue=32,
                                n_active_min=16, n_active_max=32,
                                ue_speed_min=5.0, ue_speed_max=30.0,
                                ppo_critic_v2=True, ppo_target_kl=0.02,
                                ppo_entropy_coef=0.02)

worlds = {"imperfect": make("type2_sparse_56bit", 0.6),
          "perfect":   make("genie", 1.0)}

# one policy, loaded once (dims identical across worlds)
ac = ActorCritic(worlds["perfect"])
ck = torch.load("/home/MYH/ML_DRL_Scheduler/Run3/MixedSpeed_L2b/ckpt/best.pt", map_location="cpu")
own = ac.state_dict()
ac.load_state_dict({k: v for k, v in ck["model"].items()
                    if k in own and own[k].shape == v.shape}, strict=False)
ac.eval()
sch = PPOScheduler(ac, deterministic=True)
print(f"L2b best.pt update {ck['update']} eval {ck.get('eval_reward'):.1f}\n", flush=True)

rows = []
for seed in SEEDS:
    rec = {}
    for wname, cfg in worlds.items():
        env = SchedulerEnv(cfg)
        env.reset(episode_idx=seed)
        na = env.traffic.n_active
        for _ in range(cfg.episode_len_main):
            env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        rec[wname] = (na, m)
    na = rec["imperfect"][0]
    row = [seed, na]
    for w in ("imperfect", "perfect"):
        m = rec[w][1]
        row += [round(m["reward"], 1), round(m["throughput_mbps"], 2),
                round(m["mean_sinr_db"], 2), round(m["completion_rate"], 4),
                round(m["mu_depth"], 3)]
    rows.append(row)
    print(f"seed {seed} n_act={na:2d} | "
          f"depth {rec['imperfect'][1]['mu_depth']:.2f}->{rec['perfect'][1]['mu_depth']:.2f} | "
          f"rew {rec['imperfect'][1]['reward']:.0f}->{rec['perfect'][1]['reward']:.0f} | "
          f"SINR {rec['imperfect'][1]['mean_sinr_db']:.1f}->{rec['perfect'][1]['mean_sinr_db']:.1f}dB",
          flush=True)

FIELDS = ["seed", "n_active",
          "imp_reward", "imp_thr", "imp_sinr", "imp_comp", "imp_depth",
          "perf_reward", "perf_thr", "perf_sinr", "perf_comp", "perf_depth"]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(FIELDS); w.writerows(rows)

# sorted by n_active for the load-controlled view
print("\n=== paired by load (sorted by n_active) ===")
print(f"{'n_act':>5s} {'seed':>5s} | {'depth i->p':>12s} | {'reward i->p':>16s} | {'SINR i->p':>14s}")
for r in sorted(rows, key=lambda x: x[1]):
    seed, na = r[0], r[1]
    print(f"{na:5d} {seed:5d} | {r[6]:5.2f} -> {r[11]:5.2f} | "
          f"{r[2]:7.0f} -> {r[7]:7.0f} | {r[4]:5.1f} -> {r[9]:5.1f} dB")
import statistics as st
print(f"\nmean depth: imperfect {st.mean(r[6] for r in rows):.2f} -> perfect {st.mean(r[11] for r in rows):.2f}")
print(f"mean reward: imperfect {st.mean(r[2] for r in rows):.0f} -> perfect {st.mean(r[7] for r in rows):.0f}")
print("saved ->", OUT)
