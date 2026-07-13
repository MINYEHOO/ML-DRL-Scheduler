"""L2b policy under PERFECT CSI (genie), zero-shot, vs baselines.

Load MixedSpeed_L2b best.pt (trained on IMPERFECT CSI) and evaluate it, with
NO retraining, in the genie world (pmi_mode=genie, p_csi=1.0 -> h_hat==h_true).
Both are hetero mode -> direct load, no surgery. Baselines get the same perfect
CSI. Paired held-out seeds. Standard 9 metrics.

Question: does L2b's imperfect-CSI-trained (SU-cautious) policy still beat the
heuristics once CSI is perfect and MU is safe -- or does its SU-bias leave gains
on the table that the MU-capable baselines now grab?
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
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/genie_zeroshot_l2b.csv"
SEEDS = list(range(10000, 10010))

cfg = phase2_hetero_config(pmi_mode="genie", p_csi=1.0, num_ue=32,
                           n_active_min=16, n_active_max=32,
                           ue_speed_min=5.0, ue_speed_max=30.0,
                           ppo_critic_v2=True, ppo_target_kl=0.02,
                           ppo_entropy_coef=0.02)
env = SchedulerEnv(cfg)

ac = ActorCritic(cfg)
ck = torch.load("Run3/MixedSpeed_L2b/ckpt/best.pt", map_location="cpu")
own = ac.state_dict()
keep = {k: v for k, v in ck["model"].items()
        if k in own and own[k].shape == v.shape}
missing, unexpected = ac.load_state_dict(keep, strict=False)
assert not unexpected, unexpected
ac.eval()
print(f"L2b best.pt update {ck['update']} eval {ck.get('eval_reward'):.1f} "
      f"loaded ({len(keep)} tensors, fresh: {sorted(set(own)-set(keep))})",
      flush=True)

scheds = [("PPO-L2b", PPOScheduler(ac, deterministic=True))] + \
         [(s.name, s) for s in all_baselines(cfg)]

# standard 9 metrics (+ n_active/speed context)
FIELDS = ["seed", "n_active", "mean_speed", "sched", "reward",
          "throughput_mbps", "mean_sinr_db", "completion_rate",
          "deadline_miss_rate", "retx_drop_rate", "total_fail_rate",
          "mu_depth", "jain"]
rows = []
for seed in SEEDS:
    for name, sch in scheds:
        env.reset(episode_idx=seed)
        for _ in range(cfg.episode_len_main):
            env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        tot = m["deadline_miss_rate"] + m["retx_drop_rate"]
        rows.append([seed, env.traffic.n_active, round(float(np.mean(env.ue_speeds_kmh)), 1),
                     name, round(m["reward"], 1), round(m["throughput_mbps"], 2),
                     round(m["mean_sinr_db"], 2), round(m["completion_rate"], 4),
                     round(m["deadline_miss_rate"], 4), round(m["retx_drop_rate"], 4),
                     round(tot, 4), round(m["mu_depth"], 3), round(m["jain"], 4)])
    print(f"seed {seed} (n_active {env.traffic.n_active}) done", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(FIELDS); w.writerows(rows)

# aggregate (paired means over seeds)
import statistics as st
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    for i, k in enumerate(FIELDS[4:], start=4):
        agg[r[3]][k].append(r[i])
order = [s[0] for s in scheds]
print("\n=== L2b zero-shot under PERFECT CSI (genie), 10 seeds, standard 9 metrics ===")
hdr = f"{'scheduler':16s} {'reward':>8s} {'thr':>7s} {'SINR':>6s} {'comp':>6s} {'miss':>6s} {'retx':>6s} {'totfail':>7s} {'depth':>6s} {'JFI':>6s}"
print(hdr)
ppo_rew = st.mean(agg["PPO-L2b"]["reward"])
for name in order:
    a = agg[name]
    mk = lambda k: st.mean(a[k])
    print(f"{name:16s} {mk('reward'):8.0f} {mk('throughput_mbps'):7.2f} {mk('mean_sinr_db'):6.2f} "
          f"{mk('completion_rate'):6.3f} {mk('deadline_miss_rate'):6.3f} {mk('retx_drop_rate'):6.3f} "
          f"{mk('total_fail_rate'):7.3f} {mk('mu_depth'):6.2f} {mk('jain'):6.3f}")
best_base = max((st.mean(agg[n]["reward"]) for n in order if n != "PPO-L2b"))
print(f"\nPPO-L2b reward {ppo_rew:.0f} vs best baseline {best_base:.0f} "
      f"= {ppo_rew-best_base:+.0f} ({(ppo_rew-best_base)/best_base*100:+.1f}%)")
print("saved ->", OUT)
