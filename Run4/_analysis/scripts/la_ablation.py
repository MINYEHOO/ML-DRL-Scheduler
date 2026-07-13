"""m-aware LA ablation (GPT-audit C-cluster): does 'near-SU optimal' survive
fair link adaptation?

For each seed (paired: same channel/episode), run every scheduler under
mu_aware_la = False (historical) and True (fair sizing), same world.
Worlds: 'type2' = Run3 hetero point (56-bit codebook, p_csi 0.6) — the world
behind the RUNS.md §3.1 claim; 'genie' = perfect CSI (isolates pure physics:
with quantization AND the LA artifact both gone, is MU good?).

Schedulers: SUS+CQI (MU champion), SU+CQI (SU champion), SUS+PF, CQI-greedy
(blind depth-4 MU — maximally affected), PPO-L2b frozen (the policy whose
'correct near-SU choice' is under question).
Usage: python3 la_ablation.py type2|genie
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
from baselines import SUSCQI, SUCQI, SUSPF, CQIGreedy
from train_phase2 import env_episode_metrics, PPOScheduler

WORLD = sys.argv[1]
assert WORLD in ("type2", "genie")
OUT = f"/home/MYH/ML_DRL_Scheduler/Run4/_analysis/la_ablation_{WORLD}.csv"
SEEDS = list(range(10000, 10010))

kw = dict(num_ue=32, n_active_min=16, n_active_max=32,
          ue_speed_min=5.0, ue_speed_max=30.0,
          ppo_critic_v2=True, ppo_target_kl=0.02, ppo_entropy_coef=0.02)
if WORLD == "genie":
    kw.update(pmi_mode="genie", p_csi=1.0)
else:
    kw.update(p_csi=0.6)
cfg = phase2_hetero_config(**kw)
env = SchedulerEnv(cfg)

ac = ActorCritic(cfg)
ck = torch.load("/home/MYH/ML_DRL_Scheduler/Run3/MixedSpeed_L2b/ckpt/best.pt",
                map_location="cpu")
own = ac.state_dict()
ac.load_state_dict({k: v for k, v in ck["model"].items()
                    if k in own and own[k].shape == v.shape}, strict=False)
ac.eval()

SCHEDS = [("PPO-L2b", PPOScheduler(ac, deterministic=True)),
          ("SUS+CQI", SUSCQI()), ("SU+CQI", SUCQI()),
          ("SUS+PF", SUSPF()), ("CQI-greedy", CQIGreedy())]
if WORLD == "genie":
    SCHEDS = SCHEDS[:3]

FIELDS = ["world", "la", "seed", "n_active", "sched", "reward",
          "throughput_mbps", "mean_sinr_db", "completion_rate",
          "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain"]
f = open(OUT, "w", newline=""); w = csv.writer(f); w.writerow(FIELDS); f.flush()

for seed in SEEDS:                       # seed-major: channel cache reuse
    for la in (False, True):
        cfg.mu_aware_la = la
        for name, sch in SCHEDS:
            env.reset(episode_idx=seed)
            for _ in range(cfg.episode_len):
                env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            w.writerow([WORLD, int(la), seed, env.traffic.n_active, name,
                        round(m["reward"], 1), round(m["throughput_mbps"], 2),
                        round(m["mean_sinr_db"], 2),
                        round(m["completion_rate"], 4),
                        round(m["deadline_miss_rate"], 4),
                        round(m["retx_drop_rate"], 4),
                        round(m["mu_depth"], 3), round(m["jain"], 4)])
            f.flush()
        print(f"{WORLD} seed {seed} la={la} done", flush=True)
f.close()

import statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(OUT)))
by = defaultdict(lambda: defaultdict(dict))
for r in rows:
    by[int(r["la"])][r["seed"]][r["sched"]] = float(r["reward"])
print(f"\n=== {WORLD}: mean reward, LA off -> on ===")
names = [s for s, _ in SCHEDS]
for n in names:
    off = st.mean(by[0][sd][n] for sd in by[0])
    on = st.mean(by[1][sd][n] for sd in by[1])
    print(f"{n:12s} {off:8.0f} -> {on:8.0f}  ({(on-off)/abs(off)*100:+.1f}%)")
for la in (0, 1):
    d = by[la]
    su = st.mean(d[sd]["SU+CQI"] for sd in d)
    sus = st.mean(d[sd]["SUS+CQI"] for sd in d)
    ppo = st.mean(d[sd]["PPO-L2b"] for sd in d)
    wins_sus = sum(d[sd]["SUS+CQI"] > d[sd]["SU+CQI"] for sd in d)
    wins_ppo = sum(d[sd]["PPO-L2b"] > max(d[sd]["SUS+CQI"], d[sd]["SU+CQI"]) for sd in d)
    print(f"la={la}: SUS+CQI vs SU+CQI: {(sus-su)/abs(su)*100:+.1f}% ({wins_sus}/10) | "
          f"PPO vs best-of-two: {(ppo-max(su,sus))/abs(max(su,sus))*100:+.1f}% ({wins_ppo}/10)")
print("saved ->", OUT)
