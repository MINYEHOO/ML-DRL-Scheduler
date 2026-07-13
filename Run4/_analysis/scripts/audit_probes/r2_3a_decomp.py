"""R2-3a probe: decompose the genie-world SUS+CQI vs SU+CQI reward gap.

Part A: exact arithmetic on the published CSV (la=0 genie rows).
Part B: rerun the identical operating point (la_ablation.py generator config,
episode_len=300 per probe rules) with the reward split into its three terms
  reward = lambda_s * r_short + lambda_c * n_comp - lambda_m * n_miss
and r_short further split into the pure-throughput part sum(useful)/B_norm
and the deadline-urgency part sum(useful * eta_d/(D+1))/B_norm.
Repo is READ-ONLY: instrumentation = wrapping env.txmgr.process_slot only.
"""
import os, sys, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np

# ---------------- Part A: CSV arithmetic ----------------
rows = list(csv.DictReader(open(
    "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/la_ablation_genie.csv")))
g = {}   # (la, seed, sched) -> row
for r in rows:
    g[(int(r["la"]), int(r["seed"]), r["sched"])] = r
SEEDS = sorted({int(r["seed"]) for r in rows})

def col(la, sched, key):
    return np.array([float(g[(la, s, sched)][key]) for s in SEEDS])

print("=== Part A: published CSV, genie la=0, SUS+CQI vs SU+CQI ===")
for key in ("reward", "throughput_mbps", "completion_rate",
            "deadline_miss_rate", "retx_drop_rate", "jain", "mu_depth"):
    a, b = col(0, "SUS+CQI", key), col(0, "SU+CQI", key)
    print(f"{key:20s} SUS {a.mean():9.3f}  SU {b.mean():9.3f}  "
          f"delta {a.mean()-b.mean():+9.3f} ({(a.mean()-b.mean())/abs(b.mean())*100:+.2f}%)")
rw_sus, rw_su = col(0, "SUS+CQI", "reward"), col(0, "SU+CQI", "reward")
th_sus, th_su = col(0, "SUS+CQI", "throughput_mbps"), col(0, "SU+CQI", "throughput_mbps")
na = col(0, "SUS+CQI", "n_active") if "n_active" in rows[0] else \
     np.array([int(g[(0, s, "SUS+CQI")]["n_active"]) for s in SEEDS])
print(f"\nreward  mean gap: {(rw_sus.mean()-rw_su.mean())/rw_su.mean()*100:+.2f}%  "
      f"wins {int((rw_sus > rw_su).sum())}/10")
print(f"thr     mean gap: {(th_sus.mean()-th_su.mean())/th_su.mean()*100:+.2f}%")
d_rw, d_th = rw_sus - rw_su, th_sus - th_su
print(f"per-seed: n_active | d_thr Mbps (%) | d_reward (%)")
for i, s in enumerate(SEEDS):
    print(f"  {s} n={int(na[i]):2d}  {d_th[i]:+7.2f} ({d_th[i]/th_su[i]*100:+6.1f}%)"
          f"   {d_rw[i]:+8.1f} ({d_rw[i]/rw_su[i]*100:+6.1f}%)")
cc = np.corrcoef(d_th, d_rw)[0, 1]
print(f"corr(d_thr, d_reward) over seeds = {cc:.3f}")
hi = na >= 27; lo = ~hi
print(f"high-load (n>=27, {hi.sum()} seeds): d_thr {d_th[hi].mean():+.2f} Mbps "
      f"({d_th[hi].mean()/th_su[hi].mean()*100:+.1f}%), d_reward {d_rw[hi].mean():+.1f} "
      f"({d_rw[hi].mean()/rw_su[hi].mean()*100:+.1f}%)")
print(f"low-load  (n<=25, {lo.sum()} seeds): d_thr {d_th[lo].mean():+.2f} Mbps "
      f"({d_th[lo].mean()/th_su[lo].mean()*100:+.1f}%), d_reward {d_rw[lo].mean():+.1f} "
      f"({d_rw[lo].mean()/rw_su[lo].mean()*100:+.1f}%)")
tot_sus = col(0, "SUS+CQI", "deadline_miss_rate") + col(0, "SUS+CQI", "retx_drop_rate")
tot_su = col(0, "SU+CQI", "deadline_miss_rate") + col(0, "SU+CQI", "retx_drop_rate")
print(f"total-fail (miss+retx): SUS {tot_sus.mean():.4f}  SU {tot_su.mean():.4f}")

# ---------------- Part B: instrumented rerun ----------------
print("\n=== Part B: instrumented rerun (genie, la=0, 300 slots) ===")
from config import phase2_hetero_config
from env import SchedulerEnv
from baselines import SUSCQI, SUCQI

cfg = phase2_hetero_config(num_ue=32, n_active_min=16, n_active_max=32,
                           ue_speed_min=5.0, ue_speed_max=30.0,
                           ppo_critic_v2=True, ppo_target_kl=0.02,
                           ppo_entropy_coef=0.02,
                           pmi_mode="genie", p_csi=1.0,
                           episode_len_main=300)
assert cfg.mu_aware_la is False and cfg.queue_size == 1
env = SchedulerEnv(cfg)

# wrap txmgr.process_slot once to capture useful_per_ue (read-only repo)
_orig = env.txmgr.process_slot
_last = {}
def _wrapped(sinr_map):
    out = _orig(sinr_map)
    _last["useful"] = out.useful_per_ue.copy()
    return out
env.txmgr.process_slot = _wrapped

CSV_NA = {s: int(g[(0, s, "SUS+CQI")]["n_active"]) for s in SEEDS}
res = {}
for seed in SEEDS:
    for name, sch in (("SUS+CQI", SUSCQI()), ("SU+CQI", SUCQI())):
        env.reset(episode_idx=seed)
        assert env.traffic.n_active == CSV_NA[seed], \
            f"n_active mismatch seed {seed}: {env.traffic.n_active} vs {CSV_NA[seed]}"
        acc = dict(r_short=0.0, thr_part=0.0, urg_part=0.0, comp=0, miss=0,
                   retx=0, reward=0.0, acked=0.0, useful=0.0)
        for _ in range(cfg.episode_len):
            _, rew, done, info = env.step(sch.schedule(env))
            u = _last["useful"]
            snap = env._deadline_snap
            w = 1.0 + cfg.eta_d / (snap + 1.0)
            rs_check = float(np.sum(w * u) / cfg.b_norm)
            assert abs(rs_check - info["reward_short"]) < 1e-6
            acc["r_short"] += info["reward_short"]
            acc["thr_part"] += float(u.sum() / cfg.b_norm)
            acc["urg_part"] += float(np.sum(u * cfg.eta_d / (snap + 1.0)) / cfg.b_norm)
            acc["comp"] += info["n_comp"]
            acc["miss"] += info["n_miss_deadline"]
            acc["retx"] += info["n_retx_drop"]
            acc["reward"] += rew
            acc["acked"] += info["acked_bits"]
            acc["useful"] += float(u.sum())
        acc["ovf"] = int(env.ep["n_retx_overflow_drop"])
        acc["arrivals"] = int(env.ep["n_arrivals"])
        # episode identity: reward = r_short + comp - 2*(miss+retx+ovf)
        ident = (cfg.lambda_s * acc["r_short"] + cfg.lambda_c * acc["comp"]
                 - cfg.lambda_m * (acc["miss"] + acc["retx"] + acc["ovf"]))
        assert abs(ident - acc["reward"]) < 1e-4, (ident, acc["reward"])
        res[(seed, name)] = acc
    print(f"seed {seed} done (n_active {env.traffic.n_active})", flush=True)

def agg(name, key):
    return np.array([res[(s, name)][key] for s in SEEDS], dtype=float)

print("\nper-scheduler means over 10 seeds (300 slots):")
hdr = ("sched", "reward", "r_short", "thr_part", "urg_part", "comp", "-2*miss",
       "-2*retx", "-2*ovf", "arrivals", "thr_Mbps")
print(("{:>9s}" + "{:>10s}" * (len(hdr) - 1)).format(*hdr))
ep_time = cfg.episode_len * cfg.slot_duration
for name in ("SUS+CQI", "SU+CQI"):
    print(f"{name:>9s}"
          f"{agg(name,'reward').mean():10.1f}{agg(name,'r_short').mean():10.1f}"
          f"{agg(name,'thr_part').mean():10.1f}{agg(name,'urg_part').mean():10.1f}"
          f"{agg(name,'comp').mean():10.1f}{-2*agg(name,'miss').mean():10.1f}"
          f"{-2*agg(name,'retx').mean():10.1f}{-2*agg(name,'ovf').mean():10.1f}"
          f"{agg(name,'arrivals').mean():10.1f}"
          f"{agg(name,'acked').mean()/ep_time/1e6:10.2f}")

print("\nmean gap decomposition (SUS - SU), fraction of total reward gap:")
d_reward = agg("SUS+CQI", "reward") - agg("SU+CQI", "reward")
d_rs = agg("SUS+CQI", "r_short") - agg("SU+CQI", "r_short")
d_thrp = agg("SUS+CQI", "thr_part") - agg("SU+CQI", "thr_part")
d_urgp = agg("SUS+CQI", "urg_part") - agg("SU+CQI", "urg_part")
d_c = agg("SUS+CQI", "comp") - agg("SU+CQI", "comp")
d_m = -2 * (agg("SUS+CQI", "miss") - agg("SU+CQI", "miss"))
d_x = -2 * (agg("SUS+CQI", "retx") - agg("SU+CQI", "retx"))
d_o = -2 * (agg("SUS+CQI", "ovf") - agg("SU+CQI", "ovf"))
tot = d_reward.mean()
for lbl, v in (("total reward gap", d_reward), ("r_short term", d_rs),
               ("  thr-proportional part", d_thrp), ("  urgency-weight part", d_urgp),
               ("completion term (+1/comp)", d_c), ("miss term (-2/miss)", d_m),
               ("retx-drop term (-2/drop)", d_x), ("retx-ovf term (-2/ovf)", d_o)):
    print(f"  {lbl:28s} {v.mean():+9.2f}  ({v.mean()/tot*100:+6.1f}% of gap)")
print(f"\ncheck: r_short + comp + miss + retx + ovf = "
      f"{(d_rs + d_c + d_m + d_x + d_o).mean():+.2f} vs total {tot:+.2f}")
d_thr300 = (agg("SUS+CQI", "acked") - agg("SU+CQI", "acked")) / ep_time / 1e6
print(f"300-slot thr gap: {d_thr300.mean():+.2f} Mbps "
      f"({d_thr300.mean()/(agg('SU+CQI','acked').mean()/ep_time/1e6)*100:+.2f}%), "
      f"reward gap {(d_reward.mean())/agg('SU+CQI','reward').mean()*100:+.2f}%")
cc300 = np.corrcoef(d_thr300, d_reward)[0, 1]
print(f"corr(d_thr, d_reward) 300-slot = {cc300:.3f}")
mw_sus = 1 + agg("SUS+CQI", "urg_part").sum() / agg("SUS+CQI", "thr_part").sum()
mw_su = 1 + agg("SU+CQI", "urg_part").sum() / agg("SU+CQI", "thr_part").sum()
print(f"mean deadline weight on useful bits: SUS {mw_sus:.3f}  SU {mw_su:.3f}")
