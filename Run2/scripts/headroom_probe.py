"""Headroom probe (READ-ONLY): measure scheduler reward spread under 4 configs.

For each config, run the 5 heuristic baselines over held-out seeds and report:
  - per-scheduler mean reward / completion / miss / throughput
  - SPREAD metrics (the headroom proxy):
      skill_gap   = (best_heuristic - Random) / best_heuristic
      spread      = (best - worst) / best
      dpf_vs_pf   = Deadline-PF advantage over PF  (deadline-awareness payoff)
      sus_vs_pf   = SUS+PF advantage over PF        (orthogonality payoff)
Wider gaps under harder configs => more room for a learned policy to win.
No source files are modified; configs are built in-memory.
"""
import json
import numpy as np
from config import phase2_main_config
from env import SchedulerEnv
from baselines import all_baselines

EP_LEN = 300          # short episodes: completion/miss are steady-state rates
SEEDS = [10000, 10001, 10002]

CONFIGS = {
    "C0_baseline   (p0.2 dl[5,30] eta1)": dict(p_arrival=0.2, deadline_min=5,  deadline_max=30, eta_d=1.0),
    "C1_user       (p0.4 dl[5,25] eta1)": dict(p_arrival=0.4, deadline_min=5,  deadline_max=25, eta_d=1.0),
    "C2_aggressive (p0.4 dl[3,12] eta1)": dict(p_arrival=0.4, deadline_min=3,  deadline_max=12, eta_d=1.0),
    "C3_aggr+wgt   (p0.4 dl[3,12] eta5)": dict(p_arrival=0.4, deadline_min=3,  deadline_max=12, eta_d=5.0),
}


def run_episode(env, sched):
    env.reset(episode_idx=None) if False else None
    return None


def eval_sched(env, sched, seeds):
    rew, comp, miss, thr = [], [], [], []
    cfg = env.cfg
    for s in seeds:
        env.reset(episode_idx=s)
        done = False
        while not done:
            alloc = sched.schedule(env)
            _, _, done, _ = env.step(alloc)
        e = env.ep
        arr = max(e["n_arrivals"], 1)
        ep_time = cfg.episode_len * cfg.slot_duration
        rew.append(e["reward"])
        comp.append(e["n_comp"] / arr)
        miss.append((e["n_miss_deadline"] + e["n_retx_drop"]) / arr)
        thr.append(e["acked_bits"] / ep_time / 1e6)
    return (float(np.mean(rew)), float(np.mean(comp)),
            float(np.mean(miss)), float(np.mean(thr)))


results = {}
for name, ov in CONFIGS.items():
    cfg = phase2_main_config(episode_len_main=EP_LEN, **ov)
    env = SchedulerEnv(cfg)
    scheds = all_baselines(cfg)               # Random, CQI-greedy, PF, Deadline-PF, SUS+PF
    row = {}
    for sch in scheds:
        r, c, m, t = eval_sched(env, sch, SEEDS)
        row[sch.name] = dict(reward=r, comp=c, miss=m, thr=t)
    results[name] = row

# ---- report ----
print("\n%-36s %-12s %8s %8s %8s %8s" % ("config", "scheduler", "reward", "comp", "miss", "Mbps"))
print("-" * 92)
summary = {}
for name, row in results.items():
    rewards = {k: v["reward"] for k, v in row.items()}
    best = max(rewards.values()); worst = min(rewards.values())
    rnd = rewards.get("Random", worst)
    pf = rewards.get("PF", worst)
    dpf = rewards.get("Deadline-PF", pf)
    sus = rewards.get("SUS+PF", pf)
    bestname = max(rewards, key=rewards.get)
    skill_gap = (best - rnd) / best * 100
    spread = (best - worst) / best * 100
    dpf_vs_pf = (dpf - pf) / pf * 100
    sus_vs_pf = (sus - pf) / pf * 100
    summary[name] = dict(best=bestname, skill_gap_pct=skill_gap, spread_pct=spread,
                         dpf_vs_pf_pct=dpf_vs_pf, sus_vs_pf_pct=sus_vs_pf,
                         mean_comp=float(np.mean([v["comp"] for v in row.values()])),
                         mean_miss=float(np.mean([v["miss"] for v in row.values()])))
    for k, v in row.items():
        print("%-36s %-12s %8.1f %8.3f %8.3f %8.2f" %
              (name, k, v["reward"], v["comp"], v["miss"], v["thr"]))
    print()

print("\n==== HEADROOM SUMMARY (wider = more room to learn) ====")
print("%-36s %10s %10s %10s %10s %8s %8s" %
      ("config", "skill_gap", "spread", "DPF-PF", "SUS-PF", "comp", "miss"))
for name, s in summary.items():
    print("%-36s %9.2f%% %9.2f%% %9.2f%% %9.2f%% %7.3f %7.3f" %
          (name, s["skill_gap_pct"], s["spread_pct"], s["dpf_vs_pf_pct"],
           s["sus_vs_pf_pct"], s["mean_comp"], s["mean_miss"]))

print("\n==== JSON ====")
print(json.dumps({"results": results, "summary": summary}, indent=1))
