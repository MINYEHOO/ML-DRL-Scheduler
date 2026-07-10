"""Oracle (upper-bound) probe, READ-ONLY.

For C0 (easy) and C2 (aggressive), run the 5 heuristics at:
  - stale CSI  (p_csi=0.2, what schedulers / PPO actually see)
  - fresh CSI  (p_csi=1.0, ORACLE: zero staleness; PMI still quantized)
Heuristics ignore Age, so fresh-vs-stale gap = reward locked behind CSI
staleness that a staleness-aware LEARNED policy could partially capture.
This is the room ABOVE the stale heuristics (the band PPO competes in).
Config built in-memory; no source files changed.
"""
import json
import numpy as np
from config import phase2_main_config
from env import SchedulerEnv
from baselines import all_baselines

EP_LEN = 300
SEEDS = [10000, 10001, 10002]

# (label, overrides) — C0 easy, C2 aggressive
SCEN = {
    "C0 (p_arr0.2 dl[5,30])": dict(p_arrival=0.2, deadline_min=5, deadline_max=30, eta_d=1.0),
    "C2 (p_arr0.4 dl[3,12])": dict(p_arrival=0.4, deadline_min=3, deadline_max=12, eta_d=1.0),
}
PCSI = {"stale(0.2)": 0.2, "fresh-oracle(1.0)": 1.0}


def eval_sched(env, sched, seeds):
    rew, comp = [], []
    cfg = env.cfg
    for s in seeds:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        e = env.ep
        arr = max(e["n_arrivals"], 1)
        rew.append(e["reward"]); comp.append(e["n_comp"] / arr)
    return float(np.mean(rew)), float(np.mean(comp))


results = {}
for sname, ov in SCEN.items():
    results[sname] = {}
    for pname, pcsi in PCSI.items():
        cfg = phase2_main_config(episode_len_main=EP_LEN, p_csi=pcsi, **ov)
        env = SchedulerEnv(cfg)
        row = {}
        for sch in all_baselines(cfg):
            r, c = eval_sched(env, sch, SEEDS)
            row[sch.name] = dict(reward=r, comp=c)
        results[sname][pname] = row

print("\n%-26s %-18s %-12s %9s %7s" % ("scenario", "CSI", "scheduler", "reward", "comp"))
print("-" * 78)
summary = {}
for sname, byp in results.items():
    best = {}
    for pname, row in byp.items():
        bn = max(row, key=lambda k: row[k]["reward"])
        best[pname] = (bn, row[bn]["reward"])
        for k, v in row.items():
            print("%-26s %-18s %-12s %9.1f %7.3f" % (sname, pname, k, v["reward"], v["comp"]))
        print()
    stale_best = best["stale(0.2)"][1]
    fresh_best = best["fresh-oracle(1.0)"][1]
    summary[sname] = dict(
        stale_best_sched=best["stale(0.2)"][0], stale_best=stale_best,
        fresh_best_sched=best["fresh-oracle(1.0)"][0], fresh_best=fresh_best,
        oracle_gap_pct=(fresh_best - stale_best) / stale_best * 100)

print("\n==== ORACLE HEADROOM (room ABOVE stale heuristics) ====")
print("%-26s %18s %18s %14s" % ("scenario", "stale best", "fresh-oracle best", "oracle gap"))
for sname, s in summary.items():
    print("%-26s %10s %7.1f %10s %7.1f %12.2f%%" % (
        sname, s["stale_best_sched"], s["stale_best"],
        s["fresh_best_sched"], s["fresh_best"], s["oracle_gap_pct"]))

print("\n==== JSON ====")
print(json.dumps({"results": results, "summary": summary}, indent=1))
