"""Speed sweep oracle probe, READ-ONLY.

C2 config (p_arrival=0.4, deadline[3,12]) fixed. Vary UE speed {3,30,60} km/h
x CSI {stale 0.2, fresh-oracle 1.0}. Faster UE -> channel decorrelates -> stale
CSI actually wrong -> fresh-CSI oracle should beat the (Age-blind) heuristics.
oracle_gap = (fresh_best - stale_best)/stale_best = room ABOVE stale heuristics
that a staleness-aware learned policy could chase.
Config built in-memory; no source files changed.
"""
import json
import numpy as np
from config import phase2_main_config
from env import SchedulerEnv
from baselines import all_baselines

EP_LEN = 300
SEEDS = [10000, 10001, 10002]
C2 = dict(p_arrival=0.4, deadline_min=3, deadline_max=12, eta_d=1.0)
SPEEDS = [3.0, 30.0, 60.0]
PCSI = {"stale(0.2)": 0.2, "fresh-oracle(1.0)": 1.0}


def eval_sched(env, sched, seeds):
    rew, comp, sinr = [], [], []
    for s in seeds:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        e = env.ep
        rew.append(e["reward"]); comp.append(e["n_comp"] / max(e["n_arrivals"], 1))
        sinr.append(10*np.log10(e["sinr_sum"]/e["sinr_count"]) if e["sinr_count"] else float("nan"))
    return float(np.mean(rew)), float(np.mean(comp)), float(np.mean(sinr))


results, summary = {}, {}
for spd in SPEEDS:
    key = f"{spd:.0f}km/h"
    results[key] = {}
    best = {}
    for pname, pcsi in PCSI.items():
        cfg = phase2_main_config(episode_len_main=EP_LEN, ue_speed_kmh=spd,
                                 p_csi=pcsi, **C2)
        env = SchedulerEnv(cfg)
        row = {}
        for sch in all_baselines(cfg):
            r, c, s = eval_sched(env, sch, SEEDS)
            row[sch.name] = dict(reward=r, comp=c, sinr=s)
        results[key][pname] = row
        bn = max(row, key=lambda k: row[k]["reward"])
        best[pname] = (bn, row[bn]["reward"], row[bn]["sinr"])
    sb, fb = best["stale(0.2)"][1], best["fresh-oracle(1.0)"][1]
    summary[key] = dict(stale_best=best["stale(0.2)"], fresh_best=best["fresh-oracle(1.0)"],
                        oracle_gap_pct=(fb - sb) / sb * 100)

print("\n%-8s %-18s %-12s %9s %7s %7s" % ("speed", "CSI", "sched", "reward", "comp", "SINRdB"))
print("-" * 70)
for spd, byp in results.items():
    for pname, row in byp.items():
        for k, v in row.items():
            print("%-8s %-18s %-12s %9.1f %7.3f %7.2f" % (spd, pname, k, v["reward"], v["comp"], v["sinr"]))
        print()

print("\n==== SPEED x ORACLE GAP (room above stale heuristics) ====")
print("%-8s %22s %22s %12s" % ("speed", "stale best", "fresh-oracle best", "oracle gap"))
for spd, s in summary.items():
    sb, fb = s["stale_best"], s["fresh_best"]
    print("%-8s %12s %7.1f(%4.1fdB) %12s %7.1f %10.2f%%" % (
        spd, sb[0], sb[1], sb[2], fb[0], fb[1], s["oracle_gap_pct"]))

print("\n==== JSON ====")
print(json.dumps({"results": results, "summary": {k: {"oracle_gap_pct": v["oracle_gap_pct"],
      "stale_best": v["stale_best"], "fresh_best": v["fresh_best"]} for k, v in summary.items()}}, indent=1))
