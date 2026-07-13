"""Recompute audit claims F3, F4, F5, F7 from archived CSVs (read-only)."""
import csv, math, os
from collections import defaultdict

R = "/home/MYH/ML_DRL_Scheduler"
A = f"{R}/Run3/_analysis_20260702"

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# ---------- F4: official9full PPO vs SUS+CQI ----------
print("=" * 70)
print("F4: official9full PPO vs SUS+CQI (20 seeds, paired)")
for run in ["MixedLoad_L2", "MixedSpeed_L2b"]:
    rows = load(f"{A}/official9full_{run}.csv")
    by = defaultdict(dict)
    for r in rows:
        by[r["scheduler"]][r["seed"]] = r
    seeds = sorted(by["PPO"].keys())
    assert len(seeds) == 20, len(seeds)
    def mean(sched, col):
        return sum(float(by[sched][s][col]) for s in seeds) / len(seeds)
    print(f"\n  {run} (n={len(seeds)} seeds):")
    for col in ["reward", "thr_mbps", "comp", "miss", "retx_drop", "total_fail", "jain"]:
        p, b = mean("PPO", col), mean("SUS+CQI", col)
        rel = (p - b) / b * 100
        print(f"    {col:12s} PPO {p:10.4f} vs SUS+CQI {b:10.4f}  rel {rel:+7.2f}%")

# ---------- F5: recompute CIs with t-critical ----------
print("\n" + "=" * 70)
print("F5: normal-1.96 vs t-critical CIs on borderline verdicts")
try:
    from scipy import stats as st
    tppf = lambda df: st.t.ppf(0.975, df)
except ImportError:
    # hardcoded two-sided 97.5% t quantiles
    T = {19: 2.0930, 39: 2.0227, 59: 2.0010}
    tppf = lambda df: T[df]

def paired_ci(diffs, label):
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = tppf(n - 1)
    lo96, hi96 = m - 1.96 * se, m + 1.96 * se
    lot, hit = m - t * se, m + t * se
    wins = sum(1 for d in diffs if d > 0)
    flip = (lo96 > 0) != (lot > 0) or (hi96 < 0) != (hit < 0)
    print(f"  {label}: n={n} mean={m:+.1f} wins={wins}/{n}")
    print(f"    1.96 CI  [{lo96:+.1f}, {hi96:+.1f}]  excl0={lo96>0 or hi96<0}")
    print(f"    t({n-1})={t:.4f} CI [{lot:+.1f}, {hit:+.1f}]  excl0={lot>0 or hit<0}  FLIP={flip}")
    return flip

def diffs_from(files, ppo_name, base_name):
    per = {}
    for fp in files:
        for r in load(fp):
            per.setdefault(r["scheduler"], {})[r["seed"]] = float(r["reward"])
    seeds = sorted(per[ppo_name].keys())
    return [per[ppo_name][s] - per[base_name][s] for s in seeds], seeds

# ScarcityK32 n=40
d, s = diffs_from([f"{A}/stats20_ScarcityK32.csv", f"{A}/stats20b_ScarcityK32.csv"],
                  "PPO-best", "SUS-CQI@0.8")
paired_ci(d, f"ScarcityK32 PPO vs SUS+CQI@0.8 (seeds {s[0]}..{s[-1]})")

# DeadlineScarcity n=60
d, s = diffs_from([f"{A}/stats20_DeadlineScarcity.csv", f"{A}/stats20b_DeadlineScarcity.csv",
                   f"{A}/stats20c_DeadlineScarcity.csv"], "PPO-best", "SU-CQI")
paired_ci(d, f"DeadlineScarcity PPO vs SU+CQI (seeds {s[0]}..{s[-1]})")

# Uniform10 n=40 (tie verdict)
d, s = diffs_from([f"{A}/stats20_Uniform10_Ent002.csv", f"{A}/stats20b_Uniform10_Ent002.csv"],
                  "PPO-best", "SU-CQI")
paired_ci(d, f"Uniform10 PPO vs SU+CQI (seeds {s[0]}..{s[-1]})")

# Envelope n=20: final20_<run>.csv — oracle envelope = per-seed max(SU-CQI, SUS-CQI@0.8)
for run in ["MixedLoad_L2", "MixedSpeed_L2b"]:
    rows = load(f"{A}/final20_{run}.csv")
    per = defaultdict(dict)
    for r in rows:
        per[r["scheduler"]][r["seed"]] = float(r["reward"])
    print(f"  [{run}] schedulers present: {sorted(per.keys())}")
    seeds = sorted(per[list(per.keys())[0]].keys())
    ppo_key = [k for k in per if "PPO" in k.upper()][0]
    env_diffs = [per[ppo_key][s] - max(per["SU-CQI"][s], per["SUS-CQI@0.8"][s]) for s in seeds]
    paired_ci(env_diffs, f"{run} PPO vs per-seed oracle envelope")

# ---------- F3: OOD per-seed inflation ----------
print("\n" + "=" * 70)
print("F3: OOD p=0.50 per-seed margins vs per-seed-best baseline")
rows = load(f"{R}/Run4/_analysis/ood_p_arrival.csv")
for p in ["0.45", "0.5"]:
    sub = [r for r in rows if r["p_arrival"] == p]
    seeds = sorted({r["seed"] for r in sub})
    pcts, ppos, bests = [], [], []
    for s in seeds:
        rr = {r["sched"]: float(r["reward"]) for r in sub if r["seed"] == s}
        ppo = rr["PPO-Mixed"]
        best = max(v for k, v in rr.items() if k != "PPO-Mixed")
        pcts.append((ppo - best) / abs(best) * 100)
        ppos.append(ppo); bests.append(best)
    mp = sum(ppos) / len(ppos); mb = sum(bests) / len(bests)
    print(f"  p={p}: n={len(seeds)} mean-of-per-seed-% = {sum(pcts)/len(pcts):+.1f}%")
    print(f"    per-seed %: {[f'{x:+.0f}' for x in pcts]}")
    print(f"    mean PPO {mp:.0f} vs mean best-baseline {mb:.0f} -> pooled {100*(mp-mb)/mb:+.1f}%")
    print(f"    per-seed best-baseline values: {[f'{b:.0f}' for b in bests]}")

# ---------- F7: MixedLoad_L2 best -> last decline ----------
print("\n" + "=" * 70)
print("F7: MixedLoad_L2 eval trajectory best vs last")
ev = f"{R}/Run3/MixedLoad_L2/csv_logs/eval_metrics.csv"
if os.path.exists(ev):
    rows = load(ev)
    ppo = [(int(r[list(r.keys())[0]]), r) for r in rows if r.get("scheduler", r.get("sched", "PPO")) == "PPO" or "PPO" in str(r.values())]
    print(f"  columns: {list(rows[0].keys())}")
else:
    print("  eval_metrics.csv not found at", ev)
