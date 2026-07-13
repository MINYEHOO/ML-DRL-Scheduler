import csv, math
from collections import defaultdict

BASE = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis"

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def t_ci(diffs, tcrit=2.262157163):  # df=9, 95% two-sided
    n = len(diffs)
    m = sum(diffs)/n
    var = sum((d-m)**2 for d in diffs)/(n-1)
    se = math.sqrt(var/n)
    return m, m - tcrit*se, m + tcrit*se, se, n

for world in ("type2", "genie"):
    rows = load(f"{BASE}/la_ablation_{world}.csv")
    for la in ("0", "1"):
        sub = [r for r in rows if r["la"] == la]
        # organize per seed
        per_seed = defaultdict(dict)
        for r in sub:
            per_seed[r["seed"]][r["sched"]] = float(r["reward"])
        seeds = sorted(per_seed)
        scheds = sorted({r["sched"] for r in sub})
        fixed = [s for s in scheds if s != "PPO-L2b"]
        print(f"\n=== {world} la={la} | seeds={len(seeds)} | fixed scheds={fixed}")

        # 1) pooled means per scheduler
        pooled = {}
        for s in scheds:
            vals = [per_seed[sd][s] for sd in seeds]
            pooled[s] = sum(vals)/len(vals)
        rank = sorted(pooled.items(), key=lambda kv: -kv[1])
        print("  pooled means:", {k: round(v,1) for k,v in rank})
        best_fixed_name = max(fixed, key=lambda s: pooled[s])
        print(f"  PPO pooled rank: {[k for k,_ in rank].index('PPO-L2b')+1}"
              f" | PPO - best fixed pooled = {pooled['PPO-L2b']-pooled[best_fixed_name]:+.1f} (vs {best_fixed_name})")
        # paired CI vs best pooled fixed scheduler
        d_best = [per_seed[sd]["PPO-L2b"] - per_seed[sd][best_fixed_name] for sd in seeds]
        m,lo,hi,se,n = t_ci(d_best)
        print(f"  paired PPO - {best_fixed_name}: {m:+.1f} CI[{lo:+.1f},{hi:+.1f}]")

        # 2) per-seed envelope
        env_wins = 0
        d_env = []
        argmax_count = defaultdict(int)
        for sd in seeds:
            env = max(per_seed[sd][s] for s in fixed)
            am = max(fixed, key=lambda s: per_seed[sd][s])
            argmax_count[am] += 1
            d = per_seed[sd]["PPO-L2b"] - env
            d_env.append(d)
            if d > 0: env_wins += 1
        m,lo,hi,se,n = t_ci(d_env)
        print(f"  ENVELOPE: PPO - max(fixed): {m:+.1f} CI[{lo:+.1f},{hi:+.1f}] "
              f"(se={se:.1f}, n={n}) | PPO wins {env_wins}/{len(seeds)} seeds")
        print(f"  envelope argmax composition: {dict(argmax_count)}")
        print(f"  per-seed diffs: {[round(d,1) for d in d_env]}")
