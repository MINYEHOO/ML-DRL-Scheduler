"""Merge the six rejection workers, verify against the official final100,
and compute the manuscript statistics for r_PPO / r_SUS+CQI.

Verification (fails loudly, never silently):
  * 100 seeds x 2 schedulers = 200 rows, no missing seed, no duplicate row,
    no NaN;
  * n_offered == n_arrivals + n_buffer_overflow for every episode;
  * buffer_overflow_rate == n_buffer_overflow / max(n_offered, 1);
  * per-seed reproduction of the official Run4/_analysis/
    queue_s40hl_cqi4_final100.csv on reward / throughput / goodput /
    completion / deadline-miss / retx-drop, within the rounding tolerance of
    that file (values there are round(.,4)).

Statistics (per scheduler and paired):
  * ACROSS-EPISODE MEAN of the per-episode rate -- this is the manuscript
    number;
  * Student-t 95% CI of that mean (df = 99);
  * total accepted / rejected / offered counts;
  * number of episodes with at least one rejection;
  * POOLED rate sum(N_rej)/sum(N_off) -- reported for reference only, NOT
    the manuscript value;
  * paired per-episode difference 100*(r_PPO,s - r_SUS,s) in percentage
    points, with its Student-t 95% CI.
"""
import csv, glob, math, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

BASE = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis"
OFFICIAL = f"{BASE}/queue_s40hl_cqi4_final100.csv"
MERGED = f"{BASE}/queue_s40hl_cqi4_final100_rejection.csv"
SUMMARY = f"{BASE}/queue_s40hl_cqi4_rejection_stats.txt"
SCHEDS = ["PPO", "SUS+CQI"]
SEEDS = list(range(20000, 20100))
VERIFY_KEYS = ["reward", "throughput_mbps", "goodput_mbps",
               "completion_rate", "deadline_miss_rate", "retx_drop_rate"]

out = []
rows = []
for p in sorted(glob.glob(f"{BASE}/queue_s40hl_cqi4_final100_rejection_w*.csv")):
    rows += list(csv.DictReader(open(p)))
say = lambda s: (print(s), out.append(s))

say("=" * 72)
say("STRUCTURAL VERIFICATION")
say("=" * 72)
say(f"rows read: {len(rows)} (expect 200)")
assert len(rows) == 200, f"expected 200 rows, got {len(rows)}"
keyset = [(int(r["seed"]), r["sched"]) for r in rows]
assert len(set(keyset)) == 200, "duplicate (seed, sched) rows present"
for s in SEEDS:
    for a in SCHEDS:
        assert (s, a) in keyset, f"missing row: seed {s} {a}"
say("  200 unique (seed, scheduler) rows, all 100 seeds x 2 schedulers present")

by = {}
for r in rows:
    rec = dict(seed=int(r["seed"]), sched=r["sched"],
               n_arrivals=int(r["n_arrivals"]),
               n_rej=int(r["n_buffer_overflow"]),
               n_off=int(r["n_offered"]),
               rate=float(r["buffer_overflow_rate"]))
    for k in VERIFY_KEYS:
        rec[k] = float(r[k])
    assert all(np.isfinite(v) for k, v in rec.items()
               if isinstance(v, float)), f"NaN/inf in seed {rec['seed']}"
    assert rec["n_off"] == rec["n_arrivals"] + rec["n_rej"], \
        f"n_offered mismatch seed {rec['seed']} {rec['sched']}"
    assert abs(rec["rate"] - rec["n_rej"] / max(rec["n_off"], 1)) < 1e-12, \
        f"rate identity violated seed {rec['seed']} {rec['sched']}"
    by[(rec["seed"], rec["sched"])] = rec
say("  no NaN/inf; n_offered identity holds in all 200 episodes")
say("  buffer_overflow_rate == n_rejected / max(n_offered,1) in all 200 episodes")

# merged CSV (single canonical artifact)
with open(MERGED, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["seed", "sched", "n_arrivals", "n_buffer_overflow",
                 "n_offered", "buffer_overflow_rate"] + VERIFY_KEYS)
    for s in SEEDS:
        for a in SCHEDS:
            r = by[(s, a)]
            wr.writerow([r["seed"], r["sched"], r["n_arrivals"], r["n_rej"],
                         r["n_off"], repr(r["rate"])]
                        + [repr(r[k]) for k in VERIFY_KEYS])
say(f"  merged -> {MERGED}")

say("")
say("=" * 72)
say("REPRODUCIBILITY vs OFFICIAL final100")
say("=" * 72)
say("The official file stores round(x, 4). The exact criterion is therefore")
say("round(new, 4) == official, not a raw absolute tolerance: a value sitting")
say("on the rounding half-boundary (e.g. 62.22175 -> 62.2218) differs by")
say("exactly 5e-5, which a naive |d| < 5e-5 test would flag as a mismatch.")
say("Both half-even and half-up rounding are accepted, since the writer's")
say("tie-breaking convention is not recoverable from the stored file.")
off = {}
for r in csv.DictReader(open(OFFICIAL)):
    if r["sched"] in SCHEDS:
        off[(int(r["seed"]), r["sched"])] = r
say(f"official rows for the two schedulers: {len(off)} (expect 200)")
assert len(off) == 200, "official final100 does not carry 200 rows for these two"
worst = {k: (0.0, None) for k in VERIFY_KEYS}
nbad = 0
for (s, a), rec in by.items():
    o = off[(s, a)]
    for k in VERIFY_KEYS:
        ov = float(o[k])
        d = abs(rec[k] - ov)
        if d > worst[k][0]:
            worst[k] = (d, (s, a))
        r_even = round(rec[k], 4)
        r_up = float(Decimal(repr(rec[k])).quantize(Decimal("0.0001"),
                                                    rounding=ROUND_HALF_UP))
        if r_even != ov and r_up != ov:
            nbad += 1
            if nbad <= 10:
                say(f"  MISMATCH seed {s} {a} {k}: new {rec[k]!r} "
                    f"-> {r_even}/{r_up} vs official {o[k]}")
for k in VERIFY_KEYS:
    d, where = worst[k]
    say(f"  max raw |diff| {k:20s} = {d:.3e}   at {where}   "
        f"(<= 5e-5 = the 4-dp rounding half-width)")
say(f"  values compared: {200 * len(VERIFY_KEYS)}   exact-rounding mismatches: {nbad}")
if nbad:
    say(f"  *** {nbad} value(s) do NOT reproduce -- DO NOT finalize; "
        f"diagnose first ***")
    sys.exit(1)
say("  ALL 200 episodes x 6 metrics reproduce the official final100 EXACTLY")
say("  under the file's own 4-dp rounding -> the re-run is protocol-identical,")
say("  so the rejection counters below come from the same evaluation.")

def tci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, m, m
    se = x.std(ddof=1) / math.sqrt(n)
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h

say("")
say("=" * 72)
say("FULL-BUFFER REJECTION STATISTICS  (seeds 20000-20099, n=100)")
say("=" * 72)
res = {}
for a in SCHEDS:
    r = np.array([by[(s, a)]["rate"] for s in SEEDS])
    acc = np.array([by[(s, a)]["n_arrivals"] for s in SEEDS])
    rej = np.array([by[(s, a)]["n_rej"] for s in SEEDS])
    offc = np.array([by[(s, a)]["n_off"] for s in SEEDS])
    m, lo, hi = tci(100.0 * r)
    pooled = 100.0 * rej.sum() / offc.sum()
    res[a] = dict(mean=m, lo=lo, hi=hi, pooled=pooled,
                  acc=int(acc.sum()), rej=int(rej.sum()), off=int(offc.sum()),
                  nz=int((rej > 0).sum()))
    say(f"[{a}]")
    say(f"  across-episode MEAN rejection rate (MANUSCRIPT VALUE)")
    say(f"      = {m:.10f} %   (95% CI [{lo:.10f}, {hi:.10f}] %)")
    say(f"  totals: accepted {res[a]['acc']:,}  rejected {res[a]['rej']:,}  "
        f"offered {res[a]['off']:,}")
    say(f"  episodes with >=1 rejection: {res[a]['nz']}/100")
    say(f"  POOLED rate (reference only, NOT the manuscript value) "
        f"= {pooled:.10f} %")
    say("")

d = 100.0 * np.array([by[(s, "PPO")]["rate"] - by[(s, "SUS+CQI")]["rate"]
                      for s in SEEDS])
m, lo, hi = tci(d)
say("[paired per-episode difference, percentage points]")
say(f"  100*(r_PPO,s - r_SUS+CQI,s):  mean {m:+.10f} pp   "
    f"95% CI [{lo:+.10f}, {hi:+.10f}] pp")
say(f"  episodes PPO strictly lower: {int((d < 0).sum())}/100; "
    f"equal: {int((d == 0).sum())}/100; higher: {int((d > 0).sum())}/100")

say("")
say("=" * 72)
say("MANUSCRIPT SENTENCE")
say("=" * 72)
say(f"  full precision : X = {res['PPO']['mean']:.10f} %, "
    f"Y = {res['SUS+CQI']['mean']:.10f} %")
for nd in (2, 3, 4):
    say(f"  rounded to {nd}dp: X = {res['PPO']['mean']:.{nd}f} %, "
        f"Y = {res['SUS+CQI']['mean']:.{nd}f} %")
say("")
say('  "The across-episode mean full-buffer rejection rates are '
    f'{res["PPO"]["mean"]:.3f}% for PPO and '
    f'{res["SUS+CQI"]["mean"]:.3f}% for SUS+CQI, respectively."')

with open(SUMMARY, "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"\nsummary -> {SUMMARY}")
