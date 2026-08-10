"""PAPER figure: main-result bars with 95% CI, IEEE 1-column 2x2 grid.

Spec (user, 2026-08-10):
  * panels: (a) Episode reward, (b) Goodput [Mbps], (c) Deadline miss
    rate, (d) MU depth -- no SINR/throughput/JFI/completion (redundant
    with the main table), no stacked failure panel (retx-drop ~1e-4).
  * bars = mean over seeds; error bars = 95% CI across seeds
    (t_{n-1} * std/sqrt(n)). NOTE: these CIs describe each scheduler's
    absolute-performance spread; the PPO-vs-baseline VERDICT is the
    paired-difference statistic in the main table -- state this role
    split in the caption.
  * schedulers: main = 7 (PPO + {SUS,SU} x {CQI, PF, DPF}); appendix
    variant = all 9 incl. Random (--appendix).
  * colors: PPO red / SUS family blue / SU family green.
  * size: IEEE single column, 3.5 in wide.

Input CSV defaults to the FINAL band (20000-20049) merged file; pass any
compatible per-seed CSV (e.g. final40) to test-render before the final
data lands:  python3 paper_fig_main_result.py [csv] [--appendix]
"""
import os, sys, csv
import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/MYH/ML_DRL_Scheduler")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
APPENDIX = "--appendix" in sys.argv
SRC = args[0] if args else "Run4/_analysis/queue_s40hl_cqi4_final100.csv"

# display set (user-final 2026-08-10): {SUS,SU} x {CQI, PF, Random};
# DPF pairs live in the appendix 9-scheduler variant only.
MAIN7 = ["PPO", "SUS+CQI", "SUS+PF", "SUS+Rnd", "SU+CQI", "SU+PF", "SU+Rnd"]
ALL9 = ["PPO", "SUS+CQI", "SUS+PF", "SUS+DPF", "SUS+Rnd",
        "SU+CQI", "SU+PF", "SU+DPF", "SU+Rnd"]
ORDER = ALL9 if APPENDIX else MAIN7
COLOR = {n: ("#c1272d" if n == "PPO" else "#2b6ca3" if n.startswith("SUS") else "#2e8b57")
         for n in ORDER}

rows = list(csv.DictReader(open(SRC)))
seeds = sorted({int(r["seed"]) for r in rows})
n = len(seeds)
tcrit = st.t.ppf(0.975, n - 1)
print(f"source {SRC}: n={n} seeds, t_crit={tcrit:.3f}")

def stat(name, key):
    v = np.array([float(r[key]) for r in rows if r["sched"] == name])
    assert len(v) == n, (name, key, len(v))
    return v.mean(), tcrit * v.std(ddof=1) / np.sqrt(n)

PANELS = [("(a) Episode reward", "reward", "{:.0f}", 1.0),
          ("(b) Goodput [Mbps]", "goodput_mbps", "{:.1f}", 1.0),
          ("(c) Deadline miss rate", "deadline_miss_rate", "{:.2f}", 1.0),
          ("(d) MU depth", "mu_depth", "{:.2f}", 1.0)]

plt.rcParams.update({"font.size": 6.0, "axes.titlesize": 6.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.4))
for ax, (title, key, fmt, _) in zip(axes.flat, PANELS):
    ms, cis = zip(*(stat(nm, key) for nm in ORDER))
    x = np.arange(len(ORDER))
    ax.bar(x, ms, color=[COLOR[nm] for nm in ORDER], width=0.72,
           yerr=cis, error_kw=dict(elinewidth=0.6, capsize=1.4, capthick=0.6,
                                   ecolor="#333333"))
    ax.set_title(title, pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=60, ha="right", fontsize=5.0)
    ax.tick_params(axis="y", labelsize=5.2, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.margins(x=0.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.0, color="#888888", lw=0.5)
    for xi, (m, ci) in enumerate(zip(ms, cis)):
        ax.annotate(fmt.format(m), (xi, m + (ci if m >= 0 else -ci)),
                    ha="center",
                    va="bottom" if m >= 0 else "top",
                    fontsize=4.4, xytext=(0, 1 if m >= 0 else -1),
                    textcoords="offset points")
fig.tight_layout(pad=0.35, h_pad=0.8, w_pad=0.6)
suffix = "_appendix9" if APPENDIX else ""
base = "Run4/QueuePostRZF_S40HL_CQI4/paper_main_result" + suffix
fig.savefig(base + ".png", dpi=300)
fig.savefig(base + ".pdf")
print("saved ->", base + ".{png,pdf}")
print("caption 메모: bars = seed-mean ± 95% CI (t_{n-1}, across-seed spread of "
      "absolute performance); PPO-vs-baseline verdicts are the paired-difference "
      "tests in the main table.")
