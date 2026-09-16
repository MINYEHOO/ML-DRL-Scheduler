"""NoUserHead ablation vs base vs key baselines -- the 9 standard metrics on
the blind holdout (seeds 21000-21099, n=100), in the same bar+CI style as the
paper's main-result figure.

Panels (project-standard order): reward / throughput / SINR / completion /
deadline-miss / retx-drop / total-fail / MU depth / JFI.
Bars = mean over the 100 episodes; error bars = 95% CI across episodes
(t_99 * sd / 10). These CIs describe each scheduler's absolute spread; the
PPO-vs-X VERDICT is the paired-difference statistic (see
plot_nouser_ablation.py and the holdout tables), so the caption states that.

Color = family: learned policies in one blue ramp ordered by alpha (base is the
darkest), FullRank in violet, rule baselines in warm neutrals. Identity is also
carried by the x label, so color is never the only cue.

PNG -> base run folder (project rule).  Also prints the 9-metric table.
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

os.chdir("/home/MYH/ML_DRL_Scheduler")
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
OUT = f"{BASE}/nouser_ablation_metrics.png"

# ---------------- load ----------------
pol = defaultdict(dict)
for f in glob.glob("Run4/_analysis/seedreplicate_final100_h21_*.csv"):
    tag = f.split("h21_")[1][:-4]
    for r in csv.DictReader(open(f)):
        if r["sched"] == "PPO":
            pol[tag][int(r["seed"])] = r
base = {}
for c in "abc":
    base.update(pol.get(f"2024_{c}", {}))
pol["2024"] = base
bl = defaultdict(dict)
for f in glob.glob("Run4/_analysis/holdout21_baselines_w*.csv"):
    for r in csv.DictReader(open(f)):
        bl[r["sched"]][int(r["seed"])] = r
seeds = sorted(base)
assert len(seeds) == 100

# display order + family color
SERIES = [  # label, rows, color
    ("PPO base\nα=1.0",  pol["2024"],   "#1f4f8f"),
    ("α=1.2",            pol["NUS120"], "#2a78d6"),
    ("α=1.4",            pol["NUS140"], "#4f95e3"),
    ("α=0.75",           pol["NUS075"], "#7fb2ec"),
    ("α=0.5",            pol["NUS050"], "#a8c9f2"),
    ("α=0\n(head off)",  pol["NUS0"],   "#cfe0f8"),
    ("FullRank\n(no stop)", pol["FullRank"], "#4a3aa7"),
    ("SUS+CQI\nFeasible", bl["SUS+CQI-Feasible"], "#eb6834"),
    ("SUS-RPS",          bl["SUS-RPS"],  "#eda100"),
    ("SUS+CQI",          bl["SUS+CQI"],  "#8a8781"),
]
for lab, rows, _ in SERIES:
    assert len(rows) == 100, (lab, len(rows))

def col(rows, key):
    if key == "total_fail":
        return np.array([float(rows[s]["deadline_miss_rate"]) +
                         float(rows[s]["retx_drop_rate"]) for s in seeds])
    return np.array([float(rows[s][key]) for s in seeds])

METRICS = [  # key, title, scale, fmt
    ("reward",             "Episode return",          1,   "{:.0f}"),
    ("throughput_mbps",    "Throughput [Mbit/s]",     1,   "{:.2f}"),
    ("mean_sinr_db",       "Mean SINR [dB]",          1,   "{:.2f}"),
    ("completion_rate",    "Completion rate [%]",     100, "{:.2f}"),
    ("deadline_miss_rate", "Deadline miss rate [%]",  100, "{:.2f}"),
    ("retx_drop_rate",     "Retx-drop rate [%]",      100, "{:.3f}"),
    ("total_fail",         "Total failure rate [%]",  100, "{:.2f}"),
    ("mu_depth",           "MU depth [streams/RBG]",  1,   "{:.2f}"),
    ("jain",               "Jain fairness index",     1,   "{:.3f}"),
]
GOODPUT = ("goodput_mbps", "Goodput [Mbit/s]", 1, "{:.2f}")   # table only

def stat(rows, key, scale):
    v = col(rows, key) * scale
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))

# ---------------- figure ----------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), facecolor=SURF)
x = np.arange(len(SERIES))
for ax, (key, title, scale, fmt) in zip(axes.flat, METRICS):
    ms, cis = zip(*(stat(rows, key, scale) for _, rows, _ in SERIES))
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_xlim(-0.6, len(SERIES) - 0.4)
    ax.set_axisbelow(True)
    # dot + CI (not bars): axes are zoomed to the data, and a truncated bar
    # would exaggerate differences -- a point estimate implies no baseline
    for xi, m, c, (_, _, hue) in zip(x, ms, cis, SERIES):
        ax.plot([xi, xi], [m - c, m + c], color=hue, linewidth=2.2,
                solid_capstyle="round", zorder=3)
        ax.plot([xi], [m], marker="o", ms=8, color=hue, markeredgecolor=SURF,
                markeredgewidth=1.5, zorder=4)
    ax.axvline(6.5, color=GRID, linewidth=1)          # learned | rule divider
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in SERIES], rotation=55, ha="right",
                       fontsize=7.6, color=INK)
    # zoom the y-range to the data so differences read, keep 0 only when close
    lo, hi = min(m - c for m, c in zip(ms, cis)), max(m + c for m, c in zip(ms, cis))
    pad = (hi - lo) * 0.18 if hi > lo else 1
    ax.set_ylim(lo - pad if lo - pad > 0 or lo < 0 else 0, hi + pad)
    # value labels on the base bar and the best bar, not on every bar
    best = int(np.argmax(ms)) if key not in ("deadline_miss_rate", "retx_drop_rate", "total_fail") else int(np.argmin(ms))
    for i in sorted({0, best}):
        ax.annotate(fmt.format(ms[i]), (x[i], ms[i] + cis[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK)
fig.suptitle("NoUserHead ablation vs base vs rule baselines — blind holdout 21000–21099 (n=100 episodes)",
             x=0.01, ha="left", color=INK, fontsize=12.5)
fig.text(0.01, 0.005,
         "Dots: mean over 100 episodes; whiskers: 95% CI across episodes (absolute spread). Left of the divider = learned policies, right = rule baselines.\n"
         "The PPO-vs-X verdict is the PAIRED difference: every α≠1.0 and FullRank are significantly below base on return; base ties SUS+CQI-Feasible.",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.035, 1, 0.965))
fig.savefig(OUT, dpi=170, facecolor=SURF)
print("saved", OUT)

# ---------------- table (standard 9 + goodput) ----------------
cols = METRICS[:2] + [GOODPUT] + METRICS[2:]
print(f"\n{'scheduler':20s}" + "".join(f"{t.split(' [')[0][:13]:>14s}" for _, t, _, _ in cols))
for lab, rows, _ in SERIES:
    line = f"{lab.replace(chr(10), ' '):20s}"
    for key, _, scale, fmt in cols:
        m, _ = stat(rows, key, scale)
        line += f"{fmt.format(m):>14s}"
    print(line)
