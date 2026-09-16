"""8-panel figure, NoUserHead-ablation edition (user request 2026-09-16).

Same layout / bar style / label placement as queue_s40hl_cqi4_mean40_figure.py,
but every bar is the MEAN over the 100 blind-holdout seeds 21000-21099, and
the schedulers are the NoUserHead ablation family plus the three strongest
rule baselines:

  red    = PPO base (alpha 1.0, best.pt@409)
  blue   = learned ablations: alpha in {1.2, 1.4, 0.75, 0.5, 0}, FullRank
  green  = rule baselines: SUS+CQI-Feasible, SUS-RPS, SUS+CQI

Sources: Run4/_analysis/seedreplicate_final100_h21_*.csv (policies) and
Run4/_analysis/holdout21_baselines_w*.csv (baselines). No episodes re-run.
"""
import csv
import glob
import math
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "Run4/QueuePostRZF_S40HL_CQI4"
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]

# ---- load per-seed rows -------------------------------------------------
pol = defaultdict(dict)
for f in glob.glob("Run4/_analysis/seedreplicate_final100_h21_*.csv"):
    tag = f.split("h21_")[1][:-4]
    for r in csv.DictReader(open(f)):
        if r["sched"] == "PPO":
            pol[tag][int(r["seed"])] = r
base = {}
for c in "abc":
    base.update(pol.get(f"2024_{c}", {}))
bl = defaultdict(dict)
for f in glob.glob("Run4/_analysis/holdout21_baselines_w*.csv"):
    for r in csv.DictReader(open(f)):
        bl[r["sched"]][int(r["seed"])] = r

ROWS = {
    "PPO α=1.0":     base,
    "α=1.2":         pol["NUS120"],
    "α=1.4":         pol["NUS140"],
    "α=0.75":        pol["NUS075"],
    "α=0.5":         pol["NUS050"],
    "α=0 (no head)": pol["NUS0"],
    "FullRank":      pol["FullRank"],
    "SUS+CQI-Feas":  bl["SUS+CQI-Feasible"],
    "SUS-RPS":       bl["SUS-RPS"],
    "SUS+CQI":       bl["SUS+CQI"],
}
ORDER = list(ROWS)
seeds = sorted(base)
assert len(seeds) == 100
for n in ORDER:
    assert len(ROWS[n]) == 100, n
mean = {n: {k: np.mean([float(ROWS[n][s][k]) for s in seeds]) for k in KEYS}
        for n in ORDER}
COLORS = ["#d62728"] + ["#1f77b4"] * 6 + ["#2ca02c"] * 3

def bars(ax, title, vals, fmt, ymax=None):
    b = ax.bar(range(len(ORDER)), vals, color=COLORS)
    ax.set_title(title)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7)
    if ymax:
        ax.set_ylim(0, ymax)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v),
                ha="center", va="bottom", fontsize=6)

g = lambda n, k: mean[n][k]
fig, axes = plt.subplots(2, 4, figsize=(13.66, 6.6))
bars(axes[0, 0], "Episode reward", [g(n, "reward") for n in ORDER], "{:.0f}")
bars(axes[0, 1], "Throughput (Mbps)", [g(n, "throughput_mbps") for n in ORDER], "{:.1f}")
bars(axes[0, 2], "Goodput (Mbps)", [g(n, "goodput_mbps") for n in ORDER], "{:.1f}")
bars(axes[0, 3], "Mean SINR (dB)", [g(n, "mean_sinr_db") for n in ORDER], "{:.1f}")
bars(axes[1, 0], "MU depth (UEs / active RBG)", [g(n, "mu_depth") for n in ORDER], "{:.2f}")
bars(axes[1, 1], "Completion rate", [g(n, "completion_rate") for n in ORDER], "{:.2f}", 1.0)
ax = axes[1, 2]
miss = [g(n, "deadline_miss_rate") for n in ORDER]
retx = [g(n, "retx_drop_rate") for n in ORDER]
ax.bar(range(len(ORDER)), miss, color="tab:orange", label="deadline-miss")
ax.bar(range(len(ORDER)), retx, bottom=miss, color="tab:purple", label="retx drop")
for i, (a, b) in enumerate(zip(miss, retx)):
    ax.text(i, a + b, f"{a + b:.2f}", ha="center", va="bottom", fontsize=6)
ax.set_title("Failure modes (total = stack height)")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7)
ax.legend(fontsize=7)
bars(axes[1, 3], "Jain fairness (JFI, active UEs)", [g(n, "jain") for n in ORDER], "{:.2f}")

# paired verdicts vs base for the subtitle
bv = np.array([float(base[s]["reward"]) for s in seeds])
def paired(n):
    d = np.array([float(ROWS[n][s]["reward"]) for s in seeds]) - bv
    h = stats.t.ppf(0.975, 99) * d.std(ddof=1) / 10
    return d.mean(), h, int((d < 0).sum())
parts = []
for n in ["α=1.2", "α=0 (no head)", "FullRank", "SUS+CQI-Feas"]:
    m, h, losses = paired(n)
    parts.append(f"{n} {m:+.0f}±{h:.0f} ({losses}/100 below base)")

fig.suptitle("QueuePostRZF_S40HL_CQI4 — NoUserHead ablation vs base vs rule baselines  "
             "(MEANS over 100 blind-holdout seeds 21000–21099;  best.pt per run;  SUS thr 0.75)",
             fontsize=9)
fig.text(0.5, 0.945,
         "paired Δ reward vs PPO α=1.0 (4787):  " + ";  ".join(parts),
         ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
png = f"{RUN}/NoUserAblation_metrics_mean100.png"
fig.savefig(png, dpi=140)
print("saved ->", png)
