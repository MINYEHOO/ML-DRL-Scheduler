"""NoUserHead ablation vs the base run -- eval curves + blind-holdout verdict.

Left : validation eval reward (seeds 10000-10002, 3 episodes, every 10 updates)
       over training, 5-point rolling mean, one line per run. Base run is the
       heavy blue line; every other run keeps a fixed hue and is direct-labeled
       at its end.
Right: the number that actually decides -- blind holdout 21000-21099 (n=100),
       mean with 95% CI, sorted. The left panel is 3 episodes and has inverted
       rankings before; the right panel is the verdict.

PNG goes in the base run's folder (project rule: figures live with the run,
_analysis keeps scripts + CSVs only).
"""
import csv
import glob
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

os.chdir("/home/MYH/ML_DRL_Scheduler")
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
OUT = f"{BASE}/nouser_ablation_eval.png"

# fixed categorical order (reference palette, slots 1-7); base is always slot 1
RUNS = [  # label, run dir, holdout tag
    ("base α=1.0",  BASE,                 "2024"),
    ("α=0 (head off)", BASE + "_NUS0",   "NUS0"),
    ("α=0.5",      BASE + "_NUS050",      "NUS050"),
    ("α=0.75",     BASE + "_NUS075",      "NUS075"),
    ("α=1.2",      BASE + "_NUS120",      "NUS120"),
    ("α=1.4",      BASE + "_NUS140",      "NUS140"),
    ("FullRank (no stop)", BASE + "_FullRank", "FullRank"),
]
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# ---------------- data ----------------
def eval_curve(d):
    rows = [r for r in csv.DictReader(open(f"{d}/csv_logs/eval_metrics.csv"))
            if r["scheduler"] == "PPO"]
    u = np.array([int(r["update"]) for r in rows])
    v = np.array([float(r["reward"]) for r in rows])
    o = np.argsort(u)
    u, v = u[o], v[o]
    k = 5                                             # rolling mean, 5 evals
    vs = np.convolve(v, np.ones(k) / k, mode="valid")
    return u[k - 1:], vs, v

pol = defaultdict(dict)
for f in glob.glob("Run4/_analysis/seedreplicate_final100_h21_*.csv"):
    tag = f.split("h21_")[1][:-4]
    for r in csv.DictReader(open(f)):
        if r["sched"] == "PPO":
            pol[tag][int(r["seed"])] = float(r["reward"])
base_h = {}
for c in "abc":
    base_h.update(pol.get(f"2024_{c}", {}))
pol["2024"] = base_h
seeds = sorted(base_h)
assert len(seeds) == 100

hold = []
base_v = np.array([pol["2024"][s] for s in seeds])
for lab, d, tag in RUNS:
    v = np.array([pol[tag][s] for s in seeds])
    dlt = v - base_v
    ci = stats.t.ppf(0.975, 99) * dlt.std(ddof=1) / 10     # paired CI
    hold.append((lab, v.mean(), dlt.mean(), ci, int((dlt > 0).sum())))

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1.55, 1]})
for a in (ax, bx):
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(True, color=GRID, linewidth=0.8)
    a.set_axisbelow(True)

# left: curves
ends = []
for (lab, d, tag), hue in zip(RUNS, HUES):
    u, vs, _ = eval_curve(d)
    lw = 2.6 if tag == "2024" else 1.6
    ax.plot(u, vs, color=hue, linewidth=lw, solid_capstyle="round", label=lab)
    ends.append((vs[-1], lab, hue, u[-1]))
# direct labels at line ends: the dot sits on the TRUE end; the text is pushed
# apart vertically so labels never collide, with a thin leader back to the dot
ends.sort()
y_prev = -1e9
for y_true, lab, hue, x in ends:
    y_txt = max(y_true, y_prev + 90)
    ax.plot([x], [y_true], marker="o", ms=4.5, color=hue, zorder=5)
    if abs(y_txt - y_true) > 1:
        ax.plot([x, x + 14], [y_true, y_txt], color=hue, linewidth=0.8,
                alpha=0.8, zorder=4)
    ax.annotate(lab, (x + 14, y_txt), xytext=(4, 0), textcoords="offset points",
                color=INK, fontsize=8.5, va="center")
    y_prev = y_txt
ax.set_xlim(0, 1020)
ax.set_xlabel("PPO update")
ax.set_ylabel("validation eval reward (3 episodes, 5-pt rolling mean)")
ax.set_title("Validation during training: base vs NoUserHead weight α", color=INK,
             fontsize=11.5, loc="left", pad=10)
ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2,
          labelcolor=INK)

# right: paired delta vs base, sorted; base itself is the zero line
order = sorted(range(len(hold)), key=lambda i: hold[i][2])
ys = np.arange(len(order))
for yi, i in zip(ys, order):
    lab, m, dm, ci, wins = hold[i]
    hue = HUES[i]
    if i == 0:                                    # base: reference line only
        bx.plot([0], [yi], marker="o", ms=8, color=hue, markeredgecolor=SURF,
                markeredgewidth=1.5)
        bx.annotate(f"{m:.0f}  (reference)", (0, yi), xytext=(0, 9),
                    textcoords="offset points", ha="center", color=INK, fontsize=8.5)
        continue
    bx.plot([dm - ci, dm + ci], [yi, yi], color=hue, linewidth=2.2,
            solid_capstyle="round")
    bx.plot([dm], [yi], marker="o", ms=8, color=hue, markeredgecolor=SURF,
            markeredgewidth=1.5)
    bx.annotate(f"{m:.0f}  ({dm:+.0f}, wins {wins}/100)", (dm, yi),
                xytext=(0, 9), textcoords="offset points", ha="center",
                color=INK, fontsize=8.5)
bx.set_yticks(ys)
bx.set_yticklabels([hold[i][0] for i in order], color=INK)
bx.axvline(0, color=HUES[0], linewidth=1, linestyle=(0, (4, 3)))
bx.set_xlabel("Δ reward vs base\nblind holdout 21000–21099, n=100, paired 95% CI")
bx.set_xlim(-820, 140)

fig.text(0.01, 0.01,
         "Left: the 3-episode validation used to pick best.pt (it has inverted rankings before). "
         "Right is the verdict. α multiplies the no-user logit (fwd + grad); FullRank removes the stop action itself.",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=170, facecolor=SURF)
print("saved", OUT)
for lab, m, dm, ci, wins in sorted(hold, key=lambda h: -h[1]):
    print(f"  {lab:22s} {m:6.0f}  d {dm:+5.0f} ± {ci:.0f}  wins {wins}/100")
