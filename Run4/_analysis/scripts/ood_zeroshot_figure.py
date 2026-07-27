"""3-panel summary figure for the OOD zero-shot probe (2026-07-27).

Panels: (1) goodput PPO vs best-baseline per config with paired-win
annotation, (2) deadline-miss rate, (3) PPO MU depth (adaptation view).
PNG goes to the policy's run folder per the figure-location rule.
"""
import csv
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/ood_zeroshot.csv"
OUT = ("/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HighLoad/"
       "OOD_zeroshot_probe.png")
ORDER = ["P055", "P065", "V50", "V60", "CSI04", "STORM"]
BEST = {"P055": "SUS+CQI@0.70", "P065": "SUS+CQI@0.70", "V50": "SUS+CQI@0.70",
        "V60": "SUS+CQI@0.70", "CSI04": "SUS+CQI@0.70", "STORM": "SUS+CQI@0.75"}
SUB = {"P055": "p=0.55", "P065": "p=0.65", "V50": "50 km/h", "V60": "60 km/h",
       "CSI04": "p_csi=0.4", "STORM": "storm"}

by = defaultdict(lambda: defaultdict(dict))   # config -> seed -> sched -> row
for r in csv.DictReader(open(SRC)):
    by[r["config"]][int(r["seed"])][r["sched"]] = r

def agg(cfg, sched, key):
    return np.mean([float(by[cfg][s][sched][key]) for s in by[cfg]])

def wins(cfg):
    return sum(float(by[cfg][s]["PPO-HL"]["goodput_mbps"]) >
               float(by[cfg][s][BEST[cfg]]["goodput_mbps"]) for s in by[cfg])

x = np.arange(len(ORDER))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("OOD zero-shot probe — frozen HighLoad best@639, seeds 30000-30007"
             " (8, paired)  ·  baseline = per-world swept best (SUS+CQI)",
             fontsize=12)

ax = axes[0]
pg = [agg(c, "PPO-HL", "goodput_mbps") for c in ORDER]
sg = [agg(c, BEST[c], "goodput_mbps") for c in ORDER]
ax.bar(x - 0.19, pg, 0.38, color="tab:red", label="PPO (frozen)")
ax.bar(x + 0.19, sg, 0.38, color="tab:blue", label="best baseline")
for i, c in enumerate(ORDER):
    ax.text(x[i] - 0.19, pg[i], f"{pg[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(x[i] + 0.19, sg[i], f"{sg[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(x[i], 4, f"{(pg[i]-sg[i])/sg[i]*100:+.1f}%\n{wins(c)}/8",
            ha="center", fontsize=8, fontweight="bold", color="white")
ax.set_title("Goodput (Mbps)")
ax.legend(fontsize=8)

ax = axes[1]
pm = [agg(c, "PPO-HL", "deadline_miss_rate") for c in ORDER]
sm = [agg(c, BEST[c], "deadline_miss_rate") for c in ORDER]
ax.bar(x - 0.19, pm, 0.38, color="tab:red")
ax.bar(x + 0.19, sm, 0.38, color="tab:blue")
for i in range(len(ORDER)):
    ax.text(x[i] - 0.19, pm[i], f"{pm[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(x[i] + 0.19, sm[i], f"{sm[i]:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_title("Deadline-miss rate (lower is better)")

ax = axes[2]
pd_ = [agg(c, "PPO-HL", "mu_depth") for c in ORDER]
sd = [agg(c, BEST[c], "mu_depth") for c in ORDER]
ax.bar(x - 0.19, pd_, 0.38, color="tab:red")
ax.bar(x + 0.19, sd, 0.38, color="tab:blue")
for i in range(len(ORDER)):
    ax.text(x[i] - 0.19, pd_[i], f"{pd_[i]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(x[i] + 0.19, sd[i], f"{sd[i]:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 4.6)
ax.set_title("MU depth (PPO adapts, SUS pins 4)")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{SUB[c]}" for c in ORDER], fontsize=9)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print("saved ->", OUT)
