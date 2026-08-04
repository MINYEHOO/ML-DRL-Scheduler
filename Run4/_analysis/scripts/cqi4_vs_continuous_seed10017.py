"""Seed-10017 grouped comparison: continuous-CQI vs 4-bit-CQI S40HighLoad world.

Reads the two banked final20 CSVs (no episodes re-run). Valid absolute
comparison at the same seed: the two worlds share channel, topology, traffic
draws and the noise calibration (sigma^2 depends on the DIRECTION quantization
only, which cqi_mode does not touch) -- the sole difference is the CQI report
granularity. SUS thr 0.7 was the in-world swept optimum in BOTH worlds.
PNG -> Run4/QueuePostRZF_S40HL_CQI4/CQI4_vs_continuous_seed10017.png
"""
import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 10017
ORDER = ["PPO","SUS+CQI","SUS+DPF","SUS+PF","SUS+Rnd","SU+CQI","SU+DPF","SU+PF","SU+Rnd"]
def seedrows(path):
    out={}
    for r in csv.DictReader(open(path)):
        if int(r['seed'])==SEED:
            n=r.get('sched')
            if n in ORDER: out[n]=r
    return out
cont=seedrows('Run4/_analysis/queue_s40highload_final20.csv')
cqi4=seedrows('Run4/_analysis/queue_s40hl_cqi4_final20.csv')

x=np.arange(len(ORDER)); w=0.38
C_CONT, C_CQI4 = "#1f77b4", "#ff7f0e"
def gv(d,n,k): return float(d[n][k])

def panel(ax, title, key, fmt, ymax=None, derived=None):
    a=[derived(cont,n) if derived else gv(cont,n,key) for n in ORDER]
    b=[derived(cqi4,n) if derived else gv(cqi4,n,key) for n in ORDER]
    ax.bar(x-w/2, a, w, color=C_CONT, label="continuous CQI")
    ax.bar(x+w/2, b, w, color=C_CQI4, label="4-bit CQI")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=6.5)
    if ymax: ax.set_ylim(0, ymax)
    for xi, v in zip(x-w/2, a):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=5)
    for xi, v in zip(x+w/2, b):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=5)

fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.8))
panel(axes[0,0], "Episode reward", "reward", "{:.0f}")
panel(axes[0,1], "Throughput (Mbps)", "throughput_mbps", "{:.1f}")
panel(axes[0,2], "Goodput (Mbps)", "goodput_mbps", "{:.1f}")
panel(axes[0,3], "Mean SINR (dB)", "mean_sinr_db", "{:.1f}")
panel(axes[1,0], "MU depth (UEs / active RBG)", "mu_depth", "{:.2f}")
panel(axes[1,1], "Completion rate", "completion_rate", "{:.2f}", ymax=1.0)
panel(axes[1,2], "Total failure (miss + retx drop)", None, "{:.2f}",
      derived=lambda d,n: float(d[n]['deadline_miss_rate'])+float(d[n]['retx_drop_rate']))
panel(axes[1,3], "Jain fairness (JFI, active UEs)", "jain", "{:.2f}")
axes[0,0].legend(fontsize=7, loc="upper right")

fig.suptitle("S40HighLoad world, seed 10017 (17 active UEs, p_a = 0.454): continuous vs 4-bit CQI report\n"
             "same channel, topology and noise calibration per seed — only the CQI report granularity differs;  "
             "SUS thr 0.7 (in-world swept in BOTH worlds);  PPO = each world's own best.pt (@639 / @409)",
             fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
png="Run4/QueuePostRZF_S40HL_CQI4/CQI4_vs_continuous_seed10017.png"
fig.savefig(png, dpi=140)
print("saved ->", png)
