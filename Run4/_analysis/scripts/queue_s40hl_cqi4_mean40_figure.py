"""8-panel figure, 40-seed MEAN edition (user request 2026-08-07).

Same layout/colors/order as the seed-10017 showcase figure
(queue_s40hl_cqi4_final_figure.py), but every bar is the mean over the
40 held-out seeds 10000-10039 from queue_s40hl_cqi4_final40.csv
(pilot-separated threshold T*=0.75). No episodes re-run.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "Run4/QueuePostRZF_S40HL_CQI4"
SRC = "Run4/_analysis/queue_s40hl_cqi4_final40.csv"
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]

rows = list(csv.DictReader(open(SRC)))
seeds = sorted({int(r["seed"]) for r in rows})
assert len(seeds) == 40
mean = {n: {k: np.mean([float(r[k]) for r in rows if r["sched"] == n])
            for k in KEYS} for n in ORDER}
COLORS = ["#d62728"] + ["#1f77b4"] * 4 + ["#2ca02c"] * 4

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

fig.suptitle("QueuePostRZF_S40HL_CQI4 — PPO vs baselines  (MEANS over 40 held-out seeds 10000–10039;  "
             "SUS thr 0.75 pilot-separated (seeds 40000–40007);  best.pt@409;  "
             "4-bit NR CQI, post-RZF + beta_m(CQI4), speed U(5,40), p U(0.15,0.50))", fontsize=9)
fig.text(0.5, 0.945,
         "PPO vs SUS+CQI: +15.2% reward (paired +625±82, t=15.33, 40/40 wins)",
         ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
png = f"{RUN}/QueueS40HLCQI4_metrics_mean40.png"
fig.savefig(png, dpi=140)
print("saved ->", png)
