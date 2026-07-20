"""Redraw the three existing 8-panel figures with throughput at 1 decimal
(user directive 2026-07-20). Pure replot from the saved final20 CSVs --
no re-evaluation; env.reset only for the showcase seed's subtitle
(n_active + speeds, deterministic). Outputs overwrite the PNGs in their
run folders (png-in-run-folder rule).
"""
import os, sys, csv, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config
from env import SchedulerEnv

A = "Run4/_analysis"
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
KEYS = ["reward", "throughput_mbps", "mean_sinr_db", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain"]
COLORS = ["#d62728"] + ["#1f77b4"] * 4 + ["#2ca02c"] * 4

def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d[(int(r["seed"]), r["sched"])] = {k: float(r[k]) for k in KEYS}
    return d

def make_env(cfg_json):
    raw = json.load(open(cfg_json))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    return SchedulerEnv(Config(**raw))

def draw(data, show, ctx, title, out_png, num_ue):
    na, spd = ctx
    def bars(ax, t, vals, fmt, ymax=None):
        b = ax.bar(range(len(ORDER)), vals, color=COLORS)
        ax.set_title(t)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7)
        if ymax:
            ax.set_ylim(0, ymax)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=6)
    fig, axes = plt.subplots(2, 4, figsize=(13.66, 6.6))
    g = lambda n, k: data[(show, n)][k]
    bars(axes[0, 0], "Episode reward", [g(n, "reward") for n in ORDER], "{:.0f}")
    bars(axes[0, 1], "Throughput (Mbps)", [g(n, "throughput_mbps") for n in ORDER], "{:.1f}")
    bars(axes[0, 2], "Mean SINR (dB)", [g(n, "mean_sinr_db") for n in ORDER], "{:.1f}")
    bars(axes[0, 3], "Completion rate", [g(n, "completion_rate") for n in ORDER], "{:.2f}", 1.0)
    bars(axes[1, 0], "MU depth (UEs / active RBG)", [g(n, "mu_depth") for n in ORDER], "{:.2f}")
    ax = axes[1, 1]
    miss = [g(n, "deadline_miss_rate") for n in ORDER]
    retx = [g(n, "retx_drop_rate") for n in ORDER]
    ax.bar(range(len(ORDER)), miss, color="tab:orange", label="deadline-miss")
    ax.bar(range(len(ORDER)), retx, bottom=miss, color="tab:purple", label="retx drop")
    for i, (a, b) in enumerate(zip(miss, retx)):
        ax.text(i, a + b, f"{a + b:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_title("Failure modes")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    bars(axes[1, 2], "TOTAL failure",
         [g(n, "deadline_miss_rate") + g(n, "retx_drop_rate") for n in ORDER], "{:.2f}")
    bars(axes[1, 3], "Jain fairness (JFI, active UEs)", [g(n, "jain") for n in ORDER], "{:.2f}")
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.945,
             f"K = {num_ue} UEs ({na} active)  ·  per-UE speeds (km/h, sorted): "
             + " ".join(f"{s:.1f}" for s in spd), ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("->", out_png, flush=True)

genie = make_env("Run4/GenieFineTune/config.json")
genie.reset(episode_idx=10014)
gna = int(genie.traffic.n_active)
gctx = (gna, np.sort(genie.ue_speeds_kmh[:gna]))
draw(load(f"{A}/geniefinetune_final20.csv"), 10014, gctx,
     "GenieFineTune final metrics — PPO vs SUS-MU vs SU baselines   "
     "(showcase seed 10014 = best-margin episode;  SUS threshold 0.5;  "
     "PPO = best.pt@729;  perfect CSI: genie PMI + p_csi=1.0)",
     "Run4/GenieFineTune/GenieFineTune_metrics_seed10014.png", 32)
draw(load(f"{A}/geniefinetune_final20_thr07.csv"), 10014, gctx,
     "GenieFineTune — PPO vs baselines  (seed 10014 best-margin;  "
     "SUS thr 0.7 in-world swept;  best.pt@729;  perfect CSI)",
     "Run4/GenieFineTune/GenieFineTune_metrics_seed10014_thr07.png", 32)

q = make_env("Run4/QueuePostRZF_Ent02/config.json")
q.reset(episode_idx=10017)
qna = int(q.traffic.n_active)
qctx = (qna, np.sort(q.ue_speeds_kmh[:qna]))
draw(load(f"{A}/queue_ent02_final20.csv"), 10017, qctx,
     "QueuePostRZF_Ent02 — PPO vs baselines  (seed 10017 best-margin;  "
     "SUS thr 0.7 in-world swept;  best.pt@769;  "
     "post-RZF + beta_m LA-surrogate world)",
     "Run4/QueuePostRZF_Ent02/QueueEnt02_metrics_seed10017.png", 32)
print("done")
