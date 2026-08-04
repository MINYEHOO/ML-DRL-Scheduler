"""GenieS40HL_Fresh final-metrics figure -- retired-run held-out verdict.

Fresh's best.pt@869 never changed between the FineTune final eval
(2026-08-03, where it was evaluated as the 'Fresh(interim)' supplementary
row) and Fresh's retirement (2026-08-04, best still @869). The interim
rows in genies40hl_finetune_final20.csv are therefore Fresh's FINAL
20-seed numbers: same checkpoint, same held-out seeds 10000-10019, same
in-world swept SUS threshold 0.6. This script re-runs NO episodes -- it
renders Fresh's own 8-panel figure and summary from that CSV, per the
run-folder figure convention. One env.reset() is used only to recover the
showcase seed's (n_active, speeds) context line.
"""
import os, sys, csv, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config
from env import SchedulerEnv

os.chdir("/home/MYH/ML_DRL_Scheduler")
RUN = "Run4/GenieS40HL_Fresh"
SRC = "Run4/_analysis/genies40hl_finetune_final20.csv"
SEEDS = list(range(10000, 10020))
SUS_THR = 0.6  # in-world swept (genies40hl_finetune_final_figure.out)

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
REMAP = {"Fresh(interim)": "PPO"}   # Fresh policy is THIS run's PPO
DROP = {"PPO"}                      # FineTune's rows are not part of this figure

data = {}
for r in csv.DictReader(open(SRC)):
    name = r["sched"]
    if name in DROP and name not in REMAP:
        continue
    name = REMAP.get(name, name)
    if name not in ORDER:
        continue
    data[(int(r["seed"]), name)] = {k: float(r[k]) for k in KEYS}
assert all((s, n) in data for s in SEEDS for n in ORDER), "missing rows in source CSV"

ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
print(f"PPO(Fresh) best.pt update {ck['update']} run-eval {ck.get('eval_reward', float('nan')):.1f}; "
      f"SUS thr {SUS_THR} (in-world swept, from the FineTune final); no episodes re-run", flush=True)

show = max(SEEDS, key=lambda s: data[(s, "PPO")]["reward"] -
           max(data[(s, n)]["reward"] for n in ORDER[1:]))

# context line (n_active, speeds) for the showcase seed -- one reset, no scheduling
raw = json.load(open(f"{RUN}/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.pmi_mode == "genie" and cfg.p_csi == 1.0
env = SchedulerEnv(cfg)
env.reset(episode_idx=show)
na = int(env.traffic.n_active)
spd = np.sort(env.ue_speeds_kmh[:na])

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

fig, axes = plt.subplots(2, 4, figsize=(13.66, 6.6))
g = lambda n, k: data[(show, n)][k]
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

fig.suptitle(f"GenieS40HL_Fresh — PPO vs baselines  (seed {show} best-margin;  "
             f"SUS thr {SUS_THR} in-world swept;  best.pt@{ck['update']};  "
             f"PERFECT CSI (genie, p_csi=1, beta=1), post-RZF, speed U(5,40), p U(0.15,0.50);  "
             f"trained from scratch — A/B control of the warm-started FineTune)", fontsize=9)
fig.text(0.5, 0.945,
         f"K = {cfg.num_ue} UEs ({na} active)  ·  per-UE speeds (km/h, sorted): "
         + " ".join(f"{s:.1f}" for s in spd), ha="center", fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
png = f"{RUN}/GenieS40HLFresh_metrics_seed{show}.png"
fig.savefig(png, dpi=140)
print("showcase seed", show, "->", png, flush=True)

print("\n20-seed means:")
for n in ORDER:
    mm = {k: np.mean([data[(s, n)][k] for s in SEEDS]) for k in KEYS}
    print(f"{n:8s} rew {mm['reward']:7.0f}  depth {mm['mu_depth']:.2f}  "
          f"miss {mm['deadline_miss_rate']:.3f}  good {mm['goodput_mbps']:.1f}", flush=True)
ppo = np.mean([data[(s, 'PPO')]['reward'] for s in SEEDS])
bb = max(np.mean([data[(s, n)]['reward'] for s in SEEDS]) for n in ORDER[1:])
diff = [data[(s, 'PPO')]['reward'] - max(data[(s, n)]['reward'] for n in ORDER[1:]) for s in SEEDS]
wins = sum(x > 0 for x in diff)
print(f"\nPPO(Fresh) {ppo:.0f} vs best baseline {bb:.0f} = {(ppo-bb)/bb*100:+.1f}%; "
      f"vs seed-best {wins}/20 wins", flush=True)
