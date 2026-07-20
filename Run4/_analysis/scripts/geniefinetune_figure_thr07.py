"""GenieFineTune figure v2 -- SUS baselines at their in-world strongest
threshold 0.7 (from genie_sus_threshold_sweep: 0.7=9417 > 0.5=9334).

Re-evaluates ONLY the 4 SUS-family baselines (threshold affects nothing
else) on the same 20 seeds; PPO and SU rows are reused from
geniefinetune_final20.csv. Redraws the 8-panel figure, fair-hard style.
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
from baselines import all_baselines
from train_phase2 import env_episode_metrics

OUTDIR = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis"
SEEDS = list(range(10000, 10020))
THR = 0.7

raw = json.load(open("Run4/GenieFineTune/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
cfg.sus_ortho_threshold = THR
env = SchedulerEnv(cfg)

KEYS = ["reward", "throughput_mbps", "mean_sinr_db", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain"]
SHORT = {"SUS+Deadline-PF": "SUS+DPF", "SUS+Random": "SUS+Rnd",
         "SU+Deadline-PF": "SU+DPF", "SU+Random": "SU+Rnd"}
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]

# reuse PPO + SU rows from the thr-0.5 pass
data, ctx = {}, {}
for r in csv.DictReader(open(f"{OUTDIR}/geniefinetune_final20.csv")):
    if not (r["sched"].startswith("SUS")):
        data[(int(r["seed"]), r["sched"])] = {k: float(r[k]) for k in KEYS}

sus = [(SHORT.get(s.name, s.name), s) for s in all_baselines(cfg)
       if s.name.startswith("SUS")]
assert len(sus) == 4
for seed in SEEDS:
    for name, sch in sus:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        data[(seed, name)] = {k: float(m[k]) for k in KEYS}
    ctx[seed] = (int(env.traffic.n_active), np.sort(env.ue_speeds_kmh[:int(env.traffic.n_active)]))
    print(f"seed {seed} SUS@{THR} done", flush=True)

with open(f"{OUTDIR}/geniefinetune_final20_thr07.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "sched"] + KEYS)
    for (seed, name), m in sorted(data.items()):
        w.writerow([seed, name] + [round(m[k], 4) for k in KEYS])

show = max(SEEDS, key=lambda s: data[(s, "PPO")]["reward"] -
           max(data[(s, n)]["reward"] for n in ORDER[1:]))
na, spd = ctx[show]
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

fig.suptitle(
    f"GenieFineTune final metrics — PPO vs SUS-MU vs SU baselines   "
    f"(showcase seed {show} = best-margin episode;  "
    f"SUS threshold {THR} = in-world swept strongest;  PPO = best.pt@729;  "
    f"perfect CSI: genie PMI + p_csi=1.0)", fontsize=10)
fig.text(0.5, 0.945,
         f"K = {cfg.num_ue} UEs ({na} active)  ·  per-UE constant speeds this "
         f"episode (km/h, sorted): " + " ".join(f"{s:.1f}" for s in spd),
         ha="center", fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
png = f"Run4/GenieFineTune/GenieFineTune_metrics_seed{show}_thr07.png"
fig.savefig(png, dpi=140)
print("showcase seed", show, "->", png, flush=True)

print("\n20-seed means (SUS@0.7):")
for n in ORDER:
    mm = {k: np.mean([data[(s, n)][k] for s in SEEDS]) for k in KEYS}
    print(f"{n:8s} rew {mm['reward']:7.0f}  depth {mm['mu_depth']:.2f}  "
          f"miss {mm['deadline_miss_rate']:.3f}  retx {mm['retx_drop_rate']:.4f}", flush=True)
ppo = np.mean([data[(s, 'PPO')]['reward'] for s in SEEDS])
best = max(np.mean([data[(s, n)]['reward'] for s in SEEDS]) for n in ORDER[1:])
print(f"\nPPO {ppo:.0f} vs best baseline {best:.0f} = {(ppo-best)/best*100:+.1f}%")
