"""GenieFineTune figure v3 -- Ent02 panel layout (user directive 2026-07-20).

Panels: reward, throughput, goodput, SINR / MU depth, completion,
failure modes (total = stack height), JFI. TOTAL-failure panel dropped.
SUS family at the in-world swept threshold 0.7; PPO = best.pt@729.
Full 20-seed re-eval because goodput was not stored in the old CSVs
(legacy world: env_episode_metrics hides it, but env.ep["completed_bits"]
accrues in every LA mode -- computed manually here).
"""
import os, sys, csv, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

OUTDIR = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis"
SEEDS = list(range(10000, 10020))
THR = 0.7                                    # in-world swept strongest

raw = json.load(open("Run4/GenieFineTune/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.pmi_mode == "genie" and cfg.p_csi == 1.0
cfg.sus_ortho_threshold = THR
env = SchedulerEnv(cfg)

ac = ActorCritic(cfg)
ck = torch.load("Run4/GenieFineTune/ckpt/best.pt", map_location="cpu")
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
print(f"best.pt update {ck['update']}; SUS thr {THR}", flush=True)

SHORT = {"SUS+Deadline-PF": "SUS+DPF", "SUS+Random": "SUS+Rnd",
         "SU+Deadline-PF": "SU+DPF", "SU+Random": "SU+Rnd"}
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
scheds = [("PPO", PPOScheduler(ac, deterministic=True))] + \
         [(SHORT.get(s.name, s.name), s) for s in all_baselines(cfg)
          if SHORT.get(s.name, s.name) in ORDER]
assert sorted(n for n, _ in scheds) == sorted(ORDER)

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]
ep_time = cfg.episode_len_main * cfg.slot_duration
data, ctx = {}, {}
for seed in SEEDS:
    for name, sch in scheds:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        m["goodput_mbps"] = env.ep["completed_bits"] / ep_time / 1e6  # legacy-safe
        data[(seed, name)] = {k: float(m[k]) for k in KEYS}
    na = int(env.traffic.n_active)
    ctx[seed] = (na, np.sort(env.ue_speeds_kmh[:na]))
    margin = data[(seed, 'PPO')]['reward'] - max(
        data[(seed, n)]['reward'] for n in ORDER[1:])
    print(f"seed {seed}: PPO {data[(seed,'PPO')]['reward']:.0f} margin {margin:+.0f}", flush=True)

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

fig.suptitle(f"GenieFineTune — PPO vs baselines  (seed {show} best-margin;  "
             f"SUS thr {THR} in-world swept;  best.pt@{ck['update']};  perfect CSI)",
             fontsize=10)
fig.text(0.5, 0.945,
         f"K = {cfg.num_ue} UEs ({na} active)  ·  per-UE speeds (km/h, sorted): "
         + " ".join(f"{s:.1f}" for s in spd), ha="center", fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
png = f"Run4/GenieFineTune/GenieFineTune_metrics_seed{show}_thr07.png"
fig.savefig(png, dpi=140)
print("showcase seed", show, "->", png, flush=True)

print("\n20-seed means:")
for n in ORDER:
    mm = {k: np.mean([data[(s, n)][k] for s in SEEDS]) for k in KEYS}
    print(f"{n:8s} rew {mm['reward']:7.0f}  thr {mm['throughput_mbps']:5.1f}  "
          f"good {mm['goodput_mbps']:5.1f}  comp {mm['completion_rate']:.3f}  "
          f"miss {mm['deadline_miss_rate']:.3f}  depth {mm['mu_depth']:.2f}", flush=True)
ppo = np.mean([data[(s, 'PPO')]['reward'] for s in SEEDS])
best = max(np.mean([data[(s, n)]['reward'] for s in SEEDS]) for n in ORDER[1:])
print(f"\nPPO {ppo:.0f} vs best baseline {best:.0f} = {(ppo-best)/best*100:+.1f}%")
