"""GenieS40HL_FineTune final-metrics figure -- retired-run held-out eval.

Same protocol as the S40HighLoad final figure, for the retired genie warm-start
arm (best.pt@329, 8014 run-eval; world = S40HighLoad queue post-RZF, speed
U(5,40), p U(0.15,0.50) with PERFECT CSI: pmi_mode=genie, p_csi=1.0, beta=1):
(1) sweep SUS+CQI threshold in THIS world (8 seeds) -> strongest T*;
(2) evaluate PPO(best.pt@329) + 8 baselines at T* on 20 held-out seeds;
(3) 8-panel figure (reward/thr/goodput/SINR // depth/comp/failure/JFI),
throughput labels at 1 decimal, PNG saved INSIDE the run folder.
Supplementary (CSV + summary only, NOT in the figure): GenieS40HL_Fresh
best.pt as an interim warm-vs-fresh reference -- Fresh is STILL TRAINING,
its row is not a final verdict.
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
from baselines import all_baselines, SUSCQI
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
RUN = "Run4/GenieS40HL_FineTune"
FRESH = "Run4/GenieS40HL_Fresh"
OUTDIR = "Run4/_analysis"
SEEDS = list(range(10000, 10020))
THRS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)

def make_cfg(thr=None):
    raw = json.load(open(f"{RUN}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    cfg = Config(**raw)
    if thr is not None:
        cfg.sus_ortho_threshold = thr
    return cfg

cfg0 = make_cfg()
assert cfg0.pmi_mode == "genie" and cfg0.p_csi == 1.0, "not the genie world"
assert cfg0.la_beta == 1.0 and not cfg0.la_beta_by_depth, "genie must run beta=1"

# --- stage 1: SUS+CQI threshold sweep (8 seeds) ---
print("=== SUS+CQI threshold sweep (GenieS40HL world, 8 seeds) ===", flush=True)
sweep_seeds = list(range(10000, 10008))
best_thr, best_val = None, -1e9
for thr in THRS:
    cfg = make_cfg(thr)
    env = SchedulerEnv(cfg)
    sch = SUSCQI()
    rr = []
    for s in sweep_seeds:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        rr.append(env_episode_metrics(env, cfg)["reward"])
    m = float(np.mean(rr))
    print(f"  thr {thr}: SUS+CQI {m:.1f}", flush=True)
    if m > best_val:
        best_val, best_thr = m, thr
print(f"strongest threshold: {best_thr}\n", flush=True)

# --- stage 2: full 20-seed eval at best_thr ---
cfg = make_cfg(best_thr)
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
ac_f = ActorCritic(cfg)
ck_f = torch.load(f"{FRESH}/ckpt/best.pt", map_location="cpu")
ac_f.load_state_dict(ck_f["model"], strict=True)
ac_f.eval()
print(f"PPO-FT best.pt update {ck['update']} run-eval {ck.get('eval_reward', float('nan')):.1f}; "
      f"Fresh(interim) best.pt update {ck_f['update']} run-eval {ck_f.get('eval_reward', float('nan')):.1f}; "
      f"SUS thr {best_thr}", flush=True)

SHORT = {"SUS+Deadline-PF": "SUS+DPF", "SUS+Random": "SUS+Rnd",
         "SU+Deadline-PF": "SU+DPF", "SU+Random": "SU+Rnd"}
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
scheds = [("PPO", PPOScheduler(ac, deterministic=True)),
          ("Fresh(interim)", PPOScheduler(ac_f, deterministic=True))] + \
         [(SHORT.get(s.name, s.name), s) for s in all_baselines(cfg)
          if SHORT.get(s.name, s.name) in ORDER]
assert sorted(n for n, _ in scheds) == sorted(ORDER + ["Fresh(interim)"])

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]
data, ctx = {}, {}
for seed in SEEDS:
    for name, sch in scheds:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        data[(seed, name)] = {k: float(m[k]) for k in KEYS}
    na = int(env.traffic.n_active)
    ctx[seed] = (na, np.sort(env.ue_speeds_kmh[:na]))
    marg = data[(seed, 'PPO')]['reward'] - max(data[(seed, n)]['reward'] for n in ORDER[1:])
    print(f"seed {seed}: PPO-FT {data[(seed,'PPO')]['reward']:.0f} margin {marg:+.0f}  "
          f"Fresh {data[(seed,'Fresh(interim)')]['reward']:.0f}", flush=True)

with open(f"{OUTDIR}/genies40hl_finetune_final20.csv", "w", newline="") as f:
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

fig.suptitle(f"GenieS40HL_FineTune — PPO vs baselines  (seed {show} best-margin;  "
             f"SUS thr {best_thr} in-world swept;  best.pt@{ck['update']};  "
             f"PERFECT CSI (genie, p_csi=1, beta=1), post-RZF, speed U(5,40), p U(0.15,0.50);  "
             f"warm-start from imperfect-CSI HighLoad best@639)", fontsize=9)
fig.text(0.5, 0.945,
         f"K = {cfg.num_ue} UEs ({na} active)  ·  per-UE speeds (km/h, sorted): "
         + " ".join(f"{s:.1f}" for s in spd), ha="center", fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
png = f"{RUN}/GenieS40HLFineTune_metrics_seed{show}.png"
fig.savefig(png, dpi=140)
print("showcase seed", show, "->", png, flush=True)

print("\n20-seed means:")
for n in ORDER + ["Fresh(interim)"]:
    mm = {k: np.mean([data[(s, n)][k] for s in SEEDS]) for k in KEYS}
    print(f"{n:15s} rew {mm['reward']:7.0f}  depth {mm['mu_depth']:.2f}  "
          f"miss {mm['deadline_miss_rate']:.3f}  good {mm['goodput_mbps']:.1f}", flush=True)
ppo = np.mean([data[(s, 'PPO')]['reward'] for s in SEEDS])
bb = max(np.mean([data[(s, n)]['reward'] for s in SEEDS]) for n in ORDER[1:])
diff = [data[(s, 'PPO')]['reward'] - max(data[(s, n)]['reward'] for n in ORDER[1:]) for s in SEEDS]
wins = sum(x > 0 for x in diff)
print(f"\nPPO-FT {ppo:.0f} vs best baseline {bb:.0f} = {(ppo-bb)/bb*100:+.1f}%; "
      f"vs seed-best {wins}/20 wins", flush=True)
fr = np.mean([data[(s, 'Fresh(interim)')]['reward'] for s in SEEDS])
dwf = [data[(s, 'PPO')]['reward'] - data[(s, 'Fresh(interim)')]['reward'] for s in SEEDS]
print(f"warm-vs-fresh (Fresh INTERIM @upd {ck_f['update']}, still training): "
      f"FT {ppo:.0f} vs Fresh {fr:.0f} = {(ppo-fr)/fr*100:+.1f}%; "
      f"paired {np.mean(dwf):+.0f}, FT wins {sum(x > 0 for x in dwf)}/20", flush=True)
