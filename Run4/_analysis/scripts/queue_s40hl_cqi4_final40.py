"""CQI4 main result: held-out evaluation extended to n=40 (user request
2026-08-07), with the OOD-grade pilot/test separation applied.

Changes vs the original final20:
  * seeds 10000-10019 -> 10000-10039 (the full reserved held-out band);
  * the SUS+CQI threshold is now selected on UNUSED pilot seeds
    40000-40007 in this same (in-distribution) world, instead of on
    10000-10007 which were also part of the evaluation set. All 40
    held-out seeds are therefore untouched by any tuning decision.
World = the CQI4 training world exactly (nr4bit, post-RZF + beta_m(CQI4),
speed U(5,40), p_arrival U(0.15,0.50), p_csi 0.6, K=32 n_active U(16,32)).
Policy = QueuePostRZF_S40HL_CQI4 best.pt@409, frozen.
Schedulers: PPO + the 8 grid baselines (the paper's display set is the 6
{SUS,SU}x{CQI,PF,Random}; DPF kept in the CSV for the appendix table).
"""
import os, sys, csv, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
from scipy import stats as st
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines, SUSCQI
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
RUN = "Run4/QueuePostRZF_S40HL_CQI4"
OUT = "Run4/_analysis/queue_s40hl_cqi4_final40.csv"
SEEDS = list(range(10000, 10040))          # full held-out band, n=40
PILOT = list(range(40000, 40008))          # unused seeds, threshold only
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
assert cfg0.cqi_mode == "nr4bit" and cfg0.la_beta_by_depth
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
print(f"device {DEV}; policy best.pt@{ck['update']} run-eval "
      f"{ck.get('eval_reward', float('nan')):.1f}", flush=True)

def episode(env, cfg, sch, seed):
    env.reset(episode_idx=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(sch.schedule(env))
    return env_episode_metrics(env, cfg)

# --- stage 1: threshold selection on UNUSED pilot seeds (40k) ---
print("=== SUS+CQI threshold sweep on pilot seeds 40000-40007 (in-dist world) ===", flush=True)
t0 = time.time()
best_thr, best_val = None, -1e9
for thr in THRS:
    cfg = make_cfg(thr)
    env = SchedulerEnv(cfg)
    rr = [episode(env, cfg, SUSCQI(), s)["reward"] for s in PILOT]
    m = float(np.mean(rr))
    print(f"  thr {thr}: SUS+CQI {m:.1f}", flush=True)
    if m > best_val:
        best_val, best_thr = m, thr
print(f"T* = {best_thr} (구 final20은 10000-10007에서 0.7 선택)  "
      f"[{(time.time()-t0)/60:.1f} min]\n", flush=True)

# --- stage 2: n=40 held-out evaluation ---
cfg = make_cfg(best_thr)
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
SHORT = {"SUS+Deadline-PF": "SUS+DPF", "SUS+Random": "SUS+Rnd",
         "SU+Deadline-PF": "SU+DPF", "SU+Random": "SU+Rnd"}
ORDER = ["PPO", "SUS+CQI", "SUS+DPF", "SUS+PF", "SUS+Rnd",
         "SU+CQI", "SU+DPF", "SU+PF", "SU+Rnd"]
scheds = [("PPO", PPOScheduler(ac.to(DEV), deterministic=True))] + \
         [(SHORT.get(s.name, s.name), s) for s in all_baselines(cfg)
          if SHORT.get(s.name, s.name) in ORDER]
assert sorted(n for n, _ in scheds) == sorted(ORDER)

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]
data = {}
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["seed", "sched"] + KEYS)
for i, seed in enumerate(SEEDS):
    for name, sch in scheds:
        m = episode(env, cfg, sch, seed)
        data[(seed, name)] = {k: float(m[k]) for k in KEYS}
        w.writerow([seed, name] + [round(float(m[k]), 4) for k in KEYS])
    f.flush()
    marg = data[(seed, "PPO")]["reward"] - max(data[(seed, n)]["reward"] for n in ORDER[1:])
    print(f"seed {seed}: PPO {data[(seed,'PPO')]['reward']:7.0f} margin {marg:+7.0f}"
          + (f"   [{i+1}/40, {(time.time()-t0)/60:.0f} min]" if (i + 1) % 10 == 0 else ""), flush=True)
f.close()

print("\n=== n=40 held-out means (SUS thr %.2f, pilot-separated) ===" % best_thr)
for n in ORDER:
    mm = {k: np.mean([data[(s, n)][k] for s in SEEDS]) for k in KEYS}
    print(f"{n:8s} rew {mm['reward']:7.0f}  depth {mm['mu_depth']:.2f}  "
          f"miss {mm['deadline_miss_rate']:.3f}  good {mm['goodput_mbps']:.1f}  "
          f"jain {mm['jain']:.3f}", flush=True)
ppo = np.array([data[(s, "PPO")]["reward"] for s in SEEDS])
champ = max(ORDER[1:], key=lambda n: np.mean([data[(s, n)]["reward"] for s in SEEDS]))
bb = np.array([data[(s, champ)]["reward"] for s in SEEDS])
d = ppo - bb
ci = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
seedbest = sum(data[(s, "PPO")]["reward"] > max(data[(s, n)]["reward"] for n in ORDER[1:])
               for s in SEEDS)
print(f"\nPPO {ppo.mean():.0f} vs {champ} {bb.mean():.0f} = {(ppo.mean()-bb.mean())/bb.mean()*100:+.1f}%"
      f"  paired {d.mean():+.0f}±{ci:.0f} [{'CI>0' if d.mean()-ci>0 else 'tie'}] t={tstat:.2f}"
      f"  vs champion {int((d>0).sum())}/40, vs seed-best {seedbest}/40", flush=True)
old = [s for s in SEEDS if s < 10020]
new = [s for s in SEEDS if s >= 10020]
for lbl, sub in (("10000-10019 (구 final20 구간)", old), ("10020-10039 (신규)", new)):
    dd = np.array([data[(s, "PPO")]["reward"] - data[(s, champ)]["reward"] for s in sub])
    print(f"  {lbl}: paired {dd.mean():+.0f}, 승 {int((dd>0).sum())}/{len(sub)}", flush=True)
print("saved ->", OUT)
