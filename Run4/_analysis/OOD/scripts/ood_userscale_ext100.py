"""User-count zero-shot EXTENSION to n=100 (seeds 30040-30099; 2026-08-10).

Same worlds/protocol as ood_userscale_cqi4.py; the pilot stage is SKIPPED
because T* was already selected on the 40k pilot seeds (K8@0.80, K48@0.75,
K60@0.80) and must stay fixed for a pure sample-size extension.

Replaces the old ratio-based K-scaling (num_ue=K, n_active~U{K/2..K}, 8
seeds, pilot/test shared). Here the TRAINING WORLD is left untouched and
ONLY the user count changes, fixed per world:

    K = 8    (below the trained 16..32 active range)
    K = 48   (above)
    K = 60   (far above, ~1.9x the trained maximum)

Everything else is the CQI4 training world: cqi_mode=nr4bit, speed
U(5,40), p_arrival U(0.15,0.50) per episode, p_csi 0.6, deadline U[3,12],
beta_m frozen at its K=32 calibration (zero-shot premise).
NOTE (paper): per-UE arrival rate is unchanged, so population and offered
load move together -- nominal rho ~0.4 (K8), ~2.4 (K48), ~3.0 (K60).
This is the "more users, same per-user demand" scenario, not a pure
population axis. Also, a given seed generates a DIFFERENT topology at each
K, so cross-K rows are not paired (within a K, PPO vs baseline is).

Protocol (identical to the paper OOD grid):
  * threshold selection on FRESH pilot seeds 40000-40007 (sweep 0.60-0.80)
  * evaluation on test seeds 30000-30039 (n=40), independent of the pilot
  * display baseline set: {SUS, SU} x {CQI, PF, Random}, PPO frozen
  * paired Student-t 95% CI verdict + n_active-free (population is fixed)
argv[1]=comma K subset, argv[2]=tag.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch, csv
torch.set_num_threads(8)
from scipy import stats as st
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
SEL = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None
S0, S1 = int(sys.argv[2]), int(sys.argv[3])
TAG = sys.argv[4] if len(sys.argv) > 4 else "all"
OUT = f"Run4/_analysis/OOD/results/ood_userscale_ext100_{TAG}.csv"
PILOT = list(range(40000, 40008))
SEEDS = list(range(S0, S1))
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
CK = torch.load(f"{BASE}/ckpt/best.pt", map_location="cpu")
KS = SEL if SEL else [8, 48, 60]
DISPLAY = ["SUS+PF", "SUS+Random", "SU+CQI", "SU+PF", "SU+Random"]  # + SUS+CQI@T*

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"userscale worker TAG={TAG}, K={KS}; device {DEV}; policy best.pt@{CK['update']}", flush=True)

def make_cfg(K, thr=None):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw["num_ue"] = K
    raw["n_active_min"] = K          # fixed population: every UE active
    raw["n_active_max"] = K
    if thr is not None:
        raw["sus_ortho_threshold"] = thr
    cfg = Config(**raw)
    assert cfg.cqi_mode == "nr4bit"
    return cfg

def episode(env, cfg, sched, seed):
    env.reset(episode_idx=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(sched.schedule(env))
    return env_episode_metrics(env, cfg)

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["K", "phase", "sched", "seed", "reward", "goodput_mbps",
            "deadline_miss_rate", "mu_depth"])

def log(K, phase, label, seed, m):
    w.writerow([K, phase, label, seed, round(m["reward"], 1),
                round(m.get("goodput_mbps", float("nan")), 2),
                round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])

summary = []
for K in KS:
    t0 = time.time()
    print(f"\n=== [K={K}] fixed population, training world otherwise ===", flush=True)

    TSTAR = {8: 0.80, 48: 0.75, 60: 0.80}
    tstar = TSTAR[K]
    print(f"  T*={tstar:.2f} (40k pilot, fixed)", flush=True)

    # --- stage 2: n=40 test with the 6 display baselines + PPO ---
    cfg = make_cfg(K, tstar)
    env = SchedulerEnv(cfg)
    ac = ActorCritic(cfg)
    ac.load_state_dict(CK["model"], strict=True)   # K-independent parameters
    ac.eval()
    ppo = PPOScheduler(ac.to(DEV), deterministic=True)
    bl = [s for s in all_baselines(cfg) if s.name in DISPLAY or s.name == "SUS+CQI"]
    assert len(bl) == 6, [s.name for s in bl]
    res = {}
    for i, seed in enumerate(SEEDS):
        for sch in bl:
            nm = f"SUS+CQI@{tstar:.2f}" if sch.name == "SUS+CQI" else sch.name
            m = episode(env, cfg, sch, seed)
            res.setdefault(nm, {})[seed] = m["reward"]
            log(K, "test", nm, seed, m)
        m = episode(env, cfg, ppo, seed)
        res.setdefault("PPO", {})[seed] = m["reward"]
        res.setdefault("_ppo_depth", {})[seed] = m["mu_depth"]
        log(K, "test", "PPO", seed, m)
        if (i + 1) % 10 == 0:
            f.flush()
            print(f"  ..{i+1}/40 ({(time.time()-t0)/60:.1f} min)", flush=True)
    f.flush()

    means = {n: np.mean(list(v.values())) for n, v in res.items() if not n.startswith("_")}
    champ = max((n for n in means if n != "PPO"), key=lambda n: means[n])
    d = np.array([res["PPO"][s] - res[champ][s] for s in SEEDS])
    ci = st.t.ppf(0.975, 39) * d.std(ddof=1) / np.sqrt(40)
    depth = np.mean(list(res["_ppo_depth"].values()))
    print("  " + "  ".join(f"{n}:{means[n]:.0f}" for n in sorted(means, key=lambda n: -means[n])), flush=True)
    print(f"  PPO {means['PPO']:.0f} vs {champ} {means[champ]:.0f}  paired {d.mean():+.0f}±{ci:.0f} "
          f"[{'CI>0' if d.mean()-ci>0 else 'tie'}]  승 {int((d>0).sum())}/40  depth {depth:.2f}", flush=True)
    summary.append((K, tstar, means, champ, d.mean(), ci, int((d > 0).sum()), depth))
    print(f"  [K={K}] {(time.time()-t0)/60:.1f} min", flush=True)

print("\n=== USER-SCALE 요약 (CQI4 best@409 동결, 학습세계+인원고정, n=40) ===")
for K, tstar, means, champ, dm, ci, wins, depth in summary:
    print(f"  K={K:2d} (T*={tstar:.2f}): PPO {means['PPO']:6.0f} vs {champ:14s} {means[champ]:6.0f}  "
          f"paired {dm:+6.0f}±{ci:.0f} [{'CI>0' if dm-ci>0 else 'tie'}]  승 {wins}/40  depth {depth:.2f}")
print("saved ->", OUT)
f.close()
