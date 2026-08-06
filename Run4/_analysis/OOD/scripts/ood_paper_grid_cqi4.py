"""Paper OOD grid, CQI4-main edition -- PRE-DECLARED n=40 (user-approved
2026-08-06). Frozen QueuePostRZF_S40HL_CQI4 best.pt@409, nr4bit world,
beta_m frozen (zero-shot premise).

Six scenarios, one per axis-direction, single-axis definitions reused
verbatim inside the combined scenario (STORM2 = P055 + V60max + CSI02):
  P055   p_arrival fixed 0.55        (load, just past the 0.50 ceiling)
  P010   p_arrival fixed 0.10        (load, below the 0.15 floor)
  V60max speed ~ U(5,60)             (upper-bound extension, shape kept)
  CSI02  p_csi = 0.2                 (realistic reporting density)
  D26    deadline ~ U[2,6]           (urgency axis; Run3 DeadlineScarcity kin)
  STORM2 P055 + V60max + CSI02 simultaneously (interaction test)

Protocol (pre-registered):
  * SUS+CQI threshold selected on the PILOT seeds 30000-30007 only
    (sweep {0.60..0.80}); then FIXED.
  * Full evaluation on seeds 30000-30039 (n=40, declared in advance, no
    optional stopping): SUS+CQI@T*, SU+CQI (anchor), PPO. Best baseline =
    higher 40-seed mean of the two.
  * Paired verdicts: mean margin, Student-t 95% CI of the paired diff,
    per-seed wins. Load-stratified split by n_active tertiles.
  * Seed loop OUTER, scheduler loop INNER (channel LRU cache = 16 < 40).
  * Final paper numbers will still be re-confirmed on reserved seeds
    20000+ with the paper-generation checkpoint.
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
# argv[1] = comma-separated subset of config names (parallel split across
# GPUs); argv[2] = output tag so parallel workers write disjoint CSVs.
SEL = sys.argv[1].split(",") if len(sys.argv) > 1 else None
TAG = sys.argv[2] if len(sys.argv) > 2 else "all"
OUT = f"Run4/_analysis/ood_paper_grid_cqi4_{TAG}.csv"
PILOT = list(range(30000, 30008))
SEEDS = list(range(30000, 30040))          # n=40, pre-declared
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
CK = torch.load(f"{BASE}/ckpt/best.pt", map_location="cpu")

CONFIGS = [
    ("P055",   dict(p_arrival_min=0.55, p_arrival_max=0.55)),
    ("P010",   dict(p_arrival_min=0.10, p_arrival_max=0.10)),
    ("V60max", dict(ue_speed_max=60.0)),
    ("CSI02",  dict(p_csi=0.2)),
    ("D26",    dict(deadline_min=2, deadline_max=6)),
    ("STORM2", dict(p_arrival_min=0.55, p_arrival_max=0.55,
                    ue_speed_max=60.0, p_csi=0.2)),
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"policy device: {DEV}; policy best.pt@{CK['update']} "
      f"run-eval {CK.get('eval_reward', float('nan')):.1f}; n={len(SEEDS)} pre-declared", flush=True)

def make_cfg(overrides):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw.update(overrides)
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
w.writerow(["config", "sched", "seed", "n_active", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])

def log_row(cname, label, seed, na, m):
    w.writerow([cname, label, seed, na, round(m["reward"], 1),
                round(m["throughput_mbps"], 2),
                round(m.get("goodput_mbps", float("nan")), 2),
                round(m["completion_rate"], 4),
                round(m["deadline_miss_rate"], 4),
                round(m["mu_depth"], 3)])

if SEL:
    CONFIGS = [(n, o) for n, o in CONFIGS if n in SEL]
    assert len(CONFIGS) == len(SEL), f"unknown config in {SEL}"
print(f"worker TAG={TAG}, configs: {[n for n, _ in CONFIGS]}", flush=True)

summary = []
for cname, overrides in CONFIGS:
    t0 = time.time()
    print(f"\n=== [{cname}] {overrides} ===", flush=True)
    cfg = make_cfg(overrides)
    env = SchedulerEnv(cfg)

    # --- stage 1: threshold selection on PILOT seeds only ---
    sus_by_thr = {t: [] for t in SUS_THRS}
    for seed in PILOT:                      # seed-outer keeps channel hot
        for thr in SUS_THRS:
            cfg.sus_ortho_threshold = thr
            sch = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
            m = episode(env, cfg, sch, seed)
            sus_by_thr[thr].append(m["reward"])
    best_thr = max(SUS_THRS, key=lambda t: np.mean(sus_by_thr[t]))
    print("  pilot sweep: " + "  ".join(f"{t:.2f}:{np.mean(sus_by_thr[t]):.0f}"
                                        for t in SUS_THRS)
          + f"  -> T*={best_thr:.2f}", flush=True)

    # --- stage 2: full n=40, seed-outer / scheduler-inner ---
    cfg.sus_ortho_threshold = best_thr
    sus = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
    su = [s for s in all_baselines(cfg) if s.name == "SU+CQI"][0]
    ac = ActorCritic(cfg)
    ac.load_state_dict(CK["model"], strict=True)
    ac.eval()
    ppo = PPOScheduler(ac.to(DEV), deterministic=True)

    res = {"SUS": {}, "SU": {}, "PPO": {}}
    nact = {}
    for i, seed in enumerate(SEEDS):
        for key, sch, label in (("SUS", sus, f"SUS+CQI@{best_thr:.2f}"),
                                ("SU", su, "SU+CQI"),
                                ("PPO", ppo, "PPO")):
            m = episode(env, cfg, sch, seed)
            na = int(env.traffic.n_active)
            nact[seed] = na
            res[key][seed] = m["reward"]
            log_row(cname, label, seed, na, m)
        if (i + 1) % 10 == 0:
            f.flush()
            print(f"  ..{i+1}/{len(SEEDS)} seeds", flush=True)
    f.flush()

    bkey = "SUS" if np.mean(list(res["SUS"].values())) >= np.mean(list(res["SU"].values())) else "SU"
    bname = f"SUS+CQI@{best_thr:.2f}" if bkey == "SUS" else "SU+CQI"
    d = np.array([res["PPO"][s] - res[bkey][s] for s in SEEDS])
    bmean = np.mean(list(res[bkey].values()))
    ci = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    wins = int((d > 0).sum())
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    # load-stratified tertiles by n_active
    order = sorted(SEEDS, key=lambda s: nact[s])
    lo, mid, hi = order[:13], order[13:27], order[27:]
    strat = {k: np.mean([res['PPO'][s] - res[bkey][s] for s in grp])
             for k, grp in (("lo", lo), ("mid", mid), ("hi", hi))}
    print(f"  PPO {np.mean(list(res['PPO'].values())):7.0f} vs {bname} {bmean:7.0f}"
          f"  마진 {(np.mean(list(res['PPO'].values())) - bmean) / abs(bmean) * 100:+.1f}%"
          f"  paired {d.mean():+.0f} CI±{ci:.0f} t={tstat:.2f}  승 {wins}/40", flush=True)
    print(f"  load-strata(n_active 하/중/상): {strat['lo']:+.0f} / {strat['mid']:+.0f} / {strat['hi']:+.0f}", flush=True)
    summary.append((cname, np.mean(list(res['PPO'].values())), bmean, bname,
                    d.mean(), ci, wins, strat))
    print(f"  [{cname}] {(time.time() - t0) / 60:.1f} min", flush=True)

print("\n=== PAPER OOD GRID 요약 (CQI4 best@409 동결, n=40, seeds 30000-30039) ===")
for cname, r, b, bn, dm, ci, wins, strat in summary:
    sig = "CI>0 확정" if dm - ci > 0 else ("tie" if abs(dm) < ci else "CI<0")
    print(f"  {cname:7s}: PPO {r:6.0f} vs {bn:13s} {b:6.0f}  "
          f"paired {dm:+6.0f}±{ci:.0f} [{sig}]  승 {wins}/40  "
          f"strata {strat['lo']:+.0f}/{strat['mid']:+.0f}/{strat['hi']:+.0f}")
print("saved ->", OUT)
f.close()
