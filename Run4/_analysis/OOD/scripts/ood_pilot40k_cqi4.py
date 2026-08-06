"""Fresh-seed pilot for the paper OOD grid (user redesign 2026-08-06).

Fixes two audit findings at once:
  (1) pilot/test separation -- SUS threshold selection moves to FRESH seeds
      40000-40007 (never used anywhere in the repo), so the n=40 eval set
      30000-30039 becomes a fully independent test set;
  (2) baseline scope -- the pilot runs the FULL 12-baseline tuned grid per
      world, verifying the champion's identity before we write "strongest
      heuristic" (D26 especially: deadline-aware EDF/DPF could plausibly
      gain there).

Protocol per world:
  a. sweep SUS+CQI over thresholds {0.60..0.80} on seeds 40000-40007;
     T* = argmax (this is the only tuned knob, and it belongs to the
     baseline side -- PPO has no eval-time parameter);
  b. at T*, run the remaining 11 grid baselines on the same pilot seeds;
  c. report champion identity + gap to runner-up; FLAG if the champion is
     not in {SUS+CQI@T*, SU+CQI} or if any non-CQI baseline comes within
     10% of the champion (then a dedicated threshold sweep for that family
     is required before the n=40 eval).
Baselines only -- the frozen policy is not involved in any pilot decision.
argv[1]=comma world subset, argv[2]=tag (3-GPU split as in the main grid).
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, csv
from config import Config
from env import SchedulerEnv
from baselines import all_baselines
from train_phase2 import env_episode_metrics

os.chdir("/home/MYH/ML_DRL_Scheduler")
SEL = sys.argv[1].split(",") if len(sys.argv) > 1 else None
TAG = sys.argv[2] if len(sys.argv) > 2 else "all"
OUT = f"Run4/_analysis/OOD/results/ood_pilot40k_{TAG}.csv"
PILOT = list(range(40000, 40008))
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]
BASE = "Run4/QueuePostRZF_S40HL_CQI4"

OLD_TSTAR = {"P055": 0.70, "P010": 0.80, "V60max": 0.70,
             "CSI02": 0.75, "D26": 0.75, "STORM2": 0.75}

CONFIGS = [
    ("P055",   dict(p_arrival_min=0.55, p_arrival_max=0.55)),
    ("P010",   dict(p_arrival_min=0.10, p_arrival_max=0.10)),
    ("V60max", dict(ue_speed_max=60.0)),
    ("CSI02",  dict(p_csi=0.2)),
    ("D26",    dict(deadline_min=2, deadline_max=6)),
    ("STORM2", dict(p_arrival_min=0.55, p_arrival_max=0.55,
                    ue_speed_max=60.0, p_csi=0.2)),
]
if SEL:
    CONFIGS = [(n, o) for n, o in CONFIGS if n in SEL]
    assert len(CONFIGS) == len(SEL)
print(f"pilot40k worker TAG={TAG}, worlds: {[n for n, _ in CONFIGS]}", flush=True)

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
w.writerow(["world", "sched", "seed", "reward", "goodput_mbps",
            "deadline_miss_rate", "mu_depth"])

for cname, overrides in CONFIGS:
    t0 = time.time()
    print(f"\n=== [{cname}] pilot seeds 40000-40007 ===", flush=True)
    cfg = make_cfg(overrides)
    env = SchedulerEnv(cfg)

    # (a) SUS+CQI threshold sweep on fresh seeds
    sus_by_thr = {t: [] for t in SUS_THRS}
    for seed in PILOT:
        for thr in SUS_THRS:
            cfg.sus_ortho_threshold = thr
            sch = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
            m = episode(env, cfg, sch, seed)
            sus_by_thr[thr].append(m["reward"])
            w.writerow([cname, f"SUS+CQI@{thr:.2f}", seed, round(m["reward"], 1),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        f.flush()
    tstar = max(SUS_THRS, key=lambda t: np.mean(sus_by_thr[t]))
    print("  sweep: " + "  ".join(f"{t:.2f}:{np.mean(sus_by_thr[t]):.0f}" for t in SUS_THRS)
          + f"  -> T*={tstar:.2f} (구 30k-pilot T*={OLD_TSTAR[cname]:.2f}"
          + (", 동일)" if abs(tstar - OLD_TSTAR[cname]) < 1e-9 else ", 변경!)"), flush=True)

    # (b) full 12-baseline grid at T* on the same fresh seeds
    cfg.sus_ortho_threshold = tstar
    others = [s for s in all_baselines(cfg) if s.name != "SUS+CQI"]
    res = {f"SUS+CQI@{tstar:.2f}": dict(zip(PILOT, sus_by_thr[tstar]))}
    for seed in PILOT:
        for sch in others:
            m = episode(env, cfg, sch, seed)
            res.setdefault(sch.name, {})[seed] = m["reward"]
            w.writerow([cname, sch.name, seed, round(m["reward"], 1),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        f.flush()

    # (c) champion verdict
    means = {n: np.mean(list(v.values())) for n, v in res.items()}
    order = sorted(means, key=lambda n: -means[n])
    champ, runner = order[0], order[1]
    print(f"  champion: {champ} {means[champ]:.0f}  (2위 {runner} {means[runner]:.0f})", flush=True)
    top = means[champ]
    close = [n for n in order[1:] if means[n] > top - abs(top) * 0.10
             and not n.startswith("SUS+CQI") and n != "SU+CQI"]
    if not (champ.startswith("SUS+CQI") or champ == "SU+CQI"):
        print(f"  ⚠️ FLAG: 챔피언이 CQI 계열이 아님 -> n=40 재평가 필요", flush=True)
    if close:
        print(f"  ⚠️ FLAG: 챔피언 10% 이내 비-CQI baseline: {close} -> 전용 스윕 검토", flush=True)
    print(f"  [{cname}] {(time.time() - t0) / 60:.1f} min", flush=True)

print("\nsaved ->", OUT)
f.close()
