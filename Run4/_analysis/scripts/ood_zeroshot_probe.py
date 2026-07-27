"""OOD zero-shot probe: frozen HighLoad policy in worlds OUTSIDE its
training distribution (user-approved grid 2026-07-27, priority order).

Policy = QueuePostRZF_S40HighLoad best.pt@639 (held-out +14.7%, 20/20),
frozen, no retraining. Training distribution: speed U(5,40) km/h,
p_arrival U(0.15,0.50), p_csi 0.6, K=32 active U(16,32).

Configs (each changes ONLY the listed knobs; run in this order):
  P065  p fixed 0.65  -- deep load extrapolation. ALSO evaluates frozen
        Ent02 on the same seeds so the adaptation-gap tail (in-load- vs
        off-load-trained, both OOD here) is a direct pair. NOTE: the 7/24
        load-response point "@0.65" was U(0.15,0.65) (mean 0.40), NOT
        fixed 0.65 -- do not compare those numbers 1:1.
  V50   speed fixed 50 km/h  -- +25% past the training max (Doppler)
  V60   speed fixed 60 km/h  -- +50% past
  P055  p fixed 0.55  -- just past the load ceiling
  CSI04 p_csi 0.4     -- CSI famine; beta_m stays at the 0.6-calibrated
        values (zero-shot premise, K48-style footnote)
  STORM speed U(40,60) + p fixed 0.55 + p_csi 0.5 -- all axes OOD at once

Protocol: seeds 30000-30007 (8; 20000+ stays reserved for final paper
numbers). Per config, SUS+CQI threshold re-swept over {0.6..0.8} on the
full 8 seeds; best baseline = max(best SUS+CQI, SU+CQI); wins counted
paired per seed. One env per config so the per-seed channel cache is
reused across schedulers (threshold/policy don't touch the channel).
8-seed verdict rule: low single-digit margins read as "does not lose",
not "wins" (K48 rule).
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch, csv
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
OUT = "Run4/_analysis/ood_zeroshot.csv"
SEEDS = list(range(30000, 30008))
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]
BASE = "Run4/QueuePostRZF_S40HighLoad"
CK = torch.load(f"{BASE}/ckpt/best.pt", map_location="cpu")
CK_ENT02 = torch.load("Run4/QueuePostRZF_Ent02/ckpt/best.pt", map_location="cpu")

CONFIGS = [  # (name, overrides) -- priority order, results flush as they land
    ("P065",  dict(p_arrival_min=0.65, p_arrival_max=0.65)),
    ("V50",   dict(ue_speed_min=50.0, ue_speed_max=50.0)),
    ("V60",   dict(ue_speed_min=60.0, ue_speed_max=60.0)),
    ("P055",  dict(p_arrival_min=0.55, p_arrival_max=0.55)),
    ("CSI04", dict(p_csi=0.4)),
    ("STORM", dict(ue_speed_min=40.0, ue_speed_max=60.0,
                   p_arrival_min=0.55, p_arrival_max=0.55, p_csi=0.5)),
]

def make_cfg(overrides):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw.update(overrides)
    return Config(**raw)

def load_actor(cfg, ck):
    ac = ActorCritic(cfg)
    ac.load_state_dict(ck["model"], strict=True)
    ac.eval()
    return ac

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["config", "sched", "seed", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])

def run_sched(cname, label, env, cfg, sched):
    """Run all seeds, write rows, return per-seed reward list + mean stats."""
    ms = []
    for seed in SEEDS:
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        m = env_episode_metrics(env, cfg)
        ms.append(m)
        w.writerow([cname, label, seed, round(m["reward"], 1),
                    round(m["throughput_mbps"], 2),
                    round(m.get("goodput_mbps", float("nan")), 2),
                    round(m["completion_rate"], 4),
                    round(m["deadline_miss_rate"], 4),
                    round(m["mu_depth"], 3)])
    f.flush()
    return ([m["reward"] for m in ms],
            np.mean([m["mu_depth"] for m in ms]),
            np.mean([m["deadline_miss_rate"] for m in ms]))

summary = []
for cname, overrides in CONFIGS:
    t0 = time.time()
    print(f"\n=== [{cname}] {overrides} ===", flush=True)
    cfg = make_cfg(overrides)
    env = SchedulerEnv(cfg)          # one env/config: channel cache reuse

    # baseline anchor: SUS threshold re-sweep + SU+CQI (all 8 seeds, paired)
    base_rewards = {}                # label -> per-seed reward list
    for thr in SUS_THRS:
        cfg.sus_ortho_threshold = thr
        sch = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
        rs, _, _ = run_sched(cname, f"SUS+CQI@{thr:.2f}", env, cfg, sch)
        base_rewards[f"SUS+CQI@{thr:.2f}"] = rs
        print(f"  SUS thr {thr:.2f}: {np.mean(rs):7.0f}", flush=True)
    su = [s for s in all_baselines(cfg) if s.name == "SU+CQI"][0]
    rs, _, _ = run_sched(cname, "SU+CQI", env, cfg, su)
    base_rewards["SU+CQI"] = rs
    print(f"  SU+CQI      : {np.mean(rs):7.0f}", flush=True)
    bname = max(base_rewards, key=lambda k: np.mean(base_rewards[k]))
    bb = base_rewards[bname]

    # frozen policies (Ent02 pair only on P065)
    policies = [("PPO-HL", CK)] + ([("PPO-Ent02", CK_ENT02)] if cname == "P065" else [])
    for pname, ck in policies:
        ppo_sch = PPOScheduler(load_actor(cfg, ck), deterministic=True)
        rs, depth, miss = run_sched(cname, pname, env, cfg, ppo_sch)
        marg = (np.mean(rs) - np.mean(bb)) / abs(np.mean(bb)) * 100
        wins = sum(p > b for p, b in zip(rs, bb))
        print(f"  {pname}: {np.mean(rs):7.0f} vs {bname} {np.mean(bb):7.0f}"
              f"  마진 {marg:+.1f}%  승 {wins}/8  depth {depth:.2f}  miss {miss:.3f}",
              flush=True)
        if pname == "PPO-HL":
            summary.append((cname, np.mean(rs), np.mean(bb), bname, marg, wins, depth, miss))
    print(f"  [{cname}] {(time.time()-t0)/60:.1f} min", flush=True)

print("\n=== OOD zero-shot 요약 (HighLoad best@639 동결, seeds 30000-30007) ===")
for cname, r, b, bn, marg, wins, d, miss in summary:
    print(f"  {cname:6s}: PPO {r:6.0f} vs {bn:13s} {b:6.0f}  마진 {marg:+6.1f}%"
          f"  승 {wins}/8  depth {d:.2f}  miss {miss:.3f}")
print("\n판독: 8-seed라 한 자릿수 초반 마진은 '지지 않는다'로 읽을 것 (K48 규칙).")
print("saved ->", OUT)
f.close()
