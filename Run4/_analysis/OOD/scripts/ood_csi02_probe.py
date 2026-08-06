"""CSI02 probe: p_csi = 0.2 zero-shot (user-requested 2026-08-05).

Motivation: the trained p_csi = 0.6 (mean report interval 0.83 ms) is
generous vs real CSI periodicity (5-20 ms). p_csi = 0.2 gives a mean
report interval of 2.5 ms and mean CSI age ~2 ms -- much closer to a
dense real reporting configuration (age/coherence ~0.45 at 30 km/h).
Deeper famine than the OOD probe's CSI04 (p_csi 0.4).

Two editions, mirroring the two OOD probes; everything else stays at the
respective training world (speed U(5,40), p_arrival U(0.15,0.50), K=32,
n_active U(16,32); beta_m frozen at its p_csi-0.6-calibrated tuple --
zero-shot premise, the LA margin is knowingly mis-calibrated here):
  CSI02-cont : continuous-CQI world, frozen S40HighLoad best.pt@639
  CSI02-cqi4 : nr4bit world,        frozen S40HL_CQI4 best.pt@409

Protocol: seeds 30000-30007, per-world SUS+CQI threshold re-sweep
{0.60..0.80}, SU+CQI anchor, paired wins. 8-seed verdict rule applies.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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
OUT = "Run4/_analysis/ood_csi02.csv"
SEEDS = list(range(30000, 30008))
SUS_THRS = [0.60, 0.65, 0.70, 0.75, 0.80]

EDITIONS = [
    ("CSI02-cont", "Run4/QueuePostRZF_S40HighLoad"),
    ("CSI02-cqi4", "Run4/QueuePostRZF_S40HL_CQI4"),
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"policy device: {DEV} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})", flush=True)

def make_cfg(base):
    raw = json.load(open(f"{base}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw["p_csi"] = 0.2
    return Config(**raw)

def load_actor(cfg, ck):
    ac = ActorCritic(cfg)
    ac.load_state_dict(ck["model"], strict=True)
    ac.eval()
    return ac.to(DEV)

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["config", "sched", "seed", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])

def run_sched(cname, label, env, cfg, sched):
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
            np.mean([m["deadline_miss_rate"] for m in ms]),
            np.mean([m["first_ack_rate"] for m in ms]) if "first_ack_rate" in ms[0] else float("nan"))

summary = []
for cname, base in EDITIONS:
    t0 = time.time()
    ck = torch.load(f"{base}/ckpt/best.pt", map_location="cpu")
    print(f"\n=== [{cname}] base {base}, p_csi 0.2, policy best.pt@{ck['update']} ===", flush=True)
    cfg = make_cfg(base)
    assert cfg.p_csi == 0.2
    env = SchedulerEnv(cfg)

    base_rewards = {}
    for thr in SUS_THRS:
        cfg.sus_ortho_threshold = thr
        sch = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
        rs, _, _, _ = run_sched(cname, f"SUS+CQI@{thr:.2f}", env, cfg, sch)
        base_rewards[f"SUS+CQI@{thr:.2f}"] = rs
        print(f"  SUS thr {thr:.2f}: {np.mean(rs):7.0f}", flush=True)
    su = [s for s in all_baselines(cfg) if s.name == "SU+CQI"][0]
    rs, _, _, _ = run_sched(cname, "SU+CQI", env, cfg, su)
    base_rewards["SU+CQI"] = rs
    print(f"  SU+CQI      : {np.mean(rs):7.0f}", flush=True)
    bname = max(base_rewards, key=lambda k: np.mean(base_rewards[k]))
    bb = base_rewards[bname]

    ppo_sch = PPOScheduler(load_actor(cfg, ck), deterministic=True)
    rs, depth, miss, fack = run_sched(cname, "PPO", env, cfg, ppo_sch)
    marg = (np.mean(rs) - np.mean(bb)) / abs(np.mean(bb)) * 100
    wins = sum(p > b for p, b in zip(rs, bb))
    print(f"  PPO: {np.mean(rs):7.0f} vs {bname} {np.mean(bb):7.0f}"
          f"  마진 {marg:+.1f}%  승 {wins}/8  depth {depth:.2f}  miss {miss:.3f}  first-ACK {fack:.3f}",
          flush=True)
    summary.append((cname, np.mean(rs), np.mean(bb), bname, marg, wins, depth, miss))
    print(f"  [{cname}] {(time.time()-t0)/60:.1f} min", flush=True)

print("\n=== CSI02 (p_csi 0.2) zero-shot 요약, seeds 30000-30007 ===")
for cname, r, b, bn, marg, wins, d, miss in summary:
    print(f"  {cname:11s}: PPO {r:6.0f} vs {bn:13s} {b:6.0f}  마진 {marg:+6.1f}%"
          f"  승 {wins}/8  depth {d:.2f}  miss {miss:.3f}")
print("\n판독: 8-seed 규칙 적용. 비교점: CSI04(p_csi 0.4)는 continuous +14.8%, cqi4 +10.6%;")
print("       학습분포(p_csi 0.6) in-dist는 continuous +14.7%, cqi4 +13.9% (20-seed).")
print("saved ->", OUT)
f.close()
