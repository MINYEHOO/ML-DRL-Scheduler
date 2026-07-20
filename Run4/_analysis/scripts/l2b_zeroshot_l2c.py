"""MixedSpeed_L2b policy ZERO-SHOT in the L2c no-beta world, vs baselines.

Load Run3/MixedSpeed_L2b best.pt (trained in the LEGACY-LA world, speed
U(5,30)) and evaluate it with NO retraining in the L2c world (post_rzf +
rbg_major + beta=1, speed U(5,40)) -- exactly the world the live
Run4/MixedSpeed_L2c training uses (config read from its config.json).
Also evaluates Run4/MixedSpeed_L2c best.pt (trained in-world) on the same
seeds for reference. Baselines are NOT re-run: l2c_world_baselines.csv
already covers the identical protocol (same cfg source, same seeds
10000-10007, same env_episode_metrics).

Cross-world load is STRICT: legacy-vs-post_rzf must not change any tensor
shape; if it does we abort rather than silently evaluate a part-fresh net.
"""
import os, sys, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # eval is CPU-only
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import json
import numpy as np, torch
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import env_episode_metrics, PPOScheduler

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/l2b_zeroshot_l2c.csv"
SEEDS = list(range(10000, 10008))                # == l2c_world_baselines.py

raw = json.load(open("Run4/MixedSpeed_L2c/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.resolved_la_mode() == "post_rzf" and cfg.decode_order == "rbg_major"
assert cfg.la_beta == 1.0 and not cfg.la_beta_by_depth
env = SchedulerEnv(cfg)

def load_policy(path, tag):
    # Critic input width differs across worlds (L2b critic_v2 161 vs L2c 134),
    # but the critic is unused in deterministic eval. Require an EXACT match
    # for every actor/encoder tensor; only value_head/return-normalizer may
    # stay fresh -- anything else fresh would invalidate the zero-shot claim.
    ac = ActorCritic(cfg)
    ck = torch.load(path, map_location="cpu")
    own = ac.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    fresh = sorted(set(own) - set(keep))
    bad = [k for k in fresh if not k.startswith(("value_head", "ret_"))]
    assert not bad, f"non-critic tensors failed to load: {bad}"
    ac.load_state_dict(keep, strict=False)
    ac.eval()
    print(f"{tag}: update {ck['update']} eval {ck.get('eval_reward', float('nan')):.1f} "
          f"loaded {len(keep)}/{len(own)} tensors from {path} (fresh: {fresh})", flush=True)
    return ac

scheds = [
    ("PPO-L2b-zeroshot", PPOScheduler(load_policy("Run3/MixedSpeed_L2b/ckpt/best.pt", "L2b donor"), deterministic=True)),
    ("PPO-L2c-trained",  PPOScheduler(load_policy("Run4/MixedSpeed_L2c/ckpt/best.pt", "L2c best"),  deterministic=True)),
]

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain",
        "first_ack_rate", "attempts_per_acked"]
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["baseline", "seed"] + KEYS)
for name, sch in scheds:
    ms = []
    for s in SEEDS:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        em = env_episode_metrics(env, cfg)
        ms.append(em)
        w.writerow([name, s] + [round(float(em[k]), 4) for k in KEYS])
        print(f"  {name} seed {s}: rew {em['reward']:.0f} miss {em['deadline_miss_rate']:.3f}", flush=True)
    print(f"{name:18s} rew {np.mean([m['reward'] for m in ms]):8.1f}  "
          f"thr {np.mean([m['throughput_mbps'] for m in ms]):7.2f}  "
          f"miss {np.mean([m['deadline_miss_rate'] for m in ms]):.4f}  "
          f"retx {np.mean([m['retx_drop_rate'] for m in ms]):.4f}  "
          f"depth {np.mean([m['mu_depth'] for m in ms]):.3f}", flush=True)

# combined table with the existing baseline grid (same world, same seeds)
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))
for f in ("Run4/_analysis/l2c_world_baselines.csv", OUT):
    for r in csv.DictReader(open(f)):
        for k in KEYS:
            agg[r["baseline"]][k].append(float(r[k]))
print("\n=== L2c no-beta world, 8 paired seeds (10000-10007) ===")
print(f"{'scheduler':18s} {'reward':>8s} {'thr':>7s} {'miss':>7s} {'retx':>7s} {'depth':>6s}")
for name in sorted(agg, key=lambda n: -np.mean(agg[n]["reward"])):
    a = agg[name]
    print(f"{name:18s} {np.mean(a['reward']):8.0f} {np.mean(a['throughput_mbps']):7.2f} "
          f"{np.mean(a['deadline_miss_rate']):7.4f} {np.mean(a['retx_drop_rate']):7.4f} "
          f"{np.mean(a['mu_depth']):6.3f}")
print("saved ->", OUT)
