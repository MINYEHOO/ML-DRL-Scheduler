"""Confirm Run2 PPO win: eval best.pt (update 479) on 10 held-out seeds vs the
5 heuristics (stale CSI, what PPO competes with) AND a fresh-CSI oracle ceiling
(p_csi=1.0, same greedy heuristics with zero staleness). Reports reward +
completion + drop + throughput so we see (a) is +36% robust beyond 3 seeds,
(b) is PPO below/above the oracle ceiling, (c) the win mechanism. READ-ONLY:
loads a checkpoint copy, runs on CPU; does not touch the live GPU training."""
import numpy as np, torch
from config import phase2_hard_main_config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines

SEEDS = list(range(10000, 10010))   # 10 held-out seeds (training eval used 3)
CKPT = "runs/Run2_HardMain/ckpt/best.pt"


def run_episode(env, alloc_fn, seed):
    env.reset(episode_idx=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(alloc_fn())
    e = env.ep
    arr = max(e["n_arrivals"], 1)
    return dict(reward=e["reward"], comp=e["n_comp"] / arr,
                drop=(e["n_miss_deadline"] + e["n_retx_drop"]) / arr,
                thr=e["acked_bits"] / (env.cfg.episode_len * env.cfg.slot_duration) / 1e6)


def eval_scheduler(env, sched_factory, seeds):
    recs = [run_episode(env, sched_factory(env), s) for s in seeds]
    return {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}, \
           [r["reward"] for r in recs]


# ---- stale (p_csi=0.2): PPO + heuristics ----
cfg = phase2_hard_main_config()           # K=16, 1000-slot, 30km/h, p_csi=0.2
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ck = torch.load(CKPT, map_location="cpu")
ac.load_state_dict(ck["model"]); ac.eval()
print("loaded best.pt update=%d (train-eval reward %.1f), evaluating on %d seeds\n"
      % (ck["update"], ck.get("eval_reward", float("nan")), len(SEEDS)))

def ppo_factory(env):
    return lambda: ac.decode(env.get_observation(), deterministic=True)["action_sequence"]

def base_factory(sch):
    # factory: takes env, returns a 0-arg alloc fn (matches eval_scheduler)
    return lambda e: (lambda: sch.schedule(e))

results = {}
agg, ppo_rews = eval_scheduler(env, ppo_factory, SEEDS)
results["PPO (stale)"] = agg
for sch in all_baselines(cfg):
    a, _ = eval_scheduler(env, base_factory(sch), SEEDS)
    results["%s (stale)" % sch.name] = a

# ---- fresh-CSI oracle ceiling (p_csi=1.0): same heuristics, zero staleness ----
cfg_f = phase2_hard_main_config(p_csi=1.0)
env_f = SchedulerEnv(cfg_f)
for sch in all_baselines(cfg_f):
    a, _ = eval_scheduler(env_f, base_factory(sch), SEEDS)
    results["%s (FRESH oracle)" % sch.name] = a

# ---- report ----
print("%-26s %9s %7s %7s %8s" % ("scheduler", "reward", "comp", "drop", "Mbps"))
print("-" * 62)
for k, v in results.items():
    print("%-26s %9.1f %7.3f %7.3f %8.2f" % (k, v["reward"], v["comp"], v["drop"], v["thr"]))

ppo = results["PPO (stale)"]["reward"]
best_stale = max(v["reward"] for k, v in results.items() if "(stale)" in k and "PPO" not in k)
best_fresh = max(v["reward"] for k, v in results.items() if "FRESH" in k)
print("\n=== VERDICT (10 seeds) ===")
print("PPO (stale)              : %.1f" % ppo)
print("best stale heuristic     : %.1f  -> PPO %+.1f%%" % (best_stale, (ppo-best_stale)/best_stale*100))
print("fresh-CSI oracle ceiling : %.1f  -> PPO %+.1f%% (negative = PPO below ceiling = legit)"
      % (best_fresh, (ppo-best_fresh)/best_fresh*100))
print("PPO per-seed rewards:", [round(r) for r in ppo_rews])
print("PPO reward std across seeds: %.1f (%.1f%% of mean)" % (np.std(ppo_rews), np.std(ppo_rews)/np.mean(ppo_rews)*100))
