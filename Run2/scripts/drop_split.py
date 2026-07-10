"""Split PPO/genie 'drop' into its two distinct components to resolve the
apparent discrepancy: eval_metrics.csv 'drop' = retx_drop_rate ONLY, while my
earlier confirm script lumped deadline-miss + retx-drop. READ-ONLY, CPU."""
import numpy as np, torch
from config import phase2_hard_main_config
from env import SchedulerEnv
from policy import ActorCritic
from phy import _rbg_sinr, mi_bits, predict_b_tx

SEEDS = list(range(10000, 10005))   # 5 seeds (enough to confirm the split)


def metrics(env, alloc_fn, seeds):
    out = {k: [] for k in ("comp", "miss_dl", "retx_drop", "overflow", "thr")}
    for s in seeds:
        env.reset(episode_idx=s); done = False
        while not done:
            _, _, done, _ = env.step(alloc_fn())
        e = env.ep; arr = max(e["n_arrivals"], 1)
        out["comp"].append(e["n_comp"] / arr)
        out["miss_dl"].append(e["n_miss_deadline"] / arr)
        out["retx_drop"].append(e["n_retx_drop"] / arr)
        out["overflow"].append(e.get("n_retx_overflow_drop", 0) / arr)
        out["thr"].append(e["acked_bits"] / (env.cfg.episode_len * env.cfg.slot_duration) / 1e6)
    return {k: float(np.mean(v)) for k, v in out.items()}


# PPO (stale)
cfg = phase2_hard_main_config()
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ac.load_state_dict(torch.load("runs/Run2_HardMain/ckpt/best.pt", map_location="cpu")["model"])
ac.eval()
ppo = metrics(env, lambda: ac.decode(env.get_observation(), deterministic=True)["action_sequence"], SEEDS)

# genie (perfect CSI) — reuse the greedy-marginal scheduler
from genie_probe import GenieScheduler
cfg_f = phase2_hard_main_config(p_csi=1.0)
env_f = SchedulerEnv(cfg_f)
g = GenieScheduler()
genie = metrics(env_f, lambda: g.schedule(env_f), SEEDS)

print("%-18s %8s %9s %10s %9s %8s" %
      ("scheduler", "comp", "miss_dl", "retx_drop", "overflow", "Mbps"))
print("-" * 66)
for name, m in [("PPO (stale)", ppo), ("Genie (true CSI)", genie)]:
    print("%-18s %8.3f %9.3f %10.4f %9.3f %8.2f" %
          (name, m["comp"], m["miss_dl"], m["retx_drop"], m["overflow"], m["thr"]))
print("\nNote: eval_metrics.csv 'drop' column == retx_drop only.")
print("My earlier 'drop' = miss_dl + retx_drop (that is why it looked ~0.15).")
