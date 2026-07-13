"""Probe 2: magnitude of the bootstrap value at truncation on a REAL critic.

Loads Run3/MixedSpeed_L2b best.pt (CPU), rolls one episode with the trained
policy exactly like ppo.collect_rollout, then reports:
  A. last_value = V(mixed final obs)   -- what training actually uses
  B. V(final obs with CORRECTED retx structures + age+1)  -- best buildable
     approximation of a true s_T (compaction rerun at slot T; CSI feedback
     for slot T cannot exist -> kept at last value, age incremented)
  C. V(s_{T-1}) and value stats over the episode  -- scale reference
  D. delta std over episode vs the injected error gamma*(A - 0) if the
     episodic-consistent target were terminal (last_value=0), plus the
     (gamma*lambda)^k decay horizon.
"""
import dataclasses
import numpy as np
import torch

from config import Config
from env import SchedulerEnv
from policy import ActorCritic

CKPT = "Run3/MixedSpeed_L2b/ckpt/best.pt"
ck = torch.load(CKPT, map_location="cpu")
ck_cfg = ck["cfg"]
fields = {f.name for f in dataclasses.fields(Config)}
kw = {k: v for k, v in ck_cfg.items() if k in fields}
# debug-length episode for CPU speed; slot-phase feature uses episode_len so
# the critic sees phases identical to training (slot/episode_len in [0,1])
kw["debug"] = True
kw["episode_len_debug"] = 200
cfg = Config(**kw)

env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ac.load_state_dict(ck["model"])
ac.eval()
print(f"loaded {CKPT} update={ck.get('update')} eval_reward={ck.get('eval_reward')}")

torch.manual_seed(0)
obs = env.reset(episode_idx=10000)
values, rewards = [], []
done = False
prev_obs = None
with torch.no_grad():
    while not done:
        out = ac.decode(obs, deterministic=False)
        values.append(float(out["value"]))
        prev_obs = obs
        obs, r, done, info = env.step(out["action_sequence"])
        rewards.append(float(r))

    final_obs = obs
    lv_mixed = float(ac.state_value(final_obs))          # A: what code uses

    # B: corrected retx structures via a real compaction at slot T + age+1
    res = env.txmgr.compact_pending(env.traffic)
    fixed_obs = dict(final_obs)
    fixed_obs["fixed_allocation"] = res["fixed_allocation"]
    fixed_obs["fixed_mask"] = res["fixed_mask"]
    fixed_obs["fixed_unit_map"] = res["fixed_unit_map"]
    fixed_obs["initial_S_r"] = res["initial_S_r"]
    fixed_obs["occupied"] = res["fixed_allocation"].copy()
    fixed_obs["age"] = final_obs["age"] + 1.0
    lv_corr = float(ac.state_value(fixed_obs))

    lv_zero = 0.0                                        # terminal-consistent

values = np.array(values); rewards = np.array(rewards)
gamma, lam = cfg.ppo_gamma, cfg.ppo_gae_lambda
# one-step TD errors as trained (bootstrap with lv_mixed)
nv = np.append(values[1:], lv_mixed)
deltas = rewards + gamma * nv - values

print(f"\nepisode_len={cfg.episode_len}  mean reward/slot={rewards.mean():.3f}")
print(f"V stats over episode: mean={values.mean():.3f} "
      f"min={values.min():.3f} max={values.max():.3f}  "
      f"V(s_0)={values[0]:.3f} V(s_T-1)={values[-1]:.3f}")
print(f"stale fixed_mask.sum={int(final_obs['fixed_mask'].sum())}  "
      f"corrected fixed_mask.sum={int(res['fixed_mask'].sum())}")
print(f"A. last_value (mixed obs, AS TRAINED) = {lv_mixed:.4f}")
print(f"B. V(corrected s_T approx)           = {lv_corr:.4f}   "
      f"staleness error A-B = {lv_mixed - lv_corr:+.4f}")
print(f"C. terminal-consistent choice        = 0.0        "
      f"bootstrap-vs-terminal error gamma*A = {gamma*lv_mixed:+.4f}")
print(f"delta (TD error) std over episode = {deltas.std():.4f}, "
      f"mean = {deltas.mean():+.4f}")
gl = gamma * lam
for k in (0, 5, 10, 20, 40, 78):
    print(f"  (gamma*lambda)^{k} = {gl**k:.4f} -> adv perturbation at t=T-1-{k}: "
          f"staleness {gl**k*gamma*abs(lv_mixed-lv_corr):.4f}, "
          f"vs-terminal {gl**k*gamma*abs(lv_mixed):.4f}")
n1pct = int(np.ceil(np.log(0.01) / np.log(gl)))
print(f"perturbation falls below 1% of its t=T-1 size after {n1pct} slots")
