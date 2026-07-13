"""E2 probe: is the terminal obs (used for GAE bootstrap V(s_T)) prepared
like a real next-slot obs?  Shows the concrete staleness of the obs that
policy.state_value() receives at done."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np
from config import debug_config
from env import SchedulerEnv

cfg = debug_config(num_ue=8, episode_len_debug=20, p_arrival=0.5)
env = SchedulerEnv(cfg)
rng = np.random.default_rng(0)

obs = env.reset(0)
done = False
prev_obs = obs
while not done:
    # random-ish scheduler (from env.py __main__) so retx units exist
    alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    for l in range(cfg.l_max):
        for r in range(cfg.num_rbg):
            sel_ue = set((alloc[r][alloc[r] > 0] - 1).tolist())
            cand = env.position_candidates(r, sel_ue)
            ues = np.where(cand)[0]
            if ues.size and rng.random() < 0.8:
                alloc[r, l] = int(rng.choice(ues)) + 1
    prev_obs = obs
    obs, reward, done, info = env.step(alloc)

obs_done = obs
print(f"terminal obs: slot field = {obs_done['slot']} (episode_len={cfg.episode_len})")
print(f"age at slot T-1 obs : {prev_obs['age']}")
print(f"age at terminal obs : {obs_done['age']}  "
      f"(identical -> Age NOT ticked for the elapsed slot: "
      f"{np.array_equal(prev_obs['age'], obs_done['age'])})")

# fixed grid staleness: unit ids shown in the terminal obs vs actually alive
shown = set(int(u) for u in obs_done["fixed_unit_map"].ravel() if u >= 0)
alive = set(u.unit_id for u in env.txmgr.units)
print(f"fixed_unit_map unit ids in terminal obs : {sorted(shown)}")
print(f"units actually still pending after slot : {sorted(alive)}")
print(f"stale entries (shown but already ACKed/dropped): "
      f"{sorted(shown - alive)}")
print(f"missing entries (pending but not shown): {sorted(alive - shown)}")

# arrivals: a real prepared s_T would draw slot-T arrivals; terminal obs can't
print(f"active UEs in terminal obs: {int(obs_done['active'].sum())} "
      f"(no slot-T arrivals drawn, no slot-T CSI feedback, no compaction)")

# and the reason the env cannot prepare a true s_T even if it wanted to:
try:
    env._prepare_slot(env.slot)
except IndexError as e:
    print(f"env._prepare_slot({env.slot}) -> IndexError ({e}): the channel "
          f"precompute has only {cfg.episode_len} slots; a true s_T obs "
          f"does not exist in this design")
