"""E4 probe: (a) no-user invalid at first layer when any candidate valid;
(b) forced-commit lock: a lone UE gets committed at the FIRST RBG index where
it is valid, even when its CQI there is far worse than at another RBG --
i.e. the pruning is NOT always optimal-preserving."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np
import torch
from config import debug_config
from env import SchedulerEnv
from policy import ActorCritic, obs_to_tensors

cfg = debug_config(num_ue=8, episode_len_debug=20, p_arrival=0.5)
env = SchedulerEnv(cfg)
torch.manual_seed(1)
ac = ActorCritic(cfg)

obs = env.reset(0)
while not obs["active"].any():
    obs, r, d_, i = env.step(np.zeros((cfg.num_rbg, cfg.l_max), np.int64))

# (a) mask check at a retx-free RBG, first layer
obs_t = obs_to_tensors(obs, torch.device("cpu"))
e = ac.encode(obs_t)
tu = torch.from_numpy(obs["uncommitted"].astype(np.float32)).clone()
logits, valid_mask, _ = ac._position_logits_and_mask(
    obs_t, e, r=0, l=0, S_r_r=set(),
    in_slot_count=np.zeros(cfg.num_ue, np.int64), temp_uncommit=tu)
print(f"(a) at (r=0,l=0), S_r empty, {int(valid_mask[1:].sum())} valid UE "
      f"candidate(s) -> no-user valid = {bool(valid_mask[0])}")
assert not bool(valid_mask[0])

# (b) forced-commit lock with a single active UE
active = np.where(obs["active"])[0]
u = int(active[0])
cqi_u = obs["cqi_fb"][u].copy()
unc = float(obs["uncommitted"][u])
pred0 = cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate * cqi_u[0]
print(f"(b) active UEs: {active.tolist()}; take UE {u}, "
      f"uncommitted={unc:.0f} bits")
print(f"    CQI per RBG: {np.round(cqi_u, 2)}  "
      f"(worst RBG={int(cqi_u.argmin())}, best RBG={int(cqi_u.argmax())})")
print(f"    pred_btx at RBG0 = {pred0:.0f} >= uncommitted? {pred0 >= unc}")

# deactivate all other UEs so UE u is the only candidate anywhere
for v in active[1:]:
    env.traffic.remove_packet(int(v))
obs2 = env.get_observation()
out = ac.decode(obs2, deterministic=True)
seq = out["action_sequence"]
sched = [(r, l) for r in range(cfg.num_rbg) for l in range(cfg.l_max)
         if seq[r, l] == u + 1]
print(f"    decode(): UE {u} scheduled at positions {sched}")
print(f"    -> whole packet committed at RBG {sched[0][0]} "
      f"(CQI {cqi_u[sched[0][0]]:.2f}); policy had NO action that idles "
      f"RBG {sched[0][0]} to commit at RBG {int(cqi_u.argmax())} "
      f"(CQI {cqi_u.max():.2f}) instead"
      if sched else "    (not scheduled?)")

# how much MI per slot does that cost in prediction terms
mi = lambda c: cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate * c
if sched:
    r_forced = sched[0][0]
    print(f"    predicted per-slot bits: forced RBG {r_forced}: "
          f"{mi(cqi_u[r_forced]):.0f} vs best RBG: {mi(cqi_u.max()):.0f} "
          f"({100*(mi(cqi_u.max())-mi(cqi_u[r_forced]))/max(mi(cqi_u[r_forced]),1e-9):.0f}% more)")
