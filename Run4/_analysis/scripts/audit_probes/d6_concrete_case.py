"""D6 concrete case: force ghost streams with a backlog-UNAWARE scheduler.

A scheduler that pins one UE across ALL 8 RBGs (ignoring backlog) makes the
pre-pass count it on every RBG while the creation loop exhausts its packet
after ~2 RBGs -> later RBGs carry ghost counts -> co-scheduled UEs there get
over-de-rated B_tx. Demonstrates the mechanism is real code-reachable, but
only for schedulers that do NOT thread the SU-sized budget (none in repo).
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "4"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
sys.path.insert(0, "/tmp/gpt_r2_verify")
import numpy as np
from config import debug_config
from d6_ghost_probe import ProbeEnv


class NaiveTwoUE:
    """Backlog-unaware: smallest-backlog UE (layer 0, exhausts early -> ghost)
    + largest-backlog UE (layer 1, keeps bits -> over-de-rated) on ALL RBGs."""
    def schedule(self, env):
        cfg = env.cfg
        obs = env.get_observation()
        act = obs["active"] & (obs["uncommitted"] > 0)
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        unc = np.where(act, obs["uncommitted"], np.nan)
        if act.sum() < 2:
            return alloc
        a = int(np.nanargmin(unc))          # small backlog -> ghost source
        b = int(np.nanargmax(unc))          # large backlog -> victim
        if a == b:
            return alloc
        for r in range(cfg.num_rbg):
            alloc[r, 0] = a + 1
            alloc[r, 1] = b + 1
        return alloc


cfg = debug_config(episode_len_debug=100, p_arrival=0.4)
cfg.mu_aware_la = True
env = ProbeEnv(cfg)
env.start_stats()
env.reset(episode_idx=0)
done = False
sch = NaiveTwoUE()
while not done:
    _, _, done, _ = env.step(sch.schedule(env))
S = env.S
print("backlog-unaware scheduler, 100 slots, debug cfg, mu_aware_la=True:")
print(f"  rbg_new={S['rbg_new']} derated={S['rbg_derated']} "
      f"ghost_rbg={S['rbg_ghost']} ({100*S['rbg_ghost']/max(S['rbg_new'],1):.1f}%) "
      f"ghost_skips={S['ghost_skips']} eps_skips={S['eps_skips']}")
print(f"  units={S['units']} affected={S['units_affected']} "
      f"bit_shortfall={S['bits_shortfall']:.0f} "
      f"({100*S['bits_shortfall']/max(S['bits_committed'],1):.2f}% of committed)")
