"""Round-10 tail-swallow census in the OFFICIAL beta_m world.

Counts over-cap swallow events (B_tx > btx_cap) on real baseline traces and,
for each, whether capping at btx_cap would have changed the first-attempt
ACK outcome. Determinations:
  unit          : every (rbg, ue) entry the planner returns in close_rbg btx
  over-cap      : b > cap + 1e-9, where cap = la_beta_by_depth[m-1]-backed
                  btx_cap recomputed on the FINAL group (C in the audit
                  notation; the raw beta-free MI is M = C / beta_m)
  induced NACK  : cap - 1e-9 <= actual_MI < b - 1e-6  (i.e. capping would
                  have first-ACKed, the swallow first-NACKs)
Scope: baseline traces (SUS+CQI, CQI-greedy, Random) x 8 eval seeds
10000-10007 -- NOT the live PPO pilot's own action trace, and ACK-flip
equivalence only: full corrected-trajectory (counterfactual helper A/B)
equivalence is NOT tested here (see AUDIT_CODE_CHANGES round-9/10).

Reference result (2026-07-13, code-under-test 9ed28d0): 1,477,128 units,
126 over-cap events (0.0085%, depths 3-4 only), mean overage 0.505 bit,
total over-credit ~63.6 bit, induced first-NACKs 0.
"""
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from baselines import SUSCQI, CQIGreedy, Random
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation
from phy import _rbg_sinr, mi_bits

cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                          la_beta_by_depth=(0.9815, 0.7306, 0.6466, 0.5922),
                          p_arrival_min=0.15, p_arrival_max=0.40)
env = SchedulerEnv(cfg)

stats = dict(units=0, overcap=0, overcap_bits=0.0, overcap_nack=0,
             nack_by_depth=[0, 0, 0, 0], win_by_depth=[0, 0, 0, 0])
orig_close = SlotAllocationPlanner.close_rbg


def hooked(self, r, fixed_ues, new_ues):
    kept, btx = orig_close(self, r, fixed_ues, new_ues)
    if btx:
        grp = sorted(set(int(u) for u in fixed_ues)) + [u for u in new_ues
                                                        if u in btx]
        m = len(grp)
        h = self.h_hat_slot[grp, r, :]
        _, _, caps = predict_group_link_adaptation(
            h, self.noise_var, self.alpha, self.cfg.p_rbg, self.cfg)
        for i, u in enumerate(grp):
            if u not in btx:
                continue
            stats["units"] += 1
            b, cap = btx[u], float(caps[i])
            if b > cap + 1e-9:
                stats["overcap"] += 1
                stats["overcap_bits"] += b - cap
                stats["win_by_depth"][min(m, 4) - 1] += 1
                mi = float(mi_bits(_rbg_sinr(
                    ENV.h_true_slot[grp, r, :], ENV.h_hat_slot[grp, r, :],
                    ENV.noise_var, self.cfg.p_rbg, ENV.noise_var),
                    self.cfg)[i])
                if cap - 1e-9 <= mi < b - 1e-6:
                    stats["overcap_nack"] += 1
                    stats["nack_by_depth"][min(m, 4) - 1] += 1
    return kept, btx


if __name__ == "__main__":
    SlotAllocationPlanner.close_rbg = hooked
    ENV = env
    for sch in (SUSCQI(), CQIGreedy(), Random(seed=1)):
        for s in range(10000, 10008):
            env.reset(episode_idx=s)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
    SlotAllocationPlanner.close_rbg = orig_close
    print(f"units created           : {stats['units']}")
    print(f"over-cap swallow events : {stats['overcap']} "
          f"({stats['overcap'] / max(stats['units'], 1) * 100:.4f}%)")
    print(f"over-cap bits total     : {stats['overcap_bits']:.2f} "
          f"(mean {stats['overcap_bits'] / max(stats['overcap'], 1):.3f})")
    print(f"induced first-NACK      : {stats['overcap_nack']}")
    print(f"window events by depth  : {stats['win_by_depth']}")
    print(f"induced-NACK by depth   : {stats['nack_by_depth']}")
