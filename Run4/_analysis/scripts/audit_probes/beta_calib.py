"""Scheduler-independent la_beta calibration (post_rzf, imperfect CSI).

Samples random (RBG, depth-m, UE-group) combinations over episodes advanced
with EMPTY allocations (no scheduler in the loop; CSI age/speed/load dynamics
evolve naturally). For each group member:
    ratio = MI_actual(h_true, W(h_hat)) / btx_cap_pred(h_hat, beta=1)
A full-cap unit first-ACKs iff la_beta <= ratio, so the beta that hits a
90% first-ACK target is the 10th percentile of the ratio distribution.
Genie world: ratio == 1 identically -> la_beta stays 1.0.
"""
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from la_planner import predict_group_link_adaptation
from phy import _rbg_sinr, mi_bits

cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                          p_arrival_min=0.15, p_arrival_max=0.40)
print(f"world: pmi={cfg.pmi_mode} p_csi={cfg.p_csi} K={cfg.num_ue} "
      f"R={cfg.num_rbg} beta_rate={cfg.beta_rate}")
env = SchedulerEnv(cfg)
rng = np.random.default_rng(777)
empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)

ratios = {m: [] for m in (1, 2, 3, 4)}
ages = []
for ep in range(12):
    env.reset(episode_idx=50000 + ep)
    for t in range(cfg.episode_len):
        if t % 7 == 0:
            nv = env.noise_var
            for m in (1, 2, 3, 4):
                for _ in range(2):
                    r = int(rng.integers(cfg.num_rbg))
                    grp = rng.choice(cfg.num_ue, size=m, replace=False)
                    h_hat = env.h_hat_slot[grp, r, :]
                    h_true = env.h_true_slot[grp, r, :]
                    _, sp, caps = predict_group_link_adaptation(
                        h_hat, nv, nv, cfg.p_rbg, cfg)
                    mi_act = mi_bits(_rbg_sinr(h_true, h_hat, nv,
                                               cfg.p_rbg, nv), cfg)
                    ratios[m].extend((mi_act / np.maximum(caps, 1e-12))
                                     .tolist())
            ages.append(float(env.csi.age.mean()))
        env.step(empty)

allr = np.concatenate([np.array(v) for v in ratios.values()])
print(f"samples: {len(allr)} member-ratios; mean CSI age {np.mean(ages):.2f}")
print(f"{'depth':>5s} {'n':>6s} {'mean':>7s} {'p10':>7s} {'p25':>7s} "
      f"{'p50':>7s}")
for m in (1, 2, 3, 4):
    v = np.array(ratios[m])
    print(f"{m:5d} {len(v):6d} {v.mean():7.3f} "
          f"{np.percentile(v, 10):7.3f} {np.percentile(v, 25):7.3f} "
          f"{np.percentile(v, 50):7.3f}")
for target in (0.85, 0.90, 0.95):
    beta = float(np.percentile(allr, (1 - target) * 100))
    ack = float((allr >= beta).mean())
    print(f"target first-ACK {target:.0%} -> la_beta = {beta:.4f} "
          f"(achieved {ack:.1%})")
beta90 = float(np.percentile(allr, 10))
by_depth = {m: float((np.array(ratios[m]) >= beta90).mean())
            for m in (1, 2, 3, 4)}
print(f"\nglobal la_beta(90%) = {beta90:.4f}; per-depth first-ACK at that "
      f"beta: " + ", ".join(f"m={m}:{p:.1%}" for m, p in by_depth.items()))
