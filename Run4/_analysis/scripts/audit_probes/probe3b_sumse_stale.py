"""Probe 3b: PHY sum-SE vs depth at the REAL Run3 operating point
(type2, p_csi=0.6, 10 and 30 km/h) -- greedy top-m AND SUS@0.8 selection."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import collections
import numpy as np

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config              # noqa: E402
from env import SchedulerEnv           # noqa: E402
from phy import _rbg_sinr              # noqa: E402


def sus_select(order, C2, th, mmax=4):
    sel = [order[0]]
    for cand in order[1:]:
        if len(sel) >= mmax:
            break
        if 1.0 - max(C2[cand, s] for s in sel) >= th:
            sel.append(cand)
    return np.array(sel)


for speed in (10.0, 30.0):
    cfg = Config(debug=True, episode_len_debug=60, num_ue=16,
                 p_csi=0.6, ue_speed_kmh=speed,
                 pmi_mode="type2_sparse_56bit")
    env = SchedulerEnv(cfg)
    env.reset(episode_idx=0)
    acc = collections.defaultdict(list)
    sus_m = []
    zero = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    done = False
    while not done:
        t = env.slot
        if t in (10, 25, 40, 55):
            h_true, h_hat = env.h_true_slot, env.h_hat_slot
            cqi, dirs = env.csi.cqi_fb, env.csi.direction_fb
            nv = env.noise_var
            for r in range(cfg.num_rbg):
                order = np.argsort(-cqi[:, r])
                D = dirs[:, r, :]
                C2 = np.abs(D @ D.conj().T) ** 2
                for m in (1, 2, 3, 4):
                    sel = order[:m]
                    s = _rbg_sinr(h_true[sel, r], h_hat[sel, r],
                                  alpha=nv, p_rbg=cfg.p_rbg, noise_var=nv)
                    acc[m].append(float(np.log2(1 + s).sum()))
                sel = sus_select(order, C2, 0.8)
                s = _rbg_sinr(h_true[sel, r], h_hat[sel, r],
                              alpha=nv, p_rbg=cfg.p_rbg, noise_var=nv)
                acc["sus0.8"].append(float(np.log2(1 + s).sum()))
                sus_m.append(len(sel))
        _, _, done, _ = env.step(zero)
    base = np.array(acc[1]).mean()
    print(f"\n=== type2 p_csi=0.6 {speed:.0f} km/h: per-RBG sum-SE, "
          f"greedy top-m (4 slots x 8 RBGs) ===")
    for m in (1, 2, 3, 4):
        v = np.array(acc[m])
        print(f"  m={m}: sum-SE {v.mean():6.3f}  (vs m=1: {v.mean()/base-1:+7.1%})")
    v = np.array(acc["sus0.8"])
    print(f"  SUS@0.8 (mean |S|={np.mean(sus_m):.2f}): sum-SE {v.mean():6.3f} "
          f"(vs m=1: {v.mean()/base-1:+7.1%})")
