"""Probe 3 (C3 separation): PHY-layer sum-SE vs MU depth, NO b_tx/LA layer.

If true sum-SE (bits/RBG/slot capacity) already decreases with depth under
type2 CSI, near-SU is genuine physics independent of the b_tx artifact.
Under genie CSI sum-SE should increase with depth (power split only)
-> at the genie operating point the artifact is the main anti-MU force.

K=16, fresh CSI (p_csi=1.0, sampled slots), greedy top-m CQI selection
per RBG, RZF from h_hat, SINR evaluated on h_true (repo functions).
"""
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

for mode in ("type2_sparse_56bit", "genie"):
    cfg = Config(debug=True, episode_len_debug=60, num_ue=16,
                 p_csi=1.0, ue_speed_kmh=10.0, pmi_mode=mode)
    env = SchedulerEnv(cfg)
    env.reset(episode_idx=0)
    acc = collections.defaultdict(list)
    zero = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    done = False
    while not done:
        t = env.slot
        if t in (0, 15, 30, 45):
            h_true, h_hat = env.h_true_slot, env.h_hat_slot
            cqi = env.csi.cqi_fb
            nv = env.noise_var
            for r in range(cfg.num_rbg):
                order = np.argsort(-cqi[:, r])
                for m in (1, 2, 3, 4):
                    sel = order[:m]
                    s = _rbg_sinr(h_true[sel, r], h_hat[sel, r],
                                  alpha=nv, p_rbg=cfg.p_rbg, noise_var=nv)
                    acc[m].append(float(np.log2(1 + s).sum()))
        _, _, done, _ = env.step(zero)
    print(f"\n=== {mode}: per-RBG sum-SE [bit/s/Hz], greedy top-m, "
          f"fresh CSI (mean over 4 slots x 8 RBGs) ===")
    for m in (1, 2, 3, 4):
        v = np.array(acc[m])
        print(f"  m={m}: sum-SE {v.mean():6.3f}  (vs m=1: "
              f"{v.mean()/np.array(acc[1]).mean()-1:+7.1%})")
