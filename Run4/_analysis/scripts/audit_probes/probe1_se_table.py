"""Probe 1: verify C1's SE table with the repo's own PHY functions.

Perfect orthogonality + perfect CSI at target SNR 10 dB:
  - m orthonormal channel rows, h_hat == h_true
  - RZF precoder (alpha = sigma^2, per-stream power P_r/m)
  - b_tx from predict_b_tx(SU CQI at full P_r)
  - IR rounds needed = b_tx / per-slot MI, and the exact env IR loop
    (TransmissionManager) to count actual slots-to-ACK.
Also computes effective per-RBG throughput with the artifact vs a
hypothetical m-aware ("correct") LA.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
import sys
import numpy as np

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                      # noqa: E402
from phy import rzf_precoder, _rbg_sinr, predict_b_tx, mi_bits  # noqa: E402
from transmission import TransmissionManager   # noqa: E402
from traffic import Packet                     # noqa: E402

cfg = Config()
P = cfg.p_rbg                     # 1.0
S = 10.0 ** (cfg.target_snr_db / 10.0)   # 10.0 (10 dB)
sigma2 = P / S                    # |h| = 1 -> SU SNR = P*1/sigma2 = S
cqi_su = np.log2(1.0 + S)         # fed-back SU CQI (csi.py formula, fresh)
b_tx = float(predict_b_tx(cqi_su, cfg))
print(f"P_r={P}, sigma2={sigma2:.4f}, SU CQI={cqi_su:.4f}, "
      f"b_tx={b_tx:.1f} bits, N_RE={cfg.n_re_rbg}")

print(f"\n{'m':>2} {'SINR':>7} {'SINR_dB':>8} {'MI/slot':>9} {'rounds':>7} "
      f"{'slots(env IR)':>13} {'thr/RBG(art)':>13} {'thr/RBG(corr)':>14} "
      f"{'gain_art':>9} {'gain_corr':>10}")
su_thr = None
for m in range(1, 5):
    # perfectly orthogonal channels, perfect CSI
    H = np.zeros((m, cfg.num_bs_ant), dtype=complex)
    for i in range(m):
        H[i, i] = 1.0
    sinr = _rbg_sinr(H, H, alpha=sigma2, p_rbg=P, noise_var=sigma2)
    s = float(sinr[0])
    mi = float(mi_bits(s, cfg))                 # per-slot MI per stream
    rounds = b_tx / mi

    # exact env IR loop: slots until ACK for one unit
    mgr = TransmissionManager(cfg)
    mgr.reset()
    pkt = Packet(packet_id=0, ue_id=0, arrival_slot=0, size=10**9,
                 deadline=10**9)
    u = mgr.create_unit(pkt, rbg_id=0, layer_id=0, b_tx=b_tx)
    smap = np.zeros((cfg.num_rbg, cfg.l_max)); smap[0, 0] = s
    slots = 0
    while not u.is_acked and slots < 20:
        mgr.process_slot(smap)
        slots += 1

    thr_artifact = m * b_tx / slots             # bits per RBG-slot, artifact LA
    thr_correct = m * mi                        # m-aware LA: ACK every slot
    if m == 1:
        su_thr = thr_artifact
    print(f"{m:2d} {s:7.3f} {10*np.log10(s):8.2f} {mi:9.1f} {rounds:7.3f} "
          f"{slots:13d} {thr_artifact:13.1f} {thr_correct:14.1f} "
          f"{thr_artifact/su_thr-1:+9.1%} {thr_correct/su_thr-1:+10.1%}")

print("\nClaimed rounds at 10 dB: m=2:1.34  m=3:1.64  m=4:1.91")
