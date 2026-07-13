"""Part A: verify GPT R2-2's projection-loss claim with phy.rzf_precoder.

Claim: with perfect CSI at the 10 dB operating point, post-RZF SINR <=
SU_SNR/m STRICTLY, with mean ratios ~0.973 / 0.947 / 0.921 for m = 2/3/4,
so a cap-limited unit sized at b_tx = NRE*log2(1 + SU_SNR/m) still NACKs
on its first attempt (deficit small -> ACK on attempt 2).

Setup mirrors the simulator: M=32 antennas, p_rbg = 1.0 (phase4 preset:
p_total 8 / 8 RBG), sigma^2 calibrated so median SU SNR = 10 dB,
alpha = sigma^2 (rzf_alpha_mode='noise'). i.i.d. CN(0,1) channels
(the perfect-CSI abstraction GPT used).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np
from phy import rzf_precoder, _rbg_sinr

M = 32
P = 1.0
TARGET_SNR_DB = 10.0
rng = np.random.default_rng(7)

# calibration: median SU beamformed SNR (perfect direction => gain = |h|^2)
h_cal = (rng.standard_normal((200000, M)) + 1j * rng.standard_normal((200000, M))) / np.sqrt(2.0)
g = P * np.sum(np.abs(h_cal) ** 2, axis=1)
sigma2 = float(np.median(g) / 10 ** (TARGET_SNR_DB / 10.0))
print(f"calibrated sigma2 = {sigma2:.4f}  (median |h|^2 = {np.median(np.sum(np.abs(h_cal)**2,1)):.3f})")

N_GROUPS = 20000
print(f"\n{'m':>2} {'mean ratio':>10} {'p5':>8} {'p95':>8} {'max':>10} {'frac>=1':>8} "
      f"{'IRnew_mean':>10} {'IRnew>1frac':>11} {'IRold_mean':>10} {'IRold>1frac':>11}")
for m in (2, 3, 4):
    ratios = []
    ir_new = []   # required IR rounds under fair-LA b_tx = log2(1+snr/m)
    ir_old = []   # required IR rounds under old  b_tx = log2(1+snr)
    for _ in range(N_GROUPS):
        H = (rng.standard_normal((m, M)) + 1j * rng.standard_normal((m, M))) / np.sqrt(2.0)
        snr_su = P * np.sum(np.abs(H) ** 2, axis=1) / sigma2          # perfect-CSI SU SNR
        sinr = _rbg_sinr(H, H, alpha=sigma2, p_rbg=P, noise_var=sigma2)  # perfect CSI
        ratios.append(sinr / (snr_su / m))
        se_act = np.log2(1.0 + sinr)
        ir_new.append(np.log2(1.0 + snr_su / m) / se_act)
        ir_old.append(np.log2(1.0 + snr_su) / se_act)
    ratios = np.concatenate(ratios); ir_new = np.concatenate(ir_new); ir_old = np.concatenate(ir_old)
    print(f"{m:>2} {ratios.mean():>10.4f} {np.percentile(ratios,5):>8.4f} "
          f"{np.percentile(ratios,95):>8.4f} {ratios.max():>10.6f} "
          f"{(ratios>=1).mean():>8.4f} {ir_new.mean():>10.4f} {(ir_new>1.0).mean():>11.4f} "
          f"{ir_old.mean():>10.4f} {(ir_old>1.0).mean():>11.4f}")

# sanity: the strict bound argument -- desired power (P/m)|h^H v|^2 <= (P/m)|h|^2
# check on one group that unit-norm columns imply |h^H v|^2 <= |h|^2
H = (rng.standard_normal((3, M)) + 1j * rng.standard_normal((3, M))) / np.sqrt(2.0)
W = rzf_precoder(H, sigma2, P)
eff = np.abs(H @ W) ** 2
print("\nCauchy-Schwarz check: desired gain / (P/m * |h|^2) =",
      np.diag(eff) / ((P / 3) * np.sum(np.abs(H) ** 2, axis=1)))
