"""Probe for audit claims D1/D2 (noise calibration & genie fairness).

D1: does sigma2 differ between pmi_mode=type2 and pmi_mode=genie on the SAME
    episode channel, and by how much (dB)? Is the median SU SNR pinned to the
    target in both worlds (i.e. codebook loss compensated at the median)?
D2: how non-causal is the calibration in practice — sigma2 from the full
    episode vs prefix-only (first 10 / first half slots), and how much does
    sigma2 vary across episodes (upper bound on exploitable future info)?

CPU only, read-only on the repo, small config (episode_len=200 debug).
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "4"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np
from config import Config
from channel import ChannelGenerator
from codebook import make_codebook
from phy import sigma2_from_gain, calibrate_noise

def gains(h_ep, cb, cfg):
    T = h_ep.shape[0]
    gs = []
    for t in range(T):
        _, d = cb.quantize(h_ep[t])
        g = np.abs(np.einsum("...t,...t->...", np.conj(h_ep[t]), d)) ** 2
        gs.append(g)
    return cfg.p_rbg * np.stack(gs)          # [T,K,R]

cfg_t2 = Config(debug=True, episode_len_debug=200, num_ue=8,
                pmi_mode="type2_sparse_56bit",
                ue_speed_min=5.0, ue_speed_max=30.0)
cfg_ge = Config(debug=True, episode_len_debug=200, num_ue=8,
                pmi_mode="genie",
                ue_speed_min=5.0, ue_speed_max=30.0)

cb_t2 = make_codebook(cfg_t2)
cb_ge = make_codebook(cfg_ge)

print("== D1: same episode, type2 vs genie sigma2 ==")
rows = []
for ep_seed in (2024, 2025, 2026):
    ch = ChannelGenerator(cfg_t2)
    h_ep = ch.reset(episode_seed=ep_seed)     # identical channel for both worlds
    g_t2 = gains(h_ep, cb_t2, cfg_t2)
    g_ge = gains(h_ep, cb_ge, cfg_ge)
    s_t2 = sigma2_from_gain(g_t2, cfg_t2)
    s_ge = sigma2_from_gain(g_ge, cfg_ge)
    med_snr_t2 = 10*np.log10(np.median(g_t2)/s_t2)
    med_snr_ge = 10*np.log10(np.median(g_ge)/s_ge)
    # genie-world SU SNR if we had kept the type2-calibrated sigma2 (fixed link budget)
    med_snr_ge_fixed = 10*np.log10(np.median(g_ge)/s_t2)
    rows.append((ep_seed, s_t2, s_ge, s_ge/s_t2,
                 10*np.log10(s_ge/s_t2), med_snr_t2, med_snr_ge, med_snr_ge_fixed))
    print(f" ep{ep_seed}: sigma2 t2={s_t2:.4e} genie={s_ge:.4e} "
          f"ratio={s_ge/s_t2:.3f} ({10*np.log10(s_ge/s_t2):+.2f} dB) | "
          f"median SU SNR t2={med_snr_t2:.2f} dB, genie={med_snr_ge:.2f} dB, "
          f"genie@fixed-sigma2={med_snr_ge_fixed:.2f} dB")

print("\n== D2: causality — full-episode vs prefix sigma2 (type2 world) ==")
for ep_seed in (2024, 2025, 2026):
    ch = ChannelGenerator(cfg_t2)
    h_ep = ch.reset(episode_seed=ep_seed)
    g = gains(h_ep, cb_t2, cfg_t2)
    s_full = sigma2_from_gain(g, cfg_t2)
    s_10   = sigma2_from_gain(g[:10], cfg_t2)
    s_half = sigma2_from_gain(g[:100], cfg_t2)
    s_slot0= sigma2_from_gain(g[:1], cfg_t2)
    print(f" ep{ep_seed}: full={s_full:.4e}  slot0-only={s_slot0:.4e} "
          f"({(s_slot0/s_full-1)*100:+.1f}%)  first10={s_10:.4e} "
          f"({(s_10/s_full-1)*100:+.1f}%)  first100={s_half:.4e} "
          f"({(s_half/s_full-1)*100:+.1f}%)")

print("\n== D2b: cross-episode sigma2 spread (how much future info exists at all) ==")
svals = []
for ep_seed in range(2024, 2032):
    ch = ChannelGenerator(cfg_t2)
    h_ep = ch.reset(episode_seed=ep_seed)
    svals.append(sigma2_from_gain(gains(h_ep, cb_t2, cfg_t2), cfg_t2))
svals = np.array(svals)
print(f" sigma2 across 8 episodes: mean={svals.mean():.4e} "
      f"std={svals.std():.4e} rel-std={svals.std()/svals.mean()*100:.1f}% "
      f"min/max ratio={(svals.max()/svals.min()):.2f} "
      f"({10*np.log10(svals.max()/svals.min()):.2f} dB)")
print("\ndone")
