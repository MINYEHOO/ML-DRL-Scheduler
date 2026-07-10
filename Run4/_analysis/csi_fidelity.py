"""How close is the BS's known channel (h_hat) to the genie channel (h_true)?

Measured at the real Run4 operating point (phase4_queue_config: K=32,
p_csi=0.6, speeds U(5,30)). For each active (UE,RBG) sample we compare the
BS's reconstructed h_hat against the true h_true.

Metrics (phase-ambiguity aware -- a PMI codeword is defined up to a global
phase, so a raw ||h_hat - h_true|| is meaningless; we align phase first):
  rho      = |<u_hat, u_true>|        direction cosine in [0,1] (1 = perfect beam)
  rho^2    = direction_corr           what env logs
  chordal  = 1 - rho^2
  magratio = ||h_hat|| / ||h_true||   does CQI-derived gain match true gain
  phNMSE   = (||h_hat||^2 + ||h||^2 - 2||h_hat||||h||*rho) / ||h||^2
             = min-over-global-phase NMSE (direction + magnitude error combined)

Decomposition: ACTUAL (stale fed-back CSI, = quantization + staleness) vs
FRESH (quantize the CURRENT true channel, = quantization only). The gap
between them is the staleness contribution.

SINR/rate cost: SU MRT beamforming -- genie precoder (w = h_true/||h_true||)
vs BS precoder (w = h_hat/||h_hat||), both EVALUATED on h_true. This is what
the imperfect channel knowledge actually costs in dB and bit/s/Hz.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from csi import generate_true_csi
from phy import reconstruct_h_hat
from baselines import SUSCQI

cfg = phase4_queue_config()
SEEDS = [10000, 10004, 10010]
SLOTS = 200

def fidelity(h_hat, h_true):
    """Per-(UE,RBG) fidelity arrays. h_*: [...,M] complex."""
    nh = np.linalg.norm(h_hat, axis=-1)
    nt = np.linalg.norm(h_true, axis=-1)
    ok = nt > 1e-12
    ip = np.abs(np.einsum("...t,...t->...", np.conj(h_hat), h_true))
    rho = np.where(ok & (nh > 1e-12), ip / np.maximum(nh * nt, 1e-12), 0.0)
    magr = np.where(ok, nh / np.maximum(nt, 1e-12), 0.0)
    phnmse = np.where(ok, (nh**2 + nt**2 - 2*nh*nt*rho) / np.maximum(nt**2, 1e-12), 0.0)
    return rho[ok], magr[ok], phnmse[ok]

def su_sinr_db(w, h_true, p_rbg, noise):
    """SU SINR (dB) for unit-norm precoder w evaluated on h_true."""
    g = np.abs(np.einsum("...t,...t->...", h_true, w))**2 * p_rbg
    return 10*np.log10(np.maximum(g, 1e-30) / noise)

RHO_a, MAG_a, NM_a = [], [], []
RHO_f, NM_f = [], []
DB_genie, DB_bs = [], []
sch = SUSCQI()

for seed in SEEDS:
    env = SchedulerEnv(cfg)
    env.reset(episode_idx=seed)
    na = env.traffic.n_active
    pr, nv = cfg.p_rbg, env.noise_var
    for t in range(SLOTS):
        ht = env.h_true_slot[:na]          # [na,R,M] genie
        hh = env.h_hat_slot[:na]           # [na,R,M] BS estimate (stale fed-back)
        # actual fidelity (quantization + staleness)
        r, m, n = fidelity(hh, ht); RHO_a += [r]; MAG_a += [m]; NM_a += [n]
        # fresh fidelity (quantize CURRENT true channel = quantization only)
        _, dir_f, cqi_f = generate_true_csi(ht, env.codebook, pr, nv)
        hh_f = reconstruct_h_hat(dir_f, cqi_f, pr, nv)
        rf, _, nf = fidelity(hh_f, ht); RHO_f += [rf]; NM_f += [nf]
        # SINR cost: genie MRT vs BS-estimate MRT, both on h_true
        wt = ht / np.maximum(np.linalg.norm(ht, axis=-1, keepdims=True), 1e-12)
        wh = hh / np.maximum(np.linalg.norm(hh, axis=-1, keepdims=True), 1e-12)
        DB_genie += [su_sinr_db(wt, ht, pr, nv).ravel()]
        DB_bs += [su_sinr_db(wh, ht, pr, nv).ravel()]
        env.step(sch.schedule(env))
    print(f"seed {seed} (n_active {na}) done", flush=True)

def stat(a): a = np.concatenate(a); return a.mean(), np.median(a), np.percentile(a, 5)
ra, rma, na_ = stat(RHO_a), stat(MAG_a), stat(NM_a)
rf, nf = stat(RHO_f), stat(NM_f)
dbg = np.concatenate(DB_genie); dbb = np.concatenate(DB_bs)

print("\n=== BS h_hat vs genie h_true fidelity (Run4 op point, p_csi=0.6) ===")
print(f"{'metric':26s} {'mean':>8s} {'median':>8s} {'5th pct':>8s}")
print(f"{'direction cosine rho':26s} {ra[0]:8.4f} {ra[1]:8.4f} {ra[2]:8.4f}")
print(f"{'  rho^2 (=direction_corr)':26s} {ra[0]**2:8.4f} {ra[1]**2:8.4f} {ra[2]**2:8.4f}")
print(f"{'chordal dist (1-rho^2)':26s} {1-ra[0]**2:8.4f}")
print(f"{'magnitude ratio':26s} {rma[0]:8.4f} {rma[1]:8.4f} {rma[2]:8.4f}")
print(f"{'phase-aligned NMSE':26s} {na_[0]:8.4f} {na_[1]:8.4f}")
print(f"\n--- decomposition (direction cosine rho) ---")
print(f"{'ACTUAL (quant+stale)':26s} rho={ra[0]:.4f}  phNMSE={na_[0]:.4f}")
print(f"{'FRESH (quant only)':26s} rho={rf[0]:.4f}  phNMSE={nf[0]:.4f}")
print(f"{'staleness cost':26s} d_rho={rf[0]-ra[0]:.4f}  d_phNMSE={na_[0]-nf[0]:.4f}")
print(f"\n--- SU beamforming SINR cost (genie MRT vs BS-estimate MRT on h_true) ---")
print(f"genie  SINR: mean {dbg.mean():6.2f} dB  median {np.median(dbg):6.2f} dB")
print(f"BS-est SINR: mean {dbb.mean():6.2f} dB  median {np.median(dbb):6.2f} dB")
print(f"loss from imperfect CSI: mean {dbg.mean()-dbb.mean():5.2f} dB "
      f"= {np.log2(1+10**(dbg.mean()/10))-np.log2(1+10**(dbb.mean()/10)):.3f} bit/s/Hz SU-rate")
