"""How well does the BS know the channel, as a function of CSI age?

For the L2c world (hetero, U(5,40) km/h, fc=3.5GHz, slot=0.5ms, Nt=32,
Type-2 56-bit PMI): for every slot t and age a in 0..5, compare the CSI
report generated at slot t-a (quantized direction + CQI, exactly what the
feedback buffer would hold) against the TRUE channel at slot t.

Metrics per (age, speed-bin):
  * align  = |h_true(t)^H d_fb(t-a)|^2 / ||h_true(t)||^2   (cos^2, 1.0=perfect)
             -> age 0 shows the pure PMI-quantization floor
  * dB     = 10 log10(align)  (beamforming-gain loss vs perfect alignment)
  * cqi_bias = CQI_fb(t-a) - log2(1 + p*|h(t)^H d_fb|^2/sigma^2)
             (bits/s/Hz the stale report OVER-promises about right now)

Age occurrence under p_csi=0.6 (geometric): P(a)=0.6*0.4^a ->
60%, 24%, 9.6%, 3.8%, 1.5%, 0.6% for a=0..5. Age axis chosen to 5 to
mirror the 5-attempt retx-drop rule (user request 2026-07-20).
"""
import os, sys, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import json
import numpy as np
from config import Config
from env import SchedulerEnv

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/csi_age_accuracy.csv"
SEEDS = [10000, 10001, 10002]
AGES = range(6)
BINS = [(5.0, 15.0), (15.0, 30.0), (30.0, 40.0)]

raw = json.load(open("Run4/MixedSpeed_L2c/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
env = SchedulerEnv(cfg)

# accum[(age, bin)] -> [sum_align, sum_bias, n]   (bin=-1 is "all speeds")
accum = {(a, b): [0.0, 0.0, 0] for a in AGES for b in list(range(len(BINS))) + [-1]}

for s in SEEDS:
    env.reset(episode_idx=s)
    ep = env._episode_csi                      # direction [T,K,R,Nt], cqi [T,K,R]
    sig2, T = ep.sigma2, ep.direction.shape[0]
    speeds = env.ue_speeds_kmh                 # [K]
    ue_bin = np.full(len(speeds), -2, dtype=int)
    for i, (lo, hi) in enumerate(BINS):
        ue_bin[(speeds >= lo) & (speeds <= hi)] = i
    h_ep = np.stack([env.channel.get_channel(t) for t in range(T)])  # [T,K,R,Nt]
    h_norm2 = np.einsum("tkrn,tkrn->tkr", np.conj(h_ep), h_ep).real
    for a in AGES:
        # report born at t-a, judged against the true channel at t
        g = np.abs(np.einsum("tkrn,tkrn->tkr",
                             np.conj(h_ep[a:]), ep.direction[:T - a])) ** 2
        align = g / np.maximum(h_norm2[a:], 1e-30)                 # [T-a,K,R]
        actual_se = np.log2(1.0 + cfg.p_rbg * g / sig2)
        bias = ep.cqi[:T - a] - actual_se
        for b in list(range(len(BINS))) + [-1]:
            m = np.ones(len(speeds), bool) if b == -1 else (ue_bin == b)
            if not m.any():
                continue
            acc = accum[(a, b)]
            acc[0] += float(align[:, m].sum())
            acc[1] += float(bias[:, m].sum())
            acc[2] += int(align[:, m].size)
    print(f"seed {s} done (speeds {speeds.min():.0f}-{speeds.max():.0f} km/h)", flush=True)

w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["age", "speed_bin", "mean_align", "align_db", "mean_cqi_bias_bps"])
LBL = {-1: "all(5-40)", 0: "5-15", 1: "15-30", 2: "30-40"}
print(f"\n=== BS channel knowledge vs CSI age (L2c world, {len(SEEDS)} episodes, "
      f"slot={cfg.slot_duration*1e3:.1f}ms, Nt={cfg.num_bs_ant}) ===")
print(f"{'age':>3s} {'P(age)':>7s} | " + " | ".join(
    f"{LBL[b]:>20s}" for b in (-1, 0, 1, 2)))
print(f"{'':>3s} {'':>7s} | " + " | ".join(f"{'cos2    dB   dCQI':>20s}" for _ in range(4)))
for a in AGES:
    cells = []
    for b in (-1, 0, 1, 2):
        sa, sb, n = accum[(a, b)]
        al, bi = sa / n, sb / n
        db = 10 * np.log10(al)
        cells.append(f"{al:5.3f} {db:5.2f} {bi:6.3f}")
        w.writerow([a, LBL[b], round(al, 5), round(db, 3), round(bi, 4)])
    p_age = 0.6 * 0.4 ** a
    print(f"{a:3d} {p_age:7.1%} | " + " | ".join(f"{c:>20s}" for c in cells))
print("\nsaved ->", OUT)
