"""MU physics probe: why does CQI-greedy MU packing win?

Uses the exact MixedSpeed_L2 config (K=32, U(5,30) km/h, p_csi=0.6) on an
eval-family seed. Walks a zero-allocation episode (traffic irrelevant) and, at
sampled slots, compares per-RBG sum-SE for:
  - greedy top-m CQI packing (m=1..4)   [what CQI-greedy does]
  - SUS selection at thresholds 0.5 / 0.8 (CQI metric = SUS-CQI baseline)
  - random-m packing
under three precoders/eval:
  - RZF from fed-back h_hat  (the env's actual model)
  - RZF from true channel     (genie CSI -> isolates CSI-error effect)
  - CBF/matched-filter from h_hat (no nulling -> the user's mental model)
Also records pairwise direction correlations and interference-to-noise ratios.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config          # noqa: E402
from env import SchedulerEnv       # noqa: E402

t0 = time.time()

cj = json.load(open("/home/MYH/ML_DRL_Scheduler/Run3/MixedSpeed_L2/config.json"))
fields = {}
for k, v in cj.items():
    if k in Config.__dataclass_fields__:
        fields[k] = tuple(v) if isinstance(v, list) else v
cfg = Config(**fields)
cfg.debug = True
cfg.episode_len_debug = 220
print(f"config: K={cfg.num_ue} R={cfg.num_rbg} L={cfg.l_max} "
      f"p_csi={cfg.p_csi} speeds=U({cfg.ue_speed_min},{cfg.ue_speed_max}) "
      f"target_snr={cfg.target_snr_db}dB p_rbg={cfg.p_rbg}")

env = SchedulerEnv(cfg)
env.reset(10000)                       # eval-family seed
print(f"episode ready ({time.time()-t0:.0f}s). ue_speeds sample: "
      f"{np.round(env.ue_speeds_kmh[:8],1)}")

P = cfg.p_rbg


def sinr_general(h_true_rows, W, noise_var):
    eff = h_true_rows @ W
    pw = np.abs(eff) ** 2
    des = np.diag(pw)
    intf = pw.sum(axis=1) - des
    return des / (intf + noise_var), intf


def rzf_w(h_hat_rows, alpha):
    m = h_hat_rows.shape[0]
    g = h_hat_rows @ h_hat_rows.conj().T + alpha * np.eye(m)
    v = np.linalg.solve(g, h_hat_rows).conj().T
    v /= np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-30)
    return np.sqrt(P / m) * v


def cbf_w(h_hat_rows):
    m = h_hat_rows.shape[0]
    v = h_hat_rows.conj().T.copy()
    v /= np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-30)
    return np.sqrt(P / m) * v


def sus_select(order, C2, th, mmax=4):
    sel = [order[0]]
    for cand in order[1:]:
        if len(sel) >= mmax:
            break
        if 1.0 - max(C2[cand, s] for s in sel) >= th:
            sel.append(cand)
    return np.array(sel)


sample_slots = {3, 30, 60, 100, 140, 180, 210}
zero = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
rng = np.random.default_rng(7)
rows = []
pair_corr_true_all = []

done = False
while not done:
    t = env.slot
    if t in sample_slots:
        h_true, h_hat = env.h_true_slot, env.h_hat_slot
        cqi, dirs, age = env.csi.cqi_fb, env.csi.direction_fb, env.csi.age
        nv = env.noise_var
        # true unit directions for ground-truth correlation stats
        tn = h_true / np.maximum(np.linalg.norm(h_true, axis=-1,
                                                keepdims=True), 1e-30)
        for r in range(cfg.num_rbg):
            D = dirs[:, r, :]
            C2 = np.abs(D @ D.conj().T) ** 2          # fed-back corr^2
            Dt = tn[:, r, :]
            C2t = np.abs(Dt @ Dt.conj().T) ** 2       # true corr^2
            iu = np.triu_indices(cfg.num_ue, 1)
            pair_corr_true_all.append(C2t[iu])
            order = np.argsort(-cqi[:, r])

            def record(mode, sel):
                m = len(sel)
                ht, hh = h_true[sel, r], h_hat[sel, r]
                s_rzf, i_rzf = sinr_general(ht, rzf_w(hh, nv), nv)
                s_gen, _ = sinr_general(ht, rzf_w(ht, nv), nv)
                s_cbf, i_cbf = sinr_general(ht, cbf_w(hh), nv)
                mc = 0.0 if m == 1 else max(
                    C2t[a, b] for ii, a in enumerate(sel) for b in sel[:ii])
                rows.append(dict(
                    t=t, r=r, mode=mode, m=m,
                    se_rzf=float(np.log2(1 + s_rzf).sum()),
                    se_genie=float(np.log2(1 + s_gen).sum()),
                    se_cbf=float(np.log2(1 + s_cbf).sum()),
                    inr_rzf=float((i_rzf / nv).mean()),
                    inr_cbf=float((i_cbf / nv).mean()),
                    maxcorr_true=float(mc),
                    mean_age=float(age[sel].mean()),
                    mean_speed=float(env.ue_speeds_kmh[sel].mean()),
                ))

            for m in (1, 2, 3, 4):
                record("greedy", order[:m])
            record("sus0.5", sus_select(order, C2, 0.5))
            record("sus0.8", sus_select(order, C2, 0.8))
            record("random4", rng.choice(cfg.num_ue, 4, replace=False))
    _, _, done, _ = env.step(zero)

import collections  # noqa: E402

print(f"\nprobe done in {time.time()-t0:.0f}s, {len(rows)} records")
pc = np.concatenate(pair_corr_true_all)
print(f"\n=== TRUE pairwise direction |corr|^2 across UE pairs ===")
print(f"  mean {pc.mean():.3f}  median {np.median(pc):.3f}  "
      f"p90 {np.percentile(pc,90):.3f}  p99 {np.percentile(pc,99):.3f}  "
      f"frac>0.5: {(pc>0.5).mean():.3%}  frac>0.2: {(pc>0.2).mean():.3%}")

groups = collections.defaultdict(list)
for row in rows:
    groups[(row["mode"], row["m"])].append(row)

print(f"\n=== per-RBG sum-SE [bit/s/Hz] and interference (mean over "
      f"{len(sample_slots)} slots x {cfg.num_rbg} RBGs) ===")
print(f"{'mode':10s}{'m':>3s} {'SE_rzf':>7s} {'SE_genie':>9s} {'SE_cbf':>7s}"
      f" {'INR_rzf':>8s} {'INR_cbf':>8s} {'maxcorr':>8s} {'age':>5s}")
order_keys = sorted(groups.keys(),
                    key=lambda k: (k[0] not in ("greedy",), k))
for key in order_keys:
    g = groups[key]
    f = lambda name: np.mean([x[name] for x in g])   # noqa: E731
    print(f"{key[0]:10s}{key[1]:3d} {f('se_rzf'):7.2f} {f('se_genie'):9.2f} "
          f"{f('se_cbf'):7.2f} {f('inr_rzf'):8.2f} {f('inr_cbf'):8.2f} "
          f"{f('maxcorr_true'):8.3f} {f('mean_age'):5.1f}  (n={len(g)})")

# stale vs fresh split for greedy m=4
g4 = groups[("greedy", 4)]
fresh = [x for x in g4 if x["mean_age"] <= 1]
stale = [x for x in g4 if x["mean_age"] > 1]
print(f"\n=== greedy m=4: CSI freshness effect ===")
for name, sub in (("fresh(age<=1)", fresh), ("stale(age>1)", stale)):
    if sub:
        print(f"  {name:14s} n={len(sub):3d}  "
              f"SE_rzf {np.mean([x['se_rzf'] for x in sub]):.2f}  "
              f"INR {np.mean([x['inr_rzf'] for x in sub]):.2f}  "
              f"speed {np.mean([x['mean_speed'] for x in sub]):.0f}km/h")

# SUS set sizes
for th in ("sus0.5", "sus0.8"):
    ms = [k[1] for k in groups if k[0] == th]
    sizes = [x["m"] for k in groups if k[0] == th for x in groups[k]]
    print(f"\n{th}: selected-set-size mean {np.mean(sizes):.2f} "
          f"(dist: {np.bincount(sizes, minlength=5)[1:]})")
