"""Critic-ceiling probe: how much of the return variance is predictable
from state AT ALL, and does the proposed rich feature set unlock it?

Design:
  - freeze MixedSpeed_L2 best.pt (update 29), stochastic decode (training-like
    state distribution), 30 fresh episodes (seeds 20000+, unseen by training
    [0..139] and eval [10000-10002])
  - per slot record: 6-scalar baseline features (what build_value_input's
    global scalars carry), rich handcrafted features (~30 dims: deadline
    histogram, backlog-weighted urgency, retx-grid structure, CQI/age
    aggregates, slot phase), reward + components
  - offline: discounted MC returns (gamma=0.99), drop last 300 slots
    (truncation tail), fit ridge + small MLP, R^2 on held-out EPISODES
  - per-component returns (short-bits / completion / miss-penalty) regressed
    separately on rich features -> which component is predictable
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
import torch

torch.set_num_threads(4)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config          # noqa: E402
from env import SchedulerEnv       # noqa: E402
from policy import ActorCritic     # noqa: E402

RUN = "/home/MYH/ML_DRL_Scheduler/Run3/MixedSpeed_L2"
N_EP, SEED0, GAMMA, TAIL = 30, 20000, 0.99, 300
N_TRAIN_EP = 24

cj = json.load(open(f"{RUN}/config.json"))
fields = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in cj.items() if k in Config.__dataclass_fields__}
cfg = Config(**fields)
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
ac.load_state_dict(ck["model"])
ac.eval()
print(f"policy: best.pt update {ck['update']} (eval {ck.get('eval_reward'):.0f}), "
      f"K={cfg.num_ue}, episode_len={cfg.episode_len}", flush=True)

K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
EPS = cfg.b_tx_epsilon
D_BINS = [(0, 1.5), (1.5, 2.5), (2.5, 4.5), (4.5, 8.5), (8.5, 1e9)]


def feats_baseline(obs):
    """The 6 global scalars the current value input carries."""
    act = obs["active"].astype(np.float64)
    n_act = act.sum()
    age = np.minimum(obs["age"], cfg.age_norm_max)
    md = (obs["deadline"] * act).sum() / max(n_act, 1) / cfg.deadline_max
    mb = (obs["backlog"] * act).sum() / max(n_act, 1) / cfg.b_norm
    has_data = act.astype(bool) & (obs["uncommitted"] > EPS)
    return np.array([n_act / K, obs["fixed_mask"].sum() / (R * L),
                     age.mean() / cfg.age_norm_max, md, mb,
                     has_data.mean()])


def feats_rich(obs, slot):
    act = obs["active"].astype(bool)
    dl, bk, unc = obs["deadline"], obs["backlog"], obs["uncommitted"]
    f = []
    # deadline histogram (counts + backlog-weighted), active packets only
    for lo, hi in D_BINS:
        m = act & (dl > lo) & (dl <= hi)
        f.append(m.sum() / K)
        f.append(bk[m].sum() / (K * cfg.b_norm))
    # backlog totals
    f.append(unc.sum() / (K * cfg.b_norm))
    f.append(bk.sum() / (K * cfg.b_norm))
    f.append((act & (unc > EPS)).sum() / K)
    # retx-grid structure
    per_rbg_fixed = obs["fixed_mask"].sum(axis=1)                # [R]
    for c in range(L + 1):
        f.append((per_rbg_fixed == c).sum() / R)
    f.append(obs["fixed_mask"].sum() / (R * L))
    # CQI / age aggregates
    best_cqi = obs["cqi_fb"].max(axis=1)                          # [K]
    f.append(best_cqi[act].mean() / 8.0 if act.any() else 0.0)
    f.append(best_cqi[act].max() / 8.0 if act.any() else 0.0)
    f.append(best_cqi.mean() / 8.0)
    age = np.minimum(obs["age"], cfg.age_norm_max) / cfg.age_norm_max
    f.append(age[act].mean() if act.any() else 0.0)
    wa = (age * bk).sum() / max(bk.sum(), 1e-9)                   # backlog-wt age
    f.append(wa)
    # fairness / phase / level
    f.append(obs["avg_throughput"].mean() / cfg.b_norm)
    f.append(act.sum() / K)
    f.append(slot / cfg.episode_len)
    return np.array(f, dtype=np.float64)


t0 = time.time()
ep_data = []
for e in range(N_EP):
    obs = env.reset(SEED0 + e)
    Xb, Xr, rew, r_s, r_c, r_m = [], [], [], [], [], []
    done = False
    while not done:
        Xb.append(feats_baseline(obs))
        Xr.append(feats_rich(obs, env.slot))
        with torch.no_grad():
            out = ac.decode(obs, deterministic=False)
        obs, r, done, info = env.step(out["action_sequence"])
        rew.append(r)
        r_s.append(cfg.lambda_s * info["reward_short"])
        r_c.append(cfg.lambda_c * info["n_comp"])
        r_m.append(r - cfg.lambda_s * info["reward_short"]
                   - cfg.lambda_c * info["n_comp"])   # = -lambda_m * misses
    ep_data.append((np.array(Xb), np.array(Xr),
                    np.array(rew), np.array(r_s), np.array(r_c),
                    np.array(r_m)))
    if (e + 1) % 5 == 0:
        print(f"  rollout {e+1}/{N_EP}  ep_rew {np.sum(rew):8.0f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


def disc_return(r):
    g = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + GAMMA * acc
        g[t] = acc
    return g


def stack(idx, col, ret_col):
    X = np.concatenate([ep_data[i][col][:-TAIL] for i in idx])
    G = np.concatenate([disc_return(ep_data[i][ret_col])[:-TAIL] for i in idx])
    return X, G


def r2(y, p):
    return 1.0 - np.mean((y - p) ** 2) / np.var(y)


def fit_ridge(Xtr, ytr, Xte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    w = np.linalg.solve(Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]),
                        Xtr.T @ ytr)
    return Xte @ w, w


def fit_mlp(Xtr, ytr, Xte, hidden=128, epochs=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    gm, gs = ytr.mean(), ytr.std() + 1e-9
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((ytr - gm) / gs, dtype=torch.float32)
    Xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 1024):
            i = perm[s:s + 1024]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[i]).squeeze(-1), yt[i])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(Xe).squeeze(-1).numpy() * gs + gm


tr_idx, te_idx = range(N_TRAIN_EP), range(N_TRAIN_EP, N_EP)
print(f"\nrollouts done ({time.time()-t0:.0f}s). samples: "
      f"train {(1000-TAIL)*N_TRAIN_EP}, test {(1000-TAIL)*(N_EP-N_TRAIN_EP)}")

RET_COLS = {"TOTAL": 2, "short(bits)": 3, "completion": 4, "miss(-2x)": 5}
print(f"\n=== held-out R^2 (== achievable-ev proxy), gamma={GAMMA}, "
      f"tail {TAIL} dropped ===")
print(f"{'return target':16s} {'6-scalar ridge':>15s} {'rich ridge':>11s} "
      f"{'rich MLP':>9s}")
w_saved = None
for name, rc in RET_COLS.items():
    Xb_tr, G_tr = stack(tr_idx, 0, rc)
    Xb_te, G_te = stack(te_idx, 0, rc)
    Xr_tr, _ = stack(tr_idx, 1, rc)
    Xr_te, _ = stack(te_idx, 1, rc)
    pb, _ = fit_ridge(Xb_tr, G_tr, Xb_te)
    pr, w = fit_ridge(Xr_tr, G_tr, Xr_te)
    pm = fit_mlp(Xr_tr, G_tr, Xr_te)
    if name == "TOTAL":
        w_saved = w
    print(f"{name:16s} {r2(G_te, pb):15.3f} {r2(G_te, pr):11.3f} "
          f"{r2(G_te, pm):9.3f}")

FEAT_NAMES = ([f"dl_cnt{i}" for i in range(5) for _ in [0]] and
              [x for i in range(5) for x in (f"dlcnt_b{i}", f"dlbk_b{i}")]
              + ["unc_tot", "bk_tot", "n_hasdata",
                 "rbg_fix0", "rbg_fix1", "rbg_fix2", "rbg_fix3", "rbg_fix4",
                 "fix_frac", "cqi_mean_act", "cqi_max_act", "cqi_mean_all",
                 "age_mean_act", "age_bkwt", "avgthr", "n_active", "slot_phase"])
imp = np.abs(w_saved[:-1])
top = np.argsort(-imp)[:10]
print("\ntop-10 |ridge coef| (standardized), TOTAL return:")
for i in top:
    print(f"  {FEAT_NAMES[i]:14s} {w_saved[i]:+9.1f}")
print(f"\ntotal time {time.time()-t0:.0f}s")
