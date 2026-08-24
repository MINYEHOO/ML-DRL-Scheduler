"""Critic diagnostic: what does the logged explained_variance actually measure,
and would per-episode latent parameters help the critic AT ALL? (2026-08-21)

WHY THIS EXISTS
---------------
An external review proposed adding privileged per-episode latents (K_act, p_a,
per-UE speed) to the critic, predicting explained_variance 0.5 -> 0.7. That
prediction is provably unreachable through the channel it describes:

  ppo.py:79   returns = advantages + values
  ppo.py:284  values_arr = np.array([t.value for t in trajs])
  ppo.py:286  explained_var = 1 - np.var(returns - values_arr)/np.var(returns)

so the EV residual IS the GAE advantage, and EV = 1 - Var(A)/Var(A+V) is
invariant to a constant shift in V.  One PPO update's batch is ONE episode
(train_phase2.py:659, collect_rollout(..., episode_idx=update)), so K_act,
p_a and the speed distribution are CONSTANT across the batch.  Advantages are
additionally mean-centred per rollout (ppo.py:136-140), so a constant bias in
V is removed from the policy gradient too.

BUT both the review and the first response then stopped, and both were wrong to.
The value loss

  ppo.py:208  value_loss = ((rep["value"] - ret) / ret_sigma).pow(2)

is a per-slot regression that DOES see a per-episode level error, and
ppo.py:223 backprops a single total_loss with no detach, so that error reaches
the SHARED ENCODER.  Latents cannot move EV but can still move value_loss.
Neither quantity has ever been measured.  This probe measures them.

WHAT IT COMPUTES  (frozen best.pt, stochastic decode = training state
distribution, fresh unused seed band)
--------------------------------------------------------------------
  variance decomposition   Var(G_MC) = Var_between(episode means) + Var_within
                           -> how big a slice the logged within-episode EV is

  a1  trained critic, LOGGED definition (GAE lambda=0.95 residual, all slots)
        sanity check: must land near the training log's EV
  a2  trained critic, MC definition (lambda=1), within-episode and pooled
        the comparable form
  b*  episode phase t/T alone (cubic, per episode)
        replaces the review's proposed constant baseline, which is
        DEFINITIONALLY ZERO (Var(G - mean(G)) == Var(G)) and carries no
        information.  Answers "is within-episode EV structurally near zero?"
  c1  ridge on the critic's ACTUAL 167-d value input, held-out BY EPISODE
        honest ceiling given what the critic already sees
  c2  ridge on the same 167-d input PLUS the latents, held-out BY EPISODE
        ** c2 - c1 IS THE ANSWER FOR THE PRIVILEGED-CRITIC PROPOSAL **
  d   per-episode value bias  mean_t(G) - mean_t(V), and the share of the
        critic's total MSE attributable to that level offset
        ** the quantity latents could actually fix, i.e. the value_loss channel **
        Reported against BOTH targets, because they answer different questions:
          - vs the GAE return (lambda=0.95): this is literally what ppo.py:208
            regresses on, so its bias share IS the value_loss channel;
          - vs the MC return (lambda=1): the value function's true target,
            i.e. how wrong the critic is about the world.

DECISION RULE
-------------
  c2 - c1 ~ 0  AND  bias share small   -> close the proposal permanently
  c2 - c1 > 0  OR   bias share large   -> a reduced version is worth testing
  a2 ~ c1                              -> ceiling is structural; stop quoting
                                          EV as a critic-quality diagnostic

SAFETY / ISOLATION
------------------
Read-only w.r.t. the repository: this file is new, no root *.py is touched, so
the three live seed-replicate runs' pin_check (Run4/_wrap_*_s{3024,4024,5024}.sh)
stays green.  Runs on GPU 0 (3/4/5 are the replicates).  Seeds 60000-60039 are
a previously unused band (train 2024/3024/4024/5024, eval 10000-10040, final
20000-20099, OOD 30000-30100, pilot 40000-40008, beta 50000/70000).
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "6"

import csv                                                      # noqa: E402
import json                                                     # noqa: E402
import time                                                     # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

torch.set_num_threads(6)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                                       # noqa: E402
from env import SchedulerEnv                                    # noqa: E402
from policy import ActorCritic, build_value_input, obs_to_tensors   # noqa: E402
from ppo import compute_gae, SlotTrajectory                     # noqa: E402

RUN = "/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4"
OUTDIR = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis"
SEED0 = 60000
N_EP = int(os.environ.get("PROBE_N_EP", 40))        # smoke-test override
N_TRAIN = int(os.environ.get("PROBE_N_TRAIN", 24))  # remaining episodes after
N_INNER = int(os.environ.get("PROBE_N_INNER", 6))   # these two are the test set
TAIL = int(os.environ.get("PROBE_TAIL", 300))       # truncation tail to drop
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

log_lines = []


def say(s=""):
    print(s, flush=True)
    log_lines.append(s)


# ---------------------------------------------------------------------------
# rollout collection: exactly the training path (stochastic decode)
# ---------------------------------------------------------------------------
def collect(env, ac, cfg, seed):
    """One episode. Returns per-slot V, reward, the 167-d value input, and the
    episode's latent parameters."""
    obs = env.reset(seed)
    vals, rews, feats, dones = [], [], [], []
    done = False
    while not done:
        out = ac.decode(obs, deterministic=False)     # training decode
        with torch.no_grad():
            obs_t = obs_to_tensors(obs, ac.device)
            e = ac.encode(obs_t)
            vi = build_value_input(e, obs_t, cfg, obs["fixed_mask"],
                                   obs["slot"])
        feats.append(vi.detach().cpu().numpy().astype(np.float64))
        vals.append(float(out["value"]))
        next_obs, reward, done, _ = env.step(out["action_sequence"])
        rews.append(float(reward))
        dones.append(bool(done))
        obs = next_obs                       # identical to ppo.collect_rollout
    with torch.no_grad():
        last_value = float(ac.state_value(obs))
    sp = np.asarray(env.ue_speeds_kmh, dtype=np.float64)
    latent = np.array([
        float(env.traffic.n_active) / cfg.num_ue,
        float(env.traffic.p_arrival_ep if env.traffic.p_arrival_ep
              is not None else 0.0),
        sp.mean() / 40.0,
        sp.std() / 40.0,
    ])
    return (np.array(vals), np.array(rews), np.array(feats),
            np.array(dones), last_value, latent)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def ev_var(y, yhat):
    """1 - Var(y - yhat)/Var(y): the definition ppo.py:286 uses (shift-blind)."""
    vy = np.var(y)
    return float(1.0 - np.var(y - yhat) / max(vy, 1e-12))


def r2_mse(y, yhat):
    """1 - MSE/Var(y): penalizes a constant offset, unlike ev_var."""
    vy = np.var(y)
    return float(1.0 - np.mean((y - yhat) ** 2) / max(vy, 1e-12))


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    A = Z.T @ Z
    idx = np.arange(A.shape[0] - 1)          # do not penalize the intercept
    A[idx, idx] += alpha
    w = np.linalg.solve(A, Z.T @ y)
    return mu, sd, w


def ridge_pred(model, X):
    mu, sd, w = model
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


def fit_select_eval(Xtr, ytr, Xin, yin, Xte, yte, ep_te):
    """alpha chosen on the inner-validation EPISODES, scored on the test ones."""
    best = None
    for a in ALPHAS:
        m = ridge_fit(Xtr, ytr, a)
        s = r2_mse(yin, ridge_pred(m, Xin))
        if best is None or s > best[0]:
            best = (s, a, m)
    _, alpha, m = best
    pred = ridge_pred(m, Xte)
    within = [ev_var(yte[ep_te == g], pred[ep_te == g])
              for g in np.unique(ep_te)]
    return dict(alpha=alpha,
                pooled_ev=ev_var(yte, pred), pooled_r2=r2_mse(yte, pred),
                within_ev=float(np.mean(within)))


# ---------------------------------------------------------------------------
def main():
    raw = json.load(open(f"{RUN}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    cfg = Config(**raw)
    assert cfg.cqi_mode == "nr4bit" and cfg.ppo_critic_v2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
    ac = ActorCritic(cfg)
    ac.load_state_dict(ck["model"], strict=True)
    ac.to(dev).eval()
    say("=" * 78)
    say("CRITIC EV DECOMPOSITION")
    say("=" * 78)
    say(f"policy   : {RUN}/ckpt/best.pt @update {ck['update']}")
    say(f"device   : {dev}   ret_mean={float(ac.ret_mean):.2f} "
        f"ret_std={float(ac.ret_std):.2f}")
    say(f"episodes : seeds {SEED0}-{SEED0 + N_EP - 1} (n={N_EP}), "
        f"stochastic decode")
    say(f"gamma={cfg.ppo_gamma} lambda={cfg.ppo_gae_lambda} "
        f"episode_len={cfg.episode_len}  MC tail dropped={TAIL}")
    say()

    env = SchedulerEnv(cfg)
    V, G_MC, G_GAE, F, L, EP, PH = [], [], [], [], [], [], []
    ep_rows = []
    t0 = time.time()
    for i in range(N_EP):
        seed = SEED0 + i
        v, r, f, d, lv, lat = collect(env, ac, cfg, seed)
        T = len(v)
        # GAE returns exactly as ppo_update sees them (lambda = 0.95)
        trajs = [SlotTrajectory(None, None, None, None, 0.0, [], 0.0, 0,
                                float(v[t]), float(r[t]), bool(d[t]))
                 for t in range(T)]
        _, ret_gae = compute_gae(trajs, cfg.ppo_gamma, cfg.ppo_gae_lambda,
                                 last_value=lv)
        # Monte-Carlo return with the same truncation bootstrap (lambda = 1)
        g = np.zeros(T)
        acc = lv
        for t in reversed(range(T)):
            acc = r[t] + cfg.ppo_gamma * acc
            g[t] = acc
        V.append(v); G_MC.append(g); G_GAE.append(ret_gae)
        F.append(f); L.append(np.repeat(lat[None, :], T, axis=0))
        EP.append(np.full(T, i)); PH.append(np.arange(T) / T)
        ep_rows.append(dict(
            seed=seed, K_act=int(round(lat[0] * cfg.num_ue)), p_a=lat[1],
            speed_mean=lat[2] * 40.0,
            ev_gae_all=ev_var(ret_gae, v),
            ev_mc_all=ev_var(g, v),
            bias=float(g[:T - TAIL].mean() - v[:T - TAIL].mean()),
            g_mean=float(g[:T - TAIL].mean()), g_sd=float(g[:T - TAIL].std()),
        ))
        say(f"  ep {i:2d} seed {seed}  K_act={ep_rows[-1]['K_act']:2d} "
            f"p_a={lat[1]:.3f}  EV(gae)={ep_rows[-1]['ev_gae_all']:+.3f} "
            f"EV(mc)={ep_rows[-1]['ev_mc_all']:+.3f} "
            f"bias={ep_rows[-1]['bias']:+8.1f}   "
            f"[{(time.time() - t0) / 60:.1f} min]")

    # ---- flatten, with and without the truncation tail --------------------
    def cat(xs, tail):
        return np.concatenate([x[:len(x) - tail] for x in xs])

    v_all, g_all = cat(V, 0), cat(G_MC, 0)
    rg_all = cat(G_GAE, 0)
    ep_all = cat(EP, 0)
    v_c, g_c = cat(V, TAIL), cat(G_MC, TAIL)
    f_c, l_c, ep_c, ph_c = cat(F, TAIL), cat(L, TAIL), cat(EP, TAIL), cat(PH, TAIL)

    say()
    say("=" * 78)
    say("VARIANCE DECOMPOSITION of the MC return  (tail dropped)")
    say("=" * 78)
    means = np.array([g_c[ep_c == i].mean() for i in range(N_EP)])
    wvars = np.array([g_c[ep_c == i].var() for i in range(N_EP)])
    vb, vw = float(means.var()), float(wvars.mean())
    say(f"  Var_between (episode means) = {vb:12.1f}   "
        f"{100 * vb / (vb + vw):5.1f} %")
    say(f"  Var_within  (mean of within) = {vw:12.1f}   "
        f"{100 * vw / (vb + vw):5.1f} %")
    say(f"  episode means: mean {means.mean():.1f}  sd {means.std():.1f}  "
        f"range [{means.min():.1f}, {means.max():.1f}]")
    say("  -> the logged EV is a WITHIN-episode quantity; it can only ever")
    say(f"     speak to the {100 * vw / (vb + vw):.1f}% slice.")

    say()
    say("=" * 78)
    say("a1/a2  TRAINED CRITIC")
    say("=" * 78)
    w_gae = [ev_var(G_GAE[i], V[i]) for i in range(N_EP)]
    w_mc = [ev_var(G_MC[i][:-TAIL], V[i][:-TAIL]) for i in range(N_EP)]
    say(f"  a1  logged definition (GAE lambda={cfg.ppo_gae_lambda}, all slots)")
    say(f"        within-episode mean EV = {np.mean(w_gae):+.4f} "
        f"(sd {np.std(w_gae):.4f})   [training log: mean 0.1785, last-100 0.3105]")
    say(f"  a2  MC definition (lambda=1, tail dropped)")
    say(f"        within-episode mean EV = {np.mean(w_mc):+.4f} "
        f"(sd {np.std(w_mc):.4f})")
    say(f"        pooled EV              = {ev_var(g_c, v_c):+.4f}")
    say(f"        pooled R2 (MSE-based)  = {r2_mse(g_c, v_c):+.4f}")
    say("        (pooled EV >> within EV means the critic's strength is")
    say("         tracking the episode LEVEL, not within-episode shape.)")

    say()
    say("=" * 78)
    say("b*  EPISODE PHASE t/T ALONE  (cubic, per episode)")
    say("=" * 78)
    say("  NOTE: the review proposed a constant baseline V = mean(G). That is")
    say("  DEFINITIONALLY zero under either metric, so it is replaced here.")
    ph_r2 = []
    for i in range(N_EP):
        y = g_c[ep_c == i]
        x = ph_c[ep_c == i]
        B = np.vstack([np.ones_like(x), x, x ** 2, x ** 3]).T
        yh = B @ np.linalg.lstsq(B, y, rcond=None)[0]
        ph_r2.append(ev_var(y, yh))
    say(f"  within-episode EV from phase alone = {np.mean(ph_r2):+.4f} "
        f"(sd {np.std(ph_r2):.4f})")
    say("  -> if this is near 0, within-episode return variation is mostly")
    say("     unpredictable noise and EV is a poor critic-quality diagnostic.")

    say()
    say("=" * 78)
    say("c1/c2  RIDGE CEILING, held out BY EPISODE  (** the Task-2 verdict **)")
    say("=" * 78)
    tr = ep_c < N_TRAIN
    inn = (ep_c >= N_TRAIN) & (ep_c < N_TRAIN + N_INNER)
    te = ep_c >= N_TRAIN + N_INNER
    say(f"  fit {N_TRAIN} ep / alpha-select {N_INNER} ep / test "
        f"{N_EP - N_TRAIN - N_INNER} ep   (features: {f_c.shape[1]} + 4 latent)")
    c1 = fit_select_eval(f_c[tr], g_c[tr], f_c[inn], g_c[inn],
                         f_c[te], g_c[te], ep_c[te])
    fl_c = np.hstack([f_c, l_c])
    c2 = fit_select_eval(fl_c[tr], g_c[tr], fl_c[inn], g_c[inn],
                         fl_c[te], g_c[te], ep_c[te])
    say(f"  c1  observed features only     "
        f"pooled EV {c1['pooled_ev']:+.4f}  pooled R2 {c1['pooled_r2']:+.4f}  "
        f"within EV {c1['within_ev']:+.4f}   (alpha={c1['alpha']:g})")
    say(f"  c2  observed + LATENT          "
        f"pooled EV {c2['pooled_ev']:+.4f}  pooled R2 {c2['pooled_r2']:+.4f}  "
        f"within EV {c2['within_ev']:+.4f}   (alpha={c2['alpha']:g})")
    say(f"  ** c2 - c1  (marginal contribution of the latents) **")
    say(f"       pooled EV {c2['pooled_ev'] - c1['pooled_ev']:+.4f}   "
        f"pooled R2 {c2['pooled_r2'] - c1['pooled_r2']:+.4f}   "
        f"within EV {c2['within_ev'] - c1['within_ev']:+.4f}")

    say()
    say("=" * 78)
    say("d  PER-EPISODE VALUE BIAS  (the value_loss channel)")
    say("=" * 78)
    rg_c = cat(G_GAE, TAIL)
    sigma = float(ac.ret_std)
    for label, tgt in (("GAE return (lambda=%.2f)  <- ppo.py:208 target"
                        % cfg.ppo_gae_lambda, rg_c),
                       ("MC  return (lambda=1)     <- true value target", g_c)):
        bias = np.array([tgt[ep_c == i].mean() - v_c[ep_c == i].mean()
                         for i in range(N_EP)])
        wres = np.array([np.var(tgt[ep_c == i] - v_c[ep_c == i])
                         for i in range(N_EP)])
        mse = float(np.mean((tgt - v_c) ** 2))
        b2, wv = float(np.mean(bias ** 2)), float(np.mean(wres))
        say(f"  vs {label}")
        say(f"      per-episode bias  mean {bias.mean():+9.2f}  "
            f"sd {bias.std():9.2f}  |bias| mean {np.abs(bias).mean():9.2f}"
            f"   range [{bias.min():+.1f}, {bias.max():+.1f}]")
        say(f"      total MSE = {mse:12.2f}")
        say(f"        level (bias^2) = {b2:12.2f}   {100 * b2 / mse:5.1f} %"
            f"   <- what latents could remove")
        say(f"        within-episode = {wv:12.2f}   {100 * wv / mse:5.1f} %")
        say(f"      in NORMALIZED units (the loss divides by ret_sigma="
            f"{sigma:.2f}): |bias|/sigma mean = "
            f"{np.abs(bias).mean() / sigma:.2f}")
        say()
    say(f"  scale: sd(MC return)={g_c.std():.1f}  sd(GAE return)={rg_c.std():.1f}"
        f"  ret_mean={float(ac.ret_mean):.1f}  ret_std={sigma:.2f}")
    say("  NOTE ret_std tracks the WITHIN-episode spread of the GAE return, so a")
    say("  per-episode LEVEL error is measured against that much smaller scale.")

    # ---- artifacts --------------------------------------------------------
    # raw per-slot dump so any follow-up question never needs a 50-minute re-run
    np.savez_compressed(
        f"{OUTDIR}/critic_ev_decomposition_raw.npz",
        V=cat(V, 0), G_MC=cat(G_MC, 0), G_GAE=cat(G_GAE, 0),
        EP=cat(EP, 0), PH=cat(PH, 0),
        FEAT=cat(F, 0).astype(np.float32), LATENT=cat(L, 0).astype(np.float32),
        seeds=np.arange(SEED0, SEED0 + N_EP), tail=np.array([TAIL]),
        gamma=np.array([cfg.ppo_gamma]), lam=np.array([cfg.ppo_gae_lambda]),
        ret_mean=np.array([float(ac.ret_mean)]),
        ret_std=np.array([float(ac.ret_std)]),
        best_update=np.array([int(ck["update"])]))
    with open(f"{OUTDIR}/critic_ev_decomposition_per_episode.csv", "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ep_rows[0]))
        w.writeheader()
        for r in ep_rows:
            w.writerow(r)
    with open(f"{OUTDIR}/critic_ev_decomposition.txt", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    say()
    say(f"raw per-slot    -> {OUTDIR}/critic_ev_decomposition_raw.npz")
    say(f"per-episode CSV -> {OUTDIR}/critic_ev_decomposition_per_episode.csv")
    say(f"summary         -> {OUTDIR}/critic_ev_decomposition.txt")


if __name__ == "__main__":
    main()
