"""
Numeric scout for FirstFullRun PPO scheduler diagnosis (CPU-only, READ-ONLY).
Loads runs/FirstFullRun/ckpt/latest.pt into ActorCritic(phase2_main_config())
and computes returns/critic/gradient/reward-decomp/policy numbers.
"""
import json
import numpy as np
import torch

from config import phase2_main_config
from env import SchedulerEnv
from policy import ActorCritic
import ppo as ppo_mod
from baselines import SUSPF, PF

torch.manual_seed(0)
np.random.seed(0)

CKPT = "runs/FirstFullRun/ckpt/latest.pt"
SEED = 10000  # held-out eval seed (env.reset(ep_idx + 10000))

cfg = phase2_main_config()
ck = torch.load(CKPT, map_location="cpu")
print(f"[ckpt] update={ck['update']}  best_eval_reward={ck.get('best_eval_reward')}")

ac = ActorCritic(cfg)
ac.load_state_dict(ck["model"])
ac.eval()

env = SchedulerEnv(cfg)

OUT = {}

# ---------------------------------------------------------------------------
# 1+2+3+4: sampling rollout (the distribution PPO actually trains on)
# ---------------------------------------------------------------------------
torch.manual_seed(123)
trajs, last_value = ppo_mod.collect_rollout(env, ac, episode_idx=SEED)
T = len(trajs)
rewards = np.array([t.reward for t in trajs], dtype=np.float64)
values = np.array([t.value for t in trajs], dtype=np.float64)
print(f"[rollout] T={T}  reward_sum={rewards.sum():.2f}  last_value(V(s_T))={last_value:.4f}")

adv, returns = ppo_mod.compute_gae(trajs, cfg.ppo_gamma, cfg.ppo_gae_lambda,
                                   last_value=last_value)

# RETURNS scale
OUT["returns"] = {
    "rollout_seed": SEED,
    "sampling": True,
    "T": int(T),
    "returns_min": float(returns.min()),
    "returns_max": float(returns.max()),
    "returns_mean": float(returns.mean()),
    "returns_std": float(returns.std()),
    "adv_raw_mean": float(adv.mean()),
    "adv_raw_std": float(adv.std()),
    "per_slot_reward_min": float(rewards.min()),
    "per_slot_reward_max": float(rewards.max()),
    "per_slot_reward_mean": float(rewards.mean()),
    "per_slot_reward_std": float(rewards.std()),
    "episode_reward_sum": float(rewards.sum()),
    "last_value_bootstrap": float(last_value),
}

# CRITIC: explained variance recomputed + corr(V, returns)
V = values
var_ret = float(np.var(returns))
ev = float(1.0 - np.var(returns - V) / max(var_ret, 1e-8))
corr = float(np.corrcoef(V, returns)[0, 1])
OUT["critic"] = {
    "explained_variance_recomputed": ev,
    "V_min": float(V.min()),
    "V_max": float(V.max()),
    "V_mean": float(V.mean()),
    "V_std": float(V.std()),
    "returns_std": float(returns.std()),
    "corr_V_returns": corr,
    "mean_abs_resid": float(np.mean(np.abs(returns - V))),
    "rmse_resid": float(np.sqrt(np.mean((returns - V) ** 2))),
}
print(f"[critic] EV={ev:.4f}  corr(V,ret)={corr:.4f}  "
      f"V[mean={V.mean():.2f} std={V.std():.2f}]  ret[mean={returns.mean():.2f} std={returns.std():.2f}]")

# ---------------------------------------------------------------------------
# 5: GRADIENTS — one ppo_update-style minibatch, three terms backward SEPARATELY
# ---------------------------------------------------------------------------
# Replicate ppo_update minibatch loss exactly, but split grad contributions.
if adv.std() > 1e-8:
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
else:
    adv_norm = adv - adv.mean()

mb_size = cfg.ppo_minibatch_size  # 256
rng = np.random.default_rng(cfg.seed + 31337 + 0)
indices = np.arange(T)
rng.shuffle(indices)
mb_idx = indices[:mb_size]
n_mb = len(mb_idx)
clip_eps = cfg.ppo_clip_eps


def build_terms():
    """Re-run replay for the minibatch and return list of (policy_loss, value_loss, neg_entropy_mean) tensors."""
    pol_terms, val_terms, ent_terms = [], [], []
    for idx in mb_idx:
        traj = trajs[idx]
        rep = ac.replay(traj.obs, traj.action_sequence, traj.policy_decision_mask)
        new_lp = rep["log_prob_sum"]
        old_lp = float(traj.old_logprob_sum)
        log_ratio = torch.clamp(new_lp - old_lp, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        a = float(adv_norm[idx])
        ret = float(returns[idx])
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a
        policy_loss = -torch.minimum(unclipped, clipped)
        value_loss = (rep["value"] - ret).pow(2)
        n_dec = max(int(rep["num_policy_decisions"]), 1)
        entropy_mean = rep["entropy_sum"] / n_dec
        pol_terms.append(policy_loss)
        val_terms.append(value_loss)
        ent_terms.append(entropy_mean)
    return pol_terms, val_terms, ent_terms


def grad_norm_of(scalar_loss):
    for p in ac.parameters():
        if p.grad is not None:
            p.grad = None
    scalar_loss.backward(retain_graph=False)
    total = 0.0
    for p in ac.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return float(np.sqrt(total))


# To get exact per-term contributions we must rebuild the graph for each term
# (backward frees the graph). The loss is the MEAN over the minibatch of
# (total_loss / n_mb) summed -> equivalently mean of each term / n_mb summed.
# We reproduce: each sample contributes (term / n_mb).backward(); summed across mb.

def term_grad_norm(which):
    """which in {'policy','value','entropy','total'}. Backward the chosen term(s)
    over the whole minibatch (gradients accumulate as in ppo_update)."""
    for p in ac.parameters():
        if p.grad is not None:
            p.grad = None
    for idx in mb_idx:
        traj = trajs[idx]
        rep = ac.replay(traj.obs, traj.action_sequence, traj.policy_decision_mask)
        new_lp = rep["log_prob_sum"]
        old_lp = float(traj.old_logprob_sum)
        log_ratio = torch.clamp(new_lp - old_lp, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        a = float(adv_norm[idx])
        ret = float(returns[idx])
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a
        policy_loss = -torch.minimum(unclipped, clipped)
        value_loss = (rep["value"] - ret).pow(2)
        n_dec = max(int(rep["num_policy_decisions"]), 1)
        entropy_mean = rep["entropy_sum"] / n_dec
        if which == "policy":
            term = policy_loss
        elif which == "value":
            term = cfg.ppo_value_coef * value_loss
        elif which == "entropy":
            term = -cfg.ppo_entropy_coef * entropy_mean
        elif which == "total":
            term = (policy_loss + cfg.ppo_value_coef * value_loss
                    - cfg.ppo_entropy_coef * entropy_mean)
        (term / n_mb).backward()
    total = 0.0
    for p in ac.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return float(np.sqrt(total))


gn_policy = term_grad_norm("policy")
gn_value = term_grad_norm("value")
gn_entropy = term_grad_norm("entropy")
gn_total = term_grad_norm("total")
OUT["gradients"] = {
    "minibatch_size": int(n_mb),
    "grad_norm_total_combined": gn_total,
    "grad_norm_policy_loss": gn_policy,
    "grad_norm_value_term": gn_value,   # includes value_coef=0.5
    "grad_norm_entropy_term": gn_entropy,  # includes entropy_coef=0.01
    "value_coef": cfg.ppo_value_coef,
    "entropy_coef": cfg.ppo_entropy_coef,
    "ppo_max_grad_norm": cfg.ppo_max_grad_norm,
    "scale_factor_clip": gn_total / cfg.ppo_max_grad_norm if gn_total > 0 else None,
}
print(f"[grad] total={gn_total:.2f}  policy={gn_policy:.2f}  "
      f"value(x0.5)={gn_value:.2f}  entropy(x0.01)={gn_entropy:.4f}")

# ---------------------------------------------------------------------------
# 6: REWARD DECOMPOSITION — PPO (det + sampled) vs SUS+PF + PF on same seed
# ---------------------------------------------------------------------------
def decompose(sched_fn, seed, sched_is_alloc_fn):
    """Run one episode; accumulate three reward components from info.
    n_miss_total = n_miss_deadline + n_retx_drop + overflow, where overflow is
    recovered from reward = lambda_s*short + lambda_c*comp - lambda_m*n_miss."""
    obs = env.reset(seed)
    done = False
    short = 0.0
    comp = 0
    miss_deadline = 0
    retx_drop = 0
    overflow = 0
    total = 0.0
    nslots = 0
    while not done:
        alloc = sched_is_alloc_fn(env, obs)
        obs, reward, done, info = env.step(alloc)
        short += info["reward_short"]
        comp += info["n_comp"]
        miss_deadline += info["n_miss_deadline"]
        retx_drop += info["n_retx_drop"]
        # recover overflow this slot:
        # reward = ls*short + lc*comp - lm*(miss_dl + retx_drop + overflow)
        n_miss_recon = (cfg.lambda_s * info["reward_short"]
                        + cfg.lambda_c * info["n_comp"] - reward) / cfg.lambda_m
        ov = round(n_miss_recon) - info["n_miss_deadline"] - info["n_retx_drop"]
        overflow += int(ov)
        total += reward
        nslots += 1
    n_miss_total = miss_deadline + retx_drop + overflow
    term_short = cfg.lambda_s * short
    term_comp = cfg.lambda_c * comp
    term_miss = -cfg.lambda_m * n_miss_total
    return {
        "episode_total_reward": float(total),
        "term_lambda_s_short": float(term_short),
        "term_lambda_c_comp": float(term_comp),
        "term_neg_lambda_m_miss": float(term_miss),
        "raw_short_sum": float(short),
        "raw_n_comp": int(comp),
        "raw_n_miss_deadline": int(miss_deadline),
        "raw_n_retx_drop": int(retx_drop),
        "raw_n_overflow_drop": int(overflow),
        "raw_n_miss_total": int(n_miss_total),
        "sum_of_three_terms_check": float(term_short + term_comp + term_miss),
    }


def ppo_det_alloc(env, obs):
    with torch.no_grad():
        out = ac.decode(env.get_observation(), deterministic=True)
    return out["action_sequence"]


def ppo_sample_alloc(env, obs):
    with torch.no_grad():
        out = ac.decode(env.get_observation(), deterministic=False)
    return out["action_sequence"]


suspf = SUSPF()
pf = PF()
def suspf_alloc(env, obs):
    return suspf.schedule(env)
def pf_alloc(env, obs):
    return pf.schedule(env)

torch.manual_seed(7)
dec_ppo_det = decompose(None, SEED, ppo_det_alloc)
torch.manual_seed(7)
dec_ppo_smp = decompose(None, SEED, ppo_sample_alloc)
dec_suspf = decompose(None, SEED, suspf_alloc)
dec_pf = decompose(None, SEED, pf_alloc)

OUT["reward_decomp"] = {
    "seed": SEED,
    "PPO_deterministic_argmax": dec_ppo_det,
    "PPO_sampled": dec_ppo_smp,
    "SUSPF": dec_suspf,
    "PF": dec_pf,
}
print(f"[decomp] PPO(det) total={dec_ppo_det['episode_total_reward']:.1f} "
      f"short={dec_ppo_det['term_lambda_s_short']:.1f} comp={dec_ppo_det['term_lambda_c_comp']:.1f} "
      f"miss={dec_ppo_det['term_neg_lambda_m_miss']:.1f}")
print(f"[decomp] SUSPF   total={dec_suspf['episode_total_reward']:.1f} "
      f"short={dec_suspf['term_lambda_s_short']:.1f} comp={dec_suspf['term_lambda_c_comp']:.1f} "
      f"miss={dec_suspf['term_neg_lambda_m_miss']:.1f}")

# ---------------------------------------------------------------------------
# 7: POLICY — per-decision entropy, forced fraction, mean #valid UEs
# Instrument a manual decode loop matching policy.decode but recording masks.
# ---------------------------------------------------------------------------
def policy_stats(seed, deterministic):
    obs = env.reset(seed)
    done = False
    per_dec_entropy = []
    n_valid_options = []      # incl no-user
    n_valid_ues = []          # excl no-user
    forced = 0
    total_decisions = 0
    K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
    device = ac.device
    while not done:
        obs_t = __import__("policy").obs_to_tensors(obs, device)
        with torch.no_grad():
            e = ac.encode(obs_t)
        S_r = [set(obs["initial_S_r"][r]) for r in range(R)]
        in_slot_count = np.zeros(K, dtype=np.int64)
        for r in range(R):
            for u in S_r[r]:
                in_slot_count[u] += 1
        temp_uncommit = torch.from_numpy(obs["uncommitted"].astype(np.float32)).to(device).clone()
        rbg_closed = [False] * R
        fixed_mask = obs["fixed_mask"]
        fixed_alloc = obs["fixed_allocation"]
        action_seq = np.zeros((R, L), dtype=np.int64)
        for l in range(L):
            for r in range(R):
                if fixed_mask[r, l]:
                    action_seq[r, l] = int(fixed_alloc[r, l])
                    continue
                if rbg_closed[r]:
                    action_seq[r, l] = 0
                    continue
                with torch.no_grad():
                    logits, valid_mask, pred_btx = ac._position_logits_and_mask(
                        obs_t, e, r, l, S_r[r], in_slot_count, temp_uncommit)
                nvalid = int(valid_mask.sum().item())
                nvalid_ue = int(valid_mask[1:].sum().item())  # exclude no-user slot 0
                n_valid_options.append(nvalid)
                n_valid_ues.append(nvalid_ue)
                total_decisions += 1
                if nvalid == 1:
                    forced += 1
                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                per_dec_entropy.append(float(dist.entropy().item()))
                if deterministic:
                    action = int(masked_logits.argmax().item())
                else:
                    action = int(dist.sample().item())
                action_seq[r, l] = action
                if action == 0:
                    rbg_closed[r] = True
                else:
                    u = action - 1
                    S_r[r].add(u)
                    in_slot_count[u] += 1
                    new_u = torch.clamp(temp_uncommit[u] - pred_btx[u], min=0.0)
                    temp_uncommit = temp_uncommit.clone()
                    temp_uncommit[u] = new_u
        obs, reward, done, info = env.step(action_seq)
    per_dec_entropy = np.array(per_dec_entropy)
    return {
        "deterministic": deterministic,
        "total_policy_decisions": int(total_decisions),
        "mean_per_decision_entropy": float(per_dec_entropy.mean()),
        "std_per_decision_entropy": float(per_dec_entropy.std()),
        "median_per_decision_entropy": float(np.median(per_dec_entropy)),
        "frac_forced_single_option": float(forced / max(total_decisions, 1)),
        "n_forced": int(forced),
        "mean_valid_options_incl_nouser": float(np.mean(n_valid_options)),
        "mean_valid_ues_excl_nouser": float(np.mean(n_valid_ues)),
        "max_valid_ues": int(np.max(n_valid_ues)),
        "frac_decisions_zero_valid_ues": float(np.mean(np.array(n_valid_ues) == 0)),
    }

torch.manual_seed(7)
pol_sampled = policy_stats(SEED, deterministic=False)
pol_det = policy_stats(SEED, deterministic=True)
OUT["policy"] = {
    "sampled_rollout": pol_sampled,
    "deterministic_rollout": pol_det,
}
print(f"[policy] sampled mean_ent={pol_sampled['mean_per_decision_entropy']:.4f} "
      f"forced_frac={pol_sampled['frac_forced_single_option']:.4f} "
      f"mean_valid_ues={pol_sampled['mean_valid_ues_excl_nouser']:.2f}")

print("\n==== JSON OUTPUT ====")
print(json.dumps(OUT, indent=2))
