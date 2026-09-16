"""
ppo.py -- PPO update + rollout collection for ML DRL Scheduler Phase 2.

Pieces:
  SlotTrajectory  -- per-slot data needed for PPO update.
  collect_rollout -- run one episode with the current actor, store trajectories.
  compute_gae     -- running-accumulator GAE with bootstrap at episode end.
  ppo_update      -- clipped PPO step using teacher-forcing replay through
                     the *current* actor/critic.

Spec details:
  * macro PPO mode: ratio = exp(new_macro_logprob - old_macro_logprob),
    log-ratio clamped to +-20 before exp (float32 overflow guard)
  * v1 = 1 episode per update, num_envs=1
  * entropy bonus uses entropy_mean (per-decision) for stability
  * advantage normalization per rollout (whole episode)
  * NO reward normalization in v1
  * the fixed-length episode end is a TIME LIMIT (truncation), not a true
    terminal state: GAE bootstraps through the boundary with V(s_T) of the
    final observation (partial-episode bootstrapping)
"""

from __future__ import annotations

import dataclasses
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from config import Config


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SlotTrajectory:
    obs: dict                            # raw obs dict from env.get_observation
    action_sequence: np.ndarray          # [R, Lmax] int
    policy_decision_mask: np.ndarray     # [R, Lmax] bool
    fixed_unit_map: np.ndarray           # [R, Lmax] int (unit_id at fixed pos)
    old_logprob_sum: float
    per_subaction_logprobs: list
    entropy_sum: float
    num_policy_decisions: int
    value: float
    reward: float
    done: bool
    # head inputs cached by decode(emit_context=True); consumed by
    # ActorCritic.replay_batch. None when cfg.ppo_batched_replay is off.
    ctx: dict | None = None


# ---------------------------------------------------------------------------
# GAE with running accumulator
# ---------------------------------------------------------------------------
def compute_gae(trajs: List[SlotTrajectory], gamma: float,
                gae_lambda: float, last_value: float = 0.0):
    T = len(trajs)
    advantages = np.zeros(T, dtype=np.float64)
    values = np.array([t.value for t in trajs], dtype=np.float64)
    rewards = np.array([t.reward for t in trajs], dtype=np.float64)
    dones = np.array([t.done for t in trajs], dtype=np.float64)

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            # the episode ends by time limit (truncation), not at a true
            # terminal state -> bootstrap through the boundary with V(s_T)
            # (pass last_value=0.0 to treat the end as terminal instead)
            next_value = last_value
            next_nonterminal = 1.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# rollout collection
# ---------------------------------------------------------------------------
def collect_rollout(env, ac, episode_idx: int):
    """Run one full episode with the current actor.

    Returns (trajs, last_value): the SlotTrajectory list plus V(s_T) of the
    final observation, used to bootstrap GAE through the episode time limit.
    (env returns its last prepared observation at done -- one _prepare_slot
    short of a true s_T, an accepted approximation for the boundary value.)
    """
    emit = bool(getattr(ac.cfg, "ppo_batched_replay", False))
    obs = env.reset(episode_idx)
    trajs: List[SlotTrajectory] = []
    done = False
    while not done:
        out = ac.decode(obs, deterministic=False, emit_context=emit)
        action = out["action_sequence"]
        # store BEFORE stepping (obs at decision time)
        traj_obs = obs                            # env.get_observation returns
                                                  # a fresh snapshot; safe to keep
        next_obs, reward, done, info = env.step(action)
        trajs.append(SlotTrajectory(
            obs=traj_obs,
            action_sequence=out["action_sequence"].copy(),
            policy_decision_mask=out["policy_decision_mask"].copy(),
            fixed_unit_map=traj_obs["fixed_unit_map"].copy(),
            old_logprob_sum=float(out["log_prob_sum"]),
            per_subaction_logprobs=list(out["per_subaction_logprobs"]),
            entropy_sum=float(out["entropy_sum"]),
            num_policy_decisions=int(out["num_policy_decisions"]),
            value=float(out["value"]),
            reward=float(reward),
            done=bool(done),
            ctx=out.get("context"),
        ))
        obs = next_obs
    last_value = float(ac.state_value(obs))
    return trajs, last_value


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def _minibatch_batched(ac, trajs, mb_idx, adv_norm, returns, cfg,
                       ret_sigma: float, actor_frozen: bool, n_mb: int):
    """One minibatch through the batched replay path.

    Produces exactly the six accumulators the sequential loop produces and
    performs the identical backward. Two deliberate deviations, both required:

      * the six logging sums are accumulated in FLOAT64 (``.double()`` before
        the reduction), because the sequential loop accumulates them in Python
        floats. sum_k3_kl feeds the latching actor-freeze test against a hard
        0.03 threshold, and the live KL sits near it -- a float32 tree
        reduction there could flip the freeze schedule, an O(1) behavioural
        change from an O(1e-7) numeric one.
      * ``(ratio - 1) - log_ratio`` is widened BEFORE the subtraction, matching
        the sequential ``float(ratio.item()) - 1.0`` ordering.

    No padding is used at the sample dimension: replay_batch is called with
    exactly len(mb_idx) slots, so the short final minibatch stays short.
    """
    dev = ac.device
    rep = ac.replay_batch([trajs[i].ctx for i in mb_idx])
    # Inspect every actual minibatch row, including rows outside the periodic
    # sequential cross-check. NaN can otherwise evade a comparison such as
    # ``abs(batch - seq) > tolerance`` and contaminate Adam's moments.
    finite = torch.stack([
        torch.isfinite(rep[key]).all()
        for key in ("log_prob_sum", "entropy_sum", "value",
                    "num_policy_decisions")
    ]).all()
    if not bool(finite):
        raise RuntimeError(
            "Batched replay produced nonfinite outputs; no optimizer step "
            "was applied to this minibatch. Inspect replay_batch or start "
            "a new run with sequential replay.")

    old_lp = torch.tensor([trajs[i].old_logprob_sum for i in mb_idx],
                          dtype=torch.float32, device=dev)
    adv = torch.tensor(adv_norm[mb_idx], dtype=torch.float32, device=dev)
    ret = torch.tensor(returns[mb_idx], dtype=torch.float32, device=dev)

    # same clamp as the sequential path: the ~30-subaction log-ratio sum can
    # drift past float32 exp overflow (~88)
    log_ratio = torch.clamp(rep["log_prob_sum"] - old_lp, -20.0, 20.0)
    ratio = torch.exp(log_ratio)

    clip_eps = cfg.ppo_clip_eps
    policy_loss = -torch.minimum(ratio * adv,
                                 torch.clamp(ratio, 1.0 - clip_eps,
                                             1.0 + clip_eps) * adv)
    value_loss = ((rep["value"] - ret) / ret_sigma).pow(2)
    n_dec = rep["num_policy_decisions"].clamp(min=1).to(value_loss.dtype)
    entropy_mean = rep["entropy_sum"] / n_dec

    # sum_i(loss_i)/n_mb == mean, i.e. the same scaled gradient the sequential
    # loop accumulates one .backward() at a time
    if actor_frozen:
        (cfg.ppo_value_coef * value_loss).mean().backward()
    else:
        (policy_loss + cfg.ppo_value_coef * value_loss
         - cfg.ppo_entropy_coef * entropy_mean).mean().backward()

    with torch.no_grad():
        d = lambda t: float(t.double().sum().item())
        return (d(policy_loss), d(value_loss), d(rep["entropy_sum"]),
                d(entropy_mean),
                float(((ratio.double() - 1.0)
                       - log_ratio.double()).sum().item()),
                int(((ratio - 1.0).abs() > clip_eps).sum().item()),
                rep)


def _verify_batched(ac, trajs, mb_idx, cfg, update_idx: int, rep=None) -> bool:
    """Cross-check the batched path against sequential replay on a few slots.

    Caching decode's contexts means replay's trajectory-consistency asserts no
    longer execute per sample during the update; replay_batch restores them in
    batched form, and this restores the numeric check.

    Three properties, each the result of an audit finding:

    * It verifies the TENSOR THE OPTIMIZER ACTUALLY CONSUMED. ``rep`` is the
      real minibatch output (256 slots); we slice its first k rows rather than
      re-running replay_batch on k slots alone. A batch-size-dependent defect
      (padding, a GEMM path that only engages above a row threshold) would slip
      past a 4-slot re-run that reported OK.
    * The tolerance is SCALE-FREE. ``value`` is de-normalized into raw return
      units (v_norm * ret_std + ret_mean), so its float32 noise floor grows with
      ret_std; an absolute 1e-3 gate has only ~30x headroom at this reward scale
      and none at a larger one. log_prob_sum / entropy_sum are compared
      absolutely (they are O(10) by construction); value is compared in the
      NORMALIZED units the loss actually uses.
    * A failure returns False; ppo_update then clears the minibatch gradients
      and raises BEFORE optimizer.step. It must not silently use the already
      computed, unverified gradients or mutate the recorded configuration.
      The supervisor must not automatically retry this deterministic failure.
    """
    k = min(int(cfg.ppo_batch_verify_slots), len(mb_idx))
    idxs = list(mb_idx[:k])
    raw_sigma = float(ac.ret_std)
    sigma = max(raw_sigma, 1e-6)
    tol = float(cfg.ppo_batch_verify_tol)
    if not (math.isfinite(raw_sigma) and math.isfinite(tol) and tol > 0):
        print(f"    [batch-verify] update {update_idx}: invalid scale or "
              f"tolerance -- FAILED", flush=True)
        return False
    worst, where = 0.0, ""
    for j, i in enumerate(idxs):
        seq = ac.replay(trajs[i].obs, trajs[i].action_sequence,
                        trajs[i].policy_decision_mask)
        seq_n = float(seq["num_policy_decisions"])
        batch_n = float(rep["num_policy_decisions"][j])
        if not (math.isfinite(seq_n) and math.isfinite(batch_n)
                and seq_n == batch_n):
            print(f"    [batch-verify] update {update_idx}: n_decisions "
                  f"mismatch/nonfinite at slot {i} -- FAILED",
                  flush=True)
            return False
        for key, scale in (("log_prob_sum", 1.0), ("entropy_sum", 1.0),
                           ("value", sigma)):
            seq_value, batch_value = float(seq[key]), float(rep[key][j])
            if not (math.isfinite(seq_value) and math.isfinite(batch_value)):
                print(f"    [batch-verify] update {update_idx}: nonfinite "
                      f"{key} at slot {i} -- FAILED", flush=True)
                return False
            d = abs(seq_value - batch_value) / scale
            if d > worst:
                worst, where = d, f"slot {i} {key}"
    if worst >= tol:
        print(f"    [batch-verify] update {update_idx}: FAILED, max scaled "
              f"|diff| {worst:.3e} >= {tol} at {where}", flush=True)
        return False
    print(f"    [batch-verify] update {update_idx}: {k} slots, max scaled "
          f"|diff| {worst:.2e} OK", flush=True)
    return True


def ppo_update(ac, trajs: List[SlotTrajectory], optimizer, cfg: Config,
               update_idx: int = 0, last_value: float = 0.0):
    """One PPO update: ppo_epochs × minibatch passes over the rollout.

    ``update_idx`` seeds the minibatch-shuffle RNG so that each PPO update
    sees a different shuffle pattern (otherwise the same seed would repeat
    the identical shuffle every update).
    ``last_value`` is V(s_T) from collect_rollout (truncation bootstrap).
    """
    advantages, returns = compute_gae(trajs, cfg.ppo_gamma, cfg.ppo_gae_lambda,
                                      last_value=last_value)
    # per-rollout advantage normalization (across the full episode rollout)
    if advantages.std() > 1e-8:
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        adv_norm = advantages - advantages.mean()

    # update the return normalizer from this rollout's (raw) returns; the value
    # head then regresses NORMALIZED returns so its gradient is O(1) instead of
    # the ~10^4 that dominated the shared grad clip and the shared encoder.
    ac.update_return_normalizer(returns)
    ret_sigma = float(ac.ret_std)

    # actor vs critic params for SEPARATE grad clipping (fix #3): the value
    # gradient can no longer set the policy's clip budget (the shared encoder
    # rides with the actor group; only value_head is clipped on its own).
    actor_params = [p for n, p in ac.named_parameters()
                    if not n.startswith("value_head")]
    critic_params = [p for n, p in ac.named_parameters()
                     if n.startswith("value_head")]

    T = len(trajs)
    indices = np.arange(T)
    mb_size = cfg.ppo_minibatch_size

    policy_losses, value_losses = [], []
    entropy_sums_log, entropy_means_log = [], []
    approx_kls, clip_fracs, grad_norms = [], [], []

    rng = np.random.default_rng(cfg.seed + 31337 + update_idx)
    actor_frozen = False        # set by KL early-stop; persists to update end
    # batched replay: only when enabled AND every slot carries a cached
    # context (a resumed rollout collected with the flag off would not)
    batched = (bool(getattr(cfg, "ppo_batched_replay", False))
               and cfg.decode_order == "rbg_major"
               and all(t.ctx is not None for t in trajs))
    verify_every = int(getattr(cfg, "ppo_batch_verify_every", 0)) if batched else 0
    for epoch in range(cfg.ppo_epochs):
        rng.shuffle(indices)
        for start in range(0, T, mb_size):
            mb_idx = indices[start:start + mb_size]
            n_mb = len(mb_idx)

            optimizer.zero_grad()

            sum_policy = 0.0
            sum_value = 0.0
            sum_ent_sum = 0.0
            sum_ent_mean = 0.0
            sum_k3_kl = 0.0       # KL = mean((ratio - 1) - log_ratio)  [k3, unbiased]
            n_clipped = 0

            if batched:
                try:
                    (sum_policy, sum_value, sum_ent_sum, sum_ent_mean,
                     sum_k3_kl, n_clipped, _rep) = _minibatch_batched(
                        ac, trajs, mb_idx, adv_norm, returns, cfg,
                        ret_sigma, actor_frozen, n_mb)
                    if (verify_every and epoch == 0 and start == 0
                            and update_idx % verify_every == 0):
                        # Check the actual backward's outputs. Its gradients
                        # are discarded on failure, before any optimizer step.
                        with torch.no_grad():
                            valid = _verify_batched(ac, trajs, mb_idx, cfg,
                                                    update_idx, rep=_rep)
                        if not valid:
                            raise RuntimeError(
                                f"Batched replay verification failed at update "
                                f"{update_idx}; no optimizer step was applied "
                                "to this minibatch. Inspect replay_batch or "
                                "start a new run with sequential replay. "
                                "Do not automatically retry this failure.")
                except Exception:
                    optimizer.zero_grad(set_to_none=True)
                    raise
                mb_iter = ()          # sequential body below is skipped
            else:
                mb_iter = mb_idx

            for idx in mb_iter:
                traj = trajs[idx]
                rep = ac.replay(traj.obs, traj.action_sequence,
                                traj.policy_decision_mask)
                # (frozen-actor passes still need the full replay forward:
                #  the value shares it; see value-only backward below)

                new_lp = rep["log_prob_sum"]
                old_lp = float(traj.old_logprob_sum)
                # clamp the macro log-ratio before exp: the ~32-subaction sum
                # can drift far beyond single-action ranges, and float32 exp
                # overflows to inf above ~88 -> inf loss -> NaN gradients.
                # Inactive in normal operation (clip region is ~+-0.18).
                log_ratio = torch.clamp(new_lp - old_lp, -20.0, 20.0)
                ratio = torch.exp(log_ratio)

                adv = float(adv_norm[idx])
                ret = float(returns[idx])

                clip_eps = cfg.ppo_clip_eps
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - clip_eps,
                                       1.0 + clip_eps) * adv
                policy_loss = -torch.minimum(unclipped, clipped)

                # value loss in NORMALIZED return space: (V_raw-ret)/sigma ==
                # V_norm - ret_norm, so the value gradient is O(1) (fix #1)
                value_loss = ((rep["value"] - ret) / ret_sigma).pow(2)

                n_dec = max(int(rep["num_policy_decisions"]), 1)
                entropy_mean = rep["entropy_sum"] / n_dec

                total_loss = (policy_loss
                              + cfg.ppo_value_coef * value_loss
                              - cfg.ppo_entropy_coef * entropy_mean)

                # accumulate scaled gradient (mean over minibatch); once the
                # actor is KL-frozen this update, backward ONLY the value term
                # (critic keeps training; actor grads are dropped before step)
                if actor_frozen:
                    ((cfg.ppo_value_coef * value_loss) / n_mb).backward()
                else:
                    (total_loss / n_mb).backward()

                with torch.no_grad():
                    sum_policy += float(policy_loss.item())
                    sum_value += float(value_loss.item())
                    sum_ent_sum += float(rep["entropy_sum"].item())
                    sum_ent_mean += float(entropy_mean.item())
                    lr_val = float(log_ratio.item())
                    r_val = float(ratio.item())
                    # k3 KL estimator: unbiased, always >= 0
                    sum_k3_kl += (r_val - 1.0) - lr_val
                    if abs(r_val - 1.0) > clip_eps:
                        n_clipped += 1

            # KL early-stop (active only when cfg.ppo_target_kl > 0): the
            # minibatch k3-KL measures how far the policy has already drifted
            # from the rollout policy. Past 1.5x target the surrogate is no
            # longer trustworthy for the ACTOR -- freeze the actor (and the
            # shared encoder) for the update's remaining passes but KEEP
            # training the value head: KL bounds actor drift only, and
            # halting the critic too starves it exactly when its warm-up
            # noise caused the drift (v2 fix, 2026-07-02; the v1 full-stop
            # spent 79% of L2b's early updates skipping critic epochs).
            # Clip alone does not bound this aggregate drift (MixedSpeed_L2
            # collapses at update 39/52: KL 0.054/0.066, clip 0.43/0.47).
            if (cfg.ppo_target_kl > 0 and not actor_frozen
                    and (sum_k3_kl / n_mb) > 1.5 * cfg.ppo_target_kl):
                actor_frozen = True
                print(f"    [kl-stop] update {update_idx}: minibatch KL "
                      f"{sum_k3_kl / n_mb:.4f} > "
                      f"{1.5 * cfg.ppo_target_kl:.4f} "
                      f"(epoch {epoch + 1}/{cfg.ppo_epochs}) -- actor frozen, "
                      f"value-only for remaining passes")

            # separate clips so the (now O(1)) value gradient cannot throttle
            # the policy gradient through one shared global clip (fix #3)
            if actor_frozen:
                # drop actor-side grads (encoder included: it is shared, and
                # stepping it would move the policy despite the freeze); Adam
                # skips None-grad params entirely, so the policy is bit-frozen
                for p in actor_params:
                    p.grad = None
                gn_actor = torch.zeros(())
            else:
                gn_actor = nn.utils.clip_grad_norm_(actor_params,
                                                    cfg.ppo_max_grad_norm)
            gn_critic = nn.utils.clip_grad_norm_(critic_params,
                                                 cfg.ppo_max_grad_norm)
            grad_norm = gn_actor + gn_critic
            optimizer.step()

            policy_losses.append(sum_policy / n_mb)
            value_losses.append(sum_value / n_mb)
            entropy_sums_log.append(sum_ent_sum / n_mb)
            entropy_means_log.append(sum_ent_mean / n_mb)
            # k3 estimator: approx_KL = mean((ratio-1) - log_ratio)
            # always non-negative, unbiased; more accurate than k1.
            approx_kls.append(sum_k3_kl / n_mb)
            clip_fracs.append(n_clipped / n_mb)
            grad_norms.append(float(grad_norm.item()))

    values_arr = np.array([t.value for t in trajs])
    var_ret = float(np.var(returns))
    explained_var = float(1.0 - np.var(returns - values_arr) / max(var_ret, 1e-8))

    return dict(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy_sum=float(np.mean(entropy_sums_log)),
        entropy_mean=float(np.mean(entropy_means_log)),
        approx_KL=float(np.mean(approx_kls)),
        clip_fraction=float(np.mean(clip_fracs)),
        grad_norm=float(np.mean(grad_norms)),
        explained_variance=explained_var,
    )


# ---------------------------------------------------------------------------
# smoke test (run inside the GPU container)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import phase2_debug_config
    from env import SchedulerEnv
    from policy import ActorCritic

    cfg = phase2_debug_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ppo.py smoke test  device={device}  episode_len={cfg.episode_len}")

    env = SchedulerEnv(cfg)
    ac = ActorCritic(cfg).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.ppo_learning_rate)

    # 1 rollout
    t0 = __import__("time").time()
    trajs, last_value = collect_rollout(env, ac, episode_idx=0)
    rollout_time = __import__("time").time() - t0
    print(f"  rollout: {len(trajs)} slots in {rollout_time:.2f}s  "
          f"reward_sum={sum(t.reward for t in trajs):.2f}  "
          f"V(s_T)={last_value:.3f}")
    assert len(trajs) == cfg.episode_len
    assert trajs[-1].done
    assert np.isfinite(last_value)

    # GAE (truncation bootstrap with V(s_T))
    adv, ret = compute_gae(trajs, cfg.ppo_gamma, cfg.ppo_gae_lambda,
                           last_value=last_value)
    print(f"  gae: adv mean={adv.mean():.3f} std={adv.std():.3f}  "
          f"ret mean={ret.mean():.3f}")
    assert adv.shape == (len(trajs),)

    # PPO update
    t0 = __import__("time").time()
    stats = ppo_update(ac, trajs, opt, cfg, last_value=last_value)
    update_time = __import__("time").time() - t0
    print(f"  ppo_update: {update_time:.2f}s")
    for k, v in stats.items():
        print(f"    {k}: {v:.4f}")
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])
    assert -10 < stats["approx_KL"] < 10
    assert 0.0 <= stats["clip_fraction"] <= 1.0

    print("ppo.py smoke test passed.")
