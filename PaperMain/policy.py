"""
policy.py -- PyTorch PPO actor/critic for ML DRL Scheduler Phase 2.

Architecture (Phase 2 spec):
  Actor  = SharedEncoder + ScoreNet + NoUserHead + masked categorical
  Critic = SharedEncoder + ValueHead  (encoder shared with actor by default)

Sequential autoregressive decoding handles the 32 (RBG, layer) sub-actions
per slot in layer-major order. Pending retransmissions are forced into
``env.fixed_*`` and skipped by the actor. The no-user / RBG-closure rule
is implemented per Section 4 of the Phase 2 spec.

For PPO update, ``replay`` re-runs the *current* SharedEncoder on the stored
observation and teacher-forces the stored ``action_sequence`` to recompute
log-probabilities, entropies and value under the updated weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import Config
from la_planner import SlotAllocationPlanner
from phy import reconstruct_h_hat


# ---------------------------------------------------------------------------
# Tensor / feature helpers
# ---------------------------------------------------------------------------
def obs_to_tensors(obs: dict, device: torch.device) -> dict:
    """Convert the numpy obs (env.get_observation output) to torch tensors."""
    return {
        "direction_fb": torch.from_numpy(obs["direction_fb"]).to(
            torch.complex64).to(device),
        "cqi_fb": torch.from_numpy(obs["cqi_fb"]).to(torch.float32).to(device),
        "age": torch.from_numpy(obs["age"]).to(torch.float32).to(device),
        "deadline": torch.from_numpy(obs["deadline"]).to(torch.float32).to(device),
        "backlog": torch.from_numpy(obs["backlog"]).to(torch.float32).to(device),
        "uncommitted": torch.from_numpy(obs["uncommitted"]).to(
            torch.float32).to(device),
        "avg_throughput": torch.from_numpy(obs["avg_throughput"]).to(
            torch.float32).to(device),
        "active": torch.from_numpy(obs["active"]).to(torch.bool).to(device),
        # queue state (Run4; harmless constants in legacy mode)
        "queue_len": torch.from_numpy(obs["queue_len"]).to(
            torch.float32).to(device),
        "queue_bits": torch.from_numpy(obs["queue_bits"]).to(
            torch.float32).to(device),
        "next_deadline": torch.from_numpy(obs["next_deadline"]).to(
            torch.float32).to(device),
    }


def build_encoder_input(obs_t: dict, cfg: Config) -> torch.Tensor:
    """Build the per-(UE, RBG) encoder input, [K, R, 70] or [K, R, 73].

    70 base channels; queue mode (cfg.queue_size > 1) appends three per-UE
    fields -- queue_len, queue_bits, next_deadline -- broadcast over RBG, for
    73. ActorCritic sizes the encoder to match:
    ``enc_in = 70 + (3 if cfg.queue_size > 1 else 0)``.
    """
    K, R = cfg.num_ue, cfg.num_rbg
    pmi_re = obs_t["direction_fb"].real
    pmi_im = obs_t["direction_fb"].imag
    pmi_rep = torch.cat([pmi_re, pmi_im], dim=-1)               # [K, R, 64]

    cqi_norm = (obs_t["cqi_fb"] / cfg.cqi_norm_const).unsqueeze(-1)  # [K, R, 1]

    age_clipped = torch.clamp(obs_t["age"], max=cfg.age_norm_max)
    age_norm = (age_clipped / cfg.age_norm_max).view(K, 1, 1).expand(K, R, 1)
    backlog_norm = (obs_t["backlog"] / cfg.b_norm).view(K, 1, 1).expand(K, R, 1)
    deadline_norm = (obs_t["deadline"] / cfg.deadline_max).view(K, 1, 1).expand(K, R, 1)
    avg_thr_norm = (obs_t["avg_throughput"] / cfg.b_norm).view(K, 1, 1).expand(K, R, 1)
    active_flag = obs_t["active"].to(torch.float32).view(K, 1, 1).expand(K, R, 1)

    out = torch.cat([pmi_rep, cqi_norm, age_norm, backlog_norm,
                     deadline_norm, avg_thr_norm, active_flag], dim=-1)  # [K,R,70]
    if cfg.queue_size > 1:
        # Run4 queue fields (per-UE, broadcast over RBG) -> [K, R, 73]
        qs = float(cfg.queue_size)
        qlen = (obs_t["queue_len"] / qs).view(K, 1, 1).expand(K, R, 1)
        qbits = (obs_t["queue_bits"] / (qs * cfg.b_norm)
                 ).view(K, 1, 1).expand(K, R, 1)
        nxt = (obs_t["next_deadline"] / cfg.deadline_max
               ).view(K, 1, 1).expand(K, R, 1)
        out = torch.cat([out, qlen, qbits, nxt], dim=-1)
    return out


# deadline bins (slots) for the v2 value features; matches the offline
# critic-ceiling probe that validated them
_V2_DEADLINE_BINS = ((0.0, 1.5), (1.5, 2.5), (2.5, 4.5), (4.5, 8.5),
                     (8.5, float("inf")))


def build_value_feats_v2(obs_t: dict, cfg: Config, fixed_mask: np.ndarray,
                         slot: int) -> torch.Tensor:
    """Structured value features (critic v2, cfg.ppo_critic_v2): 27, or 33 in
    queue mode where a 6-feature queue-pressure block is appended.

    Same observation content the encoder already receives, re-expressed so
    the return's drivers survive pooling: imminent-deadline structure (miss
    spikes), backlog stock (bits headroom), retx-grid occupancy (next-slot
    capacity), CQI/age aggregates (extractable rate, CSI reliability) and
    episode phase. Offline probe (30 held-out episodes, MC returns): R^2
    rises from -0.99 (6 global scalars alone) to +0.52 with these appended.
    """
    device = obs_t["cqi_fb"].device
    K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
    actf = obs_t["active"].to(torch.float32)
    dl = obs_t["deadline"]
    bk = obs_t["backlog"]
    unc = obs_t["uncommitted"]

    parts = []
    # deadline histogram: packet count + remaining-backlog mass per bin  [10]
    for lo, hi in _V2_DEADLINE_BINS:
        m = actf * (dl > lo).to(torch.float32) * (dl <= hi).to(torch.float32)
        parts.append(m.sum() / K)
        parts.append((bk * m).sum() / (K * cfg.b_norm))
    # backlog totals  [3]
    parts.append(unc.sum() / (K * cfg.b_norm))
    parts.append(bk.sum() / (K * cfg.b_norm))
    parts.append((actf * (unc > cfg.b_tx_epsilon).to(torch.float32)).sum() / K)
    # CQI / age aggregates  [5]
    best_cqi = obs_t["cqi_fb"].max(dim=1).values                    # [K]
    n_act = actf.sum().clamp(min=1.0)
    parts.append((best_cqi * actf).sum() / n_act / cfg.cqi_norm_const)
    parts.append((best_cqi * actf).max() / cfg.cqi_norm_const)
    parts.append(best_cqi.mean() / cfg.cqi_norm_const)
    age = torch.clamp(obs_t["age"], max=cfg.age_norm_max) / cfg.age_norm_max
    parts.append((age * actf).sum() / n_act)
    parts.append((age * bk).sum() / bk.sum().clamp(min=1e-9))
    # fairness / contention context  [2]
    parts.append(obs_t["avg_throughput"].mean() / cfg.b_norm)
    parts.append(actf.sum() / K)
    if cfg.queue_size > 1:
        # queue-pressure features (Run4)  [+6]
        ql = obs_t["queue_len"]
        parts.append((ql <= 1).to(torch.float32).mean())
        parts.append(((ql >= 2) & (ql <= 3)).to(torch.float32).mean())
        parts.append((ql >= 4).to(torch.float32).mean())
        parts.append((ql >= cfg.queue_size).to(torch.float32).mean())  # full buffers
        parts.append(obs_t["queue_bits"].sum()
                     / (K * cfg.queue_size * cfg.b_norm))
        parts.append((obs_t["next_deadline"] / cfg.deadline_max).mean())
    t_feats = torch.stack(parts)                    # [20] (+6 in queue mode)

    # retx-grid occupancy histogram + episode phase (numpy/python side)  [7]
    per_rbg_fixed = fixed_mask.sum(axis=1)                          # [R]
    tail = [float((per_rbg_fixed == c).sum()) / R for c in range(L + 1)]
    tail.append(float(fixed_mask.sum()) / (R * L))
    tail.append(float(slot) / cfg.episode_len)
    return torch.cat([t_feats,
                      torch.tensor(tail, device=device,
                                   dtype=torch.float32)])     # [27] or [33]


def build_value_tail(obs_t: dict, cfg: Config, fixed_mask: np.ndarray,
                     slot: int = 0) -> torch.Tensor:
    """The WEIGHT-INDEPENDENT part of the value-head input.

    = 6 global scalars, + 27 structured v2 features when cfg.ppo_critic_v2
      (33 in queue mode: the queue block adds 6).

    Split out of build_value_input so a once-per-rollout pre-pass can cache it
    per slot: only the two pooled projections of ``e`` depend on the network
    weights, so everything here is fixed for the whole PPO update.
    """
    device = obs_t["cqi_fb"].device
    active = obs_t["active"].to(torch.float32)         # [K]
    n_active = active.sum()
    pending_retx = torch.tensor(float(fixed_mask.sum()) / cfg.num_positions,
                                device=device, dtype=torch.float32)

    age_clipped = torch.clamp(obs_t["age"], max=cfg.age_norm_max)
    mean_age = age_clipped.mean() / cfg.age_norm_max

    if n_active > 0:
        mean_deadline = ((obs_t["deadline"] * active).sum() / n_active
                         / cfg.deadline_max)
        mean_backlog = ((obs_t["backlog"] * active).sum() / n_active
                        / cfg.b_norm)
    else:
        mean_deadline = torch.zeros((), device=device, dtype=torch.float32)
        mean_backlog = torch.zeros((), device=device, dtype=torch.float32)

    has_data = active.bool() & (obs_t["uncommitted"] > cfg.b_tx_epsilon)
    valid_count = has_data.to(torch.float32).sum() / cfg.num_ue
    active_count = n_active / cfg.num_ue

    out = torch.stack([active_count, pending_retx, mean_age,
                       mean_deadline, mean_backlog, valid_count])       # [6]
    if cfg.ppo_critic_v2:
        out = torch.cat([out, build_value_feats_v2(obs_t, cfg, fixed_mask,
                                                   slot)])              # [33]
    return out


def build_value_input(e: torch.Tensor, obs_t: dict, cfg: Config,
                      fixed_mask: np.ndarray, slot: int = 0) -> torch.Tensor:
    """Build the value head input vector.

    = mean_pool e [64] + max_pool e [64] + 6 global scalars  (134 dims)
    + 27 structured features when cfg.ppo_critic_v2 (161 dims; 167 in queue
      mode, where build_value_feats_v2 appends 6 queue-pressure features).
    """
    return torch.cat([e.mean(dim=(0, 1)), e.amax(dim=(0, 1)),
                      build_value_tail(obs_t, cfg, fixed_mask, slot)], dim=0)


def ortho_score_torch(direction_fb: torch.Tensor, rbg: int,
                      selected_ues: set, K: int) -> torch.Tensor:
    """OrthScore vs S_r in an RBG, using unit-norm feedback directions."""
    if not selected_ues:
        return torch.ones(K, device=direction_fb.device)
    sel = torch.tensor(sorted(selected_ues), device=direction_fb.device,
                       dtype=torch.long)
    c_all = direction_fb[:, rbg, :]                              # [K, 32]
    c_sel = direction_fb[sel, rbg, :]                            # [S, 32]
    corr = (c_all @ c_sel.conj().transpose(-1, -2)).abs().pow(2)  # [K, S]
    return 1.0 - corr.amax(dim=-1)


def predict_btx_torch(temp_uncommit: torch.Tensor, cqi_fb_r: torch.Tensor,
                      cfg: Config) -> torch.Tensor:
    """B_tx predicted at the current sub-action time, per UE in this RBG."""
    pred = cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate * cqi_fb_r
    return torch.minimum(temp_uncommit, pred)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def _mlp(dims: list[int]) -> nn.Sequential:
    """Build an MLP: ReLU between hidden layers, NO activation after the last
    layer (so embeddings and logits keep full real range)."""
    mods: list = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class SharedEncoder(nn.Module):
    """Per-(UE, RBG) encoder: in -> 128 -> 128 -> 64 (no final activation).

    ``input_dim`` is 70, or 73 in queue mode; ActorCritic passes the right one.
    """

    def __init__(self, input_dim: int = 70, hidden: int = 128,
                 out_dim: int = 64):
        super().__init__()
        self.net = _mlp([input_dim, hidden, hidden, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScoreNet(nn.Module):
    """Per-candidate UE logit: e (64) + 7 scalars -> 1 logit."""

    def __init__(self, input_dim: int = 71, hidden: tuple = (128, 64)):
        super().__init__()
        self.net = _mlp([input_dim, hidden[0], hidden[1], 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NoUserHead(nn.Module):
    """Per-position no-user logit: pooled RBG (64) + 4 scalars -> 1 logit."""

    def __init__(self, input_dim: int = 68, hidden: tuple = (128, 64)):
        super().__init__()
        self.net = _mlp([input_dim, hidden[0], hidden[1], 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ValueHead(nn.Module):
    """Per-slot value head: pooled encoder + scalars -> V(s_t).

    ``input_dim`` = 2*64 pooled + 6 global scalars = 134, plus the v2
    structured features when cfg.ppo_critic_v2 (+27, or +33 in queue mode):
    161 or, for the Run4 queue runs, 167.
    """

    def __init__(self, input_dim: int = 134, hidden: tuple = (128, 64)):
        super().__init__()
        self.net = _mlp([input_dim, hidden[0], hidden[1], 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Shared-encoder actor (ScoreNet + NoUserHead) + critic (ValueHead).

    Methods
    -------
    encode(obs_t)
        Run encoder, return e [K, R, 64].
    value(e, obs_t, fixed_mask)
        Compute V(s_t).
    decode(obs, deterministic=False)
        Sequential layer-major sample-and-step. No grad. For rollout.
    deterministic_action(obs)
        Deterministic evaluation action without rollout statistics or critic.
    replay(obs, action_sequence, policy_decision_mask)
        Teacher-forcing recompute under current policy. With grad. For PPO update.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        eo = cfg.encoder_out_dim
        enc_in = 70 + (3 if cfg.queue_size > 1 else 0)   # Run4 queue fields
        self.encoder = SharedEncoder(input_dim=enc_in,
                                     hidden=cfg.encoder_hidden, out_dim=eo)
        self.score_net = ScoreNet(input_dim=eo + 7,
                                  hidden=cfg.score_net_hidden)
        self.no_user_head = NoUserHead(input_dim=eo + 4,
                                       hidden=cfg.no_user_head_hidden)
        # value input: 2*eo pooled + 6 scalars (+27 v2 features when enabled,
        # 33 in queue mode -- see build_value_feats_v2,
        # +6 more queue-pressure features in queue mode)
        v2_dim = 27 + (6 if cfg.queue_size > 1 else 0)
        v_in = 2 * eo + 6 + (v2_dim if cfg.ppo_critic_v2 else 0)
        self.value_head = ValueHead(input_dim=v_in,
                                    hidden=cfg.value_head_hidden)

        # --- return (value-target) normalizer ---------------------------------
        # The value head predicts in NORMALIZED return space; value() de-normalizes
        # back to raw reward units so GAE/bootstrap stay in raw space. Without this,
        # raw returns (~hundreds, from the dense reward x gamma=0.99 horizon) make
        # the value gradient ~10^4 -- it dominates the shared grad-norm clip and the
        # shared encoder, starving the policy. With it, value targets are O(1), the
        # network only learns the ripple (mu carries the DC level), and grad-norms
        # return to a normal range. mu/sigma are buffers (no grad), updated per
        # rollout from the observed returns (EMA; init on the first update).
        self.register_buffer("ret_mean", torch.zeros(()))
        self.register_buffer("ret_std", torch.ones(()))
        self.register_buffer("ret_count", torch.zeros(()))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------- shared forward ----------
    def encode(self, obs_t: dict) -> torch.Tensor:
        return self.encoder(build_encoder_input(obs_t, self.cfg))

    def value(self, e: torch.Tensor, obs_t: dict, fixed_mask: np.ndarray,
              slot: int = 0) -> torch.Tensor:
        vi = build_value_input(e, obs_t, self.cfg, fixed_mask,
                               slot).unsqueeze(0)
        v_norm = self.value_head(vi).squeeze(0)
        # de-normalize to raw reward units (mu/sigma are constants here)
        return v_norm * self.ret_std + self.ret_mean

    @torch.no_grad()
    def update_return_normalizer(self, returns) -> None:
        """EMA-update the return mean/std from a rollout's raw GAE returns.

        Called once per PPO update (before the value loss uses ret_std). The
        first call initializes; later calls adapt slowly so V's calibration
        barely shifts between updates (returns are ~stationary within a run).
        """
        r = torch.as_tensor(np.asarray(returns, dtype=np.float32),
                            device=self.ret_mean.device)
        bm, bs = r.mean(), r.std().clamp_min(1e-6)
        if float(self.ret_count) == 0:
            self.ret_mean.copy_(bm)
            self.ret_std.copy_(bs)
        else:
            self.ret_mean.mul_(0.99).add_(0.01 * bm)
            self.ret_std.mul_(0.99).add_(0.01 * bs)
        self.ret_count.add_(1)

    @torch.no_grad()
    def state_value(self, obs: dict) -> float:
        """V(s) for a raw env observation, without decoding any action.

        Used for the GAE truncation bootstrap at the episode time limit
        (the fixed-length episode end is a truncation, not a terminal state).
        """
        obs_t = obs_to_tensors(obs, self.device)
        e = self.encode(obs_t)
        return float(self.value(e, obs_t, obs["fixed_mask"], obs["slot"]))

    # ---------- per-position logit computation ----------
    def _position_logits_and_mask(
        self,
        obs_t: dict,
        e: torch.Tensor,
        r: int,
        l: int,
        S_r_r: set,
        in_slot_count: np.ndarray,
        temp_uncommit: torch.Tensor,
        return_ctx: bool = False,
    ):
        """Build logits [K+1] and valid_mask [K+1] for one free position.

        Returns (logits, valid_mask, pred_btx_per_ue) by default, so every
        existing caller -- including the external audit probes that unpack a
        3-tuple (Run1/scripts/numeric_scout.py, Run4 probe_ppo_mdp.py,
        probe_e4_nouser.py) -- is unaffected.

        With ``return_ctx=True`` two more elements follow: ``aux`` [K, 7] and
        ``no_user_aux`` [4], the head input contexts.
        They are returned so the rbg-major pass can CACHE them during decode
        (see ``emit_context``): under teacher forcing they are deterministic
        functions of (obs, stored actions), so the cached copies let
        ``replay_batch`` evaluate a whole minibatch of slots with one batched
        forward per head instead of ~28 sequential ones per slot.
        """
        cfg = self.cfg
        K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
        device = e.device
        r_norm = r / R
        l_norm = l / L
        cqi_fb_r = obs_t["cqi_fb"][:, r]
        active = obs_t["active"]

        pred_btx = predict_btx_torch(temp_uncommit, cqi_fb_r, cfg)        # [K]

        u_valid = (active
                   & (temp_uncommit > cfg.b_tx_epsilon)
                   & (pred_btx >= cfg.b_tx_epsilon)).clone()
        if S_r_r:
            sel_idx = torch.tensor(sorted(S_r_r), device=device,
                                   dtype=torch.long)
            u_valid[sel_idx] = False
        any_valid = bool(u_valid.any().item())

        # no-user validity per Section 4; under the forced full-rank
        # ablation (cfg.ppo_force_full_rank) the early-close option is
        # removed entirely: no-user is legal ONLY when nothing else is
        if not any_valid:
            no_user_valid = True
        elif cfg.ppo_force_full_rank or len(S_r_r) == 0:
            no_user_valid = False
        else:
            no_user_valid = True

        # OrthScore [K]
        ortho = ortho_score_torch(obs_t["direction_fb"], r, S_r_r, K)

        # ScoreNet input: e[:,r,:] + 7 scalars per UE
        in_slot_count_t = torch.from_numpy(in_slot_count.astype(np.float32)).to(device)
        aux = torch.stack([
            ortho,
            in_slot_count_t / R,
            temp_uncommit / cfg.b_norm,
            pred_btx / cfg.b_norm,
            torch.full((K,), r_norm, device=device, dtype=torch.float32),
            torch.full((K,), l_norm, device=device, dtype=torch.float32),
            torch.full((K,), len(S_r_r) / L, device=device, dtype=torch.float32),
        ], dim=-1)                                                        # [K, 7]
        score_in = torch.cat([e[:, r, :], aux], dim=-1)                   # [K, 71]
        ue_logits = self.score_net(score_in)                              # [K]

        # NoUserHead input
        g_r = e[:, r, :].mean(dim=0)                                       # [64]
        valid_count_pos = u_valid.to(torch.float32).sum() / cfg.num_ue
        no_user_aux = torch.stack([
            torch.tensor(r_norm, device=device, dtype=torch.float32),
            torch.tensor(l_norm, device=device, dtype=torch.float32),
            torch.tensor(len(S_r_r) / L, device=device, dtype=torch.float32),
            valid_count_pos,
        ])
        no_user_in = torch.cat([g_r, no_user_aux], dim=0).unsqueeze(0)    # [1, 68]
        no_user_logit = self.no_user_head(no_user_in).squeeze(0)          # scalar
        if cfg.ppo_no_user_scale != 1.0:      # down-weighting ablation
            no_user_logit = no_user_logit * cfg.ppo_no_user_scale

        logits = torch.cat([no_user_logit.unsqueeze(0), ue_logits])       # [K+1]
        valid_mask = torch.cat([
            torch.tensor([no_user_valid], device=device, dtype=torch.bool),
            u_valid,
        ])                                                                 # [K+1]
        if return_ctx:
            return logits, valid_mask, pred_btx, aux, no_user_aux
        return logits, valid_mask, pred_btx

    # ---------- rbg-major pass (decode AND replay share this one code path,
    # so rollout/replay in-slot state can only diverge through the weights;
    # trajectory-consistency asserts mirror the layer-major replay) ----------
    def _rbg_major_pass(self, obs: dict, deterministic: bool = False,
                        stored_actions: np.ndarray | None = None,
                        stored_mask: np.ndarray | None = None,
                        emit_context: bool = False,
                        action_only: bool = False) -> dict:
        """One slot's macro action.

        ``emit_context=True`` additionally returns, under key ``context``, the
        per-position head inputs this pass built anyway (see ``replay_batch``).
        Caching them during DECODE is exact rather than merely equivalent: the
        loop state (S_r, in_slot_count, planner._remaining) evolves purely from
        the actions taken, and replay teacher-forces those same actions, so the
        contexts a replay would rebuild are identical to the ones decode used.

        ``action_only=True`` shares the deterministic action and planner logic,
        but omits outputs needed only by rollout/replay. It cannot be used for
        sampling, teacher forcing, or context collection.
        """
        if action_only and (not deterministic or stored_actions is not None
                            or stored_mask is not None or emit_context):
            raise ValueError("action_only requires deterministic evaluation "
                             "without stored actions, mask, or context")
        cfg = self.cfg
        K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
        device = self.device
        teacher = stored_actions is not None

        obs_t = obs_to_tensors(obs, device)
        enc_in = build_encoder_input(obs_t, cfg)      # weight-independent
        e = self.encoder(enc_in)                      # == self.encode(obs_t)

        # exact same h_hat the env reconstructs for this slot's precoding
        # (obs carries copies of the very buffers env used + noise_var)
        nv = float(obs["noise_var"])
        h_hat = reconstruct_h_hat(obs["direction_fb"], obs["cqi_fb"],
                                  cfg.p_rbg, nv)
        planner = SlotAllocationPlanner(
            cfg, h_hat, nv, nv, obs["uncommitted"].astype(np.float64))

        S_r = [set(obs["initial_S_r"][r]) for r in range(R)]
        in_slot_count = np.zeros(K, dtype=np.int64)
        for r in range(R):
            for u in S_r[r]:
                in_slot_count[u] += 1

        fixed_mask = obs["fixed_mask"]
        fixed_alloc = obs["fixed_allocation"]
        action_seq = np.zeros((R, L), dtype=np.int64)
        policy_mask = None if action_only else np.zeros((R, L), dtype=bool)
        log_prob_sum = None if action_only else torch.zeros((), device=device)
        entropy_sum = None if action_only else torch.zeros((), device=device)
        per_logprobs: list[float] = []
        n_decisions = 0
        budget_trace = None if action_only else np.zeros((R, K))
        ctx_r: list[int] = []                 # emit_context accumulators
        ctx_aux: list[torch.Tensor] = []
        ctx_nu: list[torch.Tensor] = []
        ctx_valid: list[torch.Tensor] = []
        ctx_act: list[int] = []

        for r in range(R):
            fixed_ues = sorted(S_r[r])        # retx members of this RBG
            entries: list[int] = []
            closed = False
            for l in range(L):
                if fixed_mask[r, l]:
                    action_seq[r, l] = int(fixed_alloc[r, l])
                    if teacher:
                        assert not stored_mask[r, l], (
                            f"replay(rm): fixed retx position (r={r}, l={l}) "
                            "is marked policy_decision_mask=True")
                    continue
                if closed:
                    action_seq[r, l] = 0
                    if teacher:
                        assert not stored_mask[r, l], (
                            f"replay(rm): (r={r}, l={l}) in closed RBG "
                            "but policy_decision_mask=True")
                    continue

                # budget features/mask come from the planner's EXACT per-UE
                # remaining (float64, debited only at group closure)
                temp_uncommit = torch.from_numpy(
                    planner._remaining.astype(np.float32)).to(device)
                logits, valid_mask, _, aux, nu_aux = \
                    self._position_logits_and_mask(
                        obs_t, e, r, l, S_r[r], in_slot_count, temp_uncommit,
                        return_ctx=True)
                assert bool(valid_mask.any().item()), \
                    f"No valid action at (r={r}, l={l})"
                if emit_context:
                    ctx_r.append(r)
                    ctx_aux.append(aux)
                    ctx_nu.append(nu_aux)
                    ctx_valid.append(valid_mask)

                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                if action_only:
                    # Categorical normally rejects invalid logits. Preserve
                    # fail-closed evaluation when bypassing that constructor.
                    # A finite maximum permits individual -inf logits just
                    # as Categorical does, while rejecting NaN, +inf, and
                    # a distribution with only negative infinities.
                    if not bool(torch.isfinite(masked_logits.max()).item()):
                        raise ValueError("Nonfinite deterministic action logits")
                    dist = None
                else:
                    dist = torch.distributions.Categorical(logits=masked_logits)
                if teacher:
                    assert stored_mask[r, l], (
                        f"replay(rm): non-fixed, non-closed (r={r}, l={l}) "
                        "has policy_decision_mask=False")
                    action = int(stored_actions[r, l])
                    assert bool(valid_mask[action].item()), (
                        f"replay(rm): stored action {action} at (r={r}, "
                        f"l={l}) invalid under current mask")
                elif deterministic:
                    action = int(masked_logits.argmax().item())
                else:
                    action = int(dist.sample().item())
                action_seq[r, l] = action
                if not action_only:
                    action_t = torch.tensor(action, device=device,
                                            dtype=torch.long)
                    log_prob = dist.log_prob(action_t)
                    policy_mask[r, l] = True
                    n_decisions += 1
                    if emit_context:
                        ctx_act.append(action)
                    log_prob_sum = log_prob_sum + log_prob
                    entropy_sum = entropy_sum + dist.entropy()
                    per_logprobs.append(float(log_prob.item()))

                if action == 0:
                    closed = True                 # spec §5 RBG closure
                else:
                    u = action - 1
                    S_r[r].add(u)
                    in_slot_count[u] += 1
                    entries.append(u)

            # group final -> exact B_tx sizing + budget debit (once per RBG)
            planner.close_rbg(r, fixed_ues, entries)
            if not action_only:
                budget_trace[r] = planner._remaining

        if action_only:
            return dict(action_sequence=action_seq)

        # identical to self.value(e, obs_t, fixed_mask, obs["slot"]), but the
        # weight-independent tail is kept so emit_context can cache it
        v_tail = build_value_tail(obs_t, cfg, fixed_mask, obs["slot"])
        vi = torch.cat([e.mean(dim=(0, 1)), e.amax(dim=(0, 1)), v_tail],
                       dim=0).unsqueeze(0)
        value = self.value_head(vi).squeeze(0) * self.ret_std + self.ret_mean

        context = None
        if emit_context:
            assert len(ctx_act) == n_decisions == len(ctx_aux)
            context = dict(
                pos_rbg=np.asarray(ctx_r, dtype=np.int64),           # [P]
                action=np.asarray(ctx_act, dtype=np.int64),          # [P]
                aux=(torch.stack(ctx_aux).detach().cpu().numpy()
                     .astype(np.float32) if ctx_aux
                     else np.zeros((0, K, 7), np.float32)),          # [P,K,7]
                nu_aux=(torch.stack(ctx_nu).detach().cpu().numpy()
                        .astype(np.float32) if ctx_nu
                        else np.zeros((0, 4), np.float32)),          # [P,4]
                valid=(torch.stack(ctx_valid).detach().cpu().numpy()
                       if ctx_valid
                       else np.zeros((0, K + 1), bool)),             # [P,K+1]
                v_tail=v_tail.detach().cpu().numpy().astype(np.float32),
                enc_in=enc_in.detach().cpu().numpy().astype(np.float32),
            )                                                        # [K,R,73]

        return dict(
            action_sequence=action_seq,
            policy_decision_mask=policy_mask,
            log_prob_sum=log_prob_sum,
            entropy_sum=entropy_sum,
            per_subaction_logprobs=per_logprobs,
            num_policy_decisions=int(n_decisions),
            value=value,
            planned_btx=planner.planned_btx_map(),     # Gate 2
            budget_trace=budget_trace,                 # Gate 3
            context=context,
        )

    @torch.no_grad()
    def deterministic_action(self, obs: dict) -> np.ndarray:
        """Exact deterministic decode action, omitting unused eval statistics.

        RBG-major evaluation uses the same logits, validity masks, argmax and
        post-RZF budget closures as ``decode(deterministic=True)``. Other
        decoding orders keep their existing decoder. Neither path changes
        module mode, parameters, return normalizers, or sampling RNG state.
        """
        if self.cfg.decode_order == "rbg_major":
            return self._rbg_major_pass(
                obs, deterministic=True, action_only=True)["action_sequence"]
        return self.decode(obs, deterministic=True)["action_sequence"]

    # ---------- decode (rollout, sampling, no grad) ----------
    @torch.no_grad()
    def decode(self, obs: dict, deterministic: bool = False,
               emit_context: bool = False) -> dict:
        """Rollout decode. ``emit_context`` caches this slot's head inputs for
        the batched replay (rbg-major only; costs nothing when False)."""
        if self.cfg.decode_order == "rbg_major":
            out = self._rbg_major_pass(obs, deterministic=deterministic,
                                       emit_context=emit_context)
            return dict(
                action_sequence=out["action_sequence"],
                policy_decision_mask=out["policy_decision_mask"],
                log_prob_sum=float(out["log_prob_sum"].item()),
                per_subaction_logprobs=out["per_subaction_logprobs"],
                entropy_sum=float(out["entropy_sum"].item()),
                num_policy_decisions=out["num_policy_decisions"],
                value=float(out["value"].item()),
                planned_btx=out["planned_btx"],
                budget_trace=out["budget_trace"],
                context=out["context"],
            )
        cfg = self.cfg
        K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
        device = self.device

        obs_t = obs_to_tensors(obs, device)
        e = self.encode(obs_t)

        # In-slot context init (retx UEs already in S_r and in_slot_count)
        S_r = [set(obs["initial_S_r"][r]) for r in range(R)]
        in_slot_count = np.zeros(K, dtype=np.int64)
        for r in range(R):
            for u in S_r[r]:
                in_slot_count[u] += 1
        temp_uncommit = torch.from_numpy(obs["uncommitted"].astype(np.float32)
                                          ).to(device).clone()
        rbg_closed = [False] * R

        fixed_mask = obs["fixed_mask"]
        fixed_alloc = obs["fixed_allocation"]
        action_seq = np.zeros((R, L), dtype=np.int64)
        policy_mask = np.zeros((R, L), dtype=bool)
        log_prob_sum = torch.zeros((), device=device)
        entropy_sum = torch.zeros((), device=device)
        per_logprobs: list[float] = []
        n_decisions = 0

        for l in range(L):
            for r in range(R):
                if fixed_mask[r, l]:
                    action_seq[r, l] = int(fixed_alloc[r, l])
                    continue
                if rbg_closed[r]:
                    action_seq[r, l] = 0
                    continue

                logits, valid_mask, pred_btx = self._position_logits_and_mask(
                    obs_t, e, r, l, S_r[r], in_slot_count, temp_uncommit)
                assert bool(valid_mask.any().item()), \
                    f"No valid action at (r={r}, l={l})"

                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                if deterministic:
                    action = int(masked_logits.argmax().item())
                else:
                    action = int(dist.sample().item())
                action_t = torch.tensor(action, device=device, dtype=torch.long)
                log_prob = dist.log_prob(action_t)
                entropy = dist.entropy()

                action_seq[r, l] = action
                policy_mask[r, l] = True
                n_decisions += 1
                log_prob_sum = log_prob_sum + log_prob
                entropy_sum = entropy_sum + entropy
                per_logprobs.append(float(log_prob.item()))

                # update in-slot context
                if action == 0:
                    rbg_closed[r] = True
                else:
                    u = action - 1
                    S_r[r].add(u)
                    in_slot_count[u] += 1
                    # clamp to >= 0 for floating-point safety (mathematically
                    # pred_btx <= temp_uncommit, so this only guards FP noise)
                    new_u = torch.clamp(temp_uncommit[u] - pred_btx[u], min=0.0)
                    temp_uncommit = temp_uncommit.clone()
                    temp_uncommit[u] = new_u

        value = self.value(e, obs_t, fixed_mask, obs["slot"])

        return dict(
            action_sequence=action_seq,
            policy_decision_mask=policy_mask,
            log_prob_sum=float(log_prob_sum.item()),
            per_subaction_logprobs=per_logprobs,
            entropy_sum=float(entropy_sum.item()),
            num_policy_decisions=int(n_decisions),
            value=float(value.item()),
        )

    # ---------- replay (PPO update, teacher forcing, with grad) ----------
    # ---------- batched replay (PPO update fast path) ----------
    def replay_batch(self, contexts: list[dict]) -> dict:
        """Teacher-forced replay of a whole minibatch of slots at once.

        ``contexts`` are the per-slot dicts cached by ``decode(..., emit_
        context=True)``. Because every head input is a deterministic function
        of (obs, stored actions) -- neither depends on the network output --
        replaying N slots needs only FOUR batched forwards (encoder, ScoreNet,
        NoUserHead, ValueHead) instead of N x (1 + 2 x ~28) sequential ones.

        Returns the same four quantities ``replay`` does, stacked over the
        minibatch: log_prob_sum [B], entropy_sum [B], num_policy_decisions [B]
        (long), value [B].

        Numerically this differs from the sequential path only in float32
        reduction ORDER (a padded row-sum instead of a running scalar add over
        ~28 terms). Measured drift is ~1e-6 relative; the equivalence test
        gates it at 1e-4.
        """
        cfg = self.cfg
        K, R = cfg.num_ue, cfg.num_rbg
        device = self.device
        B = len(contexts)

        # --- encoder: one batched pass -----------------------------------
        enc_in = torch.from_numpy(
            np.stack([c["enc_in"] for c in contexts])).to(device)   # [B,K,R,73]
        e = self.encoder(enc_in)                                    # [B,K,R,64]

        # --- value head: one batched pass --------------------------------
        v_tail = torch.from_numpy(
            np.stack([c["v_tail"] for c in contexts])).to(device)   # [B, 39]
        vi = torch.cat([e.mean(dim=(1, 2)), e.amax(dim=(1, 2)), v_tail],
                       dim=1)                                       # [B, 167]
        value = self.value_head(vi) * self.ret_std + self.ret_mean  # [B]

        counts = np.array([len(c["action"]) for c in contexts], dtype=np.int64)
        n_dec = torch.from_numpy(counts).to(device)
        if counts.sum() == 0:                       # degenerate: no decisions
            z = value.new_zeros(B)
            return dict(log_prob_sum=z, entropy_sum=z.clone(),
                        num_policy_decisions=n_dec, value=value)

        # --- flatten the ragged positions --------------------------------
        p_max = int(counts.max())
        slot_of = np.repeat(np.arange(B, dtype=np.int64), counts)
        within = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
        slot_t = torch.from_numpy(slot_of).to(device)
        within_t = torch.from_numpy(within).to(device)
        r_t = torch.from_numpy(
            np.concatenate([c["pos_rbg"] for c in contexts])).to(device)
        act_t = torch.from_numpy(
            np.concatenate([c["action"] for c in contexts])).to(device)
        aux_t = torch.from_numpy(
            np.concatenate([c["aux"] for c in contexts])).to(device)   # [P,K,7]
        nu_t = torch.from_numpy(
            np.concatenate([c["nu_aux"] for c in contexts])).to(device)  # [P,4]
        val_t = torch.from_numpy(
            np.concatenate([c["valid"] for c in contexts])).to(device)  # [P,K+1]

        # e_pos[p] = e[slot_of[p], :, pos_rbg[p], :]; permute first so the two
        # advanced indices are adjacent (unambiguous result shape [P, K, 64])
        e_pos = e.permute(0, 2, 1, 3)[slot_t, r_t]                    # [P,K,64]

        # --- the two actor heads: one batched pass each -------------------
        # reshape to rank-2 so nn.Linear takes the same ATen path the
        # sequential [K, 71] / [1, 68] calls take (H10: rank-4 input can pick
        # an unfused addmm and round the bias twice)
        score_in = torch.cat([e_pos, aux_t], dim=-1)                  # [P,K,71]
        n_pos = score_in.shape[0]
        ue_logits = self.score_net(
            score_in.reshape(-1, score_in.shape[-1])).view(n_pos, K)  # [P, K]
        no_user_logit = self.no_user_head(
            torch.cat([e_pos.mean(dim=1), nu_t], dim=-1))             # [P]
        if cfg.ppo_no_user_scale != 1.0:      # must mirror the decode path
            no_user_logit = no_user_logit * cfg.ppo_no_user_scale

        logits = torch.cat([no_user_logit.unsqueeze(-1), ue_logits], dim=-1)
        masked = logits.masked_fill(~val_t, -1e9)                     # [P,K+1]
        # Three guards, folded into ONE device sync per minibatch.
        #  (1) Categorical validates its logits and raises on NaN; the manual
        #      log_softmax below would silently propagate it into every gradient.
        #  (2)+(3) restore the sequential path's trajectory-consistency asserts,
        #      which the cached-context path would otherwise retire. Without them
        #      a ctx/traj desync stays silent: the +-20 log-ratio clamp in
        #      ppo_update turns a -1e9 log-prob into a finite, plausible loss.
        _finite = torch.isfinite(masked).all()
        _any_valid = val_t.any(dim=-1).all()
        _act_valid = val_t.gather(-1, act_t.unsqueeze(-1)).all()
        assert bool((_finite & _any_valid & _act_valid).item()), (
            "replay_batch consistency: "
            f"finite={bool(_finite)} any_valid={bool(_any_valid)} "
            f"stored_action_valid={bool(_act_valid)}")
        # torch.distributions.Categorical(logits=masked) normalizes with
        # log_softmax and defines log_prob/entropy off the normalized logits;
        # reproduce exactly, including the (logits * probs) product order.
        logp = torch.log_softmax(masked, dim=-1)
        log_prob = logp.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)   # [P]
        entropy = -(logp * logp.exp()).sum(dim=-1)                    # [P]

        # --- deterministic segment sum (unique scatter, then row sum) -----
        lp_pad = log_prob.new_zeros(B, p_max)
        lp_pad[slot_t, within_t] = log_prob
        en_pad = entropy.new_zeros(B, p_max)
        en_pad[slot_t, within_t] = entropy
        return dict(log_prob_sum=lp_pad.sum(dim=1),
                    entropy_sum=en_pad.sum(dim=1),
                    num_policy_decisions=n_dec,
                    value=value)

    def replay(self, obs: dict, action_sequence: np.ndarray,
               policy_decision_mask: np.ndarray) -> dict:
        if self.cfg.decode_order == "rbg_major":
            out = self._rbg_major_pass(obs, stored_actions=action_sequence,
                                       stored_mask=policy_decision_mask)
            return dict(
                log_prob_sum=out["log_prob_sum"],
                entropy_sum=out["entropy_sum"],
                num_policy_decisions=out["num_policy_decisions"],
                value=out["value"],
            )
        cfg = self.cfg
        K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
        device = self.device

        obs_t = obs_to_tensors(obs, device)
        e = self.encode(obs_t)

        S_r = [set(obs["initial_S_r"][r]) for r in range(R)]
        in_slot_count = np.zeros(K, dtype=np.int64)
        for r in range(R):
            for u in S_r[r]:
                in_slot_count[u] += 1
        temp_uncommit = torch.from_numpy(obs["uncommitted"].astype(np.float32)
                                          ).to(device).clone()
        rbg_closed = [False] * R

        fixed_mask = obs["fixed_mask"]
        log_prob_sum = torch.zeros((), device=device)
        entropy_sum = torch.zeros((), device=device)
        n_decisions = 0

        for l in range(L):
            for r in range(R):
                stored_action = int(action_sequence[r, l])
                was_policy = bool(policy_decision_mask[r, l])

                if fixed_mask[r, l]:
                    # fixed retx position MUST NOT be marked as a policy decision
                    assert not was_policy, (
                        f"replay: fixed retx position (r={r}, l={l}) "
                        "is marked policy_decision_mask=True")
                    continue
                if rbg_closed[r]:
                    # auto no-user after closure -- must NOT be a policy decision
                    assert not was_policy, (
                        f"replay: position (r={r}, l={l}) is in closed RBG "
                        "but policy_decision_mask=True")
                    continue

                # otherwise the position MUST be a policy decision
                assert was_policy, (
                    f"replay: non-fixed, non-closed position (r={r}, l={l}) "
                    "has policy_decision_mask=False -- trajectory inconsistent")

                logits, valid_mask, pred_btx = self._position_logits_and_mask(
                    obs_t, e, r, l, S_r[r], in_slot_count, temp_uncommit)
                # stored action must be valid under the *current* state mask
                # (replay reconstructs identical in-slot state, so the mask
                # should match rollout; failure here means trajectory drift)
                assert bool(valid_mask[stored_action].item()), (
                    f"replay: stored action {stored_action} at (r={r}, l={l}) "
                    "is invalid under current mask -- trajectory inconsistent")

                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_t = torch.tensor(stored_action, device=device,
                                        dtype=torch.long)
                log_prob_sum = log_prob_sum + dist.log_prob(action_t)
                entropy_sum = entropy_sum + dist.entropy()
                n_decisions += 1

                if stored_action == 0:
                    rbg_closed[r] = True
                else:
                    u = stored_action - 1
                    S_r[r].add(u)
                    in_slot_count[u] += 1
                    # clamp to >= 0 for floating-point safety
                    new_u = torch.clamp(temp_uncommit[u] - pred_btx[u], min=0.0)
                    temp_uncommit = temp_uncommit.clone()
                    temp_uncommit[u] = new_u

        value = self.value(e, obs_t, fixed_mask, obs["slot"])

        return dict(
            log_prob_sum=log_prob_sum,
            entropy_sum=entropy_sum,
            num_policy_decisions=int(n_decisions),
            value=value,
        )


# ---------------------------------------------------------------------------
# smoke test (run inside the GPU container)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import phase2_debug_config
    from env import SchedulerEnv

    cfg = phase2_debug_config()
    print(f"policy.py smoke test  K={cfg.num_ue}  R={cfg.num_rbg}  "
          f"Lmax={cfg.l_max}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    env = SchedulerEnv(cfg)
    obs = env.reset(0)

    ac = ActorCritic(cfg).to(device)
    n_params = sum(p.numel() for p in ac.parameters())
    print(f"  ActorCritic params: {n_params:,}")

    # rollout decode
    out = ac.decode(obs, deterministic=False)
    print(f"  decode: log_prob_sum={out['log_prob_sum']:.4f}  "
          f"entropy_sum={out['entropy_sum']:.4f}  "
          f"n_decisions={out['num_policy_decisions']}  "
          f"value={out['value']:.4f}")
    assert out["action_sequence"].shape == (cfg.num_rbg, cfg.l_max)
    assert out["policy_decision_mask"].shape == (cfg.num_rbg, cfg.l_max)
    assert out["num_policy_decisions"] == len(out["per_subaction_logprobs"])

    # replay should recover same logprob_sum if policy weights unchanged
    rep = ac.replay(obs, out["action_sequence"], out["policy_decision_mask"])
    diff = abs(out["log_prob_sum"] - float(rep["log_prob_sum"].item()))
    print(f"  replay vs decode: log_prob diff={diff:.6g} (should be ~0)")
    assert diff < 1e-3, diff
    assert rep["num_policy_decisions"] == out["num_policy_decisions"]
    assert abs(out["value"] - float(rep["value"].item())) < 1e-3

    # backward should work
    loss = -rep["log_prob_sum"] + 0.5 * rep["value"].pow(2)
    loss.backward()
    print(f"  backward ok, loss={float(loss.item()):.4f}")

    # env.step should accept the action_sequence
    next_obs, reward, done, info = env.step(out["action_sequence"])
    print(f"  env.step: reward={reward:.4f}  done={done}")

    print("policy.py smoke test passed.")
