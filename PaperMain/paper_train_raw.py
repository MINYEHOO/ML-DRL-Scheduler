#!/usr/bin/env python3
"""Additive RAW-INPUT recipe for the protected PaperMain trainer.

Use the standard training flags with ``--recipe raw_invariant``. No core
source file changes: this module subclasses ``policy.ActorCritic`` and installs
the subclass only inside its own process, so every existing run, digest and
resume/eval guard stays exactly as it was. Like ``paper_train_variants.py`` it
records its own digest in ``variant_provenance``; evaluation of a raw run must
go through ``load_model(manifest)`` here, because the checkpoint's
state_dict has raw-input layer shapes that the stock ``ActorCritic`` rejects.

WHAT THE ABLATION CHANGES (and only this):

  encoder      SharedEncoder(73 -> 128 -> 128 -> 64) is replaced by identity:
               every head sees the 73 raw per-(UE,RBG) channels.
  ScoreNet     input 64+7 -> 73+7. Per-UE, so phase carries no problem here.
  NoUserHead   the stock head pools the 64-d embedding over UEs. Pooling RAW
               input would average 64 PMI real/imag channels whose per-UE
               phases are arbitrary, i.e. ~0: the head would go blind to the
               channel geometry that decides whether to stop filling an RBG.
               Instead it receives (a) the mean over UEs of the 9 NON-PMI
               channels and (b) four PHASE-INVARIANT summaries of the valid
               candidate set at this position (best CQI, best OrthoScore vs
               the already-selected set, mean OrthoScore, best predicted
               B_tx), plus the stock 4 position scalars: 9 + 4 + 4 = 17.
  ValueHead    same pooling problem, same cure: mean/max of the 9 non-PMI
               channels (18) + three phase-invariant channel-correlation
               statistics over active UE pairs (mean, max, fraction above the
               SUS threshold) appended to the weight-independent tail
               (39 -> 42): 18 + 42 = 60.

Everything else -- masks, budgets, planner, HARQ, PPO, batched replay, the
batched-vs-sequential cross-check -- is inherited unchanged. The batched path
keeps working because the head wrappers slice the pooled raw vector
themselves, so ``replay_batch`` needs no edits.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import paper_train as pt

ROOT = Path(__file__).resolve().parent
RECIPE = 'raw_invariant'
RECIPES = {RECIPE: {'default_updates': 1000, 'lr_final': 0.0, 'lr_decay_updates': 1000}}
PMI_CHANNELS = 64          # direction_fb real (32) + imag (32) in enc_in[..., :64]
NON_PMI = slice(64, 73)    # cqi, age, backlog, deadline, avg_thr, active, qlen, qbits, nxt
NO_USER_AUX = 8            # 4 stock position scalars + 4 invariant candidate summaries
VALUE_EXTRA = 3            # correlation stats appended to the value tail
CORR_THRESHOLD = 0.25      # |h_u^H h_v|^2 > 0.25  <=>  OrthoScore < 0.75 (SUS T*)
ARCHITECTURE = {
    'schema_version': 1,
    'encoder': 'identity (raw 73-channel per-(UE,RBG) input)',
    'score_net_input': 73 + 7,
    'no_user_head_input': 9 + NO_USER_AUX,
    'no_user_aux': ['r_norm', 'l_norm', 'group_size_norm', 'valid_count',
                    'max_valid_cqi_norm', 'max_valid_ortho', 'mean_valid_ortho',
                    'max_valid_pred_btx_norm'],
    'value_head_input': 2 * 9 + 39 + VALUE_EXTRA,
    'value_tail_extra': ['pair_corr_mean', 'pair_corr_max',
                         'pair_corr_frac_above_0.25'],
    'phase_invariant_pooling': True,
}
_CORE_PLAN = pt.plan
_INSTALLED = False


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _build():
    """Construct the subclass lazily so importing this module stays cheap."""
    import numpy as np
    import torch
    from torch import nn
    import policy as P

    class RawNoUserHead(nn.Module):
        """Takes cat([mean_e(73), no_user_aux(8)]) exactly as the stock call
        sites hand it over, keeps the 9 non-PMI means + 8 scalars."""

        def __init__(self, hidden):
            super().__init__()
            self.net = P._mlp([9 + NO_USER_AUX, hidden[0], hidden[1], 1])

        def forward(self, x):
            keep = torch.cat([x[..., NON_PMI], x[..., 73:73 + NO_USER_AUX]], dim=-1)
            return self.net(keep).squeeze(-1)

    class RawValueHead(nn.Module):
        """Takes cat([mean_e(73), amax_e(73), tail(42)]); keeps 9 + 9 + 42."""

        def __init__(self, hidden, tail_dim):
            super().__init__()
            self.tail_dim = tail_dim
            self.net = P._mlp([2 * 9 + tail_dim, hidden[0], hidden[1], 1])

        def forward(self, x):
            keep = torch.cat([x[..., NON_PMI], x[..., 73 + 64:73 + 73],
                              x[..., 146:146 + self.tail_dim]], dim=-1)
            return self.net(keep).squeeze(-1)

    stock_tail = P.build_value_tail

    def pair_corr_stats(direction_fb, active):
        """Three phase-invariant statistics of |h_u^H h_v|^2 over ACTIVE UE
        pairs, pooled over RBGs. direction_fb [K,R,M] complex, active [K]."""
        K = direction_fb.shape[0]
        act = active.bool()
        n = int(act.sum().item())
        dev = direction_fb.device
        if n < 2:
            return torch.zeros(3, device=dev, dtype=torch.float32)
        d = direction_fb[act].permute(1, 0, 2)                        # [R,n,M]
        g = (d @ d.conj().transpose(-1, -2)).abs().pow(2).real           # [R,n,n]
        off = ~torch.eye(n, dtype=torch.bool, device=dev)
        vals = g[:, off].to(torch.float32)                               # [R, n*(n-1)]
        return torch.stack([vals.mean(), vals.max(),
                            (vals > CORR_THRESHOLD).to(torch.float32).mean()])

    def raw_value_tail(obs_t, cfg, fixed_mask, slot=0):
        base = stock_tail(obs_t, cfg, fixed_mask, slot)
        return torch.cat([base, pair_corr_stats(obs_t["direction_fb"],
                                                obs_t["active"])])

    class RawInvariantActorCritic(P.ActorCritic):
        """Stock ActorCritic with the encoder removed and phase-invariant
        pooling for the two pooled heads. See module docstring."""

        def __init__(self, cfg):
            super().__init__(cfg)
            if cfg.queue_size <= 1:
                raise ValueError('raw_invariant is defined for the Run4 queue world (73 channels)')
            if not cfg.ppo_critic_v2:
                raise ValueError('raw_invariant expects ppo_critic_v2 (39-dim value tail)')
            self.encoder = nn.Identity()
            self.score_net = P.ScoreNet(input_dim=73 + 7, hidden=cfg.score_net_hidden)
            self.no_user_head = RawNoUserHead(cfg.no_user_head_hidden)
            tail_dim = 6 + 27 + 6 + VALUE_EXTRA                          # 42
            self.value_head = RawValueHead(cfg.value_head_hidden, tail_dim)

        # identical to the stock method except: ScoreNet sees raw e (73+7)
        # and NoUserHead gets 8 scalars (4 stock + 4 invariant summaries)
        def _position_logits_and_mask(self, obs_t, e, r, l, S_r_r, in_slot_count,
                                      temp_uncommit, return_ctx=False):
            cfg = self.cfg
            K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
            device = e.device
            r_norm = r / R
            l_norm = l / L
            cqi_fb_r = obs_t["cqi_fb"][:, r]
            active = obs_t["active"]

            pred_btx = P.predict_btx_torch(temp_uncommit, cqi_fb_r, cfg)      # [K]

            u_valid = (active
                       & (temp_uncommit > cfg.b_tx_epsilon)
                       & (pred_btx >= cfg.b_tx_epsilon)).clone()
            if S_r_r:
                sel_idx = torch.tensor(sorted(S_r_r), device=device, dtype=torch.long)
                u_valid[sel_idx] = False
            any_valid = bool(u_valid.any().item())

            if not any_valid:
                no_user_valid = True
            elif cfg.ppo_force_full_rank or len(S_r_r) == 0:
                no_user_valid = False
            else:
                no_user_valid = True

            ortho = P.ortho_score_torch(obs_t["direction_fb"], r, S_r_r, K)

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
            score_in = torch.cat([e[:, r, :], aux], dim=-1)                   # [K, 80]
            ue_logits = self.score_net(score_in)                              # [K]

            # NoUserHead: pooled raw (the wrapper keeps only the 9 non-PMI
            # means) + 4 stock scalars + 4 phase-invariant candidate summaries
            g_r = e[:, r, :].mean(dim=0)                                       # [73]
            vf = u_valid.to(torch.float32)
            n_valid = vf.sum()
            valid_count_pos = n_valid / cfg.num_ue
            zero = torch.zeros((), device=device, dtype=torch.float32)
            if any_valid:
                neg = torch.finfo(torch.float32).min
                max_cqi = torch.where(u_valid, cqi_fb_r, torch.full_like(cqi_fb_r, neg)).max() / cfg.cqi_norm_const
                max_ortho = torch.where(u_valid, ortho, torch.full_like(ortho, neg)).max()
                mean_ortho = (ortho * vf).sum() / n_valid
                max_btx = torch.where(u_valid, pred_btx, torch.full_like(pred_btx, neg)).max() / cfg.b_norm
            else:
                max_cqi = max_ortho = mean_ortho = max_btx = zero
            no_user_aux = torch.stack([
                torch.tensor(r_norm, device=device, dtype=torch.float32),
                torch.tensor(l_norm, device=device, dtype=torch.float32),
                torch.tensor(len(S_r_r) / L, device=device, dtype=torch.float32),
                valid_count_pos,
                max_cqi.to(torch.float32), max_ortho.to(torch.float32),
                mean_ortho.to(torch.float32), max_btx.to(torch.float32),
            ])                                                                 # [8]
            no_user_in = torch.cat([g_r, no_user_aux], dim=0).unsqueeze(0)    # [1, 81]
            no_user_logit = self.no_user_head(no_user_in).squeeze(0)
            if cfg.ppo_no_user_scale != 1.0:
                no_user_logit = no_user_logit * cfg.ppo_no_user_scale

            logits = torch.cat([no_user_logit.unsqueeze(0), ue_logits])       # [K+1]
            valid_mask = torch.cat([
                torch.tensor([no_user_valid], device=device, dtype=torch.bool),
                u_valid,
            ])
            if return_ctx:
                return logits, valid_mask, pred_btx, aux, no_user_aux
            return logits, valid_mask, pred_btx

    return RawInvariantActorCritic, raw_value_tail


def _extra_baselines(cfg):
    """The two strongest rule comparators, appended to the trainer's periodic
    validation so eval_metrics.csv carries them next to PPO (user request
    2026-09-16). Same cfg (and hence the same SUS threshold) as the stock
    baseline rows; evaluated on the same fixed validation episodes."""
    import baselines as B
    return [B.SUSCQIFeasible(), B.SUSRPS()]


EXTRA_EVAL_BASELINES = ['SUS+CQI-Feasible', 'SUS-RPS']


def install():
    """Route every ActorCritic construction in this process to the raw class
    and extend the trainer's validation baseline list.

    Must run before ``train_phase2`` is imported (it binds ``ActorCritic`` and
    ``all_baselines`` at module import); modules already imported are patched
    too.
    """
    global _INSTALLED
    import policy as P
    import baselines as B
    if _INSTALLED:
        return P.ActorCritic
    cls, tail = _build()
    P.ActorCritic = cls
    P.build_value_tail = tail          # build_value_input/_rbg_major_pass look this up
    stock_all = B.all_baselines

    def all_baselines_plus(cfg):
        out = list(stock_all(cfg))
        have = {s.name for s in out}
        out += [s for s in _extra_baselines(cfg) if s.name not in have]
        return out

    B.all_baselines = all_baselines_plus
    for name in ('train_phase2', 'ppo', 'paper_eval', 'paper_run_eval'):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if hasattr(mod, 'ActorCritic'):
            mod.ActorCritic = cls
        if hasattr(mod, 'all_baselines'):
            mod.all_baselines = all_baselines_plus
    _INSTALLED = True
    return cls


def load_model(manifest, checkpoint_path, torch=None):
    """Construct the raw model for a run of this recipe and load its weights."""
    if manifest.get('recipe') != RECIPE:
        raise ValueError('load_model is for raw_invariant runs only')
    if torch is None:
        import torch
    from config import Config
    cls = install()
    model = cls(Config(**manifest['config']))
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, ckpt


# ---------------------------------------------------------------------------
# protected-trainer plumbing (mirrors paper_train_variants.py)
# ---------------------------------------------------------------------------
@contextmanager
def registered_recipes():
    previous = pt.RECIPES
    for name, recipe in RECIPES.items():
        if name in previous and previous[name] != recipe:
            raise ValueError(f'Conflicting variant recipe registration: {name}')
    pt.RECIPES = {**previous, **RECIPES}
    try:
        yield
    finally:
        pt.RECIPES = previous


def parser():
    with registered_recipes():
        return pt.parser()


def _recipe_record(root):
    """The recipe config must equal frozen Base EXACTLY: the ablation changes
    the network, not the world or the training hyperparameters."""
    root = Path(root).resolve()
    base_file = root / 'artifacts/base/config.json'
    recipe_file = root / 'artifacts' / RECIPE / 'config.json'
    base = json.loads(base_file.read_text())
    recipe = json.loads(recipe_file.read_text())
    if pt.canonical(recipe) != pt.canonical(base):
        raise ValueError('raw_invariant config must be identical to frozen Base; only the network differs')
    return recipe, {
        'schema_version': 1,
        'name': RECIPE,
        'source_sha256': {'paper_train_raw.py': pt.sha_file(root / 'paper_train_raw.py')},
        'base_recipe_sha256': pt.sha_file(base_file),
        'derived_recipe_sha256': pt.sha_file(recipe_file),
        'changed_ranges': {},
        'extra_eval_baselines': EXTRA_EVAL_BASELINES,
        'architecture': ARCHITECTURE,
        'architecture_sha256': hashlib.sha256(pt.canonical(ARCHITECTURE).encode()).hexdigest(),
    }


def validate_manifest(manifest, root=ROOT):
    if manifest.get('recipe') != RECIPE:
        raise ValueError('Variant validation requires recipe raw_invariant')
    root = Path(root).resolve()
    raw, expected_record = _recipe_record(root)
    if pt.canonical(manifest.get('variant_provenance')) != pt.canonical(expected_record):
        raise ValueError('Variant provenance changed or is missing; preserve this run and use its original variant sources')
    if manifest.get('source_sha256') != pt.source_hashes(root):
        raise ValueError('Training source changed for the variant run')
    if manifest.get('artifact_config_sha256') != expected_record['derived_recipe_sha256']:
        raise ValueError('Frozen variant configuration changed')
    cfg = manifest['config']
    expected = pt.normalize_config(raw)
    expected['seed'] = cfg['seed']
    expected['ppo_batched_replay'] = cfg['ppo_batched_replay']
    if cfg['traffic_model'] != 'bernoulli':
        raise ValueError('raw_invariant requires the original Bernoulli traffic')
    if manifest.get('calibration_profile') is not None:
        from calibration.profile import apply_profile
        expected = apply_profile(expected, manifest['calibration_profile']['document'])
    smoke = manifest['smoke_slots']
    if smoke is not None:
        expected.update(episode_len_main=smoke, episode_len_debug=smoke,
                        ppo_epochs=1, ppo_minibatch_size=max(1, smoke - 8),
                        ppo_eval_every=1, ppo_eval_episodes=1, ppo_save_every=1)
    if pt.canonical(cfg) != pt.canonical(expected):
        raise ValueError('Variant run configuration differs beyond the permitted recipe, seed, replay, calibration and smoke settings')
    return expected_record


def plan(args, root=ROOT):
    if args.recipe != RECIPE:
        return _CORE_PLAN(args, root)
    _, record = _recipe_record(root)
    if args.traffic_model not in (None, 'bernoulli'):
        raise ValueError('raw_invariant requires the original Bernoulli traffic')
    with registered_recipes():
        run_dir, checkpoint, manifest = _CORE_PLAN(args, root)
    if checkpoint is None:
        manifest['variant_provenance'] = record
        if manifest.get('calibration_profile') is not None:
            manifest['calibration_application'] = (
                'Bernoulli Base-calibrated beta, unchanged world; the raw-input '
                'network is the only difference from Base')
    validate_manifest(manifest, root)
    return run_dir, checkpoint, manifest


def main(argv=None):
    install()
    previous_plan = pt.plan
    with registered_recipes():
        pt.plan = plan
        try:
            return pt.main(argv)
        finally:
            pt.plan = previous_plan


if __name__ == '__main__':
    main()
