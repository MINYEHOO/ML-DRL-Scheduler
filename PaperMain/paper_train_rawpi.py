#!/usr/bin/env python3
"""Additive PHASE-INVARIANT raw-input recipe for the protected PaperMain trainer.

Recipes
    raw_phase_inv        config identical to frozen Base
    raw_phase_inv_lr6    config differs from Base ONLY in ppo_learning_rate
                         (3e-4 -> 6e-4), to spend the trust-region headroom the
                         raw runs leave unused (run 30: KL 0.0040 against a
                         0.02 target, clip 0.046 against Base's 0.048 at 2x the
                         gradient norm).

WHY (measured, not assumed). Multiplying a UE's fed-back direction by an
arbitrary global phase changes nothing physical: RZF, post-RZF SINR, OrthoScore
and B_tx are all invariant. Yet run 30's ScoreNet takes the PMI as 32 real + 32
imaginary channels, which are NOT invariant, and on its own checkpoint a pure
phase rotation changed the macro action in 39 of 40 slots, moving UE logits by
2.09 on average. The network therefore spends capacity on a direction that
carries no information, and sees the same physical state as many different
inputs. This is inherited from the stock encoder input; the encoder-free run
merely exposed it.

FIX. Replace the 64 real/imag PMI channels with the ADJACENT AUTOCORRELATION
    a[m] = h[m] * conj(h[m+1]),  m = 0 .. M-2      (M = 32 antennas)
carried as real and imaginary parts: 62 channels. Under h -> h*exp(j*theta)
the exp cancels exactly (measured residual 1.6e-8), while the inter-antenna
phase differences -- i.e. the beam direction on a uniform array -- are kept.
Taking |h[m]|^2 instead would also be invariant but would discard those
differences and, on a uniform array, carry almost no information; that was an
earlier, wrong proposal.

SCALE. The raw autocorrelation has std 0.0224 against 0.1250 for the stock PMI
channels and 0.3764 for the nine non-PMI ones, so it is multiplied by 8 to land
at 0.180, inside the same band. Encoder input is 62 + 9 = 71 channels.

Everything else matches paper_train_raw.py: no encoder, ScoreNet per-UE (71+7),
NoUserHead and ValueHead fed phase-invariant pooled summaries, batched replay
untouched. Core sources are not modified; run 30 and its module digest are
unaffected.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import paper_train as pt

ROOT = Path(__file__).resolve().parent
RECIPE_BASE = 'raw_phase_inv'
RECIPE_LR6 = 'raw_phase_inv_lr6'
RECIPES = {RECIPE_BASE: {'default_updates': 1000, 'lr_final': 0.0, 'lr_decay_updates': 1000},
           RECIPE_LR6: {'default_updates': 1000, 'lr_final': 0.0, 'lr_decay_updates': 1000}}
# per-recipe: the ONLY keys allowed to differ from frozen Base
EXPECTED_DIFF = {RECIPE_BASE: {}, RECIPE_LR6: {'ppo_learning_rate': 6e-4}}
AUTOCORR_SCALE = 8.0
RAW_DIM = 71               # 62 autocorrelation + 9 non-PMI
NON_PMI = slice(62, 71)
NO_USER_AUX = 8
VALUE_EXTRA = 3
VALUE_TAIL = 6 + 27 + 6 + VALUE_EXTRA          # 42
CORR_THRESHOLD = 0.25
ARCHITECTURE = {
    'schema_version': 1,
    'encoder': 'identity',
    'channel_representation': 'adjacent autocorrelation h[m]*conj(h[m+1]), real+imag, x8',
    'channel_channels': 62,
    'global_phase_invariant': True,
    'encoder_input': RAW_DIM,
    'score_net_input': RAW_DIM + 7,
    'no_user_head_input': 9 + NO_USER_AUX,
    'value_head_input': 2 * 9 + VALUE_TAIL,
    'value_tail_extra': ['pair_corr_mean', 'pair_corr_max', 'pair_corr_frac_above_0.25'],
}
EXTRA_EVAL_BASELINES = ['SUS+CQI-Feasible', 'SUS-RPS']
_CORE_PLAN = pt.plan
_INSTALLED = False


def _build():
    import numpy as np
    import torch
    from torch import nn
    import policy as P

    stock_enc_in = P.build_encoder_input
    stock_tail = P.build_value_tail

    def phase_invariant_encoder_input(obs_t, cfg):
        """[K, R, 71]: 62 phase-invariant channel features + the 9 non-PMI ones,
        the latter taken verbatim from the stock builder so they cannot drift."""
        stock = stock_enc_in(obs_t, cfg)                       # [K,R,73]
        d = obs_t["direction_fb"]                              # [K,R,M] complex
        a = d[..., :-1] * d[..., 1:].conj()                    # [K,R,M-1]
        return torch.cat([a.real * AUTOCORR_SCALE, a.imag * AUTOCORR_SCALE,
                          stock[..., 64:]], dim=-1)            # [K,R,71]

    class RawNoUserHead(nn.Module):
        """cat([mean_e(71), no_user_aux(8)]) -> keep 9 non-PMI means + 8 scalars."""

        def __init__(self, hidden):
            super().__init__()
            self.net = P._mlp([9 + NO_USER_AUX, hidden[0], hidden[1], 1])

        def forward(self, x):
            keep = torch.cat([x[..., NON_PMI], x[..., RAW_DIM:RAW_DIM + NO_USER_AUX]], dim=-1)
            return self.net(keep).squeeze(-1)

    class RawValueHead(nn.Module):
        """cat([mean_e(71), amax_e(71), tail(42)]) -> keep 9 + 9 + 42."""

        def __init__(self, hidden):
            super().__init__()
            self.net = P._mlp([2 * 9 + VALUE_TAIL, hidden[0], hidden[1], 1])

        def forward(self, x):
            keep = torch.cat([x[..., NON_PMI],
                              x[..., RAW_DIM + 62:RAW_DIM + RAW_DIM],
                              x[..., 2 * RAW_DIM:2 * RAW_DIM + VALUE_TAIL]], dim=-1)
            return self.net(keep).squeeze(-1)

    def pair_corr_stats(direction_fb, active):
        K = direction_fb.shape[0]
        act = active.bool()
        n = int(act.sum().item())
        dev = direction_fb.device
        if n < 2:
            return torch.zeros(3, device=dev, dtype=torch.float32)
        d = direction_fb[act].permute(1, 0, 2)
        g = (d @ d.conj().transpose(-1, -2)).abs().pow(2).real
        off = ~torch.eye(n, dtype=torch.bool, device=dev)
        vals = g[:, off].to(torch.float32)
        return torch.stack([vals.mean(), vals.max(),
                            (vals > CORR_THRESHOLD).to(torch.float32).mean()])

    def raw_value_tail(obs_t, cfg, fixed_mask, slot=0):
        return torch.cat([stock_tail(obs_t, cfg, fixed_mask, slot),
                          pair_corr_stats(obs_t["direction_fb"], obs_t["active"])])

    class PhaseInvariantActorCritic(P.ActorCritic):
        def __init__(self, cfg):
            super().__init__(cfg)
            if cfg.queue_size <= 1 or not cfg.ppo_critic_v2:
                raise ValueError('raw_phase_inv expects the Run4 queue world with critic v2')
            self.encoder = nn.Identity()
            self.score_net = P.ScoreNet(input_dim=RAW_DIM + 7, hidden=cfg.score_net_hidden)
            self.no_user_head = RawNoUserHead(cfg.no_user_head_hidden)
            self.value_head = RawValueHead(cfg.value_head_hidden)

        def _position_logits_and_mask(self, obs_t, e, r, l, S_r_r, in_slot_count,
                                      temp_uncommit, return_ctx=False):
            cfg = self.cfg
            K, R, L = cfg.num_ue, cfg.num_rbg, cfg.l_max
            device = e.device
            r_norm, l_norm = r / R, l / L
            cqi_fb_r = obs_t["cqi_fb"][:, r]
            active = obs_t["active"]
            pred_btx = P.predict_btx_torch(temp_uncommit, cqi_fb_r, cfg)

            u_valid = (active & (temp_uncommit > cfg.b_tx_epsilon)
                       & (pred_btx >= cfg.b_tx_epsilon)).clone()
            if S_r_r:
                u_valid[torch.tensor(sorted(S_r_r), device=device, dtype=torch.long)] = False
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
            ], dim=-1)
            ue_logits = self.score_net(torch.cat([e[:, r, :], aux], dim=-1))

            g_r = e[:, r, :].mean(dim=0)
            vf = u_valid.to(torch.float32)
            n_valid = vf.sum()
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
                n_valid / cfg.num_ue,
                max_cqi.to(torch.float32), max_ortho.to(torch.float32),
                mean_ortho.to(torch.float32), max_btx.to(torch.float32),
            ])
            no_user_logit = self.no_user_head(
                torch.cat([g_r, no_user_aux], dim=0).unsqueeze(0)).squeeze(0)
            if cfg.ppo_no_user_scale != 1.0:
                no_user_logit = no_user_logit * cfg.ppo_no_user_scale

            logits = torch.cat([no_user_logit.unsqueeze(0), ue_logits])
            valid_mask = torch.cat([
                torch.tensor([no_user_valid], device=device, dtype=torch.bool), u_valid])
            if return_ctx:
                return logits, valid_mask, pred_btx, aux, no_user_aux
            return logits, valid_mask, pred_btx

    return PhaseInvariantActorCritic, phase_invariant_encoder_input, raw_value_tail


_ORIGINALS: dict = {}


def install():
    global _INSTALLED
    import policy as P
    import baselines as B
    if _INSTALLED:
        return P.ActorCritic
    cls, enc_in, tail = _build()
    _ORIGINALS.update({'policy.ActorCritic': P.ActorCritic,
                       'policy.build_encoder_input': P.build_encoder_input,
                       'policy.build_value_tail': P.build_value_tail,
                       'baselines.all_baselines': B.all_baselines})
    P.ActorCritic = cls
    P.build_encoder_input = enc_in
    P.build_value_tail = tail
    stock_all = B.all_baselines

    def all_baselines_plus(cfg):
        out = list(stock_all(cfg))
        have = {s.name for s in out}
        out += [s for s in (B.SUSCQIFeasible(), B.SUSRPS()) if s.name not in have]
        return out

    B.all_baselines = all_baselines_plus
    for name in ('train_phase2', 'ppo', 'paper_eval', 'paper_run_eval'):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for attr, value in (('ActorCritic', cls), ('all_baselines', all_baselines_plus)):
            if hasattr(mod, attr):
                _ORIGINALS[f'{name}.{attr}'] = getattr(mod, attr)
                setattr(mod, attr, value)
    _INSTALLED = True
    return cls


def uninstall():
    global _INSTALLED
    if not _INSTALLED:
        return
    for key, value in _ORIGINALS.items():
        mod_name, attr = key.rsplit('.', 1)
        mod = sys.modules.get(mod_name)
        if mod is not None:
            setattr(mod, attr, value)
    _ORIGINALS.clear()
    _INSTALLED = False


def load_model(manifest, checkpoint_path, torch=None):
    if manifest.get('recipe') not in RECIPES:
        raise ValueError('load_model is for the raw_phase_inv recipes only')
    if torch is None:
        import torch
    from config import Config
    cls = install()
    model = cls(Config(**manifest['config']))
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, ckpt


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


def _recipe_record(root, recipe):
    root = Path(root).resolve()
    base_file = root / 'artifacts/base/config.json'
    recipe_file = root / 'artifacts' / recipe / 'config.json'
    base = json.loads(base_file.read_text())
    cfg = json.loads(recipe_file.read_text())
    expected = {**base, **EXPECTED_DIFF[recipe]}
    if pt.canonical(cfg) != pt.canonical(expected):
        raise ValueError(f'{recipe} must differ from frozen Base only in '
                         f'{sorted(EXPECTED_DIFF[recipe]) or "nothing"}')
    return cfg, {
        'schema_version': 1,
        'name': recipe,
        'source_sha256': {'paper_train_rawpi.py': pt.sha_file(root / 'paper_train_rawpi.py')},
        'base_recipe_sha256': pt.sha_file(base_file),
        'derived_recipe_sha256': pt.sha_file(recipe_file),
        'changed_ranges': {k: {'base': base[k], 'variant': v}
                           for k, v in EXPECTED_DIFF[recipe].items()},
        'architecture': ARCHITECTURE,
        'architecture_sha256': hashlib.sha256(pt.canonical(ARCHITECTURE).encode()).hexdigest(),
        'extra_eval_baselines': EXTRA_EVAL_BASELINES,
    }


def validate_manifest(manifest, root=ROOT):
    recipe = manifest.get('recipe')
    if recipe not in RECIPES:
        raise ValueError('Variant validation requires a raw_phase_inv recipe')
    root = Path(root).resolve()
    raw, expected_record = _recipe_record(root, recipe)
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
        raise ValueError('raw_phase_inv requires the original Bernoulli traffic')
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
    if args.recipe not in RECIPES:
        return _CORE_PLAN(args, root)
    _, record = _recipe_record(root, args.recipe)
    if args.traffic_model not in (None, 'bernoulli'):
        raise ValueError('raw_phase_inv requires the original Bernoulli traffic')
    with registered_recipes():
        run_dir, checkpoint, manifest = _CORE_PLAN(args, root)
    if checkpoint is None:
        manifest['variant_provenance'] = record
        if manifest.get('calibration_profile') is not None:
            manifest['calibration_application'] = (
                'Bernoulli Base-calibrated beta, unchanged world; phase-invariant '
                'raw-input network' + (' and a higher initial LR'
                                       if args.recipe == RECIPE_LR6 else ''))
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
