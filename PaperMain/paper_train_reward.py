#!/usr/bin/env python3
"""Additive REWARD-WEIGHT recipe for the protected PaperMain trainer.

``--recipe reward_c2m6`` trains the stock network in the stock Base world with
exactly two configuration values changed:

    lambda_c  1.0 -> 2.0     completion reward
    lambda_m  2.0 -> 6.0     miss / drop penalty

Why this pair (user decision 2026-09-16): every arriving packet ends as either
a completion or a failure, so over an episode
    lambda_c*n_comp - lambda_m*n_miss = lambda_c*N - (lambda_c+lambda_m)*n_miss
with N exogenous; the policy gradient therefore feels the SUM lambda_c+lambda_m
(3 -> 8, i.e. ~2.7x stronger pressure to finish packets) far more than the
ratio. The dense short-term term (urgency-weighted ACKed bits / b_norm) is left
at lambda_s = 1. This is the control arm for a goodput-aligned reward: it tests
whether re-weighting alone moves completion/miss/waste. Note it does NOT change
the marginal reward of sending bits to a packet that will miss anyway (still
positive through r_short), so waste is not expected to fall from this alone.

Rewards under this recipe are NOT comparable to Base reward numbers. Judge on
the physical KPIs (goodput, completion, miss, total-fail) and on the Base
reward recomputed by the holdout evaluator, which builds its world from the
Base configuration.

Plumbing mirrors paper_train_variants.py (narrow_mean): no core source is
modified; the recipe file must differ from frozen Base only in the two keys.
The trainer's periodic validation additionally logs SUS+CQI-Feasible and
SUS-RPS (same extension as paper_train_raw.py).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import paper_train as pt

ROOT = Path(__file__).resolve().parent
RECIPE = 'reward_c2m6'
RECIPES = {RECIPE: {'default_updates': 1000, 'lr_final': 0.0, 'lr_decay_updates': 1000}}
BASE_WEIGHTS = {'lambda_c': 1.0, 'lambda_m': 2.0}
NEW_WEIGHTS = {'lambda_c': 2.0, 'lambda_m': 6.0}
EXTRA_EVAL_BASELINES = ['SUS+CQI-Feasible', 'SUS-RPS']
_CORE_PLAN = pt.plan
_INSTALLED = False


def install():
    """Extend the trainer's validation baseline list (no model change)."""
    global _INSTALLED
    import baselines as B
    if _INSTALLED:
        return
    stock_all = B.all_baselines

    def all_baselines_plus(cfg):
        out = list(stock_all(cfg))
        have = {s.name for s in out}
        out += [s for s in (B.SUSCQIFeasible(), B.SUSRPS()) if s.name not in have]
        return out

    B.all_baselines = all_baselines_plus
    mod = sys.modules.get('train_phase2')
    if mod is not None and hasattr(mod, 'all_baselines'):
        mod.all_baselines = all_baselines_plus
    _INSTALLED = True


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
    """Require exactly the two reward weights to differ from frozen Base."""
    root = Path(root).resolve()
    base_file = root / 'artifacts/base/config.json'
    recipe_file = root / 'artifacts' / RECIPE / 'config.json'
    base = json.loads(base_file.read_text())
    recipe = json.loads(recipe_file.read_text())
    if pt.canonical({k: base.get(k) for k in BASE_WEIGHTS}) != pt.canonical(BASE_WEIGHTS):
        raise ValueError('Frozen Base reward weights differ from the recorded design')
    expected = {**base, **NEW_WEIGHTS}
    if pt.canonical(recipe) != pt.canonical(expected):
        raise ValueError('reward_c2m6 must differ from frozen Base only in lambda_c and lambda_m')
    return recipe, {
        'schema_version': 1,
        'name': RECIPE,
        'source_sha256': {'paper_train_reward.py': pt.sha_file(root / 'paper_train_reward.py')},
        'base_recipe_sha256': pt.sha_file(base_file),
        'derived_recipe_sha256': pt.sha_file(recipe_file),
        'changed_ranges': {k: {'base': BASE_WEIGHTS[k], 'variant': NEW_WEIGHTS[k]} for k in NEW_WEIGHTS},
        'extra_eval_baselines': EXTRA_EVAL_BASELINES,
        'reward_note': 'lambda_c+lambda_m 3 -> 8; rewards not comparable to Base; judge on KPIs',
    }


def validate_manifest(manifest, root=ROOT):
    if manifest.get('recipe') != RECIPE:
        raise ValueError('Variant validation requires recipe reward_c2m6')
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
        raise ValueError('reward_c2m6 requires the original Bernoulli traffic')
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
        raise ValueError('reward_c2m6 requires the original Bernoulli traffic')
    with registered_recipes():
        run_dir, checkpoint, manifest = _CORE_PLAN(args, root)
    if checkpoint is None:
        manifest['variant_provenance'] = record
        if manifest.get('calibration_profile') is not None:
            manifest['calibration_application'] = (
                'Bernoulli Base-calibrated beta, unchanged world and network; '
                'only the reward weights lambda_c/lambda_m differ from Base')
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
