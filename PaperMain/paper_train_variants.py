#!/usr/bin/env python3
"""Additive mean-matched NARROW recipe for the protected PaperMain trainer.

Use the standard training flags with ``--recipe narrow_mean``. Original recipes
and all 16 training source digests remain unchanged. This entry point records
its own digest and both frozen recipe digests in ``variant_provenance``.

Evaluation integrations must call ``validate_manifest(manifest, root)`` before
using ``registered_recipes()`` around the standard evaluator's plan function.
Registration alone is not provenance validation. It does not change the saved
policy configuration or turn NARROW ID evaluation into common-world evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import paper_train as pt

ROOT = Path(__file__).resolve().parent
RECIPE = 'narrow_mean'
RECIPES = {RECIPE: {'default_updates': 2000, 'lr_final': 0.0, 'lr_decay_updates': 2000}}
BASE_RANGES = {'n_active_min': 16, 'n_active_max': 32,
               'ue_speed_min': 5.0, 'ue_speed_max': 40.0,
               'p_arrival_min': 0.15, 'p_arrival_max': 0.5}
MEAN_RANGES = {'n_active_min': 24, 'n_active_max': 24,
               'ue_speed_min': 22.5, 'ue_speed_max': 22.5,
               'p_arrival_min': 0.325, 'p_arrival_max': 0.325}
_CORE_PLAN = pt.plan


@contextmanager
def registered_recipes():
    """Temporarily expose new recipe names without changing core source files."""
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
    """Require exactly six changed range endpoints, including raw metadata."""
    root = Path(root).resolve()
    base_file = root / 'artifacts/base/config.json'
    recipe_file = root / 'artifacts' / RECIPE / 'config.json'
    base = json.loads(base_file.read_text())
    recipe = json.loads(recipe_file.read_text())
    if pt.canonical({key: base.get(key) for key in BASE_RANGES}) != pt.canonical(BASE_RANGES):
        raise ValueError('Frozen Base ranges differ from the mean-matched design')
    expected = {**base, **MEAN_RANGES}
    if pt.canonical(recipe) != pt.canonical(expected):
        raise ValueError('narrow_mean must differ from frozen Base only in the six mean-matched ranges')
    return recipe, {
        'schema_version': 1,
        'name': RECIPE,
        'source_sha256': {'paper_train_variants.py': pt.sha_file(root / 'paper_train_variants.py')},
        'base_recipe_sha256': pt.sha_file(base_file),
        'derived_recipe_sha256': pt.sha_file(recipe_file),
        'changed_ranges': {key: {'base': BASE_RANGES[key], 'variant': MEAN_RANGES[key]}
                           for key in MEAN_RANGES},
    }


def validate_manifest(manifest, root=ROOT):
    """Validate additive provenance and full allowed config before resume/eval.

    This supplements, never replaces, the standard source, checkpoint, runtime,
    schedule, calibration and completed-run checks made by training/evaluation.
    """
    if manifest.get('recipe') != RECIPE:
        raise ValueError('Variant validation requires recipe narrow_mean')
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
        raise ValueError('narrow_mean requires the original Bernoulli traffic')
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
    """Read-only protected plan; suitable for launchers and resume preflight."""
    if args.recipe != RECIPE:
        return _CORE_PLAN(args, root)
    _, record = _recipe_record(root)
    if args.traffic_model not in (None, 'bernoulli'):
        raise ValueError('narrow_mean requires the original Bernoulli traffic')
    with registered_recipes():
        run_dir, checkpoint, manifest = _CORE_PLAN(args, root)
    if checkpoint is None:
        manifest['variant_provenance'] = record
        if manifest.get('calibration_profile') is not None:
            manifest['calibration_application'] = (
                'Bernoulli Base-calibrated beta transferred unchanged to mean-matched NARROW; '
                'no NARROW-specific holdout validation')
    validate_manifest(manifest, root)
    return run_dir, checkpoint, manifest


def main(argv=None):
    previous_plan = pt.plan
    with registered_recipes():
        pt.plan = plan
        try:
            return pt.main(argv)
        finally:
            pt.plan = previous_plan


if __name__ == '__main__':
    main()
