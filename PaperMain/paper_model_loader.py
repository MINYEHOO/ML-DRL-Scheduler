"""Strict evaluation model dispatch without changing frozen training sources.

Training wrappers install their feature builders process-wide. Evaluation may
hold several architectures at once, so each model instead activates its own
builders for the duration of a method call and restores them in ``finally``.
The shared reentrant lock also covers nested calls and concurrent *loader*
models. Do not mix these models with a running trainer in the same process.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import importlib
from pathlib import Path
import sys
import threading

import paper_train as pt

ROOT = Path(__file__).resolve().parent
CORE_RECIPES = ('base', 'lrann', 'narrow')
ARCHIVED_STOCK_RECIPES = ('base_s3024', 'base_s4024', 'base_s5024')
VARIANTS = {
    'narrow_mean': 'paper_train_variants',
    'raw_invariant': 'paper_train_raw',
    'raw_phase_inv': 'paper_train_rawpi',
    'raw_phase_inv_lr6': 'paper_train_rawpi',
}
_LOCK = threading.RLock()
_STOCK_RUNTIME = None
_ENTRY_POINTS = ('encode', 'value', 'state_value', 'deterministic_action',
                 'decode', '_rbg_major_pass', 'replay', 'replay_batch')


def recipe_names():
    """Live-run recipes supported by this evaluator (no global registration)."""
    return CORE_RECIPES + tuple(VARIANTS)


def _variant(manifest, *, archived=False):
    if not isinstance(manifest, dict):
        raise ValueError('Model manifest must be a dictionary')
    recipe = manifest.get('recipe')
    if not isinstance(recipe, str):
        raise ValueError('Evaluation recipe must be a string')
    if recipe in VARIANTS:
        return importlib.import_module(VARIANTS[recipe])
    if recipe in CORE_RECIPES or (archived and recipe in ARCHIVED_STOCK_RECIPES):
        if manifest.get('variant_provenance') is not None:
            raise ValueError('Stock recipe has unexpected variant provenance')
        return None
    raise ValueError(f'Unsupported evaluation recipe: {recipe!r}')


def validate_manifest(manifest, root=ROOT):
    """Validate variant identity in addition to each runner's existing guards.

The caller retains its version, source/config, completion, checkpoint schedule,
calibration, and episode-history checks. Variant validators additionally bind
the architecture and permitted recipe changes to the original source bytes.
"""
    module = _variant(manifest)
    if module is not None:
        return module.validate_manifest(manifest, Path(root).resolve())
    return None


def evaluation_sources(root, manifest):
    """Extra executable source digests to include in evaluation provenance."""
    root = Path(root).resolve()
    module = _variant(manifest, archived=True)
    names = ['paper_model_loader.py']
    if module is not None:
        names.append(module.__name__ + '.py')
    return {name: pt.sha_file(root / name) for name in names}


def _clean_policy():
    """Require a stock policy before constructing a new isolated model."""
    global _STOCK_RUNTIME
    import policy

    for name in ('paper_train_raw', 'paper_train_rawpi'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '_INSTALLED', False):
            raise ValueError('Cannot load evaluation models with an installed trainer variant; use a fresh process')
    current = (policy.ActorCritic, policy.build_encoder_input, policy.build_value_tail)
    if any(getattr(value, '__module__', None) != 'policy' for value in current):
        raise ValueError('Evaluation model construction requires unmodified stock policy builders')
    if _STOCK_RUNTIME is None:
        _STOCK_RUNTIME = current
    elif current != _STOCK_RUNTIME:
        raise ValueError('Stock policy class or feature builders changed in this process')
    return policy, current


@contextmanager
def _features(policy, encoder_input, value_tail):
    # Only these two symbols are dynamically resolved by inherited policy
    # methods. Classes, baselines, trainer aliases and recipe registries stay
    # untouched. Nested calls restore their enclosing model's feature pair.
    with _LOCK:
        previous = policy.build_encoder_input, policy.build_value_tail
        policy.build_encoder_input, policy.build_value_tail = encoder_input, value_tail
        try:
            yield
        finally:
            policy.build_encoder_input, policy.build_value_tail = previous


def _bound_class(cls, policy, encoder_input, value_tail):
    def bind(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            with _features(policy, encoder_input, value_tail):
                return method(self, *args, **kwargs)
        return call

    methods = {name: bind(getattr(cls, name)) for name in _ENTRY_POINTS}
    methods['__module__'] = __name__
    # Subclassing rather than adding an nn.Module wrapper preserves every
    # checkpoint key and the model's normalizer buffers verbatim.
    return type('Evaluation' + cls.__name__, (cls,), methods)


def load_model(manifest, checkpoint, policy_config=None, root=ROOT, validate=True):
    """Load an already-read checkpoint into its frozen evaluation architecture.

``validate=False`` is only for callers that already validated provenance (or
the legacy archived stock format). It never disables checkpoint/config, finite
tensor, or strict state-dictionary checks. ``policy_config`` is the caller's
validated evaluation configuration, e.g. a variable UE count or the legacy
runner's explicitly recorded calibration override; checkpoint identity is
always checked against the unmodified training configuration.
"""
    module = _variant(manifest, archived=not validate)
    if validate:
        validate_manifest(manifest, root)
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a dictionary')
    state = checkpoint.get('model')
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint must contain a nonempty model state dictionary')
    training = manifest.get('config')
    saved = checkpoint.get('cfg')
    if not isinstance(training, dict) or not isinstance(saved, dict):
        raise ValueError('Checkpoint and manifest must contain configuration dictionaries')
    training = pt.normalize_config(training)
    if pt.canonical(pt.normalize_config(saved)) != pt.canonical(training):
        raise ValueError('Checkpoint configuration differs from saved training configuration')
    if policy_config is not None and not isinstance(policy_config, dict):
        raise ValueError('Evaluation policy configuration must be a dictionary')
    effective = training if policy_config is None else pt.normalize_config(policy_config)

    import torch
    from config import Config
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Checkpoint model state is not a finite tensor: {name}')

    with _LOCK:
        policy, (cls, encoder_input, value_tail) = _clean_policy()
        recipe = manifest['recipe']
        if recipe == 'raw_invariant':
            cls, value_tail = module._build()
        elif recipe in ('raw_phase_inv', 'raw_phase_inv_lr6'):
            cls, encoder_input, value_tail = module._build()
        bound = _bound_class(cls, policy, encoder_input, value_tail)
        model = bound(Config(**effective))
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    return model
