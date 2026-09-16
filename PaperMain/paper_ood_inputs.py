"""Read-only provenance guards for completed Bernoulli OOD policy inputs.

The sole legacy bridge is the reviewed FTP3 opt-in addition: its Bernoulli
branch and random draws are unchanged, and policy, PHY, CSI and PPO sources
must be identical. It never rewrites a training manifest or checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import paper_train as pt

FROZEN_REFERENCE = 'provenance/before_ftp3_sources'
# Pin both sides of the four reviewed differences. Merely allowing filenames
# would also admit unrelated future edits to their Bernoulli behavior.
REVIEWED_BRIDGE = {
    'paper_train.py': (
        '0b4b0fe2f6359a015df21faad99cd2c4b8cd9b0898f8dcba087578efa6077531',
        '3095c067ce9ed3cf0dea43bcac5263e839986b85599e53ef4df43b84be3f64cc'),
    'train_phase2.py': (
        'fa82f7a438def16c6405cb2c79b829b256b2843abcaf99f37f983488127731bb',
        '2079a90c356a92d8cd591b378d6708b6b06d3185a84fc206d887d9bb9b3e4eca'),
    'config.py': (
        '5ab93d0212e1a86449896252514ffd4ddb7666410117a720efc527bb7db47658',
        '734936660c5ea80ad64c8b58b1d9f2e6942e5e52787f5df7d07370cb0e8845a8'),
    'traffic.py': (
        '49f1822387d29c4b8333cc73ea22f64af9fa153800651dcf7c14cb139e3fd4aa',
        '3a4c341b240924110c82c296eec3b7bf67dfcf2d7d8e2043557462ae843cdf01'),
}


def _equal(left, right, message):
    if pt.canonical(left) != pt.canonical(right):
        raise ValueError(message)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _configuration_payload(value, source_hashes):
    """Remove only the driver's verified source stamp, never Config fields.

    train_phase2 stores cfg_dump in both config.json and checkpoint['cfg'];
    paper_train supplies its content-addressed stamp instead of Git metadata.
    Plain normalized configs are also valid, but partial metadata is not.
    """
    if not isinstance(value, dict):
        raise ValueError('Saved configuration must be an object')
    metadata_keys = {'git_hash', 'git_dirty_py'}
    present = metadata_keys & set(value)
    metadata = {}
    if present:
        stamp = 'sha256:' + hashlib.sha256(pt.canonical(source_hashes).encode()).hexdigest()
        expected = {'git_hash': stamp, 'git_dirty_py': False}
        metadata = {key: value[key] for key in present}
        _equal(metadata, expected, 'Configuration provenance stamp differs from saved training sources')
    return {key: item for key, item in value.items() if key not in metadata_keys}, metadata


def _file(root, path):
    path = Path(path)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if resolved != path or not resolved.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Input must be a regular, non-symlink file inside PaperMain: {path}')
    return path


def _source_provenance(root, manifest):
    current, saved = pt.source_hashes(root), manifest.get('source_sha256')
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise ValueError('Training source inventory differs')
    if saved == current:
        return dict(mode='current-exact', current_source_sha256=current,
                    training_source_sha256=copy.deepcopy(saved), frozen_reference=None,
                    reviewed_bridge={})
    frozen = root / FROZEN_REFERENCE
    if frozen.resolve() != frozen or not frozen.is_dir():
        raise ValueError('Exact frozen pre-FTP3 source reference is missing')
    if pt.source_hashes(frozen) != saved:
        raise ValueError('Training source hashes do not match the exact frozen reference')
    changed = {name for name in current if saved[name] != current[name]}
    if changed != set(REVIEWED_BRIDGE):
        raise ValueError('Source bridge changes sources beyond the reviewed FTP3 addition')
    for name, (old, new) in REVIEWED_BRIDGE.items():
        if saved[name] != old or current[name] != new:
            raise ValueError(f'Source bridge is not the reviewed old/current revision: {name}')
    if manifest.get('recipe') != 'base' or manifest.get('calibration_reference_root') is not None:
        raise ValueError('Legacy source bridge requires the original Base calibration reference')
    return dict(mode='reviewed-bernoulli-legacy-bridge', current_source_sha256=current,
                training_source_sha256=copy.deepcopy(saved), frozen_reference=FROZEN_REFERENCE,
                reviewed_bridge={name: dict(training_sha256=old, evaluation_sha256=new)
                                 for name, (old, new) in REVIEWED_BRIDGE.items()},
                config_addition={'traffic_model': 'bernoulli'},
                rationale='FTP3 is opt-in; unchanged Bernoulli branch and RNG draws. '
                          'All policy, PPO, environment, PHY, CSI, metric and baseline sources are identical.')


def validate_run(root, run: Path, checkpoint='best'):
    """Validate one completed source without imports of Torch/Sionna or writes.

    ``training_config`` is ready for the current Config constructor;
    ``raw_training_config`` is preserved for exact checkpoint comparisons.
    Actual runtime equality and loaded tensor checks are separate preflight
    calls so the caller can configure CUDA/thread settings before importing it.
    """
    root, run = Path(root).resolve(), Path(run)
    run = (run if run.is_absolute() else root / run).resolve()
    if (run.parent != root / 'runs' or (root / 'runs').resolve() != root / 'runs'
            or not run.is_dir()):
        raise ValueError('Source must be a direct run directory below PaperMain/runs')
    if checkpoint not in ('best', 'latest'):
        raise ValueError('Checkpoint must be best or latest')
    mpath = _file(root, run / 'paper_manifest.json')
    cpath = _file(root, run / 'config.json')
    manifest = json.loads(mpath.read_text())
    if type(manifest.get('version')) is not int or manifest['version'] != pt.VERSION:
        raise ValueError('Unsupported run manifest version')
    provenance = _source_provenance(root, manifest)
    raw = manifest['config']
    normalized = pt.normalize_config(raw)
    legacy = provenance['mode'] == 'reviewed-bernoulli-legacy-bridge'
    if legacy:
        if 'traffic_model' in raw:
            raise ValueError('Legacy configuration must predate the traffic_model field')
        _equal(normalized, {**raw, 'traffic_model': 'bernoulli'},
               'Legacy configuration differs beyond the sole traffic_model default addition')
    else:
        _equal(normalized, raw, 'Saved configuration is not a complete normalized configuration')
    config_payload, config_metadata = _configuration_payload(json.loads(cpath.read_text()), manifest['source_sha256'])
    _equal(config_payload, raw, 'Run config.json differs from manifest')
    if normalized['traffic_model'] != 'bernoulli':
        raise ValueError('This OOD comparison requires original Bernoulli traffic')
    target = _integer(manifest.get('target_updates'), 'target updates', 1)
    if target > 10000:
        raise ValueError('Training target violates training/validation episode separation')
    _integer(normalized['seed'], 'training seed')
    if normalized['seed'] >= 2 ** 32:
        raise ValueError('Training seed exceeds the supported range')
    if (manifest.get('smoke_slots') is not None or normalized['debug'] is not False
            or manifest.get('purpose') != 'training'):
        raise ValueError('A diagnostic training run cannot be an OOD source')
    pt.validate_schedule(manifest['schedule'], normalized['ppo_learning_rate'])
    execution = manifest.get('execution')
    if (not isinstance(execution, dict) or execution.get('device') not in ('cuda', 'cpu')
            or type(execution.get('threads')) is not int or execution['threads'] < 1):
        raise ValueError('Missing or malformed source execution record')
    runtime = manifest.get('runtime')
    if (not isinstance(runtime, dict) or not runtime.get('python')
            or not isinstance(runtime.get('packages'), dict) or not runtime['packages']):
        raise ValueError('Missing or malformed source runtime record')
    recipe = manifest.get('recipe')
    if recipe not in ('base', 'narrow_mean'):
        raise ValueError('OOD inputs must use Base or mean-matched NARROW')
    recipe_path = _file(root, root / 'artifacts' / recipe / 'config.json')
    if manifest.get('artifact_config_sha256') != pt.sha_file(recipe_path):
        raise ValueError('Frozen recipe configuration changed')
    reference_arg = FROZEN_REFERENCE if legacy else manifest.get('calibration_reference_root')
    reference = pt.calibration_reference(root, reference_arg)
    profile = manifest.get('calibration_profile')
    if profile is None:
        raise ValueError('OOD sources require their validated CQI4 calibration profile')
    from calibration.profile import validate_profile_record, apply_profile, source_hashes
    validate_profile_record(profile, reference)
    _equal(apply_profile(normalized, profile['document']), normalized,
           'Saved calibration beta differs from run configuration')
    expected = pt.normalize_config(json.loads(recipe_path.read_text()))
    expected['seed'] = normalized['seed']
    if type(normalized['ppo_batched_replay']) is not bool:
        raise ValueError('Saved replay mode must be boolean')
    expected['ppo_batched_replay'] = normalized['ppo_batched_replay']
    expected = apply_profile(expected, profile['document'])
    _equal(normalized, expected,
           'Training configuration differs beyond permitted recipe, seed, replay and calibration settings')
    variant = None
    if recipe == 'narrow_mean':
        import paper_train_variants as variants
        variant = variants.validate_manifest(manifest, root)
    status = _file(root, root / 'review/logs' / f'{run.name}_launch.json')
    completion = json.loads(status.read_text())
    if (completion.get('status') != 'completed' or type(completion.get('returncode')) is not int
            or completion['returncode'] != 0 or completion.get('target_completed') is not True
            or _integer(completion.get('checkpoint_update'), 'completion update') + 1 != target):
        raise ValueError('Source run has no successful exact completed-target launch record')
    ckpt = _file(root, run / 'ckpt' / f'{checkpoint}.pt')
    files = {mpath, cpath, status, ckpt, recipe_path, root / 'artifacts/base/config.json',
             root / 'paper_ood_inputs.py'}
    files.update(root / name for name in pt.SOURCES)
    files.update(root / name for name in source_hashes(root))
    files.update(reference / name for name in source_hashes(reference))
    files.add(reference / 'artifacts/base/config.json')
    if legacy:
        files.update(reference / name for name in pt.SOURCES)
    if variant is not None:
        files.add(root / 'paper_train_variants.py')
    hashes = {path.relative_to(root).as_posix(): pt.sha_file(_file(root, path))
              for path in sorted(files)}
    return dict(run=run, manifest=copy.deepcopy(manifest), training_config=copy.deepcopy(normalized),
                raw_training_config=copy.deepcopy(raw), checkpoint=ckpt,
                checkpoint_kind=checkpoint, checkpoint_sha256=pt.sha_file(ckpt), checkpoint_update=None,
                source_provenance=provenance, calibration_profile=copy.deepcopy(profile),
                calibration_reference=reference.relative_to(root).as_posix(),
                variant_provenance=variant, runtime=copy.deepcopy(runtime), input_hashes=hashes,
                config_provenance_metadata=config_metadata)


def validate_loaded_checkpoint(loaded, prepared):
    """Require exact original cfg, schedule, valid update and finite model data."""
    if not isinstance(loaded, dict) or not isinstance(loaded.get('model'), dict) or not loaded['model']:
        raise ValueError('Checkpoint must contain a nonempty model state dictionary')
    cfg, metadata = _configuration_payload(loaded.get('cfg'), prepared['manifest']['source_sha256'])
    _equal(cfg, prepared['raw_training_config'],
           'Checkpoint configuration differs from exact saved training configuration')
    update = _integer(loaded.get('update'), 'checkpoint update')
    target = prepared['manifest']['target_updates']
    if update >= target or (prepared['checkpoint_kind'] == 'latest' and update != target - 1):
        raise ValueError('Checkpoint update is inconsistent with completed training target')
    _equal(loaded.get('lr_schedule'), prepared['manifest']['schedule'],
           'Checkpoint LR schedule differs from the saved manifest')
    pt.validate_checkpoint_schedule(loaded, prepared['manifest']['schedule'])
    import torch
    for name, value in loaded['model'].items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Checkpoint model state is not a finite tensor: {name}')
    prepared['checkpoint_update'] = update


def validate_runtime(runtime, prepared):
    """Changing GPU number is allowed; software and device model must match."""
    _equal(runtime, prepared['runtime'], 'Evaluation runtime differs from saved training runtime')


def assert_inputs_unchanged(root, prepared):
    """Check input bytes, including both sides of the legacy bridge, at exit."""
    root = Path(root).resolve()
    for name, digest in prepared['input_hashes'].items():
        if pt.sha_file(_file(root, root / name)) != digest:
            raise ValueError(f'Evaluation input changed during execution: {name}')
