"""Read-only loading and opt-in application of complete CQI4 calibrations.

A profile describes calibration of the frozen Base world. Applying its beta to
another training/evaluation world is a transfer experiment, not evidence that
that world has also met the first-ACK target. Callers must record that transfer.
Archived configuration files and checkpoints are never changed by this module.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
PROTOCOL = 'cqi4-feasible-v1'
BASE_CONFIG = 'artifacts/base/config.json'
SCIENCE_SOURCES = ('paper_calibration.py', 'config.py', 'env.py', 'channel.py',
                   'csi.py', 'codebook.py', 'phy.py', 'la_planner.py',
                   'traffic.py', 'transmission.py')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root=ROOT):
    """Bind the full calibration implementation, including support modules."""
    root = Path(root).resolve()
    names = set(SCIENCE_SOURCES)
    # macOS archives may contain ._*.py AppleDouble metadata. These hidden
    # sidecars are not Python modules and may disappear on extraction.
    names.update(p.relative_to(root).as_posix()
                 for p in (root / 'calibration').glob('*.py')
                 if not p.name.startswith('.'))
    return {name: sha256_file(root / name) for name in sorted(names)}


def normalized_config(raw):
    """Normalize dataclass defaults/types, excluding only the calibrated beta."""
    from config import Config
    if dataclasses.is_dataclass(raw):
        raw = dataclasses.asdict(raw)
    if not isinstance(raw, dict):
        raise ValueError('Calibration configuration must be a JSON object or Config')
    known = {field.name for field in dataclasses.fields(Config)}
    unknown = set(raw) - known - {'git_hash', 'git_dirty_py'}
    if unknown:
        raise ValueError(f'Unrecognized calibration configuration fields: {sorted(unknown)}')
    cfg = Config(**{key: value for key, value in raw.items() if key in known})
    cfg.la_beta = 1.0
    cfg.la_beta_by_depth = ()
    cfg.validate_la()
    # JSON canonicalization makes tuples/lists from a frozen JSON equivalent.
    return json.loads(json.dumps(dataclasses.asdict(cfg), allow_nan=False))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _beta(value, depth):
    if not isinstance(value, list) or len(value) != depth:
        raise ValueError(f'Calibration needs exactly {depth} rounded beta values')
    if any(not _finite(b) or not 0 < b <= 1.5 or round(b, 4) != b for b in value):
        raise ValueError('Calibration beta must be finite, positive, <=1.5 and rounded to four decimals')


def validate_profile(profile, root=ROOT):
    """Reject archived, smoke, incomplete, mismatched or stale-source profiles.

    This checks completion and reproducibility metadata; holdout estimates remain
    measured estimates and do not promise exactly 90% ACK for a learned policy.
    """
    root = Path(root).resolve()
    if not isinstance(profile, dict):
        raise ValueError('Calibration profile must be a JSON object')
    if profile.get('schema_version') != SCHEMA_VERSION or profile.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported calibration profile schema/protocol')
    if profile.get('status') != 'validated':
        raise ValueError('Calibration profile must have validated status; archived/smoke/incomplete values cannot be applied')
    if profile.get('reference_config_sha256') != sha256_file(root / BASE_CONFIG):
        raise ValueError('Calibration reference Base configuration changed or does not match')
    expected = normalized_config(json.loads((root / BASE_CONFIG).read_text()))
    if 'config' not in profile or normalized_config(profile['config']) != expected:
        raise ValueError('Calibration configuration does not match the frozen full-length Base world')
    if expected['debug'] or expected['episode_len_main'] != 1000 or expected['cqi_mode'] != 'nr4bit':
        raise ValueError('Validated calibration requires the 1000-slot Base CQI4 world')
    if expected['la_mode'] != 'post_rzf' or expected['decode_order'] != 'rbg_major':
        raise ValueError('Validated calibration requires post-RZF link adaptation and RBG-major traversal')
    if profile.get('source_sha256') != source_hashes(root):
        raise ValueError('Calibration source changed; rerun calibration before applying this profile')
    if (profile.get('membership_verified_calibration') is not True
            or profile.get('membership_verified_holdout') is not True):
        raise ValueError('Calibration and holdout final-group membership must both be verified')
    depth = expected['l_max']
    _beta(profile.get('beta_rounded'), depth)
    phases = []
    for phase, episodes in (('calibration', 36), ('holdout', 12)):
        part = profile.get(phase)
        if not isinstance(part, dict):
            raise ValueError(f'Missing calibration {phase} evidence')
        if (not _integer(part.get('episodes')) or part['episodes'] != episodes
                or not _integer(part.get('slots')) or part['slots'] != 1000
                or not _integer(part.get('start')) or part['start'] < 0
                or not _integer(part.get('sampler_seed')) or part['sampler_seed'] < 0):
            raise ValueError(f'Calibration {phase} must contain {episodes} complete 1000-slot episodes')
        phases.append(set(range(part['start'], part['start'] + episodes)))
    if phases[0] & phases[1]:
        raise ValueError('Calibration and holdout episode ranges must be disjoint')
    holdout = profile['holdout']
    if holdout.get('beta_rounded') != profile['beta_rounded']:
        raise ValueError('Holdout must use the exact frozen rounded beta from calibration')
    by_depth = holdout.get('by_depth')
    if not isinstance(by_depth, dict) or set(by_depth) != {str(m) for m in range(1, depth + 1)}:
        raise ValueError('Holdout evidence must cover every final transmission depth')
    for m, evidence in by_depth.items():
        if not isinstance(evidence, dict):
            raise ValueError(f'Invalid holdout evidence at depth {m}')
        count = evidence.get('member_count')
        ack = evidence.get('first_ack')
        acks = evidence.get('acks')
        ci = evidence.get('episode_cluster_ci95')
        if (not _integer(count) or count < 1 or not _finite(ack) or not 0 <= ack <= 1
                or not _integer(acks) or not 0 <= acks <= count
                or not math.isclose(ack, acks / count, rel_tol=0.0, abs_tol=1e-12)
                or not isinstance(ci, list) or len(ci) != 2
                or not all(_finite(v) and 0 <= v <= 1 for v in ci)
                or ci[0] > ci[1]):
            raise ValueError(f'Invalid or empty holdout evidence at depth {m}')
    return profile


def load_profile(path, root=ROOT):
    """Load a summary JSON and return its validated document; performs no writes."""
    path = Path(path)
    profile = json.loads(path.read_text())
    validate_profile(profile, root)
    return profile


def apply_profile(cfg, profile):
    """Return a separate config with only beta changed after load/validation.

    Accepts Config or a config dict. Keeping this step separate lets callers
    validate the frozen Base reference before any smoke/OOD/seed overrides.
    """
    out = copy.deepcopy(cfg)
    get = out.get if isinstance(out, dict) else lambda name: getattr(out, name)
    if (get('cqi_mode') != 'nr4bit' or get('la_mode') != 'post_rzf'
            or get('decode_order') != 'rbg_major' or get('pmi_mode') == 'genie'):
        raise ValueError('CQI4 profile applies only to imperfect-CSI post-RZF RBG-major worlds')
    depth = get('l_max')
    _beta(profile.get('beta_rounded'), depth)
    if isinstance(out, dict):
        out['la_beta'], out['la_beta_by_depth'] = 1.0, list(profile['beta_rounded'])
    else:
        out.la_beta, out.la_beta_by_depth = 1.0, tuple(profile['beta_rounded'])
        out.validate_la()
    return out


def _document_sha256(document):
    data = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def load_profile_record(path, root=ROOT):
    """Snapshot validated JSON plus its original byte hash for a run manifest."""
    path = Path(path).resolve()
    content = path.read_bytes()
    document = json.loads(content)
    validate_profile(document, root)
    return {'path': str(path), 'file_sha256': hashlib.sha256(content).hexdigest(),
            'document_sha256': _document_sha256(document), 'document': document}


def validate_profile_record(record, root=ROOT):
    """Validate a saved snapshot without requiring its original file to exist."""
    if not isinstance(record, dict) or not isinstance(record.get('document'), dict):
        raise ValueError('Invalid saved calibration profile record')
    if record.get('document_sha256') != _document_sha256(record['document']):
        raise ValueError('Saved calibration profile document changed')
    file_hash = record.get('file_sha256')
    if (not isinstance(file_hash, str) or len(file_hash) != 64
            or any(c not in '0123456789abcdef' for c in file_hash)):
        raise ValueError('Invalid saved calibration profile file hash')
    validate_profile(record['document'], root)
    return record


def reject_calibration_episode_overlap(profile, start, episodes):
    """Keep calibration and its inspected holdout out of downstream test bands."""
    end = start + episodes
    for phase in ('calibration', 'holdout'):
        used = profile[phase]
        if max(start, used['start']) < min(end, used['start'] + used['episodes']):
            raise ValueError(f'Episode range overlaps the profile {phase} episodes; use a disjoint test band')
