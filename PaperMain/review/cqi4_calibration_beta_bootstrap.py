"""Secondary calibration-only uncertainty diagnostic; never proposes new beta.

Resample the 36 calibration episodes as intact clusters, with replacement.
For each resample, recompute the ordinary pooled member-ratio 10th percentile
using NumPy's linear interpolation, exactly as in the calibration estimator.
Use shared episode draws across depths to retain cross-depth dependence.
The holdout bank is never loaded. Outputs are diagnostic, not deployable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7301)
    args = parser.parse_args()
    if args.draws < 1 or args.seed < 0:
        parser.error('draws must be positive and seed must be nonnegative')
    if args.output.exists():
        parser.error('Refusing to replace an existing diagnostic')
    start = time.monotonic()
    bank = np.load(args.bank, allow_pickle=False)
    episode_keys = sorted(int(k.rsplit('ep', 1)[1]) for k in bank.files
                          if k.startswith('ratios_m1_ep'))
    if episode_keys != list(range(36)):
        raise ValueError('This diagnostic requires the 36-episode calibration bank')
    episode_draws = np.random.default_rng(args.seed).integers(
        0, len(episode_keys), size=(args.draws, len(episode_keys)))
    by_depth = {}
    for depth in range(1, 5):
        episodes = [bank[f'ratios_m{depth}_ep{ep}'] for ep in episode_keys]
        if any(v.ndim != 1 or len(v) == 0 or not np.isfinite(v).all()
               or np.any(v < 0) for v in episodes):
            raise ValueError('Invalid calibration member ratios')
        pooled = np.concatenate(episodes)
        fitted = float(np.percentile(pooled, 10, method='linear'))
        estimates = np.empty(args.draws, dtype=float)
        for i, indices in enumerate(episode_draws):
            sample = np.concatenate([episodes[int(ep)] for ep in indices])
            estimates[i] = np.percentile(sample, 10, method='linear')
        by_depth[str(depth)] = {
            'calibration_beta_raw': fitted,
            'calibration_beta_rounded': round(fitted, 4),
            'beta_percentile_ci95': np.percentile(
                estimates, [2.5, 97.5], method='linear').tolist(),
            'bootstrap_mean': float(estimates.mean()),
            'bootstrap_median': float(np.median(estimates)),
            'bootstrap_standard_deviation': float(estimates.std(ddof=1)),
            'member_count': int(len(pooled)),
            'episode_member_counts': [int(len(v)) for v in episodes],
        }
    result = {
        'status': 'secondary_diagnostic_only_not_deployable',
        'purpose': 'Uncertainty of the calibration estimator, without holdout fitting or beta retuning',
        'method': 'Episode-cluster percentile bootstrap of the pooled member-ratio 10th percentile',
        'within_resample_quantile': {'percentile': 10, 'numpy_method': 'linear'},
        'interval_percentiles': [2.5, 97.5],
        'episodes': len(episode_keys), 'draws': args.draws, 'seed': args.seed,
        'same_episode_draws_across_depths': True,
        'holdout_loaded': False,
        'calibration_bank_sha256': hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version': np.__version__,
        'by_depth': by_depth,
        'limitations': [
            'Percentile intervals are a finite-cluster bootstrap approximation based on 36 episodes.',
            'These intervals concern beta, not holdout ACK coverage or downstream policy performance.',
            'They do not identify the cause of calibration-to-holdout differences.',
            'No holdout data were used and no deployed beta is changed or recommended.',
        ],
        'elapsed_seconds': time.monotonic() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'by_depth': by_depth, 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
