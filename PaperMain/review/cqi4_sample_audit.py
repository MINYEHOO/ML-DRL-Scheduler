"""Independently audit a complete CQI4 calibration directory, without writes.

Usage: python review/cqi4_sample_audit.py results/calibration_corrected_...

Only NumPy and the standard library are used. This checks saved sample-bank
arithmetic and provenance consistency; it does not replay channel generation,
prove wall-clock ordering, or infer unrecorded UE group identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(np.allclose(actual, expected, rtol=1e-12, atol=1e-12,
                        equal_nan=False), f'{label}: {actual!r} != {expected!r}')


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f'{path}: invalid JSON numeric constant {value}')
    return json.loads(path.read_text(), parse_constant=invalid_constant)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_quantile(values, probability):
    """Independent sorted-order interpolation, rather than the fitter call."""
    ordered = np.sort(np.asarray(values, dtype=float))
    require(len(ordered) > 0, 'Cannot compute quantile of an empty bank')
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return float(ordered[lower] + (position - lower) *
                 (ordered[upper] - ordered[lower]))


def audit_phase(directory, summary, phase, beta):
    metadata = summary[phase]
    cfg = summary['config']
    depth_count, episode_count = cfg['l_max'], metadata['episodes']
    expected_draws = 2 * ((metadata['slots'] + 6) // 7)
    filename = f'{phase}_samples.npz'
    require(sha256(directory / filename) == summary['samples_sha256'][filename],
            f'{phase}: sample file SHA256 mismatch')
    expected_keys = {f'{kind}_m{depth}_ep{ep}'
                     for kind in ('ratios', 'caps', 'mi')
                     for depth in range(1, depth_count + 1)
                     for ep in range(episode_count)}
    banks, reports = {}, {}
    with np.load(directory / filename, allow_pickle=False) as stored:
        require(set(stored.files) == expected_keys,
                f'{phase}: sample keys do not exactly cover all depths/episodes')
        for depth in range(1, depth_count + 1):
            bank = {kind: [] for kind in ('ratios', 'caps', 'mi')}
            diagnostics = metadata['diagnostics'][str(depth)]
            require(len(diagnostics) == episode_count,
                    f'{phase} depth {depth}: missing episode diagnostics')
            excluded, cqi_zero, candidates, groups = 0, 0, 0, 0
            for ep in range(episode_count):
                label = f'{phase} depth {depth} episode {ep}'
                for kind in bank:
                    values = np.asarray(stored[f'{kind}_m{depth}_ep{ep}'], dtype=float)
                    require(values.ndim == 1 and np.isfinite(values).all(),
                            f'{label}: invalid {kind} samples')
                    require(np.all(values > 0 if kind == 'caps' else values >= 0),
                            f'{label}: invalid {kind} sign or zero capacity')
                    bank[kind].append(values.copy())
                ratio, cap, mi = (bank[kind][-1] for kind in ('ratios', 'caps', 'mi'))
                require(len(ratio) == len(cap) == len(mi), f'{label}: unpaired samples')
                require(len(ratio) % depth == 0, f'{label}: incomplete group members')
                close(ratio, mi / cap, f'{label}: ratio = MI / capacity')
                d = diagnostics[ep]
                require(d['draws'] == expected_draws, f'{label}: wrong draw count')
                require(d['groups'] * depth == d['members'] == len(ratio),
                        f'{label}: group/member count mismatch')
                require(d['groups'] >= 0 and d['skipped_insufficient'] >= 0 and
                        d['groups'] + d['skipped_insufficient'] == d['draws'],
                        f'{label}: skipped/group draws mismatch')
                require(d['candidate_slots'] == d['draws'] * cfg['num_ue'],
                        f'{label}: candidate opportunity count mismatch')
                require(0 <= d['cqi_zero_candidates'] <= d['excluded_candidates'] <=
                        d['candidate_slots'], f'{label}: invalid exclusion counts')
                zero_mi = int(np.count_nonzero(mi == 0))
                require(d['real_zero_mi_members'] == zero_mi ==
                        int(np.count_nonzero(ratio == 0)), f'{label}: outage count mismatch')
                excluded += d['excluded_candidates']
                cqi_zero += d['cqi_zero_candidates']
                candidates += d['candidate_slots']
                groups += d['groups']
            all_ratio, all_cap, all_mi = (np.concatenate(bank[kind])
                                        for kind in ('ratios', 'caps', 'mi'))
            require(len(all_cap) > 0, f'{phase} depth {depth}: empty depth')
            epsilon = cfg['b_tx_epsilon']
            scaled = beta[depth - 1] * all_cap
            below = int(np.count_nonzero(scaled < epsilon))
            require(below == 0, f'{phase} depth {depth}: beta changes group membership')
            evidence = metadata['membership'][str(depth)]
            require(evidence['member_count'] == len(all_cap),
                    f'{phase} depth {depth}: membership count mismatch')
            require(evidence['below_epsilon_members'] == below,
                    f'{phase} depth {depth}: epsilon failure count mismatch')
            for key, value in (('minimum_raw_cap', all_cap.min()),
                               ('minimum_scaled_cap', scaled.min()), ('epsilon', epsilon)):
                close(evidence[key], float(value), f'{phase} depth {depth}: {key}')
            reports[str(depth)] = dict(
                groups=groups, members=len(all_cap), zero_capacity_members=0,
                real_zero_mi_members=int(np.count_nonzero(all_mi == 0)),
                below_epsilon_members=below,
                minimum_scaled_payload_bits=float(scaled.min()),
                minimum_membership_margin_bits=float(scaled.min() - epsilon),
                candidate_opportunities=candidates, excluded_candidates=excluded,
                cqi_zero_candidates=cqi_zero,
                runtime_first_ack=float(np.mean(all_mi >= scaled - 1e-6)),
                strict_ratio_coverage=float(np.mean(all_ratio >= beta[depth - 1])))
            banks[depth] = bank
    return banks, reports


def audit(directory):
    directory = Path(directory).resolve()
    summary = read_json(directory / 'summary.json')
    frozen = read_json(directory / 'calibration_fit.json')
    require(summary['schema_version'] == 1 and summary['protocol'] == 'cqi4-feasible-v1',
            'Unsupported calibration protocol/schema')
    require(summary['status'] == 'validated', 'Audit requires a completed full calibration')
    require(summary['membership_verified_calibration'] is True and
            summary['membership_verified_holdout'] is True, 'Missing membership validation')
    require(summary['calibration_fit_sha256'] == sha256(directory / 'calibration_fit.json'),
            'Frozen calibration fit SHA256 mismatch')
    require(frozen['status'] == 'in_progress', 'Unexpected frozen-fit lifecycle status')
    require('by_depth' not in frozen['holdout'] and 'beta_rounded' not in frozen['holdout'],
            'Frozen fit contains holdout results')
    for key in ('schema_version', 'protocol', 'config', 'reference_config_sha256',
                'source_sha256', 'calibration', 'beta_raw', 'beta_rounded'):
        require(frozen[key] == summary[key], f'Frozen fit and summary disagree: {key}')
    require(summary['sampling']['every_slots'] == 7 and
            summary['sampling']['groups_per_depth_per_sample'] == 2,
            'Unsupported sampling schedule')
    require(summary['config']['l_max'] == 4, 'Expected four spatial depths')
    phases = []
    for phase, start, episodes, seed in (('calibration', 50000, 36, 777),
                                       ('holdout', 70000, 12, 20250713)):
        part = summary[phase]
        require((part['start'], part['episodes'], part['sampler_seed'], part['slots']) ==
                (start, episodes, seed, 1000), f'{phase}: incorrect full protocol range')
        phases.append(set(range(start, start + episodes)))
    require(phases[0].isdisjoint(phases[1]), 'Calibration overlaps holdout')
    beta = np.asarray(summary['beta_rounded'], dtype=float)
    require(beta.shape == (4,) and np.isfinite(beta).all() and
            np.all((beta > 0) & (beta <= 1.5)), 'Invalid deployed beta')
    require(summary['holdout']['beta_rounded'] == summary['beta_rounded'],
            'Holdout beta differs from the frozen fit')
    cal_bank, cal_report = audit_phase(directory, summary, 'calibration', beta)
    raw = [linear_quantile(np.concatenate(cal_bank[m]['ratios']), 0.1)
           for m in range(1, 5)]
    rounded = [round(value, 4) for value in raw]
    close(raw, summary['beta_raw'], 'Recomputed raw calibration quantile')
    require(rounded == summary['beta_rounded'], 'Recomputed rounded beta mismatch')
    hold_bank, hold_report = audit_phase(directory, summary, 'holdout', beta)
    for depth, bank in hold_bank.items():
        label = f'holdout depth {depth}'
        reported = summary['holdout']['by_depth'][str(depth)]
        counts = np.asarray([len(values) for values in bank['mi']], dtype=np.int64)
        ack_counts = np.asarray([np.count_nonzero(mi >= beta[depth - 1] * cap - 1e-6)
                                 for mi, cap in zip(bank['mi'], bank['caps'])], dtype=np.int64)
        require(reported['episode_member_counts'] == counts.tolist(),
                f'{label}: episode member counts mismatch')
        require(reported['episode_ack_counts'] == ack_counts.tolist(),
                f'{label}: episode ACK counts mismatch')
        require(reported['member_count'] == int(counts.sum()) and
                reported['acks'] == int(ack_counts.sum()), f'{label}: pooled counts mismatch')
        close(reported['first_ack'], ack_counts.sum() / counts.sum(), f'{label}: runtime ACK')
        close(reported['strict_ratio_first_ack'], hold_report[str(depth)]['strict_ratio_coverage'],
              f'{label}: strict ratio coverage')
        close(reported['ack_tolerance_bits'], 1e-6, f'{label}: runtime ACK tolerance')
        require(reported['zero_member_episodes'] == int(np.count_nonzero(counts == 0)),
                f'{label}: empty episode count mismatch')
        indices = np.random.default_rng(42).integers(0, len(counts), size=(10000, len(counts)))
        denominators = counts[indices].sum(axis=1)
        undefined = float(np.mean(denominators == 0))
        require(undefined == 0, f'{label}: undefined bootstrap draw')
        close(reported['bootstrap_undefined_fraction'], undefined, f'{label}: bootstrap undefined fraction')
        bootstrap = ack_counts[indices].sum(axis=1) / denominators
        ci = [linear_quantile(bootstrap, q) for q in (0.025, 0.975)]
        close(reported['episode_cluster_ci95'], ci, f'{label}: episode bootstrap CI')
        hold_report[str(depth)]['episode_cluster_ci95'] = ci
        if 'old_beta_on_same_holdout' in summary:
            old = summary['old_beta_on_same_holdout'][str(depth)]
            old_ack = np.mean(np.concatenate(bank['mi']) >= old['beta'] *
                              np.concatenate(bank['caps']) - 1e-6)
            close(old['first_ack'], old_ack, f'{label}: old beta diagnostic ACK')
    return dict(status='passed', results_directory=str(directory),
                calibration_fit_sha256=summary['calibration_fit_sha256'],
                beta_raw=raw, beta_rounded=rounded,
                calibration=cal_report, holdout=hold_report,
                verified='file hashes, frozen fit, sample arithmetic, membership, ACKs, episode bootstrap',
                scope='saved evidence consistency; channel generation and temporal ordering not replayed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.results_directory), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
