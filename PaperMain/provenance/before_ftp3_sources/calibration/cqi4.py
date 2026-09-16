"""Feedback-feasible, all-new full-capacity CQI4 reference calibration.

The sampling law is uniform over feedback-eligible UEs at each sampled RBG.
It does not model queue selection, retransmissions, or a trained policy. True
CSI is used only to score a group already selected from feedback. A frozen
beta must leave every sampled group feasible, or this protocol fails closed.
"""
from dataclasses import replace
import numpy as np

from la_planner import predict_group_link_adaptation
from phy import _rbg_sinr, mi_bits, predict_b_tx


class CalibrationError(ValueError):
    """Invalid measurement or a calibration that cannot be deployed as sampled."""


def eligible_users(cqi_fb, h_hat_rows, cfg):
    cqi = np.asarray(cqi_fb, dtype=float)
    h = np.asarray(h_hat_rows)
    if cqi.ndim != 1 or h.ndim != 2 or len(cqi) != len(h):
        raise CalibrationError('Feedback and reconstructed channel shapes do not match')
    if not np.isfinite(cqi).all() or not np.isfinite(h).all():
        raise CalibrationError('Nonfinite feedback cannot be sampled')
    cap = predict_b_tx(cqi, cfg)
    return np.flatnonzero((cap >= cfg.b_tx_epsilon) & (cqi > 0)
                          & np.any(h != 0, axis=1))


def sample_group(eligible, depth, rng):
    if depth < 1:
        raise CalibrationError('Group depth must be positive')
    if len(eligible) < depth:
        return None
    return np.sort(rng.choice(eligible, size=depth, replace=False))


def measure_group(h_hat_rows, h_true_rows, cfg, noise_var):
    h, actual = np.asarray(h_hat_rows), np.asarray(h_true_rows)
    if (h.ndim != 2 or h.shape != actual.shape or not 1 <= len(h) <= cfg.l_max
            or not np.isfinite(h).all() or not np.isfinite(actual).all()
            or not np.isfinite(noise_var) or noise_var <= 0):
        raise CalibrationError('Invalid group channel or noise')
    neutral = replace(cfg, la_beta=1.0, la_beta_by_depth=())
    _, _, cap = predict_group_link_adaptation(h, noise_var, noise_var,
                                             cfg.p_rbg, neutral)
    cap = np.asarray(cap, dtype=float)
    mi = np.asarray(mi_bits(_rbg_sinr(actual, h, noise_var, cfg.p_rbg,
                                    noise_var), cfg), dtype=float)
    if not np.isfinite(cap).all() or np.any(cap <= 0):
        raise CalibrationError('Group has nonpositive/nonfinite predicted capacity; no ratio is defined')
    if not np.isfinite(mi).all() or np.any(mi < 0):
        raise CalibrationError('Group has invalid realized mutual information')
    ratio = mi / cap
    if not np.isfinite(ratio).all():
        raise CalibrationError('Nonfinite MI/capacity ratio')
    # Zero actual MI with positive predicted capacity is a real outage.
    return dict(raw_cap=cap, mi=mi, ratio=ratio)


def collect_samples(env, cfg, ep0, n_ep, seed):
    if n_ep < 1 or ep0 < 0:
        raise CalibrationError('Invalid episode range')
    rng = np.random.default_rng(seed)
    depths = range(1, cfg.l_max + 1)
    ratios, caps, mis, diagnostics = ({m: [] for m in depths} for _ in range(4))
    empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    for ep in range(n_ep):
        cur, curcaps, curmis = ({m: [] for m in depths} for _ in range(3))
        counts = {m: dict(draws=0, groups=0, members=0, skipped_insufficient=0,
                         candidate_slots=0, excluded_candidates=0,
                         cqi_zero_candidates=0, real_zero_mi_members=0) for m in depths}
        env.reset(episode_idx=ep0 + ep)
        for t in range(cfg.episode_len):
            if t % 7 == 0:
                for m in depths:
                    for _ in range(2):
                        r = int(rng.integers(cfg.num_rbg))
                        cqi = env.csi.cqi_fb[:, r]
                        eligible = eligible_users(cqi, env.h_hat_slot[:, r, :], cfg)
                        d = counts[m]
                        d['draws'] += 1
                        d['candidate_slots'] += len(cqi)
                        d['excluded_candidates'] += len(cqi) - len(eligible)
                        d['cqi_zero_candidates'] += int(np.sum(cqi <= 0))
                        group = sample_group(eligible, m, rng)
                        if group is None:
                            d['skipped_insufficient'] += 1
                            continue
                        measured = measure_group(env.h_hat_slot[group, r, :],
                                                 env.h_true_slot[group, r, :],
                                                 cfg, env.noise_var)
                        cur[m].extend(measured['ratio'].tolist())
                        curcaps[m].extend(measured['raw_cap'].tolist())
                        curmis[m].extend(measured['mi'].tolist())
                        d['groups'] += 1
                        d['members'] += m
                        d['real_zero_mi_members'] += int(np.sum(measured['mi'] == 0))
            env.step(empty)
        for m in depths:
            ratios[m].append(np.asarray(cur[m], dtype=float))
            caps[m].append(np.asarray(curcaps[m], dtype=float))
            mis[m].append(np.asarray(curmis[m], dtype=float))
            diagnostics[m].append(counts[m])
        print(f'feasible CQI4 sampler: episode {ep0 + ep} ({ep+1}/{n_ep})', flush=True)
    return dict(ratios=ratios, caps=caps, mi=mis, diagnostics=diagnostics)


def _values(episodes, label, positive=False):
    if not episodes:
        raise CalibrationError(f'{label}: no episodes')
    arrays = [np.asarray(v, dtype=float) for v in episodes]
    if any(v.ndim != 1 for v in arrays):
        raise CalibrationError(f'{label}: expected one-dimensional episode samples')
    values = np.concatenate(arrays)
    if (not len(values) or not np.isfinite(values).all()
            or np.any(values <= 0 if positive else values < 0)):
        raise CalibrationError(f'{label}: empty or invalid samples')
    return arrays, values


def _beta(beta, depth):
    b = np.asarray(beta, dtype=float)
    if b.shape != (depth,) or not np.isfinite(b).all() or np.any((b <= 0) | (b > 1.5)):
        raise CalibrationError('Beta must be finite, positive, <=1.5, and cover every depth')
    return b


def fit_beta(ratios_by_depth):
    depths = list(range(1, len(ratios_by_depth) + 1))
    if not depths or set(ratios_by_depth) != set(depths):
        raise CalibrationError('Calibration requires contiguous depths from 1')
    raw = [float(np.percentile(_values(ratios_by_depth[m], f'depth {m} ratios')[1],
                               10, method='linear')) for m in depths]
    rounded = [round(b, 4) for b in raw]
    _beta(rounded, len(depths))
    return dict(beta_raw=raw, beta_rounded=rounded)


def validate_membership(caps_by_depth, beta, cfg):
    b = _beta(beta, cfg.l_max)
    if set(caps_by_depth) != set(range(1, cfg.l_max + 1)):
        raise CalibrationError('Membership evidence must cover every depth')
    evidence = {}
    for m in range(1, cfg.l_max + 1):
        _, cap = _values(caps_by_depth[m], f'depth {m} capacities', positive=True)
        scaled = b[m-1] * cap
        below = int(np.sum(scaled < cfg.b_tx_epsilon))
        if below:
            raise CalibrationError(f'Depth {m}: frozen beta removes {below}/{len(cap)} '
                                   'members at runtime epsilon; this calibration is not deployable')
        evidence[str(m)] = dict(member_count=len(cap), below_epsilon_members=below,
                                minimum_raw_cap=float(cap.min()),
                                minimum_scaled_cap=float(scaled.min()),
                                epsilon=float(cfg.b_tx_epsilon))
    return evidence


def summarize_holdout(ratios, caps, beta, cfg, realized_mi=None):
    validate_membership(caps, beta, cfg)
    b = _beta(beta, cfg.l_max)
    stats = {}
    for m in range(1, cfg.l_max + 1):
        arrays, _ = _values(ratios[m], f'holdout depth {m} ratios')
        if len(arrays) != len(caps[m]) or any(len(v) != len(c) for v, c in zip(arrays, caps[m])):
            raise CalibrationError('Holdout ratio/capacity episode samples do not match')
        ns = np.array([len(v) for v in arrays], dtype=np.int64)
        mi = ([v * np.asarray(c) for v, c in zip(arrays, caps[m])] if realized_mi is None
              else _values(realized_mi[m], f'holdout depth {m} MI')[0])
        if len(mi) != len(arrays) or any(len(v) != len(i) for v, i in zip(arrays, mi)):
            raise CalibrationError('Holdout MI samples do not match ratio/capacity samples')
        if any(not np.allclose(i, v*np.asarray(c), rtol=1e-12, atol=1e-12)
               for i, v, c in zip(mi, arrays, caps[m])):
            raise CalibrationError('Stored MI is inconsistent with the ratio/capacity pairs')
        # Match TransmissionUnit.is_acked (absolute tolerance in bits).
        acks = np.array([int(np.sum(i >= b[m-1] * np.asarray(c) - 1e-6))
                         for i, c in zip(mi, caps[m])], dtype=np.int64)
        idx = np.random.default_rng(42).integers(0, len(arrays), (10000, len(arrays)))
        denominators = ns[idx].sum(axis=1)
        if np.any(denominators == 0):
            raise CalibrationError(f'Depth {m}: undefined zero-member episode bootstrap draw')
        boot = acks[idx].sum(axis=1) / denominators
        stats[str(m)] = dict(first_ack=float(acks.sum()/ns.sum()),
                             episode_cluster_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                             member_count=int(ns.sum()), acks=int(acks.sum()),
                             episode_member_counts=ns.tolist(), episode_ack_counts=acks.tolist(),
                             zero_member_episodes=int(np.sum(ns == 0)),
                             bootstrap_undefined_fraction=0.0, ack_tolerance_bits=1e-6,
                             strict_ratio_first_ack=float(np.mean(np.concatenate(arrays) >= b[m-1])))
    return stats
