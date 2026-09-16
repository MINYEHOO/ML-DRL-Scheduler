"""Calibrate feedback-feasible CQI4 groups in the frozen HL Base world.

Default: 36 calibration + 12 disjoint holdout episodes of 1000 slots. Short
runs are diagnostics and cannot be applied to training/evaluation. --mode
archived reproduces the known flawed sampler for historical comparison only.
No mode modifies archived configs, checkpoints, or an existing result folder.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def collect(env, cfg, ep0, episodes, seed):
    """Original sample order/RNG/math, with counters that consume no RNG."""
    import numpy as np
    from la_planner import predict_group_link_adaptation
    from phy import _rbg_sinr, mi_bits
    rng = np.random.default_rng(seed)
    empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    out = {m: [] for m in (1, 2, 3, 4)}
    diagnostics = {m: [] for m in out}
    for ep in range(episodes):
        cur = {m: [] for m in out}
        counts = {m: dict(members=0, zero_cap_members=0, below_epsilon_members=0,
                         groups=0, groups_with_zero_cap=0) for m in out}
        env.reset(episode_idx=ep0 + ep)
        for t in range(cfg.episode_len):
            if t % 7 == 0:
                nv = env.noise_var
                for m in (1, 2, 3, 4):
                    for _ in range(2):
                        r = int(rng.integers(cfg.num_rbg))
                        grp = rng.choice(cfg.num_ue, size=m, replace=False)
                        _, sp, _ = predict_group_link_adaptation(
                            env.h_hat_slot[grp, r, :], nv, nv, cfg.p_rbg, cfg)
                        cap = (cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
                               * np.log2(1 + sp))
                        mi = mi_bits(_rbg_sinr(env.h_true_slot[grp, r, :],
                                              env.h_hat_slot[grp, r, :],
                                              nv, cfg.p_rbg, nv), cfg)
                        cur[m].extend((mi / np.maximum(cap, 1e-12)).tolist())
                        counts[m]['members'] += m
                        counts[m]['groups'] += 1
                        counts[m]['zero_cap_members'] += int(np.sum(cap <= 0))
                        counts[m]['below_epsilon_members'] += int(np.sum(cap < cfg.b_tx_epsilon))
                        counts[m]['groups_with_zero_cap'] += int(np.any(cap <= 0))
            env.step(empty)
        for m in out:
            out[m].append(np.asarray(cur[m]))
            diagnostics[m].append(counts[m])
        print(f"archived sampler: episode {ep0 + ep} ({ep+1}/{episodes})", flush=True)
    return out, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['corrected', 'archived'], default='corrected')
    parser.add_argument('--config', type=Path, default=ROOT/'artifacts/base/config.json')
    parser.add_argument('--episodes', type=int, default=36, help='calibration episodes (default 36)')
    parser.add_argument('--holdout-episodes', type=int, default=12)
    parser.add_argument('--slots', type=int, help='smoke-only episode length override')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path, help='new output directory, must not exist')
    args = parser.parse_args()
    if min(args.episodes, args.holdout_episodes, args.threads) < 1 or (args.slots is not None and args.slots < 1):
        parser.error('episode counts, threads and optional slots must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    for v in ('OMP', 'MKL', 'OPENBLAS', 'NUMEXPR'):
        os.environ[f'{v}_NUM_THREADS'] = str(args.threads)
    import numpy as np
    from config import Config
    from env import SchedulerEnv
    from calibration.beta_m_calib_holdout import cluster_bootstrap_ci
    from calibration.profile import (normalized_config, source_hashes, sha256_file,
                                     validate_profile, PROTOCOL, SCHEMA_VERSION)

    raw = json.loads(args.config.read_text())
    raw = normalized_config(raw)
    reference = normalized_config(json.loads((ROOT/'artifacts/base/config.json').read_text()))
    if args.mode == 'corrected' and raw != reference:
        parser.error('corrected calibration requires the unchanged HL Base configuration')
    if args.slots is not None:
        raw.update(episode_len_main=args.slots, episode_len_debug=args.slots)
    cfg = Config(**raw)
    outdir = args.output or ROOT/'results'/('calibration_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if not outdir.is_absolute():
        outdir = ROOT/outdir
    outdir = outdir.resolve()
    if (ROOT/'results').resolve() not in outdir.parents:
        parser.error('output must be a new subdirectory of PaperMain/results/')
    outdir.mkdir(parents=True, exist_ok=False)
    if args.mode == 'corrected':
        from calibration.cqi4 import (collect_samples, fit_beta, validate_membership,
                                      summarize_holdout, CalibrationError)

        def save_json(name, value):
            (outdir/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

        def save_samples(name, bank):
            np.savez_compressed(outdir/name, **{
                f'{kind}_m{m}_ep{ep}': v for kind in ('ratios', 'caps', 'mi')
                for m, episodes in bank[kind].items() for ep, v in enumerate(episodes)})

        sources = source_hashes(ROOT)
        config_sha = sha256_file(ROOT/'artifacts/base/config.json')
        result = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL,
                      status='in_progress', config=asdict(cfg),
                      reference_config_sha256=config_sha, source_sha256=sources,
                      created_at=datetime.now(timezone.utc).isoformat(), threads=args.threads,
                      sampling=dict(every_slots=7, groups_per_depth_per_sample=2,
                                    law='uniform_without_replacement_over_feedback_eligible_UEs',
                                    traffic='empty actions; all-new full-capacity reference groups',
                                    insufficient_population='skip draw without redrawing RBG',
                                    membership='frozen beta times every raw cap must pass runtime epsilon'),
                      target_coverage=dict(target_first_ack=0.9,
                                           interpretation='reference-group estimate; not a policy or BLER guarantee'),
                      calibration=dict(start=50000, episodes=args.episodes,
                                       sampler_seed=777, slots=cfg.episode_len),
                      holdout=dict(start=70000, episodes=args.holdout_episodes,
                                   sampler_seed=20250713, slots=cfg.episode_len))
        save_json('summary.json', result)
        try:
            env = SchedulerEnv(cfg)
            cal = collect_samples(env, cfg, 50000, args.episodes, 777)
            save_samples('calibration_samples.npz', cal)
            result.update(fit_beta(cal['ratios']))
            beta = tuple(result['beta_rounded'])
            result['calibration']['membership'] = validate_membership(cal['caps'], beta, cfg)
            result['calibration']['diagnostics'] = cal['diagnostics']
            result['membership_verified_calibration'] = True
            # This write precedes collecting even the first holdout sample.
            save_json('calibration_fit.json', result)
            fit_sha = sha256_file(outdir/'calibration_fit.json')
            hold = collect_samples(env, cfg, 70000, args.holdout_episodes, 20250713)
            save_samples('holdout_samples.npz', hold)
            result['holdout']['beta_rounded'] = list(beta)
            result['holdout']['membership'] = validate_membership(hold['caps'], beta, cfg)
            result['holdout']['diagnostics'] = hold['diagnostics']
            result['holdout']['by_depth'] = summarize_holdout(hold['ratios'], hold['caps'], beta, cfg,
                                                             realized_mi=hold['mi'])
            # Same feasible sample bank, old beta: diagnostic comparison only.
            old_beta = json.loads((ROOT/'artifacts/base/config.json').read_text())['la_beta_by_depth']
            result['old_beta_on_same_holdout'] = {
                str(m): dict(beta=old_beta[m-1], first_ack=float(np.mean(
                    np.concatenate(hold['mi'][m]) >= old_beta[m-1] * np.concatenate(hold['caps'][m]) - 1e-6)))
                for m in range(1, cfg.l_max+1)}
            result['membership_verified_holdout'] = True
            if (source_hashes(ROOT) != sources
                    or sha256_file(ROOT/'artifacts/base/config.json') != config_sha
                    or sha256_file(outdir/'calibration_fit.json') != fit_sha):
                raise CalibrationError('Sources, reference config, or frozen fit changed during calibration')
            result['calibration_fit_sha256'] = fit_sha
            result['samples_sha256'] = {name: sha256_file(outdir/name) for name in
                                        ('calibration_samples.npz', 'holdout_samples.npz')}
            result['status'] = ('validated' if args.slots is None and args.episodes == 36 and args.holdout_episodes == 12
                                and cfg.episode_len == 1000 else 'diagnostic_only')
            if result['status'] == 'validated':
                validate_profile(result, ROOT)
            result['completed_at'] = datetime.now(timezone.utc).isoformat()
            save_json('summary.json', result)
            print(json.dumps(dict(status=result['status'], beta_rounded=list(beta),
                                  holdout=result['holdout']['by_depth']), indent=2))
            print(f'Calibration saved to {outdir}; apply explicitly with --calibration-profile {outdir}/summary.json')
        except Exception as error:
            result.update(status='failed', error=f'{type(error).__name__}: {error}')
            save_json('summary.json', result)
            raise
        return
    disclaimer = ('KNOWN FLAWED historical sampler: zero-cap members are retained and treated as failed ratios. '
                  'These values are not a corrected beta recommendation. No artifact was modified.')
    print(disclaimer, flush=True)
    env = SchedulerEnv(cfg)
    cal, caldiag = collect(env, cfg, 50000, args.episodes, 777)
    beta_raw = [float(np.percentile(np.concatenate(cal[m]), 10)) for m in (1, 2, 3, 4)]
    beta = [round(b, 4) for b in beta_raw]
    hold, holddiag = collect(env, cfg, 70000, args.holdout_episodes, 20250713)
    stats = {}
    for m in (1, 2, 3, 4):
        acks = [int((v >= beta[m-1]).sum()) for v in hold[m]]
        ns = [len(v) for v in hold[m]]
        stats[m] = dict(historical_first_ack=float(sum(acks)/sum(ns)),
                        episode_cluster_ci95=list(cluster_bootstrap_ci(acks, ns)),
                        member_count=sum(ns))
    result = dict(warning=disclaimer, status='historical_diagnostic_only', config=raw,
                  calibration=dict(start=50000, episodes=args.episodes, sampler_seed=777),
                  holdout=dict(start=70000, episodes=args.holdout_episodes, sampler_seed=20250713),
                  beta_raw=beta_raw, historical_beta_rounded=beta,
                  historical_holdout=stats, calibration_diagnostics=caldiag,
                  holdout_diagnostics=holddiag, threads=args.threads,
                  full_archived_counts_and_length=(args.episodes==36 and args.holdout_episodes==12 and cfg.episode_len==1000))
    (outdir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(outdir/'ratios.npz', **{
        f'{phase}_m{m}_ep{ep}': v for phase, data in [('cal',cal),('hold',hold)]
        for m, vals in data.items() for ep, v in enumerate(vals)})
    print(json.dumps(dict(historical_beta_rounded=beta, historical_holdout=stats), indent=2))
    print(f'Reproduction and diagnostics saved to {outdir}')


if __name__ == '__main__':
    main()
