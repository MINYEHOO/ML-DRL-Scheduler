"""Common evaluation of the three paper recipes; archived evidence is never edited.

All policies see the Base seed-2024 world; trained feature normalizers are frozen.
For K-scaling only the tensor dimension is changed in the policy configuration.
This runner defines a new, recorded protocol; historical CSVs remain historical.
"""
from pathlib import Path
import argparse, copy, csv, dataclasses, hashlib, importlib.metadata, json, os, sys, time

ROOT = Path(__file__).resolve().parent
RECIPES = ('base', 'lrann', 'narrow', 'base_s3024', 'base_s4024', 'base_s5024')
WORLDS = {
    'ID': ({}, .75),
    'P055': ({'p_arrival_min': .55, 'p_arrival_max': .55}, .75),
    'P010': ({'p_arrival_min': .10, 'p_arrival_max': .10}, .80),
    'V60max': ({'ue_speed_max': 60.}, .75),
    'CSI02': ({'p_csi': .2}, .70),
    'D26': ({'deadline_min': 2, 'deadline_max': 6}, .80),
    'STORM2': ({'p_arrival_min': .55, 'p_arrival_max': .55, 'ue_speed_max': 60., 'p_csi': .2}, .75),
    'K8': ({'num_ue': 8, 'n_active_min': 8, 'n_active_max': 8}, .80),
    'K48': ({'num_ue': 48, 'n_active_min': 48, 'n_active_max': 48}, .75),
    'K60': ({'num_ue': 60, 'n_active_min': 60, 'n_active_max': 60}, .80),
}

def config_from_raw(raw):
    from config import Config
    fields = {f.name: f for f in dataclasses.fields(Config)}
    unknown = set(raw) - set(fields) - {'git_hash', 'git_dirty_py'}
    if unknown:
        raise ValueError(f'Unrecognized configuration fields: {sorted(unknown)}')
    out = {k: v for k, v in raw.items() if k in fields}
    defaults = Config()
    for k, v in out.items():
        if isinstance(getattr(defaults, k), tuple): out[k] = tuple(v or ())
    cfg = Config(**out)
    cfg.validate_la()
    return cfg

def recipe_config(recipe):
    return config_from_raw(json.loads((ROOT/'artifacts'/recipe/'config.json').read_text()))

def make_configs(recipe, world, smoke_slots=None, calibration_profile=None):
    env_cfg, policy_cfg = recipe_config('base'), recipe_config(recipe)
    if calibration_profile is not None:
        from calibration.profile import apply_profile
        env_cfg = apply_profile(env_cfg, calibration_profile)
        policy_cfg = apply_profile(policy_cfg, calibration_profile)
    changes, threshold = WORLDS[world]
    for k, v in changes.items(): setattr(env_cfg, k, v)
    env_cfg.sus_ortho_threshold = threshold
    # Variable user counts change tensor shapes, not learned feature scales.
    policy_cfg.num_ue = env_cfg.num_ue
    if smoke_slots:
        env_cfg.debug = False
        env_cfg.episode_len_main = smoke_slots
    assert env_cfg.seed == 2024
    assert policy_cfg.deadline_max == 12, 'Training-time deadline normalization must stay /12'
    return env_cfg, policy_cfg

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipes', nargs='+', choices=RECIPES, default=list(RECIPES[:3]))
    p.add_argument('--worlds', nargs='+', choices=list(WORLDS), default=['ID'])
    p.add_argument('--episode-start', type=int, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--smoke-slots', type=int)
    p.add_argument('--baselines', choices=['none','strong','all'], default='strong')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--calibration-profile',
                   help='validated CQI4 summary JSON; evaluates archived policies with corrected beta')
    p.add_argument('--out', required=True, help='New directory below this package results/')
    a = p.parse_args()
    if a.episode_start < 0 or a.episodes < 1 or a.threads < 1 or (a.smoke_slots is not None and a.smoke_slots < 2):
        p.error('Invalid episode/thread/smoke dimensions')
    if len(set(a.recipes)) != len(a.recipes) or len(set(a.worlds)) != len(a.worlds):
        p.error('Duplicate recipes/worlds would duplicate evaluation rows')
    out = Path(a.out)
    if not out.is_absolute(): out = ROOT/out
    out = out.resolve()
    if not out.is_relative_to((ROOT/'results').resolve()):
        p.error('--out must be inside this package results/')
    if out.exists(): p.error(f'Refusing to overwrite {out}')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if a.device == 'cpu' else str(a.gpu)
    for name in ('OMP','MKL','OPENBLAS','NUMEXPR','VECLIB_MAXIMUM'):
        os.environ[name+'_THREADS' if name == 'VECLIB_MAXIMUM' else name+'_NUM_THREADS'] = str(a.threads)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(a.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    profile_record = None
    profile = None
    if a.calibration_profile is not None:
        from calibration.profile import load_profile_record, reject_calibration_episode_overlap
        profile_path = Path(a.calibration_profile)
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        try:
            profile_record = load_profile_record(profile_path, ROOT)
            profile = profile_record['document']
            reject_calibration_episode_overlap(profile, a.episode_start, a.episodes)
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))
    import numpy as np, torch
    torch.set_num_threads(a.threads)
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    from policy import ActorCritic
    from env import SchedulerEnv
    from train_phase2 import PPOScheduler, env_episode_metrics
    from baselines import (all_baselines, SUSRPS, SURPS, SUSCQIFeasible,
                           SUSDeadlinePFFeasible, SUSCQIDepth3, PFGreedySDS, CQIGreedySDS)
    loaded = {}
    for recipe in a.recipes:
        path = ROOT/'artifacts'/recipe/'ckpt/best.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        # Ensure archived JSON has not drifted away from the selected checkpoint.
        saved_cfg = config_from_raw(ck['cfg'])
        declared_cfg = recipe_config(recipe)
        if dataclasses.asdict(saved_cfg) != dataclasses.asdict(declared_cfg):
            raise ValueError(f'{recipe}: checkpoint/config mismatch')
        loaded[recipe] = ck
    out.mkdir(parents=True)
    manifest = dict(protocol=('paper-common-cqi4-corrected-v1' if profile else 'paper-common-v1'),
        status='running', arguments=vars(a), calibration_profile=profile_record,
        calibration_application=('Frozen archived policies with Base-calibrated beta; OOD worlds are transfer tests'
                                 if profile else 'archived beta'),
        diagnostic_only=bool(a.smoke_slots), evaluation_world_seed=2024,
        caveats=['Same seed does not guarantee identical scheduler-dependent realized traffic/CSI traces.',
                 '20000/21000 bands are previously inspected historical bands, not new blind tests.',
                 ('Corrected beta is applied without retraining archived policies; results measure sensitivity, not retrained performance.'
                  if profile else 'Original beta calibration remains unchanged and has a known CQI=0 sampling defect.'),
                 'Random baseline RNG is reset per episode in this runner; archived runners may differ.'],
        checkpoint_sha256={r:sha(ROOT/'artifacts'/r/'ckpt/best.pt') for r in a.recipes},
        source_sha256={**{f.name:sha(f) for f in ROOT.glob('*.py')},
                       **{f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'calibration').glob('*.py')}},
        versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','tensorflow','sionna','torch']},
        configurations={w:{r:{'environment':dataclasses.asdict(make_configs(r,w,a.smoke_slots,profile)[0]),
                              'policy':dataclasses.asdict(make_configs(r,w,a.smoke_slots,profile)[1])}
                           for r in a.recipes} for w in a.worlds})
    mpath = out/'manifest.json'
    mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    keys = ['reward','throughput_mbps','goodput_mbps','completion_rate','deadline_miss_rate',
            'retx_drop_rate','buffer_overflow_rate','mu_depth','jain','first_ack_rate',
            'n_arrivals','n_active','mean_speed_kmh','n_units_new','n_units_first_ack']
    keys += [f'{prefix}_m{m}' for m in range(1,5) for prefix in ('acks','units')]
    header = ['world','scheduler','episode_idx','train_seed','best_update','diagnostic_only']+keys
    t0 = time.monotonic()
    try:
        with (out/'metrics.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f,fieldnames=header); writer.writeheader()
            for world in a.worlds:
                env_cfg, _ = make_configs('base',world,a.smoke_slots,profile)
                env = SchedulerEnv(env_cfg)
                schedulers = []
                for recipe in a.recipes:
                    _, pcfg = make_configs(recipe,world,a.smoke_slots,profile)
                    model = ActorCritic(pcfg)
                    model.load_state_dict(loaded[recipe]['model'],strict=True)
                    model.eval().to(a.device)
                    schedulers.append((f'PPO/{recipe}',PPOScheduler(model),pcfg.seed,loaded[recipe]['update']))
                baseline_set = all_baselines(env_cfg) if a.baselines == 'all' else [s for s in all_baselines(env_cfg) if s.name in ('SUS+CQI','SU+CQI')]
                if a.baselines != 'none':
                    baseline_set += [c() for c in [SUSRPS,SURPS,SUSCQIFeasible,SUSDeadlinePFFeasible,SUSCQIDepth3,PFGreedySDS,CQIGreedySDS]]
                    schedulers += [(s.name,s,'','') for s in baseline_set]
                for ep in range(a.episode_start,a.episode_start+a.episodes):
                    for name, sched, seed, update in schedulers:
                        # Stable across scheduler ordering, recipe subsets, and shards.
                        if hasattr(sched,'rng'):
                            token = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
                            sched.rng = np.random.default_rng(np.random.SeedSequence([2024,ep,token]))
                        env.reset(episode_idx=ep)
                        done = False
                        while not done:
                            alloc = sched.schedule(env)
                            assert alloc.shape == (env_cfg.num_rbg,env_cfg.l_max)
                            assert alloc.min() >= 0 and alloc.max() <= env_cfg.num_ue
                            _, reward, done, _ = env.step(alloc)
                            assert np.isfinite(reward), (world,ep,name,reward)
                        metrics = env_episode_metrics(env,env_cfg)
                        metrics.update(n_units_new=int(env.ep['n_units_new']),
                                       n_units_first_ack=int(env.ep['n_units_first_ack']))
                        assert all(np.isfinite(metrics[k]) for k in keys)
                        writer.writerow(dict(world=world,scheduler=name,episode_idx=ep,
                            train_seed=seed,best_update=update,diagnostic_only=int(bool(a.smoke_slots)),
                            **{k:metrics[k] for k in keys}))
                        f.flush()
                    print(f'{world} episode={ep} schedulers={len(schedulers)} elapsed={time.monotonic()-t0:.1f}s',flush=True)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status']='failed';manifest['error']=repr(exc)
        raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-t0
        mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)

if __name__ == '__main__': main()
