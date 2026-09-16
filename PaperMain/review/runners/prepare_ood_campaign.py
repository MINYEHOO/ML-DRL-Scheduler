"""Freeze three completed checkpoints and audit fresh OOD episode bands."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, os, shutil, sys

ROOT = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
CAMPAIGN = ROOT / 'results/bernoulli_ood_runs16_21_22_20260913'
RUNS = {
    16: ('16_base_cqi4_lrrestart2000_s2024_20260910_gpu5', 1989, '04ce34d239f4ac377b4fd27f40a39640d966094dc9d01fa0eb4d17e2c69dac45'),
    21: ('21_base_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu3', 649, 'd1980adc99ec4cd14ea332fc8fb2ac4a2a5ef0339043ff4a9d49f161555f91a2'),
    22: ('22_narrow_mean_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu5', 1469, '539e72c35cd40699e17d1738b1a606d0a17c7d3b7239c71b41a91506967f9bda'),
}
RANGES = [(138000, 138002), (139000, 139008), (140000, 140100)]
CHANNEL_RANGES = [(a+2024, b+2024) for a,b in RANGES]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0, str(ROOT))
from paper_train import source_hashes, atomic_json

def overlap(a,b):
    return any(max(a,lo) < min(b,hi) for lo,hi in RANGES+CHANNEL_RANGES)

def main():
    if CAMPAIGN.exists():
        raise ValueError('Campaign already exists; never overwrite prior evaluation')
    package = json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    for name, digest in package['sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError(f'Packaged input changed: {name}')
    skip = {'.git','__pycache__','.venv','venv','node_modules','site-packages','.mypy_cache','.pytest_cache'}
    records, json_records, examined = [], [], []
    for dirname, dirs, files in os.walk(ROOT.parent):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.cache') and not (Path(dirname)/d).is_symlink()]
        for filename in files:
            path = Path(dirname)/filename
            if path.is_symlink() or filename.startswith('._'): continue
            relative = str(path.relative_to(ROOT.parent))
            if path.suffix == '.csv':
                try:
                    with path.open() as stream:
                        reader = csv.DictReader(stream)
                        keys = [k for k in ('episode_idx','episode','ep_idx','eval_seed','seed') if k in (reader.fieldnames or [])]
                        if not keys: continue
                        ids = {k:set() for k in keys}
                        for row in reader:
                            for key in keys:
                                value = row.get(key)
                                if value in ('',None): continue
                                number = float(value)
                                if number.is_integer(): ids[key].add(int(number))
                    for key, seen in ids.items():
                        if any(overlap(v,v+1) for v in seen): raise RuntimeError(f'Episode band already used in {relative}:{key}')
                        if seen: records.append(dict(path=relative,column=key,min=min(seen),max=max(seen),unique=len(seen)))
                    examined.append(relative)
                except (UnicodeDecodeError,csv.Error):
                    raise RuntimeError(f'Cannot audit evaluation CSV {relative}')
            elif path.suffix == '.json':
                try: doc = json.loads(path.read_text())
                except (ValueError,UnicodeDecodeError): continue
                def visit(value, location=''):
                    if isinstance(value,dict):
                        start = value.get('episode_start',value.get('start'))
                        count = value.get('episodes')
                        if type(start) is int and type(count) is int and count>0:
                            if overlap(start,start+count): raise RuntimeError(f'Episode range already reserved in {relative}:{location}')
                        for key, val in value.items():
                            if key in ('episode_ids','eval_episode_ids','calibration_episode_ids') and isinstance(val,list):
                                if any(type(v)is int and overlap(v,v+1) for v in val): raise RuntimeError(f'Episode IDs already recorded in {relative}:{location}/{key}')
                            visit(val,location+'/'+key)
                    elif isinstance(value,list):
                        for i,val in enumerate(value): visit(val,location+'/'+str(i))
                visit(doc)
                json_records.append(relative)
    source_records = {}
    for n,(name,update,digest) in RUNS.items():
        run = ROOT/'runs'/name
        manifest = json.loads((run/'paper_manifest.json').read_text())
        completion = json.loads((ROOT/'review/logs'/f'{name}_launch.json').read_text())
        if completion['status']!='completed' or completion['returncode']!=0 or not completion['target_completed']:
            raise ValueError(f'Run {n} not completed')
        if sha(run/'ckpt/best.pt') != digest: raise ValueError(f'Run {n} selected checkpoint changed')
        cfg=manifest['config'];seed=cfg['seed']
        if overlap(seed,seed+manifest['target_updates']) or overlap(seed+10000,seed+10000+cfg['ppo_eval_episodes']):
            raise ValueError('Training/selection episodes overlap evaluation')
        source_records[str(n)] = dict(name=name,checkpoint_update=update,checkpoint_sha256=digest,
            manifest_sha256=sha(run/'paper_manifest.json'),config_sha256=sha(run/'config.json'),
            completion_sha256=sha(ROOT/'review/logs'/f'{name}_launch.json'),source_sha256=manifest['source_sha256'])
    inputs=CAMPAIGN/'inputs';inputs.mkdir(parents=True)
    for n,(name,_,_) in RUNS.items():
        out=inputs/f'run{n}';out.mkdir()
        for rel, dest in [('paper_manifest.json','paper_manifest.json'),('config.json','config.json'),('ckpt/best.pt','best.pt')]:
            src=ROOT/'runs'/name/rel;shutil.copy2(src,out/dest)
            if sha(src)!=sha(out/dest): raise ValueError('Frozen input copy mismatch')
    audit=dict(status='validated',audited_at=datetime.now(timezone.utc).isoformat(),scope=str(ROOT.parent),
        checked_run_names=[v[0] for v in RUNS.values()],scanned_paths=sorted(set(examined+json_records)),
        allowed_episode_ranges=[dict(start=138000,episodes=2,purpose='smoke'),dict(start=139000,episodes=8,purpose='pilot'),dict(start=140000,episodes=100,purpose='eval')],
        channel_seed_ranges=CHANNEL_RANGES,world_seed=2024,historical_csv_columns=records,
        historical_csv_files=len(examined),historical_json_files=len(json_records),
        interpretation='No matching episode IDs or effective seed bands in available original-repository CSVs and structured JSON. Training/selection bands excluded; evaluator also validates calibration overlap. This is an available-record audit, not proof about unrecorded external experiments.')
    atomic_json(inputs/'episode_history_audit.json',audit)
    atomic_json(inputs/'frozen_inputs.json',dict(created_at=datetime.now(timezone.utc).isoformat(),runs=source_records,
        current_source_sha256=source_hashes(ROOT),package_manifest_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
        package_files=len(package['sha256']),history_audit_sha256=sha(inputs/'episode_history_audit.json')))
    print(json.dumps(dict(campaign=str(CAMPAIGN),runs=source_records,history_csv_files=len(examined),history_json_files=len(json_records)),indent=2))

if __name__=='__main__': main()
