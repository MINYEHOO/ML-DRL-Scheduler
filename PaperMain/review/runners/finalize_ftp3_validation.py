from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,re,sys
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler/PaperMain")
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch
torch.set_num_threads(1)
from paper_train import normalize_config,source_hashes,calibration_reference
from calibration.profile import validate_profile_record
r=Path('/home/MYH/ML_DRL_Scheduler/PaperMain');logs=r/'review/logs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
smoke_name='17_audit_ftp3_gpu4';name='18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4'
smoke=json.loads((logs/(smoke_name+'_launch.json')).read_text())
assert smoke['status']=='completed' and smoke['checkpoint_update']==1
assert smoke['resume_verified'] and smoke['actual_model_update_verified']
unit=(logs/'ftp3_regression_tests.log').read_text();n=int(re.search(r'Ran (\d+) tests',unit).group(1));assert n==166 and '\nOK\n' in unit
run=r/'runs'/name;manifest=json.loads((run/'paper_manifest.json').read_text());state=json.loads((logs/(name+'_launch.json')).read_text())
assert state['status']=='running';os.kill(state['training_pid'],0)
assert manifest['source_sha256']==source_hashes(r)==smoke['source_sha256']
base=json.loads((r/'runs/14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5/paper_manifest.json').read_text())
expected=normalize_config(base['config']);expected['traffic_model']='ftp3'
assert manifest['config']==expected
assert manifest['schedule']==base['schedule']==dict(lr_final=0.0,lr_decay_updates=1000)
assert manifest['execution']==dict(device='cuda',gpu='4',threads=4)
assert manifest['target_updates']==1000 and manifest['smoke_slots'] is None
ref=calibration_reference(r,manifest['calibration_reference_root']);validate_profile_record(manifest['calibration_profile'],ref)
old16=json.loads((r/'runs/16_base_cqi4_lrrestart2000_s2024_20260910_gpu5/paper_manifest.json').read_text())
assert source_hashes(ref)==old16['source_sha256']
assert sha(r/'results/cqi4_hl_corrected_v2/summary.json') == old16['calibration_profile']['file_sha256'] == manifest['calibration_profile']['file_sha256']
oldstate=json.loads((logs/'16_base_cqi4_lrrestart2000_s2024_20260910_gpu5_launch.json').read_text())
assert oldstate['status']=='running';os.kill(oldstate['training_pid'],0)
assert 'Traceback (most recent call last)' not in (logs/(name+'.log')).read_text()
with (run/'csv_logs/env_metrics.csv').open() as f:rows=list(csv.DictReader(f))
assert rows and int(rows[-1]['update'])>=0 and all(__import__('math').isfinite(float(row['reward'])) for row in rows)
ckpt=None
if (run/'ckpt/latest.pt').exists():
    ck=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
    assert normalize_config(ck['cfg'])==manifest['config'] and ck['lr_schedule']==manifest['schedule']
    assert all(torch.isfinite(t).all() for t in ck['model'].values())
    ckpt=dict(update=int(ck['update']),optimizer_lr=ck['optimizer']['param_groups'][0]['lr'],finite=True)
source=json.loads((r/'provenance/source_manifest.json').read_text())
for e in source['entries']:assert sha(r/e['path'])==sha(r.parent/e['source'])==e['sha256']
for f,h in source['original_root_source_sha256'].items():assert sha(r.parent/f)==h
orderpath=r/'runs/RUN_ORDER.json';order=json.loads(orderpath.read_text())
for seq,nm in [(17,smoke_name),(18,name)]:
    if not any(e['name']==nm for e in order['runs']):
        launch=json.loads((logs/(nm+'_launch.json')).read_text());assert not any(e['sequence']==seq for e in order['runs'])
        order['runs'].append(dict(original_name=nm,name=nm,sequence=seq,started_at_utc=launch['created_at'],time_basis='launch.created_at',config_created_at_utc=datetime.fromtimestamp((r/'runs'/nm/'config.json').stat().st_mtime,timezone.utc).isoformat(),legacy_alias_retained=False))
order['runs'].sort(key=lambda e:e['sequence']);order['updated_at_utc']=datetime.now(timezone.utc).isoformat()
tmp=orderpath.with_suffix('.tmp');tmp.write_text(json.dumps(order,indent=2)+'\n');os.replace(tmp,orderpath)
import paper_run_index
paper_run_index.main()
result=dict(status='passed',validated_at=datetime.now(timezone.utc).isoformat(),tests=n,new_traffic_tests=12,new_wrapper_tests=19,
    traffic=dict(model='ftp3',arrival_process='per-UE Poisson counts conditional on sampled episode intensity; slot-boundary arrivals',mean_per_ue_per_slot=[.15,.5],mean_per_ue_per_second=[300,1000],same_mean_offered_load=True,packet_bits=[4000,12000],deadline_slots=[3,12],queue_capacity=8,slot_seconds=.0005,legacy_bernoulli_rng_exact=True,shared_csi_rng_consumption_changes=True),
    calibration=dict(application=manifest['calibration_application'],reference_root=manifest['calibration_reference_root'],beta=manifest['config']['la_beta_by_depth'],original_profile_unmodified=True,ftp3_holdout_validated=False),
    smoke=dict(name=smoke_name,status=smoke['status'],phases=smoke['phases'],resume_verified=True,actual_model_update_verified=True),
    main=dict(name=name,status=state['status'],physical_gpu=4,seed=2024,target_updates=1000,initial_lr=.0003,schedule=manifest['schedule'],run_dir=str(run),training_pid=state['training_pid'],supervisor_pid=state['supervisor_pid'],completed_training_updates=int(rows[-1]['update'])+1,checkpoint=ckpt,config_diff_vs_run14=['traffic_model'],eval_csv=str(run/'eval.csv'),eval_every=10,eval_episodes=3),
    bernoulli_run16=dict(status=oldstate['status'],physical_gpu=5,training_pid=oldstate['training_pid'],source_snapshot_matches_manifest=True),
    original_code_files_unchanged=len(source['original_root_source_sha256']),original_and_copied_artifacts_unchanged=len(source['entries']),source_sha256=source_hashes(r))
(r/'review/ftp3_validation.json').write_text(json.dumps(result,indent=2)+'\n')
files={}
for f in sorted(r.rglob('*')):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(r)
    if rel.parts[0] in ('runs','results','reports') or '__pycache__' in rel.parts or f.suffix=='.pyc' or rel.as_posix().startswith('review/logs/') or f.name=='PACKAGE_MANIFEST.json' or f.name.startswith('._'):continue
    files[rel.as_posix()]=sha(f)
(r/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(version=9,scope='PaperMain optional FTP3 Poisson arrivals; Bernoulli preserved; explicit frozen CQI4 beta transfer; GPU4 fresh training',excludes='mutable runs/results/reports/logs/cache',sha256=files),indent=2)+'\n')
from verify_package import main
main()
print(json.dumps(result,indent=2))
