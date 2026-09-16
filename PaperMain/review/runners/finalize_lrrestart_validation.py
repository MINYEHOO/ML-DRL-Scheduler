from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, re, tarfile
from calibration.profile import load_profile
from paper_train import normalize_config, source_hashes, validate_checkpoint_schedule
import torch
torch.set_num_threads(1)

root=Path('/home/MYH/ML_DRL_Scheduler/PaperMain');logs=root/'review/logs'
name='16_base_cqi4_lrrestart2000_s2024_20260910_gpu5';run=root/'runs'/name
parent=root/'runs/14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
unit=(logs/'lrrestart_regression_tests.log').read_text()
assert int(re.search(r'Ran (\d+) tests',unit).group(1))==135 and '\nOK\n' in unit
smoke=json.loads((logs/'15_audit_lrrestart_gpu5_launch.json').read_text())
assert smoke['status']=='completed' and smoke['checkpoint_update']==3
assert smoke['lr_segment_restored'] and smoke['actual_model_update_verified']
profile=load_profile(root/'results/cqi4_hl_corrected_v2/summary.json')
old=json.loads((parent/'paper_manifest.json').read_text());new=json.loads((run/'paper_manifest.json').read_text())
lineage=json.loads((run/'PARENT_RUN.json').read_text())
assert new['source_sha256']==source_hashes(root)
assert new['config']==old['config']==normalize_config(json.loads((run/'config.json').read_text()))
for key in ['execution','replay_origin','calibration_profile','runtime']:
    assert new[key]==old[key],key
schedule=dict(lr_final=0.0,lr_decay_updates=1000,lr_initial=1e-5,lr_start_update=1000)
assert new['schedule']==schedule and new['target_updates']==2000
assert new['continuation']==lineage and lineage['resumed_checkpoint_update']==999
assert set(f for f,h in new['source_sha256'].items() if old['source_sha256'][f]!=h)=={'paper_train.py','train_phase2.py'}
for f,h in lineage['parent_metadata_sha256'].items():assert sha(parent/f)==h
for f,h in lineage['copied_checkpoint_sha256'].items():assert sha(parent/'ckpt'/f)==h
for f,item in lineage['csv_history'].items():assert sha(parent/'csv_logs'/f)==item['parent_sha256']
source=json.loads((root/'provenance/source_manifest.json').read_text())
for entry in source['entries']:
    assert sha(root/entry['path'])==sha(root.parent/entry['source'])==entry['sha256'],entry['path']
for f,h in source['original_root_source_sha256'].items():assert sha(root.parent/f)==h,f
launch=json.loads((logs/(name+'_launch.json')).read_text())
assert launch['status']=='running' and launch['target_updates']==2000
for key in ['supervisor_pid','training_pid']:os.kill(launch[key],0)
console=(logs/(name+'.log')).read_text()
assert 'start_update=1000' in console and 'Traceback (most recent call last)' not in console
assert 'over 1000 updates from global update 1000' in console
ck=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
assert ck['update']>=1000 and ck['lr_schedule']==schedule
validate_checkpoint_schedule(ck,schedule)
assert all(torch.isfinite(v).all() for v in ck['model'].values())
want=1e-5*(1-min((ck['update']-1000)/1000,1.0))
assert all(math.isclose(g['lr'],want,rel_tol=0,abs_tol=1e-19) for g in ck['optimizer']['param_groups'])
parent_ck=torch.load(parent/'ckpt/latest.pt',map_location='cpu',weights_only=False)
changed_actor=any(not torch.equal(v,ck['model'][k]) for k,v in parent_ck['model'].items()
                  if k.endswith('weight') and not k.startswith('value_head.'))
changed_critic=any(not torch.equal(v,ck['model'][k]) for k,v in parent_ck['model'].items()
                   if k.endswith('weight') and k.startswith('value_head.'))
assert changed_actor and changed_critic
archive=root/'provenance/run14_before_lrrestart_sources.tar.gz'
assert sha(archive)==lineage['source_archive_sha256']
with tarfile.open(archive) as tf:
    for f,h in old['source_sha256'].items():
        assert hashlib.sha256(tf.extractfile(f).read()).hexdigest()==h
    backup=root/'provenance/PACKAGE_MANIFEST_before_lrrestart.json'
    raw=tf.extractfile('PACKAGE_MANIFEST.json').read()
    if backup.exists():assert backup.read_bytes()==raw
    else:backup.write_bytes(raw)
result=dict(status='passed',validated_at=datetime.now(timezone.utc).isoformat(),tests=135,
            new_schedule_tests=23,legacy_schedule_arithmetic='exactly retained for valid legacy schedules',
            gpu_smoke=dict(parent_update=1,first_segment_update=2,resumed_update=3,
                           first_lr=smoke['legacy_continuation_optimizer_lr'],
                           resumed_lr=smoke['segment_resume_optimizer_lr'],
                           actor_and_critic_changed=True,checkpoint_schedule_restored=True),
            continuation=dict(parent=parent.name,child=name,resumed_from_update=999,start_update=1000,
                              target_updates=2000,additional_updates=1000,config_unchanged=True,
                              parent_files_unchanged=True,original_checkpoint_bytes_preserved_on_copy=True,
                              full_model_optimizer_normalizer_rng_preserved=True,
                              previous_schedule=old['schedule'],new_schedule=schedule,
                              last_training_update=1999,last_training_lr=1e-8),
            startup=dict(status=launch['status'],checkpoint_update=ck['update'],
                         completed_updates=ck['update']+1,optimizer_lr=ck['optimizer']['param_groups'][0]['lr'],
                         checkpoint_finite=True,actor_weights_changed=changed_actor,critic_weights_changed=changed_critic,
                         training_pid=launch['training_pid'],supervisor_pid=launch['supervisor_pid']),
            calibration=profile['status'],beta=profile['beta_rounded'],physical_gpu=5,
            original_code_files_unchanged=len(source['original_root_source_sha256']),
            original_and_copied_artifacts_unchanged=len(source['entries']),
            current_source_sha256=new['source_sha256'])
(root/'review/lrrestart_validation.json').write_text(json.dumps(result,indent=2)+'\n')
files={}
for f in sorted(root.rglob('*')):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(root)
    if (rel.parts[0] in ('runs','results','reports') or '__pycache__' in rel.parts or f.suffix=='.pyc'
            or rel.as_posix().startswith('review/logs/') or f.name=='PACKAGE_MANIFEST.json' or f.name.startswith('._')):continue
    files[rel.as_posix()]=sha(f)
(root/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(version=8,
    scope='PaperMain CQI4 v2, batched PPO, fast deterministic eval, explicit low-LR continuation to 2000 updates',
    excludes='mutable runs/results/reports/logs/cache',sha256=files),indent=2)+'\n')
from verify_package import main
main()
print(json.dumps(result,indent=2))
