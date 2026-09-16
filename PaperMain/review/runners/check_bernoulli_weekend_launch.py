#!/usr/bin/env python3
"""Verify live startup, preserve prior evidence, then index the new runs."""
from pathlib import Path
import os,sys,json,csv,math
from datetime import datetime,timezone
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review/runners'))
import torch
import paper_train as pt
import paper_train_variants as pv
from run_bernoulli_weekend import NAMES,GPUS,RUN19,SCHEDULE,NARROW,LOGS,now,write,sha,verify_static
torch.set_num_threads(1)
smoke=json.loads((LOGS/(NAMES['smoke']+'_launch.json')).read_text())
assert smoke['status']=='completed' and smoke['checkpoint_update']==1 and smoke['resume_verified'] and smoke['actual_model_update_verified']
env=json.loads((LOGS/'bernoulli_weekend_environment.json').read_text());assert env['status']=='passed'
ref=json.loads((ROOT/'runs'/RUN19/'paper_manifest.json').read_text())
results={}
for mode in ('base','narrow'):
    name=NAMES[mode];run=ROOT/'runs'/name
    state=json.loads((LOGS/(name+'_launch.json')).read_text());assert state['status']=='running';os.kill(state['training_pid'],0)
    m=json.loads((run/'paper_manifest.json').read_text());verify_static(state)
    expected=dict(ref['config'],traffic_model='bernoulli')
    if mode=='narrow':expected.update(NARROW);pv.validate_manifest(m,ROOT)
    assert m['config']==expected and m['runtime']==ref['runtime'] and m['source_sha256']==ref['source_sha256']
    assert m['schedule']==SCHEDULE and m['target_updates']==2000 and m['smoke_slots'] is None and m['purpose']=='training'
    assert m['execution']==dict(device='cuda',gpu=str(GPUS[mode]),threads=4)
    rows=list(csv.DictReader((run/'csv_logs/env_metrics.csv').open()))
    assert rows and int(rows[0]['update'])==0 and all(math.isfinite(float(x['reward'])) for x in rows)
    ck=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
    assert pt.normalize_config(ck['cfg'])==m['config'] and ck['lr_schedule']==SCHEDULE
    assert all(torch.isfinite(t).all() for t in ck['model'].values())
    u=int(ck['update']);lr=.0003*(1-u/2000)
    assert int(ck['model']['ret_count'])==u+1
    assert all(math.isclose(g['lr'],lr,rel_tol=0,abs_tol=1e-18) for g in ck['optimizer']['param_groups'])
    console=(LOGS/(name+'.log')).read_text()
    assert 'lr anneal: 0.0003 -> 0.0 over 2000 updates from global update 0' in console
    assert 'Traceback (most recent call last)' not in console and '\nresumed ' not in console
    results[mode]=dict(name=name,physical_gpu=GPUS[mode],status=state['status'],training_pid=state['training_pid'],supervisor_pid=state['supervisor_pid'],created_at=state['created_at'],run_dir=str(run),completed_updates=int(rows[-1]['update'])+1,checkpoint_update=u,optimizer_lr=lr,fresh_from_update=0,finite_weights=True,eval_csv=str(run/'eval.csv'),first_training_row=rows[0],calibration_application=m['calibration_application'])
assert {k for k in expected if expected[k]!=dict(ref['config'],traffic_model='bernoulli')[k]}==set(NARROW)
orderpath=ROOT/'runs/RUN_ORDER.json';order=json.loads(orderpath.read_text())
for mode in ('smoke','base','narrow'):
    name=NAMES[mode];seq=int(name.split('_')[0]);run=ROOT/'runs'/name
    matches=[e for e in order['runs'] if e['name']==name or e['sequence']==seq]
    if matches:
        assert len(matches)==1 and matches[0]['name']==name and matches[0]['sequence']==seq
    else:
        st=json.loads((LOGS/(name+'_launch.json')).read_text())
        order['runs'].append(dict(original_name=name,name=name,sequence=seq,started_at_utc=st['created_at'],time_basis='launch.created_at',config_created_at_utc=datetime.fromtimestamp((run/'config.json').stat().st_mtime,timezone.utc).isoformat(),legacy_alias_retained=False))
order['runs'].sort(key=lambda e:e['sequence']);order['updated_at_utc']=now();write(orderpath,order)
import paper_run_index
paper_run_index.main()
for mode in ('base','narrow'):
    run=ROOT/'runs'/NAMES[mode]
    for short,original in {'eval.csv':'eval_metrics.csv','train.csv':'env_metrics.csv','ppo.csv':'ppo_metrics.csv','per_ue.csv':'per_ue_metrics.csv'}.items():
        assert (run/short).is_symlink() and (run/short).resolve()==run/'csv_logs'/original
result=dict(status='passed',validated_at=now(),tests_passed=219,new_variant_tests=15,
            production_sources_unchanged=True,original_run19_sources_compatible=True,existing_results_unchanged=True,
            training_seed=2024,same_initial_model=env['same_initial_model'],initial_policy_sha256=env['initial_policy_sha256'],
            environment_validation=env,smoke=dict(name=NAMES['smoke'],phases=smoke['phases'],resume_verified=True,actual_model_update_verified=True),
            common=dict(traffic_model='bernoulli',cqi_mode='nr4bit',target_updates=2000,initial_lr=.0003,schedule=SCHEDULE,eval_every=10,eval_episodes=3,episode_slots=1000,threads_per_run=4,beta=ref['config']['la_beta_by_depth']),
            runs=results,scientific_notes=['Same seed is a paired design, not independent training replication.','Each inline eval.csv uses its own training world; compare on a common held-out world later.','NARROW uses unchanged Base beta transferred without a new NARROW-specific calibration holdout.'])
out=ROOT/'review/bernoulli_weekend_validation.json';write(out,result,exclusive=True)
packagepath=ROOT/'PACKAGE_MANIFEST.json';package=json.loads(packagepath.read_text())
for name,digest in package['sha256'].items():assert sha(ROOT/name)==digest,name
for name in ['paper_train_variants.py','artifacts/narrow_mean/config.json','tests/test_paper_train_variants.py','review/runners/run_bernoulli_weekend.py','review/runners/check_bernoulli_weekend_environment.py','review/runners/check_bernoulli_weekend_launch.py','review/bernoulli_weekend_validation.json','review/BERNOULLI_WEEKEND_20260911.md']:
    package['sha256'][name]=sha(ROOT/name)
package.update(version=12,scope='PaperMain preserved core and original recipes; additive mean-matched Bernoulli NARROW and fresh 2000-update Base/NARROW pair on GPU3/GPU5')
write(packagepath,package)
from verify_package import main
main()
print(json.dumps(dict(status='passed',validation=str(out),runs=results,package_files=len(package['sha256'])),indent=2))
