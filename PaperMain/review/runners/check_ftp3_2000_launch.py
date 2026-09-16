from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math,os,sys
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review/runners'))
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch
torch.set_num_threads(1)
from paper_train import normalize_config,source_hashes,validate_checkpoint_schedule
from run_ftp3_2000_gpu4 import NAME,PARENT,BERNOULLI,SCHEDULE,preserved_results,write_json,now
run=ROOT/'runs'/NAME;logs=ROOT/'review/logs'
state=json.loads((logs/(NAME+'_launch.json')).read_text());manifest=json.loads((run/'paper_manifest.json').read_text())
parent=json.loads((ROOT/'runs'/PARENT/'paper_manifest.json').read_text())
assert state['status']=='running';os.kill(state['training_pid'],0)
assert manifest['config']==parent['config']==normalize_config(manifest['config'])
assert manifest['source_sha256']==parent['source_sha256']==source_hashes(ROOT)
assert manifest['runtime']==parent['runtime']
assert manifest['schedule']==SCHEDULE and manifest['target_updates']==2000 and manifest['smoke_slots'] is None
assert manifest['execution']==dict(device='cuda',gpu='4',threads=4)
assert state['resume'] is None and state['training_initialization'].startswith('fresh model')
assert preserved_results()==state['previous_results_sha256']
with (run/'csv_logs/env_metrics.csv').open() as f:rows=list(csv.DictReader(f))
assert rows and int(rows[0]['update'])==0 and all(math.isfinite(float(x['reward'])) for x in rows)
console=(logs/(NAME+'.log')).read_text()
assert 'lr anneal: 0.0003 -> 0.0 over 2000 updates from global update 0' in console
assert 'Traceback (most recent call last)' not in console and '\nresumed ' not in console
ck=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
assert normalize_config(ck['cfg'])==manifest['config'] and ck['lr_schedule']==SCHEDULE
validate_checkpoint_schedule(ck,SCHEDULE)
assert all(torch.isfinite(t).all() for t in ck['model'].values())
expected_lr=.0003*(1-int(ck['update'])/2000)
assert all(math.isclose(g['lr'],expected_lr,rel_tol=0,abs_tol=1e-18) for g in ck['optimizer']['param_groups'])
result=dict(status='passed',validated_at=now(),name=NAME,run_dir=str(run),physical_gpu=4,training_pid=state['training_pid'],supervisor_pid=state['supervisor_pid'],fresh_from_update=0,target_updates=2000,completed_updates=int(rows[-1]['update'])+1,checkpoint_update=int(ck['update']),current_optimizer_lr=ck['optimizer']['param_groups'][0]['lr'],checkpoint_finite=True,initial_lr=.0003,schedule=SCHEDULE,lr_at_update={'0':.0003,'500':.000225,'1000':.00015,'1500':.000075,'1999':1.5e-7,'2000':0},comparison_run=PARENT,config_unchanged=True,production_sources_unchanged=True,runtime_unchanged=True,previous_results_unchanged=True,previous_results_sha256=state['previous_results_sha256'],first_training_row=rows[0],eval_csv=str(run/'eval.csv'),eval_every=manifest['config']['ppo_eval_every'],eval_episodes=manifest['config']['ppo_eval_episodes'],seed=2024)
# Extend package evidence only after confirming every existing tracked hash.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
package_path=ROOT/'PACKAGE_MANIFEST.json';package=json.loads(package_path.read_text())
for name,digest in package['sha256'].items():assert sha(ROOT/name)==digest,name
out=ROOT/'review/ftp3_lr2000_validation.json';write_json(out,result,exclusive=True)
for name in ['review/runners/run_ftp3_2000_gpu4.py','review/runners/check_ftp3_2000_launch.py','review/ftp3_lr2000_validation.json']:
    package['sha256'][name]=sha(ROOT/name)
package.update(version=10,scope='PaperMain optional FTP3 Poisson arrivals with preserved Bernoulli; fresh 2000-update FTP3 linear-LR run on GPU4')
write_json(package_path,package)
from verify_package import main
main()
print(json.dumps(result,indent=2))
