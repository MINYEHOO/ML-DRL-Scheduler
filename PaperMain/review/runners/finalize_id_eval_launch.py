from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,re,sys
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain');sys.path.insert(0,str(ROOT))
from paper_train import source_hashes,atomic_json
C=ROOT/'results/ftp3_id_run18_best_u0639_s2024_20260911';R18=ROOT/'runs/18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4';R19=ROOT/'runs/19_base_cqi4_ftp3_lr2000_s2024_20260911_gpu4'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
smoke=json.loads((C/'smoke_launch.json').read_text());assert smoke['status']=='completed' and smoke['serial_equivalence']['max_abs_difference']==0
main=json.loads((C/'main_launch.json').read_text());assert main['status']=='running' and [w['gpu'] for w in main['workers']]==[3,5]
assert main['hashes']==smoke['hashes']
for w in main['workers']:os.kill(w['pid'],0)
train=json.loads((ROOT/'review/logs'/f'{R19.name}_launch.json').read_text());assert train['status']=='running';os.kill(train['training_pid'],0)
assert source_hashes(ROOT)==json.loads((R19/'paper_manifest.json').read_text())['source_sha256']
assert sha(R18/'ckpt/best.pt')==sha(C/'inputs/best.pt')==main['hashes']['checkpoint']
assert sha(R18/'paper_manifest.json')==sha(C/'inputs/source_manifest.json')
assert sha(R18/'config.json')==sha(C/'inputs/source_config.json')
rows=[];progress=[]
for i in range(2):
    m=json.loads((C/f'shard{i}'/'manifest.json').read_text());p=json.loads((C/f'shard{i}'/'progress.json').read_text())
    assert m['status']=='running' and m['protocol']['checkpoint_update']==639
    assert m['protocol']['environment_config']==m['protocol']['training_config']
    assert m['protocol']['diagnostic_only'] is False and m['execution']['gpu']==str([3,5][i])
    with (C/f'shard{i}'/'metrics.csv').open() as f:part=list(csv.DictReader(f))
    assert part and part[0]['scheduler']=='ppo' and int(part[0]['slots'])==1000
    rows.extend(part);progress.append(p)
assert len({(x['episode_idx'],x['scheduler']) for x in rows})==len(rows)
log=(ROOT/'review/logs/id_eval_regression_tests.log').read_text();tests=int(re.search(r'Ran (\d+) tests',log).group(1));assert tests==204 and '\nOK\n' in log
# Ensure every previously packaged immutable file still matches before adding new tools.
package_path=ROOT/'PACKAGE_MANIFEST.json';package=json.loads(package_path.read_text())
for name,digest in package['sha256'].items():assert sha(ROOT/name)==digest,name
source=json.loads((ROOT/'provenance/source_manifest.json').read_text())
for e in source['entries']:assert sha(ROOT/e['path'])==sha(ROOT.parent/e['source'])==e['sha256']
for name,digest in source['original_root_source_sha256'].items():assert sha(ROOT.parent/name)==digest
alias=R18/'id_eval_20260911';assert not alias.exists();alias.symlink_to(os.path.relpath(C,R18));assert alias.resolve()==C
result=dict(status='passed',validated_at=datetime.now(timezone.utc).isoformat(),campaign=str(C),evaluation='ID',traffic='ftp3',source_run=R18.name,checkpoint_update=639,checkpoint_sha256=main['hashes']['checkpoint'],episodes=100,episode_start=120000,schedulers=7,expected_rows=700,physical_gpus=[3,5],workers=main['workers'],supervisor_pid=main['supervisor_pid'],tests=tests,smoke=smoke['serial_equivalence'],startup_progress=progress,first_full_episode_ppo_completed_on_both_gpus=True,run19_training_unchanged=True,training_pid=train['training_pid'],original_artifacts_unchanged=len(source['entries']),automatic_merge=str(C/'combined/summary.md'),per_run_link=str(alias),source_sha256=source_hashes(ROOT))
new=ROOT/'review/id_evaluation_validation.json';assert not new.exists();atomic_json(new,result)
for name in ['paper_run_eval.py','paper_run_eval_merge.py','tests/test_paper_run_eval.py','tests/test_paper_run_eval_merge.py','review/runners/prepare_id_campaign.py','review/runners/launch_id_campaign.py','review/runners/finalize_id_eval_launch.py','review/id-eval-stage.tar.gz','review/id_evaluation_validation.json']:
    package['sha256'][name]=sha(ROOT/name)
package.update(version=11,scope='PaperMain optional FTP3 training and frozen-checkpoint ID evaluation with validated GPU3/GPU5 sharding and paired episode aggregation')
atomic_json(package_path,package)
from verify_package import main as verify
verify();print(json.dumps(result,indent=2))
