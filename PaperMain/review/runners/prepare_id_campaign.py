from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,re,shutil,sys
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain');sys.path.insert(0,str(ROOT))
CAMPAIGN=ROOT/'results/ftp3_id_run18_best_u0639_s2024_20260911'
RUN=ROOT/'runs/18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not CAMPAIGN.exists()
manifest=json.loads((RUN/'paper_manifest.json').read_text())
from paper_train import source_hashes,calibration_reference
from calibration.profile import validate_profile_record
assert source_hashes(ROOT)==manifest['source_sha256']
validate_profile_record(manifest['calibration_profile'],calibration_reference(ROOT,manifest['calibration_reference_root']))
state=json.loads((ROOT/'review/logs'/f'{RUN.name}_launch.json').read_text());assert state['status']=='completed' and state['checkpoint_update']==999
records=[]
# Audit explicit historical evaluation IDs from both the package and original repository.
# Legacy fields called seed are conservatively treated as episode IDs too.
skip={'.git','__pycache__','.venv','venv','node_modules','site-packages','.mypy_cache','.pytest_cache'}
text_hits=[];csv_files=0;json_files=0;max_seen=-1
for dirname,dirs,files in os.walk(ROOT.parent):
    dirs[:]=[d for d in dirs if d not in skip and not d.startswith('.cache')]
    for filename in files:
        path=Path(dirname)/filename
        if path.is_symlink() or filename.startswith('._'):continue
        if path.suffix=='.csv':
            try:
                with path.open() as f:
                    reader=csv.DictReader(f);headers=reader.fieldnames or []
                    key=next((x for x in ('episode_idx','episode','ep_idx','eval_seed','seed') if x in headers),None)
                    if key is None:continue
                    ids=[]
                    for row in reader:
                        value=row.get(key,'')
                        if value not in ('',None):
                            number=float(value)
                            if number.is_integer():ids.append(int(number))
                if ids:
                    csv_files+=1;low,high=min(ids),max(ids);max_seen=max(max_seen,high)
                    assert not any(119000<=v<119002 or 120000<=v<120100 or 121024<=v<121026 or 122024<=v<122124 for v in ids),str(path)
                    records.append(dict(path=str(path.relative_to(ROOT.parent)),column=key,min=low,max=high,unique=len(set(ids))))
            except (UnicodeDecodeError,ValueError,csv.Error):continue
        elif path.name in ('manifest.json','paper_manifest.json','summary.json'):
            try:d=json.loads(path.read_text())
            except (ValueError,UnicodeDecodeError):continue
            json_files+=1
            def visit(x):
                if isinstance(x,dict):
                    start=x.get('episode_start',x.get('start'));count=x.get('episodes')
                    if type(start)is int and type(count)is int and count>0:
                        for a,b in [(119000,119002),(120000,120100),(121024,121026),(122024,122124)]:
                            assert max(start,a)>=min(start+count,b),(str(path),start,count)
                    for v in x.values():visit(v)
                elif isinstance(x,list):
                    for v in x:visit(v)
            visit(d)
# All training/inline-validation/calibration channels are far below the new band.
for path in (ROOT/'runs').glob('*/paper_manifest.json'):
    if path.parent.is_symlink():continue
    m=json.loads(path.read_text());cfg=m['config'];seed=cfg['seed']
    for lo,hi in [(seed,seed+m['target_updates']),(seed+10000,seed+10000+cfg['ppo_eval_episodes'])]:
        assert max(lo,121024)>=min(hi,121026) and max(lo,122024)>=min(hi,122124)
inputs=CAMPAIGN/'inputs';inputs.mkdir(parents=True)
for src,name in [(RUN/'ckpt/best.pt','best.pt'),(RUN/'paper_manifest.json','source_manifest.json'),(RUN/'config.json','source_config.json')]:
    shutil.copy2(src,inputs/name);assert sha(src)==sha(inputs/name)
assert sha(inputs/'best.pt')=='9afca2081ee8a5db08fe40460528e253456974b2d53abd39ca1cf0ace30ce329'
audit=dict(status='validated',audited_at=datetime.now(timezone.utc).isoformat(),scope=str(ROOT.parent),allowed_episode_ranges=[dict(start=119000,episodes=2,purpose='smoke'),dict(start=120000,episodes=100,purpose='id')],world_seed=2024,channel_seed_ranges=[[121024,121026],[122024,122124]],historical_csv_files=csv_files,historical_json_files=json_files,historical_csv_episode_max=max_seen,observed_bands=records,interpretation='No matching IDs found in available repository evaluation CSVs and structured manifests; training, inline validation and calibration bands excluded. Smoke is separate from the frozen main band.')
(inputs/'episode_history_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
plan=dict(created_at=datetime.now(timezone.utc).isoformat(),world='ID',traffic='ftp3',run=RUN.name,checkpoint='best',checkpoint_update=639,checkpoint_sha256=sha(inputs/'best.pt'),selection='maximum existing three-episode validation reward; fixed before new ID tests',episodes=100,episode_start=120000,slots=1000,world_config=manifest['config'],source_sha256=manifest['source_sha256'],schedulers=['ppo','sus_cqi','sus_cqi_m2','sus_cqi_m3','sus_rps','pf_greedy_sds','sumrate_greedy_sds'],shards=[dict(index=0,gpu=3,episode_ids=list(range(120000,120100,2))),dict(index=1,gpu=5,episode_ids=list(range(120001,120100,2)))],smoke_episode_ids=[119000,119001],training_run19='untouched on GPU4',statistical_unit='episode, 100 paired IDs; one training seed',history_audit_sha256=sha(inputs/'episode_history_audit.json'))
(inputs/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'campaign':str(CAMPAIGN),'checkpoint_sha256':plan['checkpoint_sha256'],'audited_csv_files':csv_files,'audited_json_files':json_files,'max_historical_csv_episode':max_seen,'new_band':[120000,120100]},indent=2))
