from pathlib import Path
from datetime import datetime, timezone
import copy, csv, hashlib, json, os, shutil, sys, tarfile

root = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
smoke = '--smoke' in sys.argv
parent_name = ('13_audit_eval_fast_gpu5' if smoke else
               '14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5')
name = ('15_audit_lrrestart_gpu5' if smoke else
        '16_base_cqi4_lrrestart2000_s2024_20260910_gpu5')
expected_update, target = (1, 3) if smoke else (999, 2000)
parent, child = root/'runs'/parent_name, root/'runs'/name
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from paper_train import source_hashes, normalize_config, canonical, validate_schedule
import torch
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = json.loads((parent/'paper_manifest.json').read_text())
new_hashes = source_hashes(root)
assert set(old['source_sha256']) == set(new_hashes)
changed = sorted(f for f,h in new_hashes.items() if old['source_sha256'][f] != h)
assert changed == ['paper_train.py', 'train_phase2.py'], changed
archive = root/'provenance/run14_before_lrrestart_sources.tar.gz'
assert sha(archive) == '1adeb73e4b418657efdcea1d01bfa927f62d91edbafa51d37d1368232b7774dd'
with tarfile.open(archive) as tf:
    for f,h in old['source_sha256'].items():
        assert hashlib.sha256(tf.extractfile(f).read()).hexdigest() == h, f
    if not smoke:
        assert tf.extractfile('run14_paper_manifest.json').read() == (parent/'paper_manifest.json').read_bytes()
launch = json.loads((parent/'launch.json').read_text())
assert launch['status'] == 'completed' and launch['checkpoint_update'] == expected_update
if not smoke:
    assert launch['target_completed'] and old['target_updates'] == 1000
    proof = json.loads((root/'review/logs/15_audit_lrrestart_gpu5_launch.json').read_text())
    assert proof['status'] == 'completed' and proof['checkpoint_update'] == 3
    assert proof['lr_segment_restored'] and proof['actual_model_update_verified']
checkpoint_hashes = {f:sha(parent/'ckpt'/f) for f in ['latest.pt', 'best.pt']}
if not smoke:
    assert checkpoint_hashes == {
        'latest.pt':'c7335c3cedef947ddae2ed3caf36c9835730b62d6a544c964f497adc83e7f45b',
        'best.pt':'4544298a2344cef2e4e9348f6e6011c7ac78b3a98d9fac8e9e3af8e1922d076d'}
ck = torch.load(parent/'ckpt/latest.pt', map_location='cpu', weights_only=False)
best = torch.load(parent/'ckpt/best.pt', map_location='cpu', weights_only=False)
assert ck['update'] == expected_update and best['update'] <= expected_update
assert 'lr_schedule' not in ck, 'This migration handles a legacy checkpoint without a schedule field only'
assert ck['best_eval_reward'] == best['eval_reward']
assert all(torch.isfinite(t).all() for t in ck['model'].values())
assert old['execution'] == {'device':'cuda','gpu':'5','threads':4}
assert old['config']['seed'] == 2024 and old['config']['ppo_batched_replay'] is True
assert old['config']['episode_len_main'] == (32 if smoke else 1000)
assert old['purpose'] == ('execution_smoke' if smoke else 'training')
raw_cfg = json.loads((parent/'config.json').read_text())
assert normalize_config(raw_cfg) == normalize_config(ck['cfg']) == old['config']
cfg = copy.deepcopy(raw_cfg)
cfg['git_hash'] = 'sha256:' + hashlib.sha256(canonical(new_hashes).encode()).hexdigest()
cfg['git_dirty_py'] = False
assert normalize_config(cfg) == old['config']
schedule = dict(lr_final=0.0, lr_decay_updates=1000,
                lr_initial=1e-5, lr_start_update=expected_update+1)
validate_schedule(schedule, cfg['ppo_learning_rate'])
csv_data = {}
for p in sorted((parent/'csv_logs').glob('*.csv')):
    with p.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert 'update' in reader.fieldnames
        rows = list(reader)
        kept = [r for r in rows if int(r['update']) <= expected_update]
        csv_data[p.name] = (reader.fieldnames, kept, len(rows), sha(p))
metadata_hashes = {f:sha(parent/f) for f in ['paper_manifest.json','config.json']}
assert not child.exists()
child.mkdir()
for d in ['ckpt','lineage','csv_logs']:
    (child/d).mkdir()
for f,h in checkpoint_hashes.items():
    shutil.copy2(parent/'ckpt'/f, child/'ckpt'/f)
    assert sha(child/'ckpt'/f) == h
shutil.copy2(parent/'paper_manifest.json', child/'lineage/PARENT_MANIFEST.json')
shutil.copy2(parent/'config.json', child/'lineage/PARENT_CONFIG.json')
(child/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
csv_history = {}
for filename,(fields,rows,total,parent_hash) in csv_data.items():
    p = child/'csv_logs'/filename
    with p.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    csv_history[filename] = dict(parent_sha256=parent_hash,initial_child_sha256=sha(p),
                                inherited_rows=len(rows),dropped_rows=total-len(rows),
                                inherited_through_update=expected_update)
lineage = dict(created_at=datetime.now(timezone.utc).isoformat(),parent_run=parent_name,
               reason='Explicit low-LR continuation from the last completed checkpoint; full model, critic, optimizer and RNG retained.',
               resumed_checkpoint_update=expected_update,next_update=expected_update+1,
               target_updates=target,previous_schedule=old['schedule'],continuation_schedule=schedule,
               changed_sources=changed,parent_source_sha256=old['source_sha256'],
               source_archive=str(archive.relative_to(root)),source_archive_sha256=sha(archive),
               parent_metadata_sha256=metadata_hashes,copied_checkpoint_sha256=checkpoint_hashes,
               parent_optimizer_lr=ck['optimizer']['param_groups'][0]['lr'],
               parent_best_update=best['update'],parent_best_reward=ck['best_eval_reward'],
               initial_child_config_sha256=sha(child/'config.json'),csv_history=csv_history,
               parent_checkpoint_schedule_field='absent; legacy payload, schedule retained in parent manifest',
               tensorboard=f'Child events begin at update {expected_update+1}; earlier events remain in parent run.',
               purpose='execution_smoke' if smoke else 'training')
new = copy.deepcopy(old)
new.update(source_sha256=new_hashes,schedule=schedule,schedule_origin='override',
           target_updates=target,continuation=lineage)
(child/'paper_manifest.json').write_text(json.dumps(new,indent=2)+'\n')
(child/'PARENT_RUN.json').write_text(json.dumps(lineage,indent=2)+'\n')
for f,h in checkpoint_hashes.items():assert sha(parent/'ckpt'/f)==h
for f,h in metadata_hashes.items():assert sha(parent/f)==h
print(json.dumps(dict(prepared=True,run_dir=str(child),**lineage),indent=2))
