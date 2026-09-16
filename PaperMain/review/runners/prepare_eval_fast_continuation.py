from pathlib import Path
from datetime import datetime, timezone
import copy, csv, hashlib, json, os, shutil, tarfile

root = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
from paper_train import source_hashes, normalize_config, canonical
import torch
torch.set_num_threads(1)

parent_name = '12_base_cqi4_batched_lr1000_s2024_20260909_gpu5'
name = '14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5'
parent, child = root/'runs'/parent_name, root/'runs'/name
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path = parent/'paper_manifest.json'
old = json.loads(manifest_path.read_text())
new_hashes = source_hashes(root)
assert set(old['source_sha256']) == set(new_hashes)
changed = sorted(k for k, v in new_hashes.items() if old['source_sha256'][k] != v)
assert changed == ['policy.py', 'train_phase2.py'], changed
archive = root/'provenance/run12_before_eval_fast_sources.tar.gz'
assert sha(archive) == '85be89c9e351c9449c8d5dcb6e68920b78500f9cc11c7f70a48cbd5b34ebefb4'
with tarfile.open(archive) as tf:
    for f, h in old['source_sha256'].items():
        assert hashlib.sha256(tf.extractfile(f).read()).hexdigest() == h
    assert tf.extractfile('run12_paper_manifest.json').read() == manifest_path.read_bytes()
proof_path = root/'review/logs/eval_fast_comparison/summary.json'
proof = json.loads(proof_path.read_text())
assert proof['status'] == 'passed' and proof['exact_actions'] and proof['metrics_match']
assert proof['speedup'] > 1 and proof['episodes'] == [10000, 10001, 10002]
assert proof['slots_per_episode'] == 1000 and proof['physical_gpu'] == 5
assert proof['source_sha256'] == {f: new_hashes[f] for f in changed}
smoke = json.loads((root/'review/logs/13_audit_eval_fast_gpu5_launch.json').read_text())
assert smoke['status'] == 'completed' and smoke['checkpoint_update'] == 1
stopped = json.loads((root/'review/logs'/f'{parent_name}_launch.json').read_text())
assert stopped['status'] == 'stopped_for_eval_acceleration'
for key in ['training_pid', 'supervisor_pid']:
    try:
        os.kill(stopped[key], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError(f'Parent process is still alive: {key}')
ck = torch.load(parent/'ckpt/latest.pt', map_location='cpu', weights_only=False)
assert ck['update'] == 20
assert old['target_updates'] == 1000 and old['purpose'] == 'training'
assert old['execution'] == {'device': 'cuda', 'gpu': '5', 'threads': 4}
assert old['schedule'] == {'lr_final': 0.0, 'lr_decay_updates': 1000}
old_cfg = json.loads((parent/'config.json').read_text())
assert normalize_config(old_cfg) == normalize_config(ck['cfg']) == old['config']
cfg = copy.deepcopy(old_cfg)
cfg['git_hash'] = 'sha256:' + hashlib.sha256(canonical(new_hashes).encode()).hexdigest()
cfg['git_dirty_py'] = False
assert normalize_config(cfg) == old['config']
checkpoint_hashes = {f: sha(parent/'ckpt'/f) for f in ['latest.pt', 'best.pt']}
assert proof['checkpoint_sha256'] == checkpoint_hashes['best.pt']
parent_hashes = {'paper_manifest.json': sha(manifest_path), 'config.json': sha(parent/'config.json')}
csv_data = {}
for p in sorted((parent/'csv_logs').glob('*.csv')):
    with p.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert 'update' in reader.fieldnames, p
        rows = list(reader)
        kept = [r for r in rows if int(r['update']) <= ck['update']]
        csv_data[p.name] = (reader.fieldnames, kept, len(rows), sha(p))
assert not child.exists()
child.mkdir()
(child/'ckpt').mkdir()
(child/'lineage').mkdir()
(child/'csv_logs').mkdir()
for f, h in checkpoint_hashes.items():
    shutil.copy2(parent/'ckpt'/f, child/'ckpt'/f)
    assert sha(child/'ckpt'/f) == h
shutil.copy2(manifest_path, child/'lineage/PARENT_MANIFEST.json')
shutil.copy2(parent/'config.json', child/'lineage/PARENT_CONFIG.json')
shutil.copy2(proof_path, child/'lineage/EVAL_VALIDATION.json')
(child/'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
csv_lineage = {}
for filename, (fields, rows, original_count, original_hash) in csv_data.items():
    p = child/'csv_logs'/filename
    with p.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_lineage[filename] = dict(parent_sha256=original_hash, initial_child_sha256=sha(p),
                                inherited_rows=len(rows), dropped_rows=original_count-len(rows),
                                inherited_through_update=20)
lineage = dict(created_at=datetime.now(timezone.utc).isoformat(), parent_run=parent_name,
               resumed_checkpoint_update=20, next_update=21, target_updates=1000,
               reason='Evaluation-only action path acceleration; unchanged training algorithm and configuration.',
               changed_sources=changed, parent_source_sha256=old['source_sha256'],
               source_archive=str(archive.relative_to(root)), source_archive_sha256=sha(archive),
               parent_metadata_sha256=parent_hashes, copied_checkpoint_sha256=checkpoint_hashes,
               initial_child_config_sha256=sha(child/'config.json'), csv_history=csv_lineage,
               validation_checkpoint_update=proof['checkpoint_update'],
               validation_checkpoint_sha256=proof['checkpoint_sha256'],
               evaluation_validation_sha256=sha(proof_path),
               tensorboard='Child events begin at update 21; earlier events remain in the parent run.')
new = copy.deepcopy(old)
new['source_sha256'] = new_hashes
new['continuation'] = lineage
(child/'paper_manifest.json').write_text(json.dumps(new, indent=2) + '\n')
(child/'PARENT_RUN.json').write_text(json.dumps(lineage, indent=2) + '\n')
for f, h in checkpoint_hashes.items():
    assert sha(parent/'ckpt'/f) == h
for f, h in parent_hashes.items():
    assert sha(parent/f) == h
print(json.dumps(dict(prepared=True, run_dir=str(child), **lineage), indent=2))
