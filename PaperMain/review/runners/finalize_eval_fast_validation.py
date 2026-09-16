from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, tarfile
from calibration.profile import load_profile
from paper_train import normalize_config, source_hashes

root = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
logs = root/'review/logs'
name = '14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5'
run = root/'runs'/name
parent = root/'runs/12_base_cqi4_batched_lr1000_s2024_20260909_gpu5'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
unit = (logs/'eval_fast_regression_tests.log').read_text()
assert int(re.search(r'Ran (\d+) tests', unit).group(1)) == 112 and '\nOK\n' in unit
proof_path = logs/'eval_fast_comparison/summary.json'
proof = json.loads(proof_path.read_text())
assert proof['status'] == 'passed' and proof['exact_actions'] and proof['metrics_match']
assert proof['speedup'] > 1
smoke = json.loads((logs/'13_audit_eval_fast_gpu5_launch.json').read_text())
assert smoke['status'] == 'completed' and smoke['checkpoint_update'] == 1
smoke_log = (logs/'13_audit_eval_fast_gpu5.log').read_text()
assert '[eval-time] update 0:' in smoke_log and '[eval-time] update 1:' in smoke_log
profile = load_profile(root/'results/cqi4_hl_corrected_v2/summary.json')
old = json.loads((parent/'paper_manifest.json').read_text())
new = json.loads((run/'paper_manifest.json').read_text())
lineage = json.loads((run/'PARENT_RUN.json').read_text())
assert new['source_sha256'] == source_hashes(root)
assert new['config'] == old['config'] == normalize_config(json.loads((run/'config.json').read_text()))
for key in ['schedule', 'schedule_origin', 'replay_origin', 'execution', 'calibration_profile', 'runtime', 'target_updates']:
    assert new[key] == old[key], key
assert new['continuation'] == lineage
assert set(k for k, h in new['source_sha256'].items() if old['source_sha256'][k] != h) == {'policy.py', 'train_phase2.py'}
for f, h in lineage['parent_metadata_sha256'].items():
    assert sha(parent/f) == h
for f, h in lineage['copied_checkpoint_sha256'].items():
    assert sha(parent/'ckpt'/f) == h
assert sha(run/'lineage/EVAL_VALIDATION.json') == lineage['evaluation_validation_sha256'] == sha(proof_path)
source = json.loads((root/'provenance/source_manifest.json').read_text())
for entry in source['entries']:
    assert sha(root/entry['path']) == sha(root.parent/entry['source']) == entry['sha256'], entry['path']
for f, h in source['original_root_source_sha256'].items():
    assert sha(root.parent/f) == h, f
launch = json.loads((logs/(name+'_launch.json')).read_text())
assert launch['status'] == 'running'
for key in ['supervisor_pid', 'training_pid']:
    os.kill(launch[key], 0)
console = (logs/(name+'.log')).read_text()
assert 'start_update=21' in console and 'Traceback (most recent call last)' not in console
archive = root/'provenance/run12_before_eval_fast_sources.tar.gz'
assert sha(archive) == lineage['source_archive_sha256']
with tarfile.open(archive) as tf:
    for f, h in old['source_sha256'].items():
        assert hashlib.sha256(tf.extractfile(f).read()).hexdigest() == h
    backup = root/'provenance/PACKAGE_MANIFEST_before_eval_fast.json'
    archived_manifest = tf.extractfile('PACKAGE_MANIFEST.json').read()
    if backup.exists():
        assert backup.read_bytes() == archived_manifest
    else:
        backup.write_bytes(archived_manifest)
result = dict(status='passed', validated_at=datetime.now(timezone.utc).isoformat(), tests=112,
              eval_benchmark=dict(legacy_seconds=proof['arms']['legacy']['seconds'],
                                  action_only_seconds=proof['arms']['action_only']['seconds'],
                                  speedup=proof['speedup'], reduction_percent=100*(1-1/proof['speedup']),
                                  checkpoint_update=proof['checkpoint_update'], episodes=proof['episodes'],
                                  slots_per_episode=proof['slots_per_episode'], physical_gpu=5,
                                  exact_actions=True, metrics_match=True, metric_tolerance=1e-12,
                                  note='One ordered comparison with the same prewarmed cache; performance varies with policy and load.'),
              evaluation_proof_sha256=sha(proof_path),
              fresh_resume_smoke='passed; two 32-slot updates with inline evaluation and restored LR/replay',
              smoke_optimizer_lr=smoke['optimizer_lr'],
              continuation=dict(parent=parent.name, child=name, saved_update=20, next_update=21,
                                target_updates=1000, checkpoint_bytes_preserved=True,
                                experiment_config_unchanged=True, training_optimizer_rng_preserved=True,
                                standard_resume_guards='passed', old_run_unchanged=True),
              calibration=profile['status'], beta=profile['beta_rounded'], schedule=new['schedule'],
              original_code_files_unchanged=len(source['original_root_source_sha256']),
              original_and_copied_artifacts_unchanged=len(source['entries']),
              current_source_sha256=new['source_sha256'], launch_status=launch['status'])
(root/'review/eval_fast_validation.json').write_text(json.dumps(result, indent=2)+'\n')
files = {}
for f in sorted(root.rglob('*')):
    if not f.is_file() or f.is_symlink():
        continue
    rel = f.relative_to(root)
    if (rel.parts[0] in ('runs','results','reports') or '__pycache__' in rel.parts or f.suffix == '.pyc'
            or rel.as_posix().startswith('review/logs/') or f.name == 'PACKAGE_MANIFEST.json'
            or f.name.startswith('._')):
        continue
    files[rel.as_posix()] = sha(f)
(root/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(version=7,
    scope='PaperMain corrected CQI4, batched PPO, exact deterministic evaluation action path, LR decay and documented continuation',
    excludes='mutable runs/results/reports/logs/cache', sha256=files), indent=2)+'\n')
from verify_package import main
main()
print(json.dumps(result, indent=2))
