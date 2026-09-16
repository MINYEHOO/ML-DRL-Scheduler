"""Verify the dispatched OOD campaign, expose per-run outputs, record provenance."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,sys
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain');sys.path.insert(0,str(ROOT))
import paper_train as pt
from paper_ood_campaign import C,RUNS,WORKERS,hashes,WORLDS

def main():
    smoke=json.loads((C/'smoke_launch.json').read_text())
    state=json.loads((C/'main_launch.json').read_text())
    if smoke['status']!='completed' or state['status'] not in ('running','completed'):
        raise ValueError('Campaign has not passed smoke and started main')
    if smoke['hashes']!=state['hashes'] or state['hashes']!=hashes():raise ValueError('Inputs changed after smoke')
    if smoke['cross_gpu_equivalence']['status']!='passed':raise ValueError('Cross-GPU equivalence missing')
    summary=json.loads((C/'smoke/combined/summary.json').read_text())
    if summary['rows']!=90 or summary['worlds']!=10:raise ValueError('Smoke did not cover complete grid')
    log=(ROOT/'review/logs/ood_regression_tests_20260913.log').read_text()
    tests=int(re.search(r'Ran (\d+) tests',log).group(1))
    if tests!=273 or '\nOK\n' not in log:raise ValueError('Regression proof missing')
    package_path=ROOT/'PACKAGE_MANIFEST.json';package=json.loads(package_path.read_text())
    for name,digest in package['sha256'].items():
        if pt.sha_file(ROOT/name)!=digest:raise ValueError(f'Previously packaged file changed: {name}')
    source=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    for item in source['entries']:
        if pt.sha_file(ROOT/item['path'])!=item['sha256'] or pt.sha_file(ROOT.parent/item['source'])!=item['sha256']:
            raise ValueError('Original archived evidence changed')
    for name,digest in source['original_root_source_sha256'].items():
        if pt.sha_file(ROOT.parent/name)!=digest:raise ValueError('Original root source changed')
    links=[]
    for gpu,n,_ in WORKERS:
        directory=ROOT/'runs'/RUNS[n]/'ood_20260913'
        if directory.exists():raise ValueError('Per-run OOD links already exist')
        directory.mkdir()
        targets={'eval.csv':C/f'evaluation/run{n}/metrics.csv','progress.json':C/f'evaluation/run{n}/progress.json',
                 'manifest.json':C/f'evaluation/run{n}/manifest.json','console.log':C/f'eval_gpu{gpu}.log',
                 'pilot_progress.json':C/f'pilot/gpu{gpu}/progress.json','pilot.csv':C/f'pilot/gpu{gpu}/metrics.csv',
                 'status.json':C/'main_launch.json','all_results':C,'summary.md':C/'combined/summary.md'}
        for label,target in targets.items():(directory/label).symlink_to(os.path.relpath(target,directory))
        (directory/'README.md').write_text('기존 9종 OOD + 공통 Base ID 평가입니다.\n\n'
            'eval.csv는 이 GPU에서 평가하는 PPO와 두 baseline의 결과이며 매 에피소드 후 갱신됩니다.\n'
            '본 평가 전에는 pilot.csv와 pilot_progress.json에서 비교군 설정 평가를 확인할 수 있습니다.\n'
            'status.json은 전체 진행 단계, all_results는 공통 결과, summary.md는 완료 후 통합 분석입니다.\n'
            '본 평가를 시작하기 전이나 최종 집계 전에는 해당 링크의 대상 파일이 아직 없을 수 있습니다.\n')
        links.append(dict(run=n,gpu=gpu,directory=str(directory),eval_csv=str(directory/'eval.csv')))
    result=dict(status='passed',validated_at=datetime.now(timezone.utc).isoformat(),campaign=str(C),
        current_phase=state['phase'],supervisor_pid=state['supervisor_pid'],workers=state['workers'],
        tests=tests,smoke_rows=90,smoke_worlds=10,cross_gpu_equivalence=smoke['cross_gpu_equivalence'],
        model_assignments=links,main_episodes_per_world=100,worlds=WORLDS,expected_rows=9000,
        checkpoint_preflight=json.loads((C/'inputs/checkpoint_preflight.json').read_text()),
        old_artifacts_unchanged=len(source['entries']),science_sources_unchanged=True,source_sha256=pt.source_hashes(ROOT))
    report=['# Bernoulli OOD: Run 16 / 21 / 22','',f'검증 시각: {result["validated_at"]}',
        f'현재 단계: {state["phase"]}. 전체 상태는 각 run의 `ood_20260913/status.json`에서 확인할 수 있습니다.','',
        '| GPU | 모델 | best update |','|---|---|---:|','| 3 | Run 16 — LR restart | 1989 |',
        '| 4 | Run 21 — Base fresh 2000 | 649 |','| 5 | Run 22 — mean-matched NARROW | 1469 |','',
        '공통 Base 환경과 기존 9종 OOD를 각각 100회 평가합니다. 학습 없이 best 정책을 고정합니다.',
        '기존 6 baseline은 GPU별로 2개씩 나눠 한 번만 계산하며, 모든 PPO와 같은 세계·에피소드끼리 비교합니다.','',
        '세계: ID, P055, P010, V60max(5–60 km/h), CSI02, D26strict, STORM2, K8, K48, K60.',
        '별도 pilot 8회 × threshold 5개에서 평균 reward로 SUS 설정을 선택합니다. Pilot 종료 후 본 평가와 통합 집계가 자동 실행됩니다.',
        'Pilot 에피소드 139000–139007, 본 평가 140000–140099. Smoke 138000–138001은 논문 결과에서 제외합니다.','',
        'D26에서는 환경 deadline만 2–6으로 바꾸고 정책의 /12 정규화를 유지합니다. K 실험은 사용자 수와 총 부하가 함께 바뀝니다.',
        '같은 시드는 초기 topology·속도·부하를 맞추지만, scheduler별 queue admission 때문에 실제 traffic/CSI 난수 흐름까지 동일하다고 주장하지 않습니다.','',
        f'검증: {tests}개 테스트, 10개 세계의 세 모델 smoke 90행, GPU 5↔3에서 Random 순서를 뒤집은 재평가 {smoke["cross_gpu_equivalence"]["rows"]}행 일치.',
        '평가 중 가중치·정규화 buffer·Torch RNG·입력 해시 보존 확인. 기존 학습 코드와 결과는 그대로 유지했습니다.','',
        '각 run의 `ood_20260913/eval.csv`는 매 에피소드 후 갱신됩니다. Pilot 단계에서는 `pilot.csv`를 볼 수 있습니다.','',
        f'전체 결과: `{C}`',f'완료 후 분석: `{C}/combined/summary.md`','',
        '전체 9,000행을 검증한 뒤 평균·paired Student-t 95% 구간을 집계합니다. 이는 한 training seed에서 고정한 모델의 평가 변동이며, 학습 seed 간 변동을 나타내지 않습니다.']
    pt.atomic_json(ROOT/'review/ood_launch_validation_20260913.json',result)
    (ROOT/'review/OOD_20260913.md').write_text('\n'.join(report)+'\n')
    newfiles=['paper_ood_inputs.py','paper_ood_eval.py','paper_ood_campaign.py','paper_ood_merge.py',
              'tests/test_paper_ood_inputs.py','tests/test_paper_ood_eval.py','tests/test_paper_ood_campaign.py','tests/test_paper_ood_merge.py',
              'review/runners/prepare_ood_campaign.py','review/runners/finalize_ood_launch.py',
              'review/ood-eval-inputs-stage.tar.gz','review/ood-eval-campaign-stage.tar.gz','review/ood-input-metadata-fix.tar.gz',
              'review/ood_launch_validation_20260913.json','review/OOD_20260913.md']
    for name in newfiles:package['sha256'][name]=pt.sha_file(ROOT/name)
    package.update(version=13,scope='PaperMain with frozen completed-run Bernoulli common-world OOD, independent SUS pilots, three-GPU evaluation and paired episode reporting')
    pt.atomic_json(package_path,package)
    from verify_package import main as verify
    verify()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
