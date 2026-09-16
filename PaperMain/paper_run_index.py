"""Add live per-run log aliases and a run index without touching training code.

Run again after creating new runs. CSV aliases point at the same files the
training process writes; this script does not copy, truncate or move logs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARKER = '<!-- papermain-run-index:v1 -->'
CSV_FILES = {'eval.csv': 'eval_metrics.csv', 'train.csv': 'env_metrics.csv',
             'ppo.csv': 'ppo_metrics.csv', 'per_ue.csv': 'per_ue_metrics.csv'}


def managed_text(path, body, warnings):
    if path.is_symlink() or (path.exists() and
                            (not path.is_file() or not path.read_text().startswith(MARKER))):
        warnings.append(f'Preserved existing guide: {path}')
        return
    path.write_text(MARKER + '\n' + body)


def link_file(run, name, target, allowed_root, warnings):
    if not target.exists():
        return False
    if (target.is_symlink() or not target.is_file()
            or not target.resolve().is_relative_to(allowed_root.resolve())):
        warnings.append(f'Skipped unexpected target: {target}')
        return False
    alias = run / name
    if alias.is_symlink():
        if alias.resolve() == target.resolve():
            return True
        warnings.append(f'Preserved different existing link: {alias}')
        return False
    if alias.exists():
        warnings.append(f'Preserved existing file: {alias}')
        return False
    alias.symlink_to(os.path.relpath(target, start=run))
    assert alias.resolve() == target.resolve()
    return True


def main():
    runs = ROOT / 'runs'
    if runs.is_symlink() or not runs.is_dir():
        raise ValueError('Expected a real PaperMain/runs directory')
    logs = ROOT / 'review' / 'logs'
    if logs.is_symlink() or not logs.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Console log storage must stay inside PaperMain')
    warnings, entries = [], []
    order_path = runs / 'RUN_ORDER.json'
    numbering = (json.loads(order_path.read_text())['runs']
                 if order_path.is_file() else [])
    numbered = {entry['name']: entry for entry in numbering}
    for run in sorted(runs.iterdir()):
        if run.is_symlink() or not run.is_dir():
            continue
        manifest_path = run / 'paper_manifest.json'
        if not manifest_path.is_file() or manifest_path.is_symlink():
            continue
        manifest = json.loads(manifest_path.read_text())
        order = numbered.get(run.name, {})
        original_name = order.get('original_name', run.name)
        cfg = manifest['config']
        linked = {}
        for alias, original in CSV_FILES.items():
            linked[alias] = link_file(run, alias, run / 'csv_logs' / original,
                                      run, warnings)
        for alias, suffix in (('console.log', '.log'), ('launch.json', '_launch.json')):
            linked[alias] = link_file(run, alias, logs / (original_name + suffix),
                                      logs, warnings)
        interval = cfg['ppo_eval_every']
        guide = f'''# {run.name}

이 폴더의 바로가기는 실행 중인 원본 파일을 연결한다. CSV 복사본이 아니므로
학습 프로세스가 기록하는 내용이 그대로 반영된다.

| 파일 | 내용 | 기록 시점 |
|---|---|---|
| [eval.csv](eval.csv) | 고정 검증 episode의 성능 | {interval} updates마다 전체 평가 회차 완료 후 |
| [train.csv](train.csv) | 학습 episode의 reward·goodput·completion 등 | 각 update 완료 후 |
| [ppo.csv](ppo.csv) | PPO loss·KL·entropy 등 | 각 update 완료 후 |
| [per_ue.csv](per_ue.csv) | UE별 평가 지표 | 전체 평가 회차 완료 후 |
| [tb_logs/](tb_logs/) | TensorBoard 지표 (학습률 포함) | 각 이벤트 기록 시 |
| [ckpt/](ckpt/) | 최신·최고 평가 checkpoint | 설정된 저장/평가 주기 |

첫 평가는 {interval} updates를 완료한 뒤 시작하며 CSV의 update 값은
`{interval - 1}`이다 (0부터 세는 인덱스). 매 평가에서 {cfg['ppo_eval_episodes']}개
고정 episode를 사용한다. 새 run의 첫 평가에는 PPO와 baseline 평가가 함께
포함되어, 평가 전체가 끝나기 전에는 eval.csv가 헤더만 보일 수 있다.
학습 점수(train.csv)와 고정 검증 점수(eval.csv)는 구분해서 해석한다.

기본 CSV 저장 위치는 `csv_logs/`이다. 파일 내용을 이동하거나 덮어쓰지 않았다.
'''
        if linked['console.log']:
            guide += '\n[console.log](console.log)는 실행 중 출력이다. 일반 진행 줄은 5 updates마다 출력된다.\n'
        if linked['launch.json']:
            guide += '\n[launch.json](launch.json)에서 PID와 프로세스 종료 상태를 확인할 수 있다.\n'
        guide += '\n[paper_manifest.json](paper_manifest.json)에 전체 설정·LR 일정·보정 profile 출처(사용한 경우)가 저장되어 있다.\n'
        if order:
            guide += (f"\n실행 순서: {order['sequence']}번. 최초 실행 시각(UTC): "
                      f"`{order['started_at_utc']}`. 기존 이름: `{original_name}`.\n")
        for alias, available in linked.items():
            if not available:
                guide = guide.replace(f'[{alias}]({alias})', f'`{alias}` (아직 생성되지 않음)')
        managed_text(run / 'RUN_FILES.md', guide, warnings)
        entries.append(dict(name=run.name, purpose=manifest.get('purpose', 'unknown'),
                            sequence=order.get('sequence', float('inf')), linked=linked))
    entries.sort(key=lambda x: (x['sequence'], x['name']))
    index = ['# PaperMain run별 로그', '',
             'CSV 바로가기는 원본과 연결되어 계속 갱신된다. Run을 새로 만들면',
             '`python paper_run_index.py`를 다시 실행해 같은 구조로 정리한다.', '',
             '번호는 최초 실행 순서다. 이름 변경 내역과 시각 근거는',
             '[RUN_ORDER.json](RUN_ORDER.json)에 기록한다. 남겨 둔 기존 이름은',
             '과거 경로를 유지하는 호환용 바로가기이며, 아래 목록에는 한 번만 표시한다.', '',
             '| Run | 종류 | 평가 | 학습 | PPO | 콘솔 | 상태 |',
             '|---|---|---|---|---|---|---|']
    for e in entries:
        name = e['name']
        label = name.replace('|', '\\|').replace('\n', ' ')
        kind = '정식 학습' if e['purpose'] == 'training' else '실행 검사'
        links = [f'[{filename}]({name}/{filename})' if e['linked'][filename] else '—'
                 for filename in ('eval.csv', 'train.csv', 'ppo.csv', 'console.log', 'launch.json')]
        index.append(f'| [{label}]({name}/RUN_FILES.md) | {kind} | ' + ' | '.join(links) + ' |')
    managed_text(runs / 'RUN_INDEX.md', '\n'.join(index) + '\n', warnings)
    print(json.dumps(dict(runs=len(entries), aliases_available=sum(sum(e['linked'].values())
                     for e in entries), warnings=warnings,
                     organized_at=datetime.now(timezone.utc).isoformat()), indent=2))


if __name__ == '__main__':
    main()
