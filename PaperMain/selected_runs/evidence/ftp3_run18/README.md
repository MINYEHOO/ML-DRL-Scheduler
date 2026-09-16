# FTP3 run18 고정 checkpoint ID 평가

평가 대상은 완료된 18번 run의 best checkpoint(update 639)이다. 기존 학습 중
3개 validation episode의 최고 reward로 선택된 checkpoint를 새 ID 평가 전에
고정했다. GPU 4의 19번 학습은 그대로 진행한다.

학습과 동일한 FTP3·부하·이동속도·CSI·패킷·deadline·큐·CQI4 beta 설정을 사용한다.
이번 작업은 ID 평가이며 OOD 조건을 추가하지 않았다. 정책 가중치와 학습된
normalizer를 고정하고 deterministic action을 사용한다.

- 본 평가: episode 120000–120099, 각각 1000 slots.
- GPU 3: 짝수 episode 50개, `shard0/`.
- GPU 5: 홀수 episode 50개, `shard1/`.
- 스케줄러: PPO, SUS+CQI, SUS+CQI@m=2, SUS+CQI@m=3, SUS-RPS,
  PF-Greedy-SDS, SumRate-Greedy-SDS.
- RPS·Greedy-SDS는 본 시스템에 맞춘 기존 adapted baseline 구현이다.
- 합계 700개 episode×scheduler 행. 통계의 독립 단위는 100개 episode이다.

| 파일 | 내용 |
|---|---|
| `main_launch.json` | 두 worker PID·GPU·전체 작업 상태 |
| `shard0/progress.json`, `shard1/progress.json` | GPU별 진행 행 수 |
| `shard0/metrics.csv`, `shard1/metrics.csv` | 실시간 episode별 결과 |
| `main_gpu3.log`, `main_gpu5.log` | GPU별 실행 로그 |
| `combined/summary.md` | 두 작업 완료 후 자동 생성되는 결과 표 |
| `combined/summary.json` | 전체 지표·paired bootstrap 신뢰구간 |
| `combined/mergedmetrics.csv` | 누락·중복 검사 후 합친 700개 행 |
| `inputs/experiment_plan.json` | 평가 전 고정한 조건과 대상 |
| `inputs/episode_history_audit.json` | 기존 기록과 평가 구간 분리 확인 |

평가가 끝나면 두 shard의 동일 checkpoint·조건·코드·runtime과 정확한 행 수를
확인한 뒤 자동 집계한다. 신뢰구간은 episode를 재표본하는 paired percentile
bootstrap 95% 구간이며, 한 training seed의 평가 환경 변동성을 나타낸다.
다수의 학습 seed에 대한 불확실성이나 동시 다중비교 신뢰구간은 아니다.

같은 episode ID는 같은 topology·이동속도·부하 조건을 대응시키지만, 큐 입장
여부에 따라 공유 RNG 소비가 달라져 이후 traffic/CSI 실현값이 달라질 수 있다.
완전히 동일한 packet trace를 사용했다고 주장하지 않는다. completion/miss는
admitted packet 기준, buffer overflow는 offered packet 기준이다. 1000 slots
종료 시 미완료 packet은 queue에 남으므로 모든 비율의 합이 1일 필요는 없다.
ACK depth별 성능은 raw ACK/전송 수의 합계로 계산하며 빈 구간은 undefined이다.

실행 전 검증: 전체 204개 테스트 통과. 별도 smoke episode 119000–119001에서
GPU 3·5 분할 결과 14행과 단일 GPU 3 결과 14행의 모든 비시간 지표가 정확히
일치했다. 모델 가중치·normalizer·Torch RNG가 평가 중 변하지 않음을 확인했다.
`smoke/` 결과는 본 평가와 논문 성능 결과에서 제외한다.
