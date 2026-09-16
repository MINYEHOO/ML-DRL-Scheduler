# PaperMain — Base · LRANN · NARROW

논문 작업용 독립 실행 폴더. 2026-09-09 서버의 원본 commit
`1564c45cbc56aabe341b896558b9ba2bbea699ff`에서 필요한 파일을 **복사**했다.
원본 Run1~4의 코드·결과·체크포인트는 이동하거나 수정하지 않았다.

서버 위치: `/home/MYH/ML_DRL_Scheduler/PaperMain`

| Recipe | 보존한 run | best update | 학습 조건 |
|---|---|---:|---|
| base | QueuePostRZF_S40HL_CQI4 | 409 | wide, fixed LR, sequential PPO |
| lrann | QueuePostRZF_S40HL_CQI4_LRANN | 539 | wide, LR 3e-4→0 / horizon 866, batched PPO |
| narrow | QueuePostRZF_S40HL_CQI4_NARROW | 739 | 활성 UE 24, 속도 20 km/h, arrival .30 고정, batched PPO |

Base의 별도 학습 seed 3024/4024/5024도 `artifacts/base_s*/`에 있다.
LRANN/NARROW의 독립 학습 seed 반복은 확보된 자료에 없다.
NARROW+LRANN 결합 실험과 과거 Run1~3 실행 결과는 포함하지 않았다.

## 폴더 구성

```text
paper_train.py        세 recipe의 fresh/resume 실행기
paper_eval.py         공통 환경·thread·입력 정규화를 사용하는 평가기
paper_summary.py      기존 CSV의 paired 통계와 누락 자료표
paper_plot.py         보존한 Base main figure를 다시 생성
paper_calibration.py  수정된 CQI4 보정·holdout 검증 (과거 재현은 --mode archived)
verify_package.py     복사 자료와 배포 파일 해시 확인
*.py                  공유 환경·PHY·정책·PPO 핵심 모듈 14개
artifacts/            세 recipe + Base seed 반복의 config/ckpt/학습 CSV
evidence/             기존 ID/OOD·강한 baseline·pilot·보정 결과
provenance/           원본 위치·SHA256·환경 정보·원래 실행 스크립트
docs/original/        원래 system model/논문 메모 (새 검토보다 오래된 기록)
review/              이번 검토 보고서·수치 근거·실행 로그
reports/             기존 결과의 요약 통계
results/             새 평가·그림·보정 진단 출력
runs/                새 학습 출력 (번호_audit_*는 실행 검사 전용)
tests/               설정·재개·정규화·무선 수식·PPO 동등성 검사
```

공유 모듈의 과거 preset 함수 일부는 남아 있다. 검증한 원본 알고리즘을 불필요하게
다시 작성하지 않고, 지원하는 진입점을 위 세 recipe로 제한했다. `provenance/`의
옛 스크립트는 출처 기록이며 실행 진입점이 아니다. 원본 Run 폴더나 외부 worktree를
읽지 않고 이 폴더 안의 파일로 실행한다. Python과 아래 라이브러리는 필요하다.

## 현재 서버에서 사용하기

```bash
cd /home/MYH/ML_DRL_Scheduler/PaperMain
python verify_package.py
python paper_summary.py
```

현재 서버 Python 3.11.0rc1의 관측 환경은 `requirements-server-observed.txt`와
`provenance/source_manifest.json`에 있다. 설치나 업그레이드는 이번 작업에서 하지
않았다. 다른 환경은 같은 모델을 실행할 수 있어도 과거 결과의 bitwise 재현을
보장하지 않는다. 특히 Sionna는 TensorFlow를 사용하는 1.x API다.

**GPU는 먼저 `nvidia-smi`로 확인한다.** 서버가 Exclusive_Process 모드이면 다른
프로세스가 점유한 GPU에는 들어갈 수 없다. 아래 `5`는 이번 짧은 검사 때 사용한
번호이며 영구 예약이 아니다. 실행기는 실제 CUDA 할당도 확인한 뒤 run을 만든다.

```bash
# 설정 확인만 하고 종료
python paper_train.py --recipe lrann --name lrann_new --device cuda --gpu 5 --dry-run

# 수정된 CQI4 + Base batched PPO, 선형 LR 감소, 총 1000 updates
python paper_train.py --recipe base --name base_cqi4_batched_lr1000 --device cuda --gpu 5 \
  --calibration-profile results/cqi4_hl_corrected_v2/summary.json \
  --num-updates 1000 --lr-final 0 --lr-decay-updates 1000 --replay-mode batched

# LRANN/NARROW는 저장된 과거 recipe를 실행하는 명령이다.
# Base 보정값을 적용하려면 위 --calibration-profile을 명시한다.
python paper_train.py --recipe lrann --name lrann_new --device cuda --gpu 5
python paper_train.py --recipe narrow --name narrow_new --device cuda --gpu 5

# 새 폴더에서 생성된 run만 재개; 업데이트 수는 누적 총 목표
python paper_train.py --recipe lrann --resume runs/lrann_new/ckpt/latest.pt
```

세 학습 명령은 선택해서 사용한다. Base 기본 1500 updates는 옛 wrapper의 목표이며
보존된 Base가 그만큼 학습했다는 뜻은 아니다. 기존 Base latest는 update 866이다.
LRANN/NARROW 기본 목표는 866 updates다. 정확한 과거 재학습을 주장하려면 당시
revision과 학습 중단 이력까지 맞춰야 한다.

`--lr-final`과 `--lr-decay-updates`는 함께 지정한다. LR은 3e-4에서 선형으로 줄인다.
기존 driver의 update는
0부터 시작하므로 1000번째이자 마지막 update(index 999)의 LR은 3e-7이다.
수학적 끝점 0은 index 1000에서 도달한다. `--num-updates`는 총 실행 횟수이며
LR horizon을 바꾸지 않는다. 재개할 때 LR 인자를 생략하면 저장된 schedule을
복원하고, 명시하면 같은 값만 허용한다. 고정 LR run에 감소 schedule을 덧붙여
재개할 수 없으며 새 run을 만들어야 한다. LR 설정 자체는 replay 방식을 바꾸지 않는다.

명시적인 추가 학습 구간은 기존 schedule에 `lr_initial`과 `lr_start_update`를 함께
기록한다. CLI에서는 `--lr-initial`·`--lr-start-update`이며 기존 endpoint/horizon
인자도 필요하다. LR은 `initial + (final-initial) * min((update-start)/horizon, 1)`로
계산하므로 프로세스 재시작이나 목표 update 연장으로 감소가 초기화되지 않는다.
시작점이 0보다 큰 구간은 그 직전 또는 이후 checkpoint를 가진 별도 continuation
run에서만 허용한다. 기존 run의 schedule을 CLI로 변경할 수 없다. 새 checkpoint에는
schedule 자체도 저장하고, 이후 resume 때 manifest와 다른 값이면 중단한다.

현재 16번은 완성된 14번의 latest(update 999)를 복사한 추가 학습 실험이다.
기존 1000 updates를 유지하고 update 1000부터 1999까지 1000개를 더 실행한다.
새 구간은 LR `1e-5 → 0`, 시작점 1000, horizon 1000이다. update 1000의 LR은
`1e-5`, 1500은 `5e-6`, 마지막 1999는 `1e-8`이고 수학적 끝점 0은 2000이다.
config의 과거 초기 LR `3e-4`는 보존하며 실제 추가 학습 LR은 manifest의 새 구간을
따른다. 모델·critic·normalizer·Adam·난수 상태를 이어받고, 이전 best checkpoint도
보존한다. 처음부터 2000-update schedule로 학습한 결과와는 구별해야 한다.

`--replay-mode batched`는 Base에도 배치 PPO 가속을 명시적으로 적용한다.
옵션을 생략하면 과거 recipe를 유지한다(Base 순차, LRANN/NARROW 배치).
`--replay-mode sequential`로 순차 방식을 명시할 수도 있다. 재개할 때 옵션을
생략하면 저장된 방식을 복원하고, 명시하면 같은 값만 허용한다.
배치 방식은 minibatch를 묶어서 계산하며 수치 합산 순서가 달라 순차 실행과
비트 단위의 동일한 학습 궤적은 보장하지 않는다. 전환은 새 run으로 기록한다.
배치 결과가 비유한 값이거나 순차 교차검증에 실패하면 잘못된 gradient를
optimizer에 적용하기 전에 지우고 오류로 종료한다. 실패를 자동 재시도하지 않는다.

결정론적 PPO 평가는 `ActorCritic.deterministic_action()`을 사용한다. 기존과 같은
actor logits·마스크·argmax·HARQ·LA budget 계산을 유지하고, 평가에 사용하지 않는
확률분포·log-probability·entropy·critic 계산과 학습용 trace 생성을 생략한다.
GPU5의 같은 checkpoint와 캐시를 사용한 3 × 1000 slots 비교에서 135.0초에서
99.9초로 줄었으며, 3000개 slot의 행동이 정확히 같고 모든 평가 지표도 일치했다.
이는 해당 checkpoint에서의 실측이며 학습 단계와 서버 부하에 따라 시간은 달라진다.
baseline 평가와 평가 episode 수·seed·길이는 그대로다. `paper_eval.py`도 같은
경로를 사용한다. `console.log`의 `eval_s`/`[eval-time]`과 TensorBoard의
`timing/eval/*_seconds`, `timing/eval_total_seconds`에 평가 시간이 기록된다.

14번은 12번의 checkpoint update 20을 이어받아 1000 updates를 정상 완료했다.
16번의 `PARENT_RUN.json`과 `lineage/`에 부모14의 원본 manifest/config, checkpoint
해시, 이전 schedule과 추가 학습 schedule을 기록했다. CSV는 update 999까지의
이력을 포함하고 TensorBoard의 새 event는 1000부터 시작한다. 이전 event는
14번과 그 부모12번에 보존한다. 16번의 첫 추가 정규 평가는 update 1009다.

새 run은 전체 config, source digest, 라이브러리·GPU·thread, LR schedule을
`paper_manifest.json`에 기록하며 schedule 출처(`recipe`/`override`)도 구별한다.
재개 시 이들이 달라지면 중단한다.
`artifacts/*/ckpt`는 과거 결과 보존용이므로 제자리 재개를 허용하지 않는다.

```bash
# 짧은 실행 확인: main의 32 UE/PHY는 유지하되 32 slots·1 update로 축소
python paper_train.py --recipe lrann --name my_smoke --smoke-slots 32 --num-updates 1
python paper_train.py --recipe lrann --resume runs/my_smoke/ckpt/latest.pt --num-updates 2
```

## Run별 학습 진행 확인

`runs/RUN_INDEX.md`에서 각 run의 파일로 들어갈 수 있다. 각 run 맨 위의
바로가기는 실제 기록 파일을 연결하므로 값이 계속 갱신된다.
기존 run 폴더에는 최초 실행 순서대로 `01_`부터 번호를 붙였다.
`runs/RUN_ORDER.json`에 기존 이름과 실행 시각 근거가 있다.
10번 run의 번호 없는 기존 이름은 과거 경로를 유지하는 호환용 바로가기다.
새 run은 처음부터 번호가 붙은 이름을 사용한다.

| 파일 | 내용 | 갱신 시점 |
|---|---|---|
| `eval.csv` | 고정 검증 episode의 평가 점수 | 평가 회차 전체 완료 후 |
| `train.csv` | 학습 episode의 reward·goodput·completion 등 | 각 update 완료 후 |
| `ppo.csv` | PPO loss·KL·entropy | 각 update 완료 후 |
| `console.log` | 실행 중 출력 | 출력 발생 시 (일반 진행 줄은 5 updates마다) |
| `launch.json` | PID·종료 상태 | 프로세스 시작/종료 시 |
| `RUN_FILES.md` | 해당 run의 파일 안내 | 목록 정리 시 |

CSV 원본은 각 run의 `csv_logs/`에 그대로 있다. 예를 들어
`eval.csv`는 `csv_logs/eval_metrics.csv`와 같은 파일을 가리킨다.
현재 Base의 첫 평가는 10 updates 완료 후 시작하며, CSV에는 `update=9`로
기록된다. 첫 평가에는 baseline들도 포함되므로 평가 전체가 끝나기 전까지
헤더만 보일 수 있다. 매 update의 학습 점수와 고정 검증 점수는 구분한다.

```bash
cd /home/MYH/ML_DRL_Scheduler/PaperMain
# 새 run을 추가한 뒤 같은 파일 구조로 정리
python paper_run_index.py

# 현재 run의 진행 출력
tail -f runs/16_base_cqi4_lrrestart2000_s2024_20260910_gpu5/console.log
```

학습 중 생성되는 평가는 위 run 폴더에 저장한다. 아래 `paper_eval.py`로
별도 실행한 비교 평가 결과는 지정한 `results/` 폴더에 저장한다.

## 평가와 논문 결과

```bash
# 기록된 21000번대의 재평가. 이미 열람한 band이며 새 blind test가 아니다.
python paper_eval.py --episode-start 21000 --episodes 100 --worlds ID --out results/id_recheck

# D26 재평가: 환경 deadline 2..6, 정책의 학습 정규화는 /12로 유지
python paper_eval.py --episode-start 30000 --episodes 100 --worlds D26 --out results/d26_recheck

# 모든 OOD 세계. --baselines all은 기존 12개 + 확장 7개 기준선 포함
python paper_eval.py --episode-start 30000 --episodes 100 --worlds P055 P010 V60max CSI02 D26 STORM2 K8 K48 K60 --baselines all --out results/ood_recheck

# 과거 Base main 그림 재생성
python paper_plot.py --out results/base_historical_figure
```

평가 기본값은 CPU·4 threads·세 정책이고, `--recipes`로 Base seed 반복도 선택할 수
있다. `strong` baseline에는 SUS+CQI, SU+CQI, SUS-RPS, SU-RPS,
SUS+CQI-Feasible, SUS+Deadline-PF-Feasible, SUS+CQI@m=3, PF-Greedy-SDS,
CQI-Greedy-SDS가 들어간다. 모든 정책의 평가 환경 seed는 **2024**로 고정하며
정책의 학습 seed와 구별한다. 각 policy cfg의 normalizer는 동결하고 K scaling의
텐서 크기만 환경에 맞춘다. ACK는 전체·depth별 numerator/denominator도 저장해
나중에 pooled rate를 계산할 수 있다.

새 공통 runner의 결과는 별도 `results/`에 생성된다. random baseline의 RNG는
episode별로 고정하므로 예전 runner와 동일한 random action trace를 주장하지 않는다.
`paper_summary.py`는 기존 CSV만 요약하며 새 평가 결과를 자동으로 섞지 않는다.

## 검토 판정

상세 판정은 `review/REVIEW_KO.md`를 먼저 읽는다. 실행 가능성과 논문 최종 검증은
구별된다. 이번 패키지는 경로·설정 복원·D26 정책 정규화 문제를 진입점에서 처리했지만,
다음 연구 문제는 남아 있다.

1. CQI4 sampler의 무효 표본 문제는 수정했다. 적용 범위는 feedback 조건을 만족하는
   신규 full-capacity 무작위 그룹이다. 실제 PPO 정책의 ACK 90%나 논문 성능을 보장하지
   않는다. 새 β를 쓰는 학습과 충분한 정책·baseline 평가가 필요하다.
2. NARROW는 wide 평균 22.5 km/h/.325가 아닌 20/.30을 고정했다. 다양성만 제거한
   실험이라고 쓰지 않는다.
3. 원본 PPO의 batched 검증 실패 처리 결함은 PaperMain에서 수정했다.
   검증 실패 시 gradient를 지우고 종료하며, 과거 실행 기록은 그대로 보존한다.
4. NARROW D26 기존 CSV는 재평가가 필요하며 LRANN OOD 자료는 아직 없다.
5. LRANN/NARROW 여러 training seed, recipe 동결 후 미사용 test band가 필요하다.

## 수정된 HL CQI4 사용법

기준은 `artifacts/base/config.json`의 HL CQI4 세계이며, 과거 config/ckpt는 보존한다.
새 보정은 CQI=0과 사전 예측량이 epsilon 미만인 UE를 그룹 선택 **전에** 제외한다.
최종 그룹의 전력으로 RZF를 계산하고 실제 MI=0 실패는 유지한다. 10th percentile β를
소수점 네 자리로 먼저 동결한 뒤, 별도 holdout에 적용한다. 모든 표본에서
`β × raw capacity >= b_tx_epsilon`을 만족해야 적용 가능한 profile을 발행한다.

```bash
# 새 폴더에서 전체 보정: 36 × 1000 slots + holdout 12 × 1000 slots
python paper_calibration.py --threads 4 --output results/my_cqi4_calibration

# 짧은 진단. diagnostic_only 결과는 학습/평가에 적용할 수 없다.
python paper_calibration.py --episodes 2 --holdout-episodes 2 --slots 32

# 과거 sampler 재현이 필요한 경우만 선택
python paper_calibration.py --mode archived --episodes 1 --holdout-episodes 1 --slots 32

# 기존 Base 가중치에 새 β를 적용한 평가. 재학습 성능과는 구별한다.
python paper_eval.py --recipes base --episode-start 81000 --episodes 100 \
  --calibration-profile results/cqi4_hl_corrected_v2/summary.json \
  --out results/base_corrected_sensitivity

# 새 β로 시작한 run 재개: profile은 manifest에서 복원된다.
python paper_train.py --recipe base --resume runs/base_cqi4_corrected/ckpt/latest.pt
```

`--calibration-profile`이 없으면 **과거 β**를 사용한다. 수정본을 실행하려면 위처럼
profile을 지정한다. 중간에 β를 바꿔 기존 학습을 재개하는 것은 거부한다. Profile은
보정 코드·Base config SHA256에 연결되므로 관련 소스가 바뀌면 다시 보정해야 한다.
Cal 50000–50035, holdout 70000–70011은 이미 사용한 자료이며 새 논문 test에 재사용하지
않는다. 학습 seed를 바꿀 때도 `seed + episode_idx` 충돌을 검사한다. NARROW/OOD에
Base β를 사용하는 것은 transfer 실험이며 해당 세계를 별도로 보정했다는 뜻이 아니다.

세부 설계·검증 결과는 `review/CQI4_FIX_KO.md`, 원시 표본과 동결된 fit은
`results/cqi4_hl_corrected_v2/`에서 확인한다. `review/REVIEW_KO.md`는 분리 당시의
검토 기록이며 CQI4 현황은 이번 수정 보고서가 갱신한다.

## 검사

```bash
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES= python tests/test_wireless_main.py -v
CUDA_VISIBLE_DEVICES= python tests/test_replay_batch.py
CUDA_VISIBLE_DEVICES= python tests/test_ppo_update_batched.py
```

새 평가·학습 smoke는 실행 진단용이다. 특히 짧은 episode는 noise calibration에도
영향을 주므로 논문 성능 데이터로 사용하지 않는다. 이번 작업에서 장기 재학습이나
전체 100-episode OOD 재평가를 시작하지 않았다.


## 선택형 FTP3 도착 과정 (2026-09-10)

`paper_train.py --traffic-model ftp3`로 UE별 Poisson 도착을 선택한다.
옵션 생략 또는 `--traffic-model bernoulli`는 기존 Bernoulli 도착을 사용한다.
선택값은 config/manifest/checkpoint에 저장되고 resume 시 자동 복원된다.
이미 학습 중인 run에서 traffic 모델을 바꾸는 resume는 거부한다.

FTP3는 매 슬롯 `N ~ Poisson(p_arrival_ep)`개를 UE마다 생성한다. 기존 숫자는
도착 확률 대신 평균 packets/UE/slot로 해석하며, 현재 Base의 episode별
U(0.15,0.50)는 0.5 ms 슬롯 기준 300–1000 packets/s/UE이다.
평균 offered load를 맞춘 비교이며 `-log(1-p)` 변환은 하지 않는다.
동일 슬롯의 복수 도착, busy 상태에서의 도착, 모든 초과 도착의 overflow
집계를 지원한다. 한 슬롯에 도착한 패킷의 시각은 슬롯 경계로 양자화한다.
패킷 크기 4000–12000 bits, deadline 3–12 slots, FIFO capacity 8,
UE 수·mobility·CSI·PHY·보상·PPO·평가 설정은 기존 Base 비교 run과 동일하다.
이는 FTP Model 3의 Poisson 도착 과정에 본 연구의 패킷/서비스 설정을 적용한
모델이며, 특정 3GPP 평가 표의 모든 파라미터를 그대로 사용했다는 뜻은 아니다.

기존 CQI4 v2 beta [1.0162,0.8509,0.7732,0.728]는 유지한다. 기존 calibration
JSON을 수정하지 않고, `provenance/before_ftp3_sources`의 원본 소스 전체 해시로
검증한 뒤 FTP3에 전이한다. 새 세계의 holdout ACK 성능이 검증되었다고
주장하지 않는다. 일반 `--calibration-profile`의 현재 소스 검증은 엄격하게
유지하며, 명시적 transfer에만 아래 reference 옵션을 사용한다.

```bash
python paper_train.py --recipe base --name NEW_UNIQUE_NAME   --device cuda --gpu 4 --threads 4 --seed 2024 --num-updates 1000   --lr-final 0 --lr-decay-updates 1000 --replay-mode batched   --traffic-model ftp3   --calibration-profile results/cqi4_hl_corrected_v2/summary.json   --calibration-reference-root provenance/before_ftp3_sources
```

같은 명령에서 새 run 이름과 `--traffic-model bernoulli`를 사용하면 기존
방식으로 새 학습을 시작할 수 있다. GPU 5의 run16은 수정 전 메모리상의
코드로 계속 실행되며, 원본 run16 source snapshot 및 tar archive를 보관했다.
소스가 달라진 기존 run의 무단 resume를 허용하도록 guard를 약화하지 않았다.
FTP3와 Bernoulli의 CSI 설정과 분포는 같지만 traffic/CSI가 공유하는 RNG의
소비량은 달라지므로 CSI feedback 난수열까지 동일한 짝 비교는 아니다.

- 17_audit_ftp3_gpu4: 32-slot 실행·평가·resume 검증, 논문 결과에서 제외.
- 18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4: 처음부터 1000 updates,
  LR 3e-4 → 0 (horizon 1000), 1000 slots/update, 10 updates마다 평가.
- 실시간 결과: 각 run의 `eval.csv`, `train.csv`, `ppo.csv`, `console.log`.
- 검증 근거: `review/ftp3_validation.json`, `review/logs/ftp3_regression_tests.log`.

정의 참고: [3GPP TR 38.812 Annex A.3.2](https://atisorg.s3.amazonaws.com/archive/3gpp-documents/Rel16/ATIS.3GPP.38.812.V1600.pdf).
