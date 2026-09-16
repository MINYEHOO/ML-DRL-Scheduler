# HL CQI4 수정 및 검증

2026-09-09 · 수정 위치: `/home/MYH/ML_DRL_Scheduler/PaperMain`

CQI4 보정에서 송신 불가능한 UE를 그룹에 넣던 문제를 수정했다. HL Base 설정을 기준으로 전체 보정 36개와 별도 holdout 12개, 각각 1000 slots를 완료했다. 구현과 독립 설계·코드 검토, 회귀 검사, 원시 표본 재검산을 병렬로 수행했다.

## 실제 변경

- **그룹 선택 전 후보 제외:** CQI=0, 사전 예측 전송량이 epsilon 미만인 UE 및 영벡터 채널 추정치를 제외한다. 선택한 그룹의 실제 depth와 균등 전력으로 RZF를 계산한다. 무효 행을 사후 삭제하는 방식은 사용하지 않는다.
- **실제 실패 보존:** 예측량이 양수인데 실제 MI가 0인 표본은 실패로 남긴다. 실제 채널은 그룹을 정한 뒤 성능을 계산하는 데만 사용한다.
- **β 동결과 그룹 일관성:** 기존 β를 제거한 raw capacity로 10th percentile을 구하고 네 자리로 동결한다. calibration과 holdout의 모든 `β × capacity >= 1 bit`를 확인한다. 하나라도 실패하면 이 보정 결과의 적용을 막는다.
- **ACK 일치:** 실제 runtime과 같은 `MI >= B_tx - 1e-6 bits`를 사용한다. 보존한 MI와 엄격한 ratio 기준 지표도 별도로 기록한다.
- **실행 연결:** `paper_train.py`/`paper_eval.py`에 `--calibration-profile`을 추가했다. 학습 config와 평가 환경·정책 내부 planner에 같은 β를 적용한다. 재개에서 profile 변경, stale source 및 calibration/holdout 채널 seed 재사용을 차단한다.

추가/수정한 주 코드: `calibration/cqi4.py`, `calibration/profile.py`, `paper_calibration.py`, `paper_train.py`, `paper_eval.py`. PHY·CQI 표·HARQ의 기존 동작을 수정한 것은 아니다. 문제는 보정 모집단 구성과 그것을 실행 설정에 연결하는 과정에 있었다.

## 전체 보정 결과

| 동시 stream 수 m | 과거 β | 수정 β | 새 β의 holdout first-ACK | episode-cluster 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 1.0018 | 1.0162 | 90.01% | 88.93–91.00% |
| 2 | 0.7499 | 0.8509 | 92.77% | 91.67–93.92% |
| 3 | 0.6592 | 0.7732 | 93.20% | 92.21–94.17% |
| 4 | 0.6058 | 0.7280 | 93.25% | 92.09–94.44% |

m=2–4의 신뢰구간은 90%보다 높다. 따라서 모든 spatial depth가 같은 90% reliability를 달성했다는 주장은 아직 성립하지 않는다. `validated`는 표본·그룹 유지·출처·전체 실행의 기술 검사를 통과했다는 뜻이다. 이 holdout을 다시 맞추는 데 사용하지 않았다.

기준 세계 seed는 2024다. Calibration은 episode 50000–50035 (sampler seed 777), holdout은 70000–70011 (sampler seed 20250713)이다. 7 slots마다 각 depth에서 RBG를 무작위로 두 번 선택하고, 선택한 RBG마다 유효 UE 중 한 그룹을 뽑았다. Bootstrap은 에피소드 단위로 10,000회, seed 42다. Holdout을 확인한 뒤 β를 다시 맞추지 않았다.

| m | 보정 표본 수 | holdout 표본 수 | 두 단계 전체 최소 β×capacity (bits) | 같은 holdout에서 과거 β의 ACK |
|---:|---:|---:|---:|---:|
| 1 | 10,296 | 3,432 | 208.0072 | 95.48% |
| 2 | 20,592 | 6,864 | 21.0608 | 96.37% |
| 3 | 30,888 | 10,296 | 7.4947 | 96.55% |
| 4 | 41,184 | 13,728 | 3.9362 | 96.55% |

m=1의 β가 1보다 약간 큰 것은 오류가 아니다. 현재 모델은 CQI floor로 예측률을 보수적으로 잡을 수 있으며, β는 실제 MI/예측량 비율의 분위수다. Config도 β를 1.5까지 허용한다. 이 보정은 β가 항상 1 이하라는 제약을 쓰지 않는다.

같은 holdout의 과거 β 열도 **수정된 유효 그룹 모집단**으로 계산한 비교다. 과거의 무효 표본을 포함한 ACK와 직접 섞으면 안 된다. 새 β는 특히 다중 stream에서 전송량을 더 크게 잡는다. 이것만으로 실제 정책의 goodput 개선을 단정할 수 없다.

## 검증

- 서버 회귀 테스트 **76개 통과**. 독립 reviewer가 발견한 ACK 허용오차와 seed 변경 시 채널 중복 문제도 수정·검증했다.
- 별도 스크립트가 저장된 원시 표본으로 β, MI/capacity, 그룹 수, epsilon 조건, ACK, bootstrap CI와 파일 해시를 독립 재계산해 통과했다.
- 새 β를 사용하는 Base의 GPU 5 학습 1 update와 resume 1 update를 완료했다. 모두 32-slot 실행 진단이며, profile·config·checkpoint가 일치한다.
- Base 기존 가중치로 수정 전·후 각각 1000-slot ID 1회 실행을 완료했다. 32-slot 검사에서는 Base와 9개 baseline 모두 실행됐다. 이 소규모 결과는 실행·민감도 확인이며 논문 성능 비교가 아니다.
- 원본 코드 16개, 원본 실험 파일 137개 및 PaperMain 내 보존 복사본 137개의 SHA256 불변을 확인했다.

메타데이터를 과도하게 포함하던 소스 해시 처리를 수정해 다른 컴퓨터로의 복사에도 검증이 유지되게 했다. `v2`는 같은 seed·표본 계획으로 재실행한 최종 배포 보정이며 추가 독립 표본 48개를 뜻하지 않는다. `v1` 기록은 과거 검증 근거로 보존한다. 논문·새 실행에는 `v2` profile을 사용한다.

## 수정본 실행

```bash
cd /home/MYH/ML_DRL_Scheduler/PaperMain

# 현재 비어 있는 GPU 번호를 확인한 뒤 지정한다.
python paper_train.py --recipe base --name base_cqi4_corrected --device cuda --gpu 5 \
  --calibration-profile results/cqi4_hl_corrected_v2/summary.json

# 새 run 재개: profile은 저장된 manifest에서 복원된다.
python paper_train.py --recipe base --resume runs/base_cqi4_corrected/ckpt/latest.pt
```

`--calibration-profile`을 생략하면 과거 β를 사용한다. 과거 checkpoint와 결과는 보존하며 제자리 재학습하지 않는다. 새 β로 시작한 장기 학습은 아직 실행하지 않았다.

## 논문 해석의 범위

이 보정은 feedback 조건을 만족하는 **신규 전송만으로 구성한 full-capacity 무작위 그룹**의 기준점이다. 실제 PPO가 선택한 그룹, backlog 때문에 전송량이 제한된 패킷, HARQ 혼합 그룹의 분포와 같지 않다. 따라서 전체 정책의 ACK 90% 또는 NR BLER 10% 보장으로 쓰면 안 된다. 원래의 추상 MI 모델 범위 내에서 보정 오류를 바로잡은 것이다.

MU depth의 holdout ACK가 목표를 웃도는 것은 남아 있는 보수적 보정/일반화 차이로 기록한다. 유한한 episode 표본과 채널 구성 차이가 가능한 원인이지만 확정하지는 않는다. CQI=0 후보 비율은 calibration 5.19%, holdout 8.71%였다. 보고한 신뢰구간은 동결된 β 조건의 평가 불확실성이며 β 추정 자체의 불확실성은 포함하지 않는다.

Cal/holdout은 이미 열람한 구간이다. 논문용 최종 평가는 recipe를 동결한 뒤 별도 test 구간에서 진행해야 한다. NARROW에 Base β를 적용하면 환경 간 transfer 실험이다. LRANN/NARROW 배치 검증 실패 처리, NARROW D26 재평가와 추가 seed 실험은 이전 검토의 별도 과제로 남는다.

Calibration 자료만 사용하는 2,000회 episode bootstrap(seed 7301)에서 raw β의 95% 구간은 m=1–4 순서대로 [1.01444, 1.01860], [0.83426, 0.86924], [0.75250, 0.79184], [0.70551, 0.74893]였다. 이는 β 추정 불확실성의 별도 진단이며 holdout에 맞춘 재조정은 아니다.

상세 근거: `results/cqi4_hl_corrected_v2/summary.json`, `calibration_fit.json`, `calibration_samples.npz`, `holdout_samples.npz`, `review/cqi4_sample_audit.json`, `review/cqi4_validation_summary.json`, `review/logs/cqi4_*`.
