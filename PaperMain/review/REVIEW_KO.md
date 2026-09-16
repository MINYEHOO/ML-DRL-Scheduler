# PaperMain 분리 및 집중 검토

검토일: 2026-09-09. 원본: 서버 `/home/MYH/ML_DRL_Scheduler`,
commit `1564c45cbc56aabe341b896558b9ba2bbea699ff`.
새 폴더: `/home/MYH/ML_DRL_Scheduler/PaperMain`.

**판정: 세 실험을 독립 실행 가능한 구성으로 분리했고, 핵심 실행·수식 검사는
통과했다. 다만 논문의 최종 성능 근거를 확정하려면 β 보정, 배치 검증 실패 처리,
NARROW D26 재평가와 추가 반복 실험이 필요하다.** 기존 결과를 모두 무효로
판정한 것은 아니다. 실행 성공과 연구 주장의 충분성을 구별해야 한다.

## 1. 보존한 범위

| Recipe | best update | 학습 설정 |
|---|---:|---|
| Base main | 409 | wide, fixed LR 3e-4, sequential PPO |
| LRANN | 539 | 같은 wide 환경, LR 3e-4→0 / 866 updates, batched PPO |
| NARROW | 739 | 활성 UE 24, 속도 20, arrival .30, fixed LR, batched PPO |

Base의 독립 학습 seed 3024/4024/5024도 함께 복사했다. 따라서 recipe는 3개,
학습 산출물 세트는 6개다. 각 세트의 config, best/latest checkpoint, 학습·검증
CSV 및 console log를 보존했다. Base 기존 그림, 공통 비교 CSV, 강한 baseline,
OOD, 임곗값 pilot, CQI4 보정 근거도 들어 있다. TensorBoard event 중복 저장은
제외했다. 원본은 변경하지 않았으며 NARROW+LRANN 결합 실험은 제외했다.

원본에서 복사한 137개 파일은 SHA256으로 확인한다.
`provenance/source_manifest.json`에 원래 위치·크기·해시·서버 라이브러리 버전이
기록되어 있다. 실행 후 생성되는 `runs/`, `results/`는 과거 결과와 별도다.

## 2. 분리 과정에서 처리한 실행 문제

- **고정 경로:** 삭제된 `_cqi4dev`, 옛 Git pin, 고정 GPU에 의존하는 wrapper를
  실행 경로에서 제외하고 portable `paper_train.py`/`paper_eval.py`를 추가했다.
- **설정 복원:** checkpoint의 tensor shape만 맞으면 잘못된 환경으로 재개될 수
  있었다. 새 학습기는 전체 config·source hash·runtime·thread·schedule을 기록하고
  재개 시 검사한다. LRANN horizon 866도 복원된다.
- **평가 정규화:** 공통 평가기는 환경 cfg와 학습된 정책 cfg를 분리한다.
  D26 환경은 deadline 2..6, 정책 normalizer는 학습 당시 /12다.
  K8/48/60은 정책 텐서 크기만 바꾼다.
- **공정한 평가 입력:** 모든 정책과 학습 seed 반복을 seed-2024 Base 세계에서
  비교한다. 강한 baseline 및 SU+CQI를 포함하고 thread 설정도 통일한다.
- **원본 보존:** 학습은 새 `runs/`에만 저장하며 archive checkpoint의 제자리
  resume를 거부한다. 결과 덮어쓰기와 잘못된 경로도 차단한다.
- **GPU 점유:** Exclusive_Process GPU에서 실제 CUDA allocation까지 확인한 후
  새 run을 만든다. 검사 초기에 0~4번이 다른 프로세스에 점유되어 발생한 오류는
  기록으로 남겼고, 해당 프로세스를 중지하거나 GPU 설정을 바꾸지 않았다.

환경·PHY·policy·PPO core 13개 파일은 원본과 동일하다. `train_phase2.py`의 변경은
완전한 Config를 주입하는 작은 진입점 추가이며, loss/gradient/환경 알고리즘은
바꾸지 않았다. 원래 `train_phase2.py` CLI를 직접 사용하면 옛 resume 한계가
남으므로 지원하는 진입점은 `paper_train.py`다.

## 3. 해결이 필요한 발견

### F1 — [P1] CQI4 β의 보정 모집단이 실제 전송 그룹과 다름

대상: 세 recipe 공통. 위치: `calibration/beta_m_calib_holdout.py:48–56`,
`env.py:604–613`, `la_planner.py:138–154`.

sampler는 CQI=0인 UE까지 그룹에 넣고 `MI / max(cap,1e-12)=0`을 실패로
집계한다. runtime의 신규 전송은 이 UE를 제외하고 생존 그룹의 depth·전력으로
다시 계산한다. 따라서 보정 통계의 first-ACK와 실제 전송 대상의 정의가 다르다.

독립 2인 예에서 sampler의 ratio는 `[1,0]`, 살아 있는 UE의 SINR는 5였다.
실제 closure는 1인으로 줄어 SINR가 10이 된다. 단순히 0인 행만 제거해서는
전력 재분배 차이까지 고칠 수 없다. 이전 전체 검토의 실제 1,000-slot 보정
episode에서도 depth별 9.27–11.89% 무효 표본을 확인했다. 그 비율은 이전
진단의 재인용이며 이번 전체 36+12 재보정 결과가 아니다.

**필요 작업:** 전송 가능한 그룹/최종 depth에 맞춘 sampler, 분리된 calibration/
holdout, 새 β에서 frozen 정책 및 강한 baseline의 민감도 평가.
`paper_calibration.py`는 과거 결함을 재현하고 무효 표본 수를 기록하는 도구다.
기존 β나 checkpoint를 자동 변경하지 않는다.

### F2 — [P1] 기존 NARROW D26 결과가 Base strict 평가와 다른 전처리를 사용

위치: `provenance/original_scripts/ood_policy.py:80–85`, `policy.py:74,86`.
기존 NARROW D26은 deadline normalizer까지 12→6으로 바꿨다. Base strict는
/12를 유지했다. 새 공통 평가기는 이 문제를 처리했지만, **기존 NARROW D26
100개 결과를 새 프로토콜로 다시 평가한 것은 아니다.**

기존 CSV는 보존하고 요약기의 자료표에서는 `invalid_protocol`로 표시했다.
strict 비교 통계에서는 제외한다. LRANN의 OOD CSV는 확보된 자료에 없으므로
새 공통 평가기로 생성해야 한다.

### F3 — [P2] LRANN/NARROW 배치 검증 실패 뒤 보호 동작이 잘못됨

위치: `ppo.py:288–321`. Base sequential 경로는 해당하지 않는다.
배치 cross-check가 실패하면 지역 변수만 sequential로 바꾸고, 이미 계산한
배치 gradient를 그대로 optimizer.step에 사용한다. 다음 update에는 cfg에서
batching을 다시 켠다. 정상 경로의 동등성 통과가 이 실패 처리까지 검증하지는
않는다. 이번 폴더의 LRANN 설정으로 fault injection을 다시 수행했고 update 0과 1 모두
실패를 출력하면서 배치 호출과 weight 변경을 계속하는 것을 재현했다.
근거: `review/logs/batch_failure_probe.log`.

**필요 작업:** gradient를 버리고 해당 minibatch를 sequential로 다시 계산하거나
즉시 중단하도록 수정하고, fallback 상태를 지속·기록하는 검증을 추가한다.
이번 분리는 연구 알고리즘을 보존했으며 해당 처리를 임의로 바꾸지 않았다.

### F4 — [P2] NARROW는 다양성뿐 아니라 평균 부하·속도도 변경

| 조건 | Base/LRANN wide 평균 | NARROW |
|---|---:|---:|
| 속도 km/h | 22.5 | 20 |
| arrival probability | .325 | .30 |
| 활성 UE | 24 | 24 |
| 평균 offered packets/slot | 7.8 | 7.2 |

NARROW의 기존 결과는 “고정 training scenario” 대조로 사용할 수 있다.
“평균 난이도는 같고 다양성만 제거했다”는 해석은 성립하지 않는다. diversity의
독립 효과를 주장하려면 평균 22.5/.325를 맞춘 대조와 학습 procedure/seed를
맞춘 비교가 필요하다. 평균을 맞춰도 비선형 시스템의 평균 난이도가 같다는
뜻은 아니므로 주장 범위를 실제 통제한 조건에 맞춰야 한다.

### F5 — [P2] 같은 seed가 같은 realized traffic/CSI trace를 보장하지 않음

위치: `traffic.py:93–96,121–128`, `env.py:171–172,186`.
queue overflow 시 size/deadline draw를 건너뛰고 CSI가 같은 RNG를 사용한다.
scheduler별 queue full 여부가 달라지면 후속 외생 draw가 달라질 수 있다.
main 조건 안의 제어된 counterexample에서 overflow 1개 뒤 다음 CSI mask가
13/32 UE에서 달라졌다.

이는 공통 trace라는 설명의 한계다. 이 사실만으로 paired CI가 무효이거나
정책 순위가 편향됐다고 단정할 수 없다. 엄밀한 trace matching을 원하면
arrival/packet attributes를 admission과 독립적으로 생성하고 CSI RNG를 분리한다.
그 변경은 새 seed protocol로 관리해야 한다.

## 4. 기존 실험에서 현재 읽을 수 있는 결론

이미 여러 비교에 사용한 공통 episode 21000–21099의 기록을 다시 계산했다.
각 policy는 training seed 2024의 validation-selected best checkpoint다.

| Recipe | 평균 reward | Base 대비 paired 차이, 95% CI |
|---|---:|---:|
| Base | 4786.68 | 기준 |
| LRANN | 4839.36 | +52.68 [36.74, 68.62] |
| NARROW | 4796.54 | +9.87 [−10.17, 29.91] |

이 자료에서 LRANN best policy의 평균 reward 개선은 명확하지만, NARROW와 Base의
차이는 CI가 0을 포함한다. 이는 **고정 policy의 episode 변동**이며 학습 알고리즘의
training-seed 변동까지 반영한 결론이 아니다. Base는 sequential, 두 변형은
batched라는 학습 구현 차이도 있으며 bitwise 동일한 통제 실험을 주장하지 않는다.

강한 `SUS+CQI-Feasible`의 같은 band 평균 reward는 4663.89다. Base와의 차이는
+122.79 [74.25,171.33]으로, 기존 SUS+CQI만 비교한 headline보다 작은 차이다.
이 baseline은 다른 thread 설정의 job에서 평가되어 작은 수치 불일치가 존재한다.
세 recipe 사이의 same-job SUS+CQI control은 보존 자료에서 서로 일치했다.

기존 20000번대 main 그림, 재사용된 21000번대 recipe 비교, 30000번대 OOD는
별도 자료로 유지했다. 이들을 새 blind test처럼 제시하면 안 된다.
`reports/evidence_summary/`에는 paired 통계·해시 검사·누락 자료표가 있다.

## 5. 무선통신 모델의 의미

RZF의 복소수 convention, 독립 inverse oracle, 비영 beam의 총전력, CQI floor
경계를 검사했다. 공유 planner 구조도 일관된다. 다만 다음은 구현 버그가 아니라
논문에 밝혀야 할 시스템 수준 모델 가정이다.

- CQI 값은 NR 표를 쓰지만 Shannon SE floor가 표준의 MCS/TBS/BLER 기반 CQI
  선택 전체를 구현한 것은 아니다.
- Type-II-like sparse 56-bit PMI는 표준 Type-II signaling 전체의 구현이 아니다.
- 연속 MI 누적/ideal IR/결정적 ACK, control·pilot overhead 0, 즉시 CSI 반영과
  다음 slot HARQ는 추상화다. 현실 NR/URLLC 지연·신뢰도에 직접 일반화할 수 없다.
- noise는 episode 전체 beam gain median을 이용한 운영 SNR 정규화이며 고정
  thermal-noise link budget이 아니다. 짧은 smoke는 noise도 달라진다.
- MI shaping reward와 실제 completed-packet goodput은 같은 목적함수가 아니다.
  논문에는 goodput·deadline miss·completion·fairness를 함께 제시해야 한다.

CQI 기준: [ETSI TS 38.214 §5.2.2.1](https://www.etsi.org/deliver/etsi_TS/138200_138299/138214/15.02.00_60/ts_138214v150200p.pdf).
상세 수식·행 번호·제어 실험은 `review/wireless_detail.md`에 있다.

## 6. 실제 검사 범위

서버 Python 3.11.0rc1, NumPy 1.26.4, TensorFlow 2.15.0, Sionna 1.2.1,
PyTorch 2.7.1+cu118, RTX 4090. 기존 서버 환경을 사용했고 라이브러리는 변경하지 않았다.

- 세 recipe와 Base 반복 3개의 best checkpoint를 자체 config로 strict load: 통과.
- 학습/재개 보호와 OOD 정규화 검사, 독립 무선 수치 검사: 통과.
- 32-slot main dimension의 세 recipe fresh GPU 학습·검증·checkpoint 저장: 통과.
- LRANN 1+resume1과 연속2 비교: 모델·Adam·config·CPU/GPU RNG 모두 exact equal,
  최대 parameter 차이 0. LR=`0.0002996535796766743`, horizon 866 유지.
- ID + 9 OOD 세계, 세 policy 및 baseline의 32-slot 실행: 통과.
- 정상 1,000-slot ID episode에서 세 policy + 8개 비교 scheduler 실행: 통과.
  episode 90020의 진단이며 새로운 논문 성능 비교 결과로 사용하지 않는다.
- 배치 replay/PPO·보정·마지막 package 검사의 상세 결과는
  `review/validation_summary.json` 및 `review/logs/`에 기록한다.

장기 재학습, 전체 36+12 수정된 β 보정, 전체 OOD 9×100 재평가,
LRANN/NARROW 여러 training seed 반복은 수행하지 않았다. 실행 smoke를 근거로
모든 channel realization이나 긴 학습에서 문제가 없다고 보장하지 않는다.

## 7. 다음 연구 작업의 우선순위

1. β sampler/holdout 정의를 수정하고 기존 frozen policy·강한 baseline의 민감도를 확인.
2. 배치 검증 실패 처리를 수정·검증한 후 새로운 장기 run의 코드 버전을 동결.
3. 같은 공통 평가기로 NARROW D26과 LRANN OOD 자료를 보완.
4. LRANN/NARROW 독립 학습 seed 반복 및 동결된 recipe의 미사용 test band 평가.
5. 논문 주장을 NR 추상 모델의 범위와 NARROW가 실제 통제한 조건에 맞춰 작성.

이번 작업은 폴더 분리와 그 범위의 검토까지 완료한 것이다. 위 후속 연구가
완료된 것으로 기존 결과를 다시 쓰거나 원본 데이터를 변경하지 않았다.
