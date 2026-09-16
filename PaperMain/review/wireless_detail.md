# PaperMain: main recipes의 무선통신·평가 모델 검토

2026-09-09, 원본 commit `1564c45cbc56aabe341b896558b9ba2bbea699ff`.
범위는 **Base QueuePostRZF_S40HL_CQI4, LRANN, NARROW**와 공통
channel/CSI/RZF/LA/queue/HARQ/reward 경로다. 과거 Run1/2/3, 결합 NARROW+LRANN은
대상이 아니다. 아래 행 번호는 원본 root Python 또는 원본 Run4 script 기준이며
PaperMain의 복사된 core source에서도 동일하다. 이 검토로 core source와 체크포인트를
수정하지 않았다. 추출 폴더의 portable utility/test는 별도로 추가했다.

## 결론

RZF·CSI 복원·공통 planner·큐/HARQ bookkeeping의 기본 수식과 실행은 일관된다.
독립적인 수치 oracle에서도 확인했다. 다만 현재 세 recipe에 공통 적용된 **CQI4
β 보정의 표본 정의 문제**와 **NARROW D26 OOD의 정책 전처리 변경**을 논문의 최종
결과 생성 전에 처리해야 한다. NARROW를 diversity-only ablation으로 해석하는
주장도 현재 설정으로는 성립하지 않는다. 아래 모델 단순화를 NR 구현 버그와
구분해서 논문에 명시해야 한다.

## 실제 결함·비교 프로토콜 문제

### W1 [P1] CQI4 β calibration과 실제 신규 전송 모집단이 다름 — 세 recipe 공통

위치: 원본 `Run4/_analysis/scripts/audit_probes/beta_m_calib_holdout.py:48-56`
(추출본 `calibration/beta_m_calib_holdout.py:48-56`), CQI4 호출자
`beta_m_calib_queue_s40hl_cqi4.py:25-40`.
Runtime 근거: `env.py:604-613`, `la_planner.py:138-154`, `phy.py:72-74`.

sampler는 모든 UE 중 무작위 그룹을 뽑는다. CQI=0이면 복원 h_hat=0, 예측 cap=0,
해당 stream의 실제 MI=0이며 `MI / max(cap,1e-12)`가 ratio=0 실패로 집계한다.
runtime 신규 전송은 이 UE를 배제하고 나머지 그룹의 depth와 전력을 다시 계산한다.
따라서 단순히 결과에서 0행을 제거하는 것도 완전한 수정이 아니다.

이번 독립적 2인 analytic counterexample에서 calibration ratio는 `[1,0]`, live UE
SINR는 5였지만 runtime closure는 1인 그룹으로 줄어 live UE SINR가 10이었다.
기존 전체 프로젝트 검토의 실제 첫 calibration episode(50000, 1,000 slots)에서도
깊이별 무효 표본이 9.27–11.89%였다. 이 실제 episode 수치는 이전 진단을 재인용하며
이번 turn에서 36+12 전체 보정을 재실행한 것은 아니다.

영향: 배포 β `(1.0018,.7499,.6592,.6058)`를 만든 보정/holdout의 first-ACK 정의가
실제 신규 전송과 다르다. 기존 학습 결과를 모두 무효로 만든다는 뜻은 아니다.
동일 β로 정책 간 비교는 가능하지만 “동일 약 90% first-ACK target으로 spatial
depth를 보정했다”는 근거를 보완해야 한다. full-cap uniform group의 calibration
ACK율과 작은 backlog를 포함하는 실제 scheduler의 ACK율은 원래도 서로 다르다.

조치: CQI 및 최소 payload gate를 통과하는 UE로 scheduler-independent 그룹을
구성하거나 실제 closure를 적용하고 최종 생존 depth로 분류한다. calibration과
holdout을 분리한 채 재보정하고 새 β에 대한 기존 frozen 정책 및 강한 baseline의
민감도를 측정한다. 이 변경은 연구 설정 변경이므로 과거 결과와 새 결과를 버전으로
구분해야 한다. `paper_calibration.py`는 **과거 sampler 재현 및 무효 표본 계수**만
제공하며 새 β를 자동 적용하지 않는다.

### W2 [P1] NARROW D26 OOD에서 학습한 입력 정규화가 바뀜

위치: 원본 `Run4/_analysis/scripts/ood_policy.py:80-85`, feature 연산
`policy.py:74,86-87`. 기존 Base strict runner:
`Run4/_analysis/OOD/scripts/ood_grid_extend100.py:78-84`.

`ood_policy.py`는 D26 환경의 `deadline_max=6` cfg로 ActorCritic도 생성한다.
그 결과 HOL deadline과 next deadline의 정규화 분모가 학습 시 12에서 6으로
바뀐다. 반면 기존 Base D26의 PPO-strict는 정책 cfg 분모 12를 유지한다.
NARROW의 `ood_narrow_g1.log`는 이 runner로 D26 평가를 수행한 것을 기록한다.
따라서 기존 NARROW D26 100개 결과와 Base PPO-strict 결과를 동일 전처리의
zero-shot 비교로 사용할 수 없다.

조치: 정책 cfg는 학습 정규화를 유지하고 실제 환경 cfg만 D26으로 변경하는 공통
평가기를 사용한다. K8/48/60에서는 정책 텐서 shape에 필요한 K만 맞추되 학습
normalizer를 유지한다. NARROW D26을 다시 평가한다. LRANN도 이 runner로 D26을
실행했다면 같은 문제에 해당하지만, 이번 검토만으로 LRANN D26 결과가 이미
오염됐다고 단정하지 않는다. 다른 OOD 세계 전체가 이 이유로 잘못된 것은 아니다.

### W3 [P2] NARROW는 다양성과 평균 부하·속도를 함께 바꿈

위치: `Run4/_wrap_NARROW.sh:21`, 비교 wide recipe `Run4/_wrap_LRANN.sh:21`.
잘못된 설명: `Run4/_analysis/scripts/ood_policy.py:6-10`.

| 조건 | Base/LRANN wide 평균 | NARROW |
|---|---:|---:|
| 속도 km/h | U(5,40): 22.5 | 20 |
| arrival probability | U(.15,.50): .325 | .30 |
| 활성 UE | U_integer(16,32): 24 | 24 |
| 평균 offered packets/slot | 7.8 | 7.2 |

“평균은 같고 diversity만 제거”라는 해석은 맞지 않다. NARROW는 실제로 평균 부하를
7.69%, 평균 속도를 11.11% 낮춘 recipe다. 결과가 어떻든 domain randomization의
독립 효과로 인과 해석할 수 없다. 기존 NARROW를 보존하고 논문에서는 “고정된
training scenario” 대조로 표현하거나, 22.5/.325/24의 평균을 맞춘 추가 대조를
동일 학습 seed·길이·optimizer로 수행한다. 평균을 맞춰도 비선형 시스템의 평균
난이도가 완전히 같아지는 것은 아니다.

### W4 [P2] buffer overflow 후 동일 seed가 동일 traffic/CSI trace를 보장하지 않음 — 세 recipe 공통

위치: `traffic.py:93-96,121-128`, 공유 RNG 사용 `env.py:171-172,186`.

full queue에서 거부된 arrival은 size/deadline용 RNG를 소비하지 않는다.
따라서 두 scheduler의 queue full 여부가 다르면 그 다음 다른 UE의 arrival과
다음 slot의 CSI Bernoulli draw도 달라질 수 있다. 실제 Base 학습 CSV의 CSV
868행 중 300행에서 buffer_overflow_rate>0, 최대 약 0.1276%였다.

이번 controlled probe는 main support 내부 p=.5, q8, K32, RNG seed8로 한쪽의
UE0 queue만 미리 full로 구성했다. 한쪽만 overflow 1개가 발생한 뒤 다음 CSI
mask가 13/32 UE에서 달라졌다. 이는 제어된 queue-state counterexample이지
전체 episode의 PPO-vs-baseline 변화량을 측정한 것은 아니다.

영향은 **common exogenous trace라는 실험 설명**과 paired 분산 감소의 정도다.
각 scheduler가 여전히 올바른 marginal arrival process를 갖는 한, 이 문제만으로
same-seed paired bootstrap/CI가 자동으로 무효가 되거나 정책 순위가 편향됐다고
주장하면 안 된다. 실제 차이의 크기는 별도 평가 대상이다.

조치: traffic event와 packet attributes를 admission 여부와 독립적으로 사전 생성하고
CSI feedback RNG를 분리하면 엄밀한 trace matching이 가능하다. 새 seed protocol
버전을 만들어 과거 평가와 섞지 않거나, 현재 논문 설명을 “동일 topology seed/
환경분포, overflow 이후 realized exogenous traces는 다를 수 있음”으로 한정한다.

## 구현 오류와 구별해야 할 명시적 연구 추상화

- **CQI ladder:** `csi.py:40-57`의 수치는 TS 38.214 Table 5.2.2.1-3와 일치한다.
  그러나 코드의 Shannon SE floor 양자화는 표준의 TB error probability, MCS,
  TBS 및 reference-resource 기반 CQI 선택을 구현하지 않는다. 논문에는 “NR CQI
  table을 이용한 시스템 수준 추상화”로 설명해야 한다.
  [ETSI TS 38.214 §5.2.2.1](https://www.etsi.org/deliver/etsi_TS/138200_138299/138214/15.02.00_60/ts_138214v150200p.pdf)
- **Type-II-like PMI:** sparse 56-bit codebook/OMP 방향 압축은 모델의 설계다.
  표준 Type-II 전체 signaling과 search/feedback 제약을 그대로 구현했다는
  주장은 별도 검증이 필요하다.
- **Shannon/ideal IR:** `phy.py:152-157`, `transmission.py:194-221`은 연속 MI 누적과
  결정적 ACK이다. `eta_data=1`은 control/pilot overhead를 빼지 않는다.
  `env.py:160-196`의 CSI는 feedback slot에서 즉시 쓰이고 HARQ NACK unit은
  다음 slot에 고정 RBG 재전송된다. 이는 현실 NR/URLLC latency/reliability의
  직접 예측이 아니라 주어진 추상 모델에서의 scheduling 비교다.
- **CQI0 HARQ:** `phy.py:103-105`, `csi.py:32-38`은 fixed retransmission의 zero
  beam과 전력 미사용을 의도적으로 정의한다. 이번 test도 이 동작을 확인한다.
  CQI0 자체가 물리적으로 MI=0임을 뜻하지는 않는다. 재전송 보류/추정 유지 같은
  대안의 민감도 실험 없이는 현실적 최선의 동작이라고 주장하면 안 된다.
- **Noise:** `csi.py:118-131`, `phy.py:29-37`은 episode 전체 beam gain median으로
  noise를 보정한다. 이는 미래 channel 정보가 포함된 benchmark 운영 SNR
  정규화이며 고정 thermal-noise link budget이 아니다. `channel.py:80-81`의
  RBG-center single sample 또한 모든 RE의 frequency selectivity를 직접
  시뮬레이션하지 않는다. 짧은 smoke episode는 noise calibration 자체가 바뀌므로
  논문 성능 수치로 쓸 수 없다.
- **Reward/metric:** `env.py:311-342`는 useful MI shaping+completion−failure다.
  `transmission.py:211`의 partial MI는 packet가 나중에 실패해도 과거 reward에서
  취소되지 않는다. 이는 reward 설계이고 delivered goodput과 동일한 objective가
  아니다. 논문의 주 비교에는 completed-packet goodput, completion/failure,
  deadline 및 fairness를 함께 제시해야 한다. 에피소드 끝에 남은 packet도 따로
  계수하면 finite-horizon censoring의 크기를 볼 수 있다.

## 이번 검증·추가 산출물

- `work/paper_wireless_probe.py`: main saved cfg로 analytic 200 groups 확인.
  shared-path genie SINR max error `2.84e-14`, nonzero-group 총전력 오차
  `4.44e-16`; zero-cap closure 및 queue RNG counterexample 재현.
- `PaperMain/tests/test_wireless_main.py`: unittest 4개 통과. **독립 explicit
  matrix inverse** RZF oracle와 200개 그룹 비교, CQI floor의 바로 아래 boundary,
  zero-cap sampler/runtime 차이, pinned zero beam의 명시된 전력 동작.
  NumPy만 필요하며 channel/GPU training을 생성하지 않는다.
- `PaperMain/calibration/beta_m_calib_holdout.py`: 원본 파일을 바이트 그대로 보존.
- `PaperMain/paper_calibration.py`: 저장된 Base config로 과거 CQI4 sampler를
  재현하고 zero-cap/below-epsilon/group diagnostics를 새 results 폴더에 저장.
  원본/portable sampler의 sample sequence와 ratio arrays가 **deterministic mock
  16-slot episodes 2개**에서 elementwise exact인 것을 확인했다. 실제 Sionna
  full calibration 동등성/36+12 결과를 이번 검사로 주장하지 않는다.

## 논문 확정을 위한 남은 검증

1. 허용 가능한 전송 그룹으로 정의한 β calibration/holdout과 기존 frozen 정책
   + 강한 heuristic에 대한 민감도 평가.
2. 학습 normalizer를 동결한 공통 runner에서 세 recipe의 동일-protocol 비교;
   특히 NARROW D26은 새로 평가.
3. recipe 선택 후 미사용 test band와 여러 training seeds. 한 seed의 best policy
   비교만으로 학습 알고리즘 전반의 우월성을 주장하지 않음.
4. NARROW 주장을 실제 변경한 조건에 맞추고, 논문이 diversity의 인과 효과를
   주장한다면 별도 mean-matched control 추가.
5. 실제 NR/URLLC에 관한 강한 claim을 유지하려면 BLER/MCS/TBS, overhead 및
   CSI/HARQ timing sensitivity를 추가. 현재 단순화가 모두 정책 공통이라는
   이유만으로 PPO 이득의 보수적 하한이 된다고 결론내릴 수 없음.
