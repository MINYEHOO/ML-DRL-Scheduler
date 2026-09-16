# β_m (depth-wise calibration factor) — 도입 이유와 논문 서술

작성 2026-08-01. 코드·보정 산출물 직접 확인 기준.
핵심 근거 파일: `la_planner.py:26-60`, `Run4/_analysis/scripts/audit_probes/beta_m_calib_holdout.py`,
`Run4/AUDIT_CODE_CHANGES.md:116-137`, `Run4/_analysis/beta_m_calib_*.out`.

> 명칭 주의: **back-off가 아니라 depth-wise calibration factor**로 쓸 것.
> CQI4 세계 재보정에서 β₁ = 1.0018 > 1 이 나왔으므로 "back-off"는 모순이다.

---

## 1. 왜 필요한가 — 문제의 구조

### 1.1 전송 크기는 결과를 보기 전에 정해야 한다

우리 시스템은 슬롯마다 각 유닛의 전송 크기 B_tx를 **미리** 정한다:

```
B_tx = min( backlog,  β_m · η · N_RE · log₂(1 + SINR_pred) )      (la_planner.py:56-58)
```

그리고 ACK/NACK은 **약속 대비 달성**으로 판정한다:

```
ACK  ⟺  MI_actual ≥ B_tx        (달성 상호정보가 약속 비트를 덮으면 성공)
```

즉 B_tx는 "이번 전송에서 이만큼 나른다"는 **약속**이고, 과하게 약속하면 NACK →
재전송 → 자원 낭비 → deadline miss로 이어진다. 실제 시스템에서 MCS를 고르는 행위와
같은 자리다.

### 1.2 그런데 예측은 체계적으로 낙관적이다

`SINR_pred`는 BS가 가진 추정 채널 ĥ로 계산하지만, 실제 전송은 진짜 채널 h에서 일어난다
(`la_planner.py:12-15`: 예측은 h_hat, 평가는 h_true, **프리코더 함수·정규화·전력은 동일**).
ĥ ≠ h인 이유가 셋이다:

1. **PMI 양자화** — Type-II-like 56-bit 희소 코드북으로 방향을 근사
2. **CSI 소실·노화** — p_csi = 0.6의 Bernoulli 보고, 미보고 시 직전 값 유지(Age 증가)
3. (CQI4 세계 추가) **CQI 양자화** — 4-bit 사다리로 floor

ĥ가 틀리면 RZF 빔이 살짝 어긋나고, 그 결과 **① 원하는 신호가 줄고 ② 다른 스트림으로
누설이 생긴다**. 그래서 realized SINR < predicted SINR이 평균적으로 성립한다.

### 1.3 결정적으로, 이 낙관 편향은 depth에 따라 커진다

깊이 m에서 함께 전송되는 m−1개 스트림 각각이 어긋난 빔을 쏘므로, **잔여 간섭이 깊이에
비례해 쌓인다**. 우리가 따로 분해한 결과 depth-4의 잔여 간섭(~14 dB)은 대부분
"보고 오차(ĥ≠h)" 성분이었고, 이 성분은 RZF로 제거할 수 없다(RZF는 보고된 방향들
사이의 상관만 없앤다).

**따라서 보정 없이는 깊게 갈수록 더 심하게 과약속하고, 실패율이 깊이에 따라 벌어진다.**

---

## 2. 실제 시스템은 이걸 어떻게 푸는가 — 그리고 우리는 왜 β_m인가

실제 gNB는 **link adaptation + OLLA(outer-loop link adaptation)** 로 푼다: CQI로 MCS를
고르고, ACK/NACK 통계를 보며 SINR 오프셋을 동적으로 ±조정해 목표 BLER(보통 10%)에
수렴시킨다.

우리는 MCS 사다리와 동적 OLLA 루프를 모델링하지 않는다(연속 rate 약속을 쓴다).
**β_m은 그 자리를 대신하는 정적 대리(static surrogate)** 다:

| 실제 시스템 | 본 연구 |
|---|---|
| CQI 보고 (4-bit 사다리) | 동일 (CQI4 세계) / 연속 SE (기본 세계) |
| MCS 선택 (32단 사다리) | 연속 rate 약속 B_tx |
| **OLLA 오프셋 (동적, ACK/NACK 피드백)** | **β_m (정적, 사전 보정)** |
| 목표 BLER 10% | 목표 first-ACK ≈ 90% |

**논문 문장**:
> "Transport-format granularity (MCS) and the outer-loop link-adaptation offset are
> abstracted by a continuous rate promise scaled by a depth-wise calibration factor β_m,
> which plays the role of a static OLLA operating point: it is calibrated once, offline,
> to a first-transmission success target of approximately 90% (i.e. ~10% first-transmission
> BLER, the conventional OLLA target), rather than being adapted online from ACK/NACK
> feedback. Online OLLA is left to future work."

### 왜 정적으로 충분한가 (리뷰어 예상 질문)

- 오차의 원인이 **구조적**(양자화 해상도, 보고 주기, depth별 간섭)이라 에피소드마다
  통계가 크게 변하지 않는다. 실제 OLLA가 쫓는 것은 주로 이런 정상(定常) 편향이다.
- 동적 OLLA를 넣으면 **스케줄러의 depth 선택과 LA 루프가 상호작용**해서, 관측된 성능
  차이가 "정책이 좋아서"인지 "LA 루프가 그 정책에 잘 붙어서"인지 분리가 어려워진다.
  정적 β_m은 **모든 스케줄러에 동일하게 고정**되므로 비교가 깨끗하다.
- 한계는 정직하게 밝힌다: 시변 부하·이동성에서 online OLLA가 줄 수 있는 추가 이득은
  본 연구가 측정하지 않았다.

---

## 3. 왜 depth별로 나눴는가 — global β를 쓰지 않은 이유 (핵심 논거)

이게 β_m 설계의 진짜 요점이다.

**만약 모든 깊이에 하나의 스칼라 β를 쓰면**, §1.3 때문에 first-ACK 성공률이 깊이마다
달라진다. 얕은 그룹은 과도하게 보수적이 되고(약속을 덜 해서 자원 낭비), 깊은 그룹은
여전히 과약속한다. 그러면:

> **정책이 보는 "깊이의 가격"에 물리가 아닌 보정 아티팩트가 섞인다.**

즉 학습된 스케줄러가 depth를 고르는 이유가 "물리적으로 그게 유리해서"가 아니라
"우리가 그 depth를 잘못 보정해서"가 될 수 있다. 이건 논문의 핵심 주장(정책이 세계의
좌석 가격에 맞춰 depth를 조절한다)을 통째로 무너뜨리는 교란이다.

**β_m은 모든 depth에 동일한 first-ACK 목표(~90%)를 부과**해 이 교란을 제거한다.
depth 결정이 순수하게 물리(전력 분할 vs 다중화 이득 vs 잔여 간섭)에 의해서만 내려진다.

### 실측 근거 (round-6 감사)

- 스칼라 β = 0.6469를 쓰면, m = 1에서 예측 post-RZF SINR이 피드백 SNR과 **정확히 같아지는**
  구조 때문에(`reconstruct_h_hat`의 정의상) cap-limited m=1 천장이 정확히 **35.31%** 잘린다.
  그 결과 SU+CQI가 1246으로 붕괴했다 (`Run4/AUDIT_CODE_CHANGES.md:118-123`).
- 즉 global β는 "보수적"이 아니라 **깊이 축을 왜곡**한다. depth-1 스케줄러를 부당하게
  불리하게 만들고, 그 상태로 "MU가 SU보다 낫다"고 주장하면 그건 우리 보정이 만든 결론이다.
- global β 버전은 **ablation으로 코드에 보존**되어 있다(`config.la_beta`), 삭제하지 않았다.

---

## 4. 어떻게 보정했는가 — 재현 가능한 동결 레시피

`beta_m_calib_holdout.py` (자기재현 스크립트, 실행하면 전 과정 재계산):

1. **표본 수집 (scheduler-independent)**: 에피소드 36개(50000–50035)를 **빈 할당으로
   진행**시키면서, 7슬롯마다 (RBG, depth m, UE 집합)을 **균등 무작위**로 뽑아
   `MI_actual / cap_pred` 비율을 수집한다.
   - 어떤 스케줄러도 개입하지 않으므로 **특정 정책에 유리하게 보정될 여지가 없다**.
   - 표본이 전부 full-cap이라 `first-ACK ⟺ ratio ≥ β_m`이 정확히 성립한다.
2. **β_m = 각 depth의 10th percentile** → 그 depth에서 약 90%가 첫 시도에 성공.
3. **홀드아웃 검증**: 완전히 분리된 에피소드 12개(70000–70011)에서 β_m을 **동결**한 채
   first-ACK 실측.
4. **에피소드-클러스터 부트스트랩 CI** (B=10,000): 같은 에피소드 내 표본은 채널 혼합
   (속도·활성 인원)을 공유하므로 member-level 이항 CI는 정밀도를 과대평가한다
   (m=1에서 1.2배, m=2–4에서 3.3–4.4배). **정직한 CI는 클러스터 부트스트랩**이다.

### 보정 결과 (세계별)

| 세계 | β₁ | β₂ | β₃ | β₄ | 홀드아웃 first-ACK |
|---|---|---|---|---|---|
| Queue U(5,30) | 0.9815 | 0.7306 | 0.6466 | 0.5922 | 88.8 / 93.5 / 94.2 / 94.8 % |
| Queue S40 U(5,40) | 0.9757 | 0.7180 | 0.6355 | 0.5837 | 89.3 / 93.5 / 94.2 / 94.6 % |
| **S40HL + CQI4** | **1.0018** | **0.7499** | **0.6592** | **0.6058** | 86.9 / 88.6 / 89.0 / 88.4 % |
| genie (perfect CSI) | 1.0 | 1.0 | 1.0 | 1.0 | 100% (구조적) |

읽는 법:
- **β가 depth에 따라 단조 감소**한다(≈1.0 → 0.58~0.61). 깊이가 붙일 수 있는 약속이
  줄어든다는 물리가 숫자로 나온 것 — 이 자체가 논문에 쓸 만한 관측이다.
- **세계가 바뀌면 재보정한다.** 속도 상한을 40으로 넓히자 전 depth에서 β가 소폭 하락
  (채널이 빨리 늙어 예측이 더 낙관적이 됨). CQI4에서는 반대로 상승 — floor 양자화가
  이미 보고 단계에서 보수성을 넣으므로 β가 추가로 깎을 몫이 줄어든다.
- **β₁ = 1.0018 > 1** (CQI4): depth 1에서는 양자화된 CQI 기반 예측이 실제 달성보다
  이미 낮다는 뜻. 그래서 명칭이 back-off가 아니라 calibration factor여야 한다.
- **genie는 β = 1**: ĥ = h이므로 예측 = 실현, 보정할 오차가 없다
  (`la_planner.py:12-15`, Gate 1). 여기에 β_m을 쓰면 근거 없이 과소약속이 된다.

---

## 5. 공정성 — 리뷰어가 반드시 물어볼 것

> "PPO에 유리하게 튜닝한 손잡이 아닌가?"

아니다. 세 겹으로 막혀 있다:

1. **보정이 scheduler-independent**다: 빈 할당으로 진행하며 무작위 그룹을 뽑는다.
   어떤 정책의 행동도 보정에 들어가지 않는다.
2. **모든 스케줄러가 같은 β_m을 쓴다.** PPO도, SUS+CQI도, SU+CQI도, PF/MW/EDF도
   동일한 `la_planner.predict_group_link_adaptation`을 호출한다
   (`la_planner.py:4-6`: env·baselines·PPO decoder가 공유하는 single source of truth).
3. **계획과 실제가 일치함이 게이트로 검증**된다(audit Gate 2): 스케줄러가 계획한 커밋과
   env가 실제로 커밋한 값이 정확히 같다.

> "왜 10th percentile인가?"

실제 LA의 관례적 목표(첫 전송 BLER 10%)에 대응시킨 것이다. 정확히 10%가 아니라
**"약 10% 첫전송 BLER를 목표로 독립 보정한 depth별 마진"** 이라고 서술해야 정확하다
(`AUDIT_CODE_CHANGES.md:127-129`의 표현).

---

## 6. 논문에 넣을 최소 서술 (복사용)

> Because the transmission size must be committed before the channel is realized, and the
> base station only holds a quantized, possibly stale estimate ĥ, the predicted post-RZF
> SINR is systematically optimistic, and increasingly so with multiplexing depth: each
> additional co-scheduled stream contributes precoding leakage that regularized
> zero-forcing cannot null, since it originates from the report error ĥ ≠ h rather than
> from correlation among the reported directions. We therefore size each new unit as
> B_tx = min(backlog, β_m · η · N_RE · log₂(1 + SINR_pred)), where β_m is a depth-wise
> calibration factor. β_m is obtained offline and independently of any scheduler: episodes
> are advanced with empty allocations, groups of size m are drawn uniformly at random, and
> β_m is set to the 10th percentile of the realized-to-predicted mutual-information ratio,
> targeting approximately 10% first-transmission BLER — the conventional outer-loop
> link-adaptation operating point. A single global β would impose a depth-dependent success
> rate and thus embed a calibration artifact into the very quantity the scheduler optimizes
> (the cost of depth); the depth-wise factors equalize the first-transmission target across
> spatial modes so that the depth decision reflects the physics alone. The factors are
> frozen and validated on disjoint episodes with episode-cluster bootstrap confidence
> intervals, and are shared identically by the learned policy and every baseline. β_m thus
> serves as a static surrogate for outer-loop link adaptation; online OLLA is left to
> future work.

---

## 7. 알아둘 한계 (정직하게)

- **정적이다.** 실제 OLLA의 동적 추종 이득은 측정하지 않았다.
- **세계별 재보정이 필요하다.** K, 속도 분포, CQI 해상도가 바뀌면 다시 뽑아야 한다.
  zero-shot 실험(K=16~48, OOD 6세계)에서는 **K=32 보정값을 고정**한 채 평가했으므로,
  그 세계들에서는 β_m이 최적이 아니다 — 즉 그 결과는 **보수적 하한**이다. (K48에서
  재보정하면 더 좋아질 여지가 있다는 각주를 달 것.)
- **`beta_rate`**(별개 스칼라, 항상 1.0)는 Phase-1 잔재이며 β_m이 그 역할을 대체했다.
  paper-gen 배치에서 코드에서 제거 예정 — 논문에는 등장시키지 말 것.
