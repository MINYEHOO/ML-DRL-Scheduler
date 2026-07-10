# Run3 Baseline 감사 (2026-07-02)

"PPO가 모든 baseline보다 낫다"의 비교 기준이 공정·최강이었는지에 대한 전수 감사 기록.
per-episode 원자료: scratchpad `stats20*_*.csv`, `hybrid_switch_eval.out`,
`sus_family_08_screen.out`, `stats_sus05_ScarcityK32.csv` (세션 종료 대비 사본 권장).

## 1. Regime별 챔피언 (통계 비교의 기준선)

| Run | 챔피언 | 근거 |
|---|---|---|
| Uniform10, DeadlineScarcity (K16) | **SU-CQI** | 3-seed 전수에서 모든 MU 계열에 +10% 이상 |
| ScarcityK32, MixedLoad, MixedSpeed (K32) | **SUS-CQI@0.8** | threshold sweep 0.5 < **0.8** > 0.9 |

## 2. SUS threshold 확정

- L2 sweep (3-seed): MixedSpeed 8740(@0.5) / **8939(@0.8)** / 8805(@0.9); MixedLoad 8904 / **9006** / 8754
- ScarcityK32 paired 40-seed: **@0.8 − @0.5 = +158.4, CI[+114, +203], 36/40승** → K32에서도 0.8이 최적
- 해석: 0.8은 "나쁜 pair만 걸러내고 depth(3.96+)는 유지"하는 sweet spot; 0.9부터 DoF 손실

## 3. SUS 계열 전 변형 @0.8 스크린 (3-seed, 5개 config 전부)

SUS+PF@0.8 / SUS+vPF@0.8 → 챔피언 대비 **−10.0% ~ −21.5%** (10개 측정 전부).
3% 승격 기준에 어떤 것도 미달 → 챔피언 세트 유효 확정.
참고: threshold 0.8은 PF 계열에도 이득(예: MixedLoad SUS+PF 7170→8110)이나 격차 불변.

## 4. vPF 판정 (사용자 알고리즘 변형, 솔직 평가)

| MixedLoad 3-seed | reward | comp | Jain(전체 K) |
|---|---|---|---|
| SUS+PF@0.5 / vPF@0.5 | 7207 / **7741 (+7.4%)** | 0.765 / 0.783 | 0.581 / 0.585 |
| SUS+PF@0.8 / vPF@0.8 | **8110** / 8044 (−0.8%) | 0.798 / 0.796 | 0.594 / 0.594 |
| SUS-CQI@0.8 | 9006 | 0.827 | **0.606** |

- vPF의 in-slot 분산은 loose threshold(0.5)에서만 유효(+7.4%); 최적 threshold(0.8)에선
  orthogonality 필터가 이미 다양성을 강제해 rate 비용만 남음(전 config −0.3~−2.2%)
- **Jain 개선도 없음** → 이 환경(one-packet + deadline)에서는 메인 비교 제외, appendix 1줄.
  코드는 유지 (Run4 큐잉 세팅에서 부활 가능성)

## 5. "왜 rate-greedy가 PF보다 JFI가 높은가" (사분위 실측, MixedLoad seed 10001/10002)

| | Q1(최약) | Q2 | Q3 | Q4 | 총량 | active-Jain |
|---|---|---|---|---|---|---|
| SUS-CQI@0.8 | 2.31Mb/0.19c | 10.41/0.88 | 16.04/1.00 | 15.19/1.00 | 44.0Mb | 0.764 |
| SUS+PF@0.8 | 2.34Mb/0.19c | 9.06/0.82 | 15.20/0.99 | 15.05/1.00 | 41.7Mb | 0.753 |

PF는 Q1에 +1%밖에 못 주고(comp 개선 0) Q2-Q3에서 −10%를 뺏음. 메커니즘:
(a) one-packet 모델이 강자 독식을 원천 차단(완료 즉시 pool 이탈 = cycling),
(b) 약자의 병목은 우선순위가 아니라 deadline×물리(양쪽 comp 0.19 동일),
(c) **도착이 idle UE에게만 오므로 약자는 수요 자체가 봉쇄**됨(PF가 줄 것이 없음).
→ 이 트래픽 모델에서 JFI는 슬롯 배분이 아니라 "완료 폭"의 함수. 완료율 1등이 JFI 1등.

## 6. Run4 예측 (multi-packet queue 도입 시) — 검증 가설로 이월

```
one-packet + deadline (Run3)      : greedy ≥ PF   (완료 cycling이 fairness 담당)
queue + deadline (Run4/Stage C)   : PF ≳ greedy   (약자 수요 노출 → debt 레버 작동; deadline이 잔여 제약)
queue + no-deadline (full-buffer) : PF ≫ greedy   (교과서 복원)
```
Run4 필수 추가 baseline: **MaxWeight(queue×rate)**, EDF; PF/vPF 재평가.
관측공간(큐 길이/패킷별 deadline) 변경 → 네트워크 warm-start 불가, fresh phase.

## 7. 최종 확정 결과 (2026-07-06 추가) — Run3 완결

혼합 regime 20-seed paired 확정: **MixedLoad PPO − oracle envelope = +6.35%
CI[+338,+803] (17/20), MixedSpeed_L2b = +6.88% CI[+410,+824] (16/20)** — 두 축
모두 "에피소드별 완벽 스위치" 상한을 유의하게 초과. vs SUS-CQI@0.8은 양쪽 모두
20/20 전승 (+11.8%/+12.9%). Run3 전체 판정: 단일 regime {동률, +2.62%, +1.71%}
+ 혼합 regime {envelope +6.35%, +6.88%}. 원자료: _analysis_20260702/final20_*.csv

## 8. 공식 baseline 세트(2×4 그리드) 최종 결과 (2026-07-06, 20 seeds paired)

사용자 지정 8종 — {SUS@0.8-MU | SU} × {CQI, DPF, PF, Rnd} — 전부에 대해 **PPO가
paired 유의승, 16개 비교(8종×2 run) 전부 CI가 0 제외**:
- 최소 마진: vs SUS+CQI +11.8%/+12.8% (20/20, 20/20)
- vs SU+CQI +18.1%/+18.8% (17/20, 16/20)
- 나머지 6종: +20%~+208%, 전부 20/20 전승
- 8종 전원을 한 seed 안에서 이긴 clean sweep: 17/20, 16/20 (놓친 seed는 전부
  저부하 16-19명 — 승자는 SU+CQI)
- 부수 관찰: SUS+DPF ≈ SUS+PF (SUS 게이트 하에선 deadline 가중이 무익),
  SU+DPF > SU+PF (SU에선 유익) — metric 효과가 공간 모드와 상호작용함
병합 원자료: _analysis_20260702/official9_{MixedLoad_L2,MixedSpeed_L2b}.csv
Hybrid/envelope 결과(§7)는 공식 baseline이 아닌 별도 방어 실험으로 분리 표기.

## 9. JFI 정의 수정 (2026-07-06, 사용자 확정) — active 유저 기준으로 전면 재계산

**수정**: JFI는 그 에피소드에서 트래픽을 받은 active 유저만으로 계산
(jains_index(acked_per_ue[:n_active])). 기존 전체-K 계산은 트래픽 없는 유저의 0이
분모에 들어가 지표를 (n_active/K)·상한으로 압축시키는 오류 — 예: 16명 활성 에피소드
에서 상한 0.5. 다른 8개 지표는 전수 감사 결과 오염 없음(총합·이벤트비율·scheduled-only
평균 구조라 inactive가 낄 자리 없음). §4-5의 Jain 수치는 전체-K 기준 진단값이었음.

**수정 후 20-seed 평균 (active-only)**: MixedLoad — SUS+CQI 0.733, SU+Rnd 0.731,
SUS 계열 0.72x, **PPO 0.699**, SU+DPF 0.684, SU+PF 0.673, **SU+CQI 0.589(최저)**;
MixedSpeed_L2b 동일 패턴 (PPO 0.686). 판독: PPO는 SUS 계열과 동급 범위(−0.03),
rate-greedy SU의 불공평(0.59)이 이제 명확히 드러남. Random 계열의 높은 JFI는
하향 평준화(§5 참조). 원자료: official9full_*.csv (jain=active, jain_allk=audit).
reward 등 타 지표는 재평가에서 바이트 동일 재현 확인.
