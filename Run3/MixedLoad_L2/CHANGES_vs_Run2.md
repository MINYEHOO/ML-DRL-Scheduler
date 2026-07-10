# MixedLoad_L2 — Run2 대비 변경사항

작성 2026-07-02. "Run2_HardMain 기준으로 무엇을 왜 바꿔 실행한 run인지"의 기록.

## Run2 기준점 (Run2_HardMain)

- K=16 **고정 인원**, episode 1000, p_arrival 0.4, deadline U[3,12], 속도 30 균일, p_csi 0.2
- 결과: PPO near-SU 학습(+50%). 단 한 regime에 특화된 정책 — regime이 바뀌면?

## 이 run의 변경점

| 항목 | Run2 | 이 run | 이유 |
|---|---|---|---|
| **혼잡도** | K=16 항상 전원 | **K=32, 에피소드마다 n_active ~ U[16,32] 추첨** (`--num_ue 32 --n_active_min 16 --n_active_max 32`) | **핵심 실험 변수** — Level-2: 한 정책이 에피소드별 혼잡도에 온라인 적응해야 함 |
| UE 속도 | 30 균일 | **15 균일** (`--ue_speed_kmh 15`) | 속도축은 고정(MixedSpeed와 분리), 중간값 |
| p_csi | 0.2 | 0.6 | hetero 프리셋 |
| entropy | 0.01 | 0.02 | 탐색 강화 |
| early-stop / 장치 | 없음 / GPU3 | 비활성 / **GPU 5** (CPU 포화로) | — |
| 신규 코드 | — | `n_active_min/max` (config/env/traffic, 2026-07-01) — 에피소드 seed로 n_active 추첨, 나머지 UE idle | — |

실행 커맨드:
```
python3 train_phase2.py --mode hetero --ue_speed_kmh 15 --entropy_coef 0.02 \
  --num_ue 32 --n_active_min 16 --n_active_max 32 \
  --patience_evals 100000 --seed 2024 [--resume ...]
```

## 목적 / 가설

단일 regime run들의 종합 교훈: **K16 최적=SU-CQI, K32 최적=MU-4(SUS-CQI) — 어떤 고정 heuristic도 양쪽을 다 못 맞춤.** 혼잡도가 에피소드마다 바뀌는 환경에서는 상태를 보고 SU↔MU를 오가는 학습 정책만이 전 구간 최적 → PPO의 구조적 우위가 나오는 무대라는 가설.

## 경과 (2026-07-02 기준, ~update 139+/1500) — **현재 헤드라인 결과**

- Eval: PPO 9920(최근)/10071(best.pt=update 119)
- 이 run은 신 baseline 세트가 CSV에 있음(fresh run 첫 eval 기록): SUS-CQI 8809, CQI-greedy 8577, SU-CQI 7877 등 12종
- 2026-07-02 오프라인 threshold 스윕: **SUS-CQI@0.8 = 9006** (0.5보다 강함) → **PPO +10~12% 승리, 가설 입증**
- mu_depth ~3.7 유지하며 안정 상승 중 (MixedSpeed와 달리 붕괴 없음 — KL 스파이크는 있었으나 회복)
- 잘 가고 있어 2026-07-02 개선 패키지(KL guard/critic v2)는 **적용하지 않음** (진행 중 조건 변경 회피; L2b 파일럿 검증 후 최종 레시피 통일 여부 결정)

## 공통 사항

- Eval seed 10000-10002 ×3, 10 update마다
- 채널/PHY/reward Run2와 동일 (n_active 추첨만 추가)

## Hybrid / oracle-envelope 방어 실험 (2026-07-02, seeds 10000-10002)

리뷰어 반론("적응은 switch 한 줄이면 충분") 선제 검증:
pure SU-CQI 7877 | pure SUS-CQI@0.8 9006 | 실현가능 hybrid 최강(관측 backlog-UE-수
스위치, T=18) 9035 | **oracle per-episode envelope 9563** | **PPO(best.pt upd 119) 10073**
→ **PPO가 최강 hybrid +11.5%, oracle envelope마저 +5.3% 상회.** 이득은 모드 선택이
아니라 에피소드 내부(triage·선택적 pairing)에서 발생 — 고부하 seed에서 pure SUS
홈그라운드 대비 +9%. 관측가능 신호(backlog UE 수)로는 스위치가 사실상 무력
(에피소드 n_active는 관측 불가).
⚠️ 3-seed 결과 — 학습 종료 후 20-seed로 확정 예정.
원자료: scratchpad/hybrid_switch_eval.{py,out}

## 종료 (2026-07-06)

update ~1057/1500에서 **수동 종료** (사용자 승인). 사유: 무가드 구간(전 구간)에서
KL 스파이크(0.199@364, 0.235@385 등)가 반복되는 동안 entropy가 0.84→0.09로 소진 →
eval이 ~8000까지 열화(bar 9006 하회)하고 회복 불능 정체. **best.pt(10087@219)는
안전 보존 — 논문 수치는 이것의 20-seed 재평가로 확정** (final20 평가 참조).
교훈(Run4 레시피行): KL guard 기본 장착 + entropy floor 또는 patience 조기종료
(patience-15였으면 update ~370에서 10087을 들고 종료했을 것). CSV 사용 시 주의:
update 130-138 중복 행(재시작 overlap) dedupe 필요.

## 최종 확정 (2026-07-06, 20 seeds 10000-10019 paired) — 논문 수치

PPO-best(10087@219) 9560±2095 | oracle envelope 8989 | 최강 hybrid(T=14) 8768 |
SUS-CQI@0.8 8550 | SU-CQI 8095.
**PPO − oracle envelope = +570.5 (+6.35%), CI[+338,+803], 17/20승 → 유의한 승.**
vs 최강 hybrid +9.03% (18/20), vs SUS-CQI@0.8 +11.81% (**20/20 전승**).
3-seed 추정(+5.3%)보다 강화됨. 원자료: _analysis_20260702/final20_MixedLoad_L2.csv
