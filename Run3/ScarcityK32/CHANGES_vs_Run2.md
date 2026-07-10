# ScarcityK32 — Run2 대비 변경사항

작성 2026-07-02. "Run2_HardMain 기준으로 무엇을 왜 바꿔 실행한 run인지"의 기록.

## Run2 기준점 (Run2_HardMain)

- K=16, episode 1000, p_arrival 0.4, deadline U[3,12], 속도 30 균일, p_csi 0.2, entropy 0.01
- 결과: PPO near-SU 학습(+50%), 단 K=16에선 8개 RBG의 SU 공급이 수요를 감당 → 혼잡 없음

## 이 run의 변경점 (= Uniform10_Ent002 설정 + K만 변경)

| 항목 | Run2 | 이 run | 이유 |
|---|---|---|---|
| **num_ue** | 16 | **32** (`--num_ue 32`) | **유일한 실험 변수** — 사용자 수 희소성 |
| p_csi / 속도 / entropy | 0.2 / 30 / 0.01 | 0.6 / 10 균일 / 0.02 | Uniform10_Ent002와 동일 (깨끗한 귀속을 위해) |
| early-stop / 장치 | 없음 / GPU | 비활성 / **CPU** | 위와 동일 |

실행 커맨드:
```
python3 train_phase2.py --mode hetero --ue_speed_kmh 10 --entropy_coef 0.02 \
  --num_ue 32 --patience_evals 100000 --seed 2024 [--resume ...]
```

## 목적 / 가설

유효 공급 ≈ 8 SU 슬롯/slot인데 수요를 K=32로 올리면(≈4:1 경쟁) 스케줄링 "선택"이 중요해져 학습이 heuristic을 이길 것이라는 가설. K↑는 multi-packet queue(Stage C) 없이도 즉시 혼잡을 만드는 지름길.

## 경과 (2026-07-02 기준, ~update 279)

- **PPO가 depth ~1 → 4.0으로 스스로 전환** (update 9: 0.98 → 19: 4.00) — K=16에선 SU, K=32에선 MU를 택하는 **맥락 적응 학습의 직접 증거** (고정 baseline은 항상 ~4)
- Eval: PPO 10142(최근)/10278(best) vs CQI-greedy 9778 (+3.6%)
- ⚠️ 2026-07-02 오프라인 측정: **SUS-CQI@0.8 = 10179** → PPO와 **사실상 동률** (−0.4%/+1.0%)
- 해석: K32 전원-active regime의 최적해는 "항상 MU-4" = 고정 heuristic이 이미 하는 것 → **적응의 여지가 구조적으로 없어 PPO 우위가 안 나옴**. 논문에서 이 run은 "PPO가 regime별 최적 depth를 찾는다"의 증거로 쓰고, 우위 주장은 Mixed 계열(L2)로 할 것
- 같은 이유로 2026-07-02 개선 패키지(KL guard/critic v2)를 이 run엔 적용하지 않음 (headroom 없음)

## 공통 사항

- Eval seed 10000-10002 ×3, 10 update마다; baseline CSV는 구 5종 세트 (SUS-CQI/SU-* 없음 — bar는 위 오프라인 측정 참조)
- 채널/PHY/reward Run2와 동일

## 종료 (2026-07-02)

update 288/1500에서 **수동 종료** (사용자 승인): best 10278@update 49 이후 240 update
정체 + 구조적 동률 regime(적응 여지 없음). best.pt/latest.pt 보존. watchdog 제거.

## 20-seed 통계 재평가 (2026-07-02, seeds 10000-10019 paired) — 판정 수정

PPO-best 10566±1213 | SUS-CQI@0.8 10361±1333 | CQI-greedy 9950 | SU-CQI 8360.
**PPO − SUS-CQI@0.8 = +205.0 (95% CI ±184.9), 14/20승, +1.98% — "구조적 동률"
판정을 수정: 작지만 유의한 승리** (selection-seed 제외 17 seed CI[+53,+422]).
3-seed(10000-2)가 PPO에 불리한 추첨이었음. MU-4 강제 regime에서도 deadline
triage 몫의 +2%는 남는다는 해석.

## 최종 판정 (2026-07-02, 40 seeds 10000-10039)

**PPO − SUS-CQI@0.8 = +178.6 (+1.71%), CI[+35.2, +321.9], 26/40승 → 유의한 승 확정**
(표본 2배 확대에서 생존; Uniform10과 달리 견고). vs CQI-greedy +613.9 (+6.12%),
40/40승. 해석 유지: 강제-MU regime에서도 deadline triage 몫 ~1.7%는 학습만 취득.
원자료: scratchpad/stats20_ScarcityK32.csv + stats20b_ScarcityK32.csv
