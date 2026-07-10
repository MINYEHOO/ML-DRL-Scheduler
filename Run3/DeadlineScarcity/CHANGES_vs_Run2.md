# DeadlineScarcity — Run2 대비 변경사항

작성 2026-07-02. "Run2_HardMain 기준으로 무엇을 왜 바꿔 실행한 run인지"의 기록.

## Run2 기준점 (Run2_HardMain)

- K=16, episode 1000, p_arrival 0.4, **deadline U[3,12]**, 속도 30 균일, p_csi 0.2, entropy 0.01
- 결과: PPO near-SU 학습(+50%). 단 deadline이 느슨해 urgency 우선순위의 가치가 작았음

## 이 run의 변경점 (= Uniform10_Ent002 설정 + deadline만 변경)

| 항목 | Run2 | 이 run | 이유 |
|---|---|---|---|
| **deadline** | U[3,12] | **U[2,6]** (`--deadline_min 2 --deadline_max 6`) | **유일한 실험 변수** — 긴급성 희소성 |
| p_csi / 속도 / entropy | 0.2 / 30 / 0.01 | 0.6 / 10 균일 / 0.02 | Uniform10_Ent002와 동일 |
| early-stop / 장치 | 없음 / GPU | 비활성 / **CPU** | 위와 동일 |

실행 커맨드:
```
python3 train_phase2.py --mode hetero --ue_speed_kmh 10 --entropy_coef 0.02 \
  --deadline_min 2 --deadline_max 6 --patience_evals 100000 --seed 2024 [--resume ...]
```

## 목적 / 가설

K-희소성(ScarcityK32)의 교훈("모두 MU → PPO 우위 없음") 이후, **우위가 나올 축은 deadline 긴급성**이라는 가설. 마감이 빠듯하면 rate-greedy가 아니라 "어느 패킷을 지금 살릴지"의 triage가 승부처가 되고, 이는 reward(λc, λm)를 직접 보는 학습 정책의 영역.

## 경과 (2026-07-02 기준, ~update 499)

- Eval: PPO 6758(최근)/6900(best) vs CQI-greedy 5106 (+32%), 구 세트 전부에 큰 폭 우위
- ⚠️ 2026-07-02 오프라인 측정: **SU-CQI = 6678** (진짜 bar), SUS-CQI@0.8 = 5486 → PPO **+1.2%(최근)/+3.3%(best)로 승리 유지** — 폭은 구 세트 대비 크게 줄지만 이기고는 있음
- 해석: 빠듯한 deadline에서 PPO의 urgency-aware triage가 rate-greedy SU를 근소하게 넘음. K16 계열 중 유일하게 ceiling을 넘는 run

## 공통 사항

- Eval seed 10000-10002 ×3, 10 update마다; baseline CSV는 구 5종 세트 (bar는 위 오프라인 측정 참조)
- 채널/PHY/reward Run2와 동일

## 종료 (2026-07-02)

update 556/1500에서 **수동 종료** (사용자 승인): best 6900@update 369 이후 ~190 update
정체. best.pt/latest.pt 보존. watchdog 제거. 논문 수치는 best.pt + 20-seed 재평가 사용.

## 20-seed 통계 재평가 (2026-07-02, seeds 10000-10019 paired)

PPO-best 6875±991 | SU-CQI 6677±739 | SUS-CQI@0.8 5420±1180.
**PPO − SU-CQI = +198.6 (95% CI ±155.5), 14/20승, +2.97% — 유의한 승리 확정**
(selection-seed 제외 17 seed에서도 CI[+13,+376]로 유지).

## 최종 판정 (2026-07-02, 사전 선언 n=60, seeds 10000-10059)

**PPO − SU-CQI = +169.8 (+2.62%), CI[+82, +258], 38/60승 → 유의한 승 확정.**
(n=20: +199 유의 → n=40: +97 경계 → n=60 사전선언 최종: +170 유의 — optional-stopping
방지를 위해 n=60을 미리 선언하고 확정.) vs SUS-CQI@0.8 +1371 (+25.9%), 60/60승.
평균: PPO 6659 vs SU-CQI 6489. 원자료: scratchpad/stats20{,b,c}_DeadlineScarcity.csv
