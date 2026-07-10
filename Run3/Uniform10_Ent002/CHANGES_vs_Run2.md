# Uniform10_Ent002 — Run2 대비 변경사항

작성 2026-07-02. 이 파일은 "Run2_HardMain 기준으로 무엇을 왜 바꿔 실행한 run인지"의 기록.

## Run2 기준점 (Run2_HardMain, 2026-06-15~23)

- `--mode hard_main`: K=16, episode 1000슬롯, 2000 updates, GPU 3
- p_arrival 0.4, deadline U[3,12], **UE 속도 30 km/h 균일**, **p_csi 0.2**
- entropy_coef 0.01(기본), value_coef 0.25, minibatch 256, seed 2024
- 결과: stale CSI(p=0.2)에서 MU가 유해 → PPO가 **near-SU(depth 1.11)를 스스로 학습**, heuristic 대비 +50%

## 이 run의 변경점

| 항목 | Run2 | 이 run | 이유 |
|---|---|---|---|
| p_csi | 0.2 | **0.6** (hetero 프리셋) | staleness를 완화한 중간 지점에서의 행동 확인 |
| UE 속도 | 30 균일 | **10 균일** (`--ue_speed_kmh 10`) | 채널 변화 완만 → CSI 신뢰도 ↑ |
| entropy_coef | 0.01 | **0.02** (`--entropy_coef 0.02`) | 탐색 강화 (0.05는 Ent005에서 발산 → 2×만) |
| early-stop | 없음(2000 고정) | patience **비활성** (`--patience_evals 100000`) | 충분히 길게 관찰 |
| 실행 장치 | GPU 3 | **CPU** (2026-06-30 전환; latency-bound라 CPU가 ~2× 빠름) | GPU 확보 + 속도 |
| 인프라 | nohup | auto-resume wrapper + tmux + `_watchdog.sh` | 컨테이너 재시작 대비 |

실행 커맨드:
```
python3 train_phase2.py --mode hetero --ue_speed_kmh 10 --entropy_coef 0.02 \
  --patience_evals 100000 --seed 2024 [--resume Run3/Uniform10_Ent002/ckpt/latest.pt]
```

## 목적 / 가설

Run2 후속 논쟁 "PPO가 near-SU에 머문 건 물리적 최적이라서인가, 탐색 부족(local minimum)인가?"의 직접 검증. 더 신선한 CSI(p=0.6, 10km/h) + 2× 탐색에서 MU(depth>1.3)를 배우면 탐색 부족설이 맞고, 여전히 near-SU면 물리적 최적설 확정. (Ent005: entropy 0.05는 gradient를 압도해 발산 → 0.02로 재시작한 run.)

## 경과 (2026-07-02 기준, ~update 529/1500)

- PPO는 **near-SU(depth ~1.03)로 수렴** → 물리적 최적설 확정 (depth-cap/p_csi 스윕과 일치)
- Eval: PPO 7219(최근)/7405(best) vs 구 baseline 세트 최강 SUS+PF 6093 (+20%)
- ⚠️ 단 2026-07-02 오프라인 측정으로 **진짜 bar는 SU-CQI 7325** (구 eval 세트에 없음) → PPO는 사실상 **K16 ceiling과 동률** (best +1.1%). SUS-CQI@0.8도 측정: 6332 (K16에선 MU 계열이라 약함)
- 결론적 위치: "K=16 비혼잡 regime에선 SU-CQI가 ceiling이고 PPO가 거기 도달한다"의 증거 run

## 공통 사항

- Eval: held-out seed 10000-10002, 3 에피소드, 10 update마다 (PPO만; baseline은 fresh run 첫 eval에만 기록 — 이 run은 신 baseline 세트(SUS-CQI/SU-*) 추가 **이전**에 시작돼 CSV에 구 5종만 있음)
- 채널/PHY/reward는 Run2와 동일 (TR38.901 UMi, Type-II 56bit, RZF, 10dB, λs/λc/λm=1/1/2)

## 종료 (2026-07-02)

update 560/1500에서 **수동 종료** (사용자 승인): best 7405@update 279 이후 280 update
정체 → patience-15였다면 이미 자동 종료됐을 상태. best.pt/latest.pt 보존.
watchdog 목록에서 제거됨. 논문 수치는 best.pt 기준 + 20-seed 재평가(진행 중) 사용.

## 20-seed 통계 재평가 (2026-07-02, seeds 10000-10019 paired)

PPO-best 7358±762 | SU-CQI 7280±648 | SUS-CQI@0.8 6229±1041.
**PPO − SU-CQI = +79.0 (95% CI ±75.3), 16/20승, +1.08%** — n=20에선 유의하나
selection seed(10000-2) 제외 시 CI[−9,+167]로 경계 → seed 20개 추가 검증 진행.
per-episode 원자료: scratchpad/stats20_Uniform10_Ent002.csv

## 최종 판정 (2026-07-02, 40 seeds 10000-10039)

seed 20개 추가 결과 **우위 소멸: PPO − SU-CQI = +25.3 (+0.36%), CI[−24, +75],
24/40승 → 통계적 동률.** n=20의 "+1.08% 유의"는 표본 확대에서 살아남지 못함.
확정 서사: "K16 비혼잡 uniform-10 regime에서 PPO는 SU-CQI ceiling에 도달(동률)."
원자료: scratchpad/stats20_Uniform10_Ent002.csv + stats20b_Uniform10_Ent002.csv
