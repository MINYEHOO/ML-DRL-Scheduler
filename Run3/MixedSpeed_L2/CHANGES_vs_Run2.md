# MixedSpeed_L2 — Run2 대비 변경사항 [2026-07-02 정지, L2b로 교체됨]

작성 2026-07-02. 이 run은 update ~141에서 **정지**되었고 `Run3/MixedSpeed_L2b`가 계승.
폴더는 붕괴 사례 기록 + best.pt(L2b의 warm-start 소스)로서 보존. **삭제 금지.**

## Run2 기준점 (Run2_HardMain)

- K=16 고정 인원, 속도 30 균일, p_csi 0.2, deadline U[3,12], p_arrival 0.4
- 결과: PPO near-SU 학습(+50%)

## 이 run의 변경점

| 항목 | Run2 | 이 run | 이유 |
|---|---|---|---|
| **혼잡도** | K=16 전원 | **K=32, n_active ~ U[16,32]/에피소드** | MixedLoad_L2와 동일 |
| **UE 속도** | 30 균일 | **UE별 연속 U(5,30) km/h, 에피소드마다 재추첨** (`--ue_speed_min 5 --ue_speed_max 30`) | **핵심 추가 변수** — CSI 신뢰도가 UE별로 다름 → pairing 상대 선택까지 학습 필요 |
| p_csi / entropy | 0.2 / 0.01 | 0.6 / 0.02 | hetero 프리셋 / 탐색 |
| early-stop / 장치 | 없음 / GPU3 | 비활성 / GPU 4 | — |
| 신규 코드 | — | `ue_speed_min/max` (config/channel, 2026-07-01) — 속도벡터를 UE별로 rescale | — |

실행 커맨드:
```
python3 train_phase2.py --mode hetero --entropy_coef 0.02 \
  --num_ue 32 --n_active_min 16 --n_active_max 32 \
  --ue_speed_min 5 --ue_speed_max 30 \
  --patience_evals 100000 --seed 2024 [--resume ...]
```

## 목적 / 가설

MixedLoad(혼잡도 혼합)에 **CSI 신뢰도 혼합**을 얹은 최상위 난이도: 한 정책이 (i) 에피소드별 SU/MU 전환 + (ii) UE별 신선도를 보고 pairing 상대/깊이 선택을 동시에 해야 함. 고정 heuristic으로는 흉내낼 수 없는 조합.

## 경과와 정지 사유 (붕괴 사례 기록)

- Update 29: eval **9179** — 2026-07-02 측정 bar(SUS-CQI@0.8 = **8939**)를 +2.7% 돌파 ✓
- **Update 39: KL 0.054/clip 43%, update 52: KL 0.066/clip 47%의 파괴적 update로 정책 붕괴** (eval 9179→7429). clip은 발동 중이었으나 집계 drift를 못 막음 — KL guard 부재가 원인
- 이후 110 update 동안 보수 basin(depth 3.8→2.9)에서 회복했지만 8350(@139)에 그침 (자기 best −830)
- Critic explained variance ~0.04 정체 (오프라인 프로브: 입력 표현 병목 확정, rich 피처면 R² 0.52 가능)
- → 2026-07-02 **정지**. best.pt(update 29) actor를 warm-start로 넘기고, KL early-stop + critic v2를 장착한 **MixedSpeed_L2b**로 교체 (상세는 그 폴더의 CHANGES_vs_Run2.md)

## 공통 사항

- Eval seed 10000-10002 ×3; 신 baseline 12종이 CSV에 있음 (SUS-CQI 8790, SU-CQI 7891 등)
- 채널/PHY/reward Run2와 동일 (n_active·속도 추첨만 추가)

## Hybrid / envelope 기준선 (2026-07-02, seeds 10000-10002)

pure SU-CQI 7949 | SUS-CQI@0.8 8939 | hybrid 최강(T=10) 8975 | **oracle envelope
9620** | PPO@29(best.pt) 9157 — 붕괴 전 정책도 실현가능 hybrid는 이기지만(+2.0%)
envelope에는 −4.8% (저부하 seed 10000: SU-CQI 8121 vs PPO 7818 — 저부하 모드 미숙).
→ L2b의 정량 목표 = envelope 9620 돌파 (../MixedSpeed_L2b/CHANGES_vs_Run2.md 참조).
