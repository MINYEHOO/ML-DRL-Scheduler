# Run 2 — Run2_HardMain (성공 사례: PPO가 휴리스틱을 +50% 능가)

> Run 1의 실패 원인(헤드룸 없음 + 옵티마이저 결함)을 진단해 **운영점을 어렵게
> 바꾸고 옵티마이저를 고쳐** 다시 돌린 본 학습. PPO가 stale CSI만으로 최고
> 휴리스틱을 **+50%**, 완벽-CSI 강한 참조(genie)와 대등/소폭 우위로 이겼다.
> (2026-06-15 시작, 2000 update / 5.6일 완주)

---

## 1. Run 1에서 바뀐 것 (왜 바꿨나)

| | Run 1 (실패) | **Run 2 (성공)** | 이유 |
|---|---|---|---|
| UE 속도 | 3 km/h | **30 km/h** | staleness가 헤드룸의 진짜 레버 (아래 §2) |
| 부하 p_arrival | 0.2 | **0.4** | 혼잡 생성 |
| 마감 deadline | U[5,30] | **U[3,12]** | 우선순위 결정이 중요해짐 |
| value 타깃 | 미정규화 | **정규화 (#1)** | value grad 9237→O(1), critic 회복 |
| grad clip | 전역 1개 | **actor/critic 분리 (#3)** | value 폭주가 actor 예산 잠식 방지 |
| value_coef | 0.5 | **0.25** | value 비중 완화 |

코드 변경: `policy.py`(value 정규화 버퍼·`value()` 역변환·`update_return_normalizer`),
`ppo.py`(정규화 value loss + 분리 클리핑), `config.py`(`phase2_hard_*_config` 프리셋),
`train_phase2.py`(`--mode hard_main`). (#2 encoder detach는 의도적으로 제외.)

## 2. 운영점 탐색 (왜 30 km/h인가) — `figures/{headroom,oracle,speed}_probe.png`
읽기 전용 probe(코드 미변경, config만)로 "어디에 학습 여지가 있나" 측정:
1. **headroom probe**: 부하+마감을 올리니(p0.4, dl[3,12]) 스케줄러 간 보상 spread 1.3%→8.9%.
   하지만 이건 "Random보다 낫다"일 뿐.
2. **oracle probe** (3 km/h): fresh-CSI ≈ stale (gap ~0) → 휴리스틱 *위*엔 공간 없음.
3. **speed sweep** (결정적): oracle gap이 **30 km/h +23%, 60 km/h +34%**로 열림.
   → **staleness가 진짜 레버.** 휴리스틱은 Age를 무시하지만, 학습 정책은 stale CSI를
   회피할 수 있음. eta_d 증가는 역효과 → eta_d=1 유지. **30 km/h 채택.**

## 3. 확인 런 (HardDebug_confirm) — `HardDebug_confirm/`, `figures/HardDebug_confirm.png`
본 5일 런 전, debug 설정(K=16, 300 slot, 150 update, ~2.5h)으로 수정이 작동하는지 확인:
- grad_norm ~9000→O(10) 정상화, EV 음수→양수, **PPO eval이 update ~39에 baseline을 뚫고
  +20~27%** → 통과. 이후 본 런 진행.

## 4. 본 학습 (Run2_HardMain)
### 설정
`phase2_hard_main_config`: K=16, 1000 slot, **30 km/h**, p_arrival 0.4, deadline U[3,12],
p_csi 0.2, value_coef 0.25 + 옵티마이저 수정(#1·#3). 2000 update.

### 실행 명령
```
cd /home/MYH/Y-Twin && bash ops_script.sh 3 \
  "cd ML_DRL_Scheduler && python train_phase2.py --mode hard_main --run_name Run2_HardMain \
   > Run2_HardMain_run.log 2>&1"
```
GPU 3 (RTX 4090), 2000 update / 480436 s (~5.6일), 240 s/update.

## 5. 학습 결과 — `figures/Run2_training.png`
- entropy 1.30→0.04(수렴), grad_norm ~9000→~10(결함 해소), EV ~0.06 천장(critic 약함).
- **사실상 update ~479에 수렴** (best.pt=479, eval 6740). 이후 1500 update 무익
  (KL·clip 후반 상승, 최종 6276 < best 6438) → **다음엔 early stopping 가능**.

## 6. 최종 성능 (12 held-out seed) — `figures/Run2_metrics.png`
| 스케줄러 | 보상 | 처리량 | 완료율 | 마감miss | retx실패 | SINR | UE/RBG |
|---|---|---|---|---|---|---|---|
| 최고 MU 휴리스틱 (CQI) | 4292 | 44 | 0.743 | 0.087 | **0.167** | 6.7dB | 1.63 |
| SU-CQI (신규 SU baseline) | 6241 | – | 0.843 | 0.152 | 0.003 | 22.2dB | 1.00 |
| genie (완벽 CSI, 강한 참조) | 6217 | 54 | 0.869 | 0.121 | 0.008 | 9.4dB | 2.95 |
| **PPO (학습, stale CSI)** | **6438** | **57** | 0.854 | 0.140 | **0.004** | 20.6dB | 1.12 |

- **PPO +50% vs 최고 MU 휴리스틱, +3.6% vs genie.** 12 seed 전부 승,
  PPO 최악 seed(5613) > 휴리스틱 최고 seed(5510). per-seed 분산도 최저(9.4% vs 14-22%).

## 7. 승리 메커니즘 (왜 이기나)
- **PPO는 ~SU-MIMO를 학습** (1.12 UE/RBG). 30 km/h + stale CSI에선 BS가 옛 방향으로
  RZF null을 쳐서 간섭을 못 죽임 → MU 페어링 실패. PPO는 **안 묶어** 간섭원 자체를 제거
  → SINR 20.6dB, 전송 실패 ~0.
- 휴리스틱 5종은 전부 MU(~1.6) → 간섭 → 17% 실패. **SUS+PF조차 무력** (직교성을 stale CSI로
  판정해 틀림). 그래서 stale에선 CQI-greedy가 최고 baseline.
- **실패 모드 전환**: 휴리스틱="보내고 실패(retx 0.17)", PPO="확실할 때만(마감miss 0.14)";
  총 실패는 PPO 0.144 vs 휴리스틱 0.254.

### SU baseline 비교 (정직한 해석) — `scripts/su_baseline.py`
"RBG당 1명" 손코딩 SU 휴리스틱을 추가 측정:
- **SU-CQI 6241 vs MU-CQI 4292 = +45%.** 즉 우위의 대부분은 **"SU 전략" 자체**에서 옴.
- **PPO 6438 vs SU-CQI 6241 = +3.2%.** PPO는 그 위에 "똑똑한 SU"(상황 봐서 가끔 안전한
  MU=1.12, 더 나은 UE 선택)를 더함.
- **핵심 가치**: 사람은 이 SU 전략을 *미리 알아야* 손코딩한다. 기존 문헌 휴리스틱(CQI/PF/SUS)은
  전부 MU라 실패했고, **PPO는 "이 채널 상황에선 SU"라는 비직관적 강건 전략을 스스로 발견**했다.

## 8. 한계 (정직하게)
- **약한 critic** (EV ~0.06) — 정책은 견디지만 개선 여지.
- **공정성 우위 없음** (Jain 0.675 ≈ 휴리스틱) — 보상이 공정성을 거의 안 봄.
- **genie는 진짜 상한 아님** (근시안+greedy) → "PPO>genie"는 신중히 표현.
- **단일 운영점**(30 km/h, 이 보상)만 검증 — 일반화 미확인.
- SU baseline이 +45% → "+50% 능가"의 헤드라인 가치는, "SU가 핵심·PPO는 자동발견+소폭개선"으로
  정직하게 표현해야 함.

## 9. 다음 단계
- **②MU/SU 적응 검증** (p_csi=1.0/3km/h 재학습 → PPO가 MU로 전환하는지). "SU를 외운 게 아니라
  상황 적응"의 결정적 증거.
- 일반화 스윕(속도/부하/p_csi), early stopping, critic 강화(#2 encoder detach),
  공정성 가중 보상, multi-packet 큐(포화 조건에서 MU 필요성·공정성 붕괴 관찰).

## 10. 폴더 내용
```
Run2_HardMain/        본 학습 run 디렉토리 (ckpt/best.pt=update479·latest.pt, csv_logs, tb_logs, config.json)
HardDebug_confirm/    수정 검증용 짧은 debug 런 (150 update)
logs/                 Run2_HardMain_run.log, HardDebug_run.log
figures/              Run2_metrics/training/analysis/confirm.png, HardDebug_confirm.png,
                      headroom/oracle/speed_probe.png (운영점 탐색)
scripts/              probe(headroom/oracle/speed), 평가(final_eval/genie/su_baseline/
                      density/confirm_run2/drop_split), 수정검증(verify_fix)
```
