# Run 1 — FirstFullRun (첫 실전 학습, 실패 사례)

> 단일 셀 MIMO-OFDM DL MU-MIMO PPO 스케줄러의 **첫 본 학습**. 결과적으로
> baseline 동률에 그쳐 **실패**했고, 그 원인 진단이 Run 2의 출발점이 되었다.
> (날짜: 2026-06-12 시작, update 1199에서 수동 중단)

---

## 1. 목적
Phase 2에서 구현한 PPO actor/critic + 학습 하니스(전수 코드리뷰·수정 완료본)로
실제 MIMO 스케줄링 정책을 학습할 수 있는지 처음으로 끝까지 돌려본 런.

## 2. 설정 (운영점 = "쉬운/기본")
| 항목 | 값 |
|---|---|
| config preset | `phase2_main_config` (기본값) |
| UE 수 K | 16 |
| 자원 | 8 RBG × 4 layer = 32 위치/슬롯 |
| **UE 속도** | **3 km/h** (채널 거의 정지) |
| **부하 p_arrival** | **0.2** |
| **마감 deadline** | **U[5,30] slots** |
| CSI 피드백 p_csi | 0.2 (stale) |
| 에피소드 | 1000 slot |
| PPO | gamma 0.99, λ 0.95, clip 0.2, lr 3e-4, value_coef **0.5**, max_grad_norm 0.5, minibatch 256 |
| 계획 update | 2000 (1 에피소드/update) |

### 실행 명령
```
cd /home/MYH/Y-Twin && bash ops_script.sh <gpu> \
  "cd ML_DRL_Scheduler && python train_phase2.py --mode main --run_name FirstFullRun \
   > FirstFullRun_run.log 2>&1"
```
GPU 컨테이너(Docker, RTX 4090). ~240 s/update.

## 3. 결과 (실패)
- **PPO eval이 baseline과 동률**에서 정체. update ~50부터 1190까지 평탄.
  PPO ~4046 vs 최고 휴리스틱 SUS+PF 4117 → **이기지 못함**.
- update 1199에서 수동 중단 (5일 낭비 방지). 체크포인트는 `FirstFullRun/ckpt/`에 보존
  (best.pt = update 479, eval 4162; 단발 스파이크 4458@1199는 노이즈로 판정).

## 4. 원인 진단 (수치 측정 + 적대적 검증)
`scripts/numeric_scout.py`로 측정. **두 가지 근본 원인**:

### (A) 과제에 "이길 여지(headroom)"가 없음
- 쉬운 운영점에서 **모든 스케줄러의 총보상이 1.4% 안에** 모임 (PPO·SUS+PF·PF 등).
- 자원이 수요보다 많아(비포화) 누가 스케줄하든 결과가 비슷 → **배울 게 없음**.
- 3 km/h라 CSI staleness도 무의미 → 똑똑한 스케줄링이 보상에 반영 안 됨.

### (B) 옵티마이저 결함 — value 타깃 미정규화
- value 타깃(returns) ~383을 정규화 없이 MSE → **value gradient ≈ 9237**,
  policy gradient 0.27의 **약 34000배**.
- 전역 `clip_grad_norm_(all params, 0.5)`가 거대한 value grad에 지배되어,
  critic이 학습 못 함 (explained_variance ~0.06 고정), 공유 encoder도 value에 끌려감.
- *정정*: 첫 진단 "clip이 actor를 얼린다"는 틀렸음 — Adam이 상수 클립 배율을 상쇄해
  actor는 4~7% 움직임(헤맴). 진짜 피해는 **critic + 공유 encoder**.

## 5. 결론 / 다음 단계(→ Run 2)
- **(A) 운영점을 어렵게** 만들어 학습 여지를 열어야 함.
- **(B) 옵티마이저를 고쳐야** 그 여지를 잡을 수 있음.
- 두 가지를 다 반영한 것이 **Run 2** (옆 폴더 `../Run2/` 참조).

## 6. 폴더 내용
```
FirstFullRun/            run 디렉토리 (ckpt/best.pt·latest.pt, csv_logs, tb_logs, config.json)
logs/FirstFullRun_run.log    학습 stdout 로그
figures/FirstFullRun_dashboard.png    학습 곡선 대시보드 (reward/EV/entropy/grad 등)
figures/FirstFullRun_diagnostic.png   진단 차트 (gradient 항 분해, 보상 분해, critic)
scripts/numeric_scout.py     원인 진단 측정 스크립트 (value grad 9237 등)
```
