# MixedSpeed_L2b — Run2 대비 변경사항 [2026-07-02 launch]

작성 2026-07-02. **MixedSpeed_L2의 계승 run**: 환경은 L2와 완전 동일, 바뀐 것은
학습 레시피 3가지뿐. L2가 update 29에 bar를 넘고(9179 > SUS-CQI@0.8 8939)
update 39/52의 KL 스파이크로 붕괴한 것에 대한 수술이다.

## 환경 (MixedSpeed_L2 = Run2 + 아래, 그대로 유지)

- K=32, **n_active ~ U[16,32]/에피소드** (Run2: K=16 전원)
- **UE별 속도 U(5,30) km/h/에피소드** (Run2: 30 균일)
- p_csi 0.6 (Run2: 0.2), entropy 0.02 (Run2: 0.01), deadline U[3,12]·p_arrival 0.4 동일
- 상세 근거는 `../MixedSpeed_L2/CHANGES_vs_Run2.md` 참조

## L2 → L2b 레시피 변경 3가지 (2026-07-02 신규 코드, 전부 opt-in 플래그)

| 변경 | 플래그 | 내용 / 근거 |
|---|---|---|
| **KL early-stop** | `--target_kl 0.02` | minibatch k3-KL > 1.5×0.02=0.03이면 해당 스텝 폐기 + 남은 epoch 중단 (ppo.py). L2 붕괴(upd 39: KL 0.054/clip 43%, upd 52: 0.066/47%)의 직접 방지책 — clip은 집계 drift를 못 막음이 실측됨 |
| **Critic v2** | `--critic_v2` | value 입력에 구조 피처 27개 추가(134→161): deadline histogram(5구간×count/backlog가중), backlog 총량, RBG별 retx 점유 히스토그램, CQI/age 집계, slot 위상 (policy.py). 오프라인 프로브: held-out R² −0.99(기존 6-스칼라) → **+0.52** — ev 0.04 정체는 입력 표현 병목이었음. Actor 구조는 무변경 |
| **Warm-start** | `--init_from Run3/MixedSpeed_L2/ckpt/best.pt` | L2 best(update 29, eval 9179)의 **actor 18개 tensor만** 로드; value head(차원 변경+프로브상 무가치)·return normalizer·optimizer는 fresh. 새 run_dir로 시작해 update 번호/CSV 오염 방지 |

실행 커맨드 (wrapper `Run3/_wrap_mixedspeed_l2b.sh`, GPU 4, watchdog 등록 세션 `mixedspeed_l2b`):
```
python3 train_phase2.py --mode hetero --entropy_coef 0.02 \
  --num_ue 32 --n_active_min 16 --n_active_max 32 \
  --ue_speed_min 5 --ue_speed_max 30 \
  --patience_evals 100000 --seed 2024 \
  --target_kl 0.02 --critic_v2 \
  [--init_from Run3/MixedSpeed_L2/ckpt/best.pt | --resume Run3/MixedSpeed_L2b/ckpt/latest.pt]
```
⚠️ resume 시에도 `--target_kl 0.02 --critic_v2`는 항상 필요 (cfg는 CLI에서 재구성;
critic_v2 없이 resume하면 value_head 차원 불일치로 크래시). wrapper가 자동 처리.

## 성공/실패 판정 지표

1. `csv_logs/ppo_metrics.csv`의 **explained_variance 0.04 → 0.2+** (critic 수술 효과)
2. run 로그의 `[kl-stop]` 발동 빈도 — 초반(critic 워밍업) 잦다가 잦아들면 정상, 계속 매 update면 target 하향 재검토
3. **eval reward가 8939(SUS-CQI@0.8 bar)를 넘어 유지** — 최종 목표. L2의 전철(일시 돌파 후 붕괴)과 달리 유지가 관건

## 검증 이력 (launch 전, 2026-07-02)

- 플래그 OFF 동일성: 수정 전/후 3-update 기준 run CSV **바이트 일치** (타 run 안전)
- v2 구조: params 94,147→97,603 (+27×128 정확), decode↔replay 일치, gradient 흐름 확인
- init_from smoke: 18 tensor 로드/9개 fresh 확인, KL guard 실발동 확인 (fresh critic 워밍업 보호)

## 목표 상향 (2026-07-02, hybrid/envelope 실험 반영)

성공 지표 ③을 상향: 8939(SUS-CQI@0.8 bar)가 아니라 **9620(oracle per-episode
envelope, seeds 10000-2) 돌파**가 목표. 근거: L2@29는 9157(envelope −4.8%)이었고,
MixedLoad의 PPO는 자기 envelope를 +5.3% 넘었으므로 아키텍처상 도달 가능한 선.
관건은 저부하 에피소드 성능 (critic v2의 n_active/backlog 피처가 도울 영역).

## 레시피 패치: kl-stop v2 (2026-07-02, update ~59에서 재시작 적용)

v1 kl-stop(발동 시 남은 epoch 전체 중단)의 설계 결함 발견: 발동률 79%(44/56
update)로 critic의 학습 기회까지 잘라 워밍업을 자체 연장하는 악순환 (KL은 actor
이동량이므로 critic 학습은 제한할 이유가 없음). **v2: 발동 시 actor(+공유
encoder)를 grad=None으로 동결하고 남은 pass는 value loss만 학습** (ppo.py).
검증: 플래그 OFF 동일성 바이트 일치(타 run 무영향), frozen 경로 smoke 통과.
update 61에서 정지, checkpoint 59에서 v2 코드로 재개 (2 update 손실).
관찰 포인트: ev 상승 가속 여부 + [kl-stop] "value-only" 발동률의 감소 추세.

## 종료 (2026-07-06) — 목표 달성, 수확기 수렴

update ~1146/1500에서 **수동 종료** (사용자 승인). **best 10030@859 = oracle
envelope(9620) +4.3%, bar(8939) +12.2% — 상향 목표 달성.** 이후 28 eval 라운드
미갱신(patience-15의 ~2배)이나 MixedLoad와 달리 건강한 정체: entropy 0.5~0.8 유지,
guard가 KL 0.015~0.03 관리, 열화 없음. 레시피(actor warm-start + KL guard v2 +
critic v2) 판정: **성공** — 붕괴 재발 없이 구 L2 best(9179)와 envelope를 모두 초과.
단 critic v2의 online ev는 ~0.05로 오프라인 상한(0.52)에 못 미침(이동 정책 +
에피소드 스케일 변동) — Run4 개선 과제. 논문 수치는 best.pt의 20-seed 재평가로 확정.

## 최종 확정 (2026-07-06, 20 seeds 10000-10019 paired) — 논문 수치

PPO-best(10030@859) 9581±2028 | oracle envelope 8964 | 최강 hybrid(T=14) 8698 |
SUS-CQI@0.8 8490 | SU-CQI 8068.
**PPO − oracle envelope = +617.0 (+6.88%), CI[+410,+824], 16/20승 → 유의한 승.**
vs 최강 hybrid +10.15% (19/20), vs SUS-CQI@0.8 +12.85% (**20/20 전승**).
MixedLoad(+6.35%)와 거의 동일한 크기 — 두 혼합 축의 상호 재현.
원자료: _analysis_20260702/final20_MixedSpeed_L2b.csv
