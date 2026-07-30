# PPO / DRL — 논문 서술용 정리

작성 2026-08-01. 코드 직접 확인 기준 (pin 71c0bd4 계열, `Run4/QueuePostRZF_S40HighLoad/config.json`
실제 실행값). 파일:줄 표기는 저장소 루트 기준.

이 문서의 목적: 논문의 "Method — Learning" 절과 "Implementation details"를 쓸 때
**코드에 실제로 있는 것만** 쓰기 위한 근거 모음. 추측·관례로 채운 문장이 리뷰에서
깨지는 것을 막는 것이 목표다.

---

## 1. 문제 정식화 (MDP)

| 요소 | 실제 구현 | 근거 |
|---|---|---|
| 1 step | **1 slot** (0.5 ms, 30 kHz SCS) | `env.step()` |
| 1 episode | **1000 slot** (= 0.5 s) | `config.py:243` `episode_len_main=1000` |
| 1 PPO update | **1 episode** (num_envs=1) | `ppo.py:15` |
| 상태 | 관측 = CSI feedback buffer(양자화 방향·CQI·Age) + 큐/deadline/backlog + 슬롯 내 문맥 | `policy.build_encoder_input` |
| 행동 | 매 슬롯 **[R=8 × L=4] 격자**를 순차적으로 채움. 각 위치에서 {no-user} ∪ {유효 UE}에 대한 Categorical | `policy._rbg_major_pass` |
| 전이 | RZF 프리코딩(ĥ 기반) → 실채널로 SINR 평가 → ACK/NACK → 큐·deadline 갱신 | `env.step`, `phy.compute_slot_sinr` |
| 종료 | **시간 제한(truncation)**, 진짜 terminal 아님 | `ppo.py:18-20`, `ppo.py:66-71` |

### 행동 공간의 성격 (논문에서 반드시 정확히 쓸 것)

한 슬롯의 행동은 **하나의 매크로 행동**이며, 최대 32개(=R×L)의 순차적 sub-decision으로
분해된다. 각 sub-decision은 그 시점의 슬롯 내 문맥(이미 선택된 UE 집합 S_r, 잔여
예산, RBG 폐쇄 여부)에 조건부다. 따라서 정책은 자기회귀(autoregressive) pointer
정책이고, 매크로 행동의 로그확률은 sub-decision 로그확률의 **합**이다.

- 유효하지 않은 선택은 softmax **이전에** `-1e9`로 마스킹 (`policy.py`, `masked_fill(~valid_mask, -1e9)`).
- RBG는 no-user가 선택되면 닫히고 이후 layer는 결정 대상에서 제외된다 (spec §5).
- 재전송으로 이미 고정된 위치(preempted retx)는 정책 결정 대상이 아니다
  (`fixed_mask`) — 그 위치는 `policy_decision_mask=False`로 기록되어 PPO 목적함수에
  들어가지 않는다.

### 보상 (env.py:311-341)

```
r_t = λ_s · Σ_u w_u · useful_bits_{u,t} / B_norm   (dense)
      + λ_c · n_completed_t                         (sparse, +)
      − λ_m · n_failed_t                            (sparse, −)

w_u = 1 + η_D / (D_u + 1)          # deadline urgency weight
```

실행값: λ_s = 1, λ_c = 1, λ_m = 2, η_D = 1, B_norm = 8000.
`n_failed`는 deadline miss + retx drop + overflow drop을 합한 것(현 세계군에서는
retx drop = 0이므로 사실상 deadline miss).

**측정된 성질**: dense 항이 "결국 실패할 패킷"에 지급되는 누수 비율은 PPO 3.16%,
SUS+CQI 6.28% (`Run4/_analysis/scripts/reward_leak_probe.py`). 논문에는 "이층 보상
(결정-품질 dense + 결과-정산 sparse), misalignment 실측 3.2%"로 서술하기로 결정
(2026-07-22). PBRS 재설계는 채택하지 않음.

---

## 2. 정책·가치 신경망 (policy.py)

```
SharedEncoder :  per-UE 특징 73 → 128 → 128 → 64      (34,240 params)
ScoreNet      :  [e_u ; 7 문맥특징] 71 → 128 → 64 → 1  (17,537)
NoUserHead    :  [pooled e ; 4] 68 → 128 → 64 → 1      (17,153)
ValueHead     :  [2×64 pooled ; 6 ; v2 33] 167 → 128 → 64 → 1  (29,825)
──────────────────────────────────────────────────────
actor(+shared encoder) 68,930 |  critic 29,825 |  전체 98,755
```

- 인코더는 **UE별로 동일 가중치**를 적용하고(permutation-equivariant), 점수는 UE별
  스칼라로 나온다 → **파라미터가 K와 무관**. K=16/24/32/48 zero-shot 전이 실험에서
  strict 가중치 로드가 전 K에서 통과한 것이 이 성질의 실증
  (`Run4/_analysis/scripts/scaling_zeroshot_probe.py`).
- **인코더는 actor와 critic이 공유**한다. 단, PPO 업데이트에서 그래디언트 클리핑은
  actor 그룹(인코더 포함)과 value_head를 **분리**해서 건다 (`ppo.py:151-154`,
  `ppo.py:259-271`). 논문에 "shared encoder"라고 쓸 때 이 분리를 함께 적어야 정확하다.
- **실시간성 — 수치 정정 (RESEARCH_LOG.md:100의 "~5–10 MFLOPs/슬롯"은 틀렸다)**.
  인코더는 매 슬롯 K·R개 행 전부에, ScoreNet은 자유 위치마다 K개 행 전부에 돈다.
  실측 결정 수는 슬롯당 **19–21개**(=`entropy_sum/entropy_mean`, `ppo.py:211`로 복원;
  고정 retx·폐쇄 RBG 위치는 비용 0):

  ```
  encoder    256 rows × 33,920 MAC =  8.7 MMAC
  score_net   20 pos × 32 UE × 17,344 = 11.1 MMAC
  no_user     20 pos × 16,960         =  0.3 MMAC
  ────────────────────────────────────────────
  합계 ≈ 20 MMAC ≈ 40 MFLOPs/slot  (K=32)      |  K=16: ≈10 MMAC ≈ 20 MFLOPs
  ```

  즉 **파라미터 수는 K-무관이지만 연산량은 O(K · n_positions)** 다. 논문 복잡도 절에는
  이 수치와 스케일링을 쓰고, `n_positions`는 데이터 의존(RBG 폐쇄·고정 retx)이므로
  R·L=32이 아니라 **실측 결정 수**를 보고할 것. 40 MFLOPs면 gNB가 이미 수행하는 RZF
  역행렬보다 가볍다는 결론 자체는 유지된다. (평가가 느린 것은 Python 루프 오버헤드이지
  모델 크기가 아님.)

---

## 3. PPO 구현에서 표준과 다른 점 (전부 논문에 쓸 것)

### 3.1 매크로 행동 비율 (macro-ratio PPO)

```python
log_ratio = clamp(new_logprob_sum − old_logprob_sum, −20, 20)   # ppo.py:194
ratio = exp(log_ratio)
loss  = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)                  # ppo.py:201-204
```

- 비율은 sub-decision별이 아니라 **슬롯 전체(최대 32개 결정)의 결합 로그확률**로
  계산한다. 즉 clip이 "슬롯 단위 정책 변화"를 제한한다.
- 결과적으로 log-ratio가 단일 행동보다 크게 흔들릴 수 있어 ±20 clamp를 둔다
  (float32 `exp` 오버플로 방지). 정상 동작 구간에서는 비활성.
- **논문 문장 예**: "The per-slot allocation is treated as a single macro-action;
  the importance ratio is formed from the joint log-probability of its (up to 32)
  masked sub-decisions, so the clipping constraint applies at slot granularity."

### 3.2 엔트로피는 per-decision 평균 (중요)

```python
entropy_mean = rep["entropy_sum"] / num_policy_decisions   # ppo.py:210-211
loss −= cfg.ppo_entropy_coef * entropy_mean                # ppo.py:213-215
```

- CSV에는 `entropy_sum`과 `entropy_mean`이 **둘 다** 기록되지만, **최적화에 들어가는
  것은 per-decision 평균**이다.
- 따라서 "entropy coefficient 0.02"는 슬롯당 합이 아니라 **결정 하나 기준**이다.
  그냥 "entropy coefficient 0.02"라고만 쓰면 32배 오해를 살 수 있으므로
  **"per-decision entropy"를 명시**할 것.
- dose-response 실험 결과(0.02 ≫ 0.01 > 0.03)도 이 규약 기준이다.

### 3.3 종료는 truncation, V(s_T)로 부트스트랩

```python
# ppo.py:66-71
if t == T-1:
    next_value = last_value      # V(s_T)
    next_nonterminal = 1.0       # 시간 제한이므로 terminal 아님
```

- 1000슬롯 종료는 과제 종료가 아니라 시간 제한이므로 경계에서 V=0을 넣지 않는다.
  이걸 terminal로 처리하는 구현이 흔한데, 그 경우 리턴이 체계적으로 편향된다.
- **알려진 근사** (`ppo.py:90-92`): 부트스트랩에 쓰는 최종 관측은 `_prepare_slot`
  한 번 이전 상태다. 1000개 항 중 마지막 하나에만 영향하므로 실질 무시 가능하지만,
  엄밀히 쓰려면 "the bootstrap value is taken at the last prepared observation"
  정도로 적거나 각주 처리.

### 3.4 리턴 정규화 (value target normalization)

```python
ac.update_return_normalizer(returns)                 # ppo.py:145 (EMA, no grad)
value_loss = ((V_raw − return) / ret_std)**2         # ppo.py:208
```

- value head는 **정규화 공간에서 예측**하고 `value()`에서 원래 단위로 역정규화하므로
  (`policy.py:336`), GAE·부트스트랩은 원 보상 단위로 계산된다.
- μ/σ는 gradient 없는 buffer이고 롤아웃마다 EMA(0.99/0.01)로 갱신, 첫 업데이트에서
  초기화 (`policy.py:339-355`).
- 도입 이유: 원 리턴이 O(10²)이라 value gradient가 O(10⁴)까지 커져 **공유 grad-clip
  예산과 공유 인코더를 잠식**했다. 정규화 + 분리 clip이 그 결합을 끊는다.
- 논문에 쓴다면: "value targets are normalized by a running return statistic
  (no gradient), and the actor and critic parameter groups are clipped separately,
  so the critic's gradient scale cannot throttle the policy update."

### 3.5 KL early-stop = actor만 동결, critic은 계속 학습

```python
if k3_KL > 1.5 × target_KL:  actor_frozen = True     # ppo.py:248-250
...
if actor_frozen:  p.grad = None for actor params     # ppo.py:262-264
```

- KL 추정은 k3 estimator `mean((r−1) − log r)` (비편향, 항상 ≥ 0).
- 임계 초과 시 남은 epoch 동안 **actor(공유 인코더 포함)를 비트 단위로 동결**하고
  value head만 학습한다. Adam은 grad=None 파라미터를 건너뛴다.
- 근거(코드 주석에 기록): v1의 전면 정지는 초기 업데이트의 79%에서 critic 학습까지
  굶겼다. clip만으로는 매크로 행동의 누적 드리프트가 안 잡힌다 (MixedSpeed_L2가
  update 39/52에서 붕괴: KL 0.054/0.066, clip fraction 0.43/0.47).
- 실행값 `ppo_target_kl = 0.02`.

### 3.6 어드밴티지 정규화

롤아웃(=에피소드) 전체에서 한 번 정규화하고 (`ppo.py:136-140`), 그 값을 4 epoch
동안 재사용한다. 미니배치별 재정규화 아님.

### 3.7 비대칭 actor-critic (critic 전용 특징) — 논문에 쓰면 좋은 포인트

`policy.py:92-160` (`build_value_feats_v2`, `ppo_critic_v2=True`)이 critic에만 27(+6)개
구조 특징을 준다. 여기에 **에피소드 진행도 `slot/episode_len`**(`policy.py:148`)이
포함되는데, actor의 입력(`build_encoder_input`, `policy.py:56-83`)에는 **시간 인덱스가
전혀 없다**.

> "We use an asymmetric actor-critic: the critic receives additional structured state
> features (deadline histogram, backlog, fairness, queue pressure, episode phase) that
> the actor does not. The learned scheduler is therefore time-homogeneous and
> independent of the episode length used in training."

이건 방어가 아니라 **장점**이다 — 정책이 에피소드 길이에 의존하지 않으므로 임의 길이
운용에 그대로 쓸 수 있고, critic은 학습 시에만 쓰이므로 배치에 privileged 정보가
필요 없다.

### 3.8 actor가 보는 rate 특징은 SU 상한이지 실현 rate가 아니다

`policy.py:207-211` `predict_btx_torch`는 **단일 유저 full-power 상한**(depth 1에서만
정확)이고, 실제 커밋되는 rate는 RBG 마감 시 `SlotAllocationPlanner`가 계산한다
(`policy.py:547`, `env.py:481-531`). 정직하게 쓸 것:

> "The candidate-scoring network observes the single-user full-power transmission-size
> bound (exact at multiplexing depth 1), together with the current group depth and an
> orthogonality score against the already-selected set; the realized post-RZF size is
> computed by the link-adaptation planner at group closure. **All comparator schedulers
> use the identical bound for candidate generation.**"

마지막 문장이 중요하다 — 낙관적 상한을 PPO만 쓰는 게 아니라 baseline도 동일하게
쓰므로 공정성이 유지된다.

### 3.9 value clipping은 없다 (있다고 쓰지 말 것)

`ppo.py:208`은 정규화 공간의 **평범한 MSE**다. reference 구현(SB3 등)의 value clipping은
논문 알고리즘이 아니라 구현 디테일이므로 없는 게 정상이지만, **"separate value
clipping"이라고 쓰면 거짓**이 된다. 분리된 것은 **gradient norm clipping**이다.

---

## 4. 하이퍼파라미터 (실행값, `QueuePostRZF_S40HighLoad`)

| 항목 | 값 | 비고 |
|---|---|---|
| γ | 0.99 | 유효 지평 ~100 slot (에피소드 1000) |
| GAE λ | 0.95 | |
| clip ε | 0.2 | 매크로 비율 기준 |
| learning rate | 3e-4 | Adam |
| value coef | 0.25 | 정규화 도입과 함께 0.5→0.25 |
| **entropy coef** | **0.02** | **per-decision** (§3.2) |
| max grad norm | 0.5 | actor/critic **분리** 적용 |
| epochs | 4 | |
| minibatch | 256 | 슬롯 단위 |
| target KL | 0.02 | 1.5× 초과 시 actor 동결 |
| critic v2 features | on | value 입력에 구조 특징 +33 |
| 에피소드 | 1000 slot | 1 update = 1 episode |

---

## 5. 학습·평가 프로토콜

- **학습 시드**: episode_idx = update 번호. `num_updates ≤ 10000` 가드가 있어
  평가 시드 대역(10000+)과 겹치지 않는다 (`train_phase2.py:77-79`).
- **run-eval**: 10 update마다 시드 10000–10002 3개(=3000 slot), deterministic(argmax).
  baseline 12종은 결정론적이므로 update 9에서 **한 번만** 평가하고 재사용.
- **held-out 최종 평가**: 시드 10000–10019 20개, 세계별 SUS threshold 재스윕 후
  최강 baseline과 paired 비교.
- **미사용 예비 시드**: 20000+ (최종 확정용으로 보존).

### best 체크포인트 선택과 보고 시드의 부분 중복 — **측정으로 해소됨**

`best.pt`는 run-eval 시드(10000–10002)의 reward로 선택되는데 최종 보고가
10000–10019이므로 3/20이 선택에 쓰인 시드다. **실제로 계산해 본 결과 선택 편향과
정반대 부호**다 (S40HighLoad held-out):

| 시드 집합 | reward 마진 | goodput 마진 | miss | 승수 |
|---|---|---|---|---|
| 전체 20 | +14.66% | +4.14% | −3.0%p | 20/20 |
| 선택 3 (10000–02) | **+12.46%** | +3.48% | −2.4%p | 3/3 |
| **비선택 17 (10003–19)** | **+15.07%** | +4.26% | −3.1%p | **17/17** |

즉 PPO는 자기가 선택된 시드에서 **오히려 가장 못한다**. 선택 편향이 있었다면 반대여야
한다. 서술 방식 두 가지:

1. **최종 표를 10003–10019(17개)로 보고** — 가장 깨끗하고 숫자도 좋아진다(+15.07%).
2. 20개 그대로 쓰고 각주: *"Model selection used seeds 10000–10002. Excluding the
   three selection seeds, the margin increases from +14.66% to +15.07% (17/17 seeds),
   i.e. the opposite sign from selection bias."*

권장: (2) — 20 시드 전승이라는 진술을 유지하면서 각주로 방어하는 편이 강하다.
여유가 있으면 미사용 20000번대로 최종 표를 한 번 더 뽑는 것이 최선.

### ⚠ 단일 학습 시드 — 정밀 감사에서 살아남은 유일한 major

저장소의 **모든 런이 학습 시드 2024**다 (`train_phase2.py:78` 기본값, config.json
36개 전부, best.pt 29개 전부 동일). 반면 비교 대상인 SUS+CQI/SU+CQI는 결정론적이라
사실상 분산이 0이다. 즉 PPO는 학습-시드 차원에서 **n=1**이다.

- held-out 20 시드의 CI는 **환경 잡음만** 정량화한다. 학습-런 간 분산은 어디에도 없다.
- Henderson et al.(2018)류 지적("≥5 시드, mean±std")에 답할 재료가 아티팩트에 없다.
- 다만 `--seed`는 eval 채널 추첨에도 들어가므로(`env.py:87-91`), 복제 런은 held-out
  에피소드 자체가 바뀌어 baseline도 다시 돌려야 한다. 기존 20-seed 스크립트가
  config.json에서 전부 재실행하므로 **비용은 학습 런 1개**.

대응 (택1):
1. `--seed 2025`로 대표 런 1개 복제 → "2 seeds, both directions consistent" 서술. **권장**.
2. 명시적 한계 서술: *"We report a single training seed per configuration. The result
   is corroborated by N independent runs across different environments and
   hyperparameters, all with the same sign; we do not claim recipe reproducibility
   across training seeds."*

margin이 큰 주장(+14.7%, 20/20)은 단일 시드로도 방어 가능하지만, **마진이 한 자릿수
초반인 주장은 복제 없이는 지지되지 않는다**.

---

## 6. explained variance가 낮은 이유 (리뷰어 예상 질문)

실측: HighLoad는 초반 +0.07 → 종반 **+0.26** (최대 0.48), genie 런들은 0 근방.

- 리턴 분산의 대부분이 **원리적으로 예측 불가능한 몫**이다: 에피소드마다 부하
  p~U(0.15,0.5)·활성 인구 U(16,32)·지형이 통째로 재추첨되고, 슬롯마다 Bernoulli
  도착·CSI 소실·페이딩이 굴린다. 에피소드 보상이 −7000~+8000을 오간다.
- PPO에서 critic은 정확도 목표가 아니라 **분산 감소용 baseline**이다. 부정확해도
  정책 그래디언트의 방향은 편향되지 않는다.
- 판정은 레벨이 아니라 **추세**로 한다: 0에서 상승 = critic이 세계를 붙잡는 중(건강),
  지속적 강한 음수 + value_loss 발산 + KL/clip 폭주 동반 = 문제.
- **개선 여지(정직하게 적을 것)**: update당 에피소드 1개 설계가 에피소드 추첨 잡음을
  advantage에 그대로 싣는다. 병렬 롤아웃(num_envs>1)이면 EV와 학습 속도가 함께
  올라갈 여지가 있다. 현 세대에서는 목표(휴리스틱 초과)가 일관되게 달성되어 미적용.

---

## 7. 리뷰어가 공격할 만한 지점과 준비된 답

| 공격 | 답 |
|---|---|
| "PPO를 그냥 갖다 쓴 것 아닌가" | 매크로 행동 비율, per-decision 엔트로피, truncation 부트스트랩, 리턴 정규화 + 분리 clip, actor-only KL 동결 — 구조적 행동 공간에 맞춘 적응이 다섯 군데. 각각 실패 사례(§3.5의 L2 붕괴)와 함께 제시 가능 |
| "baseline이 약한 것 아닌가" | 세계마다 SUS threshold를 **재스윕**해 최강 baseline과 비교. 12종 전체 비교도 보유(P065 보강 실험) |
| "정보 우위 아닌가" | PPO와 baseline이 **동일한 feedback buffer**만 본다. 실채널은 SINR/ACK 평가에만 사용 (`env.py:240` 주석). SUS의 직교점수도 `direction_fb` 기반 |
| "시드 몇 개로 우연 아닌가" | held-out 20 시드에서 reward/goodput/miss **20/20 전승** (paired), t=7.84 |
| "일반화 안 되는 것 아닌가" | 분포 밖 6개 세계 zero-shot에서 붕괴 없음 (OOD 프로브), K=16–48 zero-shot에서 무패 |
| "explained variance가 낮다" | §6 |
| "genie 정책을 실전에 쓰면 되지 않나" | 역방향 전이 실험: −5.8%, depth 침식이 원인 (2026-08-01 실험) |

---

## 8. 정밀 감사 결과 (2026-08-01)

6개 관점(GAE·목적함수·아키텍처·롤아웃 인터페이스·학습루프·수치안정성)으로 감사하고,
각 지적을 독립 검증자가 **반박 시도**했다. 총 43건 중 **41건이 반박되어 탈락**,
2건 생존.

**결론: 학습 알고리즘 자체에 결함 0건.** 반박된 41건에는 "value clipping 없음"(→ 표준
목적함수가 맞음), "critic에 episode phase 누출"(→ actor는 시간 비의존, 오히려 장점),
"GAE terminal 분기 버그"(→ 해당 분기는 실행되지 않음), "리턴 정규화가 PopArt 아님"
(→ PopArt라 부른 적 없음) 등이 포함된다.

생존 2건:

| # | 지적 | 판정 | 조치 |
|---|---|---|---|
| 1 | 모든 런이 학습 시드 2024 (n=1) | **major** | §5 — 복제 1개 또는 한계 명시 |
| 2 | RESEARCH_LOG의 FLOP 수치 5배 과소 | nit(문서) | §2 — ~40 MFLOPs로 정정 완료 |

특히 중요한 **반박된** 지적: "rollout/replay 분포 불일치"는 구조적으로 불가능하다.
`decode()`와 `replay()`가 **동일한 `_rbg_major_pass` 코드 경로**를 타고
(`policy.py:484-562`), 결정 마스크 일치를 assert로 강제한다(`policy.py:516-522`).
구조적 행동 공간 PPO에서 가장 흔한 치명적 버그가 설계로 차단되어 있다.

---

## 9. 아직 검증 중 / 미결

- [ ] **학습 시드 복제** (`--seed 2025`) 또는 단일 시드 한계 명시 — §5, 유일한 major.
- [ ] `docs/RESEARCH_LOG.md:100` FLOP 수치 실제로 고치기 (이 문서 §2에는 정정본 있음).
- [ ] 최종 표를 17 시드로 갈지 20 시드+각주로 갈지 결정 (§5).
- [ ] `beta_rate` 제거(항상 1.0, β_m이 대체) — paper-gen 배치에서.
- [ ] 병렬 롤아웃 도입 여부 (EV·학습속도, Run5 후보).

### 로깅 관련 사소한 주의 (결과에는 영향 없음)

- `grad_norm`은 `gn_actor + gn_critic`, 즉 **두 L2 노름의 L1 합이며 clip 이전 값**이고
  actor 동결 패스(0)도 포함한다 (`ppo.py:265-271`). 그림으로 쓸 거면 각주 필수.
- `clip_fraction`은 advantage 부호와 무관하게 `|ratio−1| > ε`를 센다(SB3 관례).
  실제로 clip이 구속하는 비율은 로그값보다 낮다.
- `config.py`의 `share_critic_encoder`는 **어디서도 읽지 않는 죽은 플래그**다.
  인코더는 "공유"라고만 쓰고, 설정 가능한 것처럼 쓰거나 비공유 ablation이 있는 것처럼
  암시하지 말 것.
- Run3의 일부 `env_metrics.csv`는 resume 이후 헤더(11열)와 행(14열)이 어긋나 있다.
  supplementary로 원본 그대로 배포하지 말 것.

---

## 부록 A. 용어 정정 이력

- **β_m은 "back-off"가 아니라 "depth-wise calibration factor"**. CQI4 세계 재보정에서
  β₁ = 1.0018 > 1 이 나왔으므로 back-off라는 표현은 모순이다.
- 에피소드 길이는 **1000 slot**. (평가에서 보이는 3000은 3 에피소드 합계.)
- CQI 양자화의 index 0 = "out of range"는 **3GPP 표준 의미**이고,
  q=0 → ĥ=0 → 신규 배제/고정 retx의 zero-beam outage는 **본 시뮬레이터의 추상화**다.
