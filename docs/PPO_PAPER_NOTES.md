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
운용에 그대로 쓸 수 있고, critic의 출력은 행동에 전혀 관여하지 않으므로 배치에
privileged 정보가 필요 없다.

> ⚠️ **표현 주의 (2026-08-21 외부 리뷰에서 지적, 사실 확인됨).**
> "the critic is not evaluated at deployment" 류로 쓰면 **틀린다**. `value_head`는
> 배포 시 **모든 슬롯에서 실행된다** (`policy.py`의 `_rbg_major_pass` 끝, decode가
> 그 경로를 탄다). 출력이 `PPOScheduler.schedule`에서 **버려질 뿐**이다
> (`train_phase2.py:209`, `eval_phase2.py:33`). 정확한 서술은
> *"the critic's output is discarded at deployment and never influences an action"*.
> 실무적 함의도 있다: critic 전용 특징을 추가하려면 반드시 `env.get_observation()`
> 을 거쳐야 한다. `obs_to_tensors`가 11개 키 화이트리스트 하드코딩이고 value 호출
> **이전에** 무조건 실행되므로, decode의 value 호출만 가드해서는 소용없다.

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

### ✅ 단일 학습 시드 — **해소됨** (2026-08-24)

정밀 감사에서 살아남은 유일한 major였다. 시드 3024/4024/5024로 **독립 복제 런 3개**를
동일 절차·동일 예산(866 update)으로 완주했고, 네 정책 전부를 **같은 예약 100
에피소드**에서 평가했다.

> ⚠️ 함정: `env.reset(k)`는 `cfg.seed + <계수>·k`로 뽑으므로 **평가 에피소드가
> cfg.seed에 의존한다**(`env.py:87/90/114/124`). 각 복제런의 자기 config를 쓰면 네
> 정책이 서로 다른 100개 에피소드에서 평가되어 비교가 무의미해진다. 세계 정의는
> 항상 seed-2024 config에서 만들고 체크포인트만 바꿨다
> (`Run4/_analysis/scripts/seedreplicate_final100.py`, assert로 강제).

| train seed | best@ | reward | goodput% | miss% | depth | vs SUS+CQI | 승 | paired 95% CI |
|---|---|---|---|---|---|---|---|---|
| 2024 | 409 | 4836 | 97.71 | 36.27 | 2.78 | **+15.32%** | 99/100 | [+596, +689] |
| 3024 | 489 | 4665 | 97.92 | 36.69 | 2.77 | **+11.24%** | 92/100 | [+416, +527] |
| 4024 | 849 | 4726 | 97.83 | 36.56 | 2.77 | **+12.68%** | 96/100 | [+485, +579] |
| 5024 | 389 | 4729 | 98.06 | 36.49 | 2.71 | **+12.76%** | 95/100 | [+480, +590] |
| SUS+CQI | — | 4194 | — | — | — | — | — | — |

```
PPO reward  4739 ± 71      범위 [4665, 4836]    변동계수 1.5%
마진(%)     +13.00 ± 1.70  범위 [+11.24, +15.32]
최저 승률   92/100
```

**§V-F 문장 (최악 시드 기준 — 평균±표준편차보다 강하고 n=4에 덜 민감하다):**

> *"Across four independent training seeds (2024/3024/4024/5024), identical procedure
> and budget, evaluated on the same reserved 100 episodes: mean reward 4739 ± 71,
> margin over the strongest baseline +13.00% ± 1.70%. The worst seed still exceeds
> SUS+CQI by +11.24% with a paired 95% CI of [+416, +527], and the lowest per-seed
> win rate is 92/100."*

**학습 로그의 best eval reward를 시드 비교에 쓰면 안 된다.** 그 값은 2786~5144로 1.8배
벌어져 있지만, 검증 3 에피소드가 **cfg.seed마다 다른 draw**이기 때문이다. 동일
에피소드에서 재면 4665~4836, **3.7% 이내**다. 시드 3024가 낮았던 건 정책이 나빠서가
아니라 그 시드가 뽑은 3개가 모두에게 어려웠기 때문이다(같은 에피소드에서 SUS+PF가
−3227까지 내려간다). s4024의 `best@849`(거의 끝)도 held-out에서는 중간값이다.

무결성 3종 통과: 800행 누락 없음 / SUS+CQI 행이 네 정책에서 완전 동일(2100건 대조,
불일치 0) / 2024 정책이 기존 `queue_s40hl_cqi4_final100.csv`를 정확히 재현(1400건,
불일치 0).

---

## 6. explained variance — **지표 자체가 부적합하다** (2026-08-21 측정)

실측: HighLoad는 초반 +0.07 → 종반 **+0.26**, CQI4 본 런은 평균 0.1785 / 마지막 100
update 0.3105. 낮아 보이는 이 값을 **critic 품질의 척도로 논문에 인용하면 안 된다.**
전용 진단(`Run4/_analysis/scripts/audit_probes/critic_ev_decomposition.py`, 동결
best.pt@409, 시드 60000–60039, n=40)이 이유를 정량화했다.

**(a) EV는 return 분산의 2.7%만 잰다.**

```
Var_between (에피소드 평균)  26068.8   97.3 %
Var_within                     730.9    2.7 %
```

한 PPO update의 배치가 **에피소드 하나**이므로(`train_phase2.py:659`), 로깅되는 EV는
within-episode 양이다. 그리고 `ppo.py:79`가 `returns = advantages + values`,
`ppo.py:284`가 같은 `values`를 다시 쓰므로 **EV의 잔차는 GAE advantage 그 자체**다:

```
explained_variance = 1 − Var(Â)/Var(Â + V)
```

즉 V의 레벨(DC 성분)에는 구조적으로 눈이 멀어 있다.

**(b) 그 2.7% 안에서도 critic은 시간 추세만큼도 못 잡는다.**

| 예측기 | within-episode EV |
|---|---|
| episode phase `t/T` 단독 (3차식) | **+0.4388** |
| 학습된 critic | **+0.0556** |
| 학습된 critic, pooled | +0.0913 |

pooled > within 이므로 critic의 강점은 within-episode 형태가 아니라 **레벨 추적**이다.

**(c) 레벨 오차는 실재하고, 그게 value_loss가 보는 것이다.**

`ppo.py:208`이 회귀하는 **GAE return** 기준으로 에피소드별 bias `|평균| 22.75`
(정규화 단위 1.61 σ), **value MSE의 85.4%가 레벨 성분**이다. MC return 기준으로는
97.7%. EV는 못 보지만 value_loss와 (detach 없는 단일 backward를 통해) 공유 encoder는
본다.

**논문 서술 권장**: EV 수치를 critic 품질 근거로 쓰지 말고, 대신
*"the logged explained variance is a within-episode quantity and accounts for only
2.7% of the return variance in this setting; we therefore do not use it as a measure
of critic quality"* 라고 명시한다. 이건 방어가 아니라 정직한 계측 서술이다.

---

## 6b. privileged critic 제안 — **측정으로 영구 종결** (2026-08-21)

외부 리뷰가 critic 입력에 에피소드 latent(`K_act`, `p_a`, per-UE 속도)를 추가하면
EV가 0.5 → 0.7이 될 것이라고 제안했다. **held-out 측정으로 기각되었다.**

에피소드 단위 held-out ridge (fit 24 / alpha-select 6 / **test 10** 에피소드):

| 특징 집합 | pooled EV | pooled R² |
|---|---|---|
| 관측 특징만 (167차원) | **+0.2647** | +0.2633 |
| 관측 + latent 4차원 | **+0.1895** | +0.1832 |
| **한계 기여 (c2 − c1)** | **−0.0751** | **−0.0801** |

**latent를 넣으면 일반화가 나빠진다.** 이미 관측 특징이 그 정보를 더 잘 나른다 —
`policy.py`의 `active_count`와 `actf.sum()/K`가 `K_act`의 거의 무잡음 대리변수이고,
`p_a`의 효과는 큐 상태 전체가 매개한다.

부수적으로, "약한 critic이 clip 포화를 유발한다"는 가설도 기각됐다. 867 update 전수에서
`corr(EV, clip_fraction) = **+0.459**`, `corr(EV, approx_KL) = +0.423` — 가설이 예측하는
음의 상관과 **부호가 반대**다 (초반 추세를 제거한 update 400 이후에도 +0.290).

§V-F 한 문장 후보:
> *"Privileged per-episode latents were measured offline on held-out episodes and
> reduced generalization (ΔEV = −0.075), so the critic's observation set was left
> unchanged."*

슬롯별 원자료는 `Run4/_analysis/critic_ev_decomposition_raw.npz`에 보존되어 있어
후속 질문에 재실행이 필요 없다.

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

- [x] ~~학습 시드 복제~~ → **완료** (시드 3×, §5). 유일한 major 해소.
- [x] ~~privileged critic 제안~~ → **측정으로 종결** (§6b).
- [x] ~~docstring 차원 수치 4건~~ → 정정 (`23bc8aa` 이래 stale, Run4 queue mode 때문).
- [ ] `docs/RESEARCH_LOG.md:100` FLOP 수치 실제로 고치기 (이 문서 §2에 정정본 있음).
- [ ] `beta_rate` 제거(항상 1.0, β_m이 대체) — paper-gen 배치에서.
- [ ] Table III 각주: deadline bin은 half-open `(·,·]`이고 decision epoch에서 `d ≥ 1`
      (`_V2_DEADLINE_BINS` 첫 bin이 `(0.0,1.5]`라 `dl==0`이 어느 bin에도 안 들어가지만,
      `env.py`가 만료 패킷을 관측 생성 전에 제거하므로 도달 불가).
- [ ] 배치 replay의 **다지점 ON/OFF 등가성** — 현재 등가성은 가중치 한 지점에서만
      측정됐다. 서로 다른 학습 단계의 체크포인트 4개(389/409/489/849)에서 재확인 필요.
- [ ] 배치 replay의 **자기 재현성** — `replay_batch`의 `e.permute(...)[slot_t, r_t]`는
      한 RBG의 4개 레이어가 같은 `(slot,r)`을 가리키는 many-to-one 인덱싱이라, 역전파가
      CUDA `atomicAdd`를 쓰면 같은 시드로 두 번 돌려도 결과가 다를 수 있다. 미측정.

## 9b. 배치 PPO update replay (2026-08-24 병합, 기본 OFF)

`--batched_replay`. update가 벽시계의 89%였고 `replay()`를 **샘플마다** 불러
update당 약 4000회의 순차 ~30위치 루프를 돌아 GPU가 7~8%에 머물렀다. teacher
forcing 하에서는 모든 head 입력과 valid_mask가 (obs, 저장된 action)의 결정론적
함수이므로, decode가 이미 만들고 버리던 텐서를 캐싱해 minibatch 전체를 **4회의
batched forward**로 처리한다.

| | |
|---|---|
| 속도 | 551.9 → **70.9 s/update** (동일 조건). 866 update: 4~5.5일 → **약 17시간** |
| 플래그 OFF | 원본 866-update 런의 update 0/1/2를 **로그 소수점까지 재현**(18건, 불일치 0). 순차 decode/replay/state_value는 병합 전 트리와 **비트 단위 동일**(24슬롯) |
| 등가성 | forward ≤3.1e-05 / gradient(랜덤 가중치, head별) ≤5.9e-06 / `ppo_update` 7개 지표 ≤1.5e-06 (**KL-frozen 분기 포함**, 그 분기에서 encoder 스텝은 양쪽 모두 정확히 0) |
| 플래그 ON | **비트 재현 불가.** float32 축약 순서가 달라 gradient가 ~1e-6 다르고, 에피소드당 약 3만 회의 categorical 샘플링 중 하나가 경계를 넘으면 궤적이 갈린다(실측: update 0·1 동일, update 2 분기) |

**용도**: 앞으로의 실험(§V-F가 약속한 architectural ablation, 추가 시드). **기존 런
재현용이 아니다** — 논문 재현 지시에는 플래그를 붙이지 않는다는 점이 명시돼야 한다.

17-에이전트 적대적 감사(BLOCKER 0, MAJOR 11, cleared 224) 후 보강:
`_position_logits_and_mask`는 원래 3-튜플 반환을 유지(외부 프로브 3개 무영향);
`replay_batch`가 순차 경로의 정합성 assert를 이미 지불 중인 device sync에 접어 복원;
런타임 교차검사는 **옵티마이저가 실제 소비한 텐서**를 검사하고 value를 정규화 단위로
비교하며 실패 시 **raise 대신 순차로 폴백**(assert였다면 프로세스가 죽고, auto-resume이
torch RNG를 복원해 같은 롤아웃·같은 검사를 재현하며 무한 크래시 루프가 된다);
`ppo_batched_replay`/`ppo_critic_v2`를 resume cfg-mismatch 경고 목록에 추가.

테스트: `tests/test_cross_tree_identity.py`, `tests/test_replay_batch.py`,
`tests/test_ppo_update_batched.py` (원 출력 `tests/results/*.log`).

## 9c. `env.py:124` 중복 시드 — 알려진 아티팩트, 의도적 미수정

`_rp = default_rng(cfg.seed + 7919 * episode_idx)` (env.py:124)가 `self.rng`
(env.py:87)와 **계수까지 동일**하다. `n_active`는 `4441 *`로 분리했는데 `p_arrival`만
누락됐다. 결과적으로 `p_arrival_ep = 0.15 + 0.35·u₀`이고 `u₀`가 `self.rng`의 첫 draw와
같아, **UE 0의 슬롯 0 도착 여부가 p_a와 결정론적으로 묶인다**(u₀ < 0.2308일 때만 도착).

바로 위 주석의 *"dedicated RNG stream ... zero impact on all other draws"* 는 절반만
참이다 — 스트림을 교란하지는 않지만 **완벽히 중복**된다.

영향: 에피소드당 약 32,000회 도착 판정 중 **정확히 1회**. 어떤 지표에도 측정 가능한
영향이 없다. 고치면 모든 에피소드의 난수열이 바뀌어 전 결과가 무효가 되므로
**수정하지 않는다.** 논문 본문이 아니라 README/재현성 부록에 기록할 것.

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
