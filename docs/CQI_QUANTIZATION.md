# 4-bit CQI 양자화 — 설계·구현·결과 (논문용)

작성 2026-08-05. 코드는 main에 병합 완료(`1d85999`), 결과 커밋 `1376c38`·`a305329`.
근거: `csi.py` (표+`quantize_cqi`), `config.py` (`cqi_mode`+가드),
`Run4/QueuePostRZF_S40HL_CQI4/`, `Run4/_analysis/queue_s40hl_cqi4_final20.csv`.

관련 문서: [BETA_M_RATIONALE.md](BETA_M_RATIONALE.md) (β_m 재보정),
[PPO_PAPER_NOTES.md](PPO_PAPER_NOTES.md) (학습 알고리즘), RESEARCH_LOG 2026-08-01·08-04.

---

## 1. 왜 했는가

기존 세계의 CSI 사슬은 **방향은 양자화(Type-II-like 56-bit PMI)되고 신선도도
제한(p_csi=0.6)되지만, CQI만 연속 실수**였다. 즉 "채널 품질 보고"만 이상화되어
있었고, 이건 리뷰어가 realism gap으로 지적할 수 있는 지점이었다.

**창설 질문**: *보고가 거칠어지면 학습된 스케줄러의 마진이 살아남는가?*

사전 예측이 있었다 — V50/V60/CSI04 프로브에서 "정보가 거칠수록 상대 우위는 유지되거나
커진다"는 패턴이 반복됐으므로, 마진이 보존될 것으로 봤다. 결과는 예측 적중(§5).

---

## 2. 무엇을 양자화했고 무엇은 안 했는가

**양자화한 것: UE의 CQI 보고(정보)뿐.**

```
연속 SE = log₂(1 + P|h^H d̂|²/σ²)
   → 3GPP TS 38.214 Table 5.2.2.1-3 (4-bit, 256QAM) 사다리에 floor 스냅
```

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| SE | **0** (out of range) | 0.1523 | 0.3770 | 0.8770 | 1.4766 | 1.9141 | 2.4063 | 2.7305 |
| idx | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| SE | 3.3223 | 3.9023 | 4.5234 | 5.1152 | 5.5547 | 6.2266 | 6.9141 | **7.4063** |

- **floor(내림)** 을 쓴 이유: 표준의 UE 규칙이 "BLER ≤ 10%를 만족하는 **가장 높은**
  인덱스"이고, 우리 세계에서 그에 대응하는 Shannon-추상화가 자기 SE 이하의 최대 표값이다.
  보수 방향이라 과약속을 만들지 않는다.
- **상한 포화**: 실측 분포에서 22.8%가 7.4063에 붙는다. 실질 비용은 작다 —
  1344 × 7.4063 ≈ **9,954 bits/RBG-slot**이고 패킷이 4–12k bits라, 상한에서도 한 RBG가
  패킷 하나를 거의 소화한다. (실제 시스템도 256QAM 상한 위 SNR은 버린다.)
- **하한 미달**(11%): index 0 = "out of range".

**양자화하지 않은 것: 전송 포맷(행동).** MCS 사다리와 동적 OLLA는 여전히 모델링하지
않고, 연속 rate 약속 + β_m이 그 자리를 대신한다. 표준에서도 CQI 생성은 MCS 결정과
독립이므로(UE는 자기 보고가 어떤 MCS가 될지 모른다) 이 조합은 정합적이다.
자세한 논거는 [BETA_M_RATIONALE.md](BETA_M_RATIONALE.md) §2.

### index 0의 의미 — 표준 vs 우리 추상화 (구분해서 쓸 것)

- **표준**: "out of range" = 그 RBG에서 스케줄하지 않는다.
- **본 시뮬레이터의 추상화**: q=0 → `reconstruct_h_hat`이 ĥ=0을 내놓음 →
  ① 신규 배치는 후보 게이트가 배제, ② **이미 고정된 HARQ 재전송은 게이트를 우회**하므로
  RZF에서 zero beam(무전력 시도) → SINR 0 → NACK. NaN 없음.
  회귀 테스트로 고정: `Run4/_analysis/scripts/audit_probes/cqi0_retx_regression.py`
  (유닛: zero row 포함 그룹에서 죽은 빔 노름 0·SINR 0·생존 스트림 정상 / 통합: 실제
  env에서 retx UE의 CQI를 0으로 강제해도 스텝 생존·reward 유한).

---

## 3. 구현 (main 병합 완료)

변경은 **4개 파일**뿐이다. 생산 경로가 단일 지점이라 아래 전체가 자동 상속한다:
BS의 ĥ 재구성 크기 → SUS+CQI의 선택 지표 → LA의 B_tx 예측 → PPO 관측.
**PPO와 baseline이 같은 거친 정보를 보므로 공정성이 자동으로 보장된다.**

| 파일 | 내용 |
|---|---|
| `config.py` | `cqi_mode: 'continuous' \| 'nr4bit'` + `validate_la()` 가드 2건 |
| `csi.py` | `NR_CQI_TABLE_256QAM` + `quantize_cqi()`; `precompute_episode_csi`의 CQI 계산 직후 적용 |
| `train_phase2.py` | `--cqi_mode` CLI + resume cfg-mismatch 경고 키 |
| `cqi0_retx_regression.py` | 회귀 테스트 (신규) |

**가드**: `pmi_mode='genie'` + `cqi_mode='nr4bit'` 조합은 `ValueError`로 금지.
genie는 정의상 perfect CSI(=양자화되지 않은 보고)이므로, 양자화하면 제3의 세계가 된다.
→ **이 가드 때문에 genie 챕터는 CQI4 세계에서 재진술이 불가능**하다. 논문이 두 세계
구성을 갖는 구조적 이유다(§6).

### 인코더 차원은 바뀌지 않는다 (자주 나오는 오해)

CQI는 여전히 **실수 스칼라 1칸**(÷8.0 정규화)이다. 양자화는 값의 *해상도*만 바꾼다.
실측: 인코더 입력 shape는 두 세계 모두 **(32, 8, 73)**, actor 파라미터 **68,930** 동일.
CQI 열의 고유값만 256개 → **16개**로, 범위는 [0.0002,1.3351] → [0.0000,0.9258]로 바뀐다.

- 덕분에 **가중치가 두 세계 사이에서 strict 로드된다** → CQI 해상도를 순수한 환경 축으로
  다룰 수 있고(동결 전이 실험 가능), 성능 차이를 모델 용량 탓으로 돌릴 여지가 없다.
- 설계 선택: 16-way one-hot이 아니라 **SE 값 그대로** 넣는다. 사다리는 순서와 크기가
  의미 있는 양(SE 3.32 ≈ 1.48의 2.2배)이라 one-hot은 그 metric 구조를 버린다.
  논문 한 줄: *"the quantized CQI is fed as its spectral-efficiency value rather than as
  a categorical index, preserving the ordinal and metric structure of the ladder."*

---

## 4. 검증 게이트 (발사 전 전부 통과)

1. **유닛**: floor 성질(q ≤ x)·표 원소 소속·경계값(0.05→0, 7.4062→6.9141, 12.25→7.4063)·
   continuous 모드 불변·genie 가드 작동.
2. **회귀**: CQI-0 retx 코너 케이스(§2) — zero beam = clean outage, NaN 없음.
3. **3-seed 동결 평가** (양자화 부담이 공평한지): PPO 5594→5029, SUS 4937→4354.
   양쪽 −10% 수준으로 동등 부담, 이상 거동 없음. 부수 소득: continuous로 학습한 정책이
   양자화 세계에서 zero-shot으로도 +15.5% 우위.
4. **smoke 12 update**: NaN 0행, update-9 eval에서 13종 스케줄러 전부 정상.
   → smoke를 폐기하지 않고 본 런으로 **승격**(동일 CLI·seed이므로 update 0–11이 정사).
5. **β_m 재보정** (frozen 10-pct recipe): §5 표.

---

## 5. 결과

### β_m 재보정 (세계가 바뀌면 반드시)

| depth | continuous S40 | **CQI4** | 변화 |
|---|---|---|---|
| 1 | 0.9757 | **1.0018** | +2.7% |
| 2 | 0.7180 | **0.7499** | +4.4% |
| 3 | 0.6355 | **0.6592** | +3.7% |
| 4 | 0.5837 | **0.6058** | +3.8% |

holdout first-ACK 0.869 / 0.886 / 0.890 / 0.884 (~90% 목표 정합).
**전 depth에서 상승**한 이유: floor 양자화가 이미 보고 단계에서 보수성을 넣으므로 β가
추가로 깎을 몫이 줄어든다(이중 보수화 제거). **β₁ = 1.0018 > 1** 이므로 명칭은
back-off가 아니라 **depth-wise calibration factor**.

### in-world SUS threshold 재스윕 (8 seeds, 이 세계 첫 스윕)

**thr 0.7 최강** 4072 (0.5: 3999 / 0.6: 4058 / **0.75(config 상속값): 4031** / 0.8: 3975 /
0.9: 3780). 상속값 0.75를 그대로 썼다면 baseline이 ~42 손해 — **세계마다 재스윕한다**는
규약의 4번째 실증. 논문에서 "baseline이 약해서 이긴 것 아니냐"는 공격의 방어 근거.

### Held-out 20 seeds (10000–10019, SUS@0.7)

| | PPO (best@409) | SUS+CQI@0.7 | 마진 |
|---|---|---|---|
| reward | **4755** | 4174 | **+13.9%, 20/20 전승** (per-seed +89~+1143) |
| goodput | 80.0 | 77.4 | +3.4% |
| miss | 0.362 | 0.388 | −2.6%p |
| MU depth | **2.74** | 3.92 | 선별 유지 |

showcase seed 10017(마진 +1143), 8-panel 그림은 run 폴더.

### ⭐ 정보-축 3점 세트 (동일 세계 계열·동일 프로토콜·세계 내 스윕된 SUS+CQI 대비)

| CSI 품질 | 마진 | 승수 |
|---|---|---|
| perfect CSI (genie) | **+4.3%** | 20/20 |
| continuous CQI (56-bit PMI + 실수 CQI) | **+14.7%** | 20/20 |
| **4-bit CQI (56-bit PMI + 16-단계 CQI)** | **+13.9%** | 20/20 |

**판독 셋**:
1. **창설 질문에 YES** — 14.7% → 13.9%는 사실상 보존. 표준 준수 보고에서도 마진이 산다.
2. **양자화 비용은 전원 부담** — PPO 5352→4755(−11.2%), SUS 4668→4174(−10.6%).
   절대 성능은 다 같이 내려가고 **상대 마진만 유지**된다.
3. **정보가 완벽해질 때만 마진이 수축**(+4.3%) — "PPO 우위는 CSI 불완전성이 만드는 결정
   문제에서 나온다"는 축의 세 번째 직접 증거. 전략도 일관: depth 2.74(CQI4) /
   2.90(continuous) / 3.48(genie).

---

## 6. 논문 서술 (복사용)

**System model 절**:
> "Each UE reports a 4-bit CQI index following the 256QAM table of 3GPP TS 38.214
> (Table 5.2.2.1-3). Index selection uses the Shannon-SE floor rule as the
> link-abstraction analog of the standard's BLER ≤ 0.1 selection criterion; index 0
> ('out of range') renders the UE unschedulable on that RBG. Transport-format
> granularity (MCS) and outer-loop adjustment remain abstracted by a continuous rate
> promise with depth-wise calibration factors β_m."

**Robustness 절 (7번째 축)**:
> "The advantage is not an artifact of an unquantized CQI report: retraining under the
> standard 4-bit ladder preserves the margin over the strongest in-world-tuned heuristic
> (+13.9%, 20/20 seeds, versus +14.7% with a continuous report), while the quantization
> cost is borne equally by both schedulers (−11.2% and −10.6% in absolute reward)."

**정보-축 문장**:
> "Across a monotone CSI-fidelity axis — 4-bit report, unquantized report, and perfect
> CSI — the learned policy's margin over the per-world-tuned heuristic is +13.9%, +14.7%
> and +4.3% respectively, indicating that the advantage originates in the decision problem
> created by CSI imperfection rather than in the physical layer itself."

---

## 7. 캐비앳 (정직하게 밝힐 것)

- **CQI4 판정은 조기 은퇴한 팔의 best@409 기준**이다. upd ~650 이후 과냉각 하강
  (창평균 4943→4595→4141, depth 2.66→1.95, entropy 0.597 — S40Ent02 병리 재현)이 있었고
  학습은 866/1500 update에서 중단됐다. entropy 스케줄이 더 좋았다면 상한이 더 높았을 수
  있으므로 **보수적 방향의 캐비앳**이며 판정에는 무해하다.
- run-eval 5039 → held-out 4755 수축은 3-seed run-eval의 노이즈 범위.
- 절대 성능을 **세계 간에 직접 비교하지 말 것** — σ² 보정이 세계마다 다르다.
  비교는 각 세계 안에서 스윕된 baseline 대비 마진으로만.
- 학습 시드는 여전히 2024 단일(전 런 공통). [PPO_PAPER_NOTES.md](PPO_PAPER_NOTES.md) §5 참조.

---

## 8. 재현 방법

```bash
# 학습 (본 런과 동일 구성)
python3 train_phase2.py --mode queue \
  --p_arrival_min 0.15 --p_arrival_max 0.50 \
  --ue_speed_min 5 --ue_speed_max 40 \
  --la_mode post_rzf --decode_order rbg_major \
  --la_beta_by_depth 1.0018,0.7499,0.6592,0.6058 \
  --cqi_mode nr4bit \
  --entropy_coef 0.02 --ppo_save_every 1 --seed 2024 \
  --run_root Run4 --run_name <name>

# β_m 재보정 (세계를 바꿨다면 필수)
python3 Run4/_analysis/scripts/audit_probes/beta_m_calib_queue_s40hl_cqi4.py

# 회귀 테스트
python3 Run4/_analysis/scripts/audit_probes/cqi0_retx_regression.py

# held-out 20-seed + 그림
python3 Run4/_analysis/scripts/queue_s40hl_cqi4_final_figure.py
```

위 스크립트들은 작성 당시 `_cqi4dev` worktree를 import했으나(당시 main에는 `cqi_mode`가
없었다), 병합 후 **전부 main 트리를 가리키도록 수정 완료**했다. 회귀 테스트는 수정 후
재실행해 PASS 확인. worktree는 제거됐고 저장소에 남은 `_cqi4dev` 참조는 0건이다.
