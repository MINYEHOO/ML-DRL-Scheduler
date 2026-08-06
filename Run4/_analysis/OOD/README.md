# OOD / Zero-Shot Generalization — 통합 아카이브

작성 2026-08-06. **메인 정책 = QueuePostRZF_S40HL_CQI4 best.pt@409** (4-bit NR CQI
세계, held-out +13.9% 20/20). 이 폴더는 OOD·zero-shot 계열의 모든 스크립트
(`scripts/`)와 결과(`results/`)를 모은 아카이브이며, 그림은 관례대로 각 run
폴더에 있다 (아래 링크). 논문 기준 확정 결과는 §2(사전등록 n=40 그리드)와
§3(K-scaling)이고, 나머지 절은 탐색 이력이다.

---

## 1. 프로토콜 불변 조건 (전 프로브 공통)

- **정책 완전 동결**: 재학습·미세조정 없음. β_m도 학습 세계 보정값 유지
  (zero-shot 전제 — OOD 세계에서는 의도적으로 미스매치 상태).
- **baseline은 세계마다 재튜닝**: 각 시나리오에서 SUS+CQI threshold를
  {0.60–0.80} 스윕해 최강값 채택 + SU+CQI anchor. 즉 비교는
  "제자리 재튜닝된 최강 휴리스틱 vs 동결 정책" — **마진은 보수적 하한**.
  (튜닝은 baseline에만 적용되는 자유도 — 정책은 평가-시 선택 파라미터 없음.)
- 시드 대역: **40000대 = pilot(threshold 선택·챔피언 검증) / 30000대 =
  test(n=40 평가)** — §2는 이 분리를 따름. 구판 8-seed 프로브(§4)는
  30000–30007에서 선택·평가를 겸했으므로 그 한계를 명시하고 인용할 것.
  **최종 논문 숫자용 20000+는 미접촉 예약**.
- 판정 규칙: n=8은 "한 자릿수 마진 = 지지 않음"(K48 규칙); n=40은 paired
  Student-t 95% CI로 "CI>0 확정 / tie" 판정.
- 에피소드 조건(활성 인원 등)은 시드가 결정 → 시나리오 간 paired 비교 성립.
  참고: 30002(활성 32명)·30006(28명)이 최대 포화 시드.

## 2. ⭐논문 그리드 (사전등록 n=40, seeds 30000–30039) — 확정 결과

**2026-08-06 감사 수정 반영판** (외부 리뷰 3건 처리):
- **pilot/test 분리**: SUS threshold 선택을 미사용 시드 **40000–40007**로 이전
  → 30000–30039는 완전 독립 test set. (구판은 30000–30007이 튜닝·평가 양쪽에
  포함 = baseline에 유리한 누출이었음 → 우리 마진의 과소평가 방향)
- **챔피언 검증**: pilot에서 **12-baseline 전수** 평가 + SUS+MW 전용 threshold
  스윕까지 실시 → 6/6 세계에서 챔피언 정체 불변(SUS+CQI, P010만 SU+CQI).
  MW는 자기 최적 threshold에서도 격차 +212~+284로 열세. D26에서
  deadline-aware(EDF/DPF)는 2위조차 못 함.
- **D26 strict**: 원판은 deadline_max 12→6 변경이 정책 관측 정규화까지 바꿔
  엄밀한 zero-shot이 아니었음. 재실행판은 **env=U[2,6] / 정규화=/12(학습값
  동결)** 분리. 표의 D26 행은 strict 결과.

설계: 축당 1개, 분포-연장 방식(V는 상한 연장), STORM2 = 단일-축 정의 3개를
문자 그대로 동시 인가. 스크립트 `scripts/ood_paper_grid_cqi4.py`(+
`ood_pilot40k_cqi4.py`, `ood_pilot40k_musweep.py`, `ood_reeval_sus_tstar.py`,
`ood_d26_strict_rerun.py`), 데이터 `results/ood_paper_grid_cqi4_merged.csv`
(+ `ood_reeval_sus_t*.csv`, `ood_d26_strict.csv`).

| 시나리오 | 정의 (학습분포 대비) | PPO | 최강 baseline (세계 내 재튜닝, 40k pilot) | paired ±CI | 판정 | 승수 | 층화(하/중/상) |
|---|---|---|---|---|---|---|---|
| P055 | p_a=0.55 고정 (상한 0.50 초과) | 2448 | SUS+CQI@0.75 1887 | **+561±168** | **CI>0 확정** | 35/40 | +809/+653/+244 |
| P010 | p_a=0.10 고정 (하한 0.15 미만) | 2971 | SU+CQI 2963 | +9±16 | tie (설계상 예상) | 28/40 | +12/+4/+11 |
| V60max | 속도 U(5,**60**) (상한 연장) | 4399 | SUS+CQI@0.75 3843 | **+556±108** | **CI>0 확정** | 36/40 | +758/+607/+300 |
| CSI02 | p_csi 0.6→**0.2** (실환경 등가 밀도) | 3589 | SUS+CQI@0.70 3145 | **+444±92** | **CI>0 확정** | 37/40 | +603/+489/+238 |
| D26 (strict) | deadline U[3,12]→**U[2,6]** | 4403 | SUS+CQI@0.80 3673 | **+730±115** | **CI>0 확정** | **39/40** | +859/+719/+570 |
| STORM2 | P055+V60max+CSI02 **동시** | 884 | SUS+CQI@0.75 386 | **+498±158** | **CI>0 확정** | 35/40 | +710/+624/+148 |

보수적 참고(구 pilot 시드 8개까지 제외한 **test32**): P055 +639±175(30/32) /
P010 +16±17(tie, 24/32) / V60max +591±112(30/32) / CSI02 +480±96(31/32) /
D26 +782±122(**32/32**) / STORM2 +584±167(30/32) — 판정 전부 동일, 마진은
오히려 상승(구 pilot 시드가 baseline에 유리했음을 재확인).

핵심 판독:
1. **5/6 확정 승 + 저부하 설계된 tie 1** — 조건 축 어디에도 붕괴 없음.
2. **D26(마감 축)이 최강** (39/40, strict 기준 +730): 긴급도 대응이 정책의
   실제 자산. 정규화 분리 후에도 결과 사실상 불변(+716→+730).
3. **STORM2 상호작용**: 단일-축 paired가 +561/+556/+444인데 3축 동시가
   +498 — **파괴적 상호작용 없음** (baseline 386 붕괴 → %말고 paired로 서술).
4. **층화(n_active 3분위)**: 확정-승 시나리오 전부 최고부하 3분위도 양수.
5. P010에서 최강 baseline이 SU+CQI로 교체 — 저부하→SU 최강 regime-map 일치.

**논문 표시 baseline 세트(2026-08-06 확정)**: 본문은 **6개 = {SUS, SU} ×
{CQI, PF, Random}**. DPF/MW/EDF는 부록 전체표·각주로만(위 챔피언 검증이
"wider 12-grid에서도 챔피언 불변"의 근거).

## 3. ⭐K-scaling (인구 축 zero-shot, CQI4판, n=8, seeds 30000–30007)

K별로 세계 재생성(num_ue=K, 활성 U{K/2..K}, K별 SUS 재스윕), 같은 ckpt 로드
(파라미터 K-독립). β_m은 K=32 보정 유지. 스크립트
`scripts/scaling_zeroshot_cqi4.py`, 데이터 `results/scaling_zeroshot_cqi4.csv`.

| K | 활성 | PPO | 최강 baseline | 마진 | 승수 |
|---|---|---|---|---|---|
| 16 | U{8..16} | 3398 | SUS@0.80 3172 | +7.1% | 7/8 |
| 24 | U{12..24} | 4829 | SUS@0.65 4260 | +13.4% | 8/8 |
| 32 (학습) | U{16..32} | 3876 | SUS@0.70 3481 | +11.4% | 7/8 |
| 48 | U{24..48} | 3130 | SUS@0.70 2591 | **+20.8%** | 8/8 |

판독: 전 구간 승리(합산 30/32). **K48에서 마진 최대(+20.8%)** — continuous판
(Ent02, +0.7% tie)과 정반대인데, CQI4 정책은 HighLoad(p~U(0.15,0.50))에서
학습돼 K48의 1.5× 부하를 오히려 잘 다룬다(adaptation-gap 발견과 정합).
주의: K 축은 인구와 부하가 얽힘(UE당 p_a 불변 → K↑ = 부하↑) — 논문에 명시.
비교용 continuous판(Ent02): K16 +4.9 / K24 +20.5 / K32 +10.7 / K48 +0.7
(`results/scaling_zeroshot.csv`).

## 4. 탐색 이력 (8-seed probes; 부록/서술용)

### 4.1 6-세계 그리드 구판 (고정값 방식; 논문 그리드로 대체됨)
- **CQI4판** (`results/ood_zeroshot_cqi4.{csv,out}`): P065 +4.3%(4/8=tie),
  V50 +11.2, V60 +11.4, P055 +21.0, CSI04 +10.6, STORM(구정의) +19.4.
  P065에서 continuous-학습 HL@639 교차 투입 시 **+56.6% (7/8)** — rich→coarse
  관측 전이 유효 (transfer 절 소재).
- **continuous판** (`results/ood_zeroshot.{csv,out}`, 정책 HL@639):
  P065 유일-양수(+285 vs −158), V50 +16.0, V60 +15.1, P055 +33.0,
  CSI04 +14.8, STORM +41.2 — 44W/1T/3L.
  그림: `Run4/QueuePostRZF_S40HighLoad/OOD_zeroshot_probe.png`.
- P065 전수확인 (`results/ood_p065_fullbase.{csv,out}`): 포화에서 전 12
  baseline 음수, PPO(continuous)만 양수.

### 4.2 CSI 밀도 축 (p_csi 0.6→0.4→0.2→0.1; 두 판 각각 자기 세계)
- continuous판 마진: +14.7%(in-dist 20-seed) / +14.8 / +15.8 / **+22.7** —
  기근일수록 확대, 전점 전승. (`results/ood_csi02.*`, `ood_csi01.*`)
- CQI4판: +13.9%(in-dist) / +10.6 / +10.9 / +12.5.
- 복합 극한 CSI01+V60(60km/h 고정+p_csi 0.1, age/Tc≈2, CQI4판):
  paired +247, 마진 +25.7%(분모 효과 큼 — 절대값으로 서술), depth 유지,
  SU 전멸. (`results/ood_csi01v60.*`)

### 4.3 부하-반응 곡선 (`results/load_response.{csv,out}`, Ent02 정책)
p_max 0.25→0.65 스윕: 마진 +11.2%→+5.4% 평탄-하락 — "부하↑→우위↑" 가설
기각; 진짜 스토리는 adaptation gap(그 부하에서 학습한 정책이 +12.9% vs
동결 +8.3%). discussion 한 문장용.

### 4.4 legacy (비공식 — 교정 전 LA 세계, 인용 금지)
`results/ood_p_arrival.csv`: QueueMixedArrival(legacy LA) 정책의 고정-p 스윕
(p 0.10~0.50, +135.7% 등). 세계 자체가 폐기된 물리라 공식 결과와 혼합 불가.

## 5. 논문 배치 가이드 (2026-08-06 확정 큐레이션)

- **본문**: §2 그리드 표(6행) + §3 K-scaling 미니표 + CSI 밀도 곡선
  (0.6/0.4/0.2/0.1, §4.2) + 정보-입도 3점(genie +4.3 / continuous +14.7 /
  4-bit +13.9, held-out 20-seed — 별도 final20 분석).
- **부록**: §4.1 구판 그리드(두 판), CSI01+V60, per-seed 승패 구조
  (승 +20~28% vs 패 한 자릿수, 포화 시드 집중).
- **transfer 절**: HL→nr4bit P065 +56.6% vs genie 역방향 −5.8% 대비.
- **제외**: §4.4 legacy, load_response(문장으로만).
- 공통 각주: ①pilot(40000대)/test(30000대) 분리, 최종 숫자는 20000+로
  재확정 예정 ②baseline만 세계 내 재튜닝(보수적 하한) + wider 12-grid에서
  챔피언 불변 검증 ③STORM2·포화 세계는 %가 아닌 paired 절대값으로
  ④D26은 관측 정규화를 학습값(/12)으로 동결한 strict 결과.

## 6. 파일 인덱스

`scripts/` — 실행 스크립트 (모든 경로는 레포 루트 기준; 이 폴더로 이동 전의
출력 경로가 docstring에 남아 있을 수 있음):
ood_paper_grid_cqi4.py(§2) · scaling_zeroshot_cqi4.py(§3) ·
ood_zeroshot_probe.py / ood_zeroshot_cqi4_probe.py(§4.1) ·
ood_csi01_probe.py / ood_csi02_probe.py / ood_csi01v60_probe.py(§4.2) ·
ood_p065_fullbase.py(§4.1) · scaling_zeroshot_probe.py(구 B3, Ent02) ·
load_response_probe.py(§4.3) · ood_probe.py(§4.4 legacy) ·
ood_zeroshot_figure.py(그림 생성기)

`results/` — CSV(per-seed 원자료) + .out(실행 로그·요약). 논문 그리드 병합본
= `ood_paper_grid_cqi4_merged.csv` (720행 = 6시나리오 × 3스케줄러 × 40시드).
