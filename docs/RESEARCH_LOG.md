# RESEARCH LOG — post-RZF 세대 이후 연구 서사

> 시간순 실험 노트 (append-only). 결정·근거·수치·교훈을 날짜별로 기록한다.
> 정정은 과거 항목을 고치지 않고 새 항목으로 쓴다. 공식 스펙은
> [SYSTEM_MODEL.md](SYSTEM_MODEL.md), run 카탈로그는 [RUNS.md](RUNS.md),
> run별 출생기록은 각 폴더의 RUN_MANIFEST.json 참조.

---

## 2026-07-13 — post-RZF 재설계 출시 (SU/MU 문제의 정식 해결)

**배경**: legacy LA는 B_tx를 SU-CQI로 산정 — MU 편성 시 구조적 과대약속
(1.34–1.91×)으로 depth≥2 첫 전송이 사실상 전부 NACK (GPT 감사 C-cluster).

**한 것**: la_mode/decode_order 축 신설. RBG 그룹 확정 후 그 그룹 그대로
h_hat 기반 RZF를 구성해 post-RZF 예측 SINR로 B_tx 산정 (la_planner.py,
RBG-major 마감). 게이트 3종 zero-tolerance 통과: G1 genie first-ACK 100%
(21,593 유닛), G2 planned-vs-actual max|Δ|=0.0 (86,986 유닛), G3
decode/replay max|Δlogp|=0.0 (11,820 결정).

**β_m 도입**: 잔여 CSI 오차(노후화 양방향 + MU 누설 낙관)에 대한 depth별
10th-percentile 보정 = (0.9815, 0.7306, 0.6466, 0.5922) @U(5,30) queue 세계.
스케줄러-독립 보정 + holdout 검증. QueuePostRZF pilot fresh 기동 (핀 9ed28d0,
entropy 0.01, seed 2024).

## 2026-07-15/16 — entropy 용량-반응 3-arm 실험

**질문**: pilot의 ~4800 정체가 탐험 부족인가?

**설계**: pilot ckpt에서 fork, 유일 델타 = entropy. 0.01(원본, GPU5) /
0.02(fork@189, GPU4) / 0.03(fork@289, GPU3). fork 기점이 달라 비교는
fork-후-경과 기준.

**부수 결정**: save_every=1을 차기 학습부터 (7/15 GPU 이동 사고 교훈 —
체크포인트 대기 규칙과 함께). MixedSpeed_L2c 기동 (L2b 계보 교정판,
속도 U(5,40)): β_m 재보정판으로 시작했다가 당일 **no-β(β=1)로 전환** —
"세계마다 재보정하는 절차 자체를 없앤 정직한 물리" (β 위계: SU/MU 미수정
= 거짓 수식 / no-β = 유효한 무-OLLA 세계 / β_m = OLLA 수렴점의 정적 근사).
pre-GPT 백업 영구 고정 (tag pre-gpt-baseline = 23bc8aa).

## 2026-07-20 (새벽) — dose-response 종결 + L2c depth-4 진단

**dose-response 최종**: 0.02 → 5743@769 ≫ 0.01 → 5380@889 > 0.03 → 5276@809.
말기 정책 entropy가 결정적 증거: 0.02 arm 0.579(최저, 확신) vs 0.03 arm
1.080(용해) vs 0.01 arm 0.832 — "뜨겁게 찾고, 찾으면 스스로 식는다"의 담금질
서사. 탐험 보너스는 '찾을 때까지'용이지 '계속 헤매기'용이 아니다.

**L2c(no-β) 진단**: fresh 0.02였음에도 depth 3.99 포화에 갇힘 (803 upd 동안
<3.5 방문 0회, critic EV≈0). 원인: 정직한 세계는 depth 비용이 후불·확률적
(첫-NACK→retx 지연→나중의 miss)이라 학습 신호가 소멸. 반면 β_m 세계는
depth 비용을 B_tx에 즉시 청구 (m4 ×0.58) → 기울기 존재 → Ent02-fork가
3.87→2.53 이동 성공. **β_m은 물리 명목(OLLA 근사)과 동시에 사실상의 reward
shaping 역할** — 이 대조 자체가 논문 소재.

**보조 실측**: SUS depth-cap 프로브 — L2c 세계 최적 depth는 내부(~3):
cap1 8052 / cap2 8372 / **cap3 8698** / cap4 8587. L2b 도너 zero-shot이
L2c 세계 1등 (8727 > in-world 학습 8632 > SUS+CQI 8587) — 세계축을 넘는
전이 실증 (단, official 파이프라인에는 타-세계 도너 금지 결정).

**CSI age 분석** (L2c 세계): age0 양자화 바닥 cos²=0.866(−0.63dB, 속도 무관);
30–40km/h age5 = 0.599(−2.23dB). p_csi=0.6이면 age≥5는 슬롯-UE의 ~1% —
오차의 주력은 양자화, 노후화는 고속 꼬리 사건.

## 2026-07-20 (오전) — Ent02 held-out 확정 + 세대 교체 + S40 기동

**Ent02 최종 판정** (20 held-out seeds, 세계 내 threshold 재스윕 후 SUS@0.7):
**PPO 5422 vs SUS+CQI 4879 = +11.1%**, 쌍별 18/20승 (t=7.84), 160판 156승.
run-eval +12.4%가 거의 유지 (L2c의 +1.8%→+0.5% 축소와 대조). PPO 전략:
depth 2.91로 얕게 → stream당 SINR 확보 → 같은 throughput으로 miss 절감
(0.319 vs 0.348; 패킷 경제: 2.9%p × ~6,600패킷 × 스윙3점 ≈ 관측 격차).

**threshold 재스윕 교훈**: 세계마다 최강 SUS threshold가 다름 (L2b 0.8 /
genie 0.7 / queue-β_m 0.7) — baseline은 반드시 세계 내 재스윕 후 비교.

**방침 확정**: β_m 유지 — "실물 LA(OLLA) 대신 그 수렴점의 정적 등가물을
쓴다"는 명목 (네 요건: 대응관계·스케줄러 중립·한계 명시·OLLA는 future work).
no-β 세계는 ablation으로 재배치. official 결과에 타-세계 도너 warm-start 금지.

**세대 교체**: queue 3-arm + L2c 은퇴 (SIGTERM 우아종료, redo 0). **S40 세대
기동** (새 핀 71c0bd4): ① fresh entropy 0.02 (fork 없는 깨끗한 파이프라인
시험) ② 속도 U(5,40) ③ β_m S40 재보정 = (0.9757, 0.7180, 0.6355, 0.5837)
(동결 레시피, holdout first-ACK 0.89–0.95) ④ save_every=1 + eval CSV 확장
(goodput/miss/total_failure/mu_depth 추가, retx 컬럼 제거 — 교정 세계에서
구조적 0). GPU4 = 메인 (p U(0.15,0.40)), GPU5 = 고부하 arm (p U(0.15,0.50);
load-headroom 프로브 근거 — 부하↑ → 스케줄러 격차 2697→7311).

## 2026-07-20 (오후) — 그림 규칙 + goodput 축 + 실시간성 산수

**그림 규칙**: 결과 PNG는 해당 run 폴더에 (analysis에는 스크립트·CSV만);
throughput 라벨 소수점 1자리; 패널 = reward/thr/goodput/SINR // depth/comp/
failure-modes(합=막대높이)/JFI (TOTAL failure 패널 제거, goodput 신설).

**goodput의 발견**: throughput은 동률인데 (74.6 vs 74.5) goodput이 가름 —
PPO 낭비 2.1Mbps(2.7%) vs SUS+CQI 4.4(5.9%). "같은 양을 나르되 완성될
패킷에 몰아준다"의 직접 시각화.

**eval 소요 실측** (tb 타임스탬프): 순수 eval 1회 ≈ 13.6분 = PPO 자기평가
11.5분 (3에피소드×1000슬롯, 슬롯당 ~230ms) + baseline 12종 2.1분. 슬롯당
230ms의 내역 = 물리 시뮬 ~4ms(배치 시 소멸) + Python 오버헤드(소멸) +
순수 정책연산 (**actor 69k 파라미터, ~5–10 MFLOPs/슬롯 → 전용 HW ~0.1ms**,
gNB가 이미 하는 RZF 역행렬보다 가벼움) — 실시간 배치에 구조적 장애물 없음.

## 2026-07-21 — 공정성 심층 분석

**JFI 압축의 원인**: 세계의 per-UE 평균 SNR 스프레드 실측 **51–57dB**
(pathloss+shadowing ON; σ² 보정은 에피소드 스칼라라 UE 간 격차 보존 —
표준 SLS와 같은 기하 격차. 참고: 표준 SLS는 SNR 정규화를 하지 않으며
우리 보정은 에피소드 간 난이도 고정용 벤치마크 장치, 비인과적임을 문서화).
바닥 UE는 물리적으로 구제 불능 (deadline 3–12슬롯 vs 슬롯당 ~10bits) →
굶는 집합을 정책이 아니라 물리가 고름 → 배달-JFI ~0.7–0.8에 전 스케줄러
수렴 (JFI 0.79 ≈ "24명 중 ~5명 0" 산수와 일치). PF의 공정성은 장기평균
기제라 hard deadline이 무력화.

**자원-JFI vs 배달-JFI 프로브** (Ent02 세계): 자원(스케줄 슬롯) 기준은
선명히 갈림 — greedy 계열 0.64–0.66 (PPO 0.66 포함) vs PF/DPF/Rnd
0.88–0.99. 그러나 배달 기준은 0.69–0.73 무반응. **자원 공정성이 결과로
전달되지 않는 세계** + PPO는 "편중하되 맞는 곳에 편중" (자원 최편중이면서
배달 공정성 동급 + reward 1등). 부수 발견: SUS의 직교성 관문은 채널 기하
기반의 접근 편향을 만들어 (만성 페어링 탈락 UE) 회전 기제로도 못 고침 —
SU+PF 0.99 vs SUS+PF 0.83의 원인.

**코드 정리 결정**: beta_rate는 Phase-1 시절의 원시 백오프 손잡이 (약속측
전용, 항상 1.0) — β_m이 정식 대체했으므로 paper-gen 배치에서 제거 (no-op).

## 2026-07-22 — 무인 복구 실전 검증 + S40 중간 판독

**복구 검증**: container 재시작 발생 → 접속 트리거로 autorecover 자동 발동
(00:51:38): tmux 재설치 → 두 run resume → watchdog 재기동, **redo 0**
(save_every=1). 무인운영 체계 첫 실전 통과.

**S40 중간 판독** (upd ~226/211): 고부하 arm이 먼저 돌파 진행 —
5115(+178)@209, depth 3.68→3.46 동반 하강 (fork-Ent02의 돌파 서명 재현).
메인 arm은 4932(−109)@189 접근 후 depth 2.6–2.8 탐색 딥. **local optimum
아님 판정**: 롤아웃 depth 3.77→3.41 단조 이동 + entropy 2.48→1.66 정상
냉각 + critic EV 0→0.21 상승 (L2c 함정의 3중 부재). 재판정 조건: depth
재고정 ∧ eval 150+ upd 평탄 ∧ entropy<0.7 동시 충족 시.

**대기 중 결정**: 시드 복제 2025/2026 (GPU0/2 제안), fresh-vs-fork 대조군
U(5,30) (GPU3, 선택), 용량-하한 프로브, SNR 분위별 완성률 표.

## 2026-07-22 (2) — reward-goodput 딜레마: 실측으로 종결

**고민** (사용자 제기): dense 항(ΔI_useful)은 최종 실패할 패킷의 bits에도
지급된다 → goodput과 misalign. 그러나 goodput을 직접 reward로 쓰면 신호가
희소해짐 (조밀함 ↔ 정렬의 딜레마). GPT가 packet-goodput 기반 PBRS
(잠정 진척 보상 + 실패 시 회수)를 제안 — 이론 검증 결과 건전했으나
(telescoping·파밍불가·부기 정확), 목적함수의 조용한 변경 3건(miss 벌점
삭제, per-packet→per-bit, 긴급도 가중 소멸)과 truncation-bootstrap 충돌이
있어 채택 보류.

**누수율 프로브** (reward_leak_probe.py — process_slot monkey-patch로
지급-귀속 계측, root py 무접촉): 실패-운명 패킷에 지급된 dense 보상 비중 =
**PPO 3.16% vs SUS+CQI 6.28%** (bits·urgency-가중 동일; goodput/throughput
낭비율과 교차검증 정합). deadline 3–12슬롯이 누수 창을 구조적으로 상계.

**결정**: 현행 reward 유지. 근거 — misalignment 실측 3.2%뿐이고, 학습된
정책의 누수가 최강 휴리스틱의 절반 (희소 정산 항이 이미 회피를 가르침).
재구성의 이득(≤3.2%) 대비 비용(전면 재학습+비교 재앵커) 불균형. 논문에는
"이층 reward(결정-품질 dense + 결과-정산 sparse), misalignment 실측 3.2%"
로 서술. reward 개편(PBRS류)은 Run5/OLLA 세대 후보로만 보존.

## 2026-07-23 — B3: 크기-축 zero-shot 일반화 (K-스케일링)

**질문**: per-UE 인코더+pointer 구조는 K-독립 파라미터 — K=32로 배운
Ent02 정책이 재학습 없이 다른 UE-집합 크기에서 작동하는가? (논문
2411.08529의 대역폭/layer 일반화에 대한 우리식 대응)

**결과** (Ent02 best@769, K별 활성 U{K/2..K}, 8 seeds, in-world baseline
대조): strict 가중치 로드 전 K 통과 (파라미터 K-독립 실증). 마진 =
K16 +4.9% (최강이 SU+CQI로 교체되는 저인원 구간) / K24 +20.5% /
K32 +10.7% (기준 재현 ✓) / **K48 +0.7% (1.5× 외삽에서도 무패배)**.
**depth의 단조 적응이 백미**: 2.22→2.66→2.91→3.19 — 인구가 늘수록
스스로 깊게. Run2 AdaptCheck(SU↔MU 적응)의 K-축 재현, 재학습 없이.

**해석·주의**: 마진 역-U (중간 인구에서 최대) — 저인원은 SU 강세로,
초과밀(K48, miss ~0.38)은 구조적 바닥으로 압축. K48의 +0.7%는 8-seed
잡음 내 동률로 읽는 게 정직 ("이긴다"보다 "지지 않는다"). β_m은 K=32
보정값 고정(zero-shot 전제) — K48 재보정 시 더 나을 여지. R/L 축은
아키텍처 확인 안 됨 (차후).

## 2026-07-24 — 부하-응답 곡선: "혼잡할수록 우위↑" 가설 기각

**질문**: Ent02 동결 정책의 baseline 대비 마진이 부하(p_arrival)와 함께
커지는가? (HighLoad 논문 주제 타당성 판단용)

**결과** (load_response_probe.py, 동결 Ent02 best, p ∈ {0.25, 0.45, 0.65},
5 seeds): 마진 +11.2% → +7.4% → +5.4% — **평탄~하강**. 예측("고부하
헤드룸") 공개 기각. 진짜 이야기는 **적응 갭**: 같은 p0.5에서 동결 정책
+8.3% vs 그 부하로 학습한 정책 +12.9% — "부하가 우위를 주는 게 아니라,
그 부하에서 학습하는 것이 우위를 준다". HighLoad는 '더 쉬운 세계 자랑'이
아니라 부하-축 적응력의 증거로 서술.

## 2026-07-26 — S40Ent02 은퇴·held-out 확정 + anneal fork (한 번의 오발)

**S40Ent02 held-out** (seeds 10000–10019, in-world SUS 재스윕 thr 0.7):
**PPO 5185 vs SUS+CQI 4696 = +10.4%, 19/20승, t=7.83**. fresh-0.02 청정
파이프라인 (fork 없음, genie 접촉 없음) 재확인. 쇼케이스 10017, 8-패널
그림 run 폴더 저장. SUS 재스윕에서 4세계 공통 thr 0.7 최강 (32-antenna
E[|corr|²]≈1/32 → 0.7이 ~10% 쌍만 차단, SUS는 사실상 항상 depth 4 충전).

**anneal fork 오발→정정**: "Ent01로 이어서"를 S40Ent02@519 fork로
구성했으나 사용자 의도는 **HighLoad best@639에서 fork**. 잘못 돌린 run은
사용자 지시로 완전 삭제 (cfa3091), QueuePostRZF_S40HighLoad_Ent01로 재발사
(6534f2a, GPU4, entropy 0.01, 그 외 동일). 판정 데드라인 upd 939 (fork
+300): 신기록(>5598) 없으면 "0.02 유지 우세"로 종결. 공유 창 조기 판독:
0.01이 평균 +57 우위, 5597 터치 — 아직 신기록은 없음.

## 2026-07-27 — HighLoad 은퇴: held-out **+14.7%, 20/20 전승** + genie 페어 발사

**HighLoad 은퇴** (d0e1a15): best 5597.5@639, 이후 500+ upd 무갱신.

**held-out 20-seed** (10000–10019, in-world 스윕 thr 0.7 = 4565 확인):
**PPO 5352 vs SUS+CQI 4668 = +14.7%, 20/20승** — 전 run 통틀어 최초의
전승이자 최대 마진. run-eval +13.4%가 held-out에서 오히려 +14.7%로 상회.
depth 2.90 vs 3.97, miss 0.345 vs 0.375, goodput 82.4 vs 79.2 Mbps.
부하-응답 결론과 정합: 학습-부하 지점별 마진 p0.4 +10.4% → p0.5 +14.7%
(적응 갭의 held-out 실증 2점). 쇼케이스 10017 (reward +21.9%, goodput
81.3 vs 76.0), 그림 run 폴더 저장.

**genie-CSI 페어 발사** (00:57, pin 71c0bd4): HighLoad 세계 그대로 +
perfect CSI (pmi_mode=genie, p_csi=1.0, β=1). GenieS40HL_FineTune (GPU5,
--init_from HighLoad best@639) vs GenieS40HL_Fresh (GPU3, 처음부터).
질문 = 교정-LA 세대에서 GenieFineTune 재연 — imperfect→perfect 전이 가치.
발사 중 fresh-dirty guard가 미커밋 분석 .py에 걸려 26초 crash-loop ×8
(빈 .stale 껍데기 정리) → 스크립트 커밋(1d67fde)으로 해소. watchdog/
autorecover 명단 동호흡 갱신 (Ent01 + genie 페어 = 3 runs).

## 2026-07-27 (2) — OOD zero-shot 프로브: 동결 HighLoad 정책, 6-config 전 구간 생존

**질문**: 학습 분포 밖 세계에서 재학습 없이 얼마나 버티는가 (사용자 승인
그리드, 우선순위 P065→V50→V60→P055→CSI04→STORM, seeds 30000–30007
8개, config별 SUS thr 재스윕, paired 비교. 이후 보고는 사용자 지정대로
goodput%·miss%p 기준).

**결과** (Δgood% = vs 그 세계 최강 baseline, 승수 = goodput paired):
P055 +2.7% (7/8·1무) / P065 +1.6% (6/8) / V50 +3.9% (8/8) / V60 +3.6%
(8/8) / CSI04 +3.7% (8/8) / STORM +2.5% (7/8). miss는 전 config에서
−1.1~−3.1%p 동반 우위. 종합 48판 44승 1무 3패 — 정책은 어느 축에서도
붕괴하지 않음. 그림 OOD_zeroshot_probe.png (run 폴더).

**축별 발견**: (a) 부하: 분포안 +4.0% → 0.55 +2.7% → 0.65 +1.6% 매끄러운
감쇠, 절벽 없음. 패배는 P065 만석 seed(30002/30006) 뿐 — 용량초과
포화에서 최적이 greedy로 퇴화, PPO도 depth 3.93으로 SUS 모사에 수렴하나
모사 정밀도에서 −2%. (b) 속도: 50/60 km/h(결맞음 5.2/4.4슬롯)에서 마진
유지 — Doppler 외삽 열화 평평. (c) CSI 기근(0.4): β_m 미재보정
핸디캡에도 +3.7% — V50/V60/CSI04가 모두 +3.6~3.9%로 수렴 = 우위의
공통 원천은 "낡은 CSI에 대한 강건성". (d) STORM(3축 동시): +2.5%,
단일축 합보다 나쁘지 않음 — 상호작용 붕괴 없음. STORM에서만 SUS 최적
thr 0.75로 이동 (유일한 예외).

**적응-갭 직접 쌍** (P065, 같은 seeds): PPO-HL +285 vs PPO-Ent02 −1108
vs SUS −158 — 고부하 학습 정책만 물 위, 저부하 학습 정책은 baseline
아래로 침몰. 갭은 지옥 seed에 집중 (30002 −3380, 30006 −3440). 각주:
Ent02는 속도축(U(5,30) 학습)도 동시 OOD — 순수 부하 귀속은 S40Ent02로
재평가해야 깨끗함. **보강**: P065에서 SUS/SU 동반 음수 → 12종 전체
baseline 재확인 규칙 발동 (ood_p065_fullbase.py, MW/PF류 검증).
