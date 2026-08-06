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

## 2026-07-29 — anneal(Ent01) 은퇴: 무이득 판정 + 원인 분석

**판정** (데드라인 939 + 300 초과 관찰): fork후 60 evals 신기록 0 —
best는 fork 시점의 5597.5@639 그대로. 공유 창 50회 paired에서 Ent01
평균 5089 vs 부모(0.02) 5283 = **−86, 12/50승** (조기 판독 +57은 표본
확대로 역전). **entropy 0.02 유지 우세 확정** — dose-response 결론 재확인.

**원인 3층**: (1) 기록 5597.5 자체가 3-episode eval 궤도(mean 5100–5300,
std 213–348)의 +1.5σ 꼬리 추첨 — 어느 팔도 평균으로 재도달 불가한 수치를
"갱신 목표"로 쫓은 셈. (2) 양팔 모두 kl-stop 348/350회(update의 ~70%)로
신뢰영역 포화 = 수렴 완료 신호, 탐색 부족이 아님. (3) **정보 천장**:
genie FineTune이 같은 세계+perfect CSI로 즉시 7995 달성 → imperfect-CSI
세계의 ~5600은 최적화가 아니라 CSI 물리로 캡. 각주: 0.01 팔의 엔트로피가
오히려 더 천천히 냉각(16.2 vs 5.3) — 예상과 반대 방향, 어느 쪽도 무관.

**부수 사건**: 정지 중 pkill 자기-매칭 재발 (변수 안 리터럴도 cmdline
노출 — [e]클래스 마스킹이 정답), 구 watchdog이 정지 직후 1회 재발사
(즉시 명단 갱신+재기동으로 정리, runs=2: genie 페어만).

**genie 페어 중간** (upd ~213/165): FineTune best 7994.9@159 =
SUS+CQI(7473) **+7.0%** 고원 — warm 전이 가치 확정 단계. Fresh는 7006
(자기 best가 아직 구조 바닥 7237@9) — greedy 동률 아래서 L2b 패턴 재현.
격차 +984. 참고: update-0 재현 평가로 fresh의 진짜 출발선 실측 —
결정론 6990 / 확률 4471 / SUS+Random 5173 (argmax 일관성 하나로 +2500;
구조 바닥이 SUS+CQI의 94%).

## 2026-08-01 — CQI4 세대 발사 + genie 역방향 전이

**CQI 4-bit 양자화** (사용자 지시): UE 보고 SE를 3GPP TS 38.214 Table
5.2.2.1-3 (4-bit 256QAM) 사다리에 floor 스냅 — "BLER≤10% 최고 인덱스"
규칙의 Shannon-추상화 대응. index 0 = out of range(표준 의미)이고,
q=0→ĥ=0(신규 배제 + 고정 HARQ retx는 zero-beam 무해 outage)는 본
시뮬레이터의 추상화 (cqi0_retx_regression으로 고정, NaN 없음 검증).
MCS/OLLA 부재와 정합: 정보(CQI)만 양자화, 행동은 연속 약속 + β_m.
사다리 상한 7.4063 → RBG-slot당 최대 ~9954 bits 약속.

**개발 위생**: 본선 pin(71c0bd4, genie 페어 가동 중) 무접촉 원칙으로
git worktree `_cqi4dev` + 브랜치 cqi4에서 구현(49550e2)·검증·테스트
(d3efef9). 게이트: 유닛(floor/표/경계/genie가드) + 3-seed 동결 평가
(PPO 5594→5029, SUS 4937→4354 — 동등 부담) + smoke 12 upd (NaN 0,
upd9 eval 13종 정상, 564s/upd).

**β_m 재보정** (frozen 10-pct recipe, nr4bit 세계): (1.0018, 0.7499,
0.6592, 0.6058) — 전 depth +2.7~4.4% 상승 (floor가 이미 보수적이라
이중 보수화 제거). β₁>1이므로 명칭은 back-off가 아니라 **depth-wise
calibration factor**. holdout first-ACK 0.869/0.886/0.890/0.884.

**발사**: QueuePostRZF_S40HL_CQI4 (GPU4, fresh 0.02, seed 2024, pin
d3efef9, worktree 실행). smoke 12 update를 폐기하지 않고 승격(동일
CLI·seed) — wrapper가 upd 12부터 resume. 첫 resume만 ALLOW_HASH_
MISMATCH=1 일회 사용 (smoke 코드 49550e2→pin diff는 주석+테스트뿐,
검증됨; 첫 저장 후 엄격 pin 자동 복원). genie 종결 후 main merge 예정.

**genie 역방향 전이** (사용자 아이디어 "hindsight 학습→실전 투입" 검증):
imperfect HighLoad × held-out 20 seeds에서 **HL 5352 > FT 5044 >
SUS 4668 > Fresh 4535**. FT는 실전에서도 SUS 20/20 전승(무너지지 않음)
이나 HL에 20전 전패(−5.8%) — genie 왕복은 순손실. 메커니즘 = depth
침식 (HL 2.90 → FT 3.56 → Fresh 3.99≈SUS): perfect CSI가 낡은-CSI
에누리 습관을 지움. Fresh(genie만)는 SUS 동률로 추락 → FT의 생존은
imperfect 유산의 공. 실용 레시피: hindsight pre-train → 실전 fine-tune
(정방향 warm-start +6.8%과 대칭); privileged distillation은 Run5 후보.

## 2026-08-03 — GenieS40HL_FineTune 은퇴 + held-out 최종판정 (+4.3%, 20/20)

**은퇴** (사용자 지시 "fine tuning은 학습 의미 거의 없음"): best 8014@329
이후 800+ upd 무갱신, depth 2.96→4.00 단조 침식으로 Fresh 수준 수렴 —
warm 이득은 초기 가속뿐, genie 세계 계속-학습이 imperfect 유산(얕은 depth)
을 지움. 로스터 동시 제거(runs=2: Fresh+CQI4), wrapper STOP 동결 후
SIGTERM 우아종료(latest@1144 저장, redo 0), watchdog 재기동, GPU5 반환.

**In-world SUS 재스윕** (8 seeds): **thr 0.6 최강** 7600 (0.5: 7582 /
0.7: 7591 / 0.75(config): 7541 / 0.8: 7467 / 0.9: 7097). genie-S40HL
세계의 최적은 0.6 — L2b-genie 0.7, imperfect-S40 0.7과 또 다름 (세계별
재스윕 규약의 3번째 실증; 0.5–0.7은 사실상 평탄).

**Held-out 20-seed (10000–10019, SUS@0.6)**: **PPO-FT(best@329) 7993 vs
SUS+CQI 7661 = +4.3%, 20/20 전승** (per-seed 마진 +6~+611); depth 3.48
vs 3.97; miss 0.274 vs 0.288; goodput 92.9 vs 91.1 Mbps. run-eval 8014
→ held-out 7993, 수축 없음. showcase seed 10011 (마진 +611), 8-panel
그림 run 폴더. 스크립트 genies40hl_finetune_final_figure.py, CSV
genies40hl_finetune_final20.csv.

**의미 3건**: (1) **정보-구조 수축** — 같은 세계에서 imperfect-CSI 마진
+14.7% → perfect-CSI +4.3%: PPO 우위의 ~2/3가 불완전-CSI 대응(양자화/
노후 에누리)에서 나온다는 직접 증거 (각 세계 내 상대 마진의 비교;
cross-world 절대보상 비교 아님). (2) **interior-depth 최적 재확인** —
β=1(깊이 무가격)·perfect CSI에서도 depth 3.48 정책이 depth-4 전원
(SUS 7661, Fresh 7773)을 이김: depth-4 포화는 genie 세계에서도 최적이
아님 (등전력 분할 + deadline triage가 여전히 interior를 만듦; L2c
cap3>cap4와 정합). (3) **warm-vs-fresh** — FT 7993 vs Fresh(interim@869,
학습 계속 중) 7773 = **+2.8% paired(+220), 19/20** (Fresh 승 1회: seed
10001, +35); Fresh도 SUS 대비 +1.5%로 튜닝 baseline은 넘음. L2b-시대
결론(warm>fresh) 재현 — 단 이번엔 fresh도 양수 마진.

## 2026-08-04 — 전 런 은퇴 + CQI4 held-out 최종판정 (+13.9%, 20/20): 정보-축 3점 완성

**전 런 은퇴** (사용자 지시 "두 실험 다 종료"): ①GenieS40HL_Fresh —
best 7761.6@869 이후 430+ upd 무갱신, depth-4 분지 고착(최근 300 upd
p05 3.83, depth<3.3 미방문). **FineTune final20의 Fresh(interim) 행이
그대로 최종값** (best 불변, 동일 시드·동일 스윕 thr 0.6): 7773 vs
SUS+CQI 7661 = +1.5%, FT에 19/20 패배. ②QueuePostRZF_S40HL_CQI4 —
best 5038.9@409, upd ~650 이후 과냉각 하강(창평균 4943→4595→4141,
depth 2.66→1.95, entropy 0.597 — S40Ent02 병리 재현). 로스터 양쪽
비움, SIGTERM 우아종료(redo 0), **라이브 런 0 / watchdog 정지 /
GPU 0–5 전부 반환**.

**CQI4 in-world SUS 재스윕** (8 seeds, 이 세계 첫 스윕): **thr 0.7 최강**
4072 (0.5: 3999 / 0.6: 4058 / 0.75(config 상속값): 4031 / 0.8: 3975 /
0.9: 3780). 상속값 0.75를 그대로 썼다면 baseline이 ~42 손해 — 재스윕
규약 4번째 실증.

**Held-out 20-seed (10000–10019, SUS@0.7)**: **PPO(best@409) 4755 vs
SUS+CQI 4174 = +13.9%, 20/20 전승** (per-seed +89~+1143); depth 2.74
vs 3.92; miss 0.362 vs 0.388; goodput 80.0 vs 77.4. showcase 10017
(마진 +1143), 8-panel run 폴더. 스크립트 queue_s40hl_cqi4_final_figure.py
(⚠️ _cqi4dev worktree 코드 import — main Config는 cqi_mode 모름), CSV
queue_s40hl_cqi4_final20.csv.

**⭐정보-축 3점 세트 완성** (동일 세계 계열·동일 프로토콜, 세계 내
스윕된 SUS+CQI 대비):
- perfect CSI (genie): **+4.3%** (20/20)
- continuous CQI (56-bit PMI + 실수 CQI): **+14.7%** (20/20)
- 4-bit CQI (56-bit PMI + 16-단계 CQI): **+13.9%** (20/20)

판독: (1) CQI4 런의 창설 질문("리포트가 거칠어지면 학습 마진이
살아남는가")에 YES — 14.7→13.9%는 사실상 보존 (V50/V60/CSI04 패턴의
예측 적중). (2) 양자화 비용은 전원 부담: PPO 5352→4755(−11.2%),
SUS+CQI 4668→4174(−10.6%) — 절대 성능은 다 같이 내려가고 상대 마진
유지. (3) 정보가 완벽해질 때만 마진이 수축(+4.3%) — "PPO 우위는 CSI
불완전성이 만드는 결정 문제에서 나온다" 축의 세 번째 직접 증거.
PPO 전략도 일관: depth 2.74(CQI4)/2.90(연속)/3.48(genie FT).

**캐비앳**: CQI4 판정은 과냉각 국면에서 조기 은퇴한 팔의 best@409
기준 — entropy 스케줄이 더 좋았다면 상한이 더 높았을 수 있음(보수적
방향, 판정에는 무해). 학습은 866/1500 upd에서 중단. run-eval 5039 →
held-out 4755 수축은 3-seed run-eval 노이즈 범위.

**후속 결정 대기**: ①cqi4 브랜치 main merge (genie 종결 조건 충족 —
launch 시 "genie 종결 후 merge 예정"이라 명기했음) ②논문세대 배치
(root-py 일괄 수정 + 다중 시드 + 최종 시드 20000+ — GPU 6대 전부 가용).

## 2026-08-05 — CQI4 OOD zero-shot probe: 6-세계 전원 양수 생존 + 역방향 놀라움 1건

**설계**: 7/27 원본 probe의 CQI4 판 (`ood_zeroshot_cqi4_probe.py`, GPU5/tmux
— 사용자 부재 대비 disconnect-safe 실행). CQI4 best@409 동결, 전 세계
nr4bit+β_m(CQI4) 유지, 시드 30000–30007, 세계별 SUS 재스윕. P065에는
continuous-CQI HL@639를 짝으로 투입(리포트-입도 적응 갭).

**결과 (vs 세계 내 스윕 최강 SUS+CQI)**: P065 +4.3% (4/8 = tie 판정) /
V50 +11.2% (6/8) / V60 +11.4% (6/8) / P055 +21.0% (5/8) / CSI04 +10.6%
(6/8) / STORM +19.4% (5/8). **어느 축에서도 붕괴 없음** — 4-bit 학습
정책도 전 OOD 세계에서 마진 양수. 단 원본(continuous판 44/48)보다
승수 32/48로 약함: ①세계 자체가 다르므로 1:1 비교 불가 ②CQI4 ckpt는
과냉각 국면 조기수확(@409)이라 정책 품질 열위 혼재.

**⭐P065 역방향 발견**: nr4bit P065 세계(전원 음수 지대)에서
**continuous-학습 HL@639 zero-shot이 −409 (+56.6%, 7/8)로, 그 세계
토박이인 CQI4 정책(−901, +4.3%)을 크게 이김.** 판독: 관측이 거칠어지는
방향의 전이(rich→coarse)는 성립 — genie 역방향(전략 자체가 오염)과
달리, 여기선 전략은 동일 물리에서 학습됐고 입력 입도만 낮아진 것이라
풍부한 훈련 신호로 배운 정책이 그대로 우세. 실용 레시피 보강:
"정밀 리포트로 학습, 거친 리포트로 배치"가 유효 (train-rich,
deploy-coarse). depth도 로드-반응 유지(2.72–3.35).

캐비앗: 8-seed probe 규칙(한 자릿수 마진 = "지지 않음"), 최종 숫자는
시드 20000+ 예약분으로. CSV/out: ood_zeroshot_cqi4.{csv,out}.

## 2026-08-06 — CSI02 probe (p_csi 0.2): 현실적 리포팅 밀도에서도 마진 보존

**동기** (사용자 질문 "실환경 적정 p_a/p_csi"): 부하 축은 학습분포가
현실 운영 구간(ρ 0.4~1.9)을 이미 커버; p_csi 0.6은 실제 CSI 주기
(5–20 ms ≈ p 0.025–0.1) 대비 후함 → p_csi 0.2(평균 간격 2.5 ms,
나이 ~2 ms, 30 km/h 기준 age/Tc ~0.45)를 zero-shot 시험.

**설계**: `ood_csi02_probe.py` (GPU5/tmux) — 두 판 동시(continuous
HL@639 / nr4bit CQI4@409), 각자 자기 세계에서 p_csi만 0.6→0.2, 시드
30000–30007, 세계별 SUS 재스윕, β_m은 0.6-보정값 동결(의도된 미스매치).

**결과**: CSI02-cont **+15.8% (8/8)** depth 2.93 first-ACK 0.875 /
CSI02-cqi4 **+10.9% (6/8)** depth 2.76 first-ACK 0.905. CSI 축 완성:
- continuous: 0.6 +14.7%(20/20) / 0.4 +14.8%(8/8) / 0.2 **+15.8%(8/8)**
- nr4bit:     0.6 +13.9%(20/20) / 0.4 +10.6%(6/8) / 0.2 **+10.9%(6/8)**

**판독 4건**: (1) CSI 밀도를 학습치의 1/3로 줄여도 마진 평탄~소폭 확대
— "p_csi 0.6이 후하다"는 우려 해소, 절대 성능은 전원 하락(SUS 4668→
3105)하되 상대 우위 보존 (CQI 양자화와 동일 패턴: 정보 열화는 전원
부담, 학습 정책이 상대적으로 덜 다침). (2) **가장 현실적인 코너**
(4-bit 리포트 × p_csi 0.2)에서도 +10.9% 양수. (3) first-ACK: cqi4판
0.905가 cont판 0.875보다 높음 — floor-snap의 보수성이 노화 낙관을
상쇄하는 내장 마진으로 작동. (4) depth 2.8~2.9 유지 — legacy-LA
시대(Run2, p_csi 0.2 → near-SU 1.12)와 대조적으로, 교정된 LA(β_m)
세계에서는 CSI 기근에도 얕은 MU가 정답이고 정책이 이를 유지 (Run2의
"SU 발견"이 부분적으로 legacy-LA 인공물이었다는 방증; 단 K/속도가
달라 정성 대조로만).

캐비앗: 8-seed 규칙; β_m 재보정 없이 잰 보수적 하한.

## 2026-08-06 (2) — CSI01 probe (p_csi 0.1): 기근이 깊을수록 마진 확대 (+22.7%)

**CSI01-cont +22.7% (8/8)** depth 2.96 first-ACK 0.818 / **CSI01-cqi4
+12.5% (6/8)** depth 2.83 first-ACK 0.849. **CSI 밀도 축 4점 완성**:
- continuous: 0.6 +14.7% / 0.4 +14.8% / 0.2 +15.8% / 0.1 **+22.7%** (전점 전승)
- nr4bit:     0.6 +13.9% / 0.4 +10.6% / 0.2 +10.9% / 0.1 **+12.5%**

판독: (1) **정보가 희소할수록 학습 이득이 커지는 단조 경향** — 평균
나이(~4.5 ms)가 30 km/h 코히런스(~4.4 ms)에 도달하는 깊은-노화
지점에서 마진 최대. Run2-시대 관찰(+30.6%@p_csi 0.2)의 재현이지만
이번엔 교정된 β_m 세계 — LA 인공물이 아님이 확정. (2) 단 Run2와
달리 정책은 SU로 도망가지 않음(depth 2.96 vs Run2 1.12): 교정 세계
에선 기근에도 얕은 MU가 정답. (3) SU+CQI는 사실상 전멸(76/34) —
리포트가 ~10슬롯마다 오면 CQI-선택+SU용량으로는 못 버팀; MU 용량과
낡은-CSI 대응이 동시에 필수인 지대. (4) first-ACK 0.82~0.85로 하락
(β_m 0.6-보정 미스매치) — 재보정 없이도 마진 확대, 보수적 하한.
캐비앗: 8-seed 규칙. 실환경 결론 확정: "CSI 가정이 현실적일수록
학습의 근거가 강해진다."

## 2026-08-06 (3) — CSI01+V60 복합 극한 (age/Tc≈2, nr4bit): +25.7% (6/8)

**설계** (사용자 지시 "아예 힘들게"): p_csi 0.1 + 전 UE 60 km/h 고정을
nr4bit 세계에 동시 인가 — 평균 CSI 나이 4.5 ms vs 코히런스 2.2 ms
(age/Tc≈2, 방향 CSI가 평균적으로 만료된 세계) + 4-bit 리포트. CQI4
정책(best@409)만, 동일 프로토콜 (`ood_csi01v60_probe.py`).

**결과**: **PPO 1205 vs SUS+CQI@0.70 958 = +25.7%, 6/8** (CQI4 시리즈
최대 마진), depth 2.91 유지, first-ACK 0.781(β_m 미스매치 최대인
조건), miss 0.458. SU+CQI **−555 완전 침몰**. 패배는 여전히 30002
(−154, −6.5%)/30006(−387, −17.1%) 두 포화 시드뿐 — 6개 프로브 전부
동일 구조 확정.

**정직한 판독**: 절대 격차는 ~+250으로 CSI01 단독(+235)과 비슷 —
상대 마진이 커진 것은 baseline 절대값 붕괴(1886→958)의 분모 효과가
큼. 즉 "복합 스트레스가 이득을 증폭"이 아니라 **"세계가 가라앉아도
학습 정책의 절대 우위는 유지되고, 상대 우위는 커진다"**가 정확한
서술. 정보-열화 축 전체(밀도 0.6→0.1 × 입도 continuous→4-bit ×
속도 →60)에서 붕괴 지점 없음 — 붕괴는 오직 용량 초과(부하 축)에서만.

## 2026-08-06 (4) — ⭐논문 OOD 그리드 (사전등록 n=40) 완료: 5/6 CI-확정 승 + K-scaling 전 구간 승

**결과** (CQI4 best@409 동결, 3-GPU 병렬, seeds 30000–30039): P055
+571±157 (35/40) / V60max +556±106 (37/40) / CSI02 +444±99 (37/40) /
**D26 +716±116 (39/40, 최강 — 신설 마감 축)** / STORM2 +498±158 (35/40)
— 전부 CI>0 확정. P010(저부하)은 +9±16 tie(설계상 예상; 최강 baseline이
SU+CQI로 교체 = regime-map 일치). **층화: 확정-승 시나리오 전부에서
최고부하 3분위 평균도 양수(+148~+570)**. STORM2는 단일-축(+571/+556/
+444) 대비 +498로 **파괴적 상호작용 없음** (baseline 386 붕괴 — %말고
paired로 서술). **K-scaling(CQI4판)**: K16 +7.1(7/8)/K24 +13.4(8/8)/
K32 +11.4(7/8)/**K48 +20.8%(8/8)** — continuous판의 K48 tie와 정반대,
HighLoad 학습의 부하 적응력 방증(단 K축=인구×부하 교란 명시 필요).

**정리**: OOD/zero-shot 전체를 `Run4/_analysis/OOD/`로 통합(scripts 12
+ results 26, git mv 이력 보존, 그리드 병합 CSV 720행) + README.md
(프로토콜·정의·전 결과표·파일 인덱스·논문 배치 가이드). legacy(교정 전
LA) 산출물은 비공식 표기로 격리.

## 2026-08-06 (5) — OOD 감사 3종 수정 완료: 판정 전부 불변, 마진은 오히려 상승

외부 리뷰(GPT) 지적 3건을 모두 처리하고 그리드를 재조립했다.

**① pilot/test 분리**: threshold 선택을 미사용 시드 40000–40007로 이전
→ 30000–30039가 완전 독립 test set. 신 T*: P055 0.70→0.75, V60max
0.70→0.75, CSI02 0.75→0.70, D26 0.75→0.80 (P010/STORM2 불변). **꼭대기가
평탄해 baseline 값은 거의 불변**(V60max 3842→3843, D26 3691→3673).

**② 챔피언 검증**: pilot에서 12-baseline 전수 + SUS+MW 전용 스윕 →
**6/6 세계 챔피언 불변**(SUS+CQI; P010만 SU+CQI). MW는 자기 최적
threshold에서도 격차 +212~+284 열세. **D26에서 deadline-aware(EDF/DPF)는
2위조차 못 함** — "왜 EDF를 본문에서 뺐나"의 선제 답변.

**③ D26 strict**: env=U[2,6] / 정책 정규화=/12(학습값 동결) 분리 재실행.
4407→4403 (사실상 불변), paired +716→**+730±115, 39/40**. 정규화 적응이
이득의 원천이 아니었음이 확정.

**갱신 판정(n=40, 신 T*, strict D26)**: P055 +561±168(35/40) / P010
+9±16(tie) / V60max +556±108(36/40) / CSI02 +444±92(37/40) / **D26
+730±115(39/40)** / STORM2 +498±158(35/40) — **5/6 CI>0 확정 + 설계상
tie 1, 구판과 동일**. 보수 참고(구 pilot 8시드 제외 test32): 전 세계
마진 상승(P055 +639, D26 +782 **32/32**, STORM2 +584) → 구 pilot 시드가
baseline에 유리했음(= 우리 마진 과소평가)이 실증됨.

**부수 결정**: 논문 본문 baseline 표시 세트 = **6개 {SUS,SU}×{CQI,PF,Random}**
(사용자 확정). DPF/MW/EDF는 부록·각주로. 12-grid 검증은 "wider grid에서도
챔피언 불변" 근거로 유지. GPU는 타 사용자 선점으로 CPU 병행 실행.
