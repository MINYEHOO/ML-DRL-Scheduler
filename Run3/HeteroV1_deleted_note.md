# HeteroV1 (삭제됨, 2026-07-06)

Run3 Stage-A: K=16, UE별 속도 4×{5,10,15,30} mix, p_csi 0.6. Update 559 조기수렴
종료 (best 7357@409). 폴더·checkpoint·로그는 2026-07-06 사용자 승인으로 삭제 —
후속 run들(Uniform10_Ent002 계열)에 대체되어 논문 수치로 미사용.

## 박제된 교훈 (원 분석은 2026-06-27~29)

1. **p_csi=0.6 K16에서 MU pairing은 순손실** — 모든 SU baseline > 모든 MU baseline.
   병목은 staleness가 아니라 Type-II 코드북 양자화 + 전력분할 (p_csi 스윕/depth-cap
   스윕으로 확정; 한계-피드백 MU-MIMO 간섭 포화, Jindal 2006 계열).
2. **retx=0은 SU/MU의 경계선** — depth 낮으면(≤1.3) 구조적으로 retx 없음 (PPO 특유
   아님). "조작 의혹" 4-agent 감사로 종결.
3. 속도별 starvation 없음 — 불평등은 채널/위치(cell-edge) 기인.
4. PPO는 이 regime에서 SU-CQI(진짜 bar)와 동률 수준 — "+24% vs baseline"은 MU-only
   구세트 대비 과대 (이 교훈이 이후 '진짜 최강 baseline' 규율의 출발점).
5. 운영 교훈: 검증/스모크 에이전트를 라이브 run_dir에 절대 접근시키지 말 것 (6/27
   오염 사고 — 이 폴더의 polluted 백업도 함께 삭제됨).
