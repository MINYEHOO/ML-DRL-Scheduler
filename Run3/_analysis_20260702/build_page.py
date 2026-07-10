
import json, re
data=json.load(open("official9full_page.json"))
css=open("_css_cache.txt").read()
def tiles(ts):
    h=""
    for i,(k,v,s) in enumerate(ts):
        h+=f"<div class='tile{' hero' if i==2 else ''}'><div class=k>{k}</div><div class=v>{v}</div>"+(f"<div class=s>{s}</div>" if s else "")+"</div>"
    return f"<div class=tiles>{h}</div>"
MET_TH=("<th>Reward (±std)</th><th>Thr Mbps</th><th>SINR dB</th><th>Comp</th><th>Miss</th>"
        "<th>Retx-drop</th><th>Total fail</th><th>Depth</th><th>JFI (active)</th>")
THS=f"<tr><th class=l>Scheduler</th>{MET_TH}<th>PPO − x (abs, %)</th><th>승수</th></tr>"
BASE_TH=("<th>SUS+CQI</th><th>SUS+DPF</th><th>SUS+PF</th><th>SUS+Rnd</th>"
         "<th>SU+CQI</th><th>SU+DPF</th><th>SU+PF</th><th>SU+Rnd</th><th>PPO</th>")
THA=f"<tr><th class=l>Seed</th><th>인원</th><th>속도 km/h</th>{BASE_TH}<th>Δ PPO−Best8 (abs, %)</th></tr>"
THB=f"<tr><th class=l>Seed</th><th>인원</th><th class=l>Scheduler</th>{MET_TH}</tr>"
secs=""
for o in data:
    secs+=f"""
<section>
  <div class=eyebrow>{'혼잡도 혼합' if o['run']=='MixedLoad_L2' else '혼잡도 + CSI 신뢰도 혼합'}</div>
  <h2>{o['run']}</h2>
  <p class=meta>PPO = <b>{o['ckpt']}</b> · UE 속도: <b>{o['speednote']}</b> · 공식 baseline 2×4 그리드 (SUS 계열 threshold 0.8) · 20 seeds (10000–10019), paired</p>
  {tiles(o['tiles'])}
  <h3>요약 — 표준 지표 9종 × 9 스케줄러 + paired 검정 (reward 기준)</h3>
  <div class=twrap><table><thead>{THS}</thead><tbody>{o['sumrows']}</tbody></table></div>
  <h3>Seed별 reward — 조건(인원·속도)과 함께</h3>
  <div class=twrap><table><thead>{THA}</thead><tbody>{o['rowsA']}</tbody></table></div>
  <div class=note><b>PPO가 8종 전원을 이긴 seed: {o['sweep']}/20.</b> 놓친 seed: {o['losstxt']} —
  전부 저부하(인원 16–19) 구간, 그 seed들의 승자는 SU+CQI.</div>
  <details><summary>전 지표 상세 (seed × 9 스케줄러 × 9 metrics, 180행)</summary>
  <div class=twrap><table><thead>{THB}</thead><tbody>{o['rowsB']}</tbody></table></div>
  </details>
</section>"""
defense="""
<section>
  <div class=eyebrow>별도: 적응성 상한 방어 실험 (공식 baseline 아님)</div>
  <h2>Switch-hybrid / Oracle-envelope 분석</h2>
  <p class=meta>리뷰어 반론("적응은 switch 한 줄이면 된다") 검증용 — 같은 20 seed.</p>
  <div class=twrap><table>
  <thead><tr><th class=l>기준</th><th>MixedLoad_L2</th><th>MixedSpeed_L2b</th></tr></thead>
  <tbody>
  <tr><td>실현가능 switch 최강 (Hybrid, backlog-UE-수 임계 sweep)</td><td class=num>8,768 (T=14)</td><td class=num>8,698 (T=14)</td></tr>
  <tr><td>oracle envelope (seed별 SU+CQI/SUS+CQI 완벽 선택 — 실현 불가 상한)</td><td class=num>8,989</td><td class=num>8,964</td></tr>
  <tr class=ppo-row><td>PPO − oracle envelope (paired)</td><td class=num>+571 (+6.4%) CI±233, 17/20</td><td class=num>+617 (+6.9%) CI±207, 16/20</td></tr>
  </tbody></table></div>
</section>"""
html=f"""<title>Run3 최종 20-seed 평가 — seed별 전체 결과</title>
<style>{css}</style>
<main>
<div class=eyebrow>ML DRL Scheduler · Run3 최종 확정 평가 · 공식 baseline + 표준 지표 9종 · 2026-07-06 (JFI = active 유저 기준)</div>
<h1>Seed별 전체 결과: PPO vs 공식 Baseline 8종</h1>
<p class=lede>Baseline = 지정된 <b>2×4 그리드</b> {{SUS-게이트 MU | SU}} × {{CQI, Deadline-PF, PF, Random}} (SUS 계열 threshold 0.8, sweep 확정 최적). Metric = 표준 9종. <b>JFI는 active 유저만으로 계산</b> (전체-K 값은 원자료 jain_allk 열에 audit용 보존). 차이값은 <b>절대값과 % 병기</b>. 각 seed는 독립 시나리오이며 9개 스케줄러가 같은 에피소드를 실행(paired). <b>굵은 값</b> = 그 seed의 1등.</p>
{secs}
{defense}
<footer>
프로토콜: episode 1000 슬롯 · K=32 · n_active ~ U[16,32]/seed · PPO는 각 run의 best.pt(argmax) · Random 계열 seed 오프셋 +100003/+100004.<br>
해석 주의: <b>SINR</b>은 scheduled position 평균이라 SU 계열이 구조적으로 높음(높은 SINR ≠ 높은 성능). <b>JFI</b>는 active 유저 기준; Random 계열의 높은 JFI는 하향 평준화 효과(총량과 함께 읽을 것). <b>Total fail</b> = miss + retx drop + overflow. 에피소드 종료 시 미결(pending) 패킷은 어느 집계에도 안 들어감 — 전 스케줄러 0.15~0.42%, 조건 동일.<br>
원자료: <code>Run3/_analysis_20260702/official9full_{{run}}.csv</code> (jain=active-only, jain_allk=audit)
</footer>
</main>"""
open("final20_report.html","w").write(html)
for t in ("table","thead","tbody","details","div","section","main"):
    o,c=len(re.findall(f"<{t}[ >]",html)),html.count(f"</{t}>")
    assert o==c,(t,o,c)
print("html ok,",len(html),"bytes")
