"""Dense-reward leak probe (user decision 2026-07-22).

Question: of the dense reward actually paid (per-slot useful-MI accrual,
the Delta-I_useful term), what fraction went to packets that ultimately
FAILED (deadline miss / retx drop)? That fraction is the goodput-
misalignment "leak" -- the number that decides whether the reward needs
restructuring (leak >~10%) or a paper paragraph (leak ~3%).

Method: monkey-patch TransmissionManager.process_slot (root py untouched)
to attribute each unit's last_useful to its parent packet (object ref
registered at pay time via env.traffic). At episode end classify every
ever-paying packet: completed (is_complete) / live-at-end (unresolved) /
failed. Report bits-weighted and urgency-weighted (1+1/(D+1)) splits.
Ent02 world, PPO best vs SUS+CQI@0.7, seeds 10000-10004.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import PPOScheduler
import transmission

os.chdir("/home/MYH/ML_DRL_Scheduler")
raw = json.load(open("Run4/QueuePostRZF_Ent02/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
cfg.sus_ortho_threshold = 0.7
env = SchedulerEnv(cfg)

ac = ActorCritic(cfg)
ac.load_state_dict(torch.load("Run4/QueuePostRZF_Ent02/ckpt/best.pt",
                              map_location="cpu")["model"], strict=True)
ac.eval()
sus = [s for s in all_baselines(cfg) if s.name == "SUS+CQI"][0]
scheds = [("PPO-Ent02", PPOScheduler(ac, deterministic=True)), ("SUS+CQI", sus)]

# ---- instrumentation state (reset per episode) ----
PAY, PAY_W, REG = {}, {}, {}          # pid -> bits / urgency-weighted / pkt ref

orig_process = transmission.TransmissionManager.process_slot
def patched_process(self, sinr_map):
    out = orig_process(self, sinr_map)
    slot = env.slot
    for u in list(self.units) + out.acked + out.dropped:
        if u.last_useful <= 0:
            continue
        pkt = env.traffic.packets[u.ue_id]
        if pkt is None or pkt.packet_id != u.packet_id:
            pkt = REG.get(u.packet_id)          # resolved mid-bookkeeping
        if pkt is not None:
            REG[u.packet_id] = pkt
            d_rem = max(int(pkt.deadline) - slot, 0)
        else:
            d_rem = 0
        w = 1.0 + 1.0 / (d_rem + 1)
        PAY[u.packet_id] = PAY.get(u.packet_id, 0.0) + u.last_useful
        PAY_W[u.packet_id] = PAY_W.get(u.packet_id, 0.0) + w * u.last_useful
    return out
transmission.TransmissionManager.process_slot = patched_process

SEEDS = range(10000, 10005)
print(f"{'sched':10s} {'seed':>5s} {'지급bits':>10s} {'완성%':>7s} {'실패%':>7s} {'미해결%':>8s}")
agg = {}
for name, sch in scheds:
    tot = np.zeros(3)      # completed, failed, unresolved (bits)
    tot_w = np.zeros(3)    # urgency-weighted
    for seed in SEEDS:
        PAY.clear(); PAY_W.clear(); REG.clear()
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        live_now = {p.packet_id for p in env.traffic.packets if p is not None}
        b = np.zeros(3); bw = np.zeros(3)
        for pid, paid in PAY.items():
            pkt = REG.get(pid)
            if pkt is not None and pkt.is_complete:
                k = 0
            elif pid in live_now:
                k = 2
            else:
                k = 1
            b[k] += paid; bw[k] += PAY_W[pid]
        tot += b; tot_w += bw
        s = b.sum()
        print(f"{name:10s} {seed:5d} {s/1e6:9.2f}M {b[0]/s:7.1%} {b[1]/s:7.1%} {b[2]/s:8.1%}", flush=True)
    s, sw = tot.sum(), tot_w.sum()
    agg[name] = (tot, tot_w)
    print(f"{name:10s} {'평균':>5s} {s/1e6:9.2f}M {tot[0]/s:7.1%} {tot[1]/s:7.1%} {tot[2]/s:8.1%}"
          f"   [urgency-가중 실패비중 {tot_w[1]/sw:.1%}]", flush=True)
print("\n누수율(=실패 패킷에 지급된 dense 보상 비중):")
for name, (t, tw) in agg.items():
    print(f"  {name:10s} bits기준 {t[1]/t.sum():.2%}  |  urgency-가중 {tw[1]/tw.sum():.2%}")
