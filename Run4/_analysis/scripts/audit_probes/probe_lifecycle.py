"""Part B: instrumented unit-lifecycle probe at the K=32 type2 point.

For SUS+CQI and CQI-greedy, mu_aware_la OFF vs ON, seeds 10000-10002
(same episode indices as Run4/_analysis/la_ablation_type2.csv rows),
300 slots each, record per transmission unit:
  - whether b_tx was backlog-capped (b_tx == uncommitted backlog < cap)
    or cap-limited (b_tx == LA cap),
  - m (co-scheduled stream count of its RBG) at first attempt,
  - first-attempt ACK or NACK, and first-attempt MI/b_tx ratio,
  - final attempts count and outcome (ack / retx-drop / purged-or-pending).
Per slot: pinned (fixed_mask) positions vs total scheduled positions.
Repo is imported read-only; nothing under the repo is written.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

from collections import Counter
import json
import numpy as np

from config import phase4_queue_config
from env import SchedulerEnv
from baselines import SUSCQI, CQIGreedy
from transmission import TransmissionManager
from phy import mi_bits

cfg = phase4_queue_config(episode_len_main=300)
env = SchedulerEnv(cfg)

records = {}          # unit_id -> record dict (per run, cleared)
ACKED, DROPPED = set(), set()

orig_create = TransmissionManager.create_unit
orig_process = TransmissionManager.process_slot


def create_probe(self, packet, rbg_id, layer_id, b_tx):
    pre_unc = packet.uncommitted_backlog
    u = orig_create(self, packet, rbg_id, layer_id, b_tx)
    records[u.unit_id] = dict(
        backlog_capped=bool(b_tx >= pre_unc - 1e-9),
        b_tx=float(b_tx), first_m=None, first_ack=None, first_ratio=None,
        obj=u)
    return u


def process_probe(self, sinr_map):
    mcount = Counter(u.rbg_id for u in self.units)
    firsts = [u for u in self.units if u.tx_attempts == 0]
    pre = {}
    for u in firsts:
        delta_i = float(mi_bits(sinr_map[u.rbg_id, u.current_layer_id], cfg))
        pre[u.unit_id] = (mcount[u.rbg_id], delta_i)
    out = orig_process(self, sinr_map)
    for u in out.acked:
        ACKED.add(u.unit_id)
    for u in out.dropped:
        DROPPED.add(u.unit_id)
    for u in firsts:
        rec = records[u.unit_id]
        m, delta_i = pre[u.unit_id]
        rec["first_m"] = m
        rec["first_ack"] = u.unit_id in ACKED
        rec["first_ratio"] = delta_i / rec["b_tx"] if rec["b_tx"] > 0 else np.inf
    return out


TransmissionManager.create_unit = create_probe
TransmissionManager.process_slot = process_probe

SCHEDS = {"SUS+CQI": SUSCQI, "CQI-greedy": CQIGreedy}


def summarize(pinned, sched_pos):
    cats = {
        "m1": lambda r: r["first_m"] == 1,
        "m2+_caplim": lambda r: r["first_m"] >= 2 and not r["backlog_capped"],
        "m2+_blkcap": lambda r: r["first_m"] >= 2 and r["backlog_capped"],
    }
    row = dict(pinned_frac=pinned / max(sched_pos, 1))
    recs = [r for r in records.values() if r["first_m"] is not None]
    row["n_units"] = len(recs)
    for name, f in cats.items():
        sub = [r for r in recs if f(r)]
        row[name + "_n"] = len(sub)
        if not sub:
            continue
        row[name + "_firstack"] = float(np.mean([r["first_ack"] for r in sub]))
        acked = [r for r in sub if r["obj"].unit_id in ACKED]
        drop = [r for r in sub if r["obj"].unit_id in DROPPED]
        row[name + "_ackedfrac"] = len(acked) / len(sub)
        row[name + "_dropfrac"] = len(drop) / len(sub)
        if acked:
            att = np.array([r["obj"].tx_attempts for r in acked])
            row[name + "_att_mean"] = float(att.mean())
            row[name + "_att_hist"] = np.bincount(att, minlength=6)[1:6].tolist()
        rat = np.array([r["first_ratio"] for r in sub])
        row[name + "_ratio_mean"] = float(np.mean(rat))
        row[name + "_ratio_med"] = float(np.median(rat))
        row[name + "_ratio_ge1"] = float(np.mean(rat >= 1.0))
    # attempts split for exactly m==2 cap-limited (GPT: unchanged 2 slots)
    for mm in (2, 3, 4):
        sub = [r for r in recs
               if r["first_m"] == mm and not r["backlog_capped"]
               and r["obj"].unit_id in ACKED]
        if sub:
            att = np.array([r["obj"].tx_attempts for r in sub])
            row[f"m{mm}_caplim_att_mean"] = float(att.mean())
            row[f"m{mm}_caplim_n"] = len(sub)
    return row


results = []
for ep_idx in (10000, 10001, 10002):
    for sched_name, sched_cls in SCHEDS.items():
        for la in (False, True):
            records.clear(); ACKED.clear(); DROPPED.clear()
            env.cfg.mu_aware_la = la
            env.reset(episode_idx=ep_idx)
            sched = sched_cls()
            pinned, sched_pos = 0, 0
            done = False
            while not done:
                pinned += int(env.fixed_mask.sum())
                alloc = sched.schedule(env)
                _, _, done, info = env.step(alloc)
                sched_pos += int(info["n_scheduled"])
            row = summarize(pinned, sched_pos)
            row.update(seed=ep_idx, sched=sched_name, la=int(la),
                       retx_drop=env.ep["n_retx_drop"],
                       miss=env.ep["n_miss_deadline"],
                       comp=env.ep["n_comp"], arrivals=env.ep["n_arrivals"])
            results.append(row)
            print(json.dumps(row, default=str))
            sys.stdout.flush()

np.save("/tmp/gpt_r2_verify/lifecycle_results.npy", results, allow_pickle=True)

# ---- aggregate table -------------------------------------------------
print("\n==== AGGREGATE (mean over seeds) ====")
hdr = (f"{'sched':>10} {'la':>2} {'pin%':>6} {'1stACK m1':>9} "
       f"{'1stACK m2+cap':>13} {'1stACK m2+blk':>13} "
       f"{'att m2+cap':>10} {'att m2':>7} {'att m3':>7} {'att m4':>7} "
       f"{'ratio_med':>9} {'drops':>6}")
print(hdr)
for sched_name in SCHEDS:
    for la in (0, 1):
        rows = [r for r in results if r["sched"] == sched_name and r["la"] == la]
        def m(key):
            vals = [r[key] for r in rows if key in r]
            return float(np.mean(vals)) if vals else float("nan")
        print(f"{sched_name:>10} {la:>2} {100*m('pinned_frac'):>6.1f} "
              f"{m('m1_firstack'):>9.3f} {m('m2+_caplim_firstack'):>13.3f} "
              f"{m('m2+_blkcap_firstack'):>13.3f} {m('m2+_caplim_att_mean'):>10.2f} "
              f"{m('m2_caplim_att_mean'):>7.2f} {m('m3_caplim_att_mean'):>7.2f} "
              f"{m('m4_caplim_att_mean'):>7.2f} {m('m2+_caplim_ratio_med'):>9.3f} "
              f"{m('retx_drop'):>6.1f}")
