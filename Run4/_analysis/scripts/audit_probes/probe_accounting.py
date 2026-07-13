"""Probe for audit claims A1/A2/A3 (packet/reward accounting).

A1: ACKed bits of a later-dropped packet stay in throughput/cum_acked/reward.
A2: throughput_mbps != application goodput; goodput-0 packet can net positive reward.
A3: quantify acked-vs-goodput gap on real episodes (correction feasibility).

Read-only on the repo; CPU only.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np

import env as envmod
from config import Config, phase2_debug_config
from env import SchedulerEnv
from traffic import Packet, TrafficModel
from metrics import jains_index

R_SEP = "=" * 72

# keep originals for restoration
ORIG_PREDICT = envmod.predict_b_tx
ORIG_SINR = envmod.compute_slot_sinr

BTX = {"val": 6000.0}
SINR = {"val": 1e9}


def patched_predict(cqi, cfg):
    return float(BTX["val"])


def patched_sinr(realized, h_true, h_hat, p_rbg, noise, alpha):
    return np.full((realized.shape[0], realized.shape[1]), SINR["val"],
                   dtype=np.float64)


def make_env():
    cfg = Config(debug=True, episode_len_debug=6, num_ue=8, p_arrival=0.0)
    e = SchedulerEnv(cfg)
    e.reset(0)
    assert all(p is None for p in e.traffic.packets), "expected no arrivals"
    return cfg, e


def inject(e, size, deadline):
    pkt = Packet(packet_id=990, ue_id=0, arrival_slot=e.slot,
                 size=size, deadline=deadline)
    e.traffic.queues[0].append(pkt)
    e.traffic._sync_hol(0)
    return pkt


def alloc_ue0(cfg):
    a = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    a[0, 0] = 1  # UE 0 on RBG0/layer0
    return a


print(R_SEP)
print("PROBE 1 (A1): partial ACK slot t, deadline-miss drop slot t+1")
print(R_SEP)
envmod.predict_b_tx = patched_predict
envmod.compute_slot_sinr = patched_sinr

cfg, e = make_env()
pkt = inject(e, size=10000, deadline=2)

BTX["val"] = 6000.0   # unit A: 6000 bits
SINR["val"] = 1e9     # huge -> ACKs in one slot
_, r1, _, i1 = e.step(alloc_ue0(cfg))
print(f"slot1: reward={r1:+.4f} r_short={i1['reward_short']:.4f} "
      f"acked_bits_slot={i1['acked_bits']:.0f} comp={i1['n_comp']} "
      f"miss={i1['n_miss_deadline']}")
print(f"       pkt.acked_bits={pkt.acked_bits:.0f}/{pkt.size} "
      f"cum_acked[0]={e.cum_acked_bits[0]:.0f} ep.acked={e.ep['acked_bits']:.0f}")
assert abs(e.cum_acked_bits[0] - 6000.0) < 1e-6
assert i1["n_comp"] == 0 and i1["n_miss_deadline"] == 0

BTX["val"] = 4000.0   # unit B: remaining 4000 bits
SINR["val"] = 0.3     # ~509 bits of MI -> NACK; deadline 1->0 -> miss
_, r2, _, i2 = e.step(alloc_ue0(cfg))
print(f"slot2: reward={r2:+.4f} r_short={i2['reward_short']:.4f} "
      f"acked_bits_slot={i2['acked_bits']:.0f} comp={i2['n_comp']} "
      f"miss={i2['n_miss_deadline']}")
assert i2["n_miss_deadline"] == 1 and i2["n_comp"] == 0
assert len(e.txmgr.units) == 0, "units purged on drop"

# --- NO ROLLBACK: the 6000 ACKed bits of the dropped packet persist ---
print(f"after drop: cum_acked[0]={e.cum_acked_bits[0]:.0f} (NOT rolled back)  "
      f"ep.acked_bits={e.ep['acked_bits']:.0f}  "
      f"ep.acked_per_ue[0]={e.ep['acked_per_ue'][0]:.0f}")
assert abs(e.cum_acked_bits[0] - 6000.0) < 1e-6
assert abs(e.ep["acked_bits"] - 6000.0) < 1e-6

done = False
while not done:
    _, _, done, _ = e.step(np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64))
ep_time = cfg.episode_len * cfg.slot_duration
thr = e.ep["acked_bits"] / ep_time / 1e6
comp_rate = e.ep["n_comp"] / max(e.ep["n_arrivals"], 1)
print(f"episode end: throughput_mbps={thr:.4f} (>0 from a 100%-failed packet), "
      f"n_comp={e.ep['n_comp']}, n_miss={e.ep['n_miss_deadline']}, "
      f"cum reward={e.ep['reward']:+.4f}")
print(f"NOTE: completion-based metrics DO see the failure: "
      f"completion_rate=0.0, miss counted -> packet-level goodput signal exists")
assert thr > 0 and e.ep["n_comp"] == 0
print("PROBE 1 PASSED: A1 mechanism confirmed (no rollback; miss penalty "
      "applied instead)")

print()
print(R_SEP)
print("PROBE 2 (A2): goodput-0 packet with NET POSITIVE cumulative reward")
print(R_SEP)
cfg, e = make_env()
pkt = inject(e, size=12000, deadline=1)
BTX["val"] = 11999.0  # unit covers all but 1 bit
SINR["val"] = 1e9     # ACKs same slot; deadline 1->0 -> miss same slot
_, r1, _, i1 = e.step(alloc_ue0(cfg))
w = 1.0 + cfg.eta_d / (1.0 + 1.0)  # deadline snapshot = 1 -> weight 1.5
expect = cfg.lambda_s * w * 11999.0 / cfg.b_norm - cfg.lambda_m * 1
print(f"slot1: reward={r1:+.4f} (expected {expect:+.4f}) "
      f"r_short={i1['reward_short']:.4f} miss={i1['n_miss_deadline']} "
      f"comp={i1['n_comp']}")
print(f"       pkt complete? {pkt.is_complete} (acked {pkt.acked_bits:.0f}"
      f"/{pkt.size}) -> application goodput = 0")
print(f"       cum_acked[0]={e.cum_acked_bits[0]:.0f} -> counted in throughput")
assert i1["n_comp"] == 0 and i1["n_miss_deadline"] == 1
assert not pkt.is_complete
assert r1 > 0, "net reward should be positive"
print(f"break-even: ACKed bits > lambda_m*b_norm/(lambda_s*w) = "
      f"{cfg.lambda_m*cfg.b_norm/(cfg.lambda_s*w):.0f} bits at weight {w}")
print("PROBE 2 PASSED: A2 'net positive reward with goodput 0' confirmed "
      "(needs near-full-size MI at tight deadline; penalty -2 usually wins)")

# restore real PHY
envmod.predict_b_tx = ORIG_PREDICT
envmod.compute_slot_sinr = ORIG_SINR

print()
print(R_SEP)
print("PROBE 3 (A3): real episodes -- acked throughput vs true packet goodput")
print(R_SEP)
from baselines import SUSPF, SUCQI  # noqa: E402

created = []
_orig_new = TrafficModel._new_packet


def _tracking_new(self, u, slot, rng):
    p = _orig_new(self, u, slot, rng)
    created.append(p)
    return p


TrafficModel._new_packet = _tracking_new

configs = {
    "default(K8,p=.2,D5-30)": phase2_debug_config(),
    "hard(K8,p=.4,D3-12,30kmh)": phase2_debug_config(
        p_arrival=0.4, deadline_min=3, deadline_max=12, ue_speed_kmh=30.0),
}
for cname, c in configs.items():
    for sched in (SUSPF(), SUCQI()):
        env2 = SchedulerEnv(c)
        rows = []
        for ep_idx in range(2):
            created.clear()
            env2.reset(episode_idx=ep_idx)
            done = False
            while not done:
                _, _, done, _ = env2.step(sched.schedule(env2))
            acked = env2.ep["acked_bits"]
            good = sum(p.size for p in created if p.is_complete)
            phantom = sum(p.acked_bits for p in created if not p.is_complete)
            good_ue = np.zeros(c.num_ue)
            for p in created:
                if p.is_complete:
                    good_ue[p.ue_id] += p.size
            j_acked = jains_index(env2.cum_acked_bits[:env2.traffic.n_active])
            j_good = jains_index(good_ue[:env2.traffic.n_active])
            rows.append((acked, good, phantom, j_acked, j_good,
                         env2.ep["n_comp"], env2.ep["n_miss_deadline"]
                         + env2.ep["n_retx_drop"]))
        acked, good, phantom, j_a, j_g, ncomp, nfail = map(
            lambda i: np.mean([r[i] for r in rows]), range(7))
        print(f"{cname:27s} {sched.name:8s} acked={acked/1e6:7.3f}Mb "
              f"goodput={good/1e6:7.3f}Mb phantom={phantom/1e6:6.3f}Mb "
              f"({100*phantom/max(acked,1):4.1f}% of acked) "
              f"Jain acked={j_a:.3f} vs goodput={j_g:.3f} "
              f"comp={ncomp:.0f} fail={nfail:.0f}")

TrafficModel._new_packet = _orig_new
print()
print("PROBE 3 done: phantom fraction = share of throughput_mbps that is not "
      "application goodput; identical accounting applied to every scheduler.")
