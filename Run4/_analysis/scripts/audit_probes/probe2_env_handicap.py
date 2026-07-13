"""Probe 2: env-level measurement of the SU-CQI b_tx artifact (C1/C2/C4).

Runs SU+CQI and SUS+CQI@0.8 on short episodes and instruments the
transmission manager to record, per attempt: depth (units in the RBG),
first-attempt ACK rate, attempts per ACKed unit, retx-pinned position
fraction. Two CSI settings:
  A) genie + p_csi=1.0  (perfect CSI -> isolates the b_tx artifact, C4)
  B) type2 + p_csi=0.6  (realistic Run3-like point)
K=16 hard point (p_arrival=0.4, deadline 3-12), episode_len=200, 2 seeds.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import sys
import time
import collections
import numpy as np

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                    # noqa: E402
from env import SchedulerEnv                 # noqa: E402
from baselines import SUCQI, SUSCQI          # noqa: E402


def run_one(cfg, sched, episodes=(0, 1)):
    env = SchedulerEnv(cfg)
    stats = dict(first=collections.defaultdict(lambda: [0, 0]),  # depth -> [acks, tries]
                 attempts_acked=[], pinned=0, occupied=0,
                 comp=0, miss=0, retx_drop=0, ovf=0, arrivals=0,
                 acked_bits=0.0, delays=[], depth_num=0.0, depth_den=0)
    for ep in episodes:
        env.reset(episode_idx=ep)
        orig = env.txmgr.process_slot

        def wrapped(sinr_map, _orig=orig, _env=env, _st=stats):
            units = list(_env.txmgr.units)
            per_rbg = collections.Counter(u.rbg_id for u in units)
            pre = [(u, u.tx_attempts, per_rbg[u.rbg_id]) for u in units]
            out = _orig(sinr_map)
            acked_ids = {u.unit_id for u in out.acked}
            for u, att0, depth in pre:
                if att0 == 0:                      # first attempt
                    _st["first"][depth][1] += 1
                    if u.unit_id in acked_ids:
                        _st["first"][depth][0] += 1
                _st["depth_num"] += depth
                _st["depth_den"] += 1
            for u in out.acked:
                _st["attempts_acked"].append(u.tx_attempts)
            return out

        env.txmgr.process_slot = wrapped
        done = False
        while not done:
            stats["pinned"] += int(env.fixed_mask.sum())
            alloc = sched.schedule(env)
            _, _, done, info = env.step(alloc)
            stats["occupied"] += info["n_scheduled"]
        stats["comp"] += env.ep["n_comp"]
        stats["miss"] += env.ep["n_miss_deadline"]
        stats["retx_drop"] += env.ep["n_retx_drop"]
        stats["ovf"] += env.ep["n_retx_overflow_drop"]
        stats["arrivals"] += env.ep["n_arrivals"]
        stats["acked_bits"] += env.ep["acked_bits"]
        stats["delays"] += env.ep["delays"]
    return stats


def report(tag, sched_name, st):
    att = np.array(st["attempts_acked"])
    print(f"\n[{tag}] {sched_name}")
    print(f"  mean depth over unit-attempts: "
          f"{st['depth_num']/max(st['depth_den'],1):.2f}")
    for d in sorted(st["first"]):
        a, n = st["first"][d]
        print(f"  first-attempt ACK rate @depth {d}: {a}/{n} = {a/max(n,1):.3f}")
    if att.size:
        print(f"  attempts per ACKed unit: mean {att.mean():.2f} "
              f"(dist {np.bincount(att, minlength=6)[1:6]})")
    print(f"  retx-pinned positions / scheduled: "
          f"{st['pinned']}/{st['occupied']} = "
          f"{st['pinned']/max(st['occupied'],1):.3f}")
    n_pk = max(st["arrivals"], 1)
    print(f"  arrivals {st['arrivals']}  comp {st['comp']} "
          f"({st['comp']/n_pk:.3f})  miss {st['miss']} "
          f"({st['miss']/n_pk:.3f})  retx_drop {st['retx_drop']} "
          f"({st['retx_drop']/n_pk:.3f})  ovf {st['ovf']}")
    if st["delays"]:
        print(f"  mean completion delay: {np.mean(st['delays']):.2f} slots")
    print(f"  acked bits: {st['acked_bits']/1e6:.2f} Mbit")


t0 = time.time()
base = dict(debug=True, episode_len_debug=200, num_ue=16,
            p_arrival=0.4, deadline_min=3, deadline_max=12,
            sus_ortho_threshold=0.8)

for tag, extra in (
        ("A genie p_csi=1.0", dict(pmi_mode="genie", p_csi=1.0,
                                   ue_speed_kmh=30.0)),
        ("B type2 p_csi=0.6", dict(pmi_mode="type2_sparse_56bit", p_csi=0.6,
                                   ue_speed_kmh=10.0))):
    cfg = Config(**base, **extra)
    for sched in (SUCQI(), SUSCQI()):
        st = run_one(cfg, sched)
        report(tag, sched.name, st)
    print(f"  [{time.time()-t0:.0f}s elapsed]")
