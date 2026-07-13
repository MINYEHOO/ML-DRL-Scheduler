"""RNG-coupling probe for audit claims B1/B2.

Part A: legacy (queue_size=1, Run3) mode -- one seed, SUS+CQI vs SU+CQI,
        diff per-slot arrival events and CSI feedback masks; hash channel.
Part B: queue mode (queue_size=8, Run4) -- high load (overflow) vs low load
        (no overflow); same diff.
Part C: variance decomposition -- fix channel seed, vary traffic/CSI-mask
        stream; between-channel vs within-channel variance of episode reward.
Part D: realized pairing correlation -- SUS+CQI vs SU+CQI on 8 seeds,
        Var(paired diff) vs Var(A)+Var(B).

CPU only, read-only on the repo, episode_len=200, K=8.
"""
import os, sys, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config
from env import SchedulerEnv
from baselines import SUSCQI, SUCQI


def instrument(env):
    """Wrap traffic.generate_arrivals and csi.step to log events."""
    logs = {"arr": [], "mask": [], "ovf": []}
    tm, orig_gen = env.traffic, env.traffic.generate_arrivals
    def gen(slot, rng):
        new, ovf = orig_gen(slot, rng)
        rec = tuple(sorted((u, tm.queues[u][-1].size, tm.queues[u][-1].deadline)
                           for u in new))
        logs["arr"].append(rec); logs["ovf"].append(ovf)
        return new, ovf
    tm.generate_arrivals = gen
    orig_step = env.csi.step
    def cstep(rp, dt, ct, rng):
        fb = orig_step(rp, dt, ct, rng)
        logs["mask"].append(fb.copy())
        return fb
    env.csi.step = cstep
    return logs


def run_episode(env, sched, episode_idx, logs):
    logs["arr"].clear(); logs["mask"].clear(); logs["ovf"].clear()
    env.reset(episode_idx)
    chash = hashlib.md5(env.channel.h_true_episode.tobytes()).hexdigest()[:10]
    cqih = hashlib.md5(np.ascontiguousarray(env._episode_csi.cqi).tobytes()
                       ).hexdigest()[:10]
    done = False
    while not done:
        _, _, done, _ = env.step(sched.schedule(env))
    return dict(chash=chash, cqih=cqih,
                arr=list(logs["arr"]), mask=[m.copy() for m in logs["mask"]],
                ovf=list(logs["ovf"]), reward=env.ep["reward"],
                acked=env.ep["acked_bits"], narr=env.ep["n_arrivals"])


def diff_runs(ra, rb, T):
    first_arr = next((t for t in range(T)
                      if ra["arr"][t] != rb["arr"][t]), None)
    n_arr_diff = sum(ra["arr"][t] != rb["arr"][t] for t in range(T))
    M = min(len(ra["mask"]), len(rb["mask"]))
    first_mask = next((t for t in range(M)
                       if not np.array_equal(ra["mask"][t], rb["mask"][t])),
                      None)
    n_mask_diff = sum(not np.array_equal(ra["mask"][t], rb["mask"][t])
                      for t in range(M))
    return first_arr, n_arr_diff, first_mask, n_mask_diff, M


class ProbeEnv(SchedulerEnv):
    """Allows overriding the main env RNG stream (traffic + CSI-feedback
    Bernoulli) while keeping channel / true-CSI / n_active identical."""
    traffic_seed_override = None
    def _prepare_slot(self, slot):
        if slot == 0 and self.traffic_seed_override is not None:
            self.rng = np.random.default_rng(self.traffic_seed_override)
        super()._prepare_slot(slot)


def part_AB():
    print("=" * 70)
    print("PART A: legacy queue_size=1 (Run3 mode), one seed, two schedulers")
    cfg = Config(debug=True, num_ue=8, episode_len_debug=200,
                 p_csi=0.6, p_arrival=0.4, deadline_min=3, deadline_max=12,
                 ue_speed_kmh=30.0, queue_size=1)
    env = SchedulerEnv(cfg); logs = instrument(env)
    ra = run_episode(env, SUSCQI(), 0, logs)
    rb = run_episode(env, SUCQI(), 0, logs)
    T = cfg.episode_len
    fa, na, fm, nm, M = diff_runs(ra, rb, T)
    print(f"  channel hash  A={ra['chash']} B={rb['chash']} "
          f"identical={ra['chash']==rb['chash']}")
    print(f"  true-CQI hash A={ra['cqih']} B={rb['cqih']} "
          f"identical={ra['cqih']==rb['cqih']}")
    print(f"  slot-0 arrivals identical: {ra['arr'][0]==rb['arr'][0]}")
    print(f"  first differing ARRIVAL slot : {fa}  "
          f"({na}/{T} slots differ; totals {ra['narr']} vs {rb['narr']})")
    print(f"  first differing CSI MASK slot: "
          f"{fm+1 if fm is not None else None}  ({nm}/{M} masks differ)")
    per_ue_fb_a = np.mean([m.mean() for m in ra["mask"]])
    per_ue_fb_b = np.mean([m.mean() for m in rb["mask"]])
    print(f"  mean fb rate A={per_ue_fb_a:.3f} B={per_ue_fb_b:.3f} "
          f"(cfg p_csi={cfg.p_csi})")

    print("=" * 70)
    print("PART B: queue mode queue_size=8 (Run4)")
    for tag, pa, dmin, dmax in [("high-load", 0.6, 20, 30),
                                ("low-load", 0.10, 5, 30)]:
        cfgq = Config(debug=True, num_ue=8, episode_len_debug=200,
                      p_csi=0.6, p_arrival=pa, deadline_min=dmin,
                      deadline_max=dmax, ue_speed_kmh=30.0, queue_size=8)
        envq = SchedulerEnv(cfgq); logsq = instrument(envq)
        ra = run_episode(envq, SUSCQI(), 0, logsq)
        rb = run_episode(envq, SUCQI(), 0, logsq)
        fa, na, fm, nm, M = diff_runs(ra, rb, cfgq.episode_len)
        print(f"  [{tag}] p_arrival={pa} overflow A={sum(ra['ovf'])} "
          f"B={sum(rb['ovf'])}")
        print(f"    first diff arrival slot: {fa} ({na}/200), "
              f"first diff mask slot: {fm+1 if fm is not None else None} "
              f"({nm}/{M}); arrivals identical={na==0} masks identical={nm==0}")


def part_CD():
    print("=" * 70)
    print("PART C: variance decomposition (SUS+CQI, legacy hard-ish point)")
    cfg = Config(debug=True, num_ue=8, episode_len_debug=200,
                 p_csi=0.6, p_arrival=0.4, deadline_min=3, deadline_max=12,
                 ue_speed_kmh=30.0, queue_size=1)
    env = ProbeEnv(cfg)
    C, J = 8, 5           # 8 channel seeds x 5 traffic/CSI-mask streams
    sched = SUSCQI()
    R = np.zeros((C, J))
    for c in range(C):
        for j in range(J):
            env.traffic_seed_override = (None if j == 0
                                         else 555_000 + 97 * c + j)
            env.reset(c)
            done = False
            while not done:
                _, _, done, _ = env.step(sched.schedule(env))
            R[c, j] = env.ep["reward"]
    env.traffic_seed_override = None
    gm = R.mean()
    msb = J * ((R.mean(axis=1) - gm) ** 2).sum() / (C - 1)
    msw = ((R - R.mean(axis=1, keepdims=True)) ** 2).sum() / (C * (J - 1))
    s2_b = max((msb - msw) / J, 0.0)
    icc = s2_b / (s2_b + msw) if (s2_b + msw) > 0 else float("nan")
    print(f"  reward matrix (rows=channel seed, cols=traffic stream):")
    for c in range(C):
        print("   ", " ".join(f"{v:8.1f}" for v in R[c]))
    print(f"  between-channel var (shared)  : {s2_b:12.1f}")
    print(f"  within-channel var (unshared) : {msw:12.1f}  "
          f"(traffic timing + CSI-mask)")
    print(f"  ICC (share of episode variance from the SHARED channel): "
          f"{icc:.3f}")

    print("=" * 70)
    print("PART D: realized pairing gain, SUS+CQI vs SU+CQI, 8 seeds")
    envd = SchedulerEnv(cfg)
    A, B = [], []
    for s in range(8):
        for sched2, acc in [(SUSCQI(), A), (SUCQI(), B)]:
            envd.reset(s)
            done = False
            while not done:
                _, _, done, _ = envd.step(sched2.schedule(envd))
            acc.append(envd.ep["reward"])
    A, B = np.array(A), np.array(B)
    d = A - B
    corr = np.corrcoef(A, B)[0, 1]
    vr = d.var(ddof=1) / (A.var(ddof=1) + B.var(ddof=1))
    print(f"  per-seed reward A(SUS+CQI): {np.round(A,1)}")
    print(f"  per-seed reward B(SU+CQI) : {np.round(B,1)}")
    print(f"  corr(A,B) across seeds = {corr:.3f}")
    print(f"  Var(A-B)/(Var A + Var B) = {vr:.3f}  "
          f"(1.0 = no pairing benefit, <1 = variance reduction)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    if "A" in which or "B" in which:
        part_AB()
    if "C" in which or "D" in which:
        part_CD()
    print("probe done.")
