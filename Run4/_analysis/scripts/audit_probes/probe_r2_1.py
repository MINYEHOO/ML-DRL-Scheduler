"""R2-1 probe: scheduler-vs-env B_tx accounting mismatch under mu_aware_la.

K=32 type2 L2b point, SUS+CQI. Three paired variants per seed:
  A) la=0  SUS+CQI as-is            (historical LA, budgets consistent by constr.)
  B) la=1  SUS+CQI as-is            (env de-rates, scheduler budget = SU sizing)
  C) la=1  SUS+CQI consistent-budget (scheduler budget threads the de-rated,
                                      m-aware estimate with re-crediting)

Variant B is instrumented per slot (replay, zero side effects):
  believed commit (SU-sized budget threading, exactly baselines.py:57)
  vs actual env commit (exact mirror of env._sanitize_and_create m_planned
  de-rate + sequential pkt-uncommitted capping + epsilon swallow)
  -> stranded UEs (scheduler thinks exhausted, env truth has backlog left),
     next-slot re-servability / actual re-selection, eventual packet outcome.

Fraction of the la-on miss rise explained by the mismatch mechanism:
  (miss_B - miss_C) / (miss_B - miss_A), paired over seeds.
"""
from __future__ import annotations

import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "8")

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np
from config import Config
from env import SchedulerEnv
from baselines import SUSCQI

EPISODE_LEN = 300
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gpt_r2_verify/probe_r2_1_results.json"


def make_cfg() -> Config:
    # Run3 MixedSpeed_L2b point (Run4/_analysis/la_ablation_type2.csv point),
    # episode shortened 1000 -> 300 per probe constraints.
    return Config(debug=False, num_ue=32, episode_len_main=EPISODE_LEN,
                  n_active_min=16, n_active_max=32,
                  ue_speed_min=5.0, ue_speed_max=30.0,
                  p_csi=0.6, p_arrival=0.4,
                  deadline_min=3, deadline_max=12,
                  sus_ortho_threshold=0.5)


# ---------------------------------------------------------------------------
# consistent-budget SUS+CQI (counterfactual C)
# ---------------------------------------------------------------------------
class SUSCQIConsistent(SUSCQI):
    """SUS+CQI whose per-UE budget threading mirrors the env's m-aware,
    de-rated commit (with re-crediting as m grows during the slot)."""
    name = "SUS+CQI-cons"

    def schedule(self, env) -> np.ndarray:
        cfg = env.cfg
        obs = env.get_observation()
        R, L, K = cfg.num_rbg, cfg.l_max, cfg.num_ue
        alloc = np.zeros((R, L), dtype=np.int64)
        occupied = obs["occupied"]
        closed = np.zeros(R, dtype=bool)
        fixed_m = obs["fixed_mask"].sum(axis=1).astype(int)
        sel_ue = [set((occupied[r][occupied[r] > 0] - 1).tolist())
                  for r in range(R)]
        unc0 = obs["uncommitted"].astype(np.float64)
        cqi = obs["cqi_fb"]                       # [K, R]
        su_coef = cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
        picks: list[tuple[int, int]] = []         # (r, u) in traversal order
        new_m = np.zeros(R, dtype=int)

        def budgets() -> np.ndarray:
            m = fixed_m + new_m
            bud = unc0.copy()
            for (r, u) in picks:
                if cfg.mu_aware_la and m[r] > 1:
                    snr = 2.0 ** float(cqi[u, r]) - 1.0
                    cap = su_coef * np.log2(1.0 + snr / float(m[r]))
                else:
                    cap = su_coef * float(cqi[u, r])
                c = min(bud[u], cap)
                if bud[u] - c < cfg.b_tx_epsilon:
                    c = bud[u]                    # env epsilon-swallow rule
                bud[u] -= c
            return bud

        for l in range(L):
            for r in range(R):
                if occupied[r, l] > 0 or closed[r]:
                    continue
                bud = budgets()
                cand = env.position_candidates(r, sel_ue[r], bud)
                if not cand.any():
                    continue
                pick = self._select(env, obs, r, l, cand, sel_ue[r], closed)
                if pick is not None:
                    alloc[r, l] = pick + 1
                    sel_ue[r].add(pick)
                    picks.append((r, pick))
                    new_m[r] += 1
        return alloc


# ---------------------------------------------------------------------------
# replay: believed (SU threading) vs actual (env mirror) commits, no side FX
# ---------------------------------------------------------------------------
def replay_commits(env, allocation):
    cfg = env.cfg
    R, L, K = cfg.num_rbg, cfg.l_max, cfg.num_ue
    su_coef = cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
    cqi = env.csi.cqi_fb

    # m_planned exactly as env._sanitize_and_create
    m_planned = env.fixed_mask.sum(axis=1).astype(int)
    if cfg.mu_aware_la:
        seen = [set(s) for s in env.initial_S_r]
        closed = np.zeros(R, dtype=bool)
        for l in range(L):
            for r in range(R):
                if env.fixed_mask[r, l] or closed[r]:
                    continue
                k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
                if k == 0:
                    closed[r] = True
                    continue
                u = k - 1
                if not (0 <= u < K) or u in seen[r]:
                    continue
                pkt = env.traffic.packets[u]
                if pkt is None or pkt.uncommitted_backlog <= 0:
                    continue
                m_planned[r] += 1
                seen[r].add(u)

    unc0 = np.array([p.uncommitted_backlog if p is not None else 0.0
                     for p in env.traffic.packets])
    believed = np.zeros(K)     # scheduler's SU-sized budget deltas
    actual = np.zeros(K)       # env's real commits
    bud = unc0.copy()
    unc_t = unc0.copy()
    sel = [set(s) for s in env.initial_S_r]
    closed = np.zeros(R, dtype=bool)
    n_eps_skip = 0
    for l in range(L):
        for r in range(R):
            if env.fixed_mask[r, l]:
                continue
            if closed[r]:
                continue
            k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
            if k == 0:
                closed[r] = True
                continue
            u = k - 1
            if not (0 <= u < K) or u in sel[r]:
                continue
            pkt = env.traffic.packets[u]
            if pkt is None or unc_t[u] <= 0:
                continue
            # scheduler-believed commit (baselines.py:57 semantics)
            bb = min(bud[u], su_coef * float(cqi[u, r]))
            believed[u] += bb
            bud[u] -= bb
            # env-actual commit
            if cfg.mu_aware_la and m_planned[r] > 1:
                snr = 2.0 ** float(cqi[u, r]) - 1.0
                cap = su_coef * np.log2(1.0 + snr / float(m_planned[r]))
            else:
                cap = su_coef * float(cqi[u, r])
            b = min(unc_t[u], cap)
            if b < cfg.b_tx_epsilon:
                n_eps_skip += 1
                continue                      # env skips unit; no sel add
            if unc_t[u] - b < cfg.b_tx_epsilon:
                b = unc_t[u]
            actual[u] += b
            unc_t[u] -= b
            sel[r].add(u)
    return dict(believed=believed, actual=actual, bud=bud, unc_after=unc_t,
                unc0=unc0, m_planned=m_planned, n_eps_skip=n_eps_skip)


# ---------------------------------------------------------------------------
# episode runners
# ---------------------------------------------------------------------------
def ep_metrics(env):
    e = env.ep
    arr = max(e["n_arrivals"], 1)
    return dict(reward=e["reward"],
                acked_mbit=e["acked_bits"] / 1e6,
                comp=e["n_comp"] / arr,
                miss=e["n_miss_deadline"] / arr,
                retx_drop=e["n_retx_drop"] / arr,
                n_arrivals=e["n_arrivals"])


def run_plain(env, sched, ep_idx):
    env.reset(episode_idx=ep_idx)
    done = False
    pinned_pos = 0
    occ_pos = 0
    while not done:
        pinned_pos += int(env.fixed_mask.sum())
        alloc = sched.schedule(env)
        _, _, done, info = env.step(alloc)
        occ_pos += int(info["n_scheduled"])
    m = ep_metrics(env)
    m["pinned_frac"] = pinned_pos / max(occ_pos, 1)
    return m


def run_instrumented(env, sched, ep_idx):
    """Variant B with per-slot mismatch logging (replay only, no perturbation)."""
    cfg = env.cfg
    eps = cfg.b_tx_epsilon
    env.reset(episode_idx=ep_idx)
    done = False
    S = dict(slots=0, n_touched=0, n_under=0, n_stranded=0,
             believed_bits=0.0, actual_bits=0.0, shortfall_bits=0.0,
             ratio_sum=0.0, ratio_cnt=0,
             stranded_missed_same_slot=0, stranded_gone_same_slot=0,
             stranded_reservable_next=0, stranded_selected_next=0,
             n_eps_skip=0, pinned_pos=0, occ_pos=0,
             mplan_ge2_units=0, new_units=0)
    # stranded packet registry: pid -> ue ; outcome counts
    open_stranded: dict[int, int] = {}
    stranded_outcome = dict(completed=0, missed=0, retx_drop=0, open_end=0)
    prev_stranded: list[tuple[int, int]] = []   # (ue, pid) stranded last slot
    prev_miss = np.zeros(cfg.num_ue, np.int64)
    prev_comp = np.zeros(cfg.num_ue, np.int64)
    prev_retx = np.zeros(cfg.num_ue, np.int64)

    while not done:
        S["pinned_pos"] += int(env.fixed_mask.sum())
        alloc = sched.schedule(env)

        # --- next-slot re-selection check for last slot's stranded set ---
        sel_now = set((alloc[alloc > 0] - 1).tolist())
        for (u, pid) in prev_stranded:
            pkt = env.traffic.packets[u]
            if pkt is not None and pkt.packet_id == pid:
                if u in sel_now:
                    S["stranded_selected_next"] += 1

        rp = replay_commits(env, alloc)
        believed, actual = rp["believed"], rp["actual"]
        touched = believed > 0
        S["slots"] += 1
        S["n_touched"] += int(touched.sum())
        S["believed_bits"] += float(believed.sum())
        S["actual_bits"] += float(actual.sum())
        S["shortfall_bits"] += float(np.maximum(believed - actual, 0.0).sum())
        S["n_eps_skip"] += rp["n_eps_skip"]
        under = touched & (actual < believed - 1.0)
        S["n_under"] += int(under.sum())
        r = actual[touched] / np.maximum(believed[touched], 1e-9)
        S["ratio_sum"] += float(r.sum())
        S["ratio_cnt"] += int(touched.sum())
        # units created at m_planned >= 2 (share of new units that get de-rated)
        for rr in range(cfg.num_rbg):
            row = alloc[rr]
            newu = 0
            for ll in range(cfg.l_max):
                if env.fixed_mask[rr, ll]:
                    continue
                if row[ll] > 0:
                    newu += 1
            S["new_units"] += newu
            if rp["m_planned"][rr] >= 2:
                S["mplan_ge2_units"] += newu

        # stranded: scheduler thinks exhausted, env truth has >= eps left
        stranded_mask = touched & (rp["bud"] < eps) & (rp["unc_after"] >= eps)
        stranded = []
        for u in np.where(stranded_mask)[0]:
            pkt = env.traffic.packets[int(u)]
            if pkt is not None:
                stranded.append((int(u), pkt.packet_id))
                open_stranded[pkt.packet_id] = int(u)
        S["n_stranded"] += len(stranded)

        # capture pids/uncommitted prediction for post-step replay validation
        pids_before = {u: env.traffic.packets[u].packet_id
                       for u in range(cfg.num_ue)
                       if env.traffic.packets[u] is not None}

        _, _, done, info = env.step(alloc)
        S["occ_pos"] += int(info["n_scheduled"])

        # replay validation: surviving same-packet uncommitted must match
        for u, pid in pids_before.items():
            pkt = env.traffic.packets[u]
            if pkt is not None and pkt.packet_id == pid:
                assert abs(pkt.uncommitted_backlog - rp["unc_after"][u]) < 1e-6, \
                    (u, pid, pkt.uncommitted_backlog, rp["unc_after"][u])

        # per-slot outcome deltas
        d_miss = env.ep["miss_per_ue"] - prev_miss
        d_comp = env.ep["comp_per_ue"] - prev_comp
        d_retx = env.ep["retx_per_ue"] - prev_retx
        prev_miss = env.ep["miss_per_ue"].copy()
        prev_comp = env.ep["comp_per_ue"].copy()
        prev_retx = env.ep["retx_per_ue"].copy()

        # same-slot fate + next-slot re-servability of THIS slot's stranded
        still = []
        for (u, pid) in stranded:
            pkt = env.traffic.packets[u]
            alive = pkt is not None and pkt.packet_id == pid
            if not alive:
                S["stranded_gone_same_slot"] += 1
                if d_miss[u] > 0:
                    S["stranded_missed_same_slot"] += 1
            else:
                if pkt.uncommitted_backlog >= eps:
                    S["stranded_reservable_next"] += 1
                still.append((u, pid))
        prev_stranded = still

        # resolve registry outcomes (legacy 1-packet/UE: per-UE delta IDs it)
        gone = []
        for pid, u in open_stranded.items():
            pkt = env.traffic.packets[u]
            if pkt is not None and pkt.packet_id == pid:
                continue
            if d_comp[u] > 0:
                stranded_outcome["completed"] += 1
            elif d_miss[u] > 0:
                stranded_outcome["missed"] += 1
            elif d_retx[u] > 0:
                stranded_outcome["retx_drop"] += 1
            else:
                stranded_outcome["missed"] += 1   # overflow drop etc: count as fail
            gone.append(pid)
        for pid in gone:
            open_stranded.pop(pid)

    stranded_outcome["open_end"] = len(open_stranded)
    m = ep_metrics(env)
    m["pinned_frac"] = S["pinned_pos"] / max(S["occ_pos"], 1)
    m["mech"] = S
    m["stranded_outcome"] = stranded_outcome
    return m


def main():
    t0 = time.time()
    cfg = make_cfg()
    env = SchedulerEnv(cfg)
    results = []
    for i in range(N_SEEDS):
        ep_idx = 10000 + i
        rec = dict(seed=ep_idx)

        env.cfg.mu_aware_la = False
        rec["A_la0"] = run_plain(env, SUSCQI(), ep_idx)

        env.cfg.mu_aware_la = True
        rec["B_la1"] = run_instrumented(env, SUSCQI(), ep_idx)

        env.cfg.mu_aware_la = True
        rec["C_la1_cons"] = run_plain(env, SUSCQIConsistent(), ep_idx)

        results.append(rec)
        print(f"[{time.time()-t0:7.1f}s] seed {ep_idx}  "
              f"miss A={rec['A_la0']['miss']:.4f} "
              f"B={rec['B_la1']['miss']:.4f} "
              f"C={rec['C_la1_cons']['miss']:.4f}  "
              f"retxdrop A={rec['A_la0']['retx_drop']:.4f} "
              f"B={rec['B_la1']['retx_drop']:.4f}", flush=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)

    # ---- aggregate ----
    A = np.array([r["A_la0"]["miss"] for r in results])
    B = np.array([r["B_la1"]["miss"] for r in results])
    C = np.array([r["C_la1_cons"]["miss"] for r in results])
    print("\n=== miss rate (deadline miss / arrivals), paired ===")
    print(f"A la=0 as-is      : {A.mean():.4f}")
    print(f"B la=1 as-is      : {B.mean():.4f}")
    print(f"C la=1 consistent : {C.mean():.4f}")
    rise = B.mean() - A.mean()
    expl = B.mean() - C.mean()
    print(f"la-on miss rise (B-A)          : {rise:+.4f}")
    print(f"mismatch-explained part (B-C)  : {expl:+.4f}"
          f"  -> fraction {expl/rise if rise else float('nan'):.2%}")
    dBC = B - C
    print(f"paired B-C per seed: mean {dBC.mean():+.5f} std {dBC.std():.5f} "
          f"win C {int((dBC>0).sum())}/{len(dBC)}")
    dBA = B - A
    print(f"paired B-A per seed: mean {dBA.mean():+.5f} std {dBA.std():.5f}")

    for k in ("comp", "acked_mbit", "retx_drop", "pinned_frac"):
        a = np.mean([r["A_la0"][k] for r in results])
        b = np.mean([r["B_la1"][k] for r in results])
        c = np.mean([r["C_la1_cons"][k] for r in results])
        print(f"{k:12s}: A {a:.4f}  B {b:.4f}  C {c:.4f}")

    # ---- mechanism stats over B episodes ----
    def s(key):
        return float(np.sum([r["B_la1"]["mech"][key] for r in results]))
    slots = s("slots"); touched = s("n_touched")
    print("\n=== variant B per-slot mismatch instrumentation ===")
    print(f"slots {slots:.0f}, UE-slot touches {touched:.0f} "
          f"({touched/slots:.2f}/slot)")
    print(f"new units at m_planned>=2: {s('mplan_ge2_units'):.0f}/"
          f"{s('new_units'):.0f} = {s('mplan_ge2_units')/max(s('new_units'),1):.2%}")
    print(f"under-committed UE-slots (actual < believed-1): "
          f"{s('n_under'):.0f} = {s('n_under')/max(touched,1):.2%} of touches")
    print(f"mean actual/believed commit ratio: "
          f"{s('ratio_sum')/max(s('ratio_cnt'),1):.4f}")
    print(f"commit shortfall: {s('shortfall_bits')/1e6:.2f} Mbit "
          f"(believed {s('believed_bits')/1e6:.2f}, "
          f"actual {s('actual_bits')/1e6:.2f})")
    nstr = s("n_stranded")
    print(f"stranded UE-slots (sched budget<eps, env backlog>=eps): "
          f"{nstr:.0f} = {nstr/slots:.2f}/slot, "
          f"{nstr/max(touched,1):.2%} of touches")
    print(f"  gone same slot   : {s('stranded_gone_same_slot'):.0f} "
          f"(missed same slot: {s('stranded_missed_same_slot'):.0f})")
    print(f"  re-servable next : {s('stranded_reservable_next'):.0f} "
          f"= {s('stranded_reservable_next')/max(nstr,1):.2%}")
    print(f"  re-SELECTED next : {s('stranded_selected_next'):.0f} "
          f"= {s('stranded_selected_next')/max(nstr,1):.2%}")
    so = {k: int(np.sum([r["B_la1"]["stranded_outcome"][k] for r in results]))
          for k in ("completed", "missed", "retx_drop", "open_end")}
    tot = max(sum(so.values()), 1)
    print(f"stranded-packet outcomes: {so} "
          f"(missed frac {so['missed']/tot:.2%})")
    print(f"epsilon-skip units: {s('n_eps_skip'):.0f}")
    print(f"\ntotal wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
