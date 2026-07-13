"""Part A probe: legacy + decode_order='rbg_major' -- scheduler RBG-major
budget plan vs env layer-major unit creation, two-RBG counterexample."""
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from phy import predict_b_tx


def build_env(decode_order):
    cfg = phase4_queue_config(la_mode="legacy", decode_order=decode_order,
                              seed=2024)
    env = SchedulerEnv(cfg)
    env.reset(episode_idx=0)
    empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    # advance with empty allocations until two UEs have packets (no units ->
    # no retx -> fixed_mask stays empty)
    for _ in range(20):
        ues = [u for u, p in enumerate(env.traffic.packets)
               if p is not None and p.uncommitted_backlog > 0]
        if len(ues) >= 2:
            break
        env.step(empty)
    assert not env.fixed_mask.any(), "fixed positions present, probe invalid"
    return cfg, env, ues


def run(decode_order):
    cfg, env, ues = build_env(decode_order)
    u, v = ues[0], ues[1]
    cap_r0 = float(predict_b_tx(env.csi.cqi_fb[u, 0], cfg))
    cap_r1 = float(predict_b_tx(env.csi.cqi_fb[u, 1], cfg))
    pkt = env.traffic.packets[u]
    # force uncommitted backlog B = cap_r0 + 0.5*cap_r1  (each cap < B < sum)
    B = cap_r0 + 0.5 * cap_r1
    pkt.size = pkt.committed_bits + B
    assert B > cap_r0 and B > cap_r1 and B < cap_r0 + cap_r1

    # ---- scheduler-side plan: EXACT baselines.py legacy rbg-major debit
    # (baselines.py:93-110, per-pick estimate_btx, r-outer / l-inner) ----
    budget = np.array([p.uncommitted_backlog if p is not None else 0.0
                       for p in env.traffic.packets])
    alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    alloc[0, 0] = v + 1
    alloc[0, 1] = u + 1
    alloc[1, 0] = u + 1
    plan = {}
    for r in range(cfg.num_rbg):          # RBG-major traversal
        for l in range(cfg.l_max):
            k = alloc[r, l]
            if k == 0:
                break                      # closure
            pick = k - 1
            d = env.estimate_btx(pick, r, budget[pick])
            budget[pick] -= d
            if pick == u:
                plan[(r, l)] = d

    # ---- env-side actual creation (record create_unit calls) ----
    created = []
    orig = env.txmgr.create_unit
    def rec(p, r, l, b):
        created.append((p.ue_id, r, l, round(b, 3)))
        return orig(p, r, l, b)
    env.txmgr.create_unit = rec
    env.step(alloc)
    env.txmgr.create_unit = orig

    actual = {(r, l): b for (uid, r, l, b) in created if uid == u}
    print(f"decode_order={decode_order!r}  UE u={u}, v={v}")
    print(f"  cap(u,r0)={cap_r0:.1f}  cap(u,r1)={cap_r1:.1f}  backlog B={B:.1f}")
    print(f"  scheduler RBG-major plan for u : {plan}")
    print(f"  env actually created for u     : {actual}")
    print(f"  plan sum={sum(plan.values()):.3f}  actual sum={sum(actual.values()):.3f}")
    print(f"  creation order of ALL create_unit calls: {created}")
    return created


c_rm = run("rbg_major")
print()
c_lm = run("layer_major")
print()
print("env creation identical under both decode_order values:",
      c_rm == c_lm)
