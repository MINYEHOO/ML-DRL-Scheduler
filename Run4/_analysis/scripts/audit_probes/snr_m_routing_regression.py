"""snr_m routing regression (audit round 8, evidence for the env.py fix).

The bug: env's SNR/m de-rate branch gated on the raw ``mu_aware_la`` flag,
so ``la_mode="snr_m"`` alone silently ran legacy and
``la_mode="legacy", mu_aware_la=True`` silently de-rated. Fixed to gate on
``resolved_la_mode() == "snr_m"`` (commit 9ed28d0).

This test goes past the config API down to ACTUAL unit creation: it drives
four identically-seeded envs with the SAME fixed depth-4 allocations for 60
slots and records every ``create_unit`` call's (rbg, layer, packet, B_tx):

  A = (la_mode="",       mu_aware_la=False)   -- legacy, historical spelling
  C = (la_mode="legacy", mu_aware_la=True)    -- explicit legacy must WIN
  B = (la_mode="",       mu_aware_la=True)    -- snr_m, historical spelling
  D = (la_mode="snr_m",  mu_aware_la=False)   -- snr_m, new spelling

PASS: A == C exactly, B == D exactly, and A != B with at least one
cap-limited m>=2 unit's B_tx strictly SMALLER in B (the de-rate acting).
(A's bit-exactness against the pre-redesign tree was verified separately
with 3-update debug trainings vs the frozen pre-redesign code -- available
in git history as the initial commit 23bc8aa, mirrored locally in
Run3/code_backup/.)

Run from the repo root:
  python3 Run4/_analysis/scripts/audit_probes/snr_m_routing_regression.py
"""
import numpy as np

from config import phase4_queue_config
from env import SchedulerEnv

COMBOS = {
    "A": dict(la_mode="", mu_aware_la=False),
    "C": dict(la_mode="legacy", mu_aware_la=True),
    "B": dict(la_mode="", mu_aware_la=True),
    "D": dict(la_mode="snr_m", mu_aware_la=False),
}
SLOTS = 60


def created_units(combo_kw):
    """Every create_unit call as (slot, rbg, layer, packet_id, ue, B_tx)."""
    cfg = phase4_queue_config(p_arrival_min=0.15, p_arrival_max=0.40,
                              **combo_kw)
    env = SchedulerEnv(cfg)
    env.reset(episode_idx=12345)          # same episode RNG for all combos
    log = []
    orig = env.txmgr.create_unit
    slot_box = [0]

    def hook(pkt, r, l, b_tx):
        log.append((slot_box[0], r, l, pkt.packet_id, pkt.ue_id,
                    round(float(b_tx), 9)))
        return orig(pkt, r, l, b_tx)

    env.txmgr.create_unit = hook
    for t in range(SLOTS):
        slot_box[0] = t
        # deterministic depth-4 allocation: RBG r wants UEs 4r+l+t (mod K);
        # env admission (active/backlog/dup checks) filters identically
        # per combo because traffic/CSI RNG streams are identical.
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        for r in range(cfg.num_rbg):
            for l in range(cfg.l_max):
                alloc[r, l] = ((4 * r + l + t) % cfg.num_ue) + 1
        env.step(alloc)
    return log


if __name__ == "__main__":
    res = {name: created_units(kw) for name, kw in COMBOS.items()}
    for name, kw in COMBOS.items():
        print(f"combo {name} {kw}: {len(res[name])} units")

    assert res["A"] == res["C"], \
        "FAIL: explicit la_mode='legacy' did not override mu_aware_la=True"
    assert res["B"] == res["D"], \
        "FAIL: la_mode='snr_m' alone does not reproduce mu_aware_la=True"
    assert res["A"] != res["B"], \
        "FAIL: snr_m de-rate changed no created unit vs legacy"
    shrunk = sum(1 for x, y in zip(res["A"], res["B"])
                 if x[:5] == y[:5] and y[5] < x[5] - 1e-9)
    print(f"A==C exactly | B==D exactly | A!=B with {shrunk} strictly "
          f"de-rated units")
    assert shrunk > 0, "FAIL: no unit strictly de-rated under snr_m"
    print("snr_m routing regression PASSED")
