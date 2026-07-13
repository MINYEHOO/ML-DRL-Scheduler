"""Gates 2 & 3 + retx-B_tx immutability + Gate-1 depth-1 supplement.

Gate 2: scheduler's in-decode planned commits vs env's actual commits,
        per (rbg, ue), across baselines AND a PPO net, both CSI worlds.
Gate 3: PPO stochastic rollout vs teacher-forced replay (same weights):
        action/mask identity (asserted inside replay), max |dlogp|,
        budget-trace and planned-B_tx equality.
"""
import numpy as np, torch
torch.manual_seed(0)
from config import phase4_queue_config
from env import SchedulerEnv
from baselines import (SUSCQI, SUSPF, SUSMaxWeight, SUCQI, CQIGreedy,
                       Random, SUSRandom)
from policy import ActorCritic

EPL = 200  # slots per episode checked (plenty of units)


BETA_M = (0.979, 0.723, 0.646, 0.590)   # official depth-wise backoff


def make_cfg(pmi, pcsi):
    # imperfect worlds run the OFFICIAL beta_m mode; genie keeps beta=1
    bm = () if pmi == "genie" else BETA_M
    return phase4_queue_config(pmi_mode=pmi, p_csi=pcsi,
                               la_mode="post_rzf", decode_order="rbg_major",
                               la_beta_by_depth=bm,
                               p_arrival_min=0.15, p_arrival_max=0.40,
                               debug=True)


# ---------------- Gate 2 + retx immutability -----------------------------
g2_max = 0.0
g2_units = 0
g2_slots = 0
g2_keyset_mismatch = 0
retx_btx_changes = 0
retx_pin_moves = 0
unit_first = {}          # uid -> (b_tx, rbg, layer)

worlds = [("type2_sparse_56bit", 0.6), ("genie", 1.0)]
scheds = [SUSCQI(), SUSPF(), SUSMaxWeight(), SUCQI(), CQIGreedy(),
          Random(seed=1), SUSRandom(seed=2)]

for pmi, pcsi in worlds:
    cfg = make_cfg(pmi, pcsi)
    env = SchedulerEnv(cfg)
    for si, sch in enumerate(scheds):
        env.reset(episode_idx=20000 + si)
        unit_first.clear()
        for t in range(EPL):
            alloc = sch.schedule(env)
            planned = sch.last_planner.planned_btx_map()
            env.step(alloc)
            actual = env.last_actual_btx
            g2_slots += 1
            if set(planned) != set(actual):
                g2_keyset_mismatch += 1
            for k in set(planned) | set(actual):
                g2_units += 1
                d = abs(planned.get(k, 0.0) - actual.get(k, 0.0))
                g2_max = max(g2_max, d)
            # retx immutability: B_tx fixed at creation; RBG pin never moves
            # (layer may shift on compaction -- allowed by spec)
            for uid, un in env.txmgr.units_by_id.items():
                cur = (un.b_tx, un.rbg_id)
                if uid in unit_first:
                    if unit_first[uid][0] != cur[0]:
                        retx_btx_changes += 1
                    if unit_first[uid][1] != cur[1]:
                        retx_pin_moves += 1
                else:
                    unit_first[uid] = cur

print(f"[Gate2/baselines] slots={g2_slots} units={g2_units} "
      f"max|planned-actual|={g2_max:.3e} keyset_mismatch={g2_keyset_mismatch}")
print(f"[retx] B_tx changes={retx_btx_changes}  pin moves={retx_pin_moves}")

# ---- Gate 2 with PPO net (random init = arbitrary policy) ---------------
cfg = make_cfg("type2_sparse_56bit", 0.6)
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ac.eval()
p2_max, p2_units, p2_mismatch = 0.0, 0, 0
for ep in range(2):
    obs = env.reset(episode_idx=30000 + ep)
    for t in range(EPL):
        out = ac.decode(obs, deterministic=False)
        obs, _, done, _ = env.step(out["action_sequence"])
        actual = env.last_actual_btx
        planned = out["planned_btx"]
        if set(planned) != set(actual):
            p2_mismatch += 1
        for k in set(planned) | set(actual):
            p2_units += 1
            p2_max = max(p2_max, abs(planned.get(k, 0.0) - actual.get(k, 0.0)))
        if done:
            break
print(f"[Gate2/PPO] units={p2_units} max|planned-actual|={p2_max:.3e} "
      f"keyset_mismatch={p2_mismatch}")

# ---------------- Gate 3: rollout vs teacher-forced replay ---------------
g3_logp_max, g3_pos_max, g3_budget_max, g3_btx_max = 0.0, 0.0, 0.0, 0.0
g3_val_max, g3_decisions = 0.0, 0
env = SchedulerEnv(make_cfg("type2_sparse_56bit", 0.6))
for ep in range(2):
    obs = env.reset(episode_idx=31000 + ep)
    for t in range(EPL):
        out = ac.decode(obs, deterministic=False)      # stochastic rollout
        with torch.no_grad():
            rep = ac._rbg_major_pass(obs,
                                     stored_actions=out["action_sequence"],
                                     stored_mask=out["policy_decision_mask"])
        assert (rep["action_sequence"] == out["action_sequence"]).all()
        assert (rep["policy_decision_mask"]
                == out["policy_decision_mask"]).all()
        g3_logp_max = max(g3_logp_max,
                          abs(out["log_prob_sum"]
                              - float(rep["log_prob_sum"].item())))
        for a, b in zip(out["per_subaction_logprobs"],
                        rep["per_subaction_logprobs"]):
            g3_pos_max = max(g3_pos_max, abs(a - b))
        g3_budget_max = max(g3_budget_max,
                            float(np.abs(out["budget_trace"]
                                         - rep["budget_trace"]).max()))
        ka, kb = out["planned_btx"], rep["planned_btx"]
        assert set(ka) == set(kb)
        for k in ka:
            g3_btx_max = max(g3_btx_max, abs(ka[k] - kb[k]))
        g3_val_max = max(g3_val_max,
                         abs(out["value"] - float(rep["value"].item())))
        g3_decisions += out["num_policy_decisions"]
        obs, _, done, _ = env.step(out["action_sequence"])
        if done:
            break
print(f"[Gate3] decisions={g3_decisions} max|dlogp_sum|={g3_logp_max:.3e} "
      f"max|dlogp_pos|={g3_pos_max:.3e} max|dV|={g3_val_max:.3e}")
print(f"[Gate3] max|d budget_trace|={g3_budget_max:.3e} "
      f"max|d planned_btx|={g3_btx_max:.3e}")

# ---------------- Gate 1 supplement: depth-1 (SU) in genie ----------------
cfg = make_cfg("genie", 1.0)
env = SchedulerEnv(cfg)
sch = SUCQI()
d1 = [0, 0]
env.reset(episode_idx=40000)
for t in range(EPL):
    alloc = sch.schedule(env)
    env.step(alloc)
    for (r, u), b in env.last_actual_btx.items():
        alive = [x for x in env.txmgr.units
                 if x.rbg_id == r and x.ue_id == u and x.tx_attempts == 1]
        d1[1] += 1
        d1[0] += (len(alive) == 0)
print(f"[Gate1 depth-1] genie SU first-ACK {d1[0]}/{d1[1]} "
      f"= {d1[0]/max(d1[1],1)*100:.2f}%")

assert g2_max == 0.0 and g2_keyset_mismatch == 0, "GATE 2 (baselines) FAIL"
assert p2_max == 0.0 and p2_mismatch == 0, "GATE 2 (PPO) FAIL"
assert retx_btx_changes == 0 and retx_pin_moves == 0, "RETX IMMUTABILITY FAIL"
assert g3_logp_max < 1e-6 and g3_budget_max == 0.0 and g3_btx_max == 0.0, \
    "GATE 3 FAIL"
assert d1[0] == d1[1], "GATE 1 depth-1 FAIL"
print("\nALL GATES PASSED")
