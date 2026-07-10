"""Stronger reference line: greedy-marginal GENIE (NOT a true optimum).

Runs at p_csi=1.0 (perfect CSI -> perfect precoding + perfect decision info).
Per RBG, layer-major, it GREEDILY adds the UE that maximizes the RBG's total
deadline-weighted delivered useful-MI under TRUE-channel RZF, stopping when no
UE improves it (standard near-optimal MU-MIMO user selection). This directly
optimizes the slot reward's dominant term, far beyond CQI-greedy/SUS+PF's fixed
rules. Backlog is threaded across RBGs. Caveat: still MYOPIC (per-slot) and
greedy (not exhaustive), so it is a STRONG REFERENCE, not a true ceiling.

Compares PPO(best.pt, stale) vs the genie vs the fresh-CSI heuristics, same 10
seeds. READ-ONLY; CPU; does not touch the live GPU training.
"""
import numpy as np, torch
from config import phase2_hard_main_config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from phy import _rbg_sinr, mi_bits, predict_b_tx

SEEDS = list(range(10000, 10010))


class GenieScheduler:
    name = "Genie (greedy-marginal, true CSI)"

    def schedule(self, env):
        cfg = env.cfg
        R, L, K = cfg.num_rbg, cfg.l_max, cfg.num_ue
        h = env.h_true_slot                         # [K, R, ant] TRUE channel
        nv, prbg = env.noise_var, cfg.p_rbg
        obs = env.get_observation()
        occupied = obs["occupied"]
        alloc = occupied.copy()
        # remaining committable backlog per UE (threaded across RBGs)
        budget = np.array([p.uncommitted_backlog if p is not None else 0.0
                           for p in env.traffic.packets], dtype=np.float64)
        deadline = np.array([p.deadline if p is not None else 0
                             for p in env.traffic.packets], dtype=np.float64)
        weight = 1.0 + cfg.eta_d / (deadline + 1.0)
        cqi_true = env.csi.cqi_fb                    # p_csi=1.0 -> true CQI

        def rbg_objective(r, ues):
            """deadline-weighted delivered useful-MI for a UE-set in RBG r."""
            if not ues:
                return 0.0, {}
            idx = np.array(ues)
            sinr = _rbg_sinr(h[idx, r, :], h[idx, r, :], nv, prbg, nv)  # perfect precoding
            mi = mi_bits(sinr, cfg)                                     # [len(ues)]
            deliv = {}
            tot = 0.0
            for j, u in enumerate(ues):
                cap = min(budget[u], float(predict_b_tx(cqi_true[u, r], cfg)))
                d = min(float(mi[j]), cap)
                deliv[u] = d
                tot += weight[u] * d
            return tot, deliv

        for r in range(R):
            sel = [int(x) - 1 for x in occupied[r] if x > 0]   # fixed (retx) UEs
            base_obj, _ = rbg_objective(r, sel)
            free_layers = [l for l in range(L) if occupied[r, l] == 0]
            for l in free_layers:
                best_u, best_obj = None, base_obj
                for u in range(K):
                    if u in sel:
                        continue
                    if budget[u] <= 0:
                        continue
                    cap = min(budget[u], float(predict_b_tx(cqi_true[u, r], cfg)))
                    if cap < cfg.b_tx_epsilon:
                        continue
                    obj, _ = rbg_objective(r, sel + [u])
                    if obj > best_obj + 1e-9:
                        best_obj, best_u = obj, u
                if best_u is None:
                    break                                # adding nobody helps -> close RBG
                sel.append(best_u)
                alloc[r, l] = best_u + 1
                base_obj = best_obj
            # thread backlog: subtract what the chosen set delivers in this RBG
            _, deliv = rbg_objective(r, sel)
            for u, d in deliv.items():
                budget[u] = max(0.0, budget[u] - d)
        return alloc


def run_episode(env, alloc_fn, seed):
    env.reset(episode_idx=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(alloc_fn())
    e = env.ep
    arr = max(e["n_arrivals"], 1)
    return dict(reward=e["reward"], comp=e["n_comp"] / arr,
                drop=(e["n_miss_deadline"] + e["n_retx_drop"]) / arr,
                thr=e["acked_bits"] / (env.cfg.episode_len * env.cfg.slot_duration) / 1e6)


def evalu(env, factory, seeds):
    recs = [run_episode(env, factory(env), s) for s in seeds]
    out = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
    out["_rews"] = [r["reward"] for r in recs]
    return out


# genie at perfect CSI
cfg_f = phase2_hard_main_config(p_csi=1.0)
env_f = SchedulerEnv(cfg_f)
g = GenieScheduler()
genie = evalu(env_f, lambda e: (lambda: g.schedule(e)), SEEDS)

print("GENIE (greedy-marginal, true CSI, p_csi=1.0):")
print("  reward %.1f  comp %.3f  drop %.3f  thr %.2f" %
      (genie["reward"], genie["comp"], genie["drop"], genie["thr"]))
print("  per-seed:", [round(x) for x in genie["_rews"]])

# reference numbers from the previous 10-seed confirm run:
PPO = 6347.6
FRESH_BEST = 5251.1      # CQI-greedy fresh
STALE_BEST = 4203.8      # CQI-greedy stale
print("\n=== reference ladder (10 seeds) ===")
print("  best STALE heuristic     : %.1f" % STALE_BEST)
print("  best FRESH-CSI heuristic : %.1f" % FRESH_BEST)
print("  GENIE (this run)         : %.1f" % genie["reward"])
print("  PPO (learned, stale)     : %.1f" % PPO)
print("  PPO vs genie             : %+.1f%%  (negative = PPO below genie = genie is a valid stronger ref)"
      % ((PPO - genie["reward"]) / genie["reward"] * 100))
