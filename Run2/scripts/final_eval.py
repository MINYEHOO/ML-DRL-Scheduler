"""Canonical final evaluation of Run2_HardMain for the post-training analysis.
12 held-out seeds. PPO best.pt(479) + latest(1999), 5 baselines (stale),
greedy-marginal genie (perfect CSI). FULL metric breakdown, separated:
reward, throughput, completion, deadline_miss, retx_drop, overflow, SINR, Jain.
READ-ONLY; CPU; training already finished."""
import json, numpy as np, torch
from config import phase2_hard_main_config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from metrics import jains_index
from phy import _rbg_sinr, mi_bits, predict_b_tx

SEEDS = list(range(10000, 10012))   # 12 held-out seeds


class GenieScheduler:
    name = "Genie(true-CSI,greedy-marg)"
    def schedule(self, env):
        cfg = env.cfg; R, L, K = cfg.num_rbg, cfg.l_max, cfg.num_ue
        h = env.h_true_slot; nv, prbg = env.noise_var, cfg.p_rbg
        occ = env.get_observation()["occupied"]; alloc = occ.copy()
        budget = np.array([p.uncommitted_backlog if p is not None else 0.0
                           for p in env.traffic.packets], dtype=np.float64)
        dl = np.array([p.deadline if p is not None else 0 for p in env.traffic.packets], dtype=np.float64)
        w = 1.0 + cfg.eta_d / (dl + 1.0); cqi = env.csi.cqi_fb
        def obj(r, ues):
            if not ues: return 0.0, {}
            idx = np.array(ues)
            sinr = _rbg_sinr(h[idx, r, :], h[idx, r, :], nv, prbg, nv)
            mi = mi_bits(sinr, cfg); deliv = {}; tot = 0.0
            for j, u in enumerate(ues):
                cap = min(budget[u], float(predict_b_tx(cqi[u, r], cfg)))
                d = min(float(mi[j]), cap); deliv[u] = d; tot += w[u] * d
            return tot, deliv
        for r in range(R):
            sel = [int(x) - 1 for x in occ[r] if x > 0]
            base, _ = obj(r, sel)
            for l in [l for l in range(L) if occ[r, l] == 0]:
                bu, bo = None, base
                for u in range(K):
                    if u in sel or budget[u] <= 0: continue
                    if min(budget[u], float(predict_b_tx(cqi[u, r], cfg))) < cfg.b_tx_epsilon: continue
                    o, _ = obj(r, sel + [u])
                    if o > bo + 1e-9: bo, bu = o, u
                if bu is None: break
                sel.append(bu); alloc[r, l] = bu + 1; base = bo
            _, dv = obj(r, sel)
            for u, d in dv.items(): budget[u] = max(0.0, budget[u] - d)
        return alloc


def full_metrics(env, alloc_fn, seeds):
    rows = []
    for s in seeds:
        env.reset(episode_idx=s); done = False
        while not done:
            _, _, done, _ = env.step(alloc_fn())
        e = env.ep; arr = max(e["n_arrivals"], 1)
        et = env.cfg.episode_len * env.cfg.slot_duration
        sinr_db = (10*np.log10(e["sinr_sum"]/e["sinr_count"]) if e["sinr_count"] else float("nan"))
        rows.append(dict(reward=e["reward"], thr=e["acked_bits"]/et/1e6,
                         comp=e["n_comp"]/arr, miss_dl=e["n_miss_deadline"]/arr,
                         retx_drop=e["n_retx_drop"]/arr,
                         overflow=e.get("n_retx_overflow_drop",0)/arr,
                         sinr_db=sinr_db, jain=jains_index(env.cum_acked_bits)))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["reward_std"] = float(np.std([r["reward"] for r in rows]))
    agg["_rews"] = [round(r["reward"]) for r in rows]
    return agg


cfg = phase2_hard_main_config()
env = SchedulerEnv(cfg)
results = {}

for tag, upd in [("PPO-best(479)", "best"), ("PPO-final(1999)", "latest")]:
    ac = ActorCritic(cfg)
    ac.load_state_dict(torch.load("runs/Run2_HardMain/ckpt/%s.pt" % upd, map_location="cpu")["model"])
    ac.eval()
    results[tag] = full_metrics(env, (lambda a: (lambda: a.decode(env.get_observation(), deterministic=True)["action_sequence"]))(ac), SEEDS)

for sch in all_baselines(cfg):
    results["%s(stale)" % sch.name] = full_metrics(env, (lambda s: (lambda: s.schedule(env)))(sch), SEEDS)

cfg_f = phase2_hard_main_config(p_csi=1.0)
env_f = SchedulerEnv(cfg_f)
g = GenieScheduler()
results["Genie(true-CSI)"] = full_metrics(env_f, lambda: g.schedule(env_f), SEEDS)

print("%-26s %8s %6s %6s %7s %8s %7s %6s" %
      ("scheduler", "reward", "thr", "comp", "miss_dl", "retx_dr", "SINRdB", "jain"))
print("-" * 80)
for k, v in results.items():
    print("%-26s %8.1f %6.1f %6.3f %7.3f %8.4f %7.2f %6.3f" %
          (k, v["reward"], v["thr"], v["comp"], v["miss_dl"], v["retx_drop"], v["sinr_db"], v["jain"]))

print("\n=== JSON ===")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "_rews"} | {"per_seed": v["_rews"]}
                  for k, v in results.items()}, indent=1))
