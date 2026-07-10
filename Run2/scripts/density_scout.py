"""Mechanism scout: does PPO win by SCHEDULING CONSERVATIVELY (fewer/more-
orthogonal co-users per RBG -> high SINR, near-zero retx failure)?
Measure, per scheduler over 4 seeds: avg occupied positions/slot, avg co-
scheduled UEs per ACTIVE RBG (MU layering depth), and the SINR distribution.
READ-ONLY, CPU."""
import numpy as np, torch
from config import phase2_hard_main_config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import CQIGreedy, SUSPF
from phy import _rbg_sinr, mi_bits, predict_b_tx

SEEDS = list(range(10000, 10004))


class GenieScheduler:
    name = "Genie"
    def schedule(self, env):
        cfg = env.cfg; R, L, K = cfg.num_rbg, cfg.l_max, cfg.num_ue
        h = env.h_true_slot; nv, prbg = env.noise_var, cfg.p_rbg
        occ = env.get_observation()["occupied"]; alloc = occ.copy()
        budget = np.array([p.uncommitted_backlog if p is not None else 0.0 for p in env.traffic.packets])
        dl = np.array([p.deadline if p is not None else 0 for p in env.traffic.packets], float)
        w = 1.0 + cfg.eta_d/(dl+1.0); cqi = env.csi.cqi_fb
        def obj(r, ues):
            if not ues: return 0.0, {}
            idx=np.array(ues); sinr=_rbg_sinr(h[idx,r,:],h[idx,r,:],nv,prbg,nv); mi=mi_bits(sinr,cfg)
            dv={}; tot=0.0
            for j,u in enumerate(ues):
                cap=min(budget[u],float(predict_b_tx(cqi[u,r],cfg))); d=min(float(mi[j]),cap); dv[u]=d; tot+=w[u]*d
            return tot,dv
        for r in range(R):
            sel=[int(x)-1 for x in occ[r] if x>0]; base,_=obj(r,sel)
            for l in [l for l in range(L) if occ[r,l]==0]:
                bu,bo=None,base
                for u in range(K):
                    if u in sel or budget[u]<=0: continue
                    if min(budget[u],float(predict_b_tx(cqi[u,r],cfg)))<cfg.b_tx_epsilon: continue
                    o,_=obj(r,sel+[u])
                    if o>bo+1e-9: bo,bu=o,u
                if bu is None: break
                sel.append(bu); alloc[r,l]=bu+1; base=bo
            _,dv=obj(r,sel)
            for u,d in dv.items(): budget[u]=max(0.0,budget[u]-d)
        return alloc


def density(env, alloc_fn, seeds):
    pos_per_slot, ues_per_active_rbg = [], []
    for s in seeds:
        env.reset(episode_idx=s); done=False
        while not done:
            a = alloc_fn()
            occ = (a > 0)
            pos_per_slot.append(int(occ.sum()))
            for r in range(env.cfg.num_rbg):
                n = int(occ[r].sum())
                if n > 0: ues_per_active_rbg.append(n)
            _,_,done,_ = env.step(a)
    return (float(np.mean(pos_per_slot)), float(np.mean(ues_per_active_rbg)),
            dict(zip(*np.unique(ues_per_active_rbg, return_counts=True))))


cfg = phase2_hard_main_config()
env = SchedulerEnv(cfg)
ac = ActorCritic(cfg)
ac.load_state_dict(torch.load("runs/Run2_HardMain/ckpt/best.pt", map_location="cpu")["model"]); ac.eval()

scheds = [("PPO-best", lambda: ac.decode(env.get_observation(), deterministic=True)["action_sequence"]),
          ("CQI-greedy", lambda: CQIGreedy().schedule(env)),
          ("SUS+PF", lambda: SUSPF().schedule(env))]

print("%-12s | pos/slot(of 32) | UEs per active RBG | layer-depth histogram")
print("-"*78)
for name, fn in scheds:
    pps, upr, hist = density(env, fn, SEEDS)
    print("%-12s | %14.1f | %18.2f | %s" % (name, pps, upr, {int(k): int(v) for k,v in hist.items()}))

env_f = SchedulerEnv(phase2_hard_main_config(p_csi=1.0)); g = GenieScheduler()
pps, upr, hist = density(env_f, lambda: g.schedule(env_f), SEEDS)
print("%-12s | %14.1f | %18.2f | %s" % ("Genie", pps, upr, {int(k): int(v) for k,v in hist.items()}))
print("\n(layer-depth histogram = how many RBGs had 1,2,3,4 co-scheduled UEs)")
