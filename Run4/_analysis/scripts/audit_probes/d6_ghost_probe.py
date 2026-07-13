"""D6 probe: m_planned pre-pass 'ghost streams' under cfg.mu_aware_la=True.

Question: the pre-pass counts a UE on every RBG it is admissibly allocated to,
using slot-START backlog; it does NOT simulate within-slot backlog commit, so a
UE whose backlog is exhausted by an earlier RBG's unit is still counted on later
RBGs -> m_planned[r] > realized stream count -> over-de-rated B_tx for the
OTHER (real) units on that RBG. Quantify frequency and bit impact at the LA
ablation operating point (K=32 mixed hetero point, la_ablation.py config),
episode_len=300, CPU only.

Also (D2 supplementary): record completed vs failed packet SIZES to check the
size-bias of completion_rate vs bit-weighted goodput.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np
import torch
torch.set_num_threads(8)

from config import phase2_hetero_config
from env import SchedulerEnv
from phy import predict_b_tx
from policy import ActorCritic
from baselines import SUSCQI, SUCQI, CQIGreedy
from train_phase2 import PPOScheduler


class ProbeEnv(SchedulerEnv):
    """Behaviorally identical to SchedulerEnv (creation loop copied verbatim
    from env.py:352-441) with recording added; asserts realized allocation
    matches what the parent would produce is implicit (same code)."""

    def start_stats(self):
        self.S = dict(slots=0, rbg_new=0, rbg_derated=0, rbg_ghost=0,
                      overcount_sum=0, ghost_skips=0, eps_skips=0,
                      units=0, units_affected=0,
                      bits_committed=0.0, bits_shortfall=0.0,
                      undercount_violations=0)
        self.completed_sizes = []
        self.failed_sizes = []

    # --- D2: classify removed packets ---
    def _remove_packet(self, ue, packet_id):
        pkt = self.traffic.packets[ue]
        if pkt is not None and pkt.packet_id == packet_id:
            if pkt.is_complete:
                self.completed_sizes.append(pkt.size)
            else:
                self.failed_sizes.append(pkt.size)
        super()._remove_packet(ue, packet_id)

    # --- D6: instrumented verbatim copy ---
    def _sanitize_and_create(self, allocation):
        cfg = self.cfg
        realized = self.fixed_allocation.copy()
        sel_ue = [s.copy() for s in self.initial_S_r]
        rbg_closed = np.zeros(cfg.num_rbg, dtype=bool)

        m_planned = None
        counted_new = [set() for _ in range(cfg.num_rbg)]
        if cfg.mu_aware_la:
            m_planned = self.fixed_mask.sum(axis=1).astype(int)   # [R]
            _seen = [set(s) for s in self.initial_S_r]
            _closed = np.zeros(cfg.num_rbg, dtype=bool)
            for l in range(cfg.l_max):
                for r in range(cfg.num_rbg):
                    if self.fixed_mask[r, l] or _closed[r]:
                        continue
                    k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
                    if k == 0:
                        _closed[r] = True
                        continue
                    u = k - 1
                    if not (0 <= u < cfg.num_ue) or u in _seen[r]:
                        continue
                    pkt = self.traffic.packets[u]
                    if pkt is None or pkt.uncommitted_backlog <= 0:
                        continue
                    m_planned[r] += 1
                    _seen[r].add(u)
                    counted_new[r].add(u)                          # [probe]

        created = [[] for _ in range(cfg.num_rbg)]                 # [probe]
        ghost_skips = eps_skips = 0                                # [probe]

        for l in range(cfg.l_max):
            for r in range(cfg.num_rbg):
                if self.fixed_mask[r, l]:
                    continue
                if rbg_closed[r]:
                    continue
                k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
                if k == 0:
                    rbg_closed[r] = True
                    continue
                u = k - 1
                if not (0 <= u < cfg.num_ue):
                    continue
                if u in sel_ue[r]:
                    continue
                pkt = self.traffic.packets[u]
                if pkt is None or pkt.uncommitted_backlog <= 0:
                    if u in counted_new[r]:                        # [probe]
                        ghost_skips += 1                           # [probe]
                    continue
                if m_planned is not None and m_planned[r] > 1:
                    snr_su = 2.0 ** float(self.csi.cqi_fb[u, r]) - 1.0
                    se_m = np.log2(1.0 + snr_su / float(m_planned[r]))
                    b_tx_cap = cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate * se_m
                else:
                    b_tx_cap = float(predict_b_tx(self.csi.cqi_fb[u, r], cfg))
                backlog_before = pkt.uncommitted_backlog            # [probe]
                b_tx = min(pkt.uncommitted_backlog, b_tx_cap)
                if b_tx < cfg.b_tx_epsilon:
                    if u in counted_new[r]:                        # [probe]
                        eps_skips += 1                             # [probe]
                    continue
                if pkt.uncommitted_backlog - b_tx < cfg.b_tx_epsilon:
                    b_tx = pkt.uncommitted_backlog
                self.txmgr.create_unit(pkt, r, l, b_tx)
                created[r].append((u, float(b_tx), float(backlog_before),
                                   float(self.csi.cqi_fb[u, r])))   # [probe]
                realized[r, l] = k
                sel_ue[r].add(u)

        # ---- post-pass stats ----
        S = self.S
        S["slots"] += 1
        S["ghost_skips"] += ghost_skips
        S["eps_skips"] += eps_skips
        realized_m = (realized > 0).sum(axis=1)                    # [R]
        for r in range(cfg.num_rbg):
            if not created[r]:
                continue
            S["rbg_new"] += 1
            mp, mr = int(m_planned[r]), int(realized_m[r])
            if mp >= 2:
                S["rbg_derated"] += 1
            if mp > mr:
                S["rbg_ghost"] += 1
                S["overcount_sum"] += mp - mr
            if mp < mr:
                S["undercount_violations"] += 1
            for (u, b_tx, backlog_before, cqi) in created[r]:
                S["units"] += 1
                S["bits_committed"] += b_tx
                # correct sizing had m been the realized count
                if mr > 1:
                    snr = 2.0 ** cqi - 1.0
                    cap_true = (cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
                                * np.log2(1.0 + snr / float(mr)))
                else:
                    cap_true = float(predict_b_tx(cqi, cfg))
                b_true = min(backlog_before, cap_true)
                if backlog_before - b_true < cfg.b_tx_epsilon:
                    b_true = backlog_before
                short = b_true - b_tx
                if short > 1e-9:
                    S["units_affected"] += 1
                    S["bits_shortfall"] += short
        return realized


def main():
    seeds = [10000, 10001, 10002]
    cfg = phase2_hetero_config(
        num_ue=32, n_active_min=16, n_active_max=32,
        ue_speed_min=5.0, ue_speed_max=30.0, p_csi=0.6,
        ppo_critic_v2=True, ppo_target_kl=0.02, ppo_entropy_coef=0.02,
        episode_len_main=300)
    cfg.mu_aware_la = True
    env = ProbeEnv(cfg)

    ac = ActorCritic(cfg)
    ck = torch.load("/home/MYH/ML_DRL_Scheduler/Run3/MixedSpeed_L2b/ckpt/best.pt",
                    map_location="cpu")
    own = ac.state_dict()
    ac.load_state_dict({k: v for k, v in ck["model"].items()
                        if k in own and own[k].shape == v.shape}, strict=False)
    ac.eval()

    scheds = [("SUS+CQI", SUSCQI()), ("CQI-greedy", CQIGreedy()),
              ("SU+CQI", SUCQI()),
              ("PPO-L2b", PPOScheduler(ac, deterministic=True))]

    agg = {}
    for seed in seeds:
        for name, sch in scheds:
            env.start_stats()
            env.reset(episode_idx=seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            S = env.S
            comp = env.completed_sizes
            fail = env.failed_sizes
            arrivals = max(env.ep["n_arrivals"], 1)
            a = agg.setdefault(name, dict(
                slots=0, rbg_new=0, rbg_derated=0, rbg_ghost=0,
                overcount_sum=0, ghost_skips=0, eps_skips=0, units=0,
                units_affected=0, bits_committed=0.0, bits_shortfall=0.0,
                undercount_violations=0, comp_sizes=[], fail_sizes=[],
                n_comp=0, n_arr=0, comp_bits=0.0))
            for k in S:
                a[k] += S[k]
            a["comp_sizes"] += comp
            a["fail_sizes"] += fail
            a["n_comp"] += len(comp)
            a["n_arr"] += arrivals
            a["comp_bits"] += float(np.sum(comp))
            print(f"seed {seed} {name:10s} slots={S['slots']} "
                  f"rbg_new={S['rbg_new']} derated={S['rbg_derated']} "
                  f"ghost={S['rbg_ghost']} ghost_skips={S['ghost_skips']} "
                  f"eps={S['eps_skips']} units={S['units']} "
                  f"aff={S['units_affected']} "
                  f"short={S['bits_shortfall']:.0f}/"
                  f"{S['bits_committed']:.0f}", flush=True)

    print("\n=== D6 summary (3 seeds x 300 slots, K=32 mixed, mu_aware_la=True) ===")
    for name, a in agg.items():
        gr = a["rbg_ghost"] / max(a["rbg_new"], 1)
        gd = (a["rbg_ghost"] / max(a["rbg_derated"], 1)
              if a["rbg_derated"] else 0.0)
        sh = a["bits_shortfall"] / max(a["bits_committed"], 1.0)
        oc = a["overcount_sum"] / max(a["rbg_ghost"], 1)
        print(f"{name:10s} RBG-slots w/ new units {a['rbg_new']:6d} | "
              f"derated(m_planned>=2) {a['rbg_derated']:6d} | "
              f"ghost(m_pl>m_real) {a['rbg_ghost']:5d} "
              f"({gr*100:5.2f}% of new, {gd*100:5.2f}% of derated) | "
              f"mean overcount {oc:.2f} | ghost_skips {a['ghost_skips']} "
              f"eps_skips {a['eps_skips']} | units aff "
              f"{a['units_affected']}/{a['units']} | bit shortfall "
              f"{a['bits_shortfall']:.0f} ({sh*100:.3f}% of committed) | "
              f"undercnt_viol {a['undercount_violations']}")

    print("\n=== D2 supplementary: packet-size bias of completion ===")
    for name, a in agg.items():
        cs, fs = np.array(a["comp_sizes"]), np.array(a["fail_sizes"])
        cr = a["n_comp"] / max(a["n_arr"], 1)
        print(f"{name:10s} comp_rate={cr:.3f} n_comp={a['n_comp']} "
              f"mean_size(completed)={cs.mean() if cs.size else 0:.0f} "
              f"mean_size(failed)={fs.mean() if fs.size else 0:.0f} "
              f"completed_bits={a['comp_bits']/1e6:.2f} Mb")


if __name__ == "__main__":
    main()
