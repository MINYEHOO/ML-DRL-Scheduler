"""Probes for GPT-audit cluster ppo-mdp (E1, E2, E4, E5, E7).

Read-only on the repo; CPU only; tiny configs (episode_len <= 20).
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")

import numpy as np

# ===========================================================================
# E5: global-phase invariance of direction features (numpy only, no env)
# ===========================================================================
print("=" * 70)
print("E5: global-phase invariance of direction_fb features")
print("=" * 70)
from config import Config
from codebook import Type2SparseCodebook, RandomUnitNormCodebook, GenieCodebook
from phy import _rbg_sinr

rng = np.random.default_rng(7)
h = (rng.standard_normal((6, 32)) + 1j * rng.standard_normal((6, 32)))
theta = 0.7
h_rot = h * np.exp(1j * theta)

cfg = Config()
for name, cb in [("type2_sparse_56bit", Type2SparseCodebook(cfg)),
                 ("random_unit_norm", RandomUnitNormCodebook(256, 32, 0)),
                 ("genie", GenieCodebook(32))]:
    _, d1 = cb.quantize(h)
    _, d2 = cb.quantize(h_rot)
    feat_diff = np.abs(np.concatenate([d1.real - d2.real,
                                       d1.imag - d2.imag], -1)).max()
    # alignment up to a global phase: |<d1,d2>| per row
    align = np.abs(np.einsum("bt,bt->b", np.conj(d1), d2))
    print(f"  {name:22s}: max |Re/Im feature diff| = {feat_diff:.4f}   "
          f"|<d1,d2>| (phase-invariant corr) min/mean = "
          f"{align.min():.4f}/{align.mean():.4f}")

# environment outcome invariance: rotate one user's h_hat row by a phase ->
# SINR unchanged (so the phase is a pure nuisance dim for the encoder)
hh = (rng.standard_normal((3, 32)) + 1j * rng.standard_normal((3, 32)))
ht = (rng.standard_normal((3, 32)) + 1j * rng.standard_normal((3, 32)))
s0 = _rbg_sinr(ht, hh, alpha=0.1, p_rbg=1.0, noise_var=0.1)
hh2 = hh.copy(); hh2[1] *= np.exp(1j * 1.3)
s1 = _rbg_sinr(ht, hh2, alpha=0.1, p_rbg=1.0, noise_var=0.1)
print(f"  SINR invariance to per-user h_hat global phase: "
      f"max diff = {np.abs(s0 - s1).max():.2e}")

# ===========================================================================
# E7: dead flags -- toggling changes nothing at runtime
# ===========================================================================
print("=" * 70)
print("E7: share_critic_encoder / rzf_alpha_mode dead-flag check")
print("=" * 70)
import torch
from policy import ActorCritic
cfgA = Config(debug=True, num_ue=4, share_critic_encoder=True,
              rzf_alpha_mode="noise")
cfgB = Config(debug=True, num_ue=4, share_critic_encoder=False,
              rzf_alpha_mode="THIS_IS_NOT_A_MODE")
torch.manual_seed(0); acA = ActorCritic(cfgA)
torch.manual_seed(0); acB = ActorCritic(cfgB)
same = all(torch.equal(a, b) for (na, a), (nb, b)
           in zip(acA.state_dict().items(), acB.state_dict().items()))
print(f"  ActorCritic state_dicts identical with flags flipped/garbage: {same}")
print(f"  (env.py:230 hardcodes alpha=self.noise_var; no code reads either flag)")

# ===========================================================================
# env-based probes (E1, E2, E4) -- tiny episode
# ===========================================================================
print("=" * 70)
print("env-based probes (Sionna channel, CPU, episode_len=20, K=8)")
print("=" * 70)
from config import debug_config
from env import SchedulerEnv

def make_env():
    cfg = debug_config(num_ue=8, episode_len_debug=20, p_arrival=0.5)
    return cfg, SchedulerEnv(cfg)

cfg, env = make_env()

def obs_equal(o1, o2):
    diffs = []
    for k in o1:
        v1, v2 = o1[k], o2[k]
        if k == "initial_S_r":
            if [sorted(s) for s in v1] != [sorted(s) for s in v2]:
                diffs.append(k)
        elif isinstance(v1, np.ndarray):
            if not np.array_equal(v1, v2):
                diffs.append(k)
        else:
            if v1 != v2:
                diffs.append(k)
    return diffs

# --- E1: same obs, different in-flight i_acc -> different outcome ----------
def run_pass(i_acc_value):
    """Deterministic pass: reset, create a pending unit on the first active
    UE, set its i_acc, advance one slot boundary, return (obs, reward, info)."""
    obs = env.reset(0)
    # advance with empty allocations until some UE is active
    while not obs["active"].any():
        obs, r, d, i = env.step(np.zeros((cfg.num_rbg, cfg.l_max), np.int64))
        assert not d
    u = int(np.where(obs["active"])[0][0])
    pkt = env.traffic.packets[u]
    b_tx = float(pkt.uncommitted_backlog)
    unit = env.txmgr.create_unit(pkt, rbg_id=0, layer_id=0, b_tx=b_tx)
    unit.i_acc = i_acc_value * b_tx          # <-- hidden state manipulated
    # next slot boundary: compaction picks the pending unit into fixed grid
    env.slot += 1
    env._prepare_slot(env.slot)
    o = env.get_observation()
    reward, info = env._finish_slot(np.zeros((cfg.num_rbg, cfg.l_max),
                                             np.int64))
    return o, reward, info, b_tx

oA, rA, iA, btx = run_pass(0.0)     # fresh unit
oB, rB, iB, _ = run_pass(0.99)      # unit 99% delivered (hidden)
d = obs_equal(oA, oB)
print(f"E1: obs fields differing between i_acc=0 and i_acc=0.99*b_tx: {d}")
print(f"    b_tx={btx:.0f}  reward A={rA:.4f} vs B={rB:.4f}   "
      f"n_comp A={iA['n_comp']} B={iB['n_comp']}   "
      f"acked A={iA['acked_bits']:.0f} B={iB['acked_bits']:.0f}")
assert d == [], "obs should be IDENTICAL (aliased states)"
print(f"    -> identical obs, different outcome: {abs(rA-rB) > 1e-9}")

# also attempt-count aliasing
def run_pass_att(attempts):
    obs = env.reset(0)
    while not obs["active"].any():
        obs, r, d_, i = env.step(np.zeros((cfg.num_rbg, cfg.l_max), np.int64))
    u = int(np.where(obs["active"])[0][0])
    pkt = env.traffic.packets[u]
    unit = env.txmgr.create_unit(pkt, 0, 0, float(pkt.uncommitted_backlog))
    unit.tx_attempts = attempts
    unit.b_tx = 1e9                     # force NACK path
    unit.i_acc = 0.0
    env.slot += 1
    env._prepare_slot(env.slot)
    o = env.get_observation()
    reward, info = env._finish_slot(np.zeros((cfg.num_rbg, cfg.l_max),
                                             np.int64))
    return o, reward, info

oC, rC, iC = run_pass_att(0)                       # 1st attempt
oD, rD, iD = run_pass_att(cfg.num_attempts - 1)    # last attempt -> drop
dd = obs_equal(oC, oD)
print(f"E1b: obs diffs for tx_attempts 0 vs {cfg.num_attempts-1}: {dd}   "
      f"retx_drop C={iC['n_retx_drop']} D={iD['n_retx_drop']}  "
      f"reward C={rC:.3f} D={rD:.3f}")

# --- E2: terminal obs vs properly prepared next-slot obs -------------------
obs = env.reset(0)
done = False
while not done:
    obs, r, done, info = env.step(np.zeros((cfg.num_rbg, cfg.l_max), np.int64))
obs_done = obs                               # what state_value() bootstraps on
age_done = obs_done["age"].copy()
env._prepare_slot(env.slot)                  # what a REAL slot-T obs would be
obs_prep = env.get_observation()
d2 = obs_equal(obs_done, obs_prep)
print(f"E2: fields differing (terminal-bootstrap obs vs prepared s_T): {d2}")
print(f"    age at done  : {age_done}")
print(f"    age prepared : {obs_prep['age']}")
print(f"    active at done/prepared: {int(obs_done['active'].sum())} / "
      f"{int(obs_prep['active'].sum())}")

# --- E4: no-user masked out at first layer when a candidate exists ---------
from policy import obs_to_tensors
torch.manual_seed(1)
ac = ActorCritic(cfg)
obs = env.reset(0)
while not obs["active"].any():
    obs, r, d_, i = env.step(np.zeros((cfg.num_rbg, cfg.l_max), np.int64))
obs_t = obs_to_tensors(obs, torch.device("cpu"))
e = ac.encode(obs_t)
tu = torch.from_numpy(obs["uncommitted"].astype(np.float32)).clone()
logits, valid_mask, _ = ac._position_logits_and_mask(
    obs_t, e, r=0, l=0, S_r_r=set(), in_slot_count=np.zeros(cfg.num_ue,
    np.int64), temp_uncommit=tu)
print(f"E4: at (r=0,l=0), S_r empty, {int(valid_mask[1:].sum())} valid UE "
      f"candidate(s) -> no-user valid = {bool(valid_mask[0])}")
assert not bool(valid_mask[0])
# commit-lock demonstration: single active UE, forced at RBG 0 even if its
# CQI at RBG 0 is the WORST of its 8 RBGs
u = int(np.where(obs["active"])[0][0])
cqi_u = obs["cqi_fb"][u]
out = ac.decode(obs, deterministic=True)
sched_rbgs = np.where(out["action_sequence"][:, 0] == u + 1)[0]
print(f"    single-UE case: UE {u} CQI per RBG = "
      f"{np.round(cqi_u, 2)} (argmin={int(cqi_u.argmin())}, "
      f"argmax={int(cqi_u.argmax())})")
print(f"    deterministic decode schedules UE at RBGs {sched_rbgs.tolist()} "
      f"(layer 0); uncommitted={obs['uncommitted'][u]:.0f}, "
      f"pred_btx@RBG0={float(min(obs['uncommitted'][u], cfg.eta_data*cfg.n_re_rbg*cfg.beta_rate*cqi_u[0])):.0f}")
print()
print("ALL PROBES DONE")
