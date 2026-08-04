"""Regression: CQI index 0 (h_hat = 0) reaching RZF via a HARQ retx.

The candidate gate excludes CQI-0 UEs from NEW placements, but a preempted
HARQ retransmission is already fixed in the RBG and bypasses the gate. If
the UE's buffer collapses to index 0 between attempts, its h_hat row is the
zero vector inside the RZF group. Intended semantics (this simulator's
abstraction, NOT a standard behavior): zero beam -> zero-power attempt ->
SINR 0 -> NACK path, group keeps p/m power split (the dead stream's share
is wasted, not redistributed), no NaN anywhere.

  A) unit: rzf_precoder/_rbg_sinr with one zero row -- dead stream gets a
     zero beam and SINR 0, live streams stay finite/positive, no NaN.
  B) integration: real HighLoad+nr4bit env; find a slot with a preempted
     retx, zero that UE's fed-back CQI, step on -- env must keep running
     with finite outputs and h_hat row exactly 0.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np
from config import Config
from env import SchedulerEnv
from baselines import all_baselines
from phy import rzf_precoder, _rbg_sinr

# ---- A) unit: one zero row in the RZF group --------------------------------
rng = np.random.default_rng(0)
m, M = 4, 32
h_hat = rng.standard_normal((m, M)) + 1j * rng.standard_normal((m, M))
h_true = h_hat + 0.1 * (rng.standard_normal((m, M)) + 1j * rng.standard_normal((m, M)))
h_hat[2] = 0.0                                    # dead-CQI retx row
W = rzf_precoder(h_hat, alpha=1.0, p_rbg=1.0)
assert np.all(np.isfinite(W)), "NaN/inf in precoder"
assert np.linalg.norm(W[:, 2]) == 0.0, "dead stream must get a zero beam"
assert all(np.linalg.norm(W[:, j]) > 0 for j in (0, 1, 3)), "live beams died"
s = _rbg_sinr(h_true, h_hat, alpha=1.0, p_rbg=1.0, noise_var=1e-3)
assert np.all(np.isfinite(s)), "NaN/inf in SINR"
assert s[2] == 0.0, "dead stream SINR must be exactly 0"
assert np.all(s[[0, 1, 3]] > 0), "live streams must keep positive SINR"
print("A) unit OK: zero beam, SINR 0, live streams finite/positive")

# ---- B) integration: forced dead-CQI retx in the live env ------------------
raw = json.load(open(
    "/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HighLoad/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = (1.0018, 0.7499, 0.6592, 0.6058)
raw["ue_speed_mix"] = ()
raw["cqi_mode"] = "nr4bit"
cfg = Config(**raw)
env = SchedulerEnv(cfg)
sch = [s_ for s_ in all_baselines(cfg) if s_.name == "SUS+CQI"][0]
env.reset(episode_idx=30010)
hit = False
for t in range(400):
    obs = env.get_observation()
    occ = obs["occupied"]
    if not hit and (occ > 0).any():
        r, l = np.argwhere(occ > 0)[0]
        u = int(occ[r, l]) - 1
        env.csi.cqi_fb[u, :] = 0.0                # collapse the report to idx 0
        _, rew, done, _ = env.step(sch.schedule(env))
        assert np.isfinite(rew), "reward NaN on dead-CQI retx slot"
        assert np.linalg.norm(env.h_hat_slot[u]) == 0.0, "h_hat row not zeroed"
        hit = True
        print(f"B) slot {t}: retx UE {u} at RBG {r} forced to CQI 0 -- "
              f"step survived, h_hat row exactly 0, reward finite")
        continue
    _, rew, done, _ = env.step(sch.schedule(env))
    assert np.isfinite(rew), f"reward NaN at slot {t}"
    if done:
        break
assert hit, "no preempted retx encountered in 400 slots (world too easy?)"
print("B) integration OK: env kept running with finite outputs")
print("REGRESSION PASS")
