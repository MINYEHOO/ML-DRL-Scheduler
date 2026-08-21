"""Equivalence gate at the ppo_update level, including the actor-frozen branch.

Why this exists (audit finding, the largest coverage gap):
    tests/test_replay_batch.py stops at replay_batch's four outputs under a
    smooth surrogate loss. Everything _minibatch_batched itself does was
    unmeasured: the .mean() scaling instead of per-sample /n_mb, the float64
    widening of the six logging accumulators, n_dec.clamp(min=1) as a tensor
    rather than a Python int, the discrete clip selector, and -- above all --
    the actor_frozen branch.

    That last one is not hypothetical. The live seeds log approx_KL 0.033-0.038
    against a 1.5*0.02 = 0.030 freeze threshold, so the frozen branch is the
    branch that actually runs in production, yet none of the three prior
    measurements touched it: the equivalence test never calls ppo_update, the
    3-update A/B is at the very start of training where KL is small, and the
    50-update soak reported 0 KL-guard fires.

What is compared, from IDENTICAL model + optimizer state on the SAME rollout:
    * the six metrics ppo_update returns
    * every parameter's post-step value, normalised by the size of the step
      that arm actually took (so the gate measures "did the two arms take the
      same step", not "are the weights close")
Two scenarios: normal, and target_kl forced tiny so the actor freezes on the
first minibatch and the frozen branch drives the remaining passes.

Usage (from the _batchdev worktree root):
    CUDA_VISIBLE_DEVICES=<free gpu> python3 tests/test_ppo_update_batched.py
Env: TEST_N_SLOTS (default 300 -> minibatches of 256 and 44, so the short
final minibatch is exercised too).
"""
import copy
import json
import os
import sys

for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "4")

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                                       # noqa: E402
from env import SchedulerEnv                                    # noqa: E402
from policy import ActorCritic                                  # noqa: E402
from ppo import ppo_update, compute_gae, SlotTrajectory         # noqa: E402

RUN = "/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4"
N_SLOTS = int(os.environ.get("TEST_N_SLOTS", 300))
SEED = 90003                      # diagnostic band, disjoint from all bands
TOL_METRIC = 1e-4                 # relative, on the six returned metrics
TOL_PARAM = 1e-3                  # relative to the size of the step taken


def build():
    raw = json.load(open(f"{RUN}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    cfg = Config(**raw)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ac = ActorCritic(cfg)
    ac.load_state_dict(torch.load(f"{RUN}/ckpt/best.pt",
                                  map_location="cpu")["model"], strict=True)
    return cfg, ac.to(dev), dev


def collect(env, ac, n_slots):
    """A partial rollout WITH contexts, so either arm can consume it."""
    obs = env.reset(SEED)
    trajs, last = [], 0.0
    for _ in range(n_slots):
        out = ac.decode(obs, deterministic=False, emit_context=True)
        nxt, reward, done, _ = env.step(out["action_sequence"])
        trajs.append(SlotTrajectory(
            obs=obs, action_sequence=out["action_sequence"].copy(),
            policy_decision_mask=out["policy_decision_mask"].copy(),
            fixed_unit_map=obs["fixed_unit_map"].copy(),
            old_logprob_sum=float(out["log_prob_sum"]),
            per_subaction_logprobs=list(out["per_subaction_logprobs"]),
            entropy_sum=float(out["entropy_sum"]),
            num_policy_decisions=int(out["num_policy_decisions"]),
            value=float(out["value"]), reward=float(reward), done=bool(done),
            ctx=out["context"]))
        obs = nxt
        if done:
            break
    last = float(ac.state_value(obs))
    return trajs, last


def run_arm(cfg, ac, trajs, last, w0, opt0, batched: bool, tag: str):
    """One ppo_update from the pinned initial state. Returns (metrics, params)."""
    ac.load_state_dict(copy.deepcopy(w0))
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.ppo_learning_rate)
    opt.load_state_dict(copy.deepcopy(opt0))
    c = copy.copy(cfg)
    c.ppo_batched_replay = batched
    c.ppo_batch_verify_every = 0          # tested separately; keep arms equal
    stats = ppo_update(ac, trajs, opt, c, update_idx=1, last_value=last)
    params = {n: p.detach().clone() for n, p in ac.named_parameters()}
    print(f"    [{tag}] {stats['policy_loss']:+.6f} pl  "
          f"{stats['value_loss']:.6f} vl  {stats['approx_KL']:.8f} kl  "
          f"{stats['clip_fraction']:.6f} clip")
    return stats, params


def compare(name, sa, pa, sb, pb, w0):
    print(f"\n  --- {name} ---")
    ok = True
    for k in ("policy_loss", "value_loss", "entropy_sum", "entropy_mean",
              "approx_KL", "clip_fraction", "grad_norm"):
        a, b = float(sa[k]), float(sb[k])
        rel = abs(a - b) / max(abs(a), 1e-12)
        good = rel < TOL_METRIC
        ok &= good
        print(f"    {k:16s} batched {a:+.8f}  sequential {b:+.8f}  "
              f"rel {rel:.2e}  {'OK' if good else 'FAIL'}")
    worst, where, worst_inf = 0.0, "", 0.0
    for n in pa:
        w = w0[n].to(pa[n].device)
        da, db = pa[n] - w, pb[n] - w
        na = float(da.norm())
        rel = float((da - db).norm()) / max(na, 1e-12)
        if rel > worst:
            worst, where = rel, n
        worst_inf = max(worst_inf, float((da - db).abs().max())
                        / max(float(da.abs().max()), 1e-12))
    good = worst < TOL_PARAM
    ok &= good
    print(f"    step vectors: worst ||Δbatched−Δsequential||₂ / ||Δ||₂ = "
          f"{worst:.3e} in '{where}'  {'OK' if good else 'FAIL'}")
    print(f"      (per-element max-norm ratio {worst_inf:.3e} -- expected to be"
          f" much larger: a fresh Adam takes lr*sign(g), so a ~1e-6 gradient"
          f" difference flips near-zero elements by a full step)")
    return ok


def main():
    cfg, ac, dev = build()
    print("=" * 78)
    print("ppo_update EQUIVALENCE GATE (batched vs sequential, incl. freeze)")
    print("=" * 78)
    print(f"  device {dev}   slots {N_SLOTS}   minibatch {cfg.ppo_minibatch_size}"
          f"   epochs {cfg.ppo_epochs}")

    # seed the SAMPLING too: collect() decodes stochastically, so without this
    # each run draws a different rollout and the Adam-amplified parameter
    # comparison below varies run to run (observed 1e-5 .. 1.1e-3).
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    env = SchedulerEnv(cfg)
    trajs, last = collect(env, ac, N_SLOTS)
    n_mb = int(np.ceil(len(trajs) / cfg.ppo_minibatch_size))
    print(f"  collected {len(trajs)} slots -> {n_mb} minibatches/epoch "
          f"(sizes {[min(cfg.ppo_minibatch_size, len(trajs) - i * cfg.ppo_minibatch_size) for i in range(n_mb)]})")

    # WARM UP ADAM FIRST. With empty moments Adam's first step is
    # lr*sign(g), a step function at g=0, so a ~1e-6 gradient difference flips
    # every element whose gradient is within 1e-6 of zero and moves it by a
    # full 2*lr. That makes an update-0 parameter comparison measure Adam's
    # discontinuity rather than the two arms' agreement -- and it is also
    # unrepresentative: from update 1 onward the live runs always have
    # populated moments. So take one sequential update, then compare the NEXT
    # one from that warmed state.
    warm = torch.optim.Adam(ac.parameters(), lr=cfg.ppo_learning_rate)
    cw = copy.copy(cfg)
    cw.ppo_batched_replay = False
    cw.ppo_batch_verify_every = 0
    ppo_update(ac, trajs, warm, cw, update_idx=0, last_value=last)
    w0 = {k: v.detach().clone() for k, v in ac.state_dict().items()}
    opt0 = copy.deepcopy(warm.state_dict())
    w0p = {n: p.detach().clone() for n, p in ac.named_parameters()}
    print("  Adam warmed with one sequential update; comparing update 1")
    ok = True

    print("\n  SCENARIO A: KL guard DISABLED (target_kl = 0) -- every"
          " minibatch takes the full actor+critic branch")
    cA = copy.copy(cfg)
    cA.ppo_target_kl = 0.0
    sa, pa = run_arm(cA, ac, trajs, last, w0, opt0, True, "batched   ")
    sb, pb = run_arm(cA, ac, trajs, last, w0, opt0, False, "sequential")
    ok &= compare("guard disabled", sa, pa, sb, pb, w0p)

    print("\n  SCENARIO B: KL guard ACTIVE (target_kl = 0.02, as in the live"
          " runs) -- the freeze fires early and the FROZEN branch drives the"
          " remaining minibatches")
    cB = copy.copy(cfg)
    cB.ppo_target_kl = 0.02
    sa2, pa2 = run_arm(cB, ac, trajs, last, w0, opt0, True, "batched   ")
    sb2, pb2 = run_arm(cB, ac, trajs, last, w0, opt0, False, "sequential")
    ok &= compare("guard active (frozen branch)", sa2, pa2, sb2, pb2, w0p)

    # the freeze must actually have engaged, else scenario 2 tested nothing
    def moved(ps, pref):
        return max(float((ps[n] - w0p[n]).norm()) for n in ps
                   if n.startswith(pref))
    a_free, v_free = moved(pa, "encoder"), moved(pa, "value_head")
    a_frz, v_frz = moved(pa2, "encoder"), moved(pa2, "value_head")
    print(f"\n    freeze actually engaged? encoder ||Δ||: guard-off "
          f"{a_free:.4e} -> guard-on {a_frz:.4e}   "
          f"(must SHRINK: the actor stops stepping)")
    print(f"                              value_head ||Δ||: guard-off "
          f"{v_free:.4e} -> guard-on {v_frz:.4e}   "
          f"(must NOT collapse: the critic keeps training)")
    if not (a_frz < a_free and v_frz > 0.1 * v_free):
        print("    *** the frozen branch was not exercised as intended ***")
        ok = False

    print()
    print("=" * 78)
    print("RESULT:", "PASS" if ok else "*** FAIL ***")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
