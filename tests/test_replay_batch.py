"""Equivalence gate for ActorCritic.replay_batch.

The batched replay is a SPEED refactor only: it must reproduce the sequential
``replay`` bit-for-bit up to float32 reduction order, in BOTH the forward
values and the gradients they produce. If this test does not pass, the
refactor is wrong and must not be merged.

What is compared, per slot, against the sequential reference:
    log_prob_sum, entropy_sum, value, num_policy_decisions
and then, over the whole minibatch:
    d/dtheta [ sum_i (log_prob_sum_i + value_i + entropy_sum_i) ]
for every parameter tensor.

The only legitimate source of difference is reduction ORDER: the sequential
path accumulates ~28 log-probs into a running scalar, the batched path sums a
padded row. That is a ~1e-6 relative effect on float32; the gate is 1e-4,
matching the existing decode-vs-replay smoke assert in policy.py.

Run (from the worktree root):
    CUDA_VISIBLE_DEVICES=<free gpu> python3 tests/test_replay_batch.py
    CUDA_VISIBLE_DEVICES= python3 tests/test_replay_batch.py       # cpu
Env overrides: TEST_N_SLOTS (default 48), TEST_RUN (config/ckpt source).
"""
import json
import os
import sys
import time

for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "4")

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                                       # noqa: E402
from env import SchedulerEnv                                    # noqa: E402
from policy import ActorCritic                                  # noqa: E402

RUN = os.environ.get(
    "TEST_RUN", "/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4")
N_SLOTS = int(os.environ.get("TEST_N_SLOTS", 48))
TOL = 1e-4
SEED = 90001            # diagnostic band, disjoint from every reserved band


def load():
    raw = json.load(open(f"{RUN}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    cfg = Config(**raw)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ac = ActorCritic(cfg)
    ck_path = f"{RUN}/ckpt/best.pt"
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu")
        ac.load_state_dict(ck["model"], strict=True)
        src = f"best.pt@{ck['update']}"
    else:
        src = "random init"
    return cfg, ac.to(dev), dev, src


def collect(env, ac, n_slots):
    """Rollout n_slots with context emission, exactly as collect_rollout will."""
    obs = env.reset(SEED)
    slots = []
    for _ in range(n_slots):
        out = ac.decode(obs, deterministic=False, emit_context=True)
        assert out["context"] is not None, "decode did not emit a context"
        slots.append(dict(obs=obs,
                          act=out["action_sequence"].copy(),
                          mask=out["policy_decision_mask"].copy(),
                          ctx=out["context"],
                          dec_lp=out["log_prob_sum"],
                          dec_v=out["value"],
                          dec_n=out["num_policy_decisions"]))
        obs, _, done, _ = env.step(out["action_sequence"])
        if done:
            break
    return slots


def grads_of(ac, loss):
    ac.zero_grad(set_to_none=True)
    loss.backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in ac.named_parameters()}


def main():
    cfg, ac, dev, src = load()
    print("=" * 74)
    print("replay_batch EQUIVALENCE GATE")
    print("=" * 74)
    print(f"  run     : {RUN}")
    print(f"  weights : {src}   device {dev}")
    print(f"  world   : K={cfg.num_ue} R={cfg.num_rbg} L={cfg.l_max} "
          f"queue={cfg.queue_size} cqi={cfg.cqi_mode} order={cfg.decode_order}")
    assert cfg.decode_order == "rbg_major", "replay_batch is rbg-major only"

    env = SchedulerEnv(cfg)
    t0 = time.time()
    slots = collect(env, ac, N_SLOTS)
    print(f"  collected {len(slots)} slots in {time.time() - t0:.1f}s")
    print(f"  decisions/slot: mean {np.mean([s['dec_n'] for s in slots]):.2f} "
          f"min {min(s['dec_n'] for s in slots)} "
          f"max {max(s['dec_n'] for s in slots)}")

    # ---- 1. sequential reference ----------------------------------------
    t0 = time.time()
    seq = [ac.replay(s["obs"], s["act"], s["mask"]) for s in slots]
    t_seq = time.time() - t0

    # ---- 2. batched ------------------------------------------------------
    t0 = time.time()
    bat = ac.replay_batch([s["ctx"] for s in slots])
    t_bat = time.time() - t0

    # ---- 3. forward comparison ------------------------------------------
    print()
    print("  forward, per slot (max |diff| over all slots):")
    worst = {}
    for key in ("log_prob_sum", "entropy_sum", "value"):
        d = max(abs(float(seq[i][key]) - float(bat[key][i]))
                for i in range(len(slots)))
        worst[key] = d
        print(f"    {key:16s} max |diff| = {d:.3e}   "
              f"{'OK' if d < TOL else 'FAIL'}")
    nd_bad = sum(int(seq[i]["num_policy_decisions"])
                 != int(bat["num_policy_decisions"][i])
                 for i in range(len(slots)))
    print(f"    {'n_decisions':16s} mismatches = {nd_bad}   "
          f"{'OK' if nd_bad == 0 else 'FAIL'}")

    # decode agreement too (the context must describe the decode that made it)
    d_dec = max(abs(s["dec_lp"] - float(seq[i]["log_prob_sum"]))
                for i, s in enumerate(slots))
    print(f"    {'vs decode lp':16s} max |diff| = {d_dec:.3e}   "
          f"(sequential replay vs the decode that produced the actions)")

    # ---- 4. gradient comparison -----------------------------------------
    # An all-ones projection sum_i(lp_i + v_i + ent_i) is a WEAK probe: any
    # error that cancels across slots, or that swaps gradient between the three
    # heads, is invisible. Use fixed RANDOM per-slot weights and back-prop each
    # head SEPARATELY, so a slot-misattribution in the ragged scatter or a
    # head mix-up cannot cancel.
    gen = torch.Generator(device="cpu").manual_seed(20260821)
    w = torch.randn(len(slots), generator=gen).to(dev)
    print()
    print("  gradients (random per-slot weights, one backward per head):")
    gmax, gname, gkey = 0.0, "", ""
    missing = []
    for key in ("log_prob_sum", "entropy_sum", "value"):
        # rebuild the sequential graph per key: one backward frees it, and
        # retain_graph would keep three full graphs alive simultaneously
        seq_k = [ac.replay(s["obs"], s["act"], s["mask"]) for s in slots]
        loss_seq = sum(w[i] * seq_k[i][key] for i in range(len(slots)))
        g_seq = grads_of(ac, loss_seq)
        bat2 = ac.replay_batch([s["ctx"] for s in slots])
        loss_bat = (w * bat2[key]).sum()
        g_bat = grads_of(ac, loss_bat)
        km, kn = 0.0, ""
        for n in g_seq:
            a, b = g_seq[n], g_bat[n]
            if a is None or b is None:
                if a is not b:
                    missing.append(f"{key}:{n}")
                continue
            rel = float((a - b).abs().max()) / max(float(a.abs().max()), 1e-12)
            if rel > km:
                km, kn = rel, n
        print(f"    {key:16s} loss seq {float(loss_seq):+12.4f} vs bat "
              f"{float(loss_bat):+12.4f}   worst rel grad {km:.3e} in '{kn}'"
              f"   {'OK' if km < TOL else 'FAIL'}")
        if km > gmax:
            gmax, gname, gkey = km, kn, key
    print(f"    WORST OVERALL = {gmax:.3e}  ({gkey} / {gname})   "
          f"{'OK' if gmax < TOL else 'FAIL'}")
    if missing:
        print(f"    *** grad presence mismatch: {missing}")

    # ---- 5. speed (indicative only; a real benchmark needs a full update) --
    print()
    print(f"  timing on {len(slots)} slots: sequential {t_seq:.2f}s   "
          f"batched {t_bat:.2f}s   speedup {t_seq / max(t_bat, 1e-9):.1f}x")
    print("    (indicative: one forward pass, no backward, small batch)")

    ok = (all(v < TOL for v in worst.values()) and nd_bad == 0
          and gmax < TOL and not missing)
    print()
    print("=" * 74)
    print("RESULT:", "PASS" if ok else "*** FAIL ***")
    print("=" * 74)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
