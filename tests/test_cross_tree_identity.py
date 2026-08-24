"""Cross-tree identity gate: does the SEQUENTIAL path still behave exactly as
the unmodified main tree?

Why this exists (audit finding, common-mode blind spot):
    tests/test_replay_batch.py compares replay_batch against replay INSIDE one
    tree. But the refactor moved code that BOTH arms use --
    build_value_input was split into build_value_tail + a cat, and
    _position_logits_and_mask gained a return_ctx parameter -- so a regression
    in those shared helpers changes both sides equally and the in-tree gate
    reports PASS. Only a comparison against the untouched main tree can see it.

This matters concretely: the three live seed-replicate runs execute the MAIN
tree. If _batchdev's sequential path has drifted, then every checkpoint they
produce would evaluate differently under the merged code, silently
invalidating cross-run comparisons in the paper.

What is compared, on identical observations with identical weights:
    decode(deterministic=True) -> action_sequence, policy_decision_mask,
                                  log_prob_sum, entropy_sum, value
    replay(stored actions)     -> log_prob_sum, entropy_sum, value
    state_value(obs)           -> V(s)
Requirement: BIT-IDENTICAL. This is not a numerical-equivalence claim; the
sequential code path was not supposed to change at all, so any difference is
a regression.

Usage (from the _batchdev worktree root):
    CUDA_VISIBLE_DEVICES=<free gpu> python3 tests/test_cross_tree_identity.py
    CUDA_VISIBLE_DEVICES= python3 tests/test_cross_tree_identity.py     # cpu
Internally it re-invokes itself once per tree so the two `policy` modules
never share an interpreter.
"""
import json
import os
import subprocess
import sys
import tempfile

for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "4")

import numpy as np                                              # noqa: E402

MAIN = "/home/MYH/ML_DRL_Scheduler"
DEV = "/home/MYH/ML_DRL_Scheduler/_batchdev"
RUN = f"{MAIN}/Run4/QueuePostRZF_S40HL_CQI4"
N_SLOTS = int(os.environ.get("TEST_N_SLOTS", 24))
SEED = 90002                       # diagnostic band, disjoint from all bands
KEYS = ("log_prob_sum", "entropy_sum", "value", "state_value")


def worker(tree: str, out: str) -> None:
    """Run in a fresh interpreter with exactly one tree on sys.path."""
    sys.path.insert(0, tree)
    import torch
    torch.set_num_threads(4)
    from config import Config
    from env import SchedulerEnv
    from policy import ActorCritic

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
    ac.to(dev).eval()

    env = SchedulerEnv(cfg)
    obs = env.reset(SEED)
    rec = {k: [] for k in KEYS}
    acts, masks = [], []
    for _ in range(N_SLOTS):
        # deterministic decode: no sampling, so any difference is a real one
        out_d = ac.decode(obs, deterministic=True)
        rep = ac.replay(obs, out_d["action_sequence"],
                        out_d["policy_decision_mask"])
        acts.append(out_d["action_sequence"].copy())
        masks.append(out_d["policy_decision_mask"].copy())
        rec["log_prob_sum"].append(float(rep["log_prob_sum"]))
        rec["entropy_sum"].append(float(rep["entropy_sum"]))
        rec["value"].append(float(rep["value"]))
        rec["state_value"].append(float(ac.state_value(obs)))
        obs, _, done, _ = env.step(out_d["action_sequence"])
        if done:
            break
    np.savez(out, action=np.stack(acts), mask=np.stack(masks),
             **{k: np.array(v, dtype=np.float64) for k, v in rec.items()})


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return

    print("=" * 74)
    print("CROSS-TREE IDENTITY GATE  (sequential path, main vs _batchdev)")
    print("=" * 74)
    print(f"  weights : {RUN}/ckpt/best.pt      slots: {N_SLOTS}  seed {SEED}")
    tmp = tempfile.mkdtemp(prefix="xtree_")
    paths = {}
    for name, tree in (("main", MAIN), ("batchdev", DEV)):
        p = os.path.join(tmp, f"{name}.npz")
        print(f"  running worker on {tree} ...", flush=True)
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", tree, p],
            capture_output=True, text=True, env={**os.environ})
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            sys.exit(f"worker for {name} failed")
        paths[name] = p

    a = np.load(paths["main"])
    b = np.load(paths["batchdev"])
    ok = True
    print()
    n_act = int((a["action"] != b["action"]).sum())
    n_msk = int((a["mask"] != b["mask"]).sum())
    print(f"  action_sequence      differing entries: {n_act}   "
          f"{'OK' if n_act == 0 else 'REGRESSION'}")
    print(f"  policy_decision_mask differing entries: {n_msk}   "
          f"{'OK' if n_msk == 0 else 'REGRESSION'}")
    ok &= (n_act == 0 and n_msk == 0)
    for k in KEYS:
        d = np.abs(a[k] - b[k])
        exact = int((d == 0).sum())
        print(f"  {k:16s} max |diff| = {d.max():.3e}   bit-identical "
              f"{exact}/{len(d)}   {'OK' if d.max() == 0.0 else 'REGRESSION'}")
        ok &= bool(d.max() == 0.0)

    print()
    print("=" * 74)
    print("RESULT:", "PASS - the sequential path is unchanged"
          if ok else "*** FAIL - the refactor changed shared code ***")
    print("=" * 74)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
