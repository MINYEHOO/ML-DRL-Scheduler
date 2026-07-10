"""
run_phase1.py -- Phase 1 driver: sanity checks + baseline comparison.

Phase 1 validates the environment and baselines before PPO (Phase 2):
  * environment reset/step works
  * RZF/SINR values are sane
  * retransmission preemption is active
  * CSI aging is active and has an observable effect
  * reward scale does not explode
  * baseline throughput / deadline / fairness metrics print correctly

Usage:
  python run_phase1.py            # debug config (200-slot episodes)
  python run_phase1.py main       # main config (1000-slot episodes)
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from config import Config, debug_config
from env import SchedulerEnv
from baselines import all_baselines, SUSPF
from metrics import evaluate, format_table


def sanity_checks(cfg: Config) -> bool:
    """Run one instrumented episode and assert Phase-1 invariants."""
    print("\n=== Sanity checks ===")
    env = SchedulerEnv(cfg)
    sched = SUSPF()
    obs = env.reset(episode_idx=0)
    assert obs["direction_fb"].shape == (cfg.num_ue, cfg.num_rbg,
                                          cfg.num_bs_ant)

    rewards, sinr_db = [], []
    max_age, preempt_slots = 0, 0
    done = False
    while not done:
        if (obs["occupied"] > 0).any():
            preempt_slots += 1
        alloc = sched.schedule(env)
        obs, reward, done, info = env.step(alloc)
        rewards.append(reward)
        if np.isfinite(info["sinr_db_mean"]):
            sinr_db.append(info["sinr_db_mean"])
        max_age = max(max_age, int(env.csi.age.max()))

    rewards = np.asarray(rewards)
    sinr_db = np.asarray(sinr_db)
    checks = [
        ("env reset/step returns finite rewards",
         bool(np.all(np.isfinite(rewards)))),
        (f"SINR sane: mean {sinr_db.mean():.1f} dB, range "
         f"[{sinr_db.min():.1f}, {sinr_db.max():.1f}]",
         bool(np.all(np.isfinite(sinr_db)) and -20 < sinr_db.mean() < 50)),
        (f"reward bounded: |max|={np.abs(rewards).max():.1f} per slot",
         bool(np.abs(rewards).max() < 1e4)),
        (f"retx preemption active: {preempt_slots} slots with preemption",
         preempt_slots > 0),
        (f"CSI aging active: max Age = {max_age} slots",
         max_age > 0),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    if not ok:
        raise AssertionError("Phase-1 sanity checks FAILED")
    print("  -> all sanity checks passed.")
    return ok


def baseline_comparison(cfg: Config, num_episodes: int,
                        pmi_mode: str | None = None) -> dict:
    """Evaluate every baseline under the configured (or overridden) pmi_mode."""
    c = cfg if pmi_mode is None else replace(cfg, pmi_mode=pmi_mode)
    env = SchedulerEnv(c)
    results = {s.name: evaluate(env, s, num_episodes)
               for s in all_baselines(c)}
    return results


def compare_pmi_modes(cfg: Config, num_episodes: int) -> None:
    """Run all baselines under each PMI codebook mode and print tables."""
    modes = ("random_unit_norm", "type2_sparse_56bit")
    for mode in modes:
        print(f"\n=== Baselines under pmi_mode = {mode} "
              f"({num_episodes} episodes) ===")
        results = baseline_comparison(cfg, num_episodes, pmi_mode=mode)
        print(format_table(results))


def csi_aging_report(cfg: Config, num_episodes: int) -> None:
    """Compare fresh vs. stale CSI to confirm the aging mechanism bites."""
    print(f"\n=== CSI aging effect, SUS+PF ({num_episodes} episodes) ===")
    print("  (UE speed 3 km/h -> weak aging by design; "
          "stronger at 30 km/h)")
    for p_csi in (1.0, cfg.p_csi):
        env = SchedulerEnv(replace(cfg, p_csi=p_csi))
        agg = evaluate(env, SUSPF(), num_episodes)
        print(f"  p_csi={p_csi:.2f}: "
              f"throughput {agg['throughput_mbps'][0]:7.3f} Mbps   "
              f"retx_drop {agg['retx_drop_rate'][0]:.3f}   "
              f"SINR {agg['mean_sinr_db'][0]:6.2f} dB")


def main() -> None:
    use_main = len(sys.argv) > 1 and sys.argv[1] == "main"
    cfg = Config() if use_main else debug_config()
    num_episodes = 3 if use_main else 3

    print(cfg.describe())
    sanity_checks(cfg)
    compare_pmi_modes(cfg, num_episodes)
    csi_aging_report(cfg, max(2, num_episodes - 1))
    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
