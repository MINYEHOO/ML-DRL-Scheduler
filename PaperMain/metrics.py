"""
metrics.py -- evaluation metrics and the baseline evaluation loop.

Runs a scheduler over one or more episodes and aggregates:
  reward, throughput (Mbps), completion / deadline-miss / retx-drop rates,
  mean SINR (dB), and Jain's fairness index over delivered bits.
"""

from __future__ import annotations

import numpy as np


def jains_index(x) -> float:
    """Jain's fairness index of a non-negative vector (1 = perfectly fair)."""
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    if s <= 0:
        return 1.0                       # nothing served -> degenerate
    return float(s * s / (len(x) * np.sum(x * x)))


def run_episode(env, scheduler, episode_idx: int) -> dict:
    """Run one full episode; return its metric record."""
    cfg = env.cfg
    env.reset(episode_idx=episode_idx)
    done = False
    while not done:
        alloc = scheduler.schedule(env)
        _, _, done, _ = env.step(alloc)

    e = env.ep
    arrivals = max(e["n_arrivals"], 1)
    ep_time = cfg.episode_len * cfg.slot_duration
    mean_sinr_db = (10.0 * np.log10(e["sinr_sum"] / e["sinr_count"])
                    if e["sinr_count"] else float("nan"))
    direction_corr = (e["dir_corr_sum"] / e["dir_corr_count"]
                      if e["dir_corr_count"] else float("nan"))
    return dict(
        reward=e["reward"],
        throughput_mbps=e["acked_bits"] / ep_time / 1e6,
        completion_rate=e["n_comp"] / arrivals,
        deadline_miss_rate=e["n_miss_deadline"] / arrivals,
        retx_drop_rate=e["n_retx_drop"] / arrivals,
        mean_sinr_db=mean_sinr_db,
        # JFI over ACTIVE UEs only (standing metric rule; active = [0, n_active))
        jain=jains_index(env.cum_acked_bits[:env.traffic.n_active]),
        direction_corr=direction_corr,
        chordal_dist=1.0 - direction_corr,
        phase_aligned_nmse=2.0 * (1.0 - np.sqrt(max(direction_corr, 0.0))),
    )


def evaluate(env, scheduler, num_episodes: int,
             episode_offset: int = 0) -> dict:
    """Run several episodes; return {metric: (mean, std)}.

    ``episode_offset`` shifts the episode indices, e.g. 10000 selects the
    held-out seed range that Phase-2 training never visits (the same range
    train_phase2.run_eval uses for best-checkpoint selection).
    """
    recs = [run_episode(env, scheduler, episode_offset + ep)
            for ep in range(num_episodes)]
    return {k: (float(np.mean([r[k] for r in recs])),
                float(np.std([r[k] for r in recs])))
            for k in recs[0]}


# columns: key -> (header, format)
_COLUMNS = [
    ("reward",             "reward",     "{:8.1f}"),
    ("throughput_mbps",    "thrpt Mbps", "{:9.3f}"),
    ("completion_rate",    "comp",       "{:6.3f}"),
    ("deadline_miss_rate", "miss",       "{:6.3f}"),
    ("retx_drop_rate",     "retxdrop",   "{:8.3f}"),
    ("mean_sinr_db",       "SINR dB",    "{:7.2f}"),
    ("direction_corr",     "dirCorr",    "{:7.4f}"),
    ("chordal_dist",       "chordal",    "{:7.4f}"),
    ("phase_aligned_nmse", "phNMSE",     "{:7.4f}"),
    ("jain",               "Jain",       "{:5.3f}"),
]


def format_table(results: dict) -> str:
    """Format {scheduler_name: agg_dict} as a comparison table."""
    name_w = max(len(n) for n in results) + 1
    head = " " * name_w + "  ".join(h for _, h, _ in _COLUMNS)
    lines = [head, "-" * len(head)]
    for name, agg in results.items():
        row = name.ljust(name_w)
        cells = []
        for key, header, fmt in _COLUMNS:
            mean, _ = agg[key]
            cells.append(fmt.format(mean).rjust(len(header)))
        lines.append(row + "  ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    from config import debug_config
    from env import SchedulerEnv
    from baselines import PF

    cfg = debug_config()
    env = SchedulerEnv(cfg)
    agg = evaluate(env, PF(), num_episodes=2)
    print("metrics.py: PF over 2 episodes")
    for k, (m, s) in agg.items():
        print(f"  {k:20s}: {m:.4f} +/- {s:.4f}")
    assert np.isfinite(agg["reward"][0])
    assert 0.0 <= agg["completion_rate"][0] <= 1.0
    print("metrics.py smoke test passed.")
