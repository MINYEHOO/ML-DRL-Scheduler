"""post-RZF world baseline evaluation -- BOTH backoff modes (audit round 8).

Runs the full baseline grid on the same 8 paired seeds in
  world "global": post_rzf + rbg_major + scalar la_beta = 0.6469 (ablation)
  world "betam" : post_rzf + rbg_major + la_beta_by_depth
                  (0.9815, 0.7306, 0.6466, 0.5922)          (official)
and writes, per world:
  new_la_baselines_<world>.csv      -- per-baseline summary (rewards etc. are
                                       seed means; per-depth first-ACK is the
                                       POOLED sum(acks)/sum(units) -- never
                                       the mean of per-episode rates)
  new_la_baselines_<world>_raw.csv  -- world x scheduler x seed raw rows
                                       (every scalar metric + per-depth
                                       acks/units), for reanalysis.

The legacy rows of the 4-way decomposition table (legacy+LM, legacy+RM)
come from control_lm_rm.py. Reference results: docs/RUNS.md 4.5.

Run from the repo root:
  python3 Run4/_analysis/scripts/audit_probes/new_la_baselines.py
"""
import csv
import numpy as np

from config import phase4_queue_config
from env import SchedulerEnv
from baselines import all_baselines
from train_phase2 import env_episode_metrics

SEEDS = list(range(10000, 10008))
OUT_DIR = "Run4/_analysis"
WORLDS = {
    "global": dict(la_beta=0.6469),
    "betam": dict(la_beta_by_depth=(0.9815, 0.7306, 0.6466, 0.5922)),
}
SUMMARY_KEYS = ["reward", "throughput_mbps", "completion_rate",
                "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain",
                "first_ack_rate", "attempts_per_acked", "pinned_fraction",
                "goodput_mbps"]

for world, la_kw in WORLDS.items():
    cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                              p_arrival_min=0.15, p_arrival_max=0.40, **la_kw)
    env = SchedulerEnv(cfg)
    raw_rows, summary = [], {}
    for sch in all_baselines(cfg):
        ms = []
        for s in SEEDS:
            env.reset(episode_idx=s)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            em = env_episode_metrics(env, cfg)
            ms.append(em)
            raw_rows.append([world, sch.name, s]
                            + [em[k] for k in SUMMARY_KEYS]
                            + [em[f"acks_m{m}"] for m in (1, 2, 3, 4)]
                            + [em[f"units_m{m}"] for m in (1, 2, 3, 4)])
        agg = {k: float(np.mean([m[k] for m in ms])) for k in SUMMARY_KEYS}
        for m in (1, 2, 3, 4):
            ub = sum(x[f"units_m{m}"] for x in ms)
            ak = sum(x[f"acks_m{m}"] for x in ms)
            agg[f"acks_m{m}"], agg[f"units_m{m}"] = ak, ub
            agg[f"first_ack_m{m}"] = ak / ub if ub > 0 else float("nan")
        summary[sch.name] = agg
        print(f"[{world}] {sch.name:16s} rew {agg['reward']:8.1f}  "
              f"depth {agg['mu_depth']:.2f}  1ACK {agg['first_ack_rate']:.3f}"
              + "".join(f"  m{m}:{agg[f'first_ack_m{m}']:.3f}"
                        f"(n={agg[f'units_m{m}']})" for m in (1, 2, 3, 4)),
              flush=True)

    with open(f"{OUT_DIR}/new_la_baselines_{world}_raw.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["world", "baseline", "seed"] + SUMMARY_KEYS
                   + [f"acks_m{m}" for m in (1, 2, 3, 4)]
                   + [f"units_m{m}" for m in (1, 2, 3, 4)])
        w.writerows(raw_rows)
    keys = SUMMARY_KEYS + [x for m in (1, 2, 3, 4)
                           for x in (f"acks_m{m}", f"units_m{m}",
                                     f"first_ack_m{m}")]
    with open(f"{OUT_DIR}/new_la_baselines_{world}.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline"] + keys)
        for name, agg in summary.items():
            w.writerow([name] + [agg[k] for k in keys])
    print(f"[{world}] saved summary + raw CSVs", flush=True)
