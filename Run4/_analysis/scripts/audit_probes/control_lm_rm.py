"""Order-effect control: legacy LA fixed, layer_major vs rbg_major traversal.

Baselines only, paired same seeds (queue op point, imperfect CSI). If the
traversal order alone barely moves rewards, any later post_rzf-vs-legacy gap
is attributable to the LA change, not the decode order.
"""
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from baselines import all_baselines

SEEDS = list(range(10000, 10008))


def run(decode_order):
    cfg = phase4_queue_config(la_mode="legacy", decode_order=decode_order,
                              p_arrival_min=0.15, p_arrival_max=0.40)
    env = SchedulerEnv(cfg)
    out = {}
    for sch in all_baselines(cfg):
        rs = []
        for s in SEEDS:
            env.reset(episode_idx=s)
            done, tot = False, 0.0
            while not done:
                _, r, done, _ = env.step(sch.schedule(env))
                tot += r
            rs.append(tot)
        out[sch.name] = np.array(rs)
    return out


lm = run("layer_major")
rm = run("rbg_major")
print(f"{'baseline':16s} {'LM mean':>9s} {'RM mean':>9s} {'d%':>7s}  paired d (mean+-sd)")
worst = 0.0
for name in lm:
    a, b = lm[name], rm[name]
    d = b - a
    pct = d.mean() / abs(a.mean()) * 100
    worst = max(worst, abs(pct))
    print(f"{name:16s} {a.mean():9.1f} {b.mean():9.1f} {pct:+6.2f}%  "
          f"{d.mean():+8.1f} +- {d.std():.1f}")
print(f"\nworst |order effect| = {worst:.2f}%  (n={len(SEEDS)} paired seeds)")
