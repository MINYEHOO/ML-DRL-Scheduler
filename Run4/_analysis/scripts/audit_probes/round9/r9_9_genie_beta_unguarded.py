"""Part C probe: genie + beta != 1 passes every guard."""
from config import phase2_hetero_config, phase4_queue_config
from env import SchedulerEnv

cfg = phase2_hetero_config(pmi_mode="genie", la_mode="post_rzf",
                           decode_order="rbg_major",
                           la_beta_by_depth=(0.9, 0.7, 0.6, 0.5))
cfg.validate_la()
print("hetero genie + la_beta_by_depth=(0.9,0.7,0.6,0.5): validate_la PASSED")

cfg2 = phase4_queue_config(pmi_mode="genie", la_mode="post_rzf",
                           decode_order="rbg_major", la_beta=0.6469)
cfg2.validate_la()
print("queue  genie + scalar la_beta=0.6469          : validate_la PASSED")

# env construction re-runs the same validate_la (env.py:61) -- the only guard
env = SchedulerEnv(cfg)
print("SchedulerEnv(genie, post_rzf, beta_m!=1) constructed WITHOUT error:",
      type(env).__name__, "| resolved_la_mode =", cfg.resolved_la_mode(),
      "| pmi_mode =", cfg.pmi_mode, "| la_beta_by_depth =", cfg.la_beta_by_depth)
