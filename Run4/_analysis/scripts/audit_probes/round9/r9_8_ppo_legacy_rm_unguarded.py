"""Part B probe: PPO decode with la_mode='legacy' + decode_order='rbg_major'
routes through _rbg_major_pass (post-RZF SlotAllocationPlanner) unguarded."""
import numpy as np
import torch
from config import phase4_queue_config
from env import SchedulerEnv
from policy import ActorCritic
from phy import predict_b_tx

cfg = phase4_queue_config(la_mode="legacy", decode_order="rbg_major",
                          seed=2024)
print("resolved_la_mode:", cfg.resolved_la_mode(),
      "| decode_order:", cfg.decode_order)
cfg.validate_la()
print("validate_la: PASSED (no legacy+rbg_major rejection)")

env = SchedulerEnv(cfg)
obs = env.reset(episode_idx=0)
empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
for _ in range(6):
    obs, *_ = env.step(empty)

torch.manual_seed(0)
model = ActorCritic(cfg)
out = model.decode(obs, deterministic=True)
print("decode() returned keys:", sorted(out.keys()))
has_plan = "planned_btx" in out and out["planned_btx"] is not None
print("decode used _rbg_major_pass (post-RZF planner exports present):",
      has_plan)
if has_plan and out["planned_btx"]:
    # compare a planned (post-RZF) B_tx against the legacy env's sizing rule
    (r, u), b_plan = next(iter(out["planned_btx"].items()))
    legacy_cap = float(predict_b_tx(env.csi.cqi_fb[u, r], cfg))
    unc = env.traffic.packets[u].uncommitted_backlog
    print(f"  sample (r={r}, u={u}): planner B_tx={b_plan:.1f}  vs  "
          f"legacy env rule min(backlog={unc:.1f}, SU-CQI cap={legacy_cap:.1f})"
          f"={min(unc, legacy_cap):.1f}")
