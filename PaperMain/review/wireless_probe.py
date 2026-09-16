"""Small deterministic diagnostics for the extracted main recipes; no source edits."""
from pathlib import Path
import json, csv, sys
import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from config import Config
from phy import rzf_precoder, _rbg_sinr, reconstruct_h_hat, mi_bits
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation
from csi import quantize_cqi
from traffic import TrafficModel, Packet

raw = json.loads((repo / "artifacts/base/config.json").read_text())
for key in ("git_hash", "git_dirty_py"):
    raw.pop(key, None)
cfg = Config(**raw)
rng = np.random.default_rng(913)
max_sinr_error = max_power_error = 0.
for depth in range(1, 5):
    for _ in range(50):
        h = rng.normal(size=(depth, 32)) + 1j*rng.normal(size=(depth, 32))
        w, pred, _ = predict_group_link_adaptation(h, 1., 1., cfg.p_rbg, cfg)
        actual = _rbg_sinr(h, h, 1., cfg.p_rbg, 1.)
        max_sinr_error = max(max_sinr_error, float(np.max(abs(pred-actual))))
        max_power_error = max(max_power_error, abs(float(np.sum(abs(w)**2))-cfg.p_rbg))
assert max_sinr_error < 1e-10 and max_power_error < 1e-12

# A CQI=0 sampled member contributes an undefined cap ratio cast to 0;
# closure would remove it and recompute the remaining user's power/depth.
h = np.zeros((cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant), complex)
h[0,:,0] = np.sqrt(10.)
group = [0, 1]
_, sp, caps = predict_group_link_adaptation(h[group,0], 1., 1., cfg.p_rbg, cfg)
mi = mi_bits(_rbg_sinr(h[group,0], h[group,0], 1., cfg.p_rbg, 1.), cfg)
ratios = mi / np.maximum(cfg.eta_data*cfg.n_re_rbg*cfg.beta_rate*np.log2(1+sp), 1e-12)
p = SlotAllocationPlanner(cfg, h, 1., 1., np.full(cfg.num_ue, 100000.))
closed = p.solve_closure(0, [], group)
assert caps[1] == mi[1] == ratios[1] == 0 and closed["survivors"] == [0]

# Controlled same-seed queue-state counterexample: fill UE0 only in A.
# Arrival probability .50 is within the exact main training support.
a, b = TrafficModel(cfg), TrafficModel(cfg)
a.reset(); b.reset()
a.p_arrival_ep = b.p_arrival_ep = .50
for i in range(cfg.queue_size):
    a.queues[0].append(Packet(1000+i, 0, 0, 8000, 12))
a._sync_hol(0)
ra, rb = np.random.default_rng(8), np.random.default_rng(8)
arr_a, ov_a = a.generate_arrivals(0, ra)
arr_b, ov_b = b.generate_arrivals(0, rb)
feedback_a, feedback_b = ra.random(cfg.num_ue)<cfg.p_csi, rb.random(cfg.num_ue)<cfg.p_csi
assert ov_a > 0 and ov_b == 0 and not np.array_equal(feedback_a, feedback_b)

with (repo/"artifacts/base/csv_logs/env_metrics.csv").open() as f:
    rows=list(csv.DictReader(f))
ovrows=[r for r in rows if float(r['buffer_overflow_rate'])>0]
out={
    "genie_groups":200, "max_pred_actual_sinr_error":max_sinr_error,
    "max_nonzero_group_power_error":max_power_error,
    "calibration_zero_member":{"sampled_nominal_depth":2,"sampled_ratios":ratios.tolist(),
        "runtime_survivors":closed['survivors'], "runtime_depth":len(closed['group']),
        "nominal_live_sinr":float(sp[0]),"runtime_live_sinr":float(closed['sinr'][0])},
    "controlled_overflow_rng":{"overflow_A":ov_a,"overflow_B":ov_b,
        "admitted_A":arr_a,"admitted_B":arr_b,"different_next_CSI_mask_entries":int(np.sum(feedback_a!=feedback_b))},
    "historical_main_training":{"episodes":len(rows),"episodes_with_buffer_overflow":len(ovrows),
        "max_buffer_overflow_rate":max(float(r['buffer_overflow_rate']) for r in rows)},
    "notes":"Analytic/controller probes, not full training/evaluation. Overflow test deliberately starts two different queue states to expose RNG dependency; it is not an observed policy-vs-policy episode."}
print(json.dumps(out, indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2))
