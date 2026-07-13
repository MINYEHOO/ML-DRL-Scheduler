"""Claim 2 probe: swallow-tail can push B_tx above the physical cap and
NACK the first transmission in genie (beta=1) mode."""
import numpy as np

from config import phase2_hetero_config
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation
from phy import _rbg_sinr, mi_bits
from transmission import TransmissionUnit

cfg = phase2_hetero_config(pmi_mode="genie", p_csi=1.0,
                           la_mode="post_rzf", decode_order="rbg_major")
assert cfg.la_beta == 1.0 and not cfg.la_beta_by_depth
eps = cfg.b_tx_epsilon
nv = 1.3   # arbitrary noise variance

rng = np.random.default_rng(7)
h = (rng.standard_normal((1, cfg.num_bs_ant))
     + 1j * rng.standard_normal((1, cfg.num_bs_ant)))

# physical cap = predicted deliverable MI (genie, beta=1 -> exact)
_, sinr_pred, caps = predict_group_link_adaptation(h, nv, nv, cfg.p_rbg, cfg)
cap = float(caps[0])

# single-user genie case: uncommitted = cap + 0.5*epsilon
h_hat_slot = np.zeros((cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant), complex)
h_hat_slot[0, 0, :] = h[0]
unc = np.zeros(cfg.num_ue)
unc[0] = cap + 0.5 * eps

pl = SlotAllocationPlanner(cfg, h_hat_slot, nv, nv, unc)
kept, btx = pl.close_rbg(0, [], [0])
b_tx = btx[0]
print(f"epsilon          = {eps}")
print(f"cap (physical)   = {cap:.9f} bits")
print(f"uncommitted      = {unc[0]:.9f} bits (cap + eps/2)")
print(f"B_tx returned    = {b_tx:.9f} bits")
print(f"B_tx - cap       = {b_tx - cap:.9f} bits  -> over-cap: {b_tx > cap}")

# actual delivered MI, genie: h_true == h_hat, same alpha/power
sinr_act = _rbg_sinr(h, h, nv, cfg.p_rbg, nv)
mi = float(mi_bits(sinr_act, cfg)[0])
print(f"delivered MI     = {mi:.9f} bits")
print(f"MI < B_tx - 1e-6 : {mi < b_tx - 1e-6}")

# drive the real ACK logic
un = TransmissionUnit(unit_id=0, packet_id=0, ue_id=0, rbg_id=0,
                      previous_layer_id=0, current_layer_id=0, b_tx=b_tx)
un.i_acc += min(mi, un.b_tx - un.i_acc)
print(f"i_acc after 1st tx = {un.i_acc:.9f}; is_acked = {un.is_acked}"
      f"  (threshold b_tx-1e-6 = {b_tx - 1e-6:.9f})")

# control: same case WITHOUT the tail (uncommitted = cap exactly) first-ACKs
pl2 = SlotAllocationPlanner(cfg, h_hat_slot, nv, nv,
                            np.array([cap] + [0.0] * (cfg.num_ue - 1)))
_, btx2 = pl2.close_rbg(0, [], [0])
un2 = TransmissionUnit(unit_id=1, packet_id=0, ue_id=0, rbg_id=0,
                       previous_layer_id=0, current_layer_id=0, b_tx=btx2[0])
un2.i_acc += min(mi, un2.b_tx - un2.i_acc)
print(f"control (unc=cap): b_tx={btx2[0]:.9f}, is_acked={un2.is_acked}")

# and what the OFFICIAL (beta_m<1) preset does with the same overshoot:
BETA_M = (0.9815, 0.7306, 0.6466, 0.5922)
cfg_b = phase2_hetero_config(pmi_mode="genie", p_csi=1.0,
                             la_mode="post_rzf", decode_order="rbg_major",
                             la_beta_by_depth=BETA_M)
_, _, caps_b = predict_group_link_adaptation(h, nv, nv, cfg_b.p_rbg, cfg_b)
cap_b = float(caps_b[0])
b_b = cap_b + 0.5 * cfg_b.b_tx_epsilon      # swallowed over-cap unit
print(f"beta_m=0.9815: cap={cap_b:.3f}, swallowed b_tx={b_b:.3f}, "
      f"delivered MI={mi:.3f} -> first-ACK still ok: {mi >= b_b - 1e-6}")
