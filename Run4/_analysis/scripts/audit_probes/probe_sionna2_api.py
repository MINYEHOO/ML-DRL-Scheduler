"""Probe A: replicate channel.py's Sionna API sequence under sionna 2.0.1.

Mirrors /home/MYH/ML_DRL_Scheduler/channel.py lines 27-30 (imports), 56-78
(PanelArray/UMi ctor kwargs), 81 (subcarrier_frequencies), 116 (config.seed),
137-148 (topology + velocity rescale + set_topology), 155-160 (CIR ->
cir_to_ofdm_channel -> np.asarray). TF is NOT importable in this venv
(sionna 2.x forced numpy 2.4.6; TF 2.15 needs numpy<2), so the tf.constant
step (channel.py line 146) is checked type-wise here and behaviorally in
probe B under the system interpreter.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings; warnings.filterwarnings("ignore")

# venv interpreter is Python 3.11.0rc1 which lacks the int-max-str-digits API
# added in 3.11.0 final; torch 2.13 _dynamo polyfills reference it. Shim it
# (environment quirk, unrelated to the Sionna API under test).
import sys
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> "int":
        return 4300
    def set_int_max_str_digits(maxdigits: "int") -> "None":
        return None
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits

import numpy as np
import torch

step = "imports"
try:
    from sionna.phy import config as sn_config
    from sionna.phy.channel.tr38901 import UMi, PanelArray
    from sionna.phy.channel import (gen_single_sector_topology,
                                    subcarrier_frequencies,
                                    cir_to_ofdm_channel)
    print("[OK] imports: same module paths as channel.py resolve in 2.0.1")

    step = "config.seed (channel.py:116)"
    sn_config.seed = 0
    print("[OK]", step)

    step = "PanelArray/UMi ctor (channel.py:56-78)"
    bs = PanelArray(num_rows_per_panel=2, num_cols_per_panel=4,
                    polarization="dual", polarization_type="cross",
                    antenna_pattern="38.901", carrier_frequency=3.5e9)
    ut = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                    polarization="single", polarization_type="V",
                    antenna_pattern="omni", carrier_frequency=3.5e9)
    umi = UMi(carrier_frequency=3.5e9, o2i_model="low", ut_array=ut,
              bs_array=bs, direction="downlink", enable_pathloss=True,
              enable_shadow_fading=True)
    print("[OK]", step)

    step = "subcarrier_frequencies (channel.py:81)"
    freqs = subcarrier_frequencies(4, 1.44e6)
    print("[OK]", step, "-> type:", type(freqs).__name__)

    step = "gen_single_sector_topology (channel.py:137-140)"
    topology = gen_single_sector_topology(
        batch_size=1, num_ut=4, scenario="umi",
        min_ut_velocity=8.333, max_ut_velocity=8.333,
        indoor_probability=0.0)
    t4 = topology[4]
    print("[OK]", step)
    print("     topology[4] type:", type(t4), "| dtype:", t4.dtype,
          "| dtype type:", type(t4.dtype))

    step = "velocity rescale as in channel.py:142-147 (numpy path)"
    v = np.asarray(t4)  # channel.py:142  np.asarray on returned tensor
    print("[OK] np.asarray(topology[4]) works -> shape", v.shape)
    # channel.py:146 does: tf.constant(v_scaled, dtype=topology[4].dtype)
    # Here topology[4].dtype is a torch.dtype -- shown behaviorally in probe B.
    is_torch_dtype = isinstance(t4.dtype, torch.dtype)
    print("     topology[4].dtype is torch.dtype:", is_torch_dtype)

    step = "set_topology with a torch replacement (channel.py:148)"
    topo = list(topology)
    topo[4] = torch.as_tensor(v, dtype=t4.dtype)  # torch, NOT tf.constant
    umi.set_topology(*tuple(topo))
    print("[OK]", step, "(only when handed a torch tensor)")

    step = "set_topology with a numpy array in slot 4 (what channel.py would produce if tf.constant were naively swapped for np)"
    topo2 = list(topology)
    topo2[4] = v.copy()
    try:
        umi.set_topology(*tuple(topo2))
        print("[OK] set_topology also accepts numpy")
    except Exception as e:
        print("[FAIL-tolerated] set_topology(numpy):",
              type(e).__name__, str(e)[:150])

    step = "CIR generation umi(num_slots, fs) (channel.py:155)"
    a, tau = umi(10, 1000.0)
    print("[OK]", step, "-> a:", type(a).__name__, tuple(a.shape),
          "| tau:", type(tau).__name__)

    step = "cir_to_ofdm_channel + np.asarray (channel.py:159-160)"
    h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)
    h = np.asarray(h_freq)[0, :, 0, 0, :, :, :]
    print("[OK]", step, "-> h", h.shape, h.dtype,
          "| finite:", bool(np.all(np.isfinite(h))),
          "| nonzero:", bool(np.abs(h).max() > 0))

    print("\nPROBE-A VERDICT: sionna 2.0.1 keeps the call signatures but "
          "returns TORCH tensors; channel.py's tf.constant bridge "
          "(line 146) is the type boundary that breaks.")
except Exception as e:
    print(f"[BROKE] at step '{step}':", type(e).__name__, str(e)[:300])
    raise SystemExit(1)
