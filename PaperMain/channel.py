"""
channel.py -- TR38.901 UMi system-level channel for the scheduler simulation.

Generates the *true* downlink channel ``h_true[slot, ue, rbg, bs_ant]`` with
temporal correlation (Doppler from UE velocity). One topology drop per
episode (``reset``); the whole episode's per-RBG channel is precomputed.

The true channel is internal to the simulator -- used for PMI/CQI generation,
RZF/SINR evaluation, and transmission success/failure. The scheduler never
observes it (only fed-back PMI/CQI/Age).

The channel model is isolated behind the ``ChannelGenerator`` interface
(``reset`` / ``get_channel``). A ray-tracing channel can later implement the
same interface without changing the rest of the simulator.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF banner

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from sionna.phy import config as sn_config
from sionna.phy.channel.tr38901 import UMi, PanelArray
from sionna.phy.channel import (gen_single_sector_topology,
                                subcarrier_frequencies, cir_to_ofdm_channel)

from config import Config


class ChannelGenerator:
    """TR38.901 UMi channel generator (per-RBG, temporally correlated).

    Usage
    -----
    >>> ch = ChannelGenerator(cfg)
    >>> ch.reset(episode_seed=0)            # new topology + episode channel
    >>> h = ch.get_channel(slot=0)          # [K, num_rbg, num_bs_ant] complex
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        pol_factor = 2 if cfg.bs_polarization == "dual" else 1
        n_ports = cfg.bs_ant_rows * cfg.bs_ant_cols * pol_factor
        if n_ports != cfg.num_bs_ant:
            raise ValueError(
                f"BS array {cfg.bs_ant_rows}x{cfg.bs_ant_cols}x{pol_factor} "
                f"= {n_ports} antennas != config.num_bs_ant={cfg.num_bs_ant}")

        # BS antenna panel: rows x cols x polarization -> num_bs_ant ports
        self.bs_array = PanelArray(
            num_rows_per_panel=cfg.bs_ant_rows,
            num_cols_per_panel=cfg.bs_ant_cols,
            polarization=cfg.bs_polarization,
            polarization_type="cross" if pol_factor == 2 else "V",
            antenna_pattern="38.901",
            carrier_frequency=cfg.carrier_frequency)

        # UT antenna panel: single omni antenna
        self.ut_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization="single", polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=cfg.carrier_frequency)

        self.umi = UMi(
            carrier_frequency=cfg.carrier_frequency,
            o2i_model=cfg.o2i_model,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="downlink",
            enable_pathloss=cfg.enable_pathloss,
            enable_shadow_fading=cfg.enable_shadow_fading)

        # one frequency per RBG -> RBG-center channel sample
        self._freqs = subcarrier_frequencies(cfg.num_rbg, cfg.rbg_bandwidth)

        # filled by reset()
        self.h_true_episode = None    # [num_slots, K, num_rbg, num_bs_ant]
        self.ut_loc = None            # [K, 3] UE positions
        self.ut_velocities = None     # [K, 3] UE velocity vectors
        self.ue_speeds_kmh = None     # [K] per-UE speed (km/h) for this episode
        # episode_seed -> precomputed channel, LRU-bounded. Training visits
        # each seed exactly once, so an unbounded dict would accumulate every
        # episode (~131 GB over a 2000-update main run); only repeated seeds
        # (eval rounds, phase-1 baselines) benefit from caching. The cap must
        # exceed ppo_eval_episodes + ppo_eval_every (3 + 10) so eval entries
        # survive the training inserts between two eval rounds.
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = 16

    def reset(self, episode_seed: int) -> np.ndarray:
        """Drop a new topology and precompute the episode's true channel.

        Parameters
        ----------
        episode_seed : int
            Seed for the topology drop (reproducible episodes).

        Returns
        -------
        ndarray, shape [num_slots, K, num_rbg, num_bs_ant], complex128
        """
        cfg = self.cfg
        if episode_seed in self._cache:
            self._cache.move_to_end(episode_seed)    # mark most-recently-used
            (self.h_true_episode, self.ut_loc, self.ut_velocities,
             self.ue_speeds_kmh) = self._cache[episode_seed]
            return self.h_true_episode

        sn_config.seed = int(episode_seed)

        # per-UE speed: continuous uniform range, heterogeneous mix, or uniform
        if getattr(cfg, "ue_speed_max", 0.0) > 0.0:
            rng_v = np.random.default_rng(int(episode_seed) + 777)
            speeds_kmh = rng_v.uniform(cfg.ue_speed_min, cfg.ue_speed_max,
                                       cfg.num_ue)             # per-UE U(min,max)
        elif tuple(getattr(cfg, "ue_speed_mix", ()) or ()):
            rng_v = np.random.default_rng(int(episode_seed) + 777)
            mix = np.asarray(cfg.ue_speed_mix, dtype=np.float64)
            reps = int(np.ceil(cfg.num_ue / mix.size))
            speeds_kmh = np.tile(mix, reps)[:cfg.num_ue].copy()
            rng_v.shuffle(speeds_kmh)                       # random UE<->speed map
        else:
            speeds_kmh = np.full(cfg.num_ue, cfg.ue_speed_kmh, dtype=np.float64)
        self.ue_speeds_kmh = speeds_kmh
        speeds_ms = speeds_kmh / 3.6

        # generate topology at the max speed, then rescale each UE's velocity
        # vector to its assigned per-UE speed (direction kept, magnitude set)
        gen_v = float(max(speeds_ms.max(), 1e-3))
        topology = gen_single_sector_topology(
            batch_size=1, num_ut=cfg.num_ue, scenario=cfg.scenario,
            min_ut_velocity=gen_v, max_ut_velocity=gen_v,
            indoor_probability=cfg.indoor_probability)
        topo = list(topology)
        v = np.asarray(topo[4])                            # [1, K, 3]
        vn = np.linalg.norm(v, axis=-1, keepdims=True)
        v_unit = np.divide(v, vn, out=np.zeros_like(v), where=vn > 1e-12)
        v_scaled = v_unit * speeds_ms.reshape(1, cfg.num_ue, 1)
        topo[4] = tf.constant(v_scaled, dtype=topology[4].dtype)
        topology = tuple(topo)
        self.umi.set_topology(*topology)
        self.ut_loc = np.asarray(topology[0])[0]          # [K, 3]
        self.ut_velocities = np.asarray(topology[4])[0]   # [K, 3]

        num_slots = cfg.episode_len
        # CIR over the whole episode; consecutive time steps are
        # Doppler-correlated (slot spacing = slot_duration).
        a, tau = self.umi(num_slots, 1.0 / cfg.slot_duration)

        # per-RBG OFDM channel response
        # [1, K, 1, 1, num_bs_ant, num_slots, num_rbg]
        h_freq = cir_to_ofdm_channel(self._freqs, a, tau, normalize=False)
        h = np.asarray(h_freq)[0, :, 0, 0, :, :, :]   # [K, ant, slots, rbg]
        h = np.transpose(h, (2, 0, 3, 1))             # [slots, K, rbg, ant]
        self.h_true_episode = np.ascontiguousarray(h.astype(np.complex128))
        self._cache[episode_seed] = (self.h_true_episode, self.ut_loc,
                                     self.ut_velocities, self.ue_speeds_kmh)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)          # evict least-recently-used
        return self.h_true_episode

    def get_channel(self, slot: int) -> np.ndarray:
        """True channel for a slot: [K, num_rbg, num_bs_ant] complex128."""
        if self.h_true_episode is None:
            raise RuntimeError("ChannelGenerator.reset() must be called first.")
        return self.h_true_episode[slot]


if __name__ == "__main__":
    # smoke test on a short episode
    from config import debug_config

    cfg = debug_config()
    print(f"channel.py smoke test: UMi, K={cfg.num_ue}, "
          f"{cfg.num_bs_ant} BS ant, {cfg.episode_len} slots, "
          f"{cfg.ue_speed_kmh} km/h")

    ch = ChannelGenerator(cfg)
    h = ch.reset(episode_seed=0)
    assert h.shape == (cfg.episode_len, cfg.num_ue, cfg.num_rbg,
                       cfg.num_bs_ant), h.shape
    assert np.all(np.isfinite(h)), "channel has non-finite values"
    assert np.abs(h).max() > 0, "channel is all-zero"

    # per-UE channel power spread (pathloss differences)
    p = np.mean(np.abs(h) ** 2, axis=(0, 2, 3))   # [K]
    print(f"  shape {h.shape}, channel power per UE: "
          f"min {p.min():.2e}  median {np.median(p):.2e}  max {p.max():.2e}")

    # temporal correlation: consecutive vs distant slots
    def corr(a, b):
        a, b = a.ravel(), b.ravel()
        return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

    c1 = corr(h[0], h[1])
    cN = corr(h[0], h[-1])
    print(f"  temporal corr: slot0-slot1 = {c1:.4f}, "
          f"slot0-slot{cfg.episode_len-1} = {cN:.4f}")
    assert c1 > 0.5, "consecutive slots not correlated -- check Doppler setup"
    print("channel.py smoke test passed.")
