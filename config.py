"""
config.py -- central configuration for the ML DRL Scheduler simulation.

All hyperparameters live here in a single dataclass. Phase 1 (environment +
baselines + metrics) and Phase 2 (PPO) both read the same ``Config`` object.
Edit values here; avoid hard-coding constants elsewhere.

System model (single-cell MIMO-OFDM downlink):
  - 32 BS antennas, 1 UE antenna, K UEs
  - 8 RBG x 4 spatial layers = 32 scheduling positions
  - RZF precoder, PHY abstraction (Shannon SE x RE count)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    # ---- reproducibility ----
    seed: int = 2024

    # ---- system: cell / antennas ----
    num_bs_ant: int = 32           # M, BS antennas
    num_ue_ant: int = 1            # 1 UE antenna (spec)
    bs_ant_rows: int = 4           # PanelArray layout: 4 x 4 x dual-pol = 32
    bs_ant_cols: int = 4
    bs_polarization: str = "dual"  # 'dual' -> x2 ports

    # ---- users ----
    num_ue: int = 16               # K, fixed during training
    # Level-2 mixed-load: if n_active_max > 0, each episode activates a random
    # n_active in [n_active_min, n_active_max] of the num_ue UEs (rest stay idle),
    # so ONE policy sees varying contention -> tests online SU/MU adaptation.
    n_active_min: int = 0
    n_active_max: int = 0
    # continuous per-UE uniform speed: if ue_speed_max > 0, each UE's speed is
    # sampled U(ue_speed_min, ue_speed_max) km/h per episode (overrides mix).
    ue_speed_min: float = 0.0
    ue_speed_max: float = 0.0

    # ---- frequency / OFDM / RBG ----
    num_rbg: int = 8               # R, scheduler frequency units
    num_rb_per_rbg: int = 8
    num_sc_per_rb: int = 12
    num_sym_per_rb: int = 14
    subcarrier_spacing: float = 30e3   # Hz
    carrier_frequency: float = 3.5e9   # Hz

    # ---- spatial layers ----
    l_max: int = 4                 # max co-scheduled streams per RBG

    # ---- channel (TR38.901 UMi system-level) ----
    scenario: str = "umi"
    o2i_model: str = "low"
    enable_pathloss: bool = True
    enable_shadow_fading: bool = True
    indoor_probability: float = 0.0    # Phase 1: all-outdoor (clean SNR spread)
    ue_speed_kmh: float = 3.0          # Phase-1 sanity-check default
    speed_ablation_kmh: tuple = (3.0, 10.0, 30.0, 60.0)  # stale-CSI experiment
    ue_speed_mix: tuple = ()           # if non-empty: per-UE heterogeneous speeds
                                       # (km/h), tiled to num_ue & shuffled per
                                       # episode -> overrides scalar ue_speed_kmh

    # ---- codebook ----
    pmi_mode: str = "type2_sparse_56bit"   # 'random_unit_norm' | 'type2_sparse_56bit'

    # random_unit_norm codebook
    codebook_size: int = 256       # M codewords

    # type2_sparse_56bit codebook (simplified Type-II-like sparse atomic feedback)
    type2_O1: int = 4              # horizontal DFT oversampling factor
    type2_O2: int = 4              # vertical DFT oversampling factor
    type2_L: int = 4               # sparse atoms per (UE, RBG)
    type2_amp_bits: int = 3        # amplitude levels = 2^amp_bits (= 8)
    type2_phase_bits: int = 2      # phase levels    = 2^phase_bits (QPSK = 4)

    # ---- CSI feedback (UE-level Bernoulli) ----
    p_csi: float = 0.2
    p_csi_ablation: tuple = (1.0, 0.5, 0.2, 0.1)
    # CQI quantization at the UE report ('continuous' | 'nr4bit').
    # 'nr4bit': floor-snap the SE to the 3GPP TS 38.214 Table 5.2.2.1-3
    # (4-bit, 256QAM) ladder; below the lowest entry -> 0 ("out of range",
    # UE unschedulable on that RBG). Forbidden with pmi_mode='genie'
    # (genie = perfect CSI; quantizing its CQI would be a third world).
    cqi_mode: str = "continuous"

    # ---- traffic (random arrivals, one HOL packet per UE) ----
    p_arrival: float = 0.2
    # Level-2 mixed-arrival (Run4): if p_arrival_max > 0, each episode draws
    # p_arrival ~ U(p_arrival_min, p_arrival_max) from a dedicated RNG stream
    # (see env.reset) -> per-episode traffic intensity decorrelated from
    # n_active, so load must be read from queue observations. 0 = disabled.
    p_arrival_min: float = 0.0
    p_arrival_max: float = 0.0
    p_arrival_ablation: tuple = (0.1, 0.2, 0.4)   # light / medium / heavy
    packet_size_min: int = 4000    # bits, Uniform
    packet_size_max: int = 12000   # bits, Uniform
    deadline_min: int = 5          # slots, Uniform
    deadline_max: int = 30         # slots, Uniform

    # ---- Run4: per-UE multi-packet queue (2026-07-06) ----
    # queue_size = 1 reproduces the Run3 one-packet model BIT-EXACTLY
    # (arrival draw only when the UE is idle -- same RNG consumption).
    # queue_size > 1 enables queue mode: every active UE draws a
    # Bernoulli(p_arrival) arrival EVERY slot (no self-throttling);
    # arrivals beyond the buffer are buffer-overflow drops (penalized
    # like miss, consistent with the retx-overflow precedent).
    # Deadlines tick for ALL queued packets (expiry in queue = miss).
    queue_size: int = 1

    # ---- PHY abstraction ----
    eta_data: float = 1.0          # fraction of REs carrying data
    beta_rate: float = 1.0         # rate backoff for B_tx prediction
    # post_rzf-only cap backoff: btx_cap = la_beta * eta * N_RE * beta_rate
    # * log2(1 + SINR_pred). Calibrated scheduler-independently to a
    # first-ACK target (90%) under imperfect CSI; genie keeps 1.0
    # (prediction exact). Separate from beta_rate so legacy SU-CQI
    # features/masks are untouched by calibration.
    la_beta: float = 1.0
    # depth-wise backoff beta_m (2026-07-13, audit round 6): when non-empty
    # (must have l_max entries), the planner uses la_beta_by_depth[m-1] for a
    # FINAL group of size m instead of the scalar la_beta -- every spatial
    # mode gets the same first-ACK target (per-rank OLLA analogue). Values
    # from the scheduler-independent calibration (per-depth 10th pct of
    # MI_actual/cap_pred): (0.979, 0.723, 0.646, 0.590) at the Run4 queue op
    # point. Scalar la_beta stays as the global-beta ABLATION mode.
    la_beta_by_depth: tuple = ()
    b_tx_epsilon: float = 1.0      # bits; predicted B_tx below this -> no unit
    # m-aware link adaptation (2026-07-10, external-audit C-cluster ablation):
    # False (default/historical) = B_tx sized from the full-power SU CQI even
    # when m>1 co-scheduled streams share the RBG power -> depth>=2 first
    # transmissions are structurally NACKed (no OLLA; an anti-MU landscape).
    # True = env de-rates B_tx by the planned stream count at unit creation:
    # SE_m = log2(1 + (2^CQI - 1)/m). Scheduler-side predictions/obs unchanged.
    # (Equivalent to la_mode="snr_m"; kept for the QueueMixedFairLA run's CLI.)
    mu_aware_la: bool = False
    # Link-adaptation mode (2026-07-13 redesign, supersedes mu_aware_la):
    #   "legacy"   B_tx from full-power SU CQI (all historical results)
    #   "snr_m"    power-split de-rate only (== mu_aware_la=True)
    #   "post_rzf" B_tx from the PREDICTED post-RZF SINR of the FINAL RBG
    #              group (same precoder/alpha/power as the actual tx, computed
    #              from h_hat only) -- requires decode_order="rbg_major" so
    #              scheduler budget accounting can match env commits exactly.
    # "" = derive from mu_aware_la (backward compatible).
    la_mode: str = ""
    # Position traversal order for scheduling + unit creation:
    #   "layer_major"  l0:r0..7, l1:r0..7, ... (historical; bit-exact default)
    #   "rbg_major"    r0:l0..3, r1:l0..3, ... (groups close per-RBG; required
    #                  by post_rzf, optional order-effect control for legacy)
    decode_order: str = "layer_major"

    def resolved_la_mode(self) -> str:
        if self.la_mode:
            return self.la_mode
        return "snr_m" if self.mu_aware_la else "legacy"

    def validate_la(self) -> None:
        m = self.resolved_la_mode()
        if m not in ("legacy", "snr_m", "post_rzf"):
            raise ValueError(f"unknown la_mode {m!r}")
        if self.decode_order not in ("layer_major", "rbg_major"):
            raise ValueError(f"unknown decode_order {self.decode_order!r}")
        if self.cqi_mode not in ("continuous", "nr4bit"):
            raise ValueError(f"unknown cqi_mode {self.cqi_mode!r}")
        if self.pmi_mode == "genie" and self.cqi_mode != "continuous":
            raise ValueError("pmi_mode='genie' requires cqi_mode='continuous' "
                             "(genie is the perfect-CSI world)")
        if m == "post_rzf" and self.decode_order != "rbg_major":
            raise ValueError(
                "la_mode='post_rzf' requires decode_order='rbg_major': under "
                "layer-major traversal the final RBG group is only known after "
                "all layers, so exact packet-budget accounting would need "
                "retroactive refunds (the very mismatch this mode removes)")
        if self.la_beta_by_depth:
            if len(self.la_beta_by_depth) != self.l_max:
                raise ValueError(
                    f"la_beta_by_depth needs {self.l_max} entries "
                    f"(one per group size), got {self.la_beta_by_depth!r}")
            if any(not (0.0 < b <= 1.5) for b in self.la_beta_by_depth):
                raise ValueError(
                    f"la_beta_by_depth values out of range: "
                    f"{self.la_beta_by_depth!r}")

    # ---- power / noise ----
    p_total: float = 8.0           # normalized total BS power (P_r = 1 per RBG)
    target_snr_db: float = 10.0    # operating SNR (calibration target)
    target_snr_ablation_db: tuple = (5.0, 10.0, 15.0)
    noise_mode: str = "calibration"   # 'calibration' | 'simple'
    rzf_alpha_mode: str = "noise"     # RZF regularization alpha = sigma^2

    # ---- transmission / retransmission ----
    max_retx: int = 4              # retransmissions after initial tx -> 5 total

    # ---- reward (two-time-scale) ----
    lambda_s: float = 1.0          # short-term (useful MI) weight
    lambda_c: float = 1.0          # completion reward weight
    lambda_m: float = 2.0          # miss/drop penalty weight
    eta_d: float = 1.0             # deadline-urgency weight
    b_norm: float = 8000.0         # reward normalization (avg packet size)

    # ---- fairness ----
    t_c: int = 100                 # EWMA window for average throughput

    # ---- baseline schedulers ----
    sus_ortho_threshold: float = 0.5   # SUS: min OrthoScore to co-schedule
    pf_epsilon: float = 1.0            # PF: floor on avg throughput [bits]

    # ---- PPO normalization constants (Phase 2) ----
    cqi_norm_const: float = 8.0        # CQI_fb / cqi_norm_const  in encoder input
    age_norm_max: float = 50.0         # min(Age, age_norm_max) / age_norm_max
    # D_max = deadline_max (= 30 by default)

    # ---- PPO hyperparameters (Phase 2) ----
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_eps: float = 0.2
    ppo_learning_rate: float = 3e-4
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 128      # debug default; main run uses 256

    # ---- PPO stability / critic upgrades (2026-07-02) ----
    # Both default OFF so live runs whose wrappers re-import edited code on
    # auto-resume keep bit-identical behavior; enable per-run via CLI.
    ppo_target_kl: float = 0.0     # >0 enables KL early-stop: a minibatch
                                   # k3-KL above 1.5x this value stops the
                                   # update's remaining passes (collapse guard;
                                   # clip alone does not bound aggregate drift)
    ppo_critic_v2: bool = False    # value-input v2: append 27 structured
                                   # features (deadline histogram, backlog
                                   # totals, retx-grid occupancy, CQI/age
                                   # aggregates, slot phase) -> ValueHead
                                   # input 134 -> 161

    # ---- PPO model architecture (Phase 2) ----
    encoder_hidden: int = 128
    encoder_out_dim: int = 64
    score_net_hidden: tuple = (128, 64)
    no_user_head_hidden: tuple = (128, 64)
    value_head_hidden: tuple = (128, 64)
    share_critic_encoder: bool = True   # Phase 2 v1: actor & critic share encoder

    # ---- PPO training schedule ----
    ppo_eval_every: int = 10            # PPO updates between eval rollouts
    ppo_eval_episodes: int = 3
    ppo_save_every: int = 10            # PPO updates between checkpoint saves

    # ---- episode ----
    episode_len_main: int = 1000
    episode_len_debug: int = 200
    debug: bool = False            # True -> use episode_len_debug

    # ----------------------------------------------------------------
    # derived quantities
    # ----------------------------------------------------------------
    @property
    def n_re_rbg(self) -> int:
        """Resource elements per RBG = 8 RB x 12 SC x 14 sym = 1344."""
        return self.num_rb_per_rbg * self.num_sc_per_rb * self.num_sym_per_rb

    @property
    def num_positions(self) -> int:
        """Total scheduling positions per slot = R x L_max = 32."""
        return self.num_rbg * self.l_max

    @property
    def rbg_bandwidth(self) -> float:
        """Bandwidth of one RBG [Hz] (~2.88 MHz)."""
        return self.num_rb_per_rbg * self.num_sc_per_rb * self.subcarrier_spacing

    @property
    def total_bandwidth(self) -> float:
        """Total system bandwidth [Hz]."""
        return self.num_rbg * self.rbg_bandwidth

    @property
    def slot_duration(self) -> float:
        """Slot duration [s] from NR numerology (SCS 30 kHz -> 0.5 ms)."""
        mu = int(round(np.log2(self.subcarrier_spacing / 15e3)))
        return 1e-3 / (2 ** mu)

    @property
    def p_rbg(self) -> float:
        """Transmit power budget per RBG (equal split of P_total)."""
        return self.p_total / self.num_rbg

    @property
    def num_attempts(self) -> int:
        """Total transmission attempts allowed per unit = 1 + max_retx."""
        return 1 + self.max_retx

    @property
    def episode_len(self) -> int:
        return self.episode_len_debug if self.debug else self.episode_len_main

    @property
    def ue_speed_ms(self) -> float:
        """UE speed in m/s."""
        return self.ue_speed_kmh / 3.6

    @property
    def avg_packet_size(self) -> float:
        return 0.5 * (self.packet_size_min + self.packet_size_max)

    @property
    def n_pol(self) -> int:
        return 2 if self.bs_polarization == "dual" else 1

    @property
    def type2_n_beams(self) -> int:
        return (self.bs_ant_rows * self.type2_O1
                * self.bs_ant_cols * self.type2_O2)

    @property
    def type2_n_atoms(self) -> int:
        return self.type2_n_beams * self.n_pol

    @property
    def type2_payload_bits(self) -> int:
        """Per-(UE,RBG) feedback payload in bits (PMI only, not CQI)."""
        atom_bits = int(np.ceil(np.log2(self.type2_n_atoms)))
        return self.type2_L * (atom_bits + self.type2_amp_bits
                               + self.type2_phase_bits)

    def describe(self) -> str:
        """Human-readable summary for sanity checks / logging."""
        lines = [
            "=== ML DRL Scheduler -- Config ===",
            f"  cell        : 1 BS ({self.num_bs_ant} ant) x {self.num_ue} UE "
            f"({self.num_ue_ant} ant), DL",
            f"  resource    : {self.num_rbg} RBG x {self.l_max} layers "
            f"= {self.num_positions} positions",
            f"  RE per RBG  : {self.n_re_rbg}",
            f"  bandwidth   : RBG {self.rbg_bandwidth/1e6:.2f} MHz, "
            f"total {self.total_bandwidth/1e6:.2f} MHz",
            f"  slot        : {self.slot_duration*1e3:.3f} ms",
            f"  channel     : TR38.901 {self.scenario.upper()}, "
            f"UE speed {self.ue_speed_kmh:.1f} km/h",
            f"  codebook    : {self.pmi_mode}"
            + (f", M={self.codebook_size}"
               if self.pmi_mode == "random_unit_norm" else
               f", L={self.type2_L}, atoms={self.type2_n_atoms}, "
               f"payload={self.type2_payload_bits} bit/(UE,RBG)"),
            f"  feedback    : p_csi={self.p_csi}, cqi={self.cqi_mode}",
            f"  traffic     : p_arrival={self.p_arrival}, "
            f"size U[{self.packet_size_min},{self.packet_size_max}] bits, "
            f"deadline U[{self.deadline_min},{self.deadline_max}] slots",
            f"  power/noise : P_total={self.p_total}, P_r={self.p_rbg}, "
            f"target SNR {self.target_snr_db} dB ({self.noise_mode})",
            f"  retx        : max_retx={self.max_retx} "
            f"({self.num_attempts} attempts total)",
            f"  reward      : lambda_s/c/m={self.lambda_s}/{self.lambda_c}/"
            f"{self.lambda_m}, eta_D={self.eta_d}, B_norm={self.b_norm}",
            f"  episode     : {self.episode_len} slots"
            f"{' (debug)' if self.debug else ''}",
        ]
        return "\n".join(lines)


def debug_config(**overrides) -> Config:
    """Short-episode config for smoke tests."""
    cfg = Config(debug=True, **overrides)
    return cfg


def phase2_debug_config(**overrides) -> Config:
    """Phase 2 PPO debug preset: small K, short episode, small minibatch."""
    base = dict(debug=True, num_ue=8, episode_len_debug=200,
                ppo_minibatch_size=128)
    base.update(overrides)
    return Config(**base)


def phase2_main_config(**overrides) -> Config:
    """Phase 2 PPO main preset: full K, long episode, larger minibatch."""
    base = dict(debug=False, num_ue=16, episode_len_main=1000,
                ppo_minibatch_size=256)
    base.update(overrides)
    return Config(**base)


# --- run #2 "hard" operating point ----------------------------------------
# Probes (headroom + oracle speed sweep) showed the default/easy point has ~0
# headroom above the heuristics (SUS+PF is already near the fresh-CSI ceiling),
# so PPO can at best tie the baselines. This preset moves to a point with real
# learnable room: heavier load (contention), tight deadlines (prioritization
# matters), and faster UEs so CSI staleness actually bites (heuristics ignore
# Age; a learned policy can avoid stale-CSI UEs) -> oracle gap ~23% at 30 km/h.
# value_coef lowered to pair with the value-normalization + separate-clip fixes.
_HARD = dict(p_arrival=0.4, deadline_min=3, deadline_max=12,
             ue_speed_kmh=30.0, ppo_value_coef=0.25)


def phase2_hard_debug_config(**overrides) -> Config:
    """Run #2 confirmation preset: K=16 hard point, short episodes (fast)."""
    base = dict(debug=True, num_ue=16, episode_len_debug=300,
                ppo_minibatch_size=128, **_HARD)
    base.update(overrides)
    return Config(**base)


def phase2_hard_main_config(**overrides) -> Config:
    """Run #2 main preset: K=16 hard point, full-length episodes."""
    base = dict(debug=False, num_ue=16, episode_len_main=1000,
                ppo_minibatch_size=256, **_HARD)
    base.update(overrides)
    return Config(**base)


def phase4_queue_config(**overrides) -> Config:
    """Run4 preset: MixedSpeed_L2b point + per-UE multi-packet queue.

    Changes vs the L2b point: queue_size 1->8 (multi-packet, finite buffer),
    p_arrival 0.4->0.22 (recalibrated so nominal rho ~= 0.8 at mean load
    n_active=24 against C_ref ~52 kbit/slot; per-episode rho then spans
    ~0.53 (n=16) to ~1.07 (n=32) -- the mixed-congestion spectrum).
    sus_ortho_threshold 0.75: re-swept at THIS operating point (paired,
    7 thresholds x SUS+CQI/SUS+MW x 10 seeds, Run4/_calib_20260706/) --
    flat optimum over 0.70-0.80, argmax 0.75; grid default 0.5 loses ~-125.
    Run3 recipe lessons ON by default: KL guard 0.02 + critic v2.
    NOTE (2026-07-08): entropy stays at the cfg default 0.01 -- Run3 runs
    used 0.02, but changing it here was reverted by user decision; the
    0.01-vs-0.02 question is measured empirically by the QueueFineTune
    (0.01) vs QueueFineTuneEnt02 (0.02) A/B fork instead."""
    base = dict(debug=False, num_ue=32, episode_len_main=1000,
                ppo_minibatch_size=256,
                n_active_min=16, n_active_max=32,
                ue_speed_min=5.0, ue_speed_max=30.0,
                p_csi=0.6,
                deadline_min=3, deadline_max=12,
                queue_size=8, p_arrival=0.22,
                sus_ortho_threshold=0.75,
                ppo_value_coef=0.25,
                ppo_target_kl=0.02, ppo_critic_v2=True)
    base.update(overrides)
    return Config(**base)


def phase2_hetero_config(**overrides) -> Config:
    """Stage-A heterogeneous-mobility preset (Run3): K=16 split 4x{5,10,15,30}
    km/h (per-UE speed shuffled each episode), p_csi=0.6, 1000-slot episodes.
    Tests per-user selective pairing under MIXED CSI reliability. Keeps the
    hard load/deadline point but replaces the scalar 30 km/h with a mix; no
    fairness term and still one-packet queue (that comes in later stages)."""
    base = dict(debug=False, num_ue=16, episode_len_main=1000,
                ppo_minibatch_size=256,
                ue_speed_mix=(5.0, 10.0, 15.0, 30.0),
                p_csi=0.6,
                p_arrival=0.4, deadline_min=3, deadline_max=12,
                ppo_value_coef=0.25)
    base.update(overrides)
    return Config(**base)


if __name__ == "__main__":
    print(Config().describe())
