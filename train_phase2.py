"""
train_phase2.py -- PPO main training loop for ML DRL Scheduler Phase 2.

Schedule:
  - one episode per PPO update (v1, num_envs=1)
  - eval every cfg.ppo_eval_every updates against all baselines
  - save best (eval reward) + latest checkpoint
  - logs to TensorBoard + CSV under runs/<run_name>/

Usage:
  python train_phase2.py --mode debug                    # K=8, 200-slot smoke
  python train_phase2.py --mode main                     # K=16, 1000-slot main
  python train_phase2.py --mode debug --num_updates 5    # quick smoke
  python train_phase2.py --mode main \
      --resume runs/<name>/ckpt/latest.pt                # continue a run

Lifecycle:
  - Ctrl+C (or normal completion) always leaves a usable latest.pt at the
    last COMPLETED update; checkpoints are written atomically.
  - a fresh run refuses to reuse a run_dir that already contains results;
    use --resume (CSV logs are then appended, config.json kept).
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime
import json
import os
import signal
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import (Config, phase2_debug_config, phase2_main_config,
                    phase2_hard_debug_config, phase2_hard_main_config,
                    phase2_hetero_config, phase4_queue_config)
from env import SchedulerEnv
from policy import ActorCritic
from ppo import collect_rollout, ppo_update
from baselines import all_baselines
from metrics import jains_index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["debug", "main", "hard_debug", "hard_main",
                            "hetero", "queue"],
                   default="debug",
                   help="hard_* = run #2 operating point (load+deadline+30km/h); "
                        "hetero = Run3 mixed-mobility 4x{5,10,15,30} + p_csi 0.6; "
                        "queue = Run4 multi-packet queue (L2b point + queue_size 8"
                        " + p_arrival 0.22 + KL guard + critic v2)")
    p.add_argument("--num_updates", type=int, default=None,
                   help="default: debug/hard_debug=250, main/hard_main=2000, "
                        "hetero=1500")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--run_root", type=str, default="runs",
                   help="parent dir for run_dir (e.g. Run3 to save under Run3/)")
    p.add_argument("--patience_evals", type=int, default=0,
                   help="early stop after this many eval rounds with no best "
                        "improvement (0 = disabled). hetero default = 15")
    p.add_argument("--patience_min_updates", type=int, default=100,
                   help="no-improvement evals are not COUNTED before this "
                        "update: on a fresh run the first eval (near-random "
                        "policy, lucky seed) sets the best-watermark and the "
                        "critic warm-up dip then eats the patience budget "
                        "(QueueMain reached 14/15 before breaking through)")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--p_csi", type=float, default=None,
                   help="override CSI-feedback probability (e.g. 1.0 = always "
                        "fresh CSI); for the MU/SU adaptation ablation")
    p.add_argument("--eval_every", type=int, default=None,
                   help="override eval cadence (e.g. 1 = eval every update, to "
                        "inspect the fine-grained early learning curve)")
    p.add_argument("--ue_speed_kmh", type=float, default=None,
                   help="override: uniform UE speed (km/h); clears ue_speed_mix "
                        "(e.g. 10 = all UEs 10 km/h instead of the hetero mix)")
    p.add_argument("--entropy_coef", type=float, default=None,
                   help="override ppo_entropy_coef (exploration strength); "
                        "e.g. 0.05 = 5x default for more pairing exploration")
    p.add_argument("--entropy_coef_final", type=float, default=None,
                   help="linearly anneal the entropy coef from its initial "
                        "value down to this over --entropy_decay_updates "
                        "(exploration early, exploitation late)")
    p.add_argument("--entropy_decay_updates", type=int, default=None,
                   help="updates to reach --entropy_coef_final (then held); "
                        "default = num_updates. Schedule is a pure function "
                        "of the update index, so --resume stays on-schedule")
    p.add_argument("--num_ue", type=int, default=None,
                   help="override number of UEs K (e.g. 32 = more contention "
                        "for the ~8 SU slots/slot; scarcity experiment)")
    p.add_argument("--p_arrival", type=float, default=None,
                   help="override packet-arrival probability per active UE per "
                        "slot (e.g. 0.33 = high-load queue regime vs preset 0.22)")
    p.add_argument("--p_arrival_min", type=float, default=None,
                   help="Level-2 mixed-arrival: min per-episode p_arrival "
                        "(needs --p_arrival_max)")
    p.add_argument("--p_arrival_max", type=float, default=None,
                   help="Level-2 mixed-arrival: max per-episode p_arrival; "
                        "enables p ~ U(min,max) drawn per episode so traffic "
                        "intensity is decorrelated from n_active")
    p.add_argument("--mu_aware_la", action="store_true",
                   help="fair link adaptation: env de-rates B_tx by the planned "
                        "co-scheduled stream count (SE_m = log2(1+(2^CQI-1)/m)); "
                        "default off = historical SU-CQI sizing")
    p.add_argument("--la_mode", type=str, default=None,
                   choices=["legacy", "snr_m", "post_rzf"],
                   help="link-adaptation mode (2026-07-13 redesign); post_rzf "
                        "sizes B_tx from the predicted post-RZF SINR of the "
                        "final RBG group and requires --decode_order rbg_major")
    p.add_argument("--decode_order", type=str, default=None,
                   choices=["layer_major", "rbg_major"],
                   help="position traversal order (default layer_major = "
                        "historical bit-exact)")
    p.add_argument("--la_beta", type=float, default=None,
                   help="post_rzf cap backoff (scheduler-independent 90%% "
                        "first-ACK calibration = 0.6469 at the Run4 queue op "
                        "point; genie worlds use 1.0). Scalar = global-beta "
                        "ablation mode; official runs use --la_beta_by_depth")
    p.add_argument("--la_beta_by_depth", type=str, default=None,
                   help="comma-separated depth-wise beta_m (l_max values), "
                        "e.g. '0.9815,0.7306,0.6466,0.5922' -- per-group-size "
                        "90%% first-ACK target (overrides scalar la_beta)")
    p.add_argument("--allow_dirty", action="store_true",
                   help="allow a FRESH run to start with uncommitted .py "
                        "changes (throwaway experiments only; official runs "
                        "must start from a clean, pushed commit)")
    p.add_argument("--deadline_min", type=int, default=None,
                   help="override deadline_min (slots); tighter = urgency scarcity")
    p.add_argument("--deadline_max", type=int, default=None,
                   help="override deadline_max (slots); e.g. 2/6 vs default 3/12")
    p.add_argument("--n_active_min", type=int, default=None,
                   help="Level-2 mixed-load: min active UEs/episode (needs --n_active_max)")
    p.add_argument("--n_active_max", type=int, default=None,
                   help="Level-2 mixed-load: max active UEs/episode; enables random "
                        "n_active in [min,max] per episode to test online SU/MU adaptation")
    p.add_argument("--ue_speed_min", type=float, default=None,
                   help="continuous per-UE uniform speed lower bound (km/h)")
    p.add_argument("--ue_speed_max", type=float, default=None,
                   help="continuous per-UE uniform speed upper bound (km/h); >0 enables "
                        "U(min,max) per-UE speed each episode (overrides mix/uniform)")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint to resume from "
                        "(e.g. runs/<name>/ckpt/latest.pt); run_dir is "
                        "derived from the path, CSV logs are appended")
    p.add_argument("--pmi_mode", type=str, default=None,
                   choices=["type2_sparse_56bit", "random_unit_norm", "genie"],
                   help="CSI codebook: 'genie' = perfect CSI (h_hat==h_true, no "
                        "quantization; pair with --p_csi 1.0 for zero staleness)")
    p.add_argument("--cqi_mode", type=str, default=None,
                   choices=["continuous", "nr4bit"],
                   help="CQI report quantization: 'nr4bit' floor-snaps the SE "
                        "to the 3GPP 4-bit 256QAM ladder (TS 38.214 Table "
                        "5.2.2.1-3); index 0 = out of range. Incompatible "
                        "with --pmi_mode genie.")
    p.add_argument("--ppo_save_every", type=int, default=None,
                   help="checkpoint cadence in updates; new official runs "
                        "pass 1 (every update; ~1.2MB overwrite, negligible "
                        "cost) so kills never redo work")
    p.add_argument("--target_kl", type=float, default=None,
                   help="override ppo_target_kl (>0 enables KL early-stop: "
                        "stop an update's remaining passes when minibatch "
                        "KL > 1.5x this; e.g. 0.02). Default: disabled")
    p.add_argument("--critic_v2", action="store_true",
                   help="enable value-input v2 (+27 structured features, "
                        "ValueHead input 134->161); breaks value_head ckpt "
                        "compat -> combine with --init_from for warm starts")
    p.add_argument("--init_from", type=str, default=None,
                   help="fresh run warm-started from a checkpoint: loads all "
                        "matching-shape tensors EXCEPT value_head and the "
                        "return normalizer (actor warm-start, fresh critic + "
                        "fresh optimizer). Mutually exclusive with --resume")
    p.add_argument("--batched_replay", action="store_true",
                   help="batch the PPO-update replay (decode caches each "
                        "slot's head inputs; ~1e-6 equivalent, NOT "
                        "bit-identical -- do not use to reproduce a run)")
    p.add_argument("--force_full_rank", action="store_true",
                   help="ablation: remove the early-close (no-user) option -- "
                        "every free position with a valid candidate must be "
                        "filled; adaptive rank comes only from budget/epsilon "
                        "exhaustion. NoUserHead is kept but masked")
    p.add_argument("--batch_verify_every", type=int, default=None,
                   help="cadence of the batched-replay cross-check against "
                        "sequential replay (default 25; 0 disables). A failure "
                        "does not raise: it logs and falls back to sequential")
    return p.parse_args()


def atomic_save(obj, path: str) -> None:
    """torch.save via a temp file + os.replace: a kill mid-write cannot
    corrupt the previous checkpoint at `path`."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class PPOScheduler:
    """Thin baseline-like wrapper around ActorCritic for eval comparisons."""
    name = "PPO"

    def __init__(self, ac, deterministic: bool = True):
        self.ac = ac
        self.deterministic = deterministic

    def schedule(self, env) -> np.ndarray:
        with torch.no_grad():
            out = self.ac.decode(env.get_observation(),
                                  deterministic=self.deterministic)
        return out["action_sequence"]


def _git_state() -> tuple:
    """(commit hash, tracked-*.py-dirty flag) of the repo this file lives in.

    Returns ("no-git"/"unknown", False) when git is unavailable -- stamping
    must never break training.
    """
    import subprocess
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=10)
        if h.returncode != 0:
            return "no-git", False
        d = subprocess.run(["git", "status", "--porcelain", "--", "*.py"],
                           cwd=root, capture_output=True, text=True,
                           timeout=10)
        return h.stdout.strip(), bool(d.stdout.strip())
    except Exception:
        return "unknown", False


def env_episode_metrics(env, cfg: Config) -> dict:
    ep = env.ep
    arrivals = max(ep["n_arrivals"], 1)
    ep_time = cfg.episode_len * cfg.slot_duration
    mean_sinr_db = (10.0 * np.log10(ep["sinr_sum"] / ep["sinr_count"])
                    if ep["sinr_count"] else float("nan"))
    dir_corr = (ep["dir_corr_sum"] / max(ep["dir_corr_count"], 1))
    ss = np.asarray(ep["sched_slots_per_ue"], dtype=float)
    pdep = np.asarray(ep["paired_depth_sum_per_ue"], dtype=float)
    act = ss > 0
    mu_depth = float(np.mean(pdep[act] / ss[act])) if act.any() else 0.0
    out = dict(
        reward=ep["reward"],
        throughput_mbps=ep["acked_bits"] / ep_time / 1e6,
        completion_rate=ep["n_comp"] / arrivals,
        deadline_miss_rate=ep["n_miss_deadline"] / arrivals,
        retx_drop_rate=ep["n_retx_drop"] / arrivals,
        retx_overflow_drop=int(ep.get("n_retx_overflow_drop", 0)),
        mean_sinr_db=mean_sinr_db,
        direction_corr=dir_corr,
        # JFI over ACTIVE UEs only (standing metric rule; active = [0, n_active))
        jain=jains_index(env.cum_acked_bits[:env.traffic.n_active]),
        n_arrivals=int(ep["n_arrivals"]),
        n_active=int(env.traffic.n_active),          # this episode's active-UE count
        mean_speed_kmh=float(np.mean(env.ue_speeds_kmh)),
        mu_depth=mu_depth,                           # realized active-only MU-depth
    )
    if cfg.queue_size > 1:                           # Run4 queue metrics
        d = ep["delays"]
        offered = ep["n_arrivals"] + ep["n_buffer_overflow"]
        out["mean_delay"] = float(np.mean(d)) if d else 0.0
        out["p95_delay"] = float(np.percentile(d, 95)) if d else 0.0
        out["mean_qlen"] = ep["qlen_sum"] / max(ep["qlen_count"], 1)
        out["buffer_overflow_rate"] = ep["n_buffer_overflow"] / max(offered, 1)
    if cfg.p_arrival_max > 0:                        # Level-2 mixed-arrival
        out["p_arrival_ep"] = float(env.traffic.p_arrival_ep)
    if cfg.resolved_la_mode() == "post_rzf":         # HARQ/LA instrumentation
        # (post_rzf runs only: legacy/live CSVs keep their historical header)
        nu = max(ep["n_units_new"], 1)
        out["first_ack_rate"] = ep["n_units_first_ack"] / nu
        out["attempts_per_acked"] = (ep["attempts_acked_sum"]
                                     / max(ep["n_units_acked"], 1))
        out["pinned_fraction"] = (ep["pinned_pos_sum"]
                                  / max(ep["sched_pos_sum"], 1))
        out["goodput_mbps"] = ep["completed_bits"] / ep_time / 1e6
        fa, ub = ep["first_ack_by_depth"], ep["units_by_depth"]
        for m in range(4):
            # raw numerator/denominator ALWAYS stored so multi-episode
            # aggregation can pool sum(acks)/sum(units); the per-episode
            # rate is NaN (not 0) when the depth bin is empty -- averaging
            # per-episode rates over episodes reproduces the 2026-07-13
            # empty-bin artifact (m1 "36%") and must not be done.
            out[f"acks_m{m + 1}"] = int(fa[m])
            out[f"units_m{m + 1}"] = int(ub[m])
            out[f"first_ack_m{m + 1}"] = (float(fa[m]) / int(ub[m])
                                          if ub[m] > 0 else float("nan"))
    return out


def run_eval(env, ac, cfg: Config, update: int,
             writer: SummaryWriter, csv_w: csv.writer,
             per_ue_csv: csv.writer = None,
             log_baselines: bool = True) -> float:
    """Evaluate PPO (argmax) + baselines, log to TB + CSV. Returns PPO mean reward.

    Baselines are deterministic on the fixed held-out eval seeds (identical
    every eval), so they are run + logged only when ``log_baselines`` is True
    (the first eval of a fresh run); afterwards only PPO is evaluated/logged.

    If ``per_ue_csv`` is given, PPO's per-UE metrics (with the episode's per-UE
    speed) are logged each eval episode for per-user / per-speed analysis."""
    print(f"  --- eval at update {update} ---")
    schedulers = [PPOScheduler(ac, deterministic=True)]
    if log_baselines:
        schedulers += all_baselines(cfg)
    summary = {}
    for sched in schedulers:
        rewards, throughputs, goodputs, comps = [], [], [], []
        misses, totfails, depths = [], [], []
        for ep_idx in range(cfg.ppo_eval_episodes):
            env.reset(ep_idx + 10000)  # held-out seed range
            done = False
            while not done:
                alloc = sched.schedule(env)
                _, _, done, _ = env.step(alloc)
            if per_ue_csv is not None and sched.name == "PPO":
                ep = env.ep
                sl = np.maximum(ep["sched_slots_per_ue"], 1)
                depth = ep["paired_depth_sum_per_ue"] / sl
                for u in range(cfg.num_ue):
                    per_ue_csv.writerow([
                        update, ep_idx, u,
                        round(float(env.ue_speeds_kmh[u]), 2),
                        round(float(ep["acked_per_ue"][u]), 1),
                        int(ep["comp_per_ue"][u]), int(ep["miss_per_ue"][u]),
                        int(ep["retx_per_ue"][u]),
                        round(float(depth[u]), 3)
                        if ep["sched_slots_per_ue"][u] else 0.0])
            m = env_episode_metrics(env, cfg)
            rewards.append(m["reward"])
            throughputs.append(m["throughput_mbps"])
            goodputs.append(m.get("goodput_mbps", float("nan")))  # post_rzf only
            comps.append(m["completion_rate"])
            misses.append(m["deadline_miss_rate"])
            totfails.append(m["deadline_miss_rate"] + m["retx_drop_rate"])
            depths.append(m["mu_depth"])
        rew = float(np.mean(rewards))
        thr = float(np.mean(throughputs))
        good = float(np.mean(goodputs))
        comp = float(np.mean(comps))
        miss = float(np.mean(misses))
        totfail = float(np.mean(totfails))
        depth = float(np.mean(depths))
        print(f"    {sched.name:12s}  reward {rew:8.2f}  thrpt {thr:6.2f}Mbps  "
              f"comp {comp:.3f}  totfail {totfail:.3f}  depth {depth:.2f}")
        csv_w.writerow([update, sched.name, rew, thr, good, comp,
                        miss, totfail, depth])
        writer.add_scalar(f"eval/{sched.name}/reward", rew, update)
        writer.add_scalar(f"eval/{sched.name}/throughput_mbps", thr, update)
        writer.add_scalar(f"eval/{sched.name}/goodput_mbps", good, update)
        writer.add_scalar(f"eval/{sched.name}/completion_rate", comp, update)
        writer.add_scalar(f"eval/{sched.name}/deadline_miss_rate", miss, update)
        writer.add_scalar(f"eval/{sched.name}/total_failure_rate", totfail, update)
        writer.add_scalar(f"eval/{sched.name}/mu_depth", depth, update)
        summary[sched.name] = rew
    return summary["PPO"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.mode == "debug":
        cfg = phase2_debug_config(seed=args.seed)
        default_updates = 250                                   # ~50k slots
    elif args.mode == "hard_debug":
        cfg = phase2_hard_debug_config(seed=args.seed)
        default_updates = 250
    elif args.mode == "hard_main":
        cfg = phase2_hard_main_config(seed=args.seed)
        default_updates = 2000
    elif args.mode == "hetero":
        cfg = phase2_hetero_config(seed=args.seed)
        default_updates = 1500
    elif args.mode == "queue":
        cfg = phase4_queue_config(seed=args.seed)
        default_updates = 1500
    else:
        cfg = phase2_main_config(seed=args.seed)
        default_updates = 2000                                  # ~2M slots
    if args.p_csi is not None:                       # ablation override
        cfg.p_csi = args.p_csi
    if args.pmi_mode is not None:                     # CSI codebook (genie etc.)
        cfg.pmi_mode = args.pmi_mode
    if args.cqi_mode is not None:                    # CQI report quantization
        cfg.cqi_mode = args.cqi_mode
    if args.eval_every is not None:                  # fine-grained eval cadence
        cfg.ppo_eval_every = args.eval_every
    if args.ue_speed_kmh is not None:                # uniform-speed override
        cfg.ue_speed_mix = ()
        cfg.ue_speed_kmh = args.ue_speed_kmh
    if args.entropy_coef is not None:                # exploration override
        cfg.ppo_entropy_coef = args.entropy_coef
    if args.ppo_save_every is not None:              # ckpt cadence (2026-07-15
        cfg.ppo_save_every = args.ppo_save_every     # decision: new runs use 1)
    if args.num_ue is not None:                      # scarcity: more UEs (K)
        cfg.num_ue = args.num_ue
    if args.p_arrival is not None:                   # load axis (Run4 HighLoad)
        cfg.p_arrival = args.p_arrival
    if args.p_arrival_min is not None:               # Level-2 mixed-arrival
        cfg.p_arrival_min = args.p_arrival_min
    if args.p_arrival_max is not None:
        cfg.p_arrival_max = args.p_arrival_max
    if args.mu_aware_la:                             # fair LA (Run5 axis)
        cfg.mu_aware_la = True
    if args.la_mode is not None:                     # 2026-07-13 LA redesign
        cfg.la_mode = args.la_mode
    if args.decode_order is not None:
        cfg.decode_order = args.decode_order
    if args.la_beta is not None:
        cfg.la_beta = args.la_beta
    if args.la_beta_by_depth is not None:
        cfg.la_beta_by_depth = tuple(
            float(x) for x in args.la_beta_by_depth.split(","))
    cfg.validate_la()                                # fail fast (env re-checks)
    if args.deadline_min is not None:                # scarcity: tighter deadlines
        cfg.deadline_min = args.deadline_min
    if args.deadline_max is not None:
        cfg.deadline_max = args.deadline_max
    if args.n_active_min is not None:                # Level-2 mixed-load
        cfg.n_active_min = args.n_active_min
    if args.n_active_max is not None:
        cfg.n_active_max = args.n_active_max
    if args.ue_speed_max is not None:                # continuous per-UE uniform speed
        cfg.ue_speed_mix = ()
        cfg.ue_speed_max = args.ue_speed_max
        if args.ue_speed_min is not None:
            cfg.ue_speed_min = args.ue_speed_min
    if args.target_kl is not None:                   # KL early-stop guard
        cfg.ppo_target_kl = args.target_kl
    if args.critic_v2:                               # value-input v2 features
        cfg.ppo_critic_v2 = True
    if args.batched_replay:                          # batched PPO update
        cfg.ppo_batched_replay = True
    if args.batch_verify_every is not None:
        cfg.ppo_batch_verify_every = args.batch_verify_every
    if args.force_full_rank:                         # full-rank ablation
        cfg.ppo_force_full_rank = True
    num_updates = args.num_updates or default_updates
    # early stopping: stop after `patience` eval rounds with no best improvement
    # (Run4 lesson from MixedLoad's entropy-exhaustion decay: auto-harvest ON)
    patience = args.patience_evals or (15 if args.mode in ("hetero", "queue")
                                       else 0)
    # training episode indices (= update) must stay below the held-out
    # eval seed range (env.reset(ep_idx + 10000) in run_eval)
    assert num_updates <= 10000, \
        "num_updates > 10000 overlaps the held-out eval seed range (10000+)"

    # run dir
    if args.resume:
        # runs/<name>/ckpt/<file>.pt -> runs/<name>
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
    else:
        if args.run_name is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.mode}_{ts}"
        else:
            run_name = args.run_name
        run_dir = os.path.join(args.run_root, run_name)
        if (os.path.exists(os.path.join(run_dir, "ckpt", "latest.pt"))
                or os.path.exists(os.path.join(run_dir, "csv_logs",
                                               "env_metrics.csv"))):
            raise SystemExit(
                f"run_dir '{run_dir}' already contains results -- refusing "
                f"to overwrite. Continue it with "
                f"--resume {run_dir}/ckpt/latest.pt or pick a new --run_name.")
    os.makedirs(os.path.join(run_dir, "tb_logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "csv_logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

    # --- reproducibility stamp (audit round 6): git hash + py-dirty flag.
    # dirty = uncommitted changes in tracked *.py only (run outputs like
    # csv_logs/ckpt are tracked too and are ALWAYS dirty during live runs).
    # A FRESH official run refuses to start dirty; resume is never blocked
    # (auto-resume wrappers must survive unrelated edits elsewhere).
    git_hash, git_dirty = _git_state()
    print(f"git: {git_hash[:12]}  ({'DIRTY py' if git_dirty else 'clean py'})")
    if args.resume is None and git_dirty and not args.allow_dirty:
        raise SystemExit(
            "refusing to START a fresh run with uncommitted .py changes "
            "(reproducibility); commit/push first or pass --allow_dirty "
            "for throwaway experiments.")

    cfg_dump = {k: v for k, v in cfg.__dict__.items()
                if not k.startswith("_")}
    cfg_dump["git_hash"] = git_hash
    cfg_dump["git_dirty_py"] = git_dirty
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):          # keep the original on resume
        with open(cfg_path, "w") as f:
            json.dump(cfg_dump, f, indent=2, default=str)

    print(cfg.describe())
    print(f"\nrun_dir: {run_dir}\nnum_updates: {num_updates}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env = SchedulerEnv(cfg)
    ac = ActorCritic(cfg).to(device)
    if args.init_from:
        # fresh run, actor warm-start: keep every matching-shape tensor
        # except the value head (probe-confirmed worthless + may change
        # shape under critic_v2) and the return normalizer (re-inits on
        # the first update). Optimizer/RNG/update counter start fresh.
        assert not args.resume, \
            "--init_from is for fresh runs; use --resume to continue a run"
        ck_init = torch.load(args.init_from, map_location=device)
        ck_qs = int((ck_init.get("cfg") or {}).get("queue_size", 1) or 1)
        if ck_qs != int(getattr(cfg, "queue_size", 1)):
            raise SystemExit(
                f"--init_from refused: checkpoint queue_size={ck_qs} vs this "
                f"run queue_size={cfg.queue_size} -- a cross-mode warm-start "
                f"silently leaves the shape-mismatched layers random "
                f"(Run4 design: queue mode trains fresh).")
        own = ac.state_dict()
        keep = {k: v for k, v in ck_init["model"].items()
                if k in own and own[k].shape == v.shape
                and not k.startswith("value_head")
                and not k.startswith("ret_")}
        ac.load_state_dict(keep, strict=False)
        fresh = sorted(set(own) - set(keep))
        print(f"init_from {args.init_from} (ckpt update "
              f"{ck_init.get('update')}, eval "
              f"{ck_init.get('eval_reward', float('nan')):.1f}): "
              f"warm-started {len(keep)} tensors; fresh: {fresh}")
    optimizer = torch.optim.Adam(ac.parameters(), lr=cfg.ppo_learning_rate)
    print(f"actor-critic params: {sum(p.numel() for p in ac.parameters()):,}\n")

    # ---- resume ----
    start_update = 0
    best_eval_reward = -float("inf")
    start_evals_no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ac.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt["update"]) + 1
        best_eval_reward = float(ckpt.get("best_eval_reward", -float("inf")))
        # patience survives restarts (else auto-resume defeats early-stop)
        start_evals_no_improve = int(ckpt.get("evals_no_improve", 0))
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if "cuda_rng_state" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [s.cpu() for s in ckpt["cuda_rng_state"]])
        for k in ("num_ue", "episode_len", "pmi_mode", "cqi_mode", "seed",
                  "queue_size", "p_arrival", "p_arrival_min", "p_arrival_max",
                  "deadline_min", "deadline_max",
                  "n_active_min", "n_active_max",
                  "ue_speed_min", "ue_speed_max", "mu_aware_la",
                  "la_mode", "decode_order", "la_beta", "la_beta_by_depth",
                  # flipping either of these across a resume changes the
                  # update math (batched vs sequential replay; critic input
                  # width) with no other trace in the run's artifacts
                  "ppo_batched_replay", "ppo_critic_v2",
                  "ppo_force_full_rank"):
            ck_v = ckpt.get("cfg", {}).get(k)
            if ck_v is not None and str(ck_v) != str(getattr(cfg, k)):
                print(f"WARNING: cfg mismatch vs checkpoint: {k}: "
                      f"ckpt={ck_v}  current={getattr(cfg, k)}")
        print(f"resumed {args.resume}: start_update={start_update}, "
              f"best_eval_reward={best_eval_reward:.2f}")
        if start_update >= num_updates:
            raise SystemExit(f"nothing to do: checkpoint already at update "
                             f"{start_update - 1}, num_updates={num_updates}")

    writer = SummaryWriter(os.path.join(run_dir, "tb_logs"))
    csv_mode = "a" if args.resume else "w"
    env_f = open(os.path.join(run_dir, "csv_logs", "env_metrics.csv"),
                 csv_mode, newline="")
    ppo_f = open(os.path.join(run_dir, "csv_logs", "ppo_metrics.csv"),
                 csv_mode, newline="")
    eval_f = open(os.path.join(run_dir, "csv_logs", "eval_metrics.csv"),
                  csv_mode, newline="")
    per_ue_f = open(os.path.join(run_dir, "csv_logs", "per_ue_metrics.csv"),
                    csv_mode, newline="")
    env_csv = csv.writer(env_f)
    ppo_csv = csv.writer(ppo_f)
    eval_csv = csv.writer(eval_f)
    per_ue_csv = csv.writer(per_ue_f)

    if not args.resume:
        per_ue_csv.writerow(["update", "ep_idx", "ue", "speed_kmh",
                             "acked_bits", "n_comp", "deadline_miss",
                             "retx_drop", "mean_mu_depth"])
        env_header = ["update", "reward", "throughput_mbps",
                      "completion_rate", "deadline_miss_rate",
                      "retx_drop_rate", "retx_overflow_drop",
                      "mean_sinr_db", "direction_corr", "jain",
                      "n_arrivals", "n_active", "mean_speed_kmh",
                      "mu_depth"]
        if cfg.queue_size > 1:
            env_header += ["mean_delay", "p95_delay", "mean_qlen",
                           "buffer_overflow_rate"]
        if cfg.p_arrival_max > 0:
            env_header += ["p_arrival_ep"]
        if cfg.resolved_la_mode() == "post_rzf":     # HARQ/LA instruments
            env_header += ["first_ack_rate", "attempts_per_acked",
                           "pinned_fraction", "goodput_mbps"]
            for m in (1, 2, 3, 4):
                env_header += [f"acks_m{m}", f"units_m{m}",
                               f"first_ack_m{m}"]
        env_csv.writerow(env_header)
        ppo_csv.writerow(["update", "policy_loss", "value_loss", "entropy_sum",
                           "entropy_mean", "approx_KL", "clip_fraction",
                           "grad_norm", "explained_variance"])
        eval_csv.writerow(["update", "scheduler", "reward", "throughput_mbps",
                            "goodput_mbps", "completion_rate",
                            "deadline_miss_rate", "total_failure_rate",
                            "mu_depth"])

    def ckpt_payload(update_idx: int) -> dict:
        """Checkpoint contents; enough state for a faithful --resume.

        Model/optimizer tensors are CLONED so the payload is a frozen
        snapshot of the calling moment: the interrupt path keeps the last
        update-boundary payload and saves THAT, because a mid-update
        SIGTERM otherwise persists half-stepped weights / mid-update RNG
        under the previous update's label, and the resumed re-run of the
        interrupted update silently diverges from the uninterrupted
        trajectory (verified empirically both ways).
        """
        payload = {"model": {k: v.detach().clone()
                             for k, v in ac.state_dict().items()},
                   "optimizer": copy.deepcopy(optimizer.state_dict()),
                   "update": update_idx,
                   "best_eval_reward": best_eval_reward,
                   "evals_no_improve": evals_no_improve,
                   "torch_rng_state": torch.get_rng_state(),
                   "cfg": cfg_dump}
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        return payload

    # route SIGTERM (docker stop / plain kill) through the same graceful
    # path as Ctrl+C: the except/finally below saves latest.pt + closes logs
    def _sigterm_to_interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm_to_interrupt)

    t0 = time.time()
    last_completed = start_update - 1
    interrupted = False
    evals_no_improve = start_evals_no_improve
    stopped_early = False
    # entropy annealing: linear from the (possibly CLI-overridden) initial
    # coef to --entropy_coef_final over --entropy_decay_updates, then held.
    ent_coef0 = cfg.ppo_entropy_coef
    if args.entropy_coef_final is not None:
        ent_horizon = args.entropy_decay_updates or num_updates
        print(f"entropy anneal: {ent_coef0} -> {args.entropy_coef_final} "
              f"over {ent_horizon} updates (linear, resume-safe)")
    # frozen state entering update start_update; refreshed at every
    # completed-update boundary -- the ONLY payload the interrupt path saves
    boundary_ckpt = ckpt_payload(last_completed)
    # baselines are logged once (first eval of a fresh run); a resumed run
    # already has them in eval_metrics.csv, so from here it logs PPO only.
    baselines_logged = args.resume
    try:
        for update in range(start_update, num_updates):
            # rollout
            ac.eval()
            trajs, last_value = collect_rollout(env, ac, episode_idx=update)

            # env metrics for this rollout episode
            em = env_episode_metrics(env, cfg)
            for k, v in em.items():
                writer.add_scalar(f"env/{k}", v, update)
            env_csv.writerow([update] + list(em.values()))

            # PPO update (entropy coef possibly annealed for this update)
            if args.entropy_coef_final is not None:
                frac = min(update / max(ent_horizon, 1), 1.0)
                cfg.ppo_entropy_coef = (ent_coef0 +
                                        (args.entropy_coef_final - ent_coef0)
                                        * frac)
            ac.train()
            ps = ppo_update(ac, trajs, optimizer, cfg, update_idx=update,
                            last_value=last_value)
            for k, v in ps.items():
                writer.add_scalar(f"ppo/{k}", v, update)
            ppo_csv.writerow([update] + list(ps.values()))

            # progress print
            if update % 5 == 0 or update == num_updates - 1:
                elapsed = time.time() - t0
                print(f"[{update:4d}/{num_updates}] "
                      f"rew={em['reward']:7.1f}  "
                      f"thr={em['throughput_mbps']:5.2f}Mbps  "
                      f"comp={em['completion_rate']:.3f}  "
                      f"drop={em['retx_drop_rate']:.3f}  "
                      f"SINR={em['mean_sinr_db']:5.1f}dB  "
                      f"KL={ps['approx_KL']:7.4f}  "
                      f"ent={ps['entropy_mean']:.3f}  "
                      f"clip={ps['clip_fraction']:.3f}  "
                      f"ev={ps['explained_variance']:+.2f}  "
                      f"t={elapsed:.0f}s"
                      + (f"  ec={cfg.ppo_entropy_coef:.4f}"
                         if args.entropy_coef_final is not None else ""))

            # eval
            if (update + 1) % cfg.ppo_eval_every == 0:
                eval_reward = run_eval(env, ac, cfg, update, writer, eval_csv,
                                       per_ue_csv=per_ue_csv,
                                       log_baselines=not baselines_logged)
                baselines_logged = True
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    evals_no_improve = 0
                    payload = ckpt_payload(update)
                    payload["eval_reward"] = eval_reward
                    atomic_save(payload,
                                os.path.join(run_dir, "ckpt", "best.pt"))
                    print(f"  >>> new best eval reward {eval_reward:.2f}, "
                          f"saved best.pt")
                else:
                    if update >= args.patience_min_updates:
                        evals_no_improve += 1
                    if patience and evals_no_improve >= patience:
                        print(f"  >>> early stop: no best-eval improvement in "
                              f"{patience} eval rounds "
                              f"(~{patience * cfg.ppo_eval_every} updates). "
                              f"best={best_eval_reward:.2f} at update<={update}.")
                        last_completed = update
                        boundary_ckpt = ckpt_payload(update)
                        stopped_early = True
                        break

            # latest checkpoint
            if (update + 1) % cfg.ppo_save_every == 0:
                atomic_save(ckpt_payload(update),
                            os.path.join(run_dir, "ckpt", "latest.pt"))

            last_completed = update
            boundary_ckpt = ckpt_payload(update)   # state entering update+1
            # flush logs every update: a crash loses at most the current row
            env_f.flush(); ppo_f.flush(); eval_f.flush(); per_ue_f.flush()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted -- last completed update: {last_completed}. "
              f"Saving latest.pt (a --resume re-runs the interrupted update "
              f"from its start).")
    finally:
        if last_completed >= 0:
            atomic_save(boundary_ckpt,
                        os.path.join(run_dir, "ckpt", "latest.pt"))
            print(f"latest.pt saved at update {last_completed}")
        writer.close()
        env_f.close(); ppo_f.close(); eval_f.close(); per_ue_f.close()

    total_time = time.time() - t0
    n_done = last_completed - start_update + 1
    status = "INTERRUPTED" if interrupted else "Done"
    per_update = total_time / max(n_done, 1)
    print(f"\n{status}. {n_done} updates in {total_time:.0f}s  "
          f"({per_update:.1f}s/update). "
          f"Best eval reward: {best_eval_reward:.2f}")


if __name__ == "__main__":
    main()
