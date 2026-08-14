"""Re-run of the ID final evaluation to extract per-episode FULL-BUFFER
REJECTION rates (manuscript placeholders r_PPO / r_SUS+CQI).

Protocol is byte-identical to queue_s40hl_cqi4_final50.py, except that only
the two schedulers needed for the placeholders are run (PPO and SUS+CQI) and
the raw arrival/rejection counters are written UNROUNDED:
  * world  = CQI4 training world from Run4/QueuePostRZF_S40HL_CQI4/config.json
             (nr4bit, post-RZF + beta_m(CQI4), speed U(5,40),
              p_arrival U(0.15,0.50), p_csi 0.6, K=32, K_act U{16..32});
  * policy = best.pt@409, frozen, deterministic (argmax) decoding;
  * T*_SUS = 0.75 (pilot seeds 40000-40007; no 20000-band seed was ever
             used for tuning);
  * seeds  = reserved final band 20000-20099 (n=100);
  * seed-outer loop with env.reset(episode_idx=seed) per scheduler, exactly
    as in final50 -- each scheduler starts from the same per-seed reset, so
    dropping the other seven baselines cannot change these two rows.

Episode rejection rate follows the existing code definition verbatim
(train_phase2.env_episode_metrics):
    buffer_overflow_rate = n_buffer_overflow / max(n_arrivals + n_buffer_overflow, 1)
i.e. N_rejected / (N_accepted + N_rejected).

Outputs a NEW file (official CSVs untouched):
    Run4/_analysis/queue_s40hl_cqi4_final100_rejection_<TAG>.csv

argv[1]=seed_start  argv[2]=seed_end(exclusive)  argv[3]=tag
"""
import os, sys, csv, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
RUN = "Run4/QueuePostRZF_S40HL_CQI4"
S0, S1, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SEEDS = list(range(S0, S1))
OUT = f"Run4/_analysis/queue_s40hl_cqi4_final100_rejection_{TAG}.csv"
TSTAR = 0.75

raw = json.load(open(f"{RUN}/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
raw["sus_ortho_threshold"] = TSTAR
cfg = Config(**raw)
assert cfg.cqi_mode == "nr4bit" and cfg.la_beta_by_depth
assert cfg.queue_size > 1, "buffer_overflow_rate is a queue-mode metric"

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{RUN}/ckpt/best.pt", map_location="cpu")
ac = ActorCritic(cfg)
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
scheds = [("PPO", PPOScheduler(ac.to(DEV), deterministic=True))] + \
         [(s.name, s) for s in all_baselines(cfg) if s.name == "SUS+CQI"]
assert [n for n, _ in scheds] == ["PPO", "SUS+CQI"]
print(f"worker {TAG}: seeds {S0}-{S1-1}, device {DEV}, T*={TSTAR}, "
      f"best.pt@{ck['update']}", flush=True)

# unrounded; the five verification keys are the ones present in final100.csv
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate"]
env = SchedulerEnv(cfg)
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["seed", "sched", "n_arrivals", "n_buffer_overflow", "n_offered",
            "buffer_overflow_rate"] + KEYS)
t0 = time.time()
for i, seed in enumerate(SEEDS):
    for name, sch in scheds:              # seed-outer, identical to final50
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        n_acc = int(env.ep["n_arrivals"])
        n_rej = int(env.ep["n_buffer_overflow"])
        n_off = n_acc + n_rej
        # recompute independently and cross-check against the env's own value
        rate = n_rej / max(n_off, 1)
        assert abs(rate - float(m["buffer_overflow_rate"])) < 1e-12, \
            f"rejection-rate mismatch seed {seed} {name}"
        w.writerow([seed, name, n_acc, n_rej, n_off, repr(rate)]
                   + [repr(float(m[k])) for k in KEYS])
    f.flush()
    print(f"seed {seed} done  [{i+1}/{len(SEEDS)}, "
          f"{(time.time()-t0)/60:.1f} min]", flush=True)
f.close()
print(f"worker {TAG} done -> {OUT}", flush=True)
