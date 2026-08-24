"""Final held-out evaluation of the TRAINING-SEED REPLICATES.

The paper's headline rested on a single training seed (2024). Three independent
replicates (3024/4024/5024) of the identical procedure have now finished, so
§V-F can report the across-training-seed spread instead of n=1.

PROTOCOL -- byte-identical to queue_s40hl_cqi4_final50/final100, except that the
policy under test varies:
  world  = the CQI4 training world (nr4bit, post-RZF + beta_m, speed U(5,40),
           p_arrival U(0.15,0.50), p_csi 0.6, K=32, K_act U{16..32})
  policy = <run>/ckpt/best.pt, frozen, deterministic (argmax) decoding
  T*_SUS = 0.75 (pilot seeds 40000-40007; no 20000-band seed ever tuned)
  seeds  = reserved final band 20000-20099 (n=100)
  loop   = seed-outer with env.reset(episode_idx=seed) per scheduler

*** THE TRAP THIS SCRIPT EXISTS TO AVOID ***
env.reset(k) draws from cfg.seed + <coef>*k (env.py:87/90/114/124), so the
EVALUATION EPISODES DEPEND ON cfg.seed. Loading each replicate's own
config.json (seed 3024/4024/5024) would evaluate the four policies on four
DIFFERENT sets of 100 episodes and the comparison would be meaningless.
The config is therefore ALWAYS built from the seed-2024 run, and only the
checkpoint varies. Asserted below.

SUS+CQI is re-run alongside every policy even though it is policy-independent:
its 100 rows MUST come out identical for all four jobs, which is a cheap
integrity check that every job really did evaluate the same world. For the
2024 policy the whole file should additionally reproduce
Run4/_analysis/queue_s40hl_cqi4_final100.csv.

argv[1]=policy run dir   argv[2]=seed_start   argv[3]=seed_end(excl)   argv[4]=tag
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "6"

import csv                                                      # noqa: E402
import json                                                     # noqa: E402
import time                                                     # noqa: E402

import torch                                                    # noqa: E402

torch.set_num_threads(6)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                                       # noqa: E402
from env import SchedulerEnv                                    # noqa: E402
from policy import ActorCritic                                  # noqa: E402
from baselines import all_baselines                             # noqa: E402
from train_phase2 import env_episode_metrics, PPOScheduler      # noqa: E402

os.chdir("/home/MYH/ML_DRL_Scheduler")
CFG_RUN = "Run4/QueuePostRZF_S40HL_CQI4"        # seed 2024 -- defines the world
POL_RUN, S0, S1, TAG = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
OUT = f"Run4/_analysis/seedreplicate_final100_{TAG}.csv"
TSTAR = 0.75
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
        "completion_rate", "deadline_miss_rate", "retx_drop_rate",
        "mu_depth", "jain"]

cfg_json = json.load(open(f"{CFG_RUN}/config.json"))   # untouched, for the diff
raw = json.load(open(f"{CFG_RUN}/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
raw["sus_ortho_threshold"] = TSTAR
cfg = Config(**raw)
assert cfg.seed == 2024, "eval episodes must be defined by the seed-2024 world"
assert cfg.cqi_mode == "nr4bit" and cfg.la_beta_by_depth

# the policy's own config must match the world in everything but its seed
pol_cfg = json.load(open(f"{POL_RUN}/config.json"))
IGN = {"seed", "run_name", "run_dir", "git_hash", "git_dirty_py",
       "sus_ortho_threshold"}
# compare the RAW json of both runs: `raw` above has had its lists coerced to
# tuples for Config(), which would spuriously differ from the on-disk lists
diff = {k for k in set(cfg_json) | set(pol_cfg)
        if k not in IGN and str(cfg_json.get(k)) != str(pol_cfg.get(k))}
assert not diff, f"policy world differs from the eval world: {diff}"

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{POL_RUN}/ckpt/best.pt", map_location="cpu")
ac = ActorCritic(cfg)
ac.load_state_dict(ck["model"], strict=True)
ac.eval()
scheds = [("PPO", PPOScheduler(ac.to(DEV), deterministic=True))] + \
         [(s.name, s) for s in all_baselines(cfg) if s.name == "SUS+CQI"]
assert [n for n, _ in scheds] == ["PPO", "SUS+CQI"]
print(f"[{TAG}] policy {POL_RUN} best.pt@{ck['update']} "
      f"(train seed {pol_cfg['seed']})  seeds {S0}-{S1-1}  dev {DEV}",
      flush=True)

env = SchedulerEnv(cfg)
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["train_seed", "best_update", "seed", "sched"] + KEYS)
t0 = time.time()
for i, seed in enumerate(range(S0, S1)):
    for name, sch in scheds:                 # seed-outer, as in final50/100
        env.reset(episode_idx=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        w.writerow([pol_cfg["seed"], int(ck["update"]), seed, name]
                   + [round(float(m[k]), 4) for k in KEYS])
    f.flush()
    print(f"[{TAG}] seed {seed} done  [{i + 1}/{S1 - S0}, "
          f"{(time.time() - t0) / 60:.1f} min]", flush=True)
f.close()
print(f"[{TAG}] done -> {OUT}", flush=True)
