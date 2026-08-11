"""OOD grid extension: seeds 30040-30099 (+60 -> n=100 total).

Approved 2026-08-10: extend every paper-OOD world uniformly to n=100 for
consistency with the main evaluation (n=100 on 20000-20099). Existing
n=40 rows (30000-30039) are reused as-is -- identical protocol (T* fixed
from the 40k pilot, policy/beta_m frozen), so this is a pure sample-size
extension applied to ALL scenarios regardless of their n=40 outcome.
Paper footnote: pre-registered n=40 verdicts were already final and are
unchanged; extension is for uniform n only.

Worlds and their fresh-pilot T* (ood_pilot40k_*):
  P055@0.75  P010@0.80  V60max@0.75  CSI02@0.70  D26@0.80  STORM2@0.75
Schedulers: PPO + {SUS+CQI@T*, SUS+PF, SUS+Rnd, SU+CQI, SU+PF, SU+Rnd}.
D26 keeps the STRICT setup for the PPO row only: env draws deadlines
U[2,6] while the policy's ActorCritic carries the training cfg
(deadline_max=12) so its observation normalization stays frozen.
argv[1]=comma world subset, argv[2]=tag.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np, torch, csv
torch.set_num_threads(8)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from train_phase2 import env_episode_metrics, PPOScheduler

os.chdir("/home/MYH/ML_DRL_Scheduler")
SEL = sys.argv[1].split(",") if len(sys.argv) > 1 else None
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"
OUT = f"Run4/_analysis/OOD/results/ood_grid_ext100_{TAG}.csv"
SEEDS = list(range(30040, 30100))          # +60 new seeds
BASE = "Run4/QueuePostRZF_S40HL_CQI4"
CK = torch.load(f"{BASE}/ckpt/best.pt", map_location="cpu")
WANT = ["SUS+PF", "SUS+Random", "SU+CQI", "SU+PF", "SU+Random"]

WORLDS = {  # name -> (overrides, T*)
    "P055":   (dict(p_arrival_min=0.55, p_arrival_max=0.55), 0.75),
    "P010":   (dict(p_arrival_min=0.10, p_arrival_max=0.10), 0.80),
    "V60max": (dict(ue_speed_max=60.0), 0.75),
    "CSI02":  (dict(p_csi=0.2), 0.70),
    "D26":    (dict(deadline_min=2, deadline_max=6), 0.80),
    "STORM2": (dict(p_arrival_min=0.55, p_arrival_max=0.55,
                    ue_speed_max=60.0, p_csi=0.2), 0.75),
}
names = SEL if SEL else list(WORLDS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ext100 worker TAG={TAG}, worlds {names}, device {DEV}, "
      f"seeds {SEEDS[0]}-{SEEDS[-1]}, best.pt@{CK['update']}", flush=True)

def make_cfg(ov, thr):
    raw = json.load(open(f"{BASE}/config.json"))
    for k in ("git_hash", "git_dirty_py"):
        raw.pop(k, None)
    raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
    raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
    raw.update(ov)
    raw["sus_ortho_threshold"] = thr
    cfg = Config(**raw)
    assert cfg.cqi_mode == "nr4bit"
    return cfg

f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["config", "sched", "seed", "n_active", "reward", "throughput_mbps",
            "goodput_mbps", "completion_rate", "deadline_miss_rate", "mu_depth"])

for world in names:
    ov, thr = WORLDS[world]
    t0 = time.time()
    print(f"\n=== [{world}] T*={thr:.2f}, +60 seeds ===", flush=True)
    cfg = make_cfg(ov, thr)
    env = SchedulerEnv(cfg)
    # policy cfg: training normalization frozen (D26 strict: deadline_max=12)
    cfg_pol = make_cfg({}, thr) if world == "D26" else cfg
    ac = ActorCritic(cfg_pol)
    ac.load_state_dict(CK["model"], strict=True)
    ac.eval()
    ppo = PPOScheduler(ac.to(DEV), deterministic=True)
    ppo_label = "PPO-strict" if world == "D26" else "PPO"
    bl = [s for s in all_baselines(cfg) if s.name in WANT or s.name == "SUS+CQI"]
    assert len(bl) == 6
    for i, seed in enumerate(SEEDS):
        for sch, label in [(ppo, ppo_label)] + \
                [(s, f"SUS+CQI@{thr:.2f}" if s.name == "SUS+CQI" else s.name) for s in bl]:
            env.reset(episode_idx=seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sch.schedule(env))
            m = env_episode_metrics(env, cfg)
            w.writerow([world, label, seed, int(env.traffic.n_active),
                        round(m["reward"], 1), round(m["throughput_mbps"], 2),
                        round(m.get("goodput_mbps", float("nan")), 2),
                        round(m["completion_rate"], 4),
                        round(m["deadline_miss_rate"], 4), round(m["mu_depth"], 3)])
        if (i + 1) % 10 == 0:
            f.flush()
            print(f"  ..{i+1}/60 ({(time.time()-t0)/60:.1f} min)", flush=True)
    f.flush()
    print(f"  [{world}] done {(time.time()-t0)/60:.1f} min", flush=True)
print("\nsaved ->", OUT, flush=True)
f.close()
