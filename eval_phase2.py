"""
eval_phase2.py -- Evaluate a trained PPO policy against the Phase-1 baselines.

Usage:
  python eval_phase2.py --checkpoint runs/<run>/ckpt/best.pt --mode debug
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from config import phase2_debug_config, phase2_main_config
from env import SchedulerEnv
from policy import ActorCritic
from baselines import all_baselines
from metrics import evaluate, format_table


class PPOScheduler:
    name = "PPO"

    def __init__(self, ac, deterministic: bool = True):
        self.ac = ac
        self.deterministic = deterministic

    def schedule(self, env) -> np.ndarray:
        with torch.no_grad():
            out = self.ac.decode(env.get_observation(),
                                  deterministic=self.deterministic)
        return out["action_sequence"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["debug", "main"], default="debug")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = (phase2_debug_config(seed=args.seed)
           if args.mode == "debug" else
           phase2_main_config(seed=args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(cfg.describe())

    env = SchedulerEnv(cfg)
    ac = ActorCritic(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ac.load_state_dict(ckpt["model"])
    ac.eval()
    print(f"loaded checkpoint  update={ckpt.get('update', '?')}  "
          f"eval_reward={ckpt.get('eval_reward', '?')}")

    schedulers = [PPOScheduler(ac, deterministic=True)] + all_baselines(cfg)
    results = {}
    for sched in schedulers:
        print(f"  evaluating {sched.name} ({args.num_episodes} episodes)...")
        # held-out seed range (episode 10000+): disjoint from the training
        # episodes 0..num_updates-1, same protocol as train_phase2.run_eval
        results[sched.name] = evaluate(env, sched, args.num_episodes,
                                       episode_offset=10000)

    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
