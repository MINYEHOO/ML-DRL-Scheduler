"""Fault injection demonstrating the UNFIXED batched fallback, no artifact writes."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import phase4_queue_config
from policy import ActorCritic
from ppo import SlotTrajectory, ppo_update

torch.set_num_threads(1)
torch.manual_seed(123)

def make_obs(cfg, seed=123):
    k, r, m, l = cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant, cfg.l_max
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal((k,r,m)) + 1j * rng.standard_normal((k,r,m))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return dict(slot=0, noise_var=1.0, direction_fb=direction,
                cqi_fb=np.full((k,r), 3.0293), age=np.zeros(k),
                deadline=np.full(k,8.), backlog=np.full(k,8000.),
                uncommitted=np.full(k,8000.), avg_throughput=np.zeros(k),
                active=np.ones(k, bool), queue_len=np.ones(k),
                queue_bits=np.full(k,8000.), next_deadline=np.zeros(k),
                fixed_allocation=np.zeros((r,l),np.int64),
                fixed_mask=np.zeros((r,l),bool),
                fixed_unit_map=np.full((r,l),-1,np.int64),
                initial_S_r=[set() for _ in range(r)])

from paper_eval import recipe_config
cfg = recipe_config('lrann')
cfg.ppo_batched_replay = True
cfg.ppo_batch_verify_every = 1
cfg.ppo_batch_verify_slots = 2
cfg.ppo_epochs = 1
cfg.ppo_minibatch_size = 2
cfg.ppo_target_kl = 0.
ac = ActorCritic(cfg)
trajs = []
for i in range(2):
    obs = make_obs(cfg,123+i)
    obs['slot'] = i
    out = ac.decode(obs, emit_context=True)
    rep = ac.replay(obs,out['action_sequence'],out['policy_decision_mask'])
    print('decode/replay',i,abs(out['log_prob_sum']-float(rep['log_prob_sum'])))
    trajs.append(SlotTrajectory(obs, out['action_sequence'],out['policy_decision_mask'],
                               obs['fixed_unit_map'],out['log_prob_sum'],
                               out['per_subaction_logprobs'],out['entropy_sum'],
                               out['num_policy_decisions'],out['value'],1.+i,i==1,
                               out['context']))
rep_b = ac.replay_batch([t.ctx for t in trajs])
print('replay_batch max logprob diff',max(abs(float(rep_b['log_prob_sum'][i])-trajs[i].old_logprob_sum) for i in range(2)))
print('baseline consistency checks passed')

calls = {'batch':0,'seq':0}
orig_b = ac.replay_batch
orig_s = ac.replay
def faulty_batch(contexts):
    calls['batch'] += 1
    rep = orig_b(contexts)
    # Inject a detected numeric error without modifying actual repository code.
    rep['value'] = rep['value'] + 100 * ac.value_head.net[-1].bias[0]
    return rep
def count_seq(*args):
    calls['seq'] += 1
    return orig_s(*args)
ac.replay_batch = faulty_batch
ac.replay = count_seq
opt = torch.optim.SGD(ac.parameters(),lr=1e-3)
for update in (0,1):
    before = ac.value_head.net[-1].bias.detach().clone()
    ppo_update(ac,trajs,opt,cfg,update_idx=update)
    print('after failed verification',json.dumps(dict(update=update,counts=calls.copy(),
          cfg_batched=cfg.ppo_batched_replay,bias_changed=bool(torch.any(ac.value_head.net[-1].bias != before)))))
