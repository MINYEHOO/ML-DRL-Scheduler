#!/usr/bin/env python3
"""Additive OUTCOME-ONLY reward recipe for the protected PaperMain trainer.

``--recipe reward_outcome``: the stock network, the stock Base world, and a
reward that is paid only when a packet's fate is decided (user design
2026-09-16). Per slot, over the packets that finish this slot:

    completed packet p        + lambda_c * size_p / b_norm
    failed packet p           - lambda_m
                              - LAMBDA_W * acked_bits_p / b_norm   (bits already
                                delivered to a packet that then missed = waste)
    dense ACK-bit term        none   (lambda_s = 0)
    urgency weighting         none   (when a packet completes does not matter)

Design logic. A bit sent to a packet that will complete with probability q
has expected value lambda_c*q - LAMBDA_W*(1-q), which is positive only for
q > q* = LAMBDA_W/(lambda_c+LAMBDA_W). With lambda_c = LAMBDA_W = 1 the reward
encodes q* = 50%: do not spend capacity on a packet you expect to lose. That
is the same threshold the strongest rule comparator (SUS+CQI-Feasible)
hard-codes with a predicted-rate test -- here the policy has to estimate q
from its own observations. lambda_m = 1 makes one failure cancel one average
completion. The stock reward's dense ACK term, which pays for bits regardless
of the packet's fate and so REWARDS waste, is removed; this is the first run
whose reward arrives only at packet outcomes.

Waste is charged at FAILURE time (when the packet is removed as missed /
retx-dropped), not when the bits were ACKed, so credit assignment points at
the decision to keep serving a doomed packet. Expired non-HOL packets and
full-buffer rejects were never served, so they carry the per-packet penalty
only. Rewards are not comparable to Base; judge on KPIs and the Base reward
recomputed by the holdout evaluator.

Plumbing mirrors paper_train_reward.py: the recipe config differs from frozen
Base ONLY in lambda_s (1 -> 0) and lambda_m (2 -> 1); lambda_c stays 1 and
eta_d stays 1 (the wrapper ignores it, and Deadline-PF baselines still need
it). The wrapper subclasses env.SchedulerEnv and is installed only inside this
process; no core source changes. Validation also logs SUS+CQI-Feasible and
SUS-RPS.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import paper_train as pt

ROOT = Path(__file__).resolve().parent
RECIPE = 'reward_outcome'
RECIPES = {RECIPE: {'default_updates': 1000, 'lr_final': 0.0, 'lr_decay_updates': 1000}}
BASE_WEIGHTS = {'lambda_s': 1.0, 'lambda_m': 2.0}
NEW_WEIGHTS = {'lambda_s': 0.0, 'lambda_m': 1.0}
LAMBDA_W = 1.0
REWARD = {
    'schema_version': 1,
    'completed': 'lambda_c * size / b_norm  (lambda_c = 1, no urgency weight)',
    'failed': '-lambda_m - LAMBDA_W * acked_bits / b_norm  (lambda_m = 1, LAMBDA_W = 1)',
    'dense_ack_term': 'none (lambda_s = 0)',
    'lambda_w': LAMBDA_W,
    'q_star': 0.5,
    'waste_charged_at': 'packet failure (miss / retx-drop removal)',
}
EXTRA_EVAL_BASELINES = ['SUS+CQI-Feasible', 'SUS-RPS']
_CORE_PLAN = pt.plan
_INSTALLED = False


def _build():
    import env as E

    class OutcomeRewardEnv(E.SchedulerEnv):
        """SchedulerEnv whose reward is paid at packet outcomes only."""

        def __init__(self, cfg):
            if float(cfg.lambda_s) != 0.0:
                raise ValueError('reward_outcome requires lambda_s = 0 (no dense ACK term)')
            super().__init__(cfg)
            self._slot_completed_bits = 0.0
            self._slot_wasted_bits = 0.0

        def reset(self, *a, **k):
            out = super().reset(*a, **k)
            self.ep['wasted_bits'] = 0.0
            self._slot_completed_bits = 0.0
            self._slot_wasted_bits = 0.0
            return out

        # every HOL removal -- completion, deadline miss, retx-drop -- passes
        # through here while the Packet is still attached, so its fate and the
        # bits already delivered to it are both known
        def _remove_packet(self, ue, packet_id):
            pkt = self.traffic.packets[ue]
            if pkt is not None and pkt.packet_id == packet_id:
                if pkt.is_complete:
                    self._slot_completed_bits += float(pkt.size)
                else:
                    self._slot_wasted_bits += float(pkt.acked_bits)
            super()._remove_packet(ue, packet_id)

        def _finish_slot(self, allocation):
            cfg = self.cfg
            self._slot_completed_bits = 0.0
            self._slot_wasted_bits = 0.0
            stock_reward, info = super()._finish_slot(allocation)
            n_fail = (info['n_miss_deadline'] + info['n_retx_drop']
                      + self._overflow_drop_this_slot + self._buffer_overflow_this_slot)
            reward = (cfg.lambda_c * self._slot_completed_bits / cfg.b_norm
                      - cfg.lambda_m * n_fail
                      - LAMBDA_W * self._slot_wasted_bits / cfg.b_norm)
            # the stock path already added its (lambda_s = 0) reward; replace it
            self.ep['reward'] += reward - stock_reward
            self.ep['wasted_bits'] = self.ep.get('wasted_bits', 0.0) + self._slot_wasted_bits
            info = dict(info, reward=reward, reward_stock=stock_reward,
                        completed_bits=self._slot_completed_bits,
                        wasted_bits=self._slot_wasted_bits, n_fail=n_fail)
            return reward, info

    return OutcomeRewardEnv


_ORIGINALS: dict = {}


def install():
    """Swap the env class and extend the validation baseline list in this
    process. Must run before train_phase2 is imported (it binds both names)."""
    global _INSTALLED
    import env as E
    import baselines as B
    if _INSTALLED:
        return E.SchedulerEnv
    cls = _build()
    stock_env, stock_all = E.SchedulerEnv, B.all_baselines
    _ORIGINALS.clear()
    _ORIGINALS['env.SchedulerEnv'] = stock_env
    _ORIGINALS['baselines.all_baselines'] = stock_all
    E.SchedulerEnv = cls

    def all_baselines_plus(cfg):
        out = list(stock_all(cfg))
        have = {s.name for s in out}
        out += [s for s in (B.SUSCQIFeasible(), B.SUSRPS()) if s.name not in have]
        return out

    B.all_baselines = all_baselines_plus
    for name in ('train_phase2', 'ppo'):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if hasattr(mod, 'SchedulerEnv'):
            _ORIGINALS[f'{name}.SchedulerEnv'] = mod.SchedulerEnv
            mod.SchedulerEnv = cls
        if hasattr(mod, 'all_baselines'):
            _ORIGINALS[f'{name}.all_baselines'] = mod.all_baselines
            mod.all_baselines = all_baselines_plus
    _INSTALLED = True
    return cls


def uninstall():
    """Restore the stock classes (tests only; a training process never needs
    this -- each recipe runs in its own process)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    for key, value in _ORIGINALS.items():
        mod_name, attr = key.rsplit('.', 1)
        mod = sys.modules.get(mod_name)
        if mod is not None:
            setattr(mod, attr, value)
    _ORIGINALS.clear()
    _INSTALLED = False


@contextmanager
def registered_recipes():
    previous = pt.RECIPES
    for name, recipe in RECIPES.items():
        if name in previous and previous[name] != recipe:
            raise ValueError(f'Conflicting variant recipe registration: {name}')
    pt.RECIPES = {**previous, **RECIPES}
    try:
        yield
    finally:
        pt.RECIPES = previous


def parser():
    with registered_recipes():
        return pt.parser()


def _recipe_record(root):
    root = Path(root).resolve()
    base_file = root / 'artifacts/base/config.json'
    recipe_file = root / 'artifacts' / RECIPE / 'config.json'
    base = json.loads(base_file.read_text())
    recipe = json.loads(recipe_file.read_text())
    if pt.canonical({k: base.get(k) for k in BASE_WEIGHTS}) != pt.canonical(BASE_WEIGHTS):
        raise ValueError('Frozen Base reward weights differ from the recorded design')
    expected = {**base, **NEW_WEIGHTS}
    if pt.canonical(recipe) != pt.canonical(expected):
        raise ValueError('reward_outcome must differ from frozen Base only in lambda_s and lambda_m')
    return recipe, {
        'schema_version': 1,
        'name': RECIPE,
        'source_sha256': {'paper_train_outcome.py': pt.sha_file(root / 'paper_train_outcome.py')},
        'base_recipe_sha256': pt.sha_file(base_file),
        'derived_recipe_sha256': pt.sha_file(recipe_file),
        'changed_ranges': {k: {'base': BASE_WEIGHTS[k], 'variant': NEW_WEIGHTS[k]} for k in NEW_WEIGHTS},
        'reward': REWARD,
        'extra_eval_baselines': EXTRA_EVAL_BASELINES,
    }


def validate_manifest(manifest, root=ROOT):
    if manifest.get('recipe') != RECIPE:
        raise ValueError('Variant validation requires recipe reward_outcome')
    root = Path(root).resolve()
    raw, expected_record = _recipe_record(root)
    if pt.canonical(manifest.get('variant_provenance')) != pt.canonical(expected_record):
        raise ValueError('Variant provenance changed or is missing; preserve this run and use its original variant sources')
    if manifest.get('source_sha256') != pt.source_hashes(root):
        raise ValueError('Training source changed for the variant run')
    if manifest.get('artifact_config_sha256') != expected_record['derived_recipe_sha256']:
        raise ValueError('Frozen variant configuration changed')
    cfg = manifest['config']
    expected = pt.normalize_config(raw)
    expected['seed'] = cfg['seed']
    expected['ppo_batched_replay'] = cfg['ppo_batched_replay']
    if cfg['traffic_model'] != 'bernoulli':
        raise ValueError('reward_outcome requires the original Bernoulli traffic')
    if manifest.get('calibration_profile') is not None:
        from calibration.profile import apply_profile
        expected = apply_profile(expected, manifest['calibration_profile']['document'])
    smoke = manifest['smoke_slots']
    if smoke is not None:
        expected.update(episode_len_main=smoke, episode_len_debug=smoke,
                        ppo_epochs=1, ppo_minibatch_size=max(1, smoke - 8),
                        ppo_eval_every=1, ppo_eval_episodes=1, ppo_save_every=1)
    if pt.canonical(cfg) != pt.canonical(expected):
        raise ValueError('Variant run configuration differs beyond the permitted recipe, seed, replay, calibration and smoke settings')
    return expected_record


def plan(args, root=ROOT):
    if args.recipe != RECIPE:
        return _CORE_PLAN(args, root)
    _, record = _recipe_record(root)
    if args.traffic_model not in (None, 'bernoulli'):
        raise ValueError('reward_outcome requires the original Bernoulli traffic')
    with registered_recipes():
        run_dir, checkpoint, manifest = _CORE_PLAN(args, root)
    if checkpoint is None:
        manifest['variant_provenance'] = record
        if manifest.get('calibration_profile') is not None:
            manifest['calibration_application'] = (
                'Bernoulli Base-calibrated beta, unchanged world and network; '
                'outcome-only reward (completed bits, per-failure penalty, wasted-bit penalty)')
    validate_manifest(manifest, root)
    return run_dir, checkpoint, manifest


def main(argv=None):
    install()
    previous_plan = pt.plan
    with registered_recipes():
        pt.plan = plan
        try:
            return pt.main(argv)
        finally:
            pt.plan = previous_plan


if __name__ == '__main__':
    main()
