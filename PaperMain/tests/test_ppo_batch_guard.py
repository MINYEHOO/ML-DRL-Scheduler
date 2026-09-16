"""Fail-closed batched replay checks with real differentiable CPU tensors.

These tests inject faults into replay's actual minibatch output and observe
optimizer calls/parameters, without constructing a wireless environment.
"""
import copy
import dataclasses
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from ppo import SlotTrajectory, ppo_update


class TinyActorCritic(torch.nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor(0.1))
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.value_head.weight.fill_(0.2)
        self.device = torch.device("cpu")
        self.ret_std = 2.0
        self.fault = fault
        self.backward_calls = 0
        self.value_head.weight.register_hook(self.record_backward)

    def record_backward(self, gradient):
        self.backward_calls += 1
        return gradient

    def update_return_normalizer(self, returns):
        # A fixed normalizer keeps the fixture transparent; the real guard
        # must not require rolling back normalizer state to protect weights.
        if isinstance(self.fault, tuple) and self.fault[0] == "normalizer_nonfinite":
            self.ret_std = self.fault[1]

    def outputs(self, xs):
        x = torch.tensor(xs, dtype=torch.float32)
        return dict(log_prob_sum=self.actor * x,
                    entropy_sum=(self.actor + 1.0) * x,
                    value=self.value_head(x[:, None]).squeeze(-1),
                    num_policy_decisions=torch.full_like(x, 2, dtype=torch.long))

    def replay_batch(self, contexts):
        rep = self.outputs([ctx["x"] for ctx in contexts])
        if self.fault == "wrong_value":
            # Differentiable and finite, so this reaches backward. A silent
            # fallback would let the wrong gradient contaminate the optimizer.
            rep["value"] = rep["value"] + 1.0
        elif self.fault == "n_decisions":
            rep["num_policy_decisions"] += 1
        elif isinstance(self.fault, tuple) and self.fault[0] == "batch_nonfinite":
            _, key, value = self.fault
            rep[key] = rep[key].to(torch.float32).clone()
            # The final row is outside the one-slot sequential sample.
            rep[key][-1] = value
        return rep

    def replay(self, obs, action_sequence, policy_decision_mask):
        rep = {key: value[0] for key, value in self.outputs([obs["x"]]).items()}
        if isinstance(self.fault, tuple) and self.fault[0] == "seq_nonfinite":
            _, key, value = self.fault
            rep[key] = torch.tensor(value)
        return rep


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def fixture(fault=None, batched=True, old_logprob=None):
    cfg = Config()
    cfg.ppo_batched_replay = batched
    cfg.decode_order = "rbg_major"
    cfg.ppo_batch_verify_every = 1
    cfg.ppo_batch_verify_slots = 1
    cfg.ppo_epochs = 1
    cfg.ppo_minibatch_size = 4
    cfg.ppo_target_kl = 0.0
    ac = TinyActorCritic(fault)
    trajs = []
    for i in range(4):
        x = float(i + 1)
        trajs.append(SlotTrajectory(
            obs={"x": x}, action_sequence=np.array([[0]]),
            policy_decision_mask=np.array([[True]]),
            fixed_unit_map=np.array([[-1]]),
            old_logprob_sum=0.1 * x if old_logprob is None else old_logprob,
            per_subaction_logprobs=[], entropy_sum=1.1 * x,
            num_policy_decisions=2, value=0.2 * x,
            reward=x / 10.0, done=i == 3, ctx={"x": x}))
    return cfg, ac, trajs, RecordingSGD(ac.parameters())


class BatchedReplayGuardTests(unittest.TestCase):
    def assert_fail_closed(self, fault, *, every=1, tolerance=None):
        cfg, ac, trajs, opt = fixture(fault)
        cfg.ppo_batch_verify_every = every
        if tolerance is not None:
            cfg.ppo_batch_verify_tol = tolerance
        config_before = dataclasses.asdict(cfg)
        before = {name: p.detach().clone() for name, p in ac.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, "Batched replay.*(failed|nonfinite)"):
            ppo_update(ac, trajs, opt, cfg)
        self.assertEqual(opt.step_calls, 0)
        self.assertEqual(dataclasses.asdict(cfg), config_before)
        for name, p in ac.named_parameters():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            self.assertIsNone(p.grad)
        return ac

    def test_incorrect_differentiable_value_never_reaches_optimizer(self):
        ac = self.assert_fail_closed("wrong_value")
        self.assertEqual(ac.backward_calls, 1)  # contaminated grads were cleared

    def test_decision_count_mismatch_never_reaches_optimizer(self):
        self.assert_fail_closed("n_decisions")

    def test_nonfinite_actual_batch_rows_fail_even_without_periodic_check(self):
        for key in ("log_prob_sum", "entropy_sum", "value", "num_policy_decisions"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value):
                    ac = self.assert_fail_closed(("batch_nonfinite", key, value), every=0)
                    self.assertEqual(ac.backward_calls, 0)

    def test_nonfinite_sequential_reference_does_not_pass_numeric_comparison(self):
        for key in ("log_prob_sum", "entropy_sum", "value", "num_policy_decisions"):
            for value in (float("nan"), float("inf")):
                with self.subTest(key=key, value=value):
                    self.assert_fail_closed(("seq_nonfinite", key, value))

    def test_invalid_normalizer_or_tolerance_fails_closed(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(normalizer=value):
                self.assert_fail_closed(("normalizer_nonfinite", value))
        for value in (float("nan"), float("inf"), 0.0):
            with self.subTest(tolerance=value):
                self.assert_fail_closed(None, tolerance=value)

    def test_exception_during_reference_check_clears_gradients(self):
        cfg, ac, trajs, opt = fixture()
        before = copy.deepcopy(ac.state_dict())

        def fail(*args):
            raise AssertionError("trajectory mismatch")

        ac.replay = fail
        with self.assertRaisesRegex(AssertionError, "trajectory mismatch"):
            ppo_update(ac, trajs, opt, cfg)
        self.assertEqual(opt.step_calls, 0)
        self.assertEqual(ac.backward_calls, 1)
        for name, p in ac.named_parameters():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            self.assertIsNone(p.grad)

    def test_success_matches_sequential_with_short_minibatch_and_kl_freeze(self):
        for freeze in (False, True):
            with self.subTest(freeze=freeze):
                arms = []
                for batched in (False, True):
                    cfg, ac, trajs, opt = fixture(
                        batched=batched, old_logprob=-2.0 if freeze else None)
                    cfg.ppo_minibatch_size = 3  # final minibatch has one slot
                    cfg.ppo_epochs = 2
                    cfg.ppo_target_kl = 0.02 if freeze else 0.0
                    before_actor = ac.actor.detach().clone()
                    before_critic = ac.value_head.weight.detach().clone()
                    stats = ppo_update(ac, trajs, opt, cfg)
                    self.assertEqual(opt.step_calls, 4)
                    self.assertEqual(cfg.ppo_batched_replay, batched)
                    self.assertFalse(torch.equal(ac.value_head.weight, before_critic))
                    if freeze:
                        torch.testing.assert_close(ac.actor, before_actor, rtol=0, atol=0)
                    arms.append((ac, stats))
                for first, second in zip(arms[0][0].parameters(), arms[1][0].parameters()):
                    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
                for key, value in arms[0][1].items():
                    self.assertAlmostEqual(value, arms[1][1][key], places=5, msg=key)


if __name__ == "__main__":
    unittest.main()
