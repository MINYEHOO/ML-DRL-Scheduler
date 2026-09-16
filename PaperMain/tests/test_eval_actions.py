"""Exact action and planner equivalence for deterministic evaluation.

These fixtures exercise CQI exclusion, HARQ reservations, early RBG closure,
and cross-RBG budget debit without constructing a channel environment.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy
from config import phase4_queue_config
from policy import ActorCritic


def make_obs(cfg, seed=123):
    k, r, m, l = cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant, cfg.l_max
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal((k, r, m)) + 1j * rng.standard_normal((k, r, m))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return dict(
        slot=37, noise_var=1.0, direction_fb=direction,
        cqi_fb=rng.choice([0., 0.1523, 1.4766, 3.0293, 5.5547], (k, r)),
        age=rng.integers(0, 20, k).astype(float),
        deadline=rng.integers(1, 12, k).astype(float),
        backlog=np.full(k, 8000.), uncommitted=np.full(k, 8000.),
        avg_throughput=rng.uniform(0, 100, k), active=np.ones(k, bool),
        queue_len=np.full(k, 2.), queue_bits=np.full(k, 16000.),
        next_deadline=np.full(k, 12.),
        fixed_allocation=np.zeros((r, l), np.int64),
        fixed_mask=np.zeros((r, l), bool),
        fixed_unit_map=np.full((r, l), -1, np.int64),
        initial_S_r=[set() for _ in range(r)])


def reserve(obs, r, l, u):
    obs["fixed_mask"][r, l] = True
    obs["fixed_allocation"][r, l] = u + 1
    obs["fixed_unit_map"][r, l] = 100 + u
    obs["initial_S_r"][r].add(u)


class EvalActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_actor(self, seed=123, **kwargs):
        cfg = phase4_queue_config(
            decode_order="rbg_major", la_mode="post_rzf",
            la_beta_by_depth=(1.0162, .8509, .7732, .728), **kwargs)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return ActorCritic(cfg)

    def recorded_pass(self, ac, obs, fast):
        positions, closures = [], []
        original_position = ac._position_logits_and_mask
        original_close = policy.SlotAllocationPlanner.close_rbg

        def position(*args, **kwargs):
            out = original_position(*args, **kwargs)
            _, _, r, l, selected, counts, remaining = args
            positions.append((r, l, sorted(selected), counts.copy(),
                              remaining.detach().cpu().numpy().copy(),
                              out[0].detach().cpu().numpy().copy(),
                              out[1].detach().cpu().numpy().copy()))
            return out

        def close(planner, r, fixed, entries):
            out = original_close(planner, r, fixed, entries)
            closures.append((r, list(fixed), list(entries),
                             planner._remaining.copy(),
                             planner.planned_btx_map()))
            return out

        with mock.patch.object(ac, "_position_logits_and_mask", side_effect=position), \
                mock.patch.object(policy.SlotAllocationPlanner, "close_rbg", close):
            out = ac.deterministic_action(obs) if fast else ac.decode(obs, deterministic=True)
        action = out if fast else out["action_sequence"]
        return action, positions, closures

    def assert_nested_equal(self, left, right, planner_tolerance=False):
        if isinstance(left, np.ndarray):
            if planner_tolerance and left.dtype == np.float64:
                # CPU BLAS can differ at ~1e-13 between identical small
                # matrix solves. Action inputs remain exact float32.
                np.testing.assert_allclose(left, right, rtol=5e-13, atol=1e-9)
            else:
                np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key], planner_tolerance)
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b, planner_tolerance)
        elif planner_tolerance and isinstance(left, float):
            np.testing.assert_allclose(left, right, rtol=5e-13, atol=1e-9)
        else:
            self.assertEqual(left, right)

    def test_exact_actions_masks_logits_and_budget_closures(self):
        for seed in (123, 2024, 9876):
            ac = self.make_actor(seed)
            for scenario in ("mixed_cqi", "harq", "zero_cqi", "no_active",
                             "no_budget", "small_budget", "all_fixed"):
                with self.subTest(seed=seed, scenario=scenario):
                    obs = make_obs(ac.cfg, seed)
                    if scenario == "harq":
                        reserve(obs, 0, 1, 0)
                        reserve(obs, 1, 0, 1)
                        reserve(obs, 1, 2, 2)
                        obs["active"][0] = False
                        obs["cqi_fb"][0, 0] = 0.0
                    elif scenario == "zero_cqi":
                        obs["cqi_fb"][:] = 0.0
                        reserve(obs, 0, 0, 0)
                    elif scenario == "no_active":
                        obs["active"][:] = False
                    elif scenario == "no_budget":
                        obs["uncommitted"][:] = 0.0
                    elif scenario == "small_budget":
                        obs["uncommitted"][:] = np.resize(
                            [0., .5, 1., 1.5, 500.], ac.cfg.num_ue)
                    elif scenario == "all_fixed":
                        for r in range(ac.cfg.num_rbg):
                            for l in range(ac.cfg.l_max):
                                reserve(obs, r, l, l)
                    before = copy.deepcopy(obs)
                    reference = self.recorded_pass(ac, obs, fast=False)
                    fast = self.recorded_pass(ac, obs, fast=True)
                    self.assert_nested_equal(reference[:2], fast[:2])
                    self.assert_nested_equal(reference[2], fast[2], planner_tolerance=True)
                    self.assert_nested_equal(obs, before)
                    np.testing.assert_array_equal(
                        fast[0][obs["fixed_mask"]],
                        obs["fixed_allocation"][obs["fixed_mask"]])
                    if scenario in ("no_active", "no_budget", "zero_cqi"):
                        self.assertFalse(np.any(fast[0][~obs["fixed_mask"]]))

    def test_early_closure_and_forced_full_rank(self):
        for full_rank in (False, True):
            ac = self.make_actor(ppo_force_full_rank=full_rank)
            with torch.no_grad():
                for module in (ac.score_net, ac.no_user_head):
                    for parameter in module.parameters():
                        parameter.zero_()
                ac.score_net.net[-1].bias.fill_(1.)
                ac.no_user_head.net[-1].bias.fill_(2.)
            obs = make_obs(ac.cfg)
            obs["cqi_fb"][:] = 3.0293
            reference = self.recorded_pass(ac, obs, fast=False)
            fast = self.recorded_pass(ac, obs, fast=True)
            self.assert_nested_equal(reference[:2], fast[:2])
            self.assert_nested_equal(reference[2], fast[2], planner_tolerance=True)
            if full_rank:
                self.assertTrue(np.all(fast[0] > 0))
            else:
                self.assertTrue(np.all(fast[0][:, 0] > 0))
                self.assertFalse(np.any(fast[0][:, 1:]))

    def test_action_only_skips_statistics_critic_and_trace_export(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        expected = ac.decode(obs, deterministic=True)["action_sequence"]
        unexpected = AssertionError("evaluation must skip this computation")
        with mock.patch("policy.torch.distributions.Categorical", side_effect=unexpected), \
                mock.patch("policy.build_value_tail", side_effect=unexpected), \
                mock.patch.object(ac.value_head, "forward", side_effect=unexpected), \
                mock.patch.object(policy.SlotAllocationPlanner, "planned_btx_map",
                                  side_effect=unexpected):
            np.testing.assert_array_equal(ac.deterministic_action(obs), expected)

    def test_no_grad_rng_state_model_state_or_mode_mutation(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        for training in (True, False):
            ac.train(training)
            # Existing gradients also belong to the training caller.
            for parameter in ac.parameters():
                parameter.grad = torch.ones_like(parameter)
            before = {name: value.clone() for name, value in ac.state_dict().items()}
            gradients = [p.grad.clone() for p in ac.parameters()]
            rng_torch = torch.get_rng_state().clone()
            rng_numpy = copy.deepcopy(np.random.get_state())
            with mock.patch.object(ac.encoder, "forward", wraps=ac.encoder.forward) as encoder:
                ac.deterministic_action(obs)
                self.assertEqual(encoder.call_count, 1)
            self.assertTrue(ac.training == training)
            self.assertTrue(all(m.training == training for m in ac.modules()))
            torch.testing.assert_close(torch.get_rng_state(), rng_torch, rtol=0, atol=0)
            self.assert_nested_equal(np.random.get_state(), rng_numpy)
            for name, value in ac.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            for p, grad in zip(ac.parameters(), gradients):
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
            observed_grad_mode = []
            handle = ac.encoder.register_forward_hook(
                lambda module, args, out: observed_grad_mode.append(out.requires_grad))
            try:
                ac.deterministic_action(obs)
            finally:
                handle.remove()
            self.assertEqual(observed_grad_mode, [False])

    def test_decode_replay_outputs_and_gradients_are_still_available(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        out = ac.decode(obs, deterministic=True, emit_context=True)
        ac.deterministic_action(obs)
        after = ac.decode(obs, deterministic=True, emit_context=True)
        self.assert_nested_equal(out, after)
        replay = ac.replay(obs, out["action_sequence"], out["policy_decision_mask"])
        for name in ("log_prob_sum", "entropy_sum", "value"):
            self.assertAlmostEqual(float(replay[name].detach()), out[name], places=5)
        self.assertEqual(replay["num_policy_decisions"], out["num_policy_decisions"])
        loss = replay["log_prob_sum"] + replay["entropy_sum"] + replay["value"]
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in ac.parameters()))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in ac.parameters()))

    def test_nonfinite_valid_logits_fail_closed(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), mock.patch.object(
                    ac.score_net, "forward", return_value=torch.full((ac.cfg.num_ue,), value)):
                with self.assertRaises(ValueError):
                    ac.decode(obs, deterministic=True)
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    ac.deterministic_action(obs)

    def test_negative_infinite_logits_match_categorical_behavior(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        scores = torch.ones(ac.cfg.num_ue)
        scores[0] = -float("inf")
        with mock.patch.object(ac.score_net, "forward", return_value=scores):
            expected = ac.decode(obs, deterministic=True)["action_sequence"]
            np.testing.assert_array_equal(ac.deterministic_action(obs), expected)

    def test_all_negative_infinite_logits_fail_closed(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        original = ac._position_logits_and_mask

        def invalid(*args, **kwargs):
            out = list(original(*args, **kwargs))
            out[0] = torch.full_like(out[0], -float("inf"))
            out[1] = torch.ones_like(out[1])
            return tuple(out)

        with mock.patch.object(ac, "_position_logits_and_mask", side_effect=invalid):
            with self.assertRaises(ValueError):
                ac.decode(obs, deterministic=True)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                ac.deterministic_action(obs)

    def test_masked_out_nonfinite_logits_do_not_change_action(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        obs["active"][0] = False
        scores = torch.ones(ac.cfg.num_ue)
        scores[0] = float("nan")
        with mock.patch.object(ac.score_net, "forward", return_value=scores):
            expected = ac.decode(obs, deterministic=True)["action_sequence"]
            np.testing.assert_array_equal(ac.deterministic_action(obs), expected)

    def test_internal_action_only_rejects_training_arguments(self):
        ac = self.make_actor()
        obs = make_obs(ac.cfg)
        for kwargs in ({}, {"deterministic": True, "emit_context": True},
                       {"deterministic": True, "stored_actions": np.zeros((8, 4))},
                       {"deterministic": True, "stored_mask": np.zeros((8, 4), bool)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "action_only"):
                ac._rbg_major_pass(obs, action_only=True, **kwargs)

    def test_legacy_decode_order_uses_existing_decoder(self):
        ac = self.make_actor()
        ac.cfg.decode_order = "layer_major"
        obs = make_obs(ac.cfg)
        expected = ac.decode(obs, deterministic=True)["action_sequence"]
        with mock.patch.object(ac, "decode", wraps=ac.decode) as decode:
            np.testing.assert_array_equal(ac.deterministic_action(obs), expected)
            decode.assert_called_once_with(obs, deterministic=True)


if __name__ == "__main__":
    unittest.main()
