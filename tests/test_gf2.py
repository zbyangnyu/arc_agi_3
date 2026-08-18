from __future__ import annotations

import math
import random
import unittest

from prp_wm.evaluation import evaluate_gate0, exact_uniform_statistics, run_episode
from prp_wm.gf2 import (
    ALL_ACTIONS,
    ALL_RULES,
    Action,
    Belief,
    Rule,
    expected_information_gain,
    predictive_distribution,
    transition,
    update_belief,
)
from prp_wm.schema import Transition, choose_information_action, make_task_bundle


class GF2EnvironmentTests(unittest.TestCase):
    def test_truth_table_matches_manual_gf2_dot_product(self) -> None:
        for state in (0, 1):
            for rule in ALL_RULES:
                for action in ALL_ACTIONS:
                    expected = state ^ ((rule.r0 * action.a0 + rule.r1 * action.a1) % 2)
                    self.assertEqual(transition(state, action, rule), expected)

    def test_uniform_belief_is_normalized(self) -> None:
        belief = Belief.uniform()
        self.assertTrue(math.isclose(sum(belief.weights), 1.0))
        self.assertEqual(len(belief.support), 4)
        self.assertAlmostEqual(belief.entropy_bits(), 2.0)

    def test_bit_inputs_are_strict_integers(self) -> None:
        for invalid in (True, 1.0, "1"):
            with self.assertRaises(ValueError):
                Rule(invalid, 0)  # type: ignore[arg-type]

    def test_zero_probe_has_no_information(self) -> None:
        belief = Belief.uniform()
        self.assertAlmostEqual(
            expected_information_gain(belief, state=0, action=Action(0, 0)),
            0.0,
        )

    def test_each_nonzero_probe_reveals_one_bit_initially(self) -> None:
        belief = Belief.uniform()
        for action in ALL_ACTIONS:
            expected = 0.0 if action == Action(0, 0) else 1.0
            self.assertAlmostEqual(expected_information_gain(belief, 0, action), expected)

    def test_dependent_probe_becomes_uninformative(self) -> None:
        belief = Belief.uniform()
        first_action = Action(0, 1)
        rule = Rule(1, 1)
        next_state = transition(0, first_action, rule)
        belief = update_belief(belief, 0, first_action, next_state)

        self.assertEqual(len(belief.support), 2)
        self.assertAlmostEqual(expected_information_gain(belief, next_state, first_action), 0.0)
        self.assertAlmostEqual(
            expected_information_gain(belief, next_state, Action(1, 0)),
            1.0,
        )

    def test_impossible_observation_is_rejected(self) -> None:
        belief = update_belief(Belief.uniform(), 0, Action(1, 0), 0)
        with self.assertRaisesRegex(ValueError, "impossible"):
            update_belief(belief, 0, Action(1, 0), 1)

    def test_predictive_distribution_sums_to_one(self) -> None:
        probabilities = predictive_distribution(Belief.uniform(), 1, Action(1, 1))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertAlmostEqual(probabilities[1], 0.5)

    def test_oracle_identifies_every_rule_in_two_actions(self) -> None:
        for rule in ALL_RULES:
            result = run_episode(rule, "oracle_eig", budget=4)
            self.assertTrue(result.identified)
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.final_belief.support, (rule,))

    def test_change_seeking_can_confuse_change_with_information(self) -> None:
        result = run_episode(Rule(0, 1), "change_seeking", budget=6)
        self.assertFalse(result.identified)
        self.assertEqual(result.steps, 6)
        self.assertEqual(len({item.action for item in result.history}), 1)

    def test_privileged_targets_are_not_in_inference_view(self) -> None:
        first_bundle = make_task_bundle(Rule(1, 0))
        perturbed_bundle = make_task_bundle(Rule(0, 1))
        self.assertEqual(first_bundle.inference, perturbed_bundle.inference)
        first_action = choose_information_action(first_bundle.inference, Belief.uniform())
        perturbed_action = choose_information_action(
            perturbed_bundle.inference,
            Belief.uniform(),
        )
        self.assertEqual(first_action, perturbed_action)

    def test_task_bundle_rejects_inconsistent_support(self) -> None:
        impossible = (Transition(0, Action(0, 0), 1),)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            make_task_bundle(Rule(0, 0), current_state=1, support=impossible)

    def test_exact_selector_handles_tiny_positive_information(self) -> None:
        epsilon = 1e-14
        belief = Belief(
            (
                math.log1p(-epsilon),
                math.log(epsilon),
                -math.inf,
                -math.inf,
            )
        )
        selected = choose_information_action(make_task_bundle(Rule(0, 0)).inference, belief)
        self.assertEqual(selected, Action(0, 1))


class Gate0EvaluationTests(unittest.TestCase):
    def test_uniform_exact_budgeted_statistics(self) -> None:
        restricted_mean, identification_rate = exact_uniform_statistics(12)
        self.assertAlmostEqual(restricted_mean, 3.3318686485290527)
        self.assertAlmostEqual(identification_rate, 0.9992676973342896)

    def test_uniform_random_has_expected_interaction_gap(self) -> None:
        results = [
            run_episode(
                ALL_RULES[index % 4],
                "uniform",
                budget=20,
                rng=random.Random(index),
            )
            for index in range(8_000)
        ]
        mean_steps = sum(result.steps for result in results) / len(results)
        self.assertGreater(mean_steps, 3.2)
        self.assertLess(mean_steps, 3.5)

    def test_gate0_passes_preregistered_headroom(self) -> None:
        report = evaluate_gate0(
            trials=504,
            repeats=4,
            budget=12,
            bootstrap_resamples=1_000,
            seed=7,
        )
        self.assertTrue(report.gate_eligible)
        self.assertTrue(report.passes)
        self.assertGreaterEqual(report.uniform_relative_step_reduction, 0.25)
        self.assertGreater(report.uniform_minus_oracle_mean_ci95[0], 0.0)

    def test_gate0_rejects_ineligible_tiny_sample(self) -> None:
        report = evaluate_gate0(
            trials=8,
            repeats=1,
            budget=12,
            bootstrap_resamples=20,
            seed=0,
        )
        self.assertFalse(report.gate_eligible)
        self.assertFalse(report.passes)

    def test_gate0_fixed_seed_is_reproducible(self) -> None:
        kwargs = dict(
            trials=16,
            repeats=2,
            budget=8,
            bootstrap_resamples=40,
            seed=123,
        )
        self.assertEqual(evaluate_gate0(**kwargs), evaluate_gate0(**kwargs))


if __name__ == "__main__":
    unittest.main()
