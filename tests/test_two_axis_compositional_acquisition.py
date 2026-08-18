"""Tests for the exact two-axis compositional acquisition control."""

from __future__ import annotations

from argparse import Namespace
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_two_axis_compositional_acquisition import (
    CANDIDATES_PER_SCENARIO,
    INITIAL_HYPOTHESES,
    QUERY_AWARE_POLICIES,
    _build_scenario,
    _validate_args,
    run_experiment,
)


class TwoAxisArgumentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> Namespace:
        values: dict[str, object] = {
            "groups_per_pair": 1,
            "trace_scenarios": 0,
            "data_master_seed": 2,
            "split": "test",
            "budgets": (0, 1, 2, 3),
            "seeds": (1, 2),
        }
        values.update(overrides)
        return Namespace(**values)

    def test_budget_and_seed_validation(self) -> None:
        self.assertEqual(
            _validate_args(
                self._args(budgets=(3, 0, 2, 1), seeds=(9, 7))
            ),
            ((0, 1, 2, 3), (9, 7)),
        )
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 1, 1)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 4)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(seeds=(1, 1)))


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class TwoAxisProtocolTests(unittest.TestCase):
    def test_rulegrid_partitions_are_axis_local_and_menu_is_balanced(
        self,
    ) -> None:
        scenario = _build_scenario(
            torch,
            query_axis_indices=(0, 2),
            group_index=0,
            split="two-axis-partition-test",
            master_seed=19,
            policy_seed=23,
        )
        self.assertEqual(
            int(scenario.hypothesis_indices.numel()),
            INITIAL_HYPOTHESES,
        )
        self.assertEqual(
            tuple(scenario.candidate_class_ids.shape),
            (CANDIDATES_PER_SCENARIO, 64),
        )
        self.assertTrue(scenario.partition_validation["passed"])
        self.assertEqual(
            sorted(scenario.candidate_categories),
            sorted(
                (
                    "relevant-axis-0",
                    "relevant-axis-0",
                    "relevant-axis-1",
                    "relevant-axis-1",
                    "nuisance-axis",
                    "nuisance-axis",
                    "neutral",
                    "neutral",
                )
            ),
        )
        class_counts = sorted(
            int(torch.unique(row).numel())
            for row in scenario.candidate_class_ids
        )
        self.assertEqual(class_counts, [1, 1, 4, 4, 4, 4, 4, 4])
        self.assertEqual(
            tuple(
                int(value)
                for value in torch.bincount(
                    scenario.door_values[scenario.hypothesis_indices],
                    minlength=4,
                )
            ),
            (4, 4, 4, 4),
        )

    def test_query_aware_policies_match_exact_compositional_control(
        self,
    ) -> None:
        result = run_experiment(
            torch=torch,
            groups_per_pair=1,
            seeds=(31,),
            budgets=(0, 1, 2, 3),
            split="two-axis-rollout-test",
            data_master_seed=29,
            trace_scenarios=0,
        )
        self.assertTrue(result["partition_validation"]["all_passed"])
        self.assertTrue(result["exact_control"]["passed"])
        aggregate = result["aggregate"]
        for policy in QUERY_AWARE_POLICIES:
            self.assertAlmostEqual(
                aggregate["0"]["policies"][policy][
                    "mean_optimal_query_success_probability"
                ],
                0.25,
            )
            self.assertAlmostEqual(
                aggregate["1"]["policies"][policy][
                    "mean_optimal_query_success_probability"
                ],
                0.5,
            )
            self.assertAlmostEqual(
                aggregate["2"]["policies"][policy][
                    "mean_optimal_query_success_probability"
                ],
                1.0,
            )
            self.assertAlmostEqual(
                aggregate["2"]["policies"][policy][
                    "b2_complementary_relevant_axes_rate"
                ],
                1.0,
            )

        # Global MI first spends its action on the four-way nuisance factor:
        # two global bits, but no information about the compositional query.
        global_budget_one = aggregate["1"]["policies"]["global-mi"]
        self.assertAlmostEqual(
            global_budget_one["mean_optimal_query_success_probability"],
            0.25,
        )
        self.assertAlmostEqual(
            global_budget_one["mean_global_information_acquired_bits"],
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            global_budget_one["selected_category_rates_by_step_audit"]["1"][
                "nuisance-axis"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
