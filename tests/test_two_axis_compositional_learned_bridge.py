"""Tests for the learned two-axis compositional bridge."""

from __future__ import annotations

from argparse import Namespace
import math
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_two_axis_compositional_acquisition import (
    CANDIDATES_PER_SCENARIO,
    _build_scenario,
)
from scripts.run_two_axis_compositional_learned_bridge import (
    CONDITIONS,
    EXACT_EXACT,
    BridgeCondition,
    LearnedScenarioPanel,
    _build_learned_panel,
    _maps_to_class_ids,
    _rollout_condition,
    _same_partition,
    _summarise_condition,
    _validate_args,
)


class LearnedBridgeArgumentTests(unittest.TestCase):
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

    def test_validation_requires_the_fixed_four_budgets(self) -> None:
        self.assertEqual(
            _validate_args(
                self._args(budgets=(3, 1, 0, 2), seeds=(9, 7))
            ),
            ((0, 1, 2, 3), (9, 7)),
        )
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 1, 2)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(seeds=(1, 1)))


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class LearnedBridgeProtocolTests(unittest.TestCase):
    @staticmethod
    def _scenario():
        return _build_scenario(
            torch,
            query_axis_indices=(0, 2),
            group_index=0,
            split="two-axis-learned-bridge-test",
            master_seed=19,
            policy_seed=23,
        )

    def test_map_class_ids_compare_partitions_up_to_relabeling(self) -> None:
        maps = torch.tensor(
            [
                [
                    [[0, 0], [0, 0]],
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                ]
            ],
            dtype=torch.long,
        )
        class_ids = _maps_to_class_ids(torch, maps)
        self.assertEqual(tuple(class_ids.shape), (1, 3))
        self.assertEqual(int(class_ids[0, 0]), int(class_ids[0, 2]))
        self.assertNotEqual(int(class_ids[0, 0]), int(class_ids[0, 1]))
        self.assertTrue(
            _same_partition(
                torch,
                class_ids[0],
                torch.tensor([7, 3, 7]),
            )
        )
        self.assertFalse(
            _same_partition(
                torch,
                class_ids[0],
                torch.tensor([7, 7, 3]),
            )
        )

    def test_exact_condition_preserves_theoretical_two_step_curve(self) -> None:
        scenario = self._scenario()
        panel = LearnedScenarioPanel(
            exact_grids=torch.zeros(
                CANDIDATES_PER_SCENARIO,
                64,
                8,
                8,
                dtype=torch.long,
            ),
            learned_maps=torch.zeros(
                CANDIDATES_PER_SCENARIO,
                64,
                8,
                8,
                dtype=torch.long,
            ),
            learned_class_ids=scenario.candidate_class_ids.clone(),
            learned_prediction=None,
            partition_alignment_passed=True,
            public_states_shape=(1, CANDIDATES_PER_SCENARIO, 8, 8),
            public_actions_shape=(1, CANDIDATES_PER_SCENARIO, 4),
        )
        condition = next(
            item for item in CONDITIONS if item.name == EXACT_EXACT
        )
        records = {budget: [] for budget in (0, 1, 2, 3)}
        for truth_index in scenario.hypothesis_indices.tolist():
            snapshots, _ = _rollout_condition(
                torch,
                scenario=scenario,
                panel=panel,
                truth_index=int(truth_index),
                condition=condition,
                budgets=(0, 1, 2, 3),
            )
            for budget, record in snapshots.items():
                records[budget].append(record)
        expected_success = (0.25, 0.5, 1.0)
        expected_entropy = (2.0, 1.0, 0.0)
        for budget in range(3):
            summary = _summarise_condition(
                records[budget],
                budget=budget,
            )
            self.assertAlmostEqual(
                summary["tie_aware_terminal_accuracy"],
                expected_success[budget],
            )
            self.assertAlmostEqual(
                summary["mean_optimal_query_success_probability"],
                expected_success[budget],
            )
            self.assertAlmostEqual(
                summary["mean_query_entropy_bits"],
                expected_entropy[budget],
            )
        self.assertAlmostEqual(
            _summarise_condition(records[2], budget=2)[
                "b2_complementary_relevant_axes_rate"
            ],
            1.0,
        )

    def test_public_panel_batches_eight_probes_by_sixty_four_codes(self) -> None:
        from prp_wm.latent_rules import OracleFactorExecutor
        from prp_wm.neural import NeuralPRPConfig

        torch.manual_seed(31)
        config = NeuralPRPConfig(
            color_embedding=8,
            position_embedding=8,
            encoder_channels=8,
            encoder_resblocks=1,
            normalization_groups=4,
            action_embedding=8,
            rule_dim=16,
            attention_heads=4,
            attention_ffn=32,
            decoder_resblocks=1,
        )
        executor = OracleFactorExecutor(config).eval()
        scenario = self._scenario()
        panel = _build_learned_panel(
            torch,
            executor=executor,
            scenario=scenario,
            query_axis_indices=(0, 2),
            group_index=0,
            split="two-axis-learned-bridge-test",
            master_seed=19,
            policy_seed=23,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            tuple(panel.exact_grids.shape),
            (CANDIDATES_PER_SCENARIO, 64, 8, 8),
        )
        self.assertEqual(panel.learned_maps.shape, panel.exact_grids.shape)
        self.assertEqual(
            tuple(panel.learned_class_ids.shape),
            (CANDIDATES_PER_SCENARIO, 64),
        )
        self.assertEqual(
            tuple(panel.learned_prediction.change_logits.shape),
            (CANDIDATES_PER_SCENARIO, 64, 8, 8),
        )
        self.assertTrue(
            torch.isfinite(panel.learned_prediction.change_logits).all()
        )
        self.assertTrue(panel.partition_alignment_passed)

        learned_update_condition = next(
            condition
            for condition in CONDITIONS
            if condition.selection == "exact"
            and condition.update == "learned"
        )
        snapshots, trace = _rollout_condition(
            torch,
            scenario=scenario,
            panel=panel,
            truth_index=int(scenario.hypothesis_indices[0]),
            condition=learned_update_condition,
            budgets=(0, 1),
        )
        self.assertEqual(
            len(trace["observed_log_predictive_probabilities"]),
            1,
        )
        self.assertTrue(
            math.isfinite(
                trace["observed_log_predictive_probabilities"][0]
            )
        )
        self.assertIsNotNone(
            snapshots[1]["mean_observed_log_predictive_probability"]
        )

    def test_condition_cross_is_complete(self) -> None:
        self.assertEqual(
            {
                (condition.selection, condition.update)
                for condition in CONDITIONS
            },
            {
                ("exact", "exact"),
                ("exact", "learned"),
                ("exact", "projected-learned"),
                ("learned", "exact"),
                ("learned", "learned"),
                ("learned", "projected-learned"),
            },
        )
        with self.assertRaises(ValueError):
            BridgeCondition("bad", "oracle", "exact")
        with self.assertRaises(ValueError):
            BridgeCondition("bad", "exact", "projected")


if __name__ == "__main__":
    unittest.main()
