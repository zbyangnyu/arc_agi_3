"""Protocol and tensor tests for the nuisance learned 2x2 bridge."""

from __future__ import annotations

from argparse import Namespace
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_nuisance_learned_bridge import (
    BridgeCondition,
    EXACT_EXACT,
    _assert_public_group_copies,
    _broadcast_group_prediction,
    _deterministic_log_likelihood,
    _public_first_choice_consistency,
    _repeat_public_groups,
    _rollout_condition,
    _validate_args,
)
from scripts.run_oracle_canonical_acquisition_ceiling import (
    PublicDoorQuery,
    _candidate_panel,
)
from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (
    PROGRAMS_PER_GROUP,
    _build_nuisance_tasks,
    _exact_candidate_outcome_maps,
    _symbolic_initial_joint_log_weights,
)


class BridgeArgumentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> Namespace:
        values: dict[str, object] = {
            "groups_per_query": 2,
            "batch_size": 1,
            "seeds": (3, 4),
            "data_master_seed": 5,
            "split": "bridge-test",
            "budgets": (3, 0, 1),
        }
        values.update(overrides)
        return Namespace(**values)

    def test_validation_sorts_budgets_and_preserves_seed_order(self) -> None:
        self.assertEqual(
            _validate_args(self._args()),
            ((0, 1, 3), (3, 4)),
        )
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 1, 1)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(seeds=(3, 3)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(groups_per_query=0))


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class BridgeTensorTests(unittest.TestCase):
    def test_public_group_repeat_and_equality_guard(self) -> None:
        group_rows = torch.tensor([[1, 2], [3, 4]])
        repeated = _repeat_public_groups(torch, group_rows)
        self.assertEqual(
            repeated.tolist(),
            [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]],
        )
        _assert_public_group_copies(torch, repeated, name="test rows")
        corrupted = repeated.clone()
        corrupted[1, 0] = 99
        with self.assertRaises(AssertionError):
            _assert_public_group_copies(
                torch,
                corrupted,
                name="corrupted rows",
            )

    def test_public_first_choice_audit_rejects_hidden_mode_leakage(
        self,
    ) -> None:
        records = []
        for slot in range(PROGRAMS_PER_GROUP):
            records.append(
                {
                    "budget": 1,
                    "seed": 7,
                    "query_axis": "trigger",
                    "group_index": 0,
                    "condition": EXACT_EXACT,
                    "hidden_query_slot": slot,
                    "selected_candidate_indices": [2],
                }
            )
        self.assertTrue(_public_first_choice_consistency(records)["passed"])
        records[1]["selected_candidate_indices"] = [3]
        audit = _public_first_choice_consistency(records)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["violations"], 1)

    def test_flattened_prediction_broadcast_preserves_task_panel_order(
        self,
    ) -> None:
        from prp_wm.neural import OutcomePrediction
        from scripts.run_oracle_canonical_acquisition_ceiling import (
            _selected_prediction,
        )

        groups = 2
        candidates = 3
        modes = 2
        colors = 7
        flattened = groups * candidates
        prediction = OutcomePrediction(
            input_colors=torch.arange(flattened).reshape(flattened, 1, 1),
            change_logits=torch.zeros(flattened, modes, 1, 1),
            new_color_logits=torch.zeros(
                flattened,
                modes,
                colors,
                1,
                1,
            ),
        )
        broadcast = _broadcast_group_prediction(
            torch=torch,
            prediction=prediction,
            groups=groups,
            candidate_count=candidates,
        )
        choices = (2, 0, 1, 1, 2, 0)
        selected = _selected_prediction(
            torch=torch,
            prediction=broadcast,
            candidate_indices=choices,
            candidate_count=candidates,
        )
        self.assertEqual(
            selected.input_colors.flatten().tolist(),
            [2, 0, 1, 4, 5, 3],
        )
        target = selected.input_colors.clone().long()
        log_likelihood = selected.log_prob(target)
        self.assertEqual(tuple(log_likelihood.shape), (6, modes))
        self.assertTrue(torch.isfinite(log_likelihood).all())

    def test_wrong_map_partition_still_has_finite_proper_likelihood(
        self,
    ) -> None:
        from prp_wm.latent_rules import outcome_map
        from prp_wm.neural import OutcomePrediction

        prediction = OutcomePrediction(
            input_colors=torch.zeros(1, 1, 1, dtype=torch.long),
            change_logits=torch.full((1, 2, 1, 1), 5.0),
            new_color_logits=torch.tensor(
                [[[[[0.0]], [[8.0]], [[-2.0]]]]]
            ).expand(1, 2, 3, 1, 1),
        )
        observed = torch.full((1, 1, 1), 2, dtype=torch.long)
        maps = outcome_map(prediction)
        hard = _deterministic_log_likelihood(
            torch=torch,
            selected_outcome_maps=maps,
            observed_feedback=observed,
        )
        proper = prediction.log_prob(observed)
        self.assertTrue(torch.isneginf(hard).all())
        self.assertTrue(torch.isfinite(proper).all())

    def test_public_nuisance_triplet_is_identical(self) -> None:
        from prp_wm.rulegrid import Axis

        tasks = _build_nuisance_tasks(
            Axis.COLLISION,
            groups=1,
            split="bridge-public-triplet-test",
            master_seed=101,
            candidate_seed=103,
        )
        self.assertEqual(len(tasks), PROGRAMS_PER_GROUP)
        self.assertTrue(
            all(task.inference == tasks[0].inference for task in tasks[1:])
        )
        states, actions, action_mask, _ = _candidate_panel(
            torch=torch,
            tasks=tasks,
            device=torch.device("cpu"),
        )
        _assert_public_group_copies(torch, states, name="states")
        _assert_public_group_copies(torch, actions, name="actions")
        _assert_public_group_copies(
            torch,
            action_mask,
            name="action mask",
        )

    def test_exact_exact_control_has_theoretical_b0_and_b1(self) -> None:
        from prp_wm.causal_filter import enumerate_factor_codes
        from prp_wm.rulegrid import Axis
        from scripts.run_active_support_calibrated_executor import (
            _canonicalize_grid_tensor,
        )

        tasks = _build_nuisance_tasks(
            Axis.TRIGGER,
            groups=1,
            split="oracle-canonical-nuisance-acquisition-ceiling",
            master_seed=2026071601,
            candidate_seed=20260873,
        )
        factor_bank = enumerate_factor_codes(device="cpu")
        initial = _symbolic_initial_joint_log_weights(
            torch=torch,
            tasks=tasks,
            factor_bank=factor_bank,
        )
        self.assertTrue(
            (torch.isfinite(initial).sum(dim=-1) == 48).all()
        )
        exact_raw, _ = _exact_candidate_outcome_maps(
            torch=torch,
            tasks=tasks,
            factor_bank=factor_bank,
        )
        exact_maps = _canonicalize_grid_tensor(
            torch,
            exact_raw,
            tasks,
        )
        _, _, _, feedback = _candidate_panel(
            torch=torch,
            tasks=tasks,
            device=torch.device("cpu"),
        )
        snapshots, _ = _rollout_condition(
            torch=torch,
            condition=BridgeCondition(EXACT_EXACT, "exact", "exact"),
            tasks=tasks,
            query=PublicDoorQuery(1),
            initial_log_weights=initial,
            factor_bank=factor_bank,
            exact_outcome_maps=exact_maps,
            learned_outcome_maps=exact_maps.clone(),
            learned_prediction=None,
            canonical_feedback=feedback,
            budgets=(0, 1),
        )
        self.assertTrue(
            all(
                abs(record["tie_aware_terminal_accuracy"] - 1.0 / 3.0)
                < 1e-6
                for record in snapshots[0]
            )
        )
        self.assertTrue(
            all(
                record["tie_aware_terminal_accuracy"] == 1.0
                for record in snapshots[1]
            )
        )
        first_indices = {
            record["selected_candidate_indices"][0]
            for record in snapshots[1]
        }
        self.assertEqual(len(first_indices), 1)
        self.assertTrue(
            all(
                record["selected_candidate_categories_audit"][0]
                == "query-atomic"
                for record in snapshots[1]
            )
        )


if __name__ == "__main__":
    unittest.main()
