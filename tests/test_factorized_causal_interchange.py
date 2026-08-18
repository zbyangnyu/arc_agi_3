"""Focused invariants for the privileged causal-interchange audit."""

from __future__ import annotations

import unittest

from scripts.eval_factorized_causal_interchange import (
    _make_support_only_batch,
    _make_randomized_geometries,
    _select_interchanges,
    _summarize,
    factor_code_to_rule_program,
    patch_factor_code,
)

try:
    import torch
except ImportError:
    torch = None


class FactorizedCausalInterchangeTests(unittest.TestCase):
    def test_patch_replaces_exactly_one_privileged_axis(self) -> None:
        source = (0, 1, 2)
        donor = (3, 2, 1)
        for axis, expected in enumerate(((3, 1, 2), (0, 2, 2), (0, 1, 1))):
            patched = patch_factor_code(source, donor, axis)
            self.assertEqual(patched, expected)
            self.assertEqual(
                sum(left != right for left, right in zip(source, patched)),
                1,
            )

    def test_patch_rejects_a_nonintervention(self) -> None:
        with self.assertRaisesRegex(ValueError, "must change"):
            patch_factor_code((0, 1, 2), (3, 1, 0), 1)

    def test_factor_code_uses_protocol_product_order(self) -> None:
        self.assertEqual(factor_code_to_rule_program((0, 0, 0)).program_id, 0)
        self.assertEqual(factor_code_to_rule_program((2, 1, 1)).program_id, 37)
        self.assertEqual(factor_code_to_rule_program((3, 3, 3)).program_id, 63)

    def test_selection_uses_other_tasks_and_preserves_single_axis(self) -> None:
        inferred = (
            ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)),
            ((1, 2, 3), (2, 3, 0), (3, 0, 1), (0, 1, 2)),
            ((2, 3, 1), (3, 0, 2), (0, 1, 3), (1, 2, 0)),
        )
        choices, skipped = _select_interchanges(inferred)
        self.assertEqual(len(choices), 9)
        self.assertEqual(skipped, ())
        for choice in choices:
            self.assertNotEqual(choice.source_task, choice.donor_task)
            self.assertNotEqual(
                choice.source_code[choice.axis], choice.donor_code[choice.axis]
            )
            self.assertEqual(
                sum(
                    left != right
                    for left, right in zip(choice.source_code, choice.patched_code)
                ),
                1,
            )

    def test_selection_reports_missing_donor_variation(self) -> None:
        inferred = (
            ((0, 0, 0),) * 4,
            ((0, 0, 0),) * 4,
        )
        choices, skipped = _select_interchanges(inferred)
        self.assertEqual(choices, ())
        self.assertEqual(len(skipped), 6)

    def test_randomized_geometries_are_unique_and_every_axis_change_is_effective(self) -> None:
        from prp_wm.rulegrid import DEFAULT_PALETTE, atomic_actions, simulate

        geometries = _make_randomized_geometries(8, seed=20260731)
        self.assertEqual(
            geometries,
            _make_randomized_geometries(8, seed=20260731),
        )
        signatures = {
            (geometry.state, geometry.action.canonical_id)
            for geometry in geometries
        }
        self.assertEqual(len(signatures), 8)
        for geometry in geometries:
            self.assertEqual(len(atomic_actions(geometry.action)), 3)
            for program_id in range(64):
                source_code = (
                    program_id // 16,
                    (program_id % 16) // 4,
                    program_id % 4,
                )
                source_target = simulate(
                    geometry.state,
                    geometry.action,
                    factor_code_to_rule_program(source_code),
                    DEFAULT_PALETTE,
                )
                for axis in range(3):
                    for donor_value in range(4):
                        if donor_value == source_code[axis]:
                            continue
                        donor = list(source_code)
                        donor[axis] = donor_value
                        patched = patch_factor_code(source_code, donor, axis)
                        patched_target = simulate(
                            geometry.state,
                            geometry.action,
                            factor_code_to_rule_program(patched),
                            DEFAULT_PALETTE,
                        )
                        self.assertNotEqual(source_target, patched_target)

    def test_summary_deduplicates_executor_and_intervention_cases(self) -> None:
        diagnostic = {
            "public_input_sha256": "input",
            "simulator_target_sha256": "target",
            "executor_map_sha256": "map",
            "map_grid_exact": True,
            "proper_mean_cell_nll": 0.01,
            "intervention_effective": True,
        }
        interchange = {
            "axis_index": 0,
            "source": {"factor_code": [0, 0, 0]},
            "patched": {"factor_code": [1, 0, 0]},
            "source_to_patched_hamming_distance": 1,
            "selected_axis_value_changed": True,
            "nonselected_axes_preserved": True,
            "source_code_support_compatible": True,
            "donor_code_support_compatible": True,
            "all_diagnostics_map_exact": True,
            "diagnostics": [dict(diagnostic) for _ in range(3)],
        }
        artifact = {
            "executor_checkpoint_sha256": "executor",
            "heldout_context_coverage_exact": True,
            "unique_heldout_context_count": 12,
            "support_inference_audit": [
                {
                    "predicted_set_exact": True,
                    "compatible_code_recall": 1.0,
                    "public_canonical_support_sha256": "support",
                }
            ],
            "interchanges": [interchange],
            "skipped_interchanges": [],
        }
        summary = _summarize((artifact, artifact))
        self.assertEqual(summary["raw_diagnostic_prediction_count"], 6)
        self.assertEqual(summary["unique_execution_case_count"], 1)
        self.assertEqual(summary["unique_intervention_case_count"], 1)
        self.assertEqual(summary["diagnostic_map_grid_exact_rate"], 1.0)
        self.assertEqual(summary["proper_mean_cell_nll"], 0.01)

    @unittest.skipIf(torch is None, "PyTorch is an optional dependency")
    def test_default_48_tasks_cover_every_heldout_context_four_times(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.pilot import TRIPLE_DIAGNOSTIC_INDICES, make_pilot_tasks
        from prp_wm.rulegrid import MASTER_SEED, version_space
        from scripts.run_expected_discrete_causal_coverage import (
            _build_context_pool,
            _support_context_key,
        )

        for fold in range(4):
            tasks = _build_context_pool(
                make_pilot_tasks=make_pilot_tasks,
                split="expected-discrete-causal-composition",
                master_seed=MASTER_SEED,
                diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
                count=48,
                heldout=True,
                factor_ids_for_program=rule_program_factor_ids,
                version_space=version_space,
                context_fold=fold,
            )
            counts: dict[tuple[int, int, int], int] = {}
            for task in tasks:
                context = _support_context_key(
                    task,
                    factor_ids_for_program=rule_program_factor_ids,
                    version_space=version_space,
                )
                counts[context] = counts.get(context, 0) + 1
            self.assertEqual(len(counts), 12)
            self.assertEqual(set(counts.values()), {4})

    @unittest.skipIf(torch is None, "PyTorch is an optional dependency")
    def test_inference_batch_contains_no_query_label(self) -> None:
        from prp_wm.pilot import make_pilot_tasks

        assert torch is not None
        tasks = make_pilot_tasks(
            split="factorized-interchange-input-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(21, 22, 23),
        )
        batch = _make_support_only_batch(
            torch,
            tasks,
            diagnostic_indices=(21, 22, 23),
            device=torch.device("cpu"),
        )
        self.assertIsNone(batch.query_targets)
        self.assertIsNone(batch.behavior_targets)
        self.assertIsNone(batch.behavior_mass)
        self.assertEqual(batch.support_states.shape[:2], (2, 6))
        self.assertEqual(batch.query_states.shape[:2], (2, 3))


if __name__ == "__main__":
    unittest.main()
