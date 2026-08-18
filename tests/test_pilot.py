"""Dependency-free invariants for the deterministic RuleGrid pilot stream."""

from __future__ import annotations

import unittest

from prp_wm.pilot import (
    NONTRIPLE_DIAGNOSTIC_INDICES,
    TRIPLE_DIAGNOSTIC_INDICES,
    assert_nontriple_training_indices,
    pilot_task_specs,
)


class PilotStreamTests(unittest.TestCase):
    def test_full_nuisance_group_is_balanced_and_replicate_is_program_independent(self) -> None:
        specs = pilot_task_specs(
            split="pilot-test", master_seed=17, start=0, count=192
        )
        self.assertEqual(len(specs), 192)
        pairs = {(program.program_id, axis) for program, axis, _ in specs}
        self.assertEqual(len(pairs), 192)
        for axis in {axis for _, axis, _ in specs}:
            axis_specs = [spec for spec in specs if spec[1] is axis]
            self.assertEqual(len(axis_specs), 64)
            self.assertEqual({program.program_id for program, _, _ in axis_specs}, set(range(64)))
            self.assertEqual(len({replicate for _, _, replicate in axis_specs}), 1)

    def test_stream_is_stable_and_changes_nuisance_with_seed(self) -> None:
        first = pilot_task_specs(
            split="pilot-test", master_seed=17, start=11, count=8
        )
        self.assertEqual(
            first,
            pilot_task_specs(split="pilot-test", master_seed=17, start=11, count=8),
        )
        second = pilot_task_specs(
            split="pilot-test", master_seed=18, start=11, count=8
        )
        self.assertNotEqual(
            [replicate for _, _, replicate in first],
            [replicate for _, _, replicate in second],
        )

    def test_training_selection_is_exactly_the_nontriple_panel(self) -> None:
        self.assertEqual(
            assert_nontriple_training_indices(NONTRIPLE_DIAGNOSTIC_INDICES),
            NONTRIPLE_DIAGNOSTIC_INDICES,
        )
        with self.assertRaisesRegex(ValueError, "forbidden triple"):
            assert_nontriple_training_indices((0, 1, *TRIPLE_DIAGNOSTIC_INDICES))
        with self.assertRaisesRegex(ValueError, "exactly"):
            assert_nontriple_training_indices((0, 1, 2))


if __name__ == "__main__":
    unittest.main()
