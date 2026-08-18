"""Small runner-level checks for GRAM candidate accounting."""

from __future__ import annotations

import unittest

from scripts.run_gram_causal_screen import (
    _four_item_gate,
    _ordered_unique,
    _support_ranked_unique,
    _uniform_four_mode_coupon_collector,
)


class GRAMCausalScreenRunnerTests(unittest.TestCase):
    def test_width_candidates_are_deduplicated_before_support_ranking(self) -> None:
        candidates = [(2, 0, 1), (2, 0, 1), (0, 0, 1), (3, 0, 1), (1, 0, 1)]
        costs = {
            (0, 0, 1): 0.1,
            (1, 0, 1): 0.1,
            (2, 0, 1): 0.4,
            (3, 0, 1): 0.2,
        }
        self.assertEqual(
            _ordered_unique(candidates),
            [(2, 0, 1), (0, 0, 1), (3, 0, 1), (1, 0, 1)],
        )
        self.assertEqual(
            _support_ranked_unique(candidates, costs, limit=3),
            [(0, 0, 1), (1, 0, 1), (3, 0, 1)],
        )

    def test_four_item_gate_reports_each_value(self) -> None:
        evaluation = {
            "coverage_at_4_mass_weighted": 0.95,
            "all_classes_covered_task_rate": 0.90,
            "factor_tuple_coverage_at_4": 0.89,
            "all_particles_support_exact_task_rate": 1.0,
        }
        gate = _four_item_gate(evaluation)
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["checks"]["factor_tuple_coverage_at_4"]["passed"]
        )
        self.assertEqual(len(gate["checks"]), 4)

    def test_coupon_collector_reference_distinguishes_recall_and_coverage(self) -> None:
        recall, all_covered = _uniform_four_mode_coupon_collector(4)
        self.assertAlmostEqual(recall, 1.0 - (3.0 / 4.0) ** 4)
        self.assertAlmostEqual(all_covered, 24.0 / 256.0)


if __name__ == "__main__":
    unittest.main()
