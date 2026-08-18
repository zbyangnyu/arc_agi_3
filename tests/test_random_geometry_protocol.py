"""Invariants for the isolated randomized-geometry executor protocol."""

from __future__ import annotations

import json
import unittest

from prp_wm.random_geometry_protocol import (
    FACTOR_CODES,
    audit_random_geometry_dataset,
    build_random_geometry_dataset,
    geometry_sha256,
)
from prp_wm.rulegrid import CompositeAction


class RandomGeometryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_random_geometry_dataset(
            train_geometry_seeds=range(4),
            eval_geometry_seeds=range(100, 104),
        )
        cls.audit = audit_random_geometry_dataset(cls.dataset)

    def test_generation_is_deterministic(self) -> None:
        repeated = build_random_geometry_dataset(
            train_geometry_seeds=range(4),
            eval_geometry_seeds=range(100, 104),
        )
        self.assertEqual(self.dataset, repeated)

    def test_train_is_singleton_pair_and_eval_is_triple_only(self) -> None:
        self.assertEqual(
            {panel.panel_kind for panel in self.dataset.train_panels},
            {"singleton", "pair"},
        )
        self.assertEqual(
            {panel.action_atom_count for panel in self.dataset.train_panels},
            {1, 2},
        )
        self.assertEqual(
            {panel.panel_kind for panel in self.dataset.eval_panels},
            {"triple"},
        )
        self.assertEqual(
            {panel.action_atom_count for panel in self.dataset.eval_panels},
            {3},
        )

    def test_geometry_hashes_are_unique_and_split_disjoint(self) -> None:
        train_hashes = {
            panel.geometry_sha256 for panel in self.dataset.train_panels
        }
        eval_hashes = {
            panel.geometry_sha256 for panel in self.dataset.eval_panels
        }
        self.assertEqual(len(train_hashes), len(self.dataset.train_panels))
        self.assertEqual(len(eval_hashes), len(self.dataset.eval_panels))
        self.assertFalse(train_hashes.intersection(eval_hashes))
        self.assertTrue(
            self.audit["static_gates"][
                "train_eval_geometry_hash_intersection_empty"
            ]
        )

    def test_geometry_hash_treats_composite_atom_order_as_nonsemantic(self) -> None:
        panel = self.dataset.eval_panels[0]
        self.assertIsInstance(panel.action, CompositeAction)
        assert isinstance(panel.action, CompositeAction)
        reversed_action = CompositeAction(tuple(reversed(panel.action.actions)))
        self.assertEqual(
            geometry_sha256(panel.state, panel.action),
            geometry_sha256(panel.state, reversed_action),
        )

    def test_every_panel_has_full_selected_axis_behavior_product(self) -> None:
        self.assertEqual(
            self.audit["train_behavior_class_count_histogram"],
            {"4": 12, "16": 12, "64": 0},
        )
        self.assertEqual(
            self.audit["eval_behavior_class_count_histogram"],
            {"4": 0, "16": 0, "64": 4},
        )
        self.assertTrue(self.audit["static_gates"]["all_panels_write_disjoint"])
        self.assertTrue(
            self.audit["static_gates"]["all_selected_values_distinguishable"]
        )

    def test_nuisance_cells_stay_outside_every_write_safety_envelope(self) -> None:
        for panel in (*self.dataset.train_panels, *self.dataset.eval_panels):
            protected = set().union(
                *(set(fixture.safety_envelope) for fixture in panel.fixtures)
            )
            self.assertFalse(set(panel.nuisance_cells).intersection(protected))

    def test_preregistered_layout_coverage_gate_is_executable(self) -> None:
        from scripts.audit_random_geometry_executor_protocol import (
            _layout_coverage_gate,
        )

        dataset = build_random_geometry_dataset(
            train_geometry_seeds=range(12),
            eval_geometry_seeds=range(100, 112),
        )
        gate = _layout_coverage_gate(audit_random_geometry_dataset(dataset))
        self.assertTrue(gate["passed"])

    def test_each_panel_materializes_all_64_privileged_factor_codes(self) -> None:
        self.assertEqual(len(FACTOR_CODES), 64)
        examples = tuple(self.dataset.iter_examples("eval"))
        self.assertEqual(len(examples), 4 * 64)
        for offset in range(0, len(examples), 64):
            self.assertEqual(
                {example.factor_code for example in examples[offset : offset + 64]},
                set(FACTOR_CODES),
            )

    def test_model_record_contains_no_split_seed_hash_axis_or_id(self) -> None:
        example = next(self.dataset.iter_examples("train"))
        record = example.model_record_jsonable()
        self.assertEqual(set(record), {"inputs", "target"})
        self.assertEqual(
            set(record["inputs"]),
            {"state", "action", "factor_code"},
        )
        serialized = json.dumps(record, sort_keys=True)
        for forbidden in (
            "split",
            "geometry_seed",
            "geometry_variant",
            "geometry_sha256",
            "panel_kind",
            "axes",
            "task_id",
            "probe_id",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            self.audit["explicit_identifier_fields_in_model_input"], []
        )

    def test_overlapping_seed_domains_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            build_random_geometry_dataset(
                train_geometry_seeds=(1, 2),
                eval_geometry_seeds=(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
