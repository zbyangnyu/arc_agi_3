"""Tests for randomized-singleton GeomSup/LocReg continuation helpers."""

from __future__ import annotations

from argparse import Namespace
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_counterfactual_locality_finetune import (
    _active_set_fiber_deviations,
    _active_set_fiber_huber_dispersion,
    _build_geometry_batch,
    _build_singleton_batch,
    _fiber_deviations,
    _fiber_huber_dispersion,
    _geometry_panels,
    _locality_ramp,
    _outcome_log_probabilities,
    _singleton_panels,
    _teacher_categorical_kl,
    _validate_args,
)


class CounterfactualLocalityArgumentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> Namespace:
        values: dict[str, object] = {
            "steps": 2,
            "batch_size": 8,
            "codes_per_task": 8,
            "eval_tasks": 8,
            "eval_batch_size": 4,
            "heldout_singleton_seeds": 2,
            "heldout_geometry_batch_panels": 3,
            "log_every": 1,
            "locality_ramp_steps": 2,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "locality_huber_beta": 1.0,
            "weight_decay": 1e-4,
            "balanced_weight": 1.0,
            "geometry_weight": 0.1,
            "locality_weight": 0.1,
            "teacher_distillation_weight": 0.0,
            "diagnostic_loss_weight": 0.4,
            "stage_loss_weights": (0.1, 0.15, 0.1, 0.25),
            "seed": 1,
            "data_master_seed": 2,
            "geometry_train_seed_base": 100,
            "geometry_train_seed_count": 0,
            "geometry_eval_seed_base": 200,
            "active_task_start_offset": 0,
            "train_split": "train",
            "eval_split": "eval",
        }
        values.update(overrides)
        return Namespace(**values)

    def test_control_and_regularized_weights_are_supported(self) -> None:
        for locality_weight in (0.0, 0.1):
            diagnostic, stages = _validate_args(
                self._args(locality_weight=locality_weight)
            )
            self.assertAlmostEqual(diagnostic + sum(stages), 1.0)

    def test_geometry_seed_streams_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must be disjoint"):
            _validate_args(
                self._args(
                    geometry_train_seed_base=100,
                    geometry_eval_seed_base=101,
                )
            )

    def test_teacher_distillation_weight_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must be non-negative"):
            _validate_args(self._args(teacher_distillation_weight=-0.1))

    def test_fixed_geometry_seed_cycle_is_used_for_disjointness(self) -> None:
        _validate_args(
            self._args(
                steps=320,
                geometry_train_seed_count=64,
                geometry_train_seed_base=100,
                geometry_eval_seed_base=164,
            )
        )
        with self.assertRaisesRegex(SystemExit, "must be disjoint"):
            _validate_args(
                self._args(
                    steps=320,
                    geometry_train_seed_count=64,
                    geometry_train_seed_base=100,
                    geometry_eval_seed_base=163,
                )
            )

    def test_locality_ramp_is_linear_and_saturates(self) -> None:
        self.assertAlmostEqual(_locality_ramp(0.1, 1, 4), 0.025)
        self.assertAlmostEqual(_locality_ramp(0.1, 4, 4), 0.1)
        self.assertAlmostEqual(_locality_ramp(0.1, 8, 4), 0.1)
        self.assertAlmostEqual(_locality_ramp(0.1, 1, 0), 0.1)
        self.assertEqual(_locality_ramp(0.0, 1, 4), 0.0)


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class CounterfactualLocalityTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panels = _singleton_panels(12345)
        cls.batch = _build_singleton_batch(
            torch=torch,
            panels=cls.panels,
            device="cpu",
        )

    def test_batch_contains_three_singletons_and_all_64_codes(self) -> None:
        from prp_wm.random_geometry_protocol import FACTOR_CODES

        self.assertEqual(len(self.panels), 3)
        self.assertEqual(
            [panel.axes[0].value for panel in self.panels],
            ["collision", "trigger", "relation"],
        )
        self.assertEqual(tuple(self.batch.states.shape), (3 * 64, 1, 8, 8))
        self.assertEqual(tuple(self.batch.actions.shape), (3 * 64, 1, 4))
        self.assertEqual(tuple(self.batch.targets.shape), (3 * 64, 1, 8, 8))
        self.assertEqual(tuple(self.batch.factor_ids.shape), (3 * 64, 3))
        expected = torch.tensor(FACTOR_CODES, dtype=torch.long)
        for panel_index in range(3):
            actual = self.batch.factor_ids.reshape(3, 64, 3)[panel_index]
            self.assertTrue(bool(actual.eq(expected).all().item()))

    def test_singleton_targets_are_constant_over_nuisance_fibers(self) -> None:
        targets = self.batch.targets.reshape(3, 64, 1, 8, 8)
        codes = self.batch.factor_ids.reshape(3, 64, 3)[0]
        for axis in range(3):
            for value in range(4):
                fiber = targets[axis, codes[:, axis].eq(value)]
                self.assertEqual(fiber.shape[0], 16)
                self.assertTrue(bool(fiber.eq(fiber[:1]).all().item()))

    def test_matched_geometry_batch_contains_singletons_and_pairs(self) -> None:
        panels = _geometry_panels(54321)
        batch, active_axis_sets = _build_geometry_batch(
            torch=torch,
            panels=panels,
            device="cpu",
        )
        self.assertEqual(len(panels), 6)
        self.assertEqual(
            [panel.panel_kind for panel in panels],
            ["singleton", "singleton", "singleton", "pair", "pair", "pair"],
        )
        self.assertEqual(
            active_axis_sets,
            ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)),
        )
        self.assertEqual(tuple(batch.states.shape), (6 * 64, 1, 8, 8))
        self.assertEqual(tuple(batch.actions.shape), (6 * 64, 1, 2, 4))
        self.assertEqual(tuple(batch.action_mask.shape), (6 * 64, 1, 2))
        self.assertEqual(
            batch.action_mask.reshape(6, 64, 1, 2)[:, 0, 0].sum(dim=1).tolist(),
            [1, 1, 1, 2, 2, 2],
        )

    def test_pair_locality_holds_active_tuple_and_penalizes_nuisance(self) -> None:
        panels = _geometry_panels(54321)
        batch, active_axis_sets = _build_geometry_batch(
            torch=torch,
            panels=panels,
            device="cpu",
        )
        codes = batch.factor_ids.reshape(6, 64, 3)[0]
        invariant_rows = []
        nuisance_rows = []
        for active_axes in active_axis_sets:
            invariant = sum(
                codes[:, axis].to(dtype=torch.float32)
                for axis in active_axes
            )
            nuisance_axis = next(axis for axis in range(3) if axis not in active_axes)
            invariant_rows.append(invariant)
            nuisance_rows.append(invariant + 0.5 * codes[:, nuisance_axis])
        invariant_scores = torch.stack(invariant_rows)
        deviations, ranges = _active_set_fiber_deviations(
            torch=torch,
            full_grid_log_likelihood=invariant_scores,
            factor_codes=codes,
            active_axis_sets=active_axis_sets,
        )
        self.assertTrue(bool(deviations.eq(0).all().item()))
        self.assertTrue(all(float(value) == 0.0 for value in ranges))
        nuisance_scores = torch.stack(nuisance_rows).requires_grad_(True)
        locality = _active_set_fiber_huber_dispersion(
            torch=torch,
            full_grid_log_likelihood=nuisance_scores,
            factor_codes=codes,
            active_axis_sets=active_axis_sets,
            beta=1.0,
        )
        self.assertGreater(float(locality.detach()), 0.0)
        locality.backward()
        self.assertIsNotNone(nuisance_scores.grad)
        assert nuisance_scores.grad is not None
        self.assertGreater(float(nuisance_scores.grad.abs().sum()), 0.0)

    def test_locality_is_zero_when_score_only_depends_on_acted_axis(self) -> None:
        codes = self.batch.factor_ids.reshape(3, 64, 3)[0]
        scores = torch.stack(
            [codes[:, axis].to(dtype=torch.float32) for axis in range(3)]
        )
        deviations, ranges = _fiber_deviations(
            torch=torch,
            full_grid_log_likelihood=scores,
            factor_codes=codes,
            acted_axis_indices=(0, 1, 2),
        )
        self.assertTrue(bool(deviations.eq(0).all().item()))
        self.assertTrue(all(float(value) == 0.0 for value in ranges))
        locality = _fiber_huber_dispersion(
            torch=torch,
            full_grid_log_likelihood=scores,
            factor_codes=codes,
            acted_axis_indices=(0, 1, 2),
            beta=1.0,
        )
        self.assertEqual(float(locality), 0.0)

    def test_locality_penalizes_nuisance_dependence_and_has_gradient(self) -> None:
        codes = self.batch.factor_ids.reshape(3, 64, 3)[0]
        base = torch.stack(
            [
                codes[:, 0] + 0.5 * codes[:, 1],
                codes[:, 1] + 0.5 * codes[:, 2],
                codes[:, 2] + 0.5 * codes[:, 0],
            ]
        ).to(dtype=torch.float32)
        scores = base.clone().requires_grad_(True)
        locality = _fiber_huber_dispersion(
            torch=torch,
            full_grid_log_likelihood=scores,
            factor_codes=codes,
            acted_axis_indices=(0, 1, 2),
            beta=1.0,
        )
        self.assertGreater(float(locality.detach()), 0.0)
        locality.backward()
        self.assertIsNotNone(scores.grad)
        assert scores.grad is not None
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_teacher_kl_is_normalized_zero_for_equal_predictions(self) -> None:
        from prp_wm.neural import OutcomePrediction

        input_colors = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
        change_logits = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        color_logits = torch.zeros((1, 1, 4, 2, 2), dtype=torch.float32)
        prediction = OutcomePrediction(
            input_colors=input_colors,
            change_logits=change_logits,
            new_color_logits=color_logits,
        )
        log_probs = _outcome_log_probabilities(
            torch=torch,
            prediction=prediction,
        )
        normalizer = torch.logsumexp(log_probs, dim=2)
        self.assertTrue(
            bool(torch.allclose(normalizer, torch.zeros_like(normalizer), atol=1e-6))
        )
        kl = _teacher_categorical_kl(
            torch=torch,
            student_prediction=prediction,
            teacher_prediction=prediction,
        )
        self.assertAlmostEqual(float(kl), 0.0, places=6)

    def test_teacher_kl_penalizes_drift_and_has_gradient(self) -> None:
        from prp_wm.neural import OutcomePrediction

        input_colors = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
        teacher = OutcomePrediction(
            input_colors=input_colors,
            change_logits=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            new_color_logits=torch.zeros((1, 1, 4, 2, 2), dtype=torch.float32),
        )
        student_change = torch.full(
            (1, 1, 2, 2),
            2.0,
            dtype=torch.float32,
            requires_grad=True,
        )
        student = OutcomePrediction(
            input_colors=input_colors,
            change_logits=student_change,
            new_color_logits=torch.zeros(
                (1, 1, 4, 2, 2),
                dtype=torch.float32,
                requires_grad=True,
            ),
        )
        kl = _teacher_categorical_kl(
            torch=torch,
            student_prediction=student,
            teacher_prediction=teacher,
        )
        self.assertGreater(float(kl.detach()), 0.0)
        kl.backward()
        self.assertIsNotNone(student_change.grad)
        assert student_change.grad is not None
        self.assertGreater(float(student_change.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
    _geometry_panels,
