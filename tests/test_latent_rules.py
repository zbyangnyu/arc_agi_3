"""Fast checks for the privileged latent-rule executor ceiling."""

from __future__ import annotations

import unittest

try:
    import torch
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        SpatialOracleFactorExecutor,
        TiedSingleBelief,
        balanced_behavior_assignment_loss,
        canonicalize_rulegrid_tensor_batch,
        injective_assignment_loss,
        outcome_map,
        rule_program_factor_ids,
        rulegrid_tasks_to_oracle_factor_batch,
    )
    from prp_wm.neural import NeuralPRPConfig, OutcomePrediction
    from prp_wm.routed_executor import CanonicalRoleRoutedOracleFactorExecutor
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class OracleFactorExecutorTests(unittest.TestCase):
    def _config(self) -> NeuralPRPConfig:
        return NeuralPRPConfig(
            color_embedding=16,
            position_embedding=16,
            encoder_channels=16,
            encoder_resblocks=1,
            normalization_groups=4,
            action_embedding=16,
            rule_dim=32,
            attention_ffn=64,
            decoder_resblocks=1,
        )

    def test_factor_code_is_axis_compositional_not_program_lookup(self) -> None:
        from prp_wm.rulegrid import RuleProgram

        self.assertEqual(rule_program_factor_ids(RuleProgram.from_program_id(0)), (0, 0, 0))
        self.assertEqual(rule_program_factor_ids(RuleProgram.from_program_id(63)), (3, 3, 3))
        self.assertEqual(rule_program_factor_ids(RuleProgram.from_program_id(37)), (2, 1, 1))

    def test_selected_panel_and_privileged_factors_materialize(self) -> None:
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="factor-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 7, 20),
        )
        batch = rulegrid_tasks_to_oracle_factor_batch(
            tasks,
            diagnostic_indices=(0, 7, 20),
        )
        self.assertEqual(batch.states.shape, (2, 3, 8, 8))
        self.assertEqual(batch.targets.shape, (2, 3, 8, 8))
        self.assertEqual(batch.factor_ids.shape, (2, 3))
        self.assertEqual(
            batch.factor_ids[0].tolist(),
            list(rule_program_factor_ids(tasks[0].privileged.true_program)),
        )
        batch.validate(self._config())

    def test_privileged_palette_canonicalization_maps_roles_not_programs(self) -> None:
        assert torch is not None
        from dataclasses import fields

        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="factor-palette-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 4, 7),
        )
        raw = rulegrid_tasks_to_oracle_factor_batch(
            tasks,
            diagnostic_indices=(0, 4, 7),
        )
        canonical = rulegrid_tasks_to_oracle_factor_batch(
            tasks,
            diagnostic_indices=(0, 4, 7),
            canonicalize_palette=True,
        )
        expected_states = raw.states.clone()
        expected_targets = raw.targets.clone()
        for task_index, task in enumerate(tasks):
            for canonical_id, field in enumerate(
                fields(task.privileged.palette), start=1
            ):
                actual_color = getattr(task.privileged.palette, field.name)
                expected_states[task_index][raw.states[task_index] == actual_color] = canonical_id
                expected_targets[task_index][raw.targets[task_index] == actual_color] = canonical_id
        self.assertFalse(raw.palette_canonicalized)
        self.assertTrue(canonical.palette_canonicalized)
        self.assertTrue(torch.equal(canonical.states, expected_states))
        self.assertTrue(torch.equal(canonical.targets, expected_targets))
        self.assertTrue(torch.equal(canonical.factor_ids, raw.factor_ids))

    def test_full_panel_loss_is_finite_and_backpropagates(self) -> None:
        assert torch is not None
        from prp_wm.pilot import make_pilot_tasks

        torch.manual_seed(31)
        tasks = make_pilot_tasks(
            split="factor-gradient-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 7, 20),
        )
        batch = rulegrid_tasks_to_oracle_factor_batch(
            tasks,
            diagnostic_indices=(0, 7, 20),
        )
        model = OracleFactorExecutor(self._config())
        loss = model.losses(batch, balanced_weight=1.0)
        self.assertTrue(torch.isfinite(loss.total).item())
        prediction = model.predict_panel(batch)
        self.assertEqual(prediction.change_logits.shape, (6, 1, 8, 8))
        self.assertEqual(prediction.new_color_logits.shape, (6, 1, 16, 8, 8))
        loss.total.backward()
        self.assertIsNotNone(model.factor_embeddings[0].weight.grad)
        self.assertTrue(torch.isfinite(model.factor_embeddings[0].weight.grad).all().item())

    def test_predict_from_rule_latent_preserves_legacy_logits(self) -> None:
        assert torch is not None
        torch.manual_seed(37)
        model = OracleFactorExecutor(self._config()).eval()
        states = torch.randint(0, 16, (3, 8, 8), dtype=torch.long)
        actions = torch.tensor(
            [
                [0, 1, 2, 1],
                [1, 4, 3, 4],
                [0, 6, 5, 3],
            ],
            dtype=torch.long,
        )
        factors = torch.tensor(
            [[0, 1, 2], [3, 2, 1], [1, 0, 3]],
            dtype=torch.long,
        )
        legacy = model.predict(states, actions, factors)
        routed = CanonicalRoleRoutedOracleFactorExecutor(self._config())
        routed.load_state_dict(model.state_dict(), strict=True)
        explicit = routed.predict_from_rule_latent(
            states,
            actions,
            model.rule_latent(factors),
        )
        self.assertTrue(
            torch.equal(legacy.change_logits, explicit.change_logits)
        )
        self.assertTrue(
            torch.equal(
                legacy.new_color_logits,
                explicit.new_color_logits,
            )
        )

    def test_canonical_routed_executor_is_parameter_and_checkpoint_matched(self) -> None:
        assert torch is not None
        global_model = OracleFactorExecutor(self._config())
        routed_model = CanonicalRoleRoutedOracleFactorExecutor(
            self._config()
        )
        routed_model.load_state_dict(global_model.state_dict(), strict=True)
        self.assertEqual(
            tuple(global_model.state_dict()),
            tuple(routed_model.state_dict()),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in global_model.parameters()),
            sum(parameter.numel() for parameter in routed_model.parameters()),
        )

    def test_canonical_routing_detects_atomic_composite_and_neutral_events(self) -> None:
        assert torch is not None
        states = torch.zeros((4, 8, 8), dtype=torch.long)
        states[0, 1, 1] = 1
        states[1, 2, 2] = 5
        states[2, 3, 3] = 3
        actions = torch.tensor(
            [
                [[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                [[1, 2, 2, 4], [0, 0, 0, 1], [0, 0, 0, 1]],
                [[0, 3, 3, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            ],
            dtype=torch.long,
        )
        mask = torch.tensor(
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [True, False, False],
            ]
        )
        routed = CanonicalRoleRoutedOracleFactorExecutor.active_factor_mask(
            states,
            actions,
            mask,
        )
        self.assertEqual(
            routed.tolist(),
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ],
        )

    def test_canonical_routed_predictions_ignore_nuisance_factor_values(self) -> None:
        assert torch is not None
        torch.manual_seed(39)
        model = CanonicalRoleRoutedOracleFactorExecutor(
            self._config()
        ).eval()
        factor_bank = torch.cartesian_prod(
            torch.arange(4),
            torch.arange(4),
            torch.arange(4),
        )
        event_specs = (
            (0, 1, [0, 2, 2, 1]),
            (1, 5, [1, 2, 2, 4]),
            (2, 3, [0, 2, 2, 1]),
        )
        for active_axis, source_color, action in event_specs:
            states = torch.zeros((64, 8, 8), dtype=torch.long)
            states[:, 2, 2] = source_color
            actions = torch.tensor(
                [action],
                dtype=torch.long,
            ).expand(64, -1)
            prediction = model.predict(states, actions, factor_bank)
            for factor_value in range(4):
                fiber = factor_bank[:, active_axis].eq(factor_value)
                self.assertTrue(
                    torch.equal(
                        prediction.change_logits[fiber],
                        prediction.change_logits[fiber][:1].expand_as(
                            prediction.change_logits[fiber]
                        ),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        prediction.new_color_logits[fiber],
                        prediction.new_color_logits[fiber][:1].expand_as(
                            prediction.new_color_logits[fiber]
                        ),
                    )
                )

        neutral_states = torch.zeros((64, 8, 8), dtype=torch.long)
        neutral_actions = torch.tensor(
            [[0, 0, 0, 1]],
            dtype=torch.long,
        ).expand(64, -1)
        neutral = model.predict(
            neutral_states,
            neutral_actions,
            factor_bank,
        )
        self.assertTrue(
            torch.equal(
                neutral.change_logits,
                neutral.change_logits[:1].expand_as(neutral.change_logits),
            )
        )
        self.assertTrue(
            torch.equal(
                neutral.new_color_logits,
                neutral.new_color_logits[:1].expand_as(
                    neutral.new_color_logits
                ),
            )
        )

    def test_canonical_routed_all_axis_composite_matches_global_executor(self) -> None:
        assert torch is not None
        torch.manual_seed(43)
        global_model = OracleFactorExecutor(self._config()).eval()
        routed_model = CanonicalRoleRoutedOracleFactorExecutor(
            self._config()
        ).eval()
        routed_model.load_state_dict(global_model.state_dict(), strict=True)
        states = torch.zeros((1, 8, 8), dtype=torch.long)
        states[0, 1, 1] = 1
        states[0, 2, 2] = 5
        states[0, 3, 3] = 3
        actions = torch.tensor(
            [[[0, 1, 1, 1], [1, 2, 2, 4], [0, 3, 3, 1]]],
            dtype=torch.long,
        )
        mask = torch.ones((1, 3), dtype=torch.bool)
        factors = torch.tensor([[2, 1, 3]], dtype=torch.long)
        global_prediction = global_model.predict(
            states,
            actions,
            factors,
            mask,
        )
        routed_prediction = routed_model.predict(
            states,
            actions,
            factors,
            mask,
        )
        self.assertTrue(
            torch.equal(
                global_prediction.change_logits,
                routed_prediction.change_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                global_prediction.new_color_logits,
                routed_prediction.new_color_logits,
            )
        )

    def test_outcome_map_compares_copy_with_changed_colors(self) -> None:
        assert torch is not None
        inputs = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
        change_logits = torch.full((1, 1, 2, 2), -8.0)
        change_logits[0, 0, 0, 1] = 8.0
        colors = torch.zeros((1, 1, 6, 2, 2))
        colors[0, 0, 5, 0, 1] = 12.0
        prediction = OutcomePrediction(
            input_colors=inputs,
            change_logits=change_logits,
            new_color_logits=colors,
        )
        expected = inputs.clone()
        expected[0, 0, 1] = 5
        self.assertTrue(torch.equal(outcome_map(prediction)[:, 0], expected))

    def test_injective_assignment_requires_all_four_classes(self) -> None:
        assert torch is not None
        cost = torch.full((1, 4, 4), 10.0)
        cost[0, torch.arange(4), torch.arange(4)] = 0.0
        self.assertEqual(float(injective_assignment_loss(cost)), 0.0)
        collapsed = torch.zeros((1, 4, 4))
        collapsed[:, :, 1:] = 4.0
        self.assertAlmostEqual(float(injective_assignment_loss(collapsed)), 3.0)

    def test_spatial_executor_is_invariant_to_composite_atom_order(self) -> None:
        assert torch is not None
        torch.manual_seed(41)
        config = self._config()
        model = SpatialOracleFactorExecutor(config).eval()
        states = torch.zeros((1, 8, 8), dtype=torch.long)
        factors = torch.tensor([[1, 2, 3]], dtype=torch.long)
        actions = torch.tensor(
            [[[0, 1, 2, 3], [1, 5, 4, 4]]], dtype=torch.long
        )
        mask = torch.ones((1, 2), dtype=torch.bool)
        first = model.predict(states, actions, factors, mask)
        second = model.predict(states, actions.flip(1), factors, mask.flip(1))
        self.assertTrue(torch.allclose(first.change_logits, second.change_logits))
        self.assertTrue(torch.allclose(first.new_color_logits, second.new_color_logits))

    def test_spatial_executor_panel_loss_backpropagates(self) -> None:
        assert torch is not None
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="spatial-factor-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 12, 20),
        )
        batch = rulegrid_tasks_to_oracle_factor_batch(
            tasks,
            diagnostic_indices=(0, 12, 20),
        )
        model = SpatialOracleFactorExecutor(self._config())
        loss = model.losses(batch)
        self.assertTrue(torch.isfinite(loss.total).item())
        loss.total.backward()
        self.assertIsNotNone(model.spatial_action_encoder.project[0].weight.grad)

    def test_full_batch_palette_canonicalization_preserves_non_grid_fields(self) -> None:
        assert torch is not None
        from prp_wm.neural import rulegrid_tasks_to_tensor_batch
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="canonical-latent-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 12, 20),
        )
        raw = rulegrid_tasks_to_tensor_batch(
            tasks,
            include_behavior_targets=True,
            diagnostic_indices=(0, 12, 20),
        )
        canonical = canonicalize_rulegrid_tensor_batch(raw, tasks)
        self.assertTrue(torch.equal(canonical.support_actions, raw.support_actions))
        self.assertTrue(torch.equal(canonical.query_actions, raw.query_actions))
        self.assertTrue(torch.equal(canonical.support_mask, raw.support_mask))
        self.assertEqual(canonical.behavior_targets.shape, raw.behavior_targets.shape)
        canonical.validate(self._config())

    def test_tied_single_belief_stays_identical_and_balanced_loss_is_finite(self) -> None:
        assert torch is not None
        from prp_wm.latent_rules import rulegrid_tasks_to_canonical_behavior_batch
        from prp_wm.pilot import make_pilot_tasks

        torch.manual_seed(47)
        tasks = make_pilot_tasks(
            split="tied-latent-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 4, 8, 12),
        )
        batch = rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )
        self.assertIsNone(batch.query_targets)
        model = TiedSingleBelief(self._config())
        inference = model.infer_support(batch)
        self.assertTrue(
            torch.equal(
                inference.modes,
                inference.modes[:, :1].expand_as(inference.modes),
            )
        )
        loss = balanced_behavior_assignment_loss(model, batch, inference)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.updater.weight_hh.grad)


if __name__ == "__main__":
    unittest.main()
