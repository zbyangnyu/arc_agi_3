"""Fast correctness checks for the optional Stage-1 neural scaffold."""

from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
    from prp_wm.neural import (
        NeuralPRPConfig,
        PersistentK4,
        encode_public_action,
        make_toy_rulegrid_batch,
        rulegrid_tasks_to_tensor_batch,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional Stage-1 dependency")
class NeuralScaffoldTests(unittest.TestCase):
    def _config(self) -> NeuralPRPConfig:
        # Small dimensions keep unit tests quick; default config remains the
        # 64-channel / 128-rule-dimension architecture specification.
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

    def test_grid_embedding_dimensions_are_additive_and_must_match(self) -> None:
        assert torch is not None
        config = self._config()
        model = PersistentK4(config)
        self.assertEqual(model.grid_encoder.stem.in_channels, config.color_embedding)
        with self.assertRaisesRegex(ValueError, "must match"):
            NeuralPRPConfig(color_embedding=16, position_embedding=32)

    def test_full_loss_path_is_finite_and_backpropagates(self) -> None:
        assert torch is not None
        torch.manual_seed(17)
        config = self._config()
        model = PersistentK4(config)
        batch = make_toy_rulegrid_batch(
            batch_size=2,
            support_steps=2,
            query_count=2,
            config=config,
            generator=torch.Generator().manual_seed(19),
        )
        result = model.losses(batch)
        self.assertTrue(torch.isfinite(result.total).item())
        self.assertEqual(result.inference.modes.shape, (2, 4, 32))
        self.assertEqual(result.inference.log_weights.shape, (2, 4))
        self.assertTrue(
            torch.allclose(
                result.inference.weights.sum(dim=-1), torch.ones(2), atol=1e-5
            )
        )
        result.total.backward()
        self.assertIsNotNone(model.initial_modes.grad)
        self.assertTrue(torch.isfinite(model.initial_modes.grad).all().item())

    def test_original_color_is_scored_only_by_no_change_branch(self) -> None:
        assert torch is not None
        config = self._config()
        model = PersistentK4(config)
        batch = make_toy_rulegrid_batch(batch_size=2, config=config)
        inference = model.initial_state(2)
        prediction = model.predict(
            batch.support_states[:, 0],
            batch.support_actions[:, 0],
            inference.modes,
        )
        unchanged = prediction.log_prob_cells(batch.support_states[:, 0])
        self.assertTrue(torch.isfinite(unchanged).all().item())
        expected = torch.nn.functional.logsigmoid(-prediction.change_logits)
        self.assertTrue(torch.allclose(unchanged, expected, atol=1e-6))

    def test_public_action_adapter_does_not_need_hidden_rule_data(self) -> None:
        assert torch is not None

        class Move:
            kind = "MOVE"
            coord = (3, 4)
            direction = "LEFT"

        encoded = encode_public_action(Move())
        self.assertEqual(encoded.tolist(), [[0, 3, 4, 2]])

    def test_rulegrid_direction_and_composite_encoding(self) -> None:
        assert torch is not None
        from prp_wm.rulegrid import ActionKind, CompositeAction, Direction, GridAction

        action = CompositeAction(
            (
                GridAction(ActionKind.MOVE, (1, 2), Direction.NORTH),
                GridAction(ActionKind.ACTIVATE, (5, 6)),
                GridAction(ActionKind.MOVE, (4, 3), Direction.EAST),
            )
        )
        # Direction IDs are N/S/W/E -> 0/1/2/3; ACTIVATE uses public sentinel 4.
        self.assertEqual(
            encode_public_action(action).tolist(),
            [[0, 1, 2, 0], [1, 5, 6, 4], [0, 4, 3, 3]],
        )

    def test_rulegrid_task_adapter_has_a_finite_training_loss(self) -> None:
        """The real RuleGrid boundary must work without leaking a rule ID."""

        assert torch is not None
        from prp_wm.rulegrid import Axis, RuleProgram, make_rulegrid_task

        tasks = (
            make_rulegrid_task(RuleProgram.from_program_id(5), Axis.COLLISION, 0),
            make_rulegrid_task(RuleProgram.from_program_id(41), Axis.COLLISION, 1),
        )
        batch = rulegrid_tasks_to_tensor_batch(tasks, prefix_length=6)
        self.assertEqual(batch.support_states.shape, (2, 6, 8, 8))
        self.assertEqual(batch.query_states.shape, (2, 24, 8, 8))
        self.assertEqual(batch.behavior_targets.shape, (2, 4, 24, 8, 8))
        self.assertEqual(batch.behavior_mass.shape, (2, 4))
        result = PersistentK4(self._config()).losses(batch)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_rulegrid_task_adapter_selects_only_requested_diagnostics(self) -> None:
        """Composition targets can be excluded before any tensor is built."""

        assert torch is not None
        from prp_wm.rulegrid import Axis, RuleProgram, make_rulegrid_task

        task = make_rulegrid_task(
            RuleProgram.from_program_id(23), Axis.RELATION, replicate=9
        )
        selected = (0, 7, 20)
        batch = rulegrid_tasks_to_tensor_batch(
            (task,), diagnostic_indices=selected, include_behavior_targets=True
        )
        self.assertEqual(batch.query_states.shape, (1, len(selected), 8, 8))
        self.assertEqual(batch.query_targets.shape, (1, len(selected), 8, 8))
        self.assertEqual(batch.behavior_targets.shape[2], len(selected))
        self.assertTrue(
            torch.equal(
                batch.query_states[0, 1],
                torch.tensor(task.inference.diagnostics[7].state, dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.query_targets[0, 2],
                torch.tensor(task.privileged.diagnostic_targets[20], dtype=torch.long),
            )
        )

    def test_rulegrid_task_adapter_rejects_ambiguous_diagnostic_selection(self) -> None:
        assert torch is not None
        from prp_wm.rulegrid import Axis, RuleProgram, make_rulegrid_task

        task = make_rulegrid_task(
            RuleProgram.from_program_id(23), Axis.RELATION, replicate=9
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            rulegrid_tasks_to_tensor_batch((task,), diagnostic_indices=(0, 0))
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            rulegrid_tasks_to_tensor_batch((task,), diagnostic_indices=())
        with self.assertRaisesRegex(ValueError, "must lie"):
            rulegrid_tasks_to_tensor_batch((task,), diagnostic_indices=(24,))

    def test_nontriple_adapter_does_not_read_guarded_triple_targets(self) -> None:
        """A selected train panel must not even index evaluation-only targets."""

        assert torch is not None
        from prp_wm.rulegrid import Axis, RuleGridTask, RuleProgram, make_rulegrid_task

        class GuardedTargets(tuple):
            def __getitem__(self, index: object) -> object:
                if isinstance(index, int) and index >= 21:
                    raise AssertionError("adapter attempted to read a triple target")
                return super().__getitem__(index)  # type: ignore[index]

        task = make_rulegrid_task(
            RuleProgram.from_program_id(51), Axis.TRIGGER, replicate=5
        )
        guarded = RuleGridTask(
            inference=task.inference,
            privileged=replace(
                task.privileged,
                diagnostic_targets=GuardedTargets(task.privileged.diagnostic_targets),
            ),
        )
        batch = rulegrid_tasks_to_tensor_batch(
            (guarded,), diagnostic_indices=tuple(range(21)), include_behavior_targets=True
        )
        self.assertEqual(batch.query_targets.shape[1], 21)

    def test_pilot_subset_train_and_triple_eval_batches_materialize_end_to_end(self) -> None:
        """The lazy RuleGrid builder and tensor adapter agree on canonical indices."""

        assert torch is not None
        from prp_wm.pilot import (
            NONTRIPLE_DIAGNOSTIC_INDICES,
            TRIPLE_DIAGNOSTIC_INDICES,
            make_pilot_tensor_batch,
        )

        train_batch = make_pilot_tensor_batch(
            split="pilot-test-train",
            master_seed=2026071601,
            start=0,
            count=1,
            diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
            include_behavior_targets=True,
        )
        train_batch.validate(self._config())
        self.assertEqual(train_batch.query_targets.shape, (1, 21, 8, 8))
        self.assertIsNotNone(train_batch.behavior_targets)
        self.assertEqual(train_batch.behavior_targets.shape[2], 21)

        eval_batch = make_pilot_tensor_batch(
            split="pilot-test-eval",
            master_seed=2026071601,
            start=0,
            count=1,
            diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
            include_behavior_targets=False,
        )
        eval_batch.validate(self._config())
        self.assertEqual(eval_batch.query_targets.shape, (1, 3, 8, 8))
        self.assertIsNone(eval_batch.behavior_targets)


if __name__ == "__main__":
    unittest.main()
