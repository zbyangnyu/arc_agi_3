"""Invariants for public-only persistent K4 rule-set abstraction."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

try:
    import torch
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
    from prp_wm.public_version_k4 import (
        FactorizedPublicVersionSpaceCausalK4,
        HistoryConditionedProbeFactorBeliefCausalK4,
        PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4,
        ProbeAwareSymmetryFactorBeliefCausalK4,
        PublicVersionSpaceCausalK4,
        SymmetryAwareFactorBeliefCausalK4,
        TransitionEvidencePublicVersionSpaceCausalK4,
        TranslationInvariantHistoryProbeFactorBeliefCausalK4,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class PublicVersionSpaceK4Tests(unittest.TestCase):
    def _model(self) -> PublicVersionSpaceCausalK4:
        assert torch is not None
        config = NeuralPRPConfig(
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
        return PublicVersionSpaceCausalK4(
            OracleFactorExecutor(config),
            attention_layers=2,
        )

    def _batch(self, count: int = 2):
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="public-version-k4-test",
            master_seed=2026072302,
            start=0,
            count=count,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    def test_canonical_targets_use_slot_value_on_varying_axis(self) -> None:
        assert torch is not None
        model = self._model()
        mask = torch.zeros(2, 64, dtype=torch.bool)
        mask[0, :4] = True  # relation varies -> bank 0,1,2,3
        mask[1, torch.tensor([5, 21, 37, 53])] = True  # collision varies
        targets, axes, compatible = model._canonical_public_targets(mask)
        self.assertTrue(torch.equal(targets[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(targets[1], torch.tensor([5, 21, 37, 53])))
        self.assertTrue(torch.equal(axes, torch.tensor([2, 0])))
        self.assertTrue(torch.equal(targets, compatible))

    def test_loss_ignores_query_behavior_and_freezes_executor(self) -> None:
        assert torch is not None
        torch.manual_seed(907)
        model = self._model().train()
        batch = self._batch(count=2)
        changed = replace(
            batch,
            query_states=(batch.query_states + 1) % model.config.num_colors,
            query_actions=batch.query_actions.roll(1, dims=1),
            behavior_targets=(batch.behavior_targets + 2) % model.config.num_colors,
            behavior_mass=batch.behavior_mass.roll(1, dims=1),
        )
        mask = torch.zeros(2, 64, dtype=torch.bool)
        mask[:, :4] = True
        with patch.object(model, "public_support_exact_mask", return_value=mask):
            first = model.hard_public_version_space_loss(batch)
            second = model.hard_public_version_space_loss(changed)
        self.assertTrue(torch.allclose(first.total, second.total))
        self.assertTrue(
            torch.allclose(
                first.inference.factor_logits,
                second.inference.factor_logits,
            )
        )
        first.total.backward()
        public_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if not name.startswith("executor.") and parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in public_gradients))
        self.assertTrue(
            all(
                gradient is None or torch.isfinite(gradient).all().item()
                for gradient in public_gradients
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.executor.parameters())
        )

    def test_external_symbolic_mask_bypasses_executor_teacher(self) -> None:
        assert torch is not None
        model = self._model().train()
        batch = self._batch(count=1)
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[:, :4] = True
        with patch.object(
            model,
            "public_support_exact_mask",
            side_effect=AssertionError("executor teacher must not be called"),
        ):
            loss = model.hard_public_version_space_loss(
                batch,
                compatible_mask=mask,
            )
        self.assertTrue(torch.isfinite(loss.total).item())
        self.assertTrue(torch.equal(loss.compatible_mask, mask))

    def test_raw_support_adapter_accepts_inference_views_only(self) -> None:
        assert torch is not None
        from prp_wm.pilot import make_pilot_tasks
        from scripts.run_gram_public_coverage_finetune import (
            _raw_public_support_batch_from_views,
        )

        tasks = make_pilot_tasks(
            split="raw-public-version-k4-test",
            master_seed=2026072303,
            start=0,
            count=2,
            diagnostic_indices=(0,),
        )
        views = tuple(task.inference for task in tasks)
        batch = _raw_public_support_batch_from_views(
            torch,
            views,
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [[transition.state for transition in view.support[:6]] for view in views],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(batch.support_states, expected))
        self.assertIsNone(batch.query_states)
        self.assertEqual(tuple(batch.support_states.shape), (2, 6, 8, 8))

    def test_independent_support_encoders_do_not_unfreeze_teacher(self) -> None:
        assert torch is not None
        base = self._model()
        model = PublicVersionSpaceCausalK4(
            base.executor,
            attention_layers=2,
            independent_support_encoders=True,
        )
        assert model.support_grid_encoder is not None
        assert model.support_action_encoder is not None
        self.assertIsNot(model.support_grid_encoder, model.executor.grid_encoder)
        self.assertIsNot(model.support_action_encoder, model.executor.action_encoder)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.support_grid_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.support_action_encoder.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.executor.parameters())
        )

    def test_raw_color_augmentation_is_consistent_and_preserves_background(self) -> None:
        assert torch is not None
        from scripts.run_public_version_space_k4 import _permute_raw_support_colors

        batch = self._model()._support_only_batch(self._batch(count=2))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        augmented = _permute_raw_support_colors(
            torch,
            batch,
            generator=generator,
            num_colors=16,
        )
        self.assertTrue(
            torch.equal(augmented.support_states == 0, batch.support_states == 0)
        )
        for task_index in range(batch.batch_size):
            original = torch.cat(
                (
                    batch.support_states[task_index].flatten(),
                    batch.support_targets[task_index].flatten(),
                )
            )
            changed = torch.cat(
                (
                    augmented.support_states[task_index].flatten(),
                    augmented.support_targets[task_index].flatten(),
                )
            )
            for color in torch.unique(original):
                mapped = torch.unique(changed[original == color])
                self.assertEqual(mapped.numel(), 1)

    def test_factorized_head_composes_exactly_four_codes(self) -> None:
        assert torch is not None
        base = self._model()
        model = FactorizedPublicVersionSpaceCausalK4(
            base.executor,
            attention_layers=2,
            independent_support_encoders=True,
        ).train()
        batch = self._batch(count=2)
        mask = torch.zeros(2, 64, dtype=torch.bool)
        mask[0, :4] = True
        mask[1, torch.tensor([5, 21, 37, 53])] = True
        loss = model.hard_public_version_space_loss(
            batch,
            compatible_mask=mask,
        )
        self.assertTrue(torch.isfinite(loss.total).item())
        for row in loss.inference.factor_ids:
            self.assertEqual(torch.unique(row, dim=0).shape[0], 4)
            varying = [
                axis
                for axis in range(3)
                if torch.unique(row[:, axis]).numel() == 4
            ]
            self.assertEqual(len(varying), 1)
        loss.total.backward()
        self.assertIsNotNone(model.varying_axis_head.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.cross_layers.parameters())
        )

    def test_transition_evidence_head_routes_public_observations(self) -> None:
        assert torch is not None
        from prp_wm.pilot import make_pilot_tasks
        from scripts.run_public_version_space_k4 import (
            _symbolic_transition_evidence_targets,
            _symbolic_version_space_mask,
        )

        base = self._model()
        model = TransitionEvidencePublicVersionSpaceCausalK4(
            base.executor,
            attention_layers=2,
            independent_support_encoders=True,
        ).train()
        tasks = make_pilot_tasks(
            split="transition-evidence-k4-test",
            master_seed=2026072304,
            start=0,
            count=2,
            diagnostic_indices=(0,),
        )
        from scripts.run_gram_public_coverage_finetune import (
            _raw_public_support_batch_from_views,
        )

        batch = _raw_public_support_batch_from_views(
            torch,
            tuple(task.inference for task in tasks),
            device=torch.device("cpu"),
        )
        compatible = _symbolic_version_space_mask(
            torch,
            model,
            tasks,
            device=torch.device("cpu"),
        )
        axes, values = _symbolic_transition_evidence_targets(
            torch,
            tasks,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.all((axes < 3).sum(dim=1) == 2))
        self.assertTrue(torch.all((axes == 3).sum(dim=1) == 4))
        loss = model.hard_public_version_space_loss(
            batch,
            compatible_mask=compatible,
            evidence_axis_targets=axes,
            evidence_value_targets=values,
        )
        self.assertTrue(torch.isfinite(loss.total).item())
        self.assertEqual(tuple(loss.inference.factor_ids.shape), (2, 4, 3))

    def test_symmetry_factor_belief_keeps_toggle_recolor_pair(self) -> None:
        assert torch is not None
        from prp_wm.rulegrid import (
            Axis,
            Collision,
            Relation,
            RuleProgram,
            Trigger,
            make_rulegrid_task,
        )
        from scripts.run_gram_public_coverage_finetune import (
            _raw_public_support_batch_from_views,
        )
        from scripts.run_public_version_space_k4 import (
            _symmetry_expanded_version_space_mask,
            _symmetry_transition_evidence_targets,
        )

        task = make_rulegrid_task(
            RuleProgram(Collision.STOP, Trigger.TOGGLE, Relation.NONE),
            Axis.COLLISION,
            0,
            split="symmetry-factor-belief-test",
            diagnostic_indices=(0,),
        )
        base = self._model()
        model = SymmetryAwareFactorBeliefCausalK4(
            base.executor,
            independent_support_encoders=True,
        ).train()
        batch = _raw_public_support_batch_from_views(
            torch,
            (task.inference,),
            device=torch.device("cpu"),
        )
        compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            (task,),
            device=torch.device("cpu"),
        )
        axes, value_mask = _symmetry_transition_evidence_targets(
            torch,
            (task,),
            device=torch.device("cpu"),
        )
        self.assertEqual(int(compatible.sum()), 8)
        trigger_step = int(torch.nonzero(axes[0] == 1)[0])
        self.assertTrue(
            torch.equal(
                value_mask[0, trigger_step, 1],
                torch.tensor([True, False, False, True]),
            )
        )
        loss = model.symmetry_aware_factor_belief_loss(
            batch,
            compatible_mask=compatible,
            evidence_axis_targets=axes,
            evidence_value_target_mask=value_mask,
        )
        self.assertTrue(torch.isfinite(loss.total).item())
        self.assertTrue(
            torch.allclose(
                loss.belief.factor_probabilities.sum(dim=-1),
                torch.ones(1, 3),
            )
        )

    def test_conditional_active_innovation_shrinks_toggle_recolor_pair(self) -> None:
        assert torch is not None
        from prp_wm.rulegrid import (
            Axis,
            Collision,
            Relation,
            RuleProgram,
            Trigger,
            make_rulegrid_task,
        )
        from scripts.run_public_version_space_k4 import (
            ControllerHistory,
            _active_break_history,
            _conditional_active_innovation_targets,
            _conditional_probe_innovation_targets,
            _informative_then_replay_controller_history,
            _informative_semantic_composite_controller_history,
            _neutral_replay_controller_history,
            _semantic_composite_replay_controller_history,
            _symmetry_expanded_version_space_mask,
            _symmetry_transition_evidence_targets,
            _translated_move_semantic_composite_replay_controller_history,
            _translated_move_informative_composite_controller_history,
        )

        tasks = tuple(
            make_rulegrid_task(
                RuleProgram(Collision.STOP, trigger, Relation.NONE),
                Axis.COLLISION,
                0,
                split="conditional-active-innovation-test",
                diagnostic_indices=(0,),
            )
            for trigger in (Trigger.TOGGLE, Trigger.RECOLOR)
        )
        base = self._model()
        model = SymmetryAwareFactorBeliefCausalK4(
            base.executor,
            independent_support_encoders=True,
        )
        device = torch.device("cpu")
        histories = tuple(_active_break_history(task) for task in tasks)
        prefix_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
        )
        active_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
            histories=histories,
        )
        prefix_factor_sets = model._factor_value_masks(
            model.factor_bank,
            prefix_compatible,
        )
        active_factor_sets = model._factor_value_masks(
            model.factor_bank,
            active_compatible,
        )

        self.assertTrue(torch.equal(prefix_compatible.sum(dim=-1), torch.tensor([8, 8])))
        self.assertTrue(torch.equal(active_compatible.sum(dim=-1), torch.tensor([4, 4])))
        self.assertTrue(
            torch.equal(
                prefix_factor_sets[:, 1],
                torch.tensor(
                    [
                        [True, False, False, True],
                        [True, False, False, True],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                active_factor_sets[:, 1],
                torch.tensor(
                    [
                        [True, False, False, False],
                        [False, False, False, True],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                prefix_factor_sets[:, (0, 2)],
                active_factor_sets[:, (0, 2)],
            )
        )

        standalone_axes, standalone_values = _symmetry_transition_evidence_targets(
            torch,
            tasks,
            device=device,
            histories=histories,
        )
        innovation_axes, innovation_values = _conditional_active_innovation_targets(
            torch,
            model,
            tasks,
            device=device,
            histories=histories,
        )
        # TOGGLE changes the frame and is already identifiable in isolation;
        # RECOLOR is a no-op in isolation, so only the prefix-relative target
        # reveals that it eliminates the TOGGLE interpretation.
        self.assertTrue(torch.equal(standalone_axes[:, -1], torch.tensor([1, 3])))
        self.assertTrue(torch.equal(innovation_axes[:, -1], torch.tensor([1, 1])))
        self.assertTrue(
            torch.equal(
                innovation_values[:, -1, 1],
                active_factor_sets[:, 1],
            )
        )
        self.assertTrue(
            torch.equal(innovation_axes[:, :-1], standalone_axes[:, :-1])
        )
        self.assertTrue(
            torch.equal(innovation_values[:, :-1], standalone_values[:, :-1])
        )

        # Provenance, not position, identifies the probe result: append an
        # ordinary observation after it and keep the conditional label at -2.
        extended = tuple(
            ControllerHistory(
                history + (history[0],),
                (False,) * 6 + (True, False),
            )
            for history in histories
        )
        extended_histories = tuple(item.transitions for item in extended)
        extended_standalone_axes, extended_standalone_values = (
            _symmetry_transition_evidence_targets(
                torch,
                tasks,
                device=device,
                histories=extended_histories,
            )
        )
        extended_axes, extended_values = _conditional_probe_innovation_targets(
            torch,
            model,
            tasks,
            device=device,
            controller_histories=extended,
        )
        self.assertTrue(torch.equal(extended_axes[:, -2], torch.tensor([1, 1])))
        self.assertTrue(
            torch.equal(
                extended_axes[:, -1],
                extended_standalone_axes[:, -1],
            )
        )
        self.assertTrue(
            torch.equal(
                extended_values[:, -1],
                extended_standalone_values[:, -1],
            )
        )

        neutral = tuple(_neutral_replay_controller_history(task) for task in tasks)
        neutral_axes, neutral_values = _conditional_probe_innovation_targets(
            torch,
            model,
            tasks,
            device=device,
            controller_histories=neutral,
        )
        neutral_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
            histories=tuple(item.transitions for item in neutral),
        )
        self.assertTrue(torch.equal(neutral_compatible, prefix_compatible))
        self.assertTrue(torch.equal(neutral_axes[:, -1], torch.tensor([3, 3])))
        self.assertFalse(bool(neutral_values[:, -1].any().item()))

        from scripts.run_gram_public_coverage_finetune import (
            _raw_public_history_batch,
        )

        neutral_batch = _raw_public_history_batch(
            torch,
            tuple(item.transitions for item in neutral),
            device=device,
        )
        neutral_repeated = (
            HistoryConditionedProbeFactorBeliefCausalK4
            ._has_identical_prior_transition(neutral_batch)
        )
        self.assertTrue(bool(neutral_repeated[:, -1].all().item()))

        hard_replay = tuple(
            _informative_then_replay_controller_history(task) for task in tasks
        )
        hard_axes, hard_values = _conditional_probe_innovation_targets(
            torch,
            model,
            tasks,
            device=device,
            controller_histories=hard_replay,
        )
        self.assertTrue(torch.equal(hard_axes[:, -2], torch.tensor([1, 1])))
        self.assertTrue(torch.equal(hard_axes[:, -1], torch.tensor([3, 3])))
        self.assertFalse(bool(hard_values[:, -1].any().item()))
        hard_batch = _raw_public_history_batch(
            torch,
            tuple(item.transitions for item in hard_replay),
            device=device,
        )
        hard_repeated = (
            HistoryConditionedProbeFactorBeliefCausalK4
            ._has_identical_prior_transition(hard_batch)
        )
        self.assertFalse(bool(hard_repeated[:, -2].any().item()))
        self.assertTrue(bool(hard_repeated[:, -1].all().item()))

        semantic_composite = tuple(
            _semantic_composite_replay_controller_history(task)
            for task in tasks
        )
        semantic_axes, semantic_values = _conditional_probe_innovation_targets(
            torch,
            model,
            tasks,
            device=device,
            controller_histories=semantic_composite,
        )
        semantic_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
            histories=tuple(item.transitions for item in semantic_composite),
        )
        self.assertTrue(torch.equal(semantic_compatible, prefix_compatible))
        self.assertTrue(torch.equal(semantic_axes[:, -1], torch.tensor([3, 3])))
        self.assertFalse(bool(semantic_values[:, -1].any().item()))
        self.assertTrue(
            all(
                item.is_agent_probe_result == (False,) * 6 + (True,)
                for item in semantic_composite
            )
        )
        self.assertTrue(
            all(
                item.transitions[-1] not in item.transitions[:-1]
                for item in semantic_composite
            )
        )
        semantic_batch = _raw_public_history_batch(
            torch,
            tuple(item.transitions for item in semantic_composite),
            device=device,
        )
        semantic_repeated = (
            HistoryConditionedProbeFactorBeliefCausalK4
            ._has_identical_prior_transition(semantic_batch)
        )
        self.assertFalse(bool(semantic_repeated[:, -1].any().item()))

        heldout_geometry = tuple(
            _translated_move_semantic_composite_replay_controller_history(task)
            for task in tasks
        )
        heldout_geometry_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
            histories=tuple(item.transitions for item in heldout_geometry),
        )
        self.assertTrue(
            torch.equal(heldout_geometry_compatible, prefix_compatible)
        )
        self.assertTrue(
            all(
                item.transitions[-1] not in item.transitions[:-1]
                for item in heldout_geometry
            )
        )

        informative_composite = tuple(
            _informative_semantic_composite_controller_history(task)
            for task in tasks
        )
        informative_axes, informative_values = (
            _conditional_probe_innovation_targets(
                torch,
                model,
                tasks,
                device=device,
                controller_histories=informative_composite,
            )
        )
        informative_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
            histories=tuple(item.transitions for item in informative_composite),
        )
        self.assertTrue(torch.equal(informative_compatible, active_compatible))
        self.assertTrue(torch.equal(informative_axes[:, -1], torch.tensor([1, 1])))
        self.assertTrue(
            torch.equal(informative_values[:, -1, 1], active_factor_sets[:, 1])
        )
        informative_geometry = tuple(
            _translated_move_informative_composite_controller_history(task)
            for task in tasks
        )
        informative_geometry_compatible = _symmetry_expanded_version_space_mask(
            torch,
            model,
            tasks,
            device=device,
            histories=tuple(item.transitions for item in informative_geometry),
        )
        self.assertTrue(
            torch.equal(informative_geometry_compatible, active_compatible)
        )

    def test_translation_invariant_head_freezes_zero_position_embeddings(
        self,
    ) -> None:
        assert torch is not None
        base = self._model()
        with self.assertRaisesRegex(ValueError, "independent support encoders"):
            TranslationInvariantHistoryProbeFactorBeliefCausalK4(
                base.executor,
                independent_support_encoders=False,
            )
        model = TranslationInvariantHistoryProbeFactorBeliefCausalK4(
            base.executor,
            independent_support_encoders=True,
        )
        assert model.support_grid_encoder is not None
        assert model.support_action_encoder is not None
        embeddings = (
            model.support_grid_encoder.row_embedding,
            model.support_grid_encoder.column_embedding,
            model.support_action_encoder.row_embedding,
            model.support_action_encoder.column_embedding,
        )
        for embedding in embeddings:
            self.assertFalse(embedding.weight.requires_grad)
            self.assertFalse(bool(embedding.weight.any().item()))

    def test_palette_invariant_atom_match_is_geometry_and_order_invariant(
        self,
    ) -> None:
        assert torch is not None
        from prp_wm.rulegrid import (
            Axis,
            Collision,
            Relation,
            RuleProgram,
            Trigger,
            make_rulegrid_task,
        )
        from scripts.run_gram_public_coverage_finetune import (
            _raw_public_history_batch,
        )
        from scripts.run_public_version_space_k4 import (
            _controller_probe_result_mask,
            _informative_semantic_composite_controller_history,
            _semantic_composite_replay_controller_history,
            _translated_move_semantic_composite_replay_controller_history,
            _translated_move_informative_composite_controller_history,
        )

        tasks = tuple(
            make_rulegrid_task(
                RuleProgram(Collision.STOP, trigger, Relation.NONE),
                Axis.COLLISION,
                0,
                split="relative-event-invariance-test",
                diagnostic_indices=(0,),
            )
            for trigger in (Trigger.TOGGLE, Trigger.RECOLOR)
        )
        base = self._model()
        model = PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4(
            base.executor,
            independent_support_encoders=True,
        ).eval()

        history_pairs = (
            (
                "neutral",
                tuple(
                    _semantic_composite_replay_controller_history(task)
                    for task in tasks
                ),
                tuple(
                    _translated_move_semantic_composite_replay_controller_history(
                        task
                    )
                    for task in tasks
                ),
            ),
            (
                "informative",
                tuple(
                    _informative_semantic_composite_controller_history(task)
                    for task in tasks
                ),
                tuple(
                    _translated_move_informative_composite_controller_history(
                        task
                    )
                    for task in tasks
                ),
            ),
        )
        for kind, standard, translated in history_pairs:
            with self.subTest(kind=kind):
                standard_batch = _raw_public_history_batch(
                    torch,
                    tuple(item.transitions for item in standard),
                    device=torch.device("cpu"),
                )
                translated_batch = _raw_public_history_batch(
                    torch,
                    tuple(item.transitions for item in translated),
                    device=torch.device("cpu"),
                )
                standard_atoms, standard_events = model.relative_event_encoder(
                    standard_batch
                )
                translated_atoms, translated_events = model.relative_event_encoder(
                    translated_batch
                )
                self.assertTrue(
                    torch.equal(
                        standard_events[:, -1],
                        translated_events[:, -1],
                    )
                )
                standard_match = model._causal_atom_match_context(
                    standard_batch,
                    standard_atoms,
                )
                translated_match = model._causal_atom_match_context(
                    translated_batch,
                    translated_atoms,
                )
                self.assertTrue(
                    torch.equal(
                        standard_match[:, -1],
                        translated_match[:, -1],
                    )
                )
                standard_pairs = model._palette_invariant_atom_pair_features(
                    standard_batch
                )
                translated_pairs = model._palette_invariant_atom_pair_features(
                    translated_batch
                )
                self.assertTrue(
                    torch.equal(
                        standard_pairs[:, -1],
                        translated_pairs[:, -1],
                    )
                )

                with torch.no_grad():
                    standard_belief = model.infer_factor_belief(
                        standard_batch,
                        is_agent_probe_result=_controller_probe_result_mask(
                            torch,
                            standard,
                            device=torch.device("cpu"),
                        ),
                    )
                    translated_belief = model.infer_factor_belief(
                        translated_batch,
                        is_agent_probe_result=_controller_probe_result_mask(
                            torch,
                            translated,
                            device=torch.device("cpu"),
                        ),
                    )
                self.assertTrue(
                    torch.equal(
                        standard_belief.factor_probabilities,
                        translated_belief.factor_probabilities,
                    )
                )

        _, standard, _ = history_pairs[0]
        standard_batch = _raw_public_history_batch(
            torch,
            tuple(item.transitions for item in standard),
            device=torch.device("cpu"),
        )
        standard_atoms, standard_events = model.relative_event_encoder(
            standard_batch
        )
        assert standard_batch.support_action_mask is not None
        swapped_actions = standard_batch.support_actions.clone()
        swapped_actions[:, -1] = swapped_actions[:, -1].flip(dims=(1,))
        swapped_mask = standard_batch.support_action_mask.clone()
        swapped_mask[:, -1] = swapped_mask[:, -1].flip(dims=(1,))
        swapped_batch = replace(
            standard_batch,
            support_actions=swapped_actions,
            support_action_mask=swapped_mask,
        )
        swapped_atoms, swapped_events = model.relative_event_encoder(swapped_batch)
        self.assertTrue(
            torch.equal(standard_events[:, -1], swapped_events[:, -1])
        )
        standard_match = model._causal_atom_match_context(
            standard_batch,
            standard_atoms,
        )
        swapped_match = model._causal_atom_match_context(
            swapped_batch,
            swapped_atoms,
        )
        self.assertTrue(
            torch.equal(standard_match[:, -1], swapped_match[:, -1])
        )
        standard_pairs = model._palette_invariant_atom_pair_features(
            standard_batch
        )
        swapped_pairs = model._palette_invariant_atom_pair_features(swapped_batch)
        self.assertTrue(
            torch.equal(
                standard_pairs[:, -1, :, :-1],
                swapped_pairs[:, -1, :, :-1].flip(dims=(1,)),
            )
        )
        with torch.no_grad():
            standard_belief = model.infer_factor_belief(
                standard_batch,
                is_agent_probe_result=_controller_probe_result_mask(
                    torch,
                    standard,
                    device=torch.device("cpu"),
                ),
            )
            swapped_belief = model.infer_factor_belief(
                swapped_batch,
                is_agent_probe_result=_controller_probe_result_mask(
                    torch,
                    standard,
                    device=torch.device("cpu"),
                ),
            )
        self.assertTrue(
            torch.equal(
                standard_belief.factor_probabilities,
                swapped_belief.factor_probabilities,
            )
        )

    def test_probe_phase_bit_only_changes_marked_transition_evidence(self) -> None:
        assert torch is not None
        base = self._model()
        model = ProbeAwareSymmetryFactorBeliefCausalK4(base.executor).eval()
        batch = model._support_only_batch(self._batch(count=2))
        is_agent_probe_result = torch.zeros_like(batch.support_mask)
        is_agent_probe_result[:, -1] = True
        with torch.no_grad():
            model.agent_probe_result_embedding.copy_(
                torch.linspace(-1.0, 1.0, model.config.rule_dim)
            )
            plain = model.infer_factor_belief(batch)
            all_observed = model.infer_factor_belief(
                batch,
                is_agent_probe_result=torch.zeros_like(batch.support_mask),
            )
            marked = model.infer_factor_belief(
                batch,
                is_agent_probe_result=is_agent_probe_result,
            )
        self.assertTrue(
            torch.equal(
                plain.factor_probabilities,
                all_observed.factor_probabilities,
            )
        )
        self.assertTrue(
            torch.equal(
                plain.evidence_axis_logits[:, :-1],
                marked.evidence_axis_logits[:, :-1],
            )
        )
        self.assertFalse(
            torch.allclose(
                plain.evidence_axis_logits[:, -1],
                marked.evidence_axis_logits[:, -1],
            )
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            model.infer_factor_belief(
                batch,
                is_agent_probe_result=torch.zeros(2, 1, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            model.infer_factor_belief(
                batch,
                is_agent_probe_result=torch.zeros_like(
                    batch.support_mask,
                    dtype=torch.long,
                ),
            )
        padded = replace(
            batch,
            support_mask=batch.support_mask.clone(),
        )
        padded.support_mask[:, -1] = False
        with self.assertRaisesRegex(ValueError, "padded"):
            model.infer_factor_belief(
                padded,
                is_agent_probe_result=is_agent_probe_result,
            )

        order = torch.tensor([5, 0, 1, 2, 3, 4])
        permuted = replace(
            batch,
            support_states=batch.support_states[:, order],
            support_actions=batch.support_actions[:, order],
            support_targets=batch.support_targets[:, order],
            support_mask=batch.support_mask[:, order],
            support_action_mask=(
                None
                if batch.support_action_mask is None
                else batch.support_action_mask[:, order]
            ),
        )
        with torch.no_grad():
            reordered = model.infer_factor_belief(
                permuted,
                is_agent_probe_result=is_agent_probe_result[:, order],
            )
        self.assertTrue(
            torch.allclose(
                marked.factor_probabilities,
                reordered.factor_probabilities,
                atol=1e-7,
                rtol=0.0,
            )
        )

    def test_support_order_is_invariant(self) -> None:
        assert torch is not None
        torch.manual_seed(911)
        model = self._model().eval()
        batch = model._support_only_batch(self._batch(count=1))
        order = torch.tensor([3, 0, 5, 2, 1, 4])
        permuted = replace(
            batch,
            support_states=batch.support_states[:, order],
            support_actions=batch.support_actions[:, order],
            support_targets=batch.support_targets[:, order],
            support_mask=batch.support_mask[:, order],
            support_action_mask=(
                None
                if batch.support_action_mask is None
                else batch.support_action_mask[:, order]
            ),
        )
        first = model.infer_support(batch)
        second = model.infer_support(permuted)
        self.assertTrue(torch.allclose(first.factor_logits, second.factor_logits))

    def test_width_interface_is_strictly_deterministic_w4(self) -> None:
        assert torch is not None
        model = self._model().eval()
        batch = self._batch(count=1)
        inference = model.sample_width_candidates(batch, width=4)
        self.assertEqual(inference.particles, 4)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            model.sample_width_candidates(batch, width=8)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            model.sample_width_candidates(batch, width=4, sample_noise=True)

    def test_noncanonical_public_set_is_rejected(self) -> None:
        assert torch is not None
        model = self._model()
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[0, torch.tensor([0, 1, 4, 5])] = True
        with self.assertRaisesRegex(ValueError, "one four-valued axis"):
            model._canonical_public_targets(mask)


if __name__ == "__main__":
    unittest.main()
