"""Checks for exact-discrete amortized causal rule inference."""

from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
    from prp_wm.discrete_causal_rules import ExpectedDiscreteCausalK4
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class ExpectedDiscreteCausalK4Tests(unittest.TestCase):
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

    def _batch(self):
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="expected-discrete-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    def test_joint_distribution_is_normalized_and_matches_axis_argmax(self) -> None:
        assert torch is not None
        torch.manual_seed(301)
        model = ExpectedDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        )
        inference = model.infer_support(self._batch())
        joint = model.joint_rule_probabilities(inference.factor_logits)
        self.assertEqual(joint.shape, (2, 4, 64))
        self.assertTrue(
            torch.allclose(
                joint.sum(dim=-1),
                torch.ones((2, 4)),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        joint_ids = model.factor_bank[joint.argmax(dim=-1)]
        self.assertTrue(torch.equal(joint_ids, inference.factor_ids))

    def test_cost_table_is_detached_and_loss_reaches_all_heads(self) -> None:
        assert torch is not None
        torch.manual_seed(307)
        executor = OracleFactorExecutor(self._config())
        model = ExpectedDiscreteCausalK4(executor)
        loss = model.losses(self._batch())
        self.assertEqual(loss.behavior_costs.shape, (2, 64, 4))
        self.assertFalse(loss.behavior_costs.requires_grad)
        support_costs = model.discrete_support_costs(self._batch())
        self.assertEqual(support_costs.shape, (2, 64))
        self.assertFalse(support_costs.requires_grad)
        self.assertTrue(torch.isfinite(loss.total).item())
        loss.total.backward()
        for head in model.factor_heads:
            gradient = head[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in executor.parameters())
        )

    def test_behavior_permutation_preserves_expected_discrete_loss(self) -> None:
        assert torch is not None
        torch.manual_seed(311)
        model = ExpectedDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        ).eval()
        batch = self._batch()
        first = model.losses(batch).total
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = replace(
            batch,
            behavior_targets=batch.behavior_targets[:, permutation],
            behavior_mass=batch.behavior_mass[:, permutation],
        )
        second = model.losses(permuted).total
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_train_mode_keeps_executor_frozen_in_eval_mode(self) -> None:
        assert torch is not None
        executor = OracleFactorExecutor(self._config())
        model = ExpectedDiscreteCausalK4(executor)
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(executor.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in executor.parameters())
        )

    def test_inference_does_not_read_query_or_behavior_supervision(self) -> None:
        assert torch is not None
        torch.manual_seed(313)
        model = ExpectedDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        ).eval()
        batch = self._batch()
        changed = replace(
            batch,
            query_states=batch.query_states.roll(1, dims=0),
            query_actions=batch.query_actions.roll(1, dims=0),
            behavior_targets=batch.behavior_targets.roll(1, dims=0),
            behavior_mass=batch.behavior_mass.roll(1, dims=0),
        )
        first = model.infer_support(batch)
        second = model.infer_support(changed)
        self.assertTrue(torch.equal(first.factor_logits, second.factor_logits))

    def test_diagonal_context_holdout_is_disjoint_and_axis_balanced(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.pilot import make_pilot_tasks
        from prp_wm.rulegrid import version_space
        from scripts.run_expected_discrete_causal_coverage import (
            _is_diagonal_holdout,
            _support_context_key,
        )

        tasks = make_pilot_tasks(
            split="expected-discrete-context-test",
            master_seed=2026071601,
            start=0,
            count=192,
            diagnostic_indices=(21, 22, 23),
        )
        contexts = {
            _support_context_key(
                task,
                factor_ids_for_program=rule_program_factor_ids,
                version_space=version_space,
            )
            for task in tasks
        }
        train = {context for context in contexts if not _is_diagonal_holdout(context)}
        heldout = {context for context in contexts if _is_diagonal_holdout(context)}
        self.assertEqual(len(contexts), 48)
        self.assertEqual(len(train), 36)
        self.assertEqual(len(heldout), 12)
        self.assertFalse(train.intersection(heldout))
        for axis in range(3):
            axis_heldout = {context for context in heldout if context[0] == axis}
            self.assertEqual(axis_heldout, {(axis, value, value) for value in range(4)})

    def test_latin_context_folds_partition_all_contexts_and_values(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.pilot import make_pilot_tasks
        from prp_wm.rulegrid import version_space
        from scripts.run_expected_discrete_causal_coverage import (
            _is_latin_holdout,
            _support_context_key,
        )

        tasks = make_pilot_tasks(
            split="expected-discrete-latin-context-test",
            master_seed=2026071601,
            start=0,
            count=192,
            diagnostic_indices=(21, 22, 23),
        )
        contexts = {
            _support_context_key(
                task,
                factor_ids_for_program=rule_program_factor_ids,
                version_space=version_space,
            )
            for task in tasks
        }
        folds = [
            {context for context in contexts if _is_latin_holdout(context, fold)}
            for fold in range(4)
        ]
        self.assertEqual(len(contexts), 48)
        self.assertEqual(set().union(*folds), contexts)
        for fold, heldout in enumerate(folds):
            self.assertEqual(len(heldout), 12)
            self.assertTrue(
                all((context[1] + context[2]) % 4 == fold for context in heldout)
            )
            for other_fold in range(fold):
                self.assertFalse(heldout.intersection(folds[other_fold]))
            for axis in range(3):
                axis_heldout = {context for context in heldout if context[0] == axis}
                self.assertEqual(len(axis_heldout), 4)
                self.assertEqual({context[1] for context in axis_heldout}, set(range(4)))
                self.assertEqual({context[2] for context in axis_heldout}, set(range(4)))


if __name__ == "__main__":
    unittest.main()
