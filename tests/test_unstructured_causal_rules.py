"""Checks for the parameter-matched opaque 64-way causal-rule control."""

from __future__ import annotations

from dataclasses import replace
import itertools
import unittest

try:
    import torch
    import torch.nn.functional as F
    from prp_wm.discrete_causal_rules import ExpectedDiscreteCausalK4
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
    from prp_wm.unstructured_causal_rules import (
        DirectRuleHead,
        RULE_CLASS_COUNT,
        UnstructuredDiscreteCausalK4,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class UnstructuredExpectedDiscreteK4Tests(unittest.TestCase):
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
            split="unstructured-discrete-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    @staticmethod
    def _trainable_parameters(model) -> int:
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def test_low_rank_head_is_parameter_matched_and_replaces_factor_heads(self) -> None:
        assert torch is not None
        factorized = ExpectedDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        )
        unstructured = UnstructuredDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        )
        factorized_count = self._trainable_parameters(factorized)
        unstructured_count = self._trainable_parameters(unstructured)
        relative_difference = abs(
            unstructured_count - factorized_count
        ) / factorized_count
        self.assertLess(relative_difference, 0.01)
        self.assertFalse(hasattr(unstructured, "factor_heads"))
        self.assertEqual(unstructured.head_kind, "low-rank")
        self.assertEqual(unstructured.head_rank, 5)
        self.assertEqual(
            unstructured.rule_head.layers[-1].out_features,
            RULE_CLASS_COUNT,
        )

    def test_head_rank_override_supports_capacity_sensitivity_control(self) -> None:
        model = UnstructuredDiscreteCausalK4(
            OracleFactorExecutor(self._config()),
            head_rank=9,
        )
        self.assertEqual(model.head_kind, "low-rank")
        self.assertEqual(model.head_rank, 9)
        self.assertEqual(model.rule_head.layers[1].out_features, 9)
        inference = model.infer_support(self._batch())
        self.assertEqual(inference.rule_logits.shape, (2, 4, 64))

    def test_direct_linear_head_is_unrestricted_capacity_sensitivity(self) -> None:
        assert torch is not None
        factorized = ExpectedDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        )
        direct = UnstructuredDiscreteCausalK4(
            OracleFactorExecutor(self._config()),
            head_kind="direct-linear",
        )
        self.assertIsInstance(direct.rule_head, DirectRuleHead)
        self.assertEqual(direct.head_kind, "direct-linear")
        self.assertIsNone(direct.head_rank)
        self.assertEqual(len(direct.rule_head.layers), 2)
        self.assertEqual(
            direct.rule_head.layers[-1].out_features,
            RULE_CLASS_COUNT,
        )
        # LN(32)->Linear(64) has 2,176 parameters versus 588 in the
        # factorized 3x4 heads, hence a 1,588-parameter sensitivity delta.
        self.assertEqual(
            self._trainable_parameters(direct)
            - self._trainable_parameters(factorized),
            1_588,
        )
        inference = direct.infer_support(self._batch())
        self.assertEqual(inference.rule_logits.shape, (2, 4, 64))
        loss = direct.losses(self._batch())
        loss.total.backward()
        gradient = direct.rule_head.layers[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_head_configuration_rejects_ambiguous_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "head_kind"):
            UnstructuredDiscreteCausalK4(
                OracleFactorExecutor(self._config()),
                head_kind="unknown",
            )
        with self.assertRaisesRegex(ValueError, "head_rank"):
            UnstructuredDiscreteCausalK4(
                OracleFactorExecutor(self._config()),
                head_kind="direct-linear",
                head_rank=9,
            )

    def test_direct_distribution_is_normalized_and_decodes_executor_ids(self) -> None:
        assert torch is not None
        torch.manual_seed(401)
        model = UnstructuredDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        )
        inference = model.infer_support(self._batch())
        self.assertEqual(inference.rule_logits.shape, (2, 4, 64))
        self.assertEqual(inference.rule_probabilities.shape, (2, 4, 64))
        self.assertTrue(
            torch.allclose(
                inference.rule_probabilities.sum(dim=-1),
                torch.ones((2, 4)),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                inference.factor_ids,
                model.factor_bank[inference.rule_ids],
            )
        )
        self.assertTrue(
            torch.equal(
                inference.rule_codes,
                F.one_hot(inference.rule_ids, 64).to(
                    dtype=inference.rule_codes.dtype
                ),
            )
        )

    def test_shared_evaluator_compatibility_views(self) -> None:
        assert torch is not None
        torch.manual_seed(409)
        model = UnstructuredDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        ).eval()
        inference = model.infer_support(self._batch())
        self.assertEqual(inference.factor_logits.shape, (2, 4, 1, 64))
        self.assertEqual(inference.factor_ids.shape, (2, 4, 3))
        margins = inference.factor_logits.topk(2, dim=-1).values
        reduced = (margins[..., 0] - margins[..., 1]).mean(dim=(1, 2))
        self.assertEqual(reduced.shape, (2,))
        prediction = model.predict_panel(self._batch(), inference)
        self.assertEqual(prediction.change_logits.shape[:2], (8, 4))

    def test_detached_cost_hard_assignment_and_gradient_isolation(self) -> None:
        assert torch is not None
        torch.manual_seed(419)
        executor = OracleFactorExecutor(self._config())
        model = UnstructuredDiscreteCausalK4(executor)
        loss = model.losses(self._batch())
        self.assertFalse(loss.behavior_costs.requires_grad)
        expected = torch.einsum(
            "bkr,brm->bkm",
            loss.joint_probabilities,
            loss.behavior_costs,
        )
        permutations = torch.tensor(
            list(itertools.permutations(range(4))),
            dtype=torch.long,
        )
        rows = torch.arange(4)
        manual = expected[:, rows[None], permutations].mean(dim=-1).amin(dim=1).mean()
        self.assertTrue(torch.allclose(loss.set_cost, manual))
        loss.total.backward()
        gradient = model.rule_head.layers[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in executor.parameters())
        )
        with self.assertRaisesRegex(ValueError, "hard 4!"):
            model.losses(self._batch(), assignment_temperature=0.1)

    def test_behavior_and_support_order_invariances(self) -> None:
        assert torch is not None
        torch.manual_seed(421)
        model = UnstructuredDiscreteCausalK4(
            OracleFactorExecutor(self._config())
        ).eval()
        batch = self._batch()
        behavior_permutation = torch.tensor([2, 0, 3, 1])
        permuted_behavior = replace(
            batch,
            behavior_targets=batch.behavior_targets[:, behavior_permutation],
            behavior_mass=batch.behavior_mass[:, behavior_permutation],
        )
        self.assertTrue(
            torch.allclose(
                model.losses(batch).total,
                model.losses(permuted_behavior).total,
                atol=1e-6,
                rtol=1e-6,
            )
        )

        support_permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        permuted_support = replace(
            batch,
            support_states=batch.support_states[:, support_permutation],
            support_actions=batch.support_actions[:, support_permutation],
            support_targets=batch.support_targets[:, support_permutation],
            support_mask=batch.support_mask[:, support_permutation],
            support_action_mask=(
                batch.support_action_mask[:, support_permutation]
                if batch.support_action_mask is not None
                else None
            ),
        )
        first = model.infer_support(batch)
        second = model.infer_support(permuted_support)
        self.assertTrue(
            torch.allclose(
                first.rule_logits,
                second.rule_logits,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_inference_does_not_read_query_or_behavior_supervision(self) -> None:
        assert torch is not None
        torch.manual_seed(431)
        model = UnstructuredDiscreteCausalK4(
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
        self.assertTrue(torch.equal(first.rule_logits, second.rule_logits))


if __name__ == "__main__":
    unittest.main()
