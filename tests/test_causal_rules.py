"""Fast checks for the axis-structured causal-mechanism ceiling."""

from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
    from prp_wm.causal_rules import AxisStructuredCausalK4
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class AxisStructuredCausalK4Tests(unittest.TestCase):
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
            split="causal-rule-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    def test_st_codes_are_hard_and_match_executor_rule_latents(self) -> None:
        assert torch is not None
        torch.manual_seed(101)
        executor = OracleFactorExecutor(self._config())
        model = AxisStructuredCausalK4(executor)
        inference = model.infer_support(self._batch())
        self.assertEqual(inference.factor_ids.shape, (2, 4, 3))
        self.assertTrue(
            torch.allclose(
                inference.factor_codes.detach(),
                torch.nn.functional.one_hot(
                    inference.factor_ids, 4
                ).to(dtype=inference.factor_codes.dtype),
                atol=1e-7,
                rtol=0.0,
            )
        )
        expected = executor.rule_latent(
            inference.factor_ids.reshape(-1, 3)
        ).reshape(2, 4, -1)
        self.assertTrue(torch.allclose(inference.rule_latents, expected))

    def test_loss_backpropagates_to_all_heads_but_not_frozen_executor(self) -> None:
        assert torch is not None
        torch.manual_seed(103)
        executor = OracleFactorExecutor(self._config())
        model = AxisStructuredCausalK4(executor)
        loss = model.losses(self._batch())
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

    def test_support_order_is_permutation_invariant(self) -> None:
        assert torch is not None
        torch.manual_seed(107)
        batch = self._batch()
        model = AxisStructuredCausalK4(
            OracleFactorExecutor(self._config())
        ).eval()
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        permuted = replace(
            batch,
            support_states=batch.support_states[:, permutation],
            support_actions=batch.support_actions[:, permutation],
            support_targets=batch.support_targets[:, permutation],
            support_mask=batch.support_mask[:, permutation],
            support_action_mask=(
                batch.support_action_mask[:, permutation]
                if batch.support_action_mask is not None
                else None
            ),
        )
        first = model.infer_support(batch)
        second = model.infer_support(permuted)
        self.assertTrue(torch.allclose(first.factor_logits, second.factor_logits))

    def test_behavior_order_does_not_change_set_loss(self) -> None:
        assert torch is not None
        torch.manual_seed(109)
        batch = self._batch()
        model = AxisStructuredCausalK4(
            OracleFactorExecutor(self._config())
        ).eval()
        first = model.losses(batch).set_nll
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = replace(
            batch,
            behavior_targets=batch.behavior_targets[:, permutation],
            behavior_mass=batch.behavior_mass[:, permutation],
        )
        second = model.losses(permuted).set_nll
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_prediction_has_one_persistent_mode_axis(self) -> None:
        assert torch is not None
        model = AxisStructuredCausalK4(
            OracleFactorExecutor(self._config())
        )
        batch = self._batch()
        inference = model.infer_support(batch)
        prediction = model.predict_panel(batch, inference)
        self.assertEqual(prediction.change_logits.shape, (8, 4, 8, 8))
        self.assertEqual(prediction.new_color_logits.shape, (8, 4, 16, 8, 8))


if __name__ == "__main__":
    unittest.main()
