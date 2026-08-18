"""Focused invariants for the public-only GRAM version-space objective."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

try:
    import torch
    from prp_wm.gram_causal_rules import GRAMFactorizedCausalK4
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig, OutcomePrediction
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class GRAMPublicCoverageTests(unittest.TestCase):
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

    def _model(self) -> GRAMFactorizedCausalK4:
        assert torch is not None
        return GRAMFactorizedCausalK4(
            OracleFactorExecutor(self._config()),
            recursive_steps=2,
            guidance_dim=8,
        )

    def _batch(self, count: int = 1):
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="gram-public-coverage-test",
            master_seed=2026072202,
            start=0,
            count=count,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    @staticmethod
    def _four_code_mask(model: GRAMFactorizedCausalK4, batch_size: int):
        assert torch is not None
        mask = torch.zeros(
            batch_size,
            model.factor_bank.shape[0],
            dtype=torch.bool,
        )
        # Codes 0..3 vary relation and fix the other two axes.
        mask[:, :4] = True
        return mask

    def _prediction_with_first_four_exact(
        self,
        model: GRAMFactorizedCausalK4,
        support,
    ) -> OutcomePrediction:
        assert torch is not None
        self.assertIsNone(support.query_states)
        self.assertIsNone(support.query_actions)
        self.assertIsNone(support.behavior_targets)
        batch_size, steps, height, width = support.support_states.shape
        hypotheses = model.factor_bank.shape[0]
        states = support.support_states.reshape(batch_size * steps, height, width)
        targets = support.support_targets.reshape(batch_size * steps, height, width)
        desired = targets[:, None].expand(-1, hypotheses, -1, -1).clone()
        wrong_color = (states[:, 0, 0] + 1) % model.config.num_colors
        desired[:, 4:, 0, 0] = wrong_color[:, None]
        changed = desired.ne(states[:, None])
        change_logits = torch.where(
            changed,
            torch.full_like(desired, 20, dtype=torch.float32),
            torch.full_like(desired, -20, dtype=torch.float32),
        )
        color_logits = torch.full(
            (
                batch_size * steps,
                hypotheses,
                model.config.num_colors,
                height,
                width,
            ),
            -20.0,
        )
        color_logits.scatter_(2, desired[:, :, None], 20.0)
        return OutcomePrediction(
            input_colors=states,
            change_logits=change_logits,
            new_color_logits=color_logits,
        )

    def test_exact_mask_is_map_equality_on_support_only(self) -> None:
        assert torch is not None
        torch.manual_seed(601)
        model = self._model().eval()
        batch = self._batch()
        changed_privileged_fields = replace(
            batch,
            query_states=(batch.query_states + 1) % model.config.num_colors,
            query_actions=batch.query_actions.roll(1, dims=1),
            behavior_targets=(
                batch.behavior_targets + 2
            ) % model.config.num_colors,
            behavior_mass=batch.behavior_mass.roll(1, dims=1),
        )

        def prediction(support):
            return self._prediction_with_first_four_exact(model, support)

        with patch.object(
            model,
            "_predict_all_support_codes",
            side_effect=prediction,
        ):
            first = model.public_support_exact_mask(batch)
            second = model.public_support_exact_mask(changed_privileged_fields)
        expected = self._four_code_mask(model, batch.batch_size)
        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(second, expected))
        self.assertFalse(first.requires_grad)

    def test_coverage_loss_is_public_prior_only_and_backpropagates(self) -> None:
        assert torch is not None
        torch.manual_seed(603)
        model = self._model().train()
        batch = self._batch()
        mask = self._four_code_mask(model, batch.batch_size)
        with patch.object(
            model,
            "public_support_exact_mask",
            return_value=mask,
        ):
            loss = model.coverage_losses(batch, seed=41)
        self.assertEqual(loss.joint_probabilities.shape, (2, 1, 4, 64))
        self.assertTrue(torch.equal(loss.compatible_indices, torch.arange(4)[None]))
        self.assertTrue(torch.isfinite(loss.total).item())
        self.assertGreaterEqual(float(loss.axis_balance.detach()), -1e-6)
        self.assertGreaterEqual(float(loss.invalid_mass.detach()), 0.0)
        self.assertLessEqual(float(loss.invalid_mass.detach()), 1.0)
        self.assertTrue(
            torch.allclose(
                loss.joint_probabilities.sum(dim=-1),
                torch.ones_like(loss.joint_probabilities[..., 0]),
                atol=1e-6,
                rtol=1e-6,
            )
        )

        loss.total.backward()
        for module in (
            model.high_core,
            model.low_core,
            model.prior_head[-1],
            model.guidance_to_high,
            *model.factor_heads,
        ):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(
                all(
                    gradient is None or torch.isfinite(gradient).all().item()
                    for gradient in gradients
                )
            )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.posterior_head.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.posterior_behavior_to_guidance.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.executor.parameters())
        )

    def test_assignment_is_invariant_to_slot_and_set_order(self) -> None:
        assert torch is not None
        torch.manual_seed(607)
        cost = torch.rand(3, 4, 4)
        baseline = GRAMFactorizedCausalK4._soft_permutation_loss(cost, 0.05)
        slots = GRAMFactorizedCausalK4._soft_permutation_loss(
            cost[:, torch.tensor([2, 0, 3, 1])],
            0.05,
        )
        codes = GRAMFactorizedCausalK4._soft_permutation_loss(
            cost[:, :, torch.tensor([1, 3, 0, 2])],
            0.05,
        )
        self.assertTrue(torch.allclose(baseline, slots, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(baseline, codes, atol=1e-7, rtol=1e-7))

    def test_non_four_version_space_is_rejected(self) -> None:
        assert torch is not None
        model = self._model().eval()
        batch = self._batch()
        bad = self._four_code_mask(model, batch.batch_size)
        bad[:, 3] = False
        with patch.object(
            model,
            "public_support_exact_mask",
            return_value=bad,
        ):
            with self.assertRaisesRegex(ValueError, "exactly four"):
                model.coverage_losses(batch, seed=43)

    def test_compatible_indices_are_gathered_independently_per_task(self) -> None:
        assert torch is not None
        model = self._model().eval()
        batch = self._batch(count=2)
        expected = torch.tensor(
            [[0, 5, 17, 63], [1, 6, 18, 62]],
            dtype=torch.long,
        )
        mask = torch.zeros(2, 64, dtype=torch.bool)
        mask.scatter_(1, expected, True)
        with patch.object(
            model,
            "public_support_exact_mask",
            return_value=mask,
        ):
            loss = model.coverage_losses(batch, seed=47)
        self.assertTrue(torch.equal(loss.compatible_indices, expected))


if __name__ == "__main__":
    unittest.main()
