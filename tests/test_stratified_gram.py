"""Focused invariants for the persistent stratified GRAM adapter."""

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
    from prp_wm.neural import NeuralPRPConfig
    from prp_wm.stratified_gram import (
        PersistentStratifiedGRAMProposal,
        nested_stratified_anchor_codes,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class StratifiedGRAMTests(unittest.TestCase):
    def _legacy(self) -> GRAMFactorizedCausalK4:
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
        return GRAMFactorizedCausalK4(
            OracleFactorExecutor(config),
            recursive_steps=2,
            guidance_dim=8,
        )

    def _batch(self, count: int = 2):
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="stratified-gram-test",
            master_seed=2026072301,
            start=0,
            count=count,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    def test_anchor_bank_has_nested_balance_and_strength_two(self) -> None:
        anchors = nested_stratified_anchor_codes()
        self.assertEqual(len(anchors), 32)
        self.assertEqual(len(set(anchors)), 32)
        for width in (4, 8, 16, 32):
            for axis in range(3):
                self.assertEqual(
                    [
                        sum(code[axis] == value for code in anchors[:width])
                        for value in range(4)
                    ],
                    [width // 4] * 4,
                )
        for width, expected in ((16, 1), (32, 2)):
            for left in range(3):
                for right in range(left + 1, 3):
                    counts = [
                        sum(
                            code[left] == x and code[right] == y
                            for code in anchors[:width]
                        )
                        for x in range(4)
                        for y in range(4)
                    ]
                    self.assertEqual(counts, [expected] * 16)

    def test_wrapper_freezes_legacy_and_exposes_only_small_adapter(self) -> None:
        assert torch is not None
        model = PersistentStratifiedGRAMProposal(self._legacy())
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.legacy.parameters())
        )
        names = [name for name, _ in model.adapter_named_parameters()]
        self.assertEqual(
            names,
            [
                "anchor_log_gain",
                "support_adapter.0.weight",
                "support_adapter.0.bias",
                "support_adapter.1.weight",
                "support_adapter.1.bias",
            ],
        )
        count = sum(parameter.numel() for _, parameter in model.adapter_named_parameters())
        self.assertLess(count, 1_200)
        model.train()
        self.assertFalse(model.legacy.training)

    def test_public_inference_is_support_permutation_invariant(self) -> None:
        assert torch is not None
        torch.manual_seed(811)
        model = PersistentStratifiedGRAMProposal(self._legacy()).eval()
        batch = self._batch(count=1)
        order = torch.tensor([4, 1, 5, 0, 3, 2])
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
        first = model.sample_trajectories(batch, width=4, sample_noise=False)
        second = model.sample_trajectories(permuted, width=4, sample_noise=False)
        self.assertTrue(torch.equal(first.factor_ids, second.factor_ids))
        self.assertTrue(torch.allclose(first.factor_logits, second.factor_logits))

    def test_hard_loss_uses_all_public_codes_and_only_updates_adapter(self) -> None:
        assert torch is not None
        torch.manual_seed(821)
        model = PersistentStratifiedGRAMProposal(self._legacy()).train()
        batch = self._batch(count=2)
        changed_privileged = replace(
            batch,
            query_states=(batch.query_states + 1) % model.config.num_colors,
            behavior_targets=(batch.behavior_targets + 2) % model.config.num_colors,
        )
        mask = torch.zeros(2, 64, dtype=torch.bool)
        # Codes 0..3 fix collision/trigger and vary relation.
        mask[:, :4] = True
        with patch.object(model, "public_support_exact_mask", return_value=mask):
            first = model.hard_public_version_space_loss(
                batch,
                sample_noise=False,
            )
            second = model.hard_public_version_space_loss(
                changed_privileged,
                sample_noise=False,
            )
        expected_targets = torch.tensor([[0, 3, 1, 2], [0, 3, 1, 2]])
        self.assertTrue(torch.equal(first.canonical_target_indices, expected_targets))
        self.assertTrue(torch.equal(first.varying_axes, torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(first.total, second.total))
        self.assertTrue(
            torch.allclose(
                first.trajectories.factor_logits,
                second.trajectories.factor_logits,
            )
        )
        self.assertTrue(torch.isfinite(first.total).item())
        first.total.backward()
        adapter_gradients = [
            parameter.grad for _, parameter in model.adapter_named_parameters()
        ]
        self.assertTrue(any(gradient is not None for gradient in adapter_gradients))
        self.assertTrue(
            all(
                gradient is None or torch.isfinite(gradient).all().item()
                for gradient in adapter_gradients
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.legacy.parameters())
        )

    def test_invalid_version_space_shape_is_rejected(self) -> None:
        assert torch is not None
        model = PersistentStratifiedGRAMProposal(self._legacy()).eval()
        batch = self._batch(count=1)
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[:, :3] = True
        with patch.object(model, "public_support_exact_mask", return_value=mask):
            with self.assertRaisesRegex(ValueError, "exactly four"):
                model.hard_public_version_space_loss(batch)

    def test_replace_mode_removes_legacy_factor_logits(self) -> None:
        assert torch is not None
        torch.manual_seed(829)
        residual = PersistentStratifiedGRAMProposal(
            self._legacy(),
            legacy_logit_mode="residual",
        ).eval()
        replacement = PersistentStratifiedGRAMProposal(
            self._legacy(),
            legacy_logit_mode="replace",
        ).eval()
        batch = self._batch(count=1)
        first = replacement.sample_trajectories(
            batch,
            width=4,
            sample_noise=False,
        )
        # Within one wrapper, recursive-step logits are identical because no
        # legacy factor logits survive the adapter boundary.
        self.assertTrue(torch.allclose(first.factor_logits[0], first.factor_logits[1]))
        varied = residual.sample_trajectories(
            batch,
            width=4,
            sample_noise=False,
        )
        self.assertFalse(torch.allclose(varied.factor_logits[0], varied.factor_logits[1]))


if __name__ == "__main__":
    unittest.main()
