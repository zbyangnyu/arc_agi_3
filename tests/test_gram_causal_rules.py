"""Checks for the GRAM-style factorized causal-rule proposer."""

from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
    from prp_wm.gram_causal_rules import (
        DiagonalGaussian,
        GRAMFactorizedCausalK4,
    )
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class GRAMFactorizedCausalK4Tests(unittest.TestCase):
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
            split="gram-causal-test",
            master_seed=2026072201,
            start=0,
            count=1,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    def _model(self, *, recursive_steps: int = 2):
        return GRAMFactorizedCausalK4(
            OracleFactorExecutor(self._config()),
            recursive_steps=recursive_steps,
            guidance_dim=8,
        )

    def test_prior_sampling_repeats_initial_state_and_scales_width(self) -> None:
        assert torch is not None
        torch.manual_seed(501)
        model = self._model().eval()
        batch = self._batch()
        first = model.sample_trajectories(batch, width=7, seed=19)
        second = model.sample_trajectories(batch, width=7, seed=19)

        self.assertEqual(first.factor_logits.shape, (2, 1, 7, 3, 4))
        self.assertEqual(first.high_states.shape, (3, 1, 7, 32))
        self.assertEqual(first.standard_noises.shape, (2, 1, 7, 8))
        self.assertFalse(first.used_training_posterior)
        self.assertTrue(torch.equal(first.factor_logits, second.factor_logits))
        self.assertTrue(torch.equal(first.standard_noises, second.standard_noises))

        repeated_high = first.high_states[0, :, :1].expand(-1, 7, -1)
        repeated_low = first.low_states[0, :, :1].expand(-1, 7, -1)
        self.assertTrue(torch.equal(first.high_states[0], repeated_high))
        self.assertTrue(torch.equal(first.low_states[0], repeated_low))
        self.assertGreater(
            float((first.standard_noises[:, :, 0] - first.standard_noises[:, :, 1]).abs().sum()),
            0.0,
        )

        candidates = model.sample_width_candidates(batch, width=7, seed=19)
        self.assertEqual(candidates.factor_ids.shape, (1, 7, 3))
        self.assertTrue(torch.equal(candidates.factor_ids, first.factor_ids[-1]))
        self.assertEqual(model.high_core.input_size, 2 * 32)
        self.assertEqual(model.low_core.input_size, 2 * 32)
        self.assertEqual(model.guidance_to_high.in_features, 8)
        self.assertEqual(model.guidance_to_high.out_features, 32)

        mean_path = model.sample_trajectories(
            batch,
            width=7,
            seed=999,
            sample_noise=False,
        )
        self.assertTrue(torch.count_nonzero(mean_path.standard_noises) == 0)
        self.assertTrue(
            torch.allclose(
                mean_path.factor_logits[:, :, :1].expand(-1, -1, 7, -1, -1),
                mean_path.factor_logits,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_prior_inference_reads_support_only(self) -> None:
        assert torch is not None
        torch.manual_seed(503)
        model = self._model().eval()
        batch = self._batch()
        changed = replace(
            batch,
            query_states=(batch.query_states + 1) % model.config.num_colors,
            query_actions=batch.query_actions.roll(1, dims=1),
            behavior_targets=(
                batch.behavior_targets + 3
            ) % model.config.num_colors,
            behavior_mass=batch.behavior_mass.roll(1, dims=1),
        )
        first = model.sample_trajectories(batch, seed=23)
        second = model.sample_trajectories(changed, seed=23)
        self.assertTrue(torch.equal(first.factor_logits, second.factor_logits))
        self.assertTrue(torch.equal(first.prior_means, second.prior_means))

        support_only = replace(
            batch,
            query_states=None,
            query_actions=None,
            query_targets=None,
            behavior_targets=None,
            behavior_mass=None,
            query_action_mask=None,
        )
        third = model.sample_trajectories(support_only, seed=23)
        self.assertTrue(torch.equal(first.factor_logits, third.factor_logits))

        support_permutation = torch.arange(batch.support_steps - 1, -1, -1)
        reordered = replace(
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
        fourth = model.sample_trajectories(reordered, seed=23)
        self.assertTrue(
            torch.allclose(first.factor_logits, fourth.factor_logits, atol=1e-6, rtol=1e-6)
        )

    def test_recursive_depth_shares_one_parameterized_core(self) -> None:
        assert torch is not None
        torch.manual_seed(507)
        shallow = self._model(recursive_steps=1)
        torch.manual_seed(507)
        deep = self._model(recursive_steps=4)
        shallow_trainable = sum(
            parameter.numel()
            for parameter in shallow.parameters()
            if parameter.requires_grad
        )
        deep_trainable = sum(
            parameter.numel()
            for parameter in deep.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(shallow_trainable, deep_trainable)
        self.assertEqual(set(shallow.state_dict()), set(deep.state_dict()))
        self.assertTrue(deep.truncate_between_steps)

    def test_posterior_is_target_conditioned_and_set_permutation_equivariant(self) -> None:
        assert torch is not None
        torch.manual_seed(509)
        model = self._model().eval()
        batch = self._batch()
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = replace(
            batch,
            behavior_targets=batch.behavior_targets[:, permutation],
            behavior_mass=batch.behavior_mass[:, permutation],
        )
        first = model.sample_training_trajectories(batch, seed=29)
        second = model.sample_training_trajectories(permuted, seed=29)

        self.assertTrue(first.used_training_posterior)
        self.assertIsNotNone(first.posterior_means)
        self.assertIsNotNone(first.posterior_log_variances)
        assert first.posterior_means is not None
        assert first.posterior_log_variances is not None
        assert second.posterior_means is not None
        self.assertGreater(
            float(
                (
                    first.posterior_means[:, :, 0]
                    - first.posterior_means[:, :, 1]
                ).abs().sum().detach()
            ),
            0.0,
        )
        self.assertTrue(
            torch.allclose(
                second.posterior_means,
                first.posterior_means[:, :, permutation],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                second.factor_logits,
                first.factor_logits[:, :, permutation],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        reconstructed = (
            first.posterior_means
            + torch.exp(0.5 * first.posterior_log_variances)
            * first.standard_noises
        )
        self.assertTrue(
            torch.allclose(
                reconstructed,
                first.guidance_samples,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        first_loss = model.losses(batch, seed=37).total
        second_loss = model.losses(permuted, seed=37).total
        self.assertTrue(
            torch.allclose(first_loss, second_loss, atol=1e-6, rtol=1e-6)
        )

    def test_analytic_diagonal_gaussian_kl(self) -> None:
        assert torch is not None
        zeros = torch.zeros(2, 4, 8)
        prior = DiagonalGaussian(zeros, zeros)
        same = GRAMFactorizedCausalK4.analytic_gaussian_kl(prior, prior)
        shifted = GRAMFactorizedCausalK4.analytic_gaussian_kl(
            DiagonalGaussian(torch.ones_like(zeros), zeros),
            prior,
        )
        self.assertTrue(torch.equal(same, torch.zeros(2, 4)))
        self.assertTrue(torch.allclose(shifted, torch.full((2, 4), 4.0)))

    def test_loss_supervises_every_step_and_keeps_executor_frozen(self) -> None:
        assert torch is not None
        torch.manual_seed(521)
        executor = OracleFactorExecutor(self._config())
        model = GRAMFactorizedCausalK4(
            executor,
            recursive_steps=2,
            guidance_dim=8,
        )
        model.train()
        loss = model.losses(self._batch(), seed=31)

        self.assertFalse(executor.training)
        self.assertEqual(loss.step_objectives.shape, (2,))
        self.assertEqual(loss.step_kls.shape, (2,))
        self.assertEqual(loss.joint_probabilities.shape, (2, 1, 4, 64))
        self.assertTrue(torch.all(loss.deep_supervision_weights > 0).item())
        self.assertTrue(
            torch.allclose(
                loss.deep_supervision_weights.sum(),
                torch.tensor(1.0),
            )
        )
        self.assertFalse(loss.behavior_costs.requires_grad)
        self.assertFalse(loss.support_costs.requires_grad)
        self.assertGreaterEqual(float(loss.kl.detach()), 0.0)
        self.assertTrue(torch.isfinite(loss.total).item())

        loss.total.backward()
        for module in (
            model.high_core,
            model.low_core,
            model.prior_head[-1],
            model.posterior_head[-1],
            model.posterior_behavior_to_guidance,
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
            all(parameter.grad is None for parameter in executor.parameters())
        )


if __name__ == "__main__":
    unittest.main()
