"""Structural checks for the capacity-matched P1 executor conditions."""

from __future__ import annotations

import unittest

try:
    import torch

    from prp_wm.latent_rules import OracleFactorExecutor
    from prp_wm.matched_executor import (
        BASE_BRANCH,
        COLLISION_BRANCH,
        MatchedFactorLocalOracleFactorExecutor,
        MatchedWiderGlobalOracleFactorExecutor,
        RELATION_BRANCH,
        TRIGGER_BRANCH,
        canonical_spatial_branch_assignment,
    )
    from prp_wm.neural import NeuralPRPConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class MatchedExecutorTests(unittest.TestCase):
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

    def _composite_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch is not None
        state = torch.zeros((1, 8, 8), dtype=torch.long)
        state[0, 1, 1] = 1
        state[0, 1, 2] = 2
        state[0, 3, 1] = 5
        state[0, 3, 2] = 6
        state[0, 3, 3] = 9
        state[0, 5, 1] = 3
        state[0, 5, 2] = 4
        actions = torch.tensor(
            [
                [
                    [0, 1, 1, 3],
                    [1, 3, 1, 4],
                    [0, 5, 1, 3],
                ]
            ],
            dtype=torch.long,
        )
        mask = torch.ones((1, 3), dtype=torch.bool)
        return state, actions, mask

    def test_parameter_and_state_dict_identity(self) -> None:
        assert torch is not None
        wider = MatchedWiderGlobalOracleFactorExecutor(self._config())
        local = MatchedFactorLocalOracleFactorExecutor(self._config())
        wider_parameters = sum(parameter.numel() for parameter in wider.parameters())
        local_parameters = sum(parameter.numel() for parameter in local.parameters())
        self.assertEqual(wider_parameters, 41_564)
        self.assertEqual(local_parameters, 41_564)
        self.assertEqual(wider_parameters, local_parameters)
        self.assertEqual(list(wider.state_dict()), list(local.state_dict()))

    def test_spatial_assignment_uses_only_public_state_and_action(self) -> None:
        assert torch is not None
        state, actions, mask = self._composite_inputs()
        assignment = canonical_spatial_branch_assignment(state, actions, mask)
        self.assertEqual(int(assignment[0, 1, 0]), COLLISION_BRANCH)
        self.assertEqual(int(assignment[0, 1, 1]), COLLISION_BRANCH)
        self.assertEqual(int(assignment[0, 1, 2]), COLLISION_BRANCH)
        self.assertEqual(int(assignment[0, 1, 3]), COLLISION_BRANCH)
        self.assertEqual(int(assignment[0, 3, 2]), TRIGGER_BRANCH)
        self.assertEqual(int(assignment[0, 3, 3]), TRIGGER_BRANCH)
        self.assertEqual(int(assignment[0, 5, 1]), RELATION_BRANCH)
        self.assertEqual(int(assignment[0, 5, 2]), RELATION_BRANCH)
        self.assertEqual(int(assignment[0, 5, 3]), RELATION_BRANCH)
        self.assertEqual(int(assignment[0, 7, 7]), BASE_BRANCH)

    def test_wider_global_initialization_is_bit_identical_to_parent(self) -> None:
        assert torch is not None
        torch.manual_seed(71)
        parent = OracleFactorExecutor(self._config()).eval()
        wider = MatchedWiderGlobalOracleFactorExecutor(self._config()).eval()
        wider.initialize_from_oracle_state_dict(parent.state_dict())
        states, actions, mask = self._composite_inputs()
        factor_ids = torch.tensor([[3, 2, 1]], dtype=torch.long)
        parent_prediction = parent.predict(states, actions, factor_ids, mask)
        wider_prediction = wider.predict(states, actions, factor_ids, mask)
        self.assertTrue(
            torch.equal(
                parent_prediction.change_logits,
                wider_prediction.change_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                parent_prediction.new_color_logits,
                wider_prediction.new_color_logits,
            )
        )

    def test_factor_local_prediction_is_nuisance_invariant(self) -> None:
        assert torch is not None
        torch.manual_seed(73)
        model = MatchedFactorLocalOracleFactorExecutor(self._config()).eval()
        state = torch.zeros((2, 8, 8), dtype=torch.long)
        state[:, 2, 2] = 1
        state[:, 2, 3] = 2
        actions = torch.tensor(
            [[0, 2, 2, 3], [0, 2, 2, 3]],
            dtype=torch.long,
        )
        factors = torch.tensor(
            [[1, 0, 0], [1, 3, 2]],
            dtype=torch.long,
        )
        prediction = model.predict(state, actions, factors)
        self.assertTrue(
            torch.equal(
                prediction.change_logits[0],
                prediction.change_logits[1],
            )
        )
        self.assertTrue(
            torch.equal(
                prediction.new_color_logits[0],
                prediction.new_color_logits[1],
            )
        )

    def test_selected_categorical_distribution_is_normalized(self) -> None:
        assert torch is not None
        torch.manual_seed(79)
        model = MatchedFactorLocalOracleFactorExecutor(self._config()).eval()
        states, actions, mask = self._composite_inputs()
        factors = torch.tensor([[0, 1, 2]], dtype=torch.long)
        prediction = model.predict(states, actions, factors, mask)
        original = states[:, None, None]
        color_ids = torch.arange(16)[None, None, :, None, None]
        masked = prediction.new_color_logits.masked_fill(
            original.eq(color_ids),
            torch.finfo(prediction.new_color_logits.dtype).min,
        )
        changed = torch.sigmoid(prediction.change_logits)[:, :, None]
        color_probability = torch.softmax(masked, dim=2)
        outcome_probability = changed * color_probability
        outcome_probability = outcome_probability.scatter(
            2,
            original.expand(-1, 1, 1, -1, -1),
            (1.0 - changed),
        )
        self.assertTrue(
            torch.allclose(
                outcome_probability.sum(dim=2),
                torch.ones_like(prediction.change_logits),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_all_four_decoders_receive_gradient_on_composite(self) -> None:
        assert torch is not None
        torch.manual_seed(83)
        model = MatchedFactorLocalOracleFactorExecutor(self._config())
        states, actions, mask = self._composite_inputs()
        factors = torch.tensor([[0, 1, 2]], dtype=torch.long)
        target = states.clone()
        target[0, 1, 1] = 0
        target[0, 3, 2] = 7
        target[0, 5, 2] = 3
        prediction = model.predict(states, actions, factors, mask)
        loss = -prediction.log_prob(target).mean()
        loss.backward()
        decoders = (model.decoder, *tuple(model.axis_decoders))
        for decoder in decoders:
            gradients = [
                parameter.grad
                for parameter in decoder.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                any(bool(gradient.abs().sum().gt(0).item()) for gradient in gradients)
            )


if __name__ == "__main__":
    unittest.main()
