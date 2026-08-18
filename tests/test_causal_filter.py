"""Checks for explicit latent causal-hypothesis filtering."""

from __future__ import annotations

import unittest

try:
    import torch
    from prp_wm.causal_filter import (
        HypothesisBankScores,
        enumerate_factor_codes,
        predict_factor_panel,
        score_hypothesis_bank,
        select_hypotheses,
        selected_factor_ids,
    )
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class CausalHypothesisFilterTests(unittest.TestCase):
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
            split="causal-filter-test",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=(21, 22, 23),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(21, 22, 23),
        )

    def test_factor_bank_contains_every_tuple_once(self) -> None:
        assert torch is not None
        codes = enumerate_factor_codes()
        self.assertEqual(codes.shape, (64, 3))
        self.assertEqual(torch.unique(codes, dim=0).shape[0], 64)
        self.assertEqual(int(codes.min()), 0)
        self.assertEqual(int(codes.max()), 3)

    def test_bank_scoring_and_selected_prediction_shapes(self) -> None:
        assert torch is not None
        torch.manual_seed(211)
        executor = OracleFactorExecutor(self._config()).eval()
        batch = self._batch()
        scores = score_hypothesis_bank(
            executor,
            batch.support_states,
            batch.support_actions,
            batch.support_targets,
            batch.support_mask,
            batch.support_action_mask,
        )
        self.assertEqual(scores.proper_nll_per_cell.shape, (2, 64))
        self.assertEqual(scores.map_exact.shape, (2, 64))
        selected = select_hypotheses(scores)
        codes = selected_factor_ids(scores, selected)
        self.assertEqual(codes.shape, (2, 4, 3))
        for row in selected:
            self.assertEqual(torch.unique(row).numel(), 4)
        prediction = predict_factor_panel(
            executor,
            batch.query_states,
            batch.query_actions,
            codes,
            batch.query_action_mask,
        )
        self.assertEqual(prediction.change_logits.shape, (6, 4, 8, 8))
        self.assertEqual(prediction.new_color_logits.shape, (6, 4, 16, 8, 8))

    def test_lexicographic_selection_prioritizes_map_error(self) -> None:
        assert torch is not None
        scores = HypothesisBankScores(
            factor_ids=torch.tensor(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]
            ),
            proper_nll_per_cell=torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            balanced_nll_per_cell=torch.tensor([[9.0, 0.1, 0.2, 0.3]]),
            map_error_cells=torch.tensor([[0, 1, 0, 2]]),
            map_exact=torch.tensor([[True, False, True, False]]),
        )
        selected = select_hypotheses(
            scores,
            particles=4,
            method="map_then_balanced_nll",
        )
        self.assertEqual(selected.tolist(), [[2, 0, 1, 3]])


if __name__ == "__main__":
    unittest.main()
