"""Checks for the verifier-guided GRAM population search.

The historical module name contains ``smc``; these tests enforce that the
implemented algorithm has no path-weight, posterior, resampling, or mutation
semantics.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch
import unittest

try:
    import torch
    from prp_wm.causal_filter import score_hypothesis_bank
    from prp_wm.gram_causal_rules import GRAMFactorizedCausalK4
    from prp_wm.gram_smc import (
        GRAMPopulationMemory,
        GRAMVerifierPopulationSearch,
    )
    from prp_wm.latent_rules import (
        OracleFactorExecutor,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class GRAMVerifierPopulationSearchTests(unittest.TestCase):
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

    def _batch(self, *, count: int = 1):
        from prp_wm.pilot import make_pilot_tasks

        tasks = make_pilot_tasks(
            split="gram-population-search-test",
            master_seed=2026072203,
            start=0,
            count=count,
            diagnostic_indices=(0, 4, 8, 12),
        )
        return rulegrid_tasks_to_canonical_behavior_batch(
            tasks,
            diagnostic_indices=(0, 4, 8, 12),
        )

    @staticmethod
    def _support_prefix(batch, steps: int):
        return replace(
            batch,
            support_states=batch.support_states[:, :steps],
            support_actions=batch.support_actions[:, :steps],
            support_targets=batch.support_targets[:, :steps],
            support_mask=batch.support_mask[:, :steps],
            support_action_mask=(
                None
                if batch.support_action_mask is None
                else batch.support_action_mask[:, :steps]
            ),
        )

    def _components(self, *, recursive_steps: int = 2):
        config = self._config()
        proposal_executor = OracleFactorExecutor(config)
        proposer = GRAMFactorizedCausalK4(
            proposal_executor,
            recursive_steps=recursive_steps,
            guidance_dim=8,
        )
        verifier_executor = OracleFactorExecutor(config)
        return proposal_executor, proposer, verifier_executor

    def _search(
        self,
        *,
        proposals: int = 12,
        carry_limit: int | None = None,
        recursive_steps: int = 2,
        proposal_mode: str = "gram",
        ranking_inverse_temperature: float = 1.0,
    ) -> GRAMVerifierPopulationSearch:
        _, proposer, verifier = self._components(
            recursive_steps=recursive_steps
        )
        return GRAMVerifierPopulationSearch(
            proposer,
            verifier_executor=verifier,
            proposals=proposals,
            carry_limit=carry_limit,
            recursive_steps=recursive_steps,
            proposal_mode=proposal_mode,
            ranking_inverse_temperature=ranking_inverse_temperature,
        )

    @staticmethod
    def _memory(
        search: GRAMVerifierPopulationSearch,
        codes: list[tuple[int, int, int]],
        *,
        ranking_weights: list[float] | None = None,
        generation: int = 1,
    ) -> GRAMPopulationMemory:
        assert torch is not None
        if len(codes) > search.carry_limit:
            raise ValueError("too many codes for test memory")
        ids = torch.full(
            (1, search.carry_limit, 3), -1, dtype=torch.long
        )
        mask = torch.zeros((1, search.carry_limit), dtype=torch.bool)
        ids[0, : len(codes)] = torch.tensor(codes, dtype=torch.long)
        mask[0, : len(codes)] = True
        weights = torch.zeros((1, search.carry_limit))
        if ranking_weights is None:
            weights[0, : len(codes)] = 1.0 / len(codes)
        else:
            weights[0, : len(codes)] = torch.tensor(ranking_weights)
        energies = torch.full_like(weights, torch.inf)
        energies[mask] = 0.0
        maximum = torch.iinfo(torch.long).max
        errors = torch.full(
            (1, search.carry_limit), maximum, dtype=torch.long
        )
        errors[mask] = 0
        return GRAMPopulationMemory(
            factor_ids=ids,
            mask=mask,
            ranking_weights=weights,
            energies=energies,
            map_error_cells=errors,
            map_exact=mask.clone(),
            generation=generation,
        )

    def test_gram_stage_uses_fresh_final_proposals_and_is_reproducible(self) -> None:
        assert torch is not None
        torch.manual_seed(701)
        search = self._search(proposals=9, recursive_steps=3).eval()
        batch = self._batch()
        result = search.search(batch, seed=83)
        repeated = search.search(batch, seed=83)
        public = search.proposer._support_only_batch(batch)
        direct = search.proposer.sample_width_candidates(
            public,
            width=9,
            recursive_steps=3,
            seed=83,
            sample_noise=True,
        )

        self.assertEqual(result.proposed_factor_ids.shape, (1, 9, 3))
        self.assertTrue(
            torch.equal(result.proposed_factor_ids, repeated.proposed_factor_ids)
        )
        self.assertTrue(
            torch.equal(result.proposed_factor_ids, direct.factor_ids)
        )
        self.assertEqual(result.verifier_bank_evaluations, 1)
        self.assertEqual(result.population.generation, 1)

    def test_independent_verifier_never_overwrites_proposer_executor(self) -> None:
        assert torch is not None
        torch.manual_seed(709)
        proposal_executor, proposer, verifier = self._components()
        search = GRAMVerifierPopulationSearch(
            proposer,
            verifier_executor=verifier,
            proposals=8,
            recursive_steps=2,
        )
        self.assertIs(proposer.executor, proposal_executor)
        self.assertIs(search.verifier_executor, verifier)
        self.assertIsNot(search.verifier_executor, proposer.executor)

        with patch(
            "prp_wm.gram_smc.score_hypothesis_bank",
            wraps=score_hypothesis_bank,
        ) as score:
            search.search(self._batch(), seed=89)
        self.assertEqual(score.call_count, 1)
        self.assertIs(score.call_args.args[0], verifier)

        search.train()
        self.assertTrue(search.training)
        self.assertTrue(search.proposer.training)
        self.assertFalse(search.proposer.executor.training)
        self.assertFalse(search.verifier_executor.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in search.proposer.executor.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in search.verifier_executor.parameters()
            )
        )

    def test_inference_strips_query_and_privileged_behavior_fields(self) -> None:
        assert torch is not None
        torch.manual_seed(719)
        search = self._search(proposals=10).eval()
        batch = self._batch()
        changed = replace(
            batch,
            query_states=(batch.query_states + 3) % search.config.num_colors,
            query_actions=batch.query_actions.roll(1, dims=1),
            query_targets=(
                None
                if batch.query_targets is None
                else (batch.query_targets + 5) % search.config.num_colors
            ),
            behavior_targets=(
                batch.behavior_targets + 7
            ) % search.config.num_colors,
            behavior_mass=batch.behavior_mass.roll(1, dims=1),
        )
        first = search.search(batch, seed=97)
        second = search.search(changed, seed=97)
        for left, right in (
            (first.proposed_factor_ids, second.proposed_factor_ids),
            (first.candidate_factor_ids, second.candidate_factor_ids),
            (first.candidate_energies, second.candidate_energies),
            (first.candidate_map_error_cells, second.candidate_map_error_cells),
            (first.population.factor_ids, second.population.factor_ids),
            (first.population.ranking_weights, second.population.ranking_weights),
        ):
            self.assertTrue(torch.equal(left, right))

    def test_uniform_control_is_fresh_iid_at_every_stage(self) -> None:
        assert torch is not None
        torch.manual_seed(727)
        search = self._search(
            proposals=32,
            carry_limit=8,
            proposal_mode="uniform",
        ).eval()
        batch = self._batch()
        first = search.search(batch, seed=101)
        with_carry = search.search(
            batch,
            carried=first.population,
            seed=101,
        )
        different_seed = search.search(
            batch,
            carried=first.population,
            seed=103,
        )

        self.assertEqual(first.proposed_factor_ids.shape, (1, 32, 3))
        self.assertTrue(
            torch.equal(first.proposed_factor_ids, with_carry.proposed_factor_ids)
        )
        self.assertFalse(
            torch.equal(
                first.proposed_factor_ids,
                different_seed.proposed_factor_ids,
            )
        )
        self.assertTrue(
            torch.all(
                (first.proposed_factor_ids >= 0)
                & (first.proposed_factor_ids < 4)
            )
        )
        self.assertEqual(with_carry.population.generation, 2)

    def test_dedup_and_compatibility_first_selection_are_explicit(self) -> None:
        assert torch is not None
        torch.manual_seed(733)
        search = self._search(
            proposals=4,
            carry_limit=4,
            ranking_inverse_temperature=1.0,
        )
        code_a = (2, 0, 1)
        code_b = (0, 0, 1)
        code_c = (3, 0, 1)
        code_d = (1, 0, 1)
        proposals = torch.tensor(
            [[code_a, code_a, code_b, code_c]], dtype=torch.long
        )
        carried = self._memory(search, [code_b, code_d])
        bank = search.proposer.factor_bank
        energy_table = torch.full((1, 64), 20.0)
        map_errors = torch.full((1, 64), 9, dtype=torch.long)
        index = {
            tuple(int(value) for value in row): position
            for position, row in enumerate(bank.tolist())
        }
        for code, energy, error in (
            (code_a, 0.1, 3),
            (code_b, 5.0, 0),
            (code_c, 1.0, 0),
            (code_d, 0.0, 2),
        ):
            energy_table[0, index[code]] = energy
            map_errors[0, index[code]] = error

        candidates = search._deduplicate_candidates(
            proposals,
            carried,
            bank=bank,
            energy_table=energy_table,
            map_error_table=map_errors,
        )
        self.assertEqual(
            candidates[0][0, :4].tolist(),
            [list(code_c), list(code_b), list(code_d), list(code_a)],
        )
        self.assertEqual(candidates[5][0, :4].tolist(), [1, 2, 1, 2])
        self.assertEqual(candidates[6][0, :4].tolist(), [1, 1, 0, 2])
        self.assertEqual(candidates[7][0, :4].tolist(), [False, True, True, False])

        population, selection_weights = search._retain_population(
            candidates[0],
            candidates[1],
            candidates[2],
            candidates[3],
            generation=2,
        )
        # Exact-MAP codes survive even though invalid D/A have lower energies.
        self.assertEqual(
            population.factor_ids[0, :2].tolist(),
            [list(code_c), list(code_b)],
        )
        self.assertEqual(population.counts.tolist(), [2])
        self.assertTrue(bool(population.map_exact[0, :2].all()))
        self.assertGreater(
            float(population.ranking_weights[0, 0]),
            float(population.ranking_weights[0, 1]),
        )
        self.assertEqual(int(torch.count_nonzero(selection_weights)), 2)

    def test_scored_candidates_never_include_unproposed_bank_codes(self) -> None:
        assert torch is not None
        torch.manual_seed(739)
        search = self._search(
            proposals=12,
            carry_limit=5,
            proposal_mode="uniform",
        ).eval()
        carried_codes = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        carried = self._memory(search, carried_codes)
        result = search.search(self._batch(), carried=carried, seed=107)

        proposed = {
            tuple(row)
            for row in result.proposed_factor_ids[0].detach().cpu().tolist()
        }
        allowed = proposed.union(carried_codes)
        returned = {
            tuple(row)
            for row in result.candidate_factor_ids[0, result.candidate_mask[0]]
            .detach()
            .cpu()
            .tolist()
        }
        self.assertEqual(returned, allowed)
        self.assertLessEqual(len(returned), 12 + len(carried_codes))
        self.assertEqual(result.verifier_bank_evaluations, 1)

    def test_current_code_rescore_ignores_path_weights_regression(self) -> None:
        """A low prior path score cannot poison a now-good final code."""

        assert torch is not None
        energies = torch.tensor([[1000.0, 0.0]])
        mask = torch.tensor([[True, True]])
        weights = GRAMVerifierPopulationSearch._stable_ranking_weights(
            energies,
            mask,
            inverse_temperature=1.0,
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights.tolist(), [[0.0, 1.0]])

        torch.manual_seed(743)
        search = self._search(
            proposals=16,
            carry_limit=4,
            proposal_mode="uniform",
        ).eval()
        codes = [(0, 0, 0), (3, 3, 3)]
        formerly_good = self._memory(
            search,
            codes,
            ranking_weights=[1.0, 0.0],
        )
        formerly_bad = self._memory(
            search,
            codes,
            ranking_weights=[0.0, 1.0],
        )
        batch = self._batch()
        first = search.search(batch, carried=formerly_good, seed=109)
        second = search.search(batch, carried=formerly_bad, seed=109)
        # Only carried codes matter; their inherited numerical scores do not.
        self.assertTrue(
            torch.equal(first.candidate_factor_ids, second.candidate_factor_ids)
        )
        self.assertTrue(
            torch.equal(first.candidate_energies, second.candidate_energies)
        )
        self.assertTrue(
            torch.equal(
                first.population.ranking_weights,
                second.population.ranking_weights,
            )
        )

    def test_full_history_is_scored_once_and_never_accumulates_weights(self) -> None:
        assert torch is not None
        torch.manual_seed(751)
        search = self._search(
            proposals=8,
            carry_limit=8,
            proposal_mode="uniform",
        ).eval()
        full = self._batch()
        prefix = self._support_prefix(full, 1)
        first = search.search(prefix, seed=113)
        second = search.search(full, carried=first.population, seed=127)

        self.assertEqual(first.applied_energy_scales.tolist(), [64.0])
        self.assertEqual(second.applied_energy_scales.tolist(), [384.0])
        self.assertEqual(first.verifier_bank_evaluations, 1)
        self.assertEqual(second.verifier_bank_evaluations, 1)
        self.assertEqual(second.population.generation, 2)
        with self.assertRaisesRegex(ValueError, "never accumulates path weights"):
            search.update(
                full,
                first.population,
                accumulate_weights=True,
                seed=127,
            )

    def test_outputs_are_gradient_free_and_topk_adapter_is_well_formed(self) -> None:
        assert torch is not None
        torch.manual_seed(757)
        search = self._search(proposals=8, carry_limit=8)
        search.train()
        result = search.search(self._batch(), seed=131)
        for tensor in (
            result.proposed_factor_ids,
            result.candidate_energies,
            result.candidate_map_error_cells,
            result.population.factor_ids,
            result.population.ranking_weights,
        ):
            self.assertFalse(tensor.requires_grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in search.parameters())
        )
        inference = search.topk_inference(result, k=4)
        self.assertEqual(inference.factor_ids.shape, (1, 4, 3))
        self.assertFalse(inference.rule_latents.requires_grad)
        retained = {
            tuple(row)
            for row in result.population.factor_ids[
                0, result.population.mask[0]
            ].tolist()
        }
        self.assertTrue(
            all(tuple(row) in retained for row in inference.factor_ids[0].tolist())
        )


if __name__ == "__main__":
    unittest.main()
