from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import unittest


from scripts.run_gram_proposal_mix_ablation import (
    FRESH_PROPOSALS,
    MIXTURE_GRAM_COUNTS,
    STAGES,
    _compose_mixture_stages,
    _gram_proposal_stages,
    _iid_uniform_proposal_stages,
    _latin_cover_proposal_stages,
    _latin_pairwise_cover_codes,
    _mixture_target_source_attribution,
    _proposal_family,
    _proposal_retention_gate_audit,
    _stream_seed,
    _uniform_analytic_baseline,
)
from scripts.run_gram_smc_active_screen import (
    _build_paired_evidence,
    _factor_code,
    _snapshot_from_particles,
    _symbolic_population_stages,
)


class MatchedProposalAblationTests(unittest.TestCase):
    def _constant_stages(self, code: tuple[int, int, int], tasks: int = 2):
        return tuple(
            tuple(
                _snapshot_from_particles([code] * FRESH_PROPOSALS)
                for _ in range(tasks)
            )
            for _ in range(STAGES)
        )

    def test_stream_seeds_are_reproducible_distinct_and_namespaced(self) -> None:
        first = _stream_seed(71, "gram", 2, 3)
        self.assertEqual(first, _stream_seed(71, "gram", 2, 3))
        self.assertEqual(
            len(
                {
                    _stream_seed(71, namespace, stage, task)
                    for namespace in ("gram", "uniform")
                    for stage in range(4)
                    for task in range(3)
                }
            ),
            24,
        )

    def test_latin_cover_has_exact_w32_pairwise_balance(self) -> None:
        codes = _latin_pairwise_cover_codes(83)
        self.assertEqual(len(codes), 32)
        self.assertEqual(len(set(codes)), 32)
        for axis in range(3):
            self.assertEqual(
                Counter(code[axis] for code in codes),
                Counter({value: 8 for value in range(4)}),
            )
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertEqual(
                    Counter((code[left], code[right]) for code in codes),
                    Counter({(x, y): 2 for x in range(4) for y in range(4)}),
                )

    def test_latin_banks_are_reproducible_stage_fresh_and_w32(self) -> None:
        first = _latin_cover_proposal_stages(task_count=2, seed=101)
        second = _latin_cover_proposal_stages(task_count=2, seed=101)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0][0].particle_codes, first[1][0].particle_codes)
        for stage in first:
            for snapshot in stage:
                self.assertEqual(len(snapshot.particle_codes), FRESH_PROPOSALS)
                self.assertEqual(len(set(snapshot.particle_codes)), FRESH_PROPOSALS)

    def test_uniform_bank_is_reproducible_stage_fresh_and_exact_width(self) -> None:
        first = _iid_uniform_proposal_stages(task_count=2, seed=109)
        second = _iid_uniform_proposal_stages(task_count=2, seed=109)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0][0].particle_codes, first[1][0].particle_codes)
        self.assertTrue(
            all(
                len(snapshot.particle_codes) == FRESH_PROPOSALS
                for stage in first
                for snapshot in stage
            )
        )

    def test_all_mixtures_are_nested_prefixes_with_exact_total_width(self) -> None:
        gram = self._constant_stages((0, 0, 0))
        uniform = self._constant_stages((3, 3, 3))
        for gram_count in MIXTURE_GRAM_COUNTS:
            mixture = _compose_mixture_stages(
                gram, uniform, gram_count=gram_count
            )
            for stage in mixture:
                for snapshot in stage:
                    self.assertEqual(len(snapshot.particle_codes), 32)
                    self.assertEqual(
                        snapshot.particle_codes[:gram_count],
                        ((0, 0, 0),) * gram_count,
                    )
                    self.assertEqual(
                        snapshot.particle_codes[gram_count:],
                        ((3, 3, 3),) * (32 - gram_count),
                    )

    def test_proposal_family_has_six_methods_and_never_changes_width(self) -> None:
        gram = self._constant_stages((0, 0, 0))
        uniform = self._constant_stages((3, 3, 3))
        latin = _latin_cover_proposal_stages(task_count=2, seed=127)
        family = _proposal_family(gram, uniform, latin)
        self.assertEqual(
            set(family),
            {
                "gram32_uniform0",
                "gram24_uniform8",
                "gram16_uniform16",
                "gram8_uniform24",
                "gram0_uniform32",
                "latin_pairwise_cover32",
            },
        )
        self.assertTrue(
            all(
                len(snapshot.particle_codes) == 32
                for stages in family.values()
                for stage in stages
                for snapshot in stage
            )
        )

    def test_gram_branches_reuse_stage_seeds(self) -> None:
        import torch

        class FakeGRAM:
            def __init__(self) -> None:
                self.seeds: list[int] = []

            def sample_width_candidates(
                self,
                batch,
                *,
                width,
                recursive_steps,
                seed,
                temperature,
                sample_noise,
            ):
                del batch, recursive_steps, temperature, sample_noise
                self.seeds.append(seed)
                generator = torch.Generator().manual_seed(seed)
                return SimpleNamespace(
                    factor_ids=torch.randint(
                        0, 4, (2, width, 3), generator=generator
                    )
                )

        model = FakeGRAM()
        left = _gram_proposal_stages(
            torch=torch,
            gram=model,
            batches=(object(),) * STAGES,
            seed=137,
            recursive_steps=4,
            temperature=1.0,
        )
        right = _gram_proposal_stages(
            torch=torch,
            gram=model,
            batches=(object(),) * STAGES,
            seed=137,
            recursive_steps=4,
            temperature=1.0,
        )
        self.assertEqual(left, right)
        self.assertEqual(model.seeds[:STAGES], model.seeds[STAGES:])
        self.assertEqual(len(set(model.seeds[:STAGES])), STAGES)

    def test_symbolic_filter_does_not_inject_missing_true_code(self) -> None:
        from prp_wm.pilot import make_pilot_tasks

        task = make_pilot_tasks(
            split="proposal-ablation-no-injection",
            master_seed=2026071601,
            start=0,
            count=1,
            diagnostic_indices=(0,),
        )[0]
        paired = _build_paired_evidence((task,))
        target = _factor_code(paired.factual_programs[0])
        absent = next(
            (x, y, z)
            for x in range(4)
            for y in range(4)
            for z in range(4)
            if (x, y, z) != target
        )
        fresh = self._constant_stages(absent, tasks=1)
        retained = _symbolic_population_stages(
            fresh_proposals_by_stage=fresh,
            tasks=(task,),
            histories_by_stage=paired.factual_histories,
            carry_limit=32,
        )
        self.assertNotIn(target, retained[-1][0].unique_codes)

    def test_fresh_union_belief_gate_definitions_and_conditionals(self) -> None:
        from prp_wm.pilot import make_pilot_tasks

        task = make_pilot_tasks(
            split="proposal-gate-audit-test",
            master_seed=2026071601,
            start=0,
            count=1,
            diagnostic_indices=(0,),
        )[0]
        target_program = task.privileged.true_program
        target = _factor_code(target_program)
        absent = next(
            (x, y, z)
            for x in range(4)
            for y in range(4)
            for z in range(4)
            if (x, y, z) != target
        )
        fresh = (
            (_snapshot_from_particles([absent] * 32),),
            (_snapshot_from_particles([absent] * 32),),
            (_snapshot_from_particles([absent] * 32),),
            (_snapshot_from_particles([target] + [absent] * 31),),
        )
        retained = (
            (_snapshot_from_particles([absent]),),
            (_snapshot_from_particles([absent]),),
            (_snapshot_from_particles([absent]),),
            (_snapshot_from_particles([target]),),
        )
        audit = _proposal_retention_gate_audit(
            fresh=fresh,
            retained=retained,
            target_programs=(target_program,),
        )
        self.assertEqual(audit["stages"][2]["p_U_t_cumulative_union_target"], 0.0)
        self.assertEqual(audit["stages"][3]["p_F_t_fresh_target"], 1.0)
        self.assertEqual(audit["stages"][3]["p_U_t_cumulative_union_target"], 1.0)
        self.assertEqual(audit["stages"][3]["p_B_t_retained_target"], 1.0)
        self.assertEqual(
            audit["fresh_recovery_p_F3_given_not_U2"]["probability"], 1.0
        )
        self.assertEqual(
            audit["belief_recovery_p_B3_given_not_B2"]["probability"], 1.0
        )

    def test_mixture_source_attribution_is_component_specific(self) -> None:
        from prp_wm.pilot import make_pilot_tasks

        task = make_pilot_tasks(
            split="proposal-source-attribution-test",
            master_seed=2026071601,
            start=0,
            count=1,
            diagnostic_indices=(0,),
        )[0]
        target_program = task.privileged.true_program
        target = _factor_code(target_program)
        absent = next(
            (x, y, z)
            for x in range(4)
            for y in range(4)
            for z in range(4)
            if (x, y, z) != target
        )
        gram = tuple(
            (_snapshot_from_particles([target] + [absent] * 31),)
            for _ in range(4)
        )
        uniform = tuple(
            (_snapshot_from_particles([absent] * 32),) for _ in range(4)
        )
        attribution = _mixture_target_source_attribution(
            gram=gram,
            uniform=uniform,
            gram_count=8,
            target_programs=(target_program,),
        )
        self.assertTrue(
            all(
                stage["counts"]
                == {"gram_only": 1, "uniform_only": 0, "both": 0, "neither": 0}
                for stage in attribution["stages"]
            )
        )

    def test_uniform_analytic_baseline_matches_closed_form(self) -> None:
        baseline = _uniform_analytic_baseline()
        expected_fresh = 1.0 - (63.0 / 64.0) ** 32
        expected_union_t3 = 1.0 - (63.0 / 64.0) ** 128
        self.assertAlmostEqual(
            baseline["stages"][0]["p_F_t_fresh_target"], expected_fresh
        )
        self.assertAlmostEqual(
            baseline["stages"][3]["p_U_t_cumulative_union_target"],
            expected_union_t3,
        )
        self.assertEqual(baseline["retention_p_B_t_given_U_t"], 1.0)


if __name__ == "__main__":
    unittest.main()
