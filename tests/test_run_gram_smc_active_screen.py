"""Runner-level checks for the fixed sequential GRAM-VPS screen."""

from __future__ import annotations

import math
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_gram_smc_active_screen import (
    _aggregate_snapshot_metrics,
    _build_paired_evidence,
    _conditional_recovery_metrics,
    _exact_snapshots,
    _fresh_stage_seed,
    _js_divergence_bits,
    _method_report,
    _snapshot_from_particles,
    _symbolic_population_stages,
    _support_batch,
    _uniform_iid_proposal_stages,
)


class SequentialBeliefMetricTests(unittest.TestCase):
    def test_duplicate_particles_are_mass_aggregated(self) -> None:
        snapshot = _snapshot_from_particles(
            [(0, 1, 2), (0, 1, 2), (3, 1, 2), (2, 1, 2)],
            [0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(
            snapshot.unique_codes,
            ((2, 1, 2), (0, 1, 2), (3, 1, 2)),
        )
        self.assertTrue(
            all(
                math.isclose(left, right)
                for left, right in zip(
                    snapshot.unique_weights, (0.4, 0.3, 0.3), strict=True
                )
            )
        )
        self.assertAlmostEqual(
            snapshot.effective_sample_size,
            1.0 / (0.1**2 + 0.2**2 + 0.3**2 + 0.4**2),
        )

    def test_js_divergence_is_zero_for_equal_and_one_for_disjoint(self) -> None:
        left = _snapshot_from_particles([(0, 0, 0), (1, 0, 0)])
        same = _snapshot_from_particles([(1, 0, 0), (0, 0, 0)])
        disjoint = _snapshot_from_particles([(2, 0, 0), (3, 0, 0)])
        self.assertAlmostEqual(_js_divergence_bits(left, same), 0.0)
        self.assertAlmostEqual(_js_divergence_bits(left, disjoint), 1.0)

    def test_conditional_recovery_reports_explicit_denominators(self) -> None:
        target = (0, 0, 0)
        before = (
            _snapshot_from_particles(((1, 0, 0),)),
            _snapshot_from_particles((target, (1, 0, 0))),
            _snapshot_from_particles(((1, 0, 0), target), (0.8, 0.2)),
        )
        after = (
            _snapshot_from_particles((target,)),
            _snapshot_from_particles((target,)),
            _snapshot_from_particles((target,)),
        )
        metrics = _conditional_recovery_metrics(
            before, after, (target, target, target)
        )
        self.assertEqual(metrics["target_missing_at_t2_tasks"], 1)
        self.assertEqual(metrics["target_recovered_in_bank_at_t3_tasks"], 1)
        self.assertEqual(metrics["top1_wrong_at_t2_tasks"], 2)
        self.assertEqual(metrics["top1_corrected_at_t3_tasks"], 2)
        self.assertAlmostEqual(metrics["p_target_in_bank_t3_given_missing_t2"], 1.0)
        self.assertAlmostEqual(metrics["p_top1_target_t3_given_wrong_t2"], 1.0)


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class SequentialEvidenceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from prp_wm.pilot import make_pilot_tasks

        cls.tasks = make_pilot_tasks(
            split="gram-vps-sequential-runner-test",
            master_seed=2026071601,
            start=0,
            count=8,
            diagnostic_indices=(0,),
        )
        cls.paired = _build_paired_evidence(cls.tasks)

    def test_fixed_sequence_has_expected_version_space_transitions(self) -> None:
        from prp_wm.rulegrid import version_space

        self.assertEqual(
            [len(self.paired.factual_histories[index][0]) for index in range(4)],
            [6, 7, 8, 9],
        )
        for task_index, task in enumerate(self.tasks):
            factual_sizes = [
                len(version_space(stage[task_index], task.privileged.palette))
                for stage in self.paired.factual_histories
            ]
            counterfactual_sizes = [
                len(version_space(stage[task_index], task.privileged.palette))
                for stage in self.paired.counterfactual_histories
            ]
            self.assertEqual(factual_sizes[0], 4)
            self.assertIn(factual_sizes[1], (1, 3))
            self.assertEqual(factual_sizes[2], factual_sizes[1])
            self.assertEqual(factual_sizes[3], 1)
            self.assertEqual(counterfactual_sizes[0], 4)
            self.assertIn(counterfactual_sizes[1], (1, 3))
            self.assertEqual(counterfactual_sizes[2], counterfactual_sizes[1])
            self.assertEqual(counterfactual_sizes[3], 1)

    def test_counterfactual_changes_targets_not_public_inputs(self) -> None:
        for task_index in range(len(self.tasks)):
            for stage in range(4):
                factual = self.paired.factual_histories[stage][task_index]
                counterfactual = self.paired.counterfactual_histories[stage][task_index]
                self.assertEqual(len(factual), len(counterfactual))
                for left, right in zip(factual, counterfactual, strict=True):
                    self.assertEqual(left.state, right.state)
                    self.assertEqual(left.action, right.action)
            self.assertNotEqual(
                self.paired.factual_histories[1][task_index][-1].next_state,
                self.paired.counterfactual_histories[1][task_index][-1].next_state,
            )
            self.assertEqual(
                self.paired.factual_histories[2][task_index][-1].next_state,
                self.paired.counterfactual_histories[2][task_index][-1].next_state,
            )

    def test_dynamic_history_batch_is_support_only_and_canonical(self) -> None:
        batch = _support_batch(
            torch,
            self.tasks,
            self.paired.factual_histories[3],
            device="cpu",
        )
        self.assertEqual(tuple(batch.support_states.shape), (8, 9, 8, 8))
        self.assertEqual(tuple(batch.support_targets.shape), (8, 9, 8, 8))
        self.assertEqual(tuple(batch.support_actions.shape), (8, 9, 4))
        self.assertTrue(bool(batch.support_mask.all()))
        self.assertIsNone(batch.query_states)
        self.assertIsNone(batch.query_actions)
        self.assertIsNone(batch.query_targets)
        self.assertIsNone(batch.behavior_targets)
        # Palette-role colors are canonicalized, while unrelated distractor
        # colors intentionally remain valid members of the 16-color palette.
        self.assertLess(int(batch.support_states.max()), 16)

    def test_exact_filter_is_neutral_stable_and_strong_exact(self) -> None:
        factual = tuple(
            _exact_snapshots(self.tasks, histories)
            for histories in self.paired.factual_histories
        )
        counterfactual = tuple(
            _exact_snapshots(self.tasks, histories)
            for histories in self.paired.counterfactual_histories
        )
        report = _method_report(
            tasks=self.tasks,
            paired=self.paired,
            factual=factual,
            counterfactual=counterfactual,
        )
        self.assertAlmostEqual(
            report["neutral_distractor_stability"][
                "factual_mean_jsd_t1_to_t2_bits"
            ],
            0.0,
        )
        final = report["stages"][3]
        self.assertEqual(
            final["factual"]["exact_version_space_size_histogram"], {"1": 8}
        )
        self.assertAlmostEqual(
            final["factual"]["target_code_in_belief_rate"], 1.0
        )
        self.assertAlmostEqual(final["factual"]["mean_target_code_weight"], 1.0)
        self.assertAlmostEqual(
            final["counterfactual"]["mean_target_code_weight"], 1.0
        )

    def test_metrics_distinguish_coverage_precision_and_recall(self) -> None:
        # At t0 each task has four compatible rules.  Keeping only one valid
        # code gives perfect precision but one-quarter recall.
        snapshots = tuple(
            _snapshot_from_particles(
                [_exact_snapshots((task,), (history,))[0].unique_codes[0]]
            )
            for task, history in zip(
                self.tasks,
                self.paired.factual_histories[0],
                strict=True,
            )
        )
        metrics = _aggregate_snapshot_metrics(
            snapshots,
            self.tasks,
            self.paired.factual_histories[0],
            self.paired.factual_programs,
        )
        self.assertAlmostEqual(metrics["mean_unique_compatible_precision"], 1.0)
        self.assertAlmostEqual(metrics["mean_exact_version_space_recall"], 0.25)

    def test_symbolic_population_carries_a_surviving_code_without_bank_fallback(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids

        task = self.tasks[0]
        target = rule_program_factor_ids(task.privileged.true_program)
        invalid = next(
            code
            for code in ((a, b, c) for a in range(4) for b in range(4) for c in range(4))
            if code not in _exact_snapshots(
                (task,), (self.paired.factual_histories[0][0],)
            )[0].unique_codes
        )
        fresh = (
            (_snapshot_from_particles((target, invalid)),),
            (_snapshot_from_particles((invalid, invalid)),),
        )
        stages = _symbolic_population_stages(
            fresh_proposals_by_stage=fresh,
            tasks=(task,),
            histories_by_stage=(
                (self.paired.factual_histories[0][0],),
                (self.paired.factual_histories[1][0],),
            ),
            carry_limit=4,
        )
        self.assertIn(target, stages[0][0].unique_codes)
        self.assertIn(target, stages[1][0].unique_codes)

    def test_uniform_iid_proposals_are_reproducible_and_stage_fresh(self) -> None:
        first = _uniform_iid_proposal_stages(
            task_count=2, stages=3, proposals=8, seed=71
        )
        second = _uniform_iid_proposal_stages(
            task_count=2, stages=3, proposals=8, seed=71
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first[0][0].particle_codes, first[1][0].particle_codes)

    def test_population_stage_seeds_are_fresh_while_pairable(self) -> None:
        seeds = [_fresh_stage_seed(900, stage) for stage in range(4)]
        self.assertEqual(seeds, [900, 901, 902, 903])
        self.assertEqual(_fresh_stage_seed(900, 2), _fresh_stage_seed(900, 2))
        with self.assertRaises(ValueError):
            _fresh_stage_seed(900, -1)


if __name__ == "__main__":
    unittest.main()
