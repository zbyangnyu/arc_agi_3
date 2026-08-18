"""Protocol and tensor tests for the forced cross-axis likelihood audit."""

from __future__ import annotations

import math
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_nuisance_learned_bridge import (
    _deterministic_log_likelihood,
)
from scripts.run_forced_cross_axis_likelihood_audit import (
    axis_project_log_likelihood,
    deduplicate_semantic_records,
    evaluate_gates,
    forced_bayes_fork,
    percentile,
    safe_true_query_log_odds,
    score_public_prediction_against_program_feedback,
)
from scripts.run_oracle_canonical_acquisition_ceiling import (
    _candidate_panel,
    _door_marginals,
    bayesian_log_likelihood_update,
)
from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (
    CANDIDATES_PER_TASK,
    _candidate_menu,
    _exact_candidate_outcome_maps,
    _neutral_support,
)


PROGRAMS_PER_PUBLIC_ENVIRONMENT = 64
QUERY_PROBES = 2
FORCED_PROBES = 6
PRIMARY_SEMANTIC_CASES = 3 * 64 * QUERY_PROBES * FORCED_PROBES
AUDIT_SPLIT = "forced-cross-axis-likelihood-locality-v1"
MASTER_SEED = 2026072401


def _public_environment(axis: object, group_index: int):
    """Build one palette with all programs and a program-independent public view."""

    from prp_wm.rulegrid import (
        ALL_PROGRAMS,
        RuleGridInferenceView,
        RuleGridPrivilegedTargets,
        RuleGridTask,
        derive_seed64,
        palette_from_seed,
        simulate,
    )

    palette = palette_from_seed(
        derive_seed64(
            AUDIT_SPLIT,
            axis,
            group_index,
            "palette",
            master_seed=MASTER_SEED,
        )
    )
    candidates, kinds = _candidate_menu(
        query_axis=axis,
        palette=palette,
        order_seed=derive_seed64(
            AUDIT_SPLIT,
            axis,
            group_index,
            "candidate_order",
            master_seed=MASTER_SEED,
        ),
    )
    support = _neutral_support(ALL_PROGRAMS[0], palette)
    inference = RuleGridInferenceView(
        task_id=f"{AUDIT_SPLIT}/Q{axis.value}/G{group_index:04d}",
        support=support,
        active_candidates=candidates,
        diagnostics=(),
    )
    return tuple(
        RuleGridTask(
            inference=inference,
            privileged=RuleGridPrivilegedTargets(
                true_program=program,
                palette=palette,
                candidate_kinds=kinds,
                active_targets=tuple(
                    simulate(
                        probe.state,
                        probe.action,
                        program,
                        palette,
                    )
                    for probe in candidates
                ),
                diagnostic_targets=(),
                diagnostic_target_indices=(),
            ),
        )
        for program in ALL_PROGRAMS
    )


def _candidate_multiset_signature(states, actions, action_mask):
    """Return an order-invariant signature for one canonical public menu."""

    rows = []
    for index in range(states.shape[0]):
        row = tuple(int(value) for value in states[index].flatten().tolist())
        row += tuple(int(value) for value in actions[index].flatten().tolist())
        if action_mask is not None:
            row += tuple(
                int(value) for value in action_mask[index].flatten().tolist()
            )
        rows.append(row)
    return tuple(sorted(rows))


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class ForcedCrossAxisOracleProtocolTests(unittest.TestCase):
    def test_public_environment_covers_all_codes_without_program_leakage(
        self,
    ) -> None:
        from prp_wm.causal_filter import enumerate_factor_codes
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.rulegrid import ALL_AXES

        factor_bank = enumerate_factor_codes(device="cpu")
        for axis in ALL_AXES:
            tasks = _public_environment(axis, 0)
            self.assertEqual(
                len(tasks),
                PROGRAMS_PER_PUBLIC_ENVIRONMENT,
            )
            self.assertEqual(
                [
                    rule_program_factor_ids(task.privileged.true_program)
                    for task in tasks
                ],
                [tuple(int(value) for value in row) for row in factor_bank],
            )
            self.assertTrue(
                all(task.inference == tasks[0].inference for task in tasks)
            )
            self.assertTrue(
                all(
                    task.privileged.palette == tasks[0].privileged.palette
                    for task in tasks
                )
            )
            kinds = tasks[0].privileged.candidate_kinds
            self.assertEqual(len(kinds), CANDIDATES_PER_TASK)
            self.assertEqual(kinds.count("query-atomic"), QUERY_PROBES)
            self.assertEqual(kinds.count("nuisance-axis-0"), 2)
            self.assertEqual(kinds.count("nuisance-axis-1"), 2)
            self.assertEqual(kinds.count("neutral"), 2)

            states, actions, action_mask, _ = _candidate_panel(
                torch=torch,
                tasks=tasks,
                device=torch.device("cpu"),
            )
            self.assertTrue(
                torch.equal(
                    states,
                    states[:1].expand_as(states),
                )
            )
            self.assertTrue(
                torch.equal(
                    actions,
                    actions[:1].expand_as(actions),
                )
            )
            if action_mask is not None:
                self.assertTrue(
                    torch.equal(
                        action_mask,
                        action_mask[:1].expand_as(action_mask),
                    )
                )

    def test_palette_repeats_have_one_canonical_public_template(
        self,
    ) -> None:
        """Palette/order repeats are invariance checks, not independent clusters."""

        from prp_wm.rulegrid import ALL_AXES

        for axis in ALL_AXES:
            signatures = []
            for group_index in range(64):
                representative = _public_environment(axis, group_index)[:1]
                states, actions, action_mask, _ = _candidate_panel(
                    torch=torch,
                    tasks=representative,
                    device=torch.device("cpu"),
                )
                signatures.append(
                    _candidate_multiset_signature(
                        states[0],
                        actions[0],
                        (
                            action_mask[0]
                            if action_mask is not None
                            else None
                        ),
                    )
                )
            self.assertEqual(len(set(signatures)), 1)

    def test_exact_query_then_independent_forced_controls(self) -> None:
        """Both query variants leave 16; cross probes leave 4; neutral leaves 16."""

        from prp_wm.causal_filter import enumerate_factor_codes
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.rulegrid import ALL_AXES

        factor_bank = enumerate_factor_codes(device="cpu")
        prior = torch.full(
            (PROGRAMS_PER_PUBLIC_ENVIRONMENT, len(factor_bank)),
            -math.log(float(len(factor_bank))),
        )
        for query_axis_index, axis in enumerate(ALL_AXES):
            tasks = _public_environment(axis, 0)
            exact_maps, feedback = _exact_candidate_outcome_maps(
                torch=torch,
                tasks=tasks,
                factor_bank=factor_bank,
            )
            kinds = tasks[0].privileged.candidate_kinds
            query_indices = [
                index
                for index, kind in enumerate(kinds)
                if kind == "query-atomic"
            ]
            forced_indices = [
                index
                for index, kind in enumerate(kinds)
                if kind != "query-atomic"
            ]
            self.assertEqual(
                (len(query_indices), len(forced_indices)),
                (QUERY_PROBES, FORCED_PROBES),
            )

            for true_index, task in enumerate(tasks):
                true_code = torch.tensor(
                    rule_program_factor_ids(task.privileged.true_program),
                    dtype=factor_bank.dtype,
                )
                code_index = torch.nonzero(
                    factor_bank.eq(true_code).all(dim=-1)
                ).flatten()
                self.assertEqual(code_index.tolist(), [true_index])
                self.assertTrue(
                    torch.equal(
                        exact_maps[true_index, :, true_index],
                        feedback[true_index],
                    )
                )

            for query_index in query_indices:
                query_likelihood = _deterministic_log_likelihood(
                    torch=torch,
                    selected_outcome_maps=exact_maps[:, query_index],
                    observed_feedback=feedback[:, query_index],
                )
                query_posterior, _ = bayesian_log_likelihood_update(
                    torch,
                    prior,
                    query_likelihood,
                )
                self.assertTrue(
                    (
                        torch.isfinite(query_posterior).sum(dim=-1) == 16
                    ).all()
                )

                query_marginals = []
                for task_index, task in enumerate(tasks):
                    surviving = factor_bank[
                        torch.isfinite(query_posterior[task_index])
                    ]
                    true_value = rule_program_factor_ids(
                        task.privileged.true_program
                    )[query_axis_index]
                    self.assertEqual(
                        surviving[:, query_axis_index].unique().tolist(),
                        [true_value],
                    )
                    for other_axis in (
                        set(range(len(ALL_AXES))) - {query_axis_index}
                    ):
                        self.assertEqual(
                            surviving[:, other_axis].unique().numel(),
                            4,
                        )
                    query_marginals.append(
                        _door_marginals(
                            torch,
                            query_posterior[task_index].exp(),
                            factor_bank[:, query_axis_index],
                        )
                    )
                query_marginals = torch.stack(query_marginals)

                # Every path must fork from this same query posterior.
                for forced_index in forced_indices:
                    forced_likelihood = _deterministic_log_likelihood(
                        torch=torch,
                        selected_outcome_maps=exact_maps[:, forced_index],
                        observed_feedback=feedback[:, forced_index],
                    )
                    forced_posterior, _ = bayesian_log_likelihood_update(
                        torch,
                        query_posterior.clone(),
                        forced_likelihood,
                    )
                    expected_survivors = (
                        16 if kinds[forced_index] == "neutral" else 4
                    )
                    self.assertTrue(
                        (
                            torch.isfinite(forced_posterior).sum(dim=-1)
                            == expected_survivors
                        ).all()
                    )
                    forced_query_marginals = torch.stack(
                        [
                            _door_marginals(
                                torch,
                                forced_posterior[task_index].exp(),
                                factor_bank[:, query_axis_index],
                            )
                            for task_index in range(len(tasks))
                        ]
                    )
                    self.assertTrue(
                        torch.equal(
                            forced_query_marginals,
                            query_marginals,
                        )
                    )


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class ForcedCrossAxisRunnerContractTests(unittest.TestCase):
    def test_full_grid_axis_projection_is_logmeanexp_and_local(self) -> None:
        from prp_wm.causal_filter import enumerate_factor_codes

        factor_bank = enumerate_factor_codes(device="cpu")
        likelihood = torch.full((1, 64), -100.0)
        likelihood[0, 0] = 0.0
        projected = axis_project_log_likelihood(
            torch,
            likelihood,
            factor_bank,
            0,
        )
        first_fiber = factor_bank[:, 0].eq(0)
        expected = torch.logsumexp(
            likelihood[:, first_fiber],
            dim=-1,
        ) - math.log(16.0)
        self.assertTrue(
            torch.allclose(
                projected[:, first_fiber],
                expected[:, None].expand(-1, 16),
            )
        )
        self.assertAlmostEqual(float(expected), -math.log(16.0), places=5)
        self.assertNotAlmostEqual(
            float(expected),
            float(likelihood[:, first_fiber].mean()),
            places=2,
        )
        for value in range(4):
            fiber = factor_bank[:, 0].eq(value)
            self.assertEqual(
                int(torch.unique(projected[:, fiber]).numel()),
                1,
            )

        shifted = axis_project_log_likelihood(
            torch,
            likelihood + 7.0,
            factor_bank,
            0,
        )
        self.assertTrue(torch.allclose(shifted, projected + 7.0))

        prior = torch.full((1, 64), -math.log(64.0))
        _, raw_evidence = bayesian_log_likelihood_update(
            torch,
            prior,
            likelihood,
        )
        _, projected_evidence = bayesian_log_likelihood_update(
            torch,
            prior,
            projected,
        )
        self.assertTrue(torch.allclose(raw_evidence, projected_evidence))

    def test_neutral_projection_is_constant_and_bayes_noop(self) -> None:
        from prp_wm.causal_filter import enumerate_factor_codes

        generator = torch.Generator().manual_seed(17)
        likelihood = torch.randn(2, 64, generator=generator)
        factor_bank = enumerate_factor_codes(device="cpu")
        projected = axis_project_log_likelihood(
            torch,
            likelihood,
            factor_bank,
            None,
        )
        expected = (
            torch.logsumexp(likelihood, dim=-1) - math.log(64.0)
        )
        self.assertTrue(
            torch.allclose(
                projected,
                expected[:, None].expand_as(projected),
            )
        )
        prior = torch.log_softmax(
            torch.randn(2, 64, generator=generator),
            dim=-1,
        )
        posterior, evidence = bayesian_log_likelihood_update(
            torch,
            prior,
            projected,
        )
        self.assertTrue(torch.isfinite(evidence).all())
        self.assertTrue(torch.allclose(posterior, prior, atol=1e-6))

    def test_ee_rr_rp_pr_pp_paths_fork_from_query_stage(self) -> None:
        query = torch.log_softmax(
            torch.tensor([[2.0, 1.0, 0.0, -1.0]]),
            dim=-1,
        )
        forced = torch.tensor(
            [[[0.0, -1.0, -2.0, -3.0], [-3.0, -2.0, -1.0, 0.0]]]
        )
        forked, evidence = forced_bayes_fork(torch, query, forced)
        self.assertEqual(tuple(forked.shape), (1, 2, 4))
        self.assertEqual(tuple(evidence.shape), (1, 2))
        for forced_index in range(2):
            expected, expected_evidence = bayesian_log_likelihood_update(
                torch,
                query,
                forced[:, forced_index],
            )
            self.assertTrue(
                torch.allclose(
                    forked[:, forced_index],
                    expected,
                )
            )
            self.assertTrue(
                torch.allclose(
                    evidence[:, forced_index],
                    expected_evidence,
                )
            )
        sequential, _ = bayesian_log_likelihood_update(
            torch,
            forked[:, 0],
            forced[:, 1],
        )
        self.assertFalse(torch.allclose(forked[:, 1], sequential))

    def test_flattened_prediction_broadcast_keeps_g_p_h_order(self) -> None:
        from prp_wm.neural import OutcomePrediction

        groups, targets, probes, hypotheses, colors = 2, 3, 2, 4, 5
        flattened = groups * probes
        input_colors = torch.zeros(
            flattened,
            1,
            1,
            dtype=torch.long,
        )
        change_logits = torch.arange(
            flattened * hypotheses,
            dtype=torch.float32,
        ).reshape(flattened, hypotheses, 1, 1) / 10.0
        new_color_logits = torch.arange(
            flattened * hypotheses * colors,
            dtype=torch.float32,
        ).reshape(flattened, hypotheses, colors, 1, 1) / 20.0
        prediction = OutcomePrediction(
            input_colors=input_colors,
            change_logits=change_logits,
            new_color_logits=new_color_logits,
        )
        feedback = torch.tensor(
            [
                [[[0]], [[1]]],
                [[[2]], [[3]]],
                [[[4]], [[0]]],
                [[[1]], [[2]]],
                [[[3]], [[4]]],
                [[[0]], [[1]]],
            ],
            dtype=torch.long,
        ).reshape(groups, targets, probes, 1, 1)
        scored = score_public_prediction_against_program_feedback(
            torch,
            prediction,
            feedback,
            probes=probes,
            target_chunk_size=2,
        )
        self.assertEqual(
            tuple(scored.shape),
            (groups, targets, probes, hypotheses),
        )
        for group in range(groups):
            for target in range(targets):
                for probe in range(probes):
                    row = group * probes + probe
                    selected = OutcomePrediction(
                        input_colors=prediction.input_colors[row : row + 1],
                        change_logits=prediction.change_logits[row : row + 1],
                        new_color_logits=prediction.new_color_logits[
                            row : row + 1
                        ],
                    )
                    expected = selected.log_prob(
                        feedback[group, target, probe : probe + 1]
                    )[0]
                    self.assertTrue(
                        torch.allclose(
                            scored[group, target, probe],
                            expected,
                        )
                    )

    def test_semantic_dedup_is_invariant_to_repeat_and_batch_count(self) -> None:
        self.assertEqual(PRIMARY_SEMANTIC_CASES, 2304)
        base = {
            "query_axis": "collision",
            "true_program_index": 0,
            "query_probe_key": "query:collision:v0",
            "forced_probe_key": "cross:trigger:v0",
            "branch": "RR",
            "candidate_index": 3,
            "query_candidate_index": 1,
            "metric": 2.0,
        }
        once = deduplicate_semantic_records(
            [{**base, "group_index": 0}]
        )
        repeated = deduplicate_semantic_records(
            [
                {
                    **base,
                    "group_index": group,
                    "candidate_index": group % 8,
                    "query_candidate_index": (group + 1) % 8,
                    "metric": 2.0 + (1e-6 if group else 0.0),
                }
                for group in range(64)
            ]
        )
        self.assertEqual(len(once), len(repeated))
        for key in base:
            if key not in {
                "candidate_index",
                "query_candidate_index",
            }:
                self.assertAlmostEqual(
                    float(once[0][key])
                    if key == "metric"
                    else 0.0,
                    float(repeated[0][key])
                    if key == "metric"
                    else 0.0,
                )
        self.assertEqual(repeated[0]["canonical_repeat_count"], 64)
        corrupted = [
            {**base, "group_index": 0},
            {**base, "group_index": 1, "metric": 2.01},
        ]
        with self.assertRaises(AssertionError):
            deduplicate_semantic_records(corrupted)

    def test_gate_thresholds_and_invalidity_are_recomputed(self) -> None:
        # Frozen "higher" P99: exactly one severe case in 100 reaches P99.
        self.assertEqual(percentile([0.0] * 99 + [10.0], 99.0), 10.0)

        summaries = {
            branch: {
                "cross": {
                    "catastrophic_reversal_rate": 0.0,
                    "p99_log_odds_drop_nats": 0.0,
                    "mean_query_nll_nats": 1.0,
                    "semantic_sequences": 1,
                },
                # Deliberately extreme neutral values: the architecture gate
                # must use only the preregistered cross-axis denominator.
                "neutral": {
                    "catastrophic_reversal_rate": 1.0,
                    "p99_log_odds_drop_nats": 100.0,
                    "mean_query_nll_nats": 1.0,
                    "semantic_sequences": 1,
                },
            }
            for branch in ("EE", "RR", "RP", "PR", "PP")
        }
        summaries["RR"]["cross"]["catastrophic_reversal_rate"] = 0.01
        # RP intentionally fails rescue.  PP is the preregistered rescue path.
        summaries["RP"]["cross"]["catastrophic_reversal_rate"] = 0.009
        summaries["PP"]["cross"]["catastrophic_reversal_rate"] = 0.001
        rows = []
        for branch, reversal in (
            ("RR", True),
            ("RP", True),
            ("PP", False),
        ):
            rows.append(
                {
                    "query_axis": "collision",
                    "true_program_index": 0,
                    "query_probe_key": "query:collision:v0",
                    "forced_probe_key": "cross:trigger:v0",
                    "branch": branch,
                    "catastrophic_reversal": reversal,
                }
            )
        gate = evaluate_gates(
            summaries,
            rows,
            expected_semantic_sequences=1,
        )
        self.assertEqual(
            gate["decision"],
            "support-factorized-jepa-executor",
        )
        self.assertTrue(gate["factor_locality_rescue_gate_passed"])

        summaries["PR"]["cross"]["mean_query_nll_nats"] = 1.05
        gate = evaluate_gates(
            summaries,
            rows,
            expected_semantic_sequences=1,
        )
        self.assertEqual(
            gate["decision"],
            "turn-to-within-axis-calibration",
        )
        invalid = evaluate_gates(
            summaries,
            rows,
            validity_passed=False,
            expected_semantic_sequences=1,
        )
        self.assertEqual(invalid["decision"], "invalid-no-decision")

        # The second disjunct has the same inclusive threshold and 90% rescue.
        summaries["PR"]["cross"]["mean_query_nll_nats"] = 1.0
        summaries["RR"]["cross"]["catastrophic_reversal_rate"] = 0.0
        summaries["PP"]["cross"]["catastrophic_reversal_rate"] = 0.0
        summaries["RR"]["cross"]["p99_log_odds_drop_nats"] = 5.0
        summaries["PP"]["cross"]["p99_log_odds_drop_nats"] = 0.5
        p99_gate = evaluate_gates(
            summaries,
            rows,
            expected_semantic_sequences=1,
        )
        self.assertEqual(
            p99_gate["decision"],
            "support-factorized-jepa-executor",
        )
        summaries["PP"]["cross"]["p99_log_odds_drop_nats"] = 0.51
        under_rescued = evaluate_gates(
            summaries,
            rows,
            expected_semantic_sequences=1,
        )
        self.assertEqual(
            under_rescued["decision"],
            "turn-to-within-axis-calibration",
        )

        summaries["RR"]["cross"]["p99_log_odds_drop_nats"] = 4.999
        summaries["PP"]["cross"]["p99_log_odds_drop_nats"] = 0.0
        no_raw_failure = evaluate_gates(
            summaries,
            rows,
            expected_semantic_sequences=1,
        )
        self.assertEqual(
            no_raw_failure["decision"],
            "turn-to-harder-compositional-benchmark",
        )

    def test_log_odds_is_stable_without_probability_clipping(self) -> None:
        from prp_wm.causal_filter import enumerate_factor_codes

        factor_bank = enumerate_factor_codes(device="cpu")
        log_weights = torch.full((1, 64), -1000.0)
        log_weights[:, factor_bank[:, 0].eq(0)] = 0.0
        odds = safe_true_query_log_odds(
            torch,
            log_weights,
            factor_bank,
            0,
            torch.tensor([0]),
        )
        self.assertTrue(torch.isfinite(odds).all())
        self.assertGreater(float(odds), 900.0)


if __name__ == "__main__":
    unittest.main()
