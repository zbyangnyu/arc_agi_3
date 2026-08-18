"""Tests for the privileged query-conditioned nuisance ceiling."""

from __future__ import annotations

from argparse import Namespace
import inspect
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (
    CANDIDATES_PER_TASK,
    PROGRAMS_PER_GROUP,
    _build_nuisance_tasks,
    _symbolic_initial_joint_log_weights,
    _select_global_information_candidate,
    _select_query_conditioned_candidate,
    _selector_boundary_audit,
    _uniform_orders,
    _validate_args,
)


class NuisanceCeilingArgumentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> Namespace:
        values: dict[str, object] = {
            "groups_per_query": 1,
            "batch_size": 2,
            "trace_tasks_per_query": 0,
            "seeds": (1, 2),
            "data_master_seed": 2,
            "split": "test",
            "budgets": (0, 1, 2, 3, 4, 8),
        }
        values.update(overrides)
        return Namespace(**values)

    def test_budget_validation(self) -> None:
        self.assertEqual(
            _validate_args(self._args(budgets=(8, 0, 2, 1))),
            ((0, 1, 2, 8), (1, 2)),
        )
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 1, 1)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 9)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(seeds=(1, 1)))


class NuisanceTaskConstructionTests(unittest.TestCase):
    def test_public_task_is_program_independent_and_menu_is_balanced(self) -> None:
        from prp_wm.rulegrid import Axis

        tasks = _build_nuisance_tasks(
            Axis.COLLISION,
            groups=1,
            split="nuisance-test",
            master_seed=19,
            candidate_seed=31,
        )
        self.assertEqual(len(tasks), PROGRAMS_PER_GROUP)
        first = tasks[0]
        self.assertEqual(len(first.inference.support), 5)
        self.assertEqual(
            len(first.inference.active_candidates),
            CANDIDATES_PER_TASK,
        )
        for task in tasks[1:]:
            self.assertEqual(task.inference, first.inference)
        self.assertEqual(
            sorted(first.privileged.candidate_kinds),
            sorted(
                (
                    "query-atomic",
                    "query-atomic",
                    "nuisance-axis-0",
                    "nuisance-axis-0",
                    "nuisance-axis-1",
                    "nuisance-axis-1",
                    "neutral",
                    "neutral",
                )
            ),
        )
        self.assertEqual(
            tuple(probe.probe_id for probe in first.inference.active_candidates),
            tuple(f"C{index:02d}" for index in range(CANDIDATES_PER_TASK)),
        )
        from prp_wm.rulegrid import GridAction

        self.assertTrue(
            all(
                isinstance(probe.action, GridAction)
                for probe in first.inference.active_candidates
            )
        )

    def test_exact_symbolic_partitions_match_query_and_nuisance_roles(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.rulegrid import Axis, RuleGridTransition, version_space

        tasks = _build_nuisance_tasks(
            Axis.TRIGGER,
            groups=1,
            split="nuisance-partition-test",
            master_seed=23,
            candidate_seed=37,
        )
        task = tasks[0]
        initial = version_space(
            task.inference.support,
            task.privileged.palette,
        )
        self.assertEqual(len(initial), 48)
        query_index = task.privileged.candidate_kinds.index("query-atomic")
        nuisance_index = task.privileged.candidate_kinds.index(
            "nuisance-axis-0"
        )

        def observed_space(candidate_index: int):
            probe = task.inference.active_candidates[candidate_index]
            history = task.inference.support + (
                RuleGridTransition(
                    probe.state,
                    probe.action,
                    task.privileged.active_targets[candidate_index],
                ),
            )
            return version_space(history, task.privileged.palette)

        query_space = observed_space(query_index)
        nuisance_space = observed_space(nuisance_index)
        self.assertEqual(
            len({rule_program_factor_ids(p)[1] for p in query_space}),
            1,
        )
        self.assertEqual(len(nuisance_space), 12)
        self.assertEqual(
            len({rule_program_factor_ids(p)[1] for p in nuisance_space}),
            3,
        )

    @unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
    def test_symbolic_initial_posterior_is_uniform_over_48_codes(self) -> None:
        from prp_wm.rulegrid import Axis

        tasks = _build_nuisance_tasks(
            Axis.RELATION,
            groups=1,
            split="nuisance-posterior-test",
            master_seed=29,
            candidate_seed=41,
        )
        bank = torch.cartesian_prod(*(torch.arange(4) for _ in range(3)))
        posterior = _symbolic_initial_joint_log_weights(
            torch=torch,
            tasks=tasks,
            factor_bank=bank,
        )
        self.assertEqual(tuple(posterior.shape), (PROGRAMS_PER_GROUP, 64))
        self.assertTrue((torch.isfinite(posterior).sum(dim=-1) == 48).all())
        self.assertTrue(
            torch.allclose(
                posterior.exp().sum(dim=-1),
                torch.ones(PROGRAMS_PER_GROUP),
            )
        )


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class NuisanceSelectorTensorTests(unittest.TestCase):
    def test_query_gain_rejects_globally_informative_nuisance(self) -> None:
        # 48 hypotheses: q has three values and one nuisance axis has four.
        # Candidate 0 reveals q; candidate 1 reveals an irrelevant axis.
        log_weights = torch.full((48,), -torch.log(torch.tensor(48.0)))
        query_values = torch.arange(3).repeat_interleave(16)
        outcomes = torch.tensor(
            [
                [[[value]] for value in query_values.tolist()],
                [[[value]] for value in torch.arange(4).repeat(12).tolist()],
            ],
            dtype=torch.long,
        )
        available = torch.ones(2, dtype=torch.bool)
        query_choice = _select_query_conditioned_candidate(
            torch,
            log_weights,
            query_values,
            outcomes,
            available,
        )
        global_choice = _select_global_information_candidate(
            torch,
            log_weights,
            outcomes,
            available,
        )
        self.assertEqual(query_choice.candidate_index, 0)
        self.assertAlmostEqual(
            query_choice.expected_door_gain,
            2.0 / 3.0,
            places=6,
        )
        self.assertEqual(global_choice.candidate_index, 1)
        self.assertAlmostEqual(
            global_choice.information_gain_nats,
            float(torch.log(torch.tensor(4.0))),
            places=6,
        )

    def test_uniform_orders_are_shared_across_each_four_task_group(self) -> None:
        orders = _uniform_orders(
            torch=torch,
            query_index=2,
            groups=2,
            candidates=CANDIDATES_PER_TASK,
            seed=31,
        )
        self.assertEqual(len(orders), 2 * PROGRAMS_PER_GROUP)
        for start in range(0, len(orders), PROGRAMS_PER_GROUP):
            self.assertEqual(
                len(set(orders[start : start + PROGRAMS_PER_GROUP])),
                1,
            )
            self.assertEqual(
                sorted(orders[start]),
                list(range(CANDIDATES_PER_TASK)),
            )


class NuisanceSelectorBoundaryTests(unittest.TestCase):
    def test_selectors_have_non_leaking_boundaries(self) -> None:
        audit = _selector_boundary_audit()
        self.assertTrue(audit["passed"])
        self.assertEqual(
            tuple(
                inspect.signature(
                    _select_query_conditioned_candidate
                ).parameters
            ),
            (
                "torch",
                "log_weights",
                "query_values",
                "candidate_outcomes",
                "available",
            ),
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    _select_global_information_candidate
                ).parameters
            ),
            (
                "torch",
                "log_weights",
                "candidate_outcomes",
                "available",
            ),
        )


if __name__ == "__main__":
    unittest.main()
