"""Runner-level checks for active-prefix executor delta calibration."""

from __future__ import annotations

from argparse import Namespace
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_active_support_calibrated_executor import (
    STAGE_NAMES,
    _active_executor_gate,
    _active_histories,
    _bank_set_counts,
    _diagnostic_panels,
    _program_conditioned_panel_batch,
    _scheduled_program_rows,
    _validate_args,
)


class ActiveExecutorArgumentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> Namespace:
        values: dict[str, object] = {
            "steps": 1,
            "batch_size": 8,
            "codes_per_task": 8,
            "eval_tasks": 48,
            "eval_batch_size": 8,
            "log_every": 1,
            "learning_rate": 5e-4,
            "tail_steps": 0,
            "tail_learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "weight_decay": 1e-4,
            "balanced_weight": 1.0,
            "diagnostic_loss_weight": 0.4,
            "stage_loss_weights": (0.1, 0.15, 0.1, 0.25),
            "seed": 1,
            "data_master_seed": 2,
            "train_split": "train",
            "eval_split": "eval",
        }
        values.update(overrides)
        return Namespace(**values)

    def test_domain_weights_must_sum_to_one(self) -> None:
        diagnostic, stages = _validate_args(self._args())
        self.assertAlmostEqual(diagnostic + sum(stages), 1.0)
        with self.assertRaises(SystemExit):
            _validate_args(
                self._args(stage_loss_weights=(0.1, 0.1, 0.1, 0.1))
            )

    def test_codes_per_task_cannot_exceed_bank(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_args(self._args(codes_per_task=65))


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class ActiveExecutorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from prp_wm.pilot import make_pilot_tasks

        cls.tasks = make_pilot_tasks(
            split="active-executor-runner-test",
            master_seed=2026071601,
            start=0,
            count=8,
            diagnostic_indices=tuple(range(21)),
        )
        cls.histories = _active_histories(cls.tasks)

    def test_fixed_active_prefix_has_expected_symbolic_spaces(self) -> None:
        from prp_wm.rulegrid import version_space

        self.assertEqual(
            [len(stage[0]) for stage in self.histories],
            [6, 7, 8, 9],
        )
        for task_index, task in enumerate(self.tasks):
            spaces = [
                version_space(stage[task_index], task.privileged.palette)
                for stage in self.histories
            ]
            self.assertEqual(len(spaces[0]), 4)
            self.assertIn(len(spaces[1]), (1, 3))
            self.assertEqual(spaces[2], spaces[1])
            self.assertEqual(spaces[3], (task.privileged.true_program,))

    def test_default_schedule_covers_all_64_codes_each_step(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids

        for step in (0, 1, 17):
            rows = _scheduled_program_rows(
                self.tasks,
                step=step,
                codes_per_task=8,
            )
            codes = {
                rule_program_factor_ids(program)
                for row in rows
                for program in row
            }
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(len(row) == 8 for row in rows))
            self.assertEqual(len(codes), 64)
        first = _scheduled_program_rows(self.tasks, step=0, codes_per_task=8)
        second = _scheduled_program_rows(self.tasks, step=1, codes_per_task=8)
        self.assertNotEqual(first[0], second[0])

    def test_program_conditioned_batch_uses_code_specific_targets(self) -> None:
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.rulegrid import ALL_PROGRAMS

        task = self.tasks[0]
        rows = ((ALL_PROGRAMS[0], ALL_PROGRAMS[-1]),)
        panel = ((self.histories[3][0][-1],),)
        batch = _program_conditioned_panel_batch(
            torch=torch,
            tasks=(task,),
            panels=panel,
            program_rows=rows,
            device="cpu",
        )
        self.assertEqual(tuple(batch.states.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(batch.targets.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(batch.factor_ids.shape), (2, 3))
        self.assertEqual(
            tuple(tuple(int(value) for value in row) for row in batch.factor_ids),
            tuple(rule_program_factor_ids(program) for program in rows[0]),
        )
        self.assertTrue(batch.palette_canonicalized)
        # The strong transition discriminates the held-out axis, so the two
        # endpoint programs cannot both have the same canonical target here.
        self.assertFalse(bool(torch.equal(batch.targets[0], batch.targets[1])))

    def test_diagnostic_panels_select_only_requested_public_probes(self) -> None:
        panels = _diagnostic_panels(self.tasks, (0, 4, 8, 12, 21))
        self.assertEqual(len(panels), len(self.tasks))
        self.assertTrue(all(len(panel) == 5 for panel in panels))
        self.assertEqual(
            tuple(probe.probe_id for probe in panels[0]),
            ("D00", "D04", "D08", "D12", "D21"),
        )

    def test_bank_comparison_recovers_exact_symbolic_set(self) -> None:
        from prp_wm.causal_filter import HypothesisBankScores, enumerate_factor_codes
        from prp_wm.latent_rules import rule_program_factor_ids
        from prp_wm.rulegrid import version_space

        tasks = self.tasks[:3]
        histories = self.histories[1][:3]
        bank = enumerate_factor_codes()
        code_to_index = {
            tuple(int(value) for value in code): index
            for index, code in enumerate(bank.tolist())
        }
        exact = torch.zeros((len(tasks), 64), dtype=torch.bool)
        for task_index, (task, history) in enumerate(
            zip(tasks, histories, strict=True)
        ):
            for program in version_space(history, task.privileged.palette):
                exact[task_index, code_to_index[rule_program_factor_ids(program)]] = True
        zeros = torch.zeros((len(tasks), 64))
        scores = HypothesisBankScores(
            factor_ids=bank,
            proper_nll_per_cell=zeros,
            balanced_nll_per_cell=zeros,
            map_error_cells=zeros.to(dtype=torch.long),
            map_exact=exact,
        )
        counts = _bank_set_counts(
            scores=scores,
            tasks=tasks,
            histories=histories,
        )
        self.assertEqual(counts["set_equal"], len(tasks))
        self.assertEqual(counts["true_exact"], len(tasks))
        self.assertEqual(counts["false_positives"], 0)
        self.assertEqual(counts["false_negatives"], 0)


class ActiveExecutorGateTests(unittest.TestCase):
    @staticmethod
    def _prefix(rate: float = 1.0, true_rate: float = 1.0) -> dict[str, object]:
        return {
            "stages": {
                name: {
                    "neural_map_exact_bank_equals_symbolic_version_space_task_rate": rate,
                    "true_rule_map_exact_task_rate": true_rate,
                }
                for name in STAGE_NAMES
            }
        }

    def test_gate_is_strict_on_every_stage_and_diagnostic_group(self) -> None:
        diagnostics = tuple({"exact_task_accuracy": 1.0} for _ in range(3))
        self.assertTrue(
            _active_executor_gate(self._prefix(), diagnostics)["passed"]
        )
        prefix = self._prefix()
        prefix["stages"]["t2_neutral"][
            "neural_map_exact_bank_equals_symbolic_version_space_task_rate"
        ] = 0.98
        self.assertFalse(_active_executor_gate(prefix, diagnostics)["passed"])
        bad_diagnostics = (
            {"exact_task_accuracy": 1.0},
            {"exact_task_accuracy": 1.0},
            {"exact_task_accuracy": 0.99},
        )
        self.assertFalse(
            _active_executor_gate(self._prefix(), bad_diagnostics)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
