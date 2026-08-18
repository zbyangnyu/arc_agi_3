"""Focused tests for the independent opaque-head capacity sensitivity suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import run_factorization_latin_suite as primary
from scripts import run_unstructured_capacity_sensitivity as sensitivity


ROOT = Path(__file__).resolve().parents[1]
MAIN_SUITE = ROOT / "runs/factorization_latin_4fold_3seed_v1"


def _evaluation(
    tasks: int, value: float, *, support_ablation: str
) -> dict[str, object]:
    return {
        "tasks": tasks,
        "support_ablation": support_ablation,
        "behavior_classes": tasks * 4,
        "covered_behavior_classes": min(tasks * 4, int(round(tasks * 4 * value))),
        "coverage_nll_threshold_per_cell": 0.05,
        "coverage_at_4_mass_weighted": value,
        "all_classes_covered_task_rate": value,
        "factor_tuple_coverage_at_4": value,
        "all_particles_support_exact_task_rate": value,
        "map_exact_class_recall": value,
        "nll_threshold_class_recall": value,
        "support_exact_particle_rate": value,
        "valid_particle_rate": value,
        "mean_unique_factor_tuples": 4.0,
        "mean_unique_map_signatures": 4.0,
    }


def _write_valid_output(
    command: list[str],
    *,
    suite_config: dict[str, object],
    spec: dict[str, object],
    stdout_path: Path,
    stderr_path: Path,
    metric: float = 1.0,
) -> None:
    output = Path(command[command.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("capacity fixture\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    (output / "progress.jsonl").write_text('{"step":600}\n', encoding="utf-8")
    expected = sensitivity._expected_result_fields(suite_config, spec)
    checkpoint_path = output / "checkpoint_last.pt"
    checkpoint = {
        **expected,
        "checkpoint_schema_version": primary.EXPECTED_CHECKPOINT_SCHEMA_VERSION,
        "model_type": "UnstructuredDiscreteCausalK4",
        "source_sha256": suite_config["inherited_main_config"][
            "runner_source_sha256"
        ],
        "model_state_dict": {"fixture": torch.tensor([1.0])},
        "latest_training_metrics": {"loss_total": 1.0},
    }
    torch.save(checkpoint, checkpoint_path)
    train_contexts, eval_contexts = primary._expected_contexts(
        int(spec["context_fold"])
    )
    result: dict[str, object] = {
        **expected,
        "source_sha256": suite_config["inherited_main_config"][
            "runner_source_sha256"
        ],
        "train_support_contexts": [list(value) for value in sorted(train_contexts)],
        "eval_support_contexts": [list(value) for value in sorted(eval_contexts)],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": primary._sha256_file(checkpoint_path),
        "model_parameters": int(spec["expected_trainable_parameters"]) + 17_177,
        "trainable_parameters": spec["expected_trainable_parameters"],
        "training_seconds": 10.0,
        "heldout_triple_coverage": _evaluation(
            48, metric, support_ablation="none"
        ),
        "seen_context_triple_coverage": _evaluation(
            144, metric, support_ablation="none"
        ),
        "shuffled_support_target_control": _evaluation(
            48, 0.125, support_ablation="shuffle-targets"
        ),
        "static_gate": {
            "coverage_at_4_gte": 0.90,
            "all_classes_covered_task_rate_gte": 0.90,
            "factor_tuple_coverage_at_4_gte": 0.90,
            "all_particles_support_exact_task_rate_gte": 0.90,
            "passed": metric >= 0.90,
        },
    }
    primary._atomic_json(output / "result.json", result)


def _capacity_record(spec: dict[str, object], metric: float) -> dict[str, object]:
    metrics = {name: metric for name in primary.METRIC_PATHS}
    metrics["shuffled.coverage_at_4_mass_weighted"] = 0.2
    metrics["training_seconds"] = 12.0
    return {
        "run_id": spec["run_id"],
        "variant": spec["variant"],
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
        "status": "succeeded",
        "execution_action": "executed",
        "attempt_index": 1,
        "static_gate_passed": metric >= 0.90,
        "head_kind": spec["head_kind"],
        "head_rank": spec["head_rank"],
        "capacity_label": spec["capacity_label"],
        "model_parameters": int(spec["expected_trainable_parameters"]) + 17_177,
        "trainable_parameters": spec["expected_trainable_parameters"],
        "metrics": metrics,
        "artifacts": {},
    }


def _factorized_record(fold: int, seed: int) -> dict[str, object]:
    metrics = {name: 1.0 for name in primary.METRIC_PATHS}
    metrics["shuffled.coverage_at_4_mass_weighted"] = 0.1
    metrics["training_seconds"] = 10.0
    return {
        "run_id": f"factorized-3x4__fold{fold}__seed{seed}",
        "model": "factorized-3x4",
        "context_fold": fold,
        "model_seed": seed,
        "status": "succeeded",
        "static_gate_passed": True,
        "trainable_parameters": sensitivity.FACTORIZED_TRAINABLE_PARAMETERS,
        "metrics": metrics,
    }


class UnstructuredCapacitySensitivityTests(unittest.TestCase):
    def _config(self, root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        self.assertTrue(MAIN_SUITE.joinpath("suite_config.json").is_file())
        args = argparse.Namespace(
            output=root,
            main_suite=MAIN_SUITE,
            max_workers=2,
        )
        config = sensitivity.build_suite_config(args)
        specs = sensitivity.build_run_specs(config, "b" * 64)
        return config, specs

    def test_matrix_is_exactly_eight_rank9_and_four_direct_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, specs = self._config(Path(directory))
        self.assertEqual(len(specs), 12)
        self.assertEqual(len({spec["run_id"] for spec in specs}), 12)
        rank9 = [spec for spec in specs if spec["variant"] == "low-rank-r9"]
        direct = [spec for spec in specs if spec["variant"] == "direct-linear"]
        self.assertEqual(len(rank9), 8)
        self.assertEqual(len(direct), 4)
        self.assertEqual({spec["model_seed"] for spec in rank9}, {20260727, 20260728})
        self.assertEqual({spec["model_seed"] for spec in direct}, {20260727})
        self.assertEqual({spec["context_fold"] for spec in rank9}, set(range(4)))
        self.assertTrue(all(spec["head_rank"] == 9 for spec in rank9))
        self.assertTrue(all(spec["head_kind"] == "direct-linear" for spec in direct))

    def test_runner_command_changes_only_capacity_head_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)
            rank_command = sensitivity._runner_command(config, specs[0], root / "r")
            direct_command = sensitivity._runner_command(config, specs[1], root / "d")
        rank_kind = rank_command[rank_command.index("--unstructured-head-kind") + 1]
        rank_value = rank_command[rank_command.index("--unstructured-head-rank") + 1]
        direct_kind = direct_command[
            direct_command.index("--unstructured-head-kind") + 1
        ]
        self.assertEqual((rank_kind, rank_value), ("low-rank", "9"))
        self.assertEqual(direct_kind, "direct-linear")
        self.assertNotIn("--unstructured-head-rank", direct_command)

    def test_valid_attempt_is_skipped_and_corrupt_checkpoint_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)
            calls = 0

            def runner(command, *, stdout_path, stderr_path, environment):
                nonlocal calls
                del environment
                calls += 1
                _write_valid_output(
                    list(command),
                    suite_config=config,
                    spec=specs[0],
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                return 0

            first = sensitivity.execute_run(
                suite_config=config,
                spec=specs[0],
                suite_root=root,
                process_runner=runner,
            )
            skipped = sensitivity.execute_run(
                suite_config=config,
                spec=specs[0],
                suite_root=root,
                process_runner=runner,
            )
            self.assertEqual(calls, 1)
            self.assertEqual(skipped["execution_action"], "skipped_valid")
            checkpoint = root / skipped["artifacts"]["checkpoint"]["path"]
            checkpoint.write_bytes(b"corrupt")
            retried = sensitivity.execute_run(
                suite_config=config,
                spec=specs[0],
                suite_root=root,
                process_runner=runner,
            )
            self.assertEqual(first["status"], "succeeded")
            self.assertEqual(calls, 2)
            self.assertEqual(retried["attempt_index"], 2)

    def test_wrong_requested_head_identity_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)

            def runner(command, *, stdout_path, stderr_path, environment):
                del environment
                _write_valid_output(
                    list(command),
                    suite_config=config,
                    spec=specs[1],
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                output = Path(command[command.index("--output") + 1])
                result_path = output / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["requested_unstructured_head_kind"] = "low-rank"
                primary._atomic_json(result_path, result)
                return 0

            record = sensitivity.execute_run(
                suite_config=config,
                spec=specs[1],
                suite_root=root,
                process_runner=runner,
            )
            self.assertEqual(record["status"], "invalid")
            self.assertEqual(record["failure"]["kind"], "identity_mismatch")

    def test_summary_can_complete_capacity_runs_while_primary_pairing_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, specs = self._config(Path(directory))
        records = [_capacity_record(spec, 0.5) for spec in specs]
        summary = sensitivity.build_summary(
            suite_config=config,
            suite_config_sha256="b" * 64,
            specs=specs,
            records=records,
            primary_state="pending_main_suite",
            primary_records={},
            primary_reason="primary still running",
        )
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["integrity_gate"]["passed"])
        self.assertFalse(summary["integrity_gate"]["paired_results_ready"])
        paired = summary["paired_comparison_to_primary_factorized"]["by_variant"]
        self.assertEqual(paired["low-rank-r9"]["status"], "pending_main_suite")
        self.assertEqual(paired["direct-linear"]["complete_pairs"], 0)

    def test_completed_pairing_uses_metric_direction_and_capacity_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, specs = self._config(Path(directory))
        records = [_capacity_record(spec, 0.5) for spec in specs]
        primary_records = {
            (fold, seed): _factorized_record(fold, seed)
            for fold in range(4)
            for seed in (20260727, 20260728)
        }
        summary = sensitivity.build_summary(
            suite_config=config,
            suite_config_sha256="b" * 64,
            specs=specs,
            records=records,
            primary_state="ready",
            primary_records=primary_records,
            primary_reason=None,
        )
        paired = summary["paired_comparison_to_primary_factorized"]["by_variant"]
        rank = paired["low-rank-r9"]
        direct = paired["direct-linear"]
        self.assertEqual(rank["status"], "complete")
        self.assertEqual(rank["complete_pairs"], 8)
        self.assertEqual(direct["complete_pairs"], 4)
        heldout = rank["metrics"]["heldout.coverage_at_4_mass_weighted"]
        shuffled = rank["metrics"]["shuffled.coverage_at_4_mass_weighted"]
        training = rank["metrics"]["training_seconds"]
        self.assertAlmostEqual(heldout["mean"], 0.5)
        self.assertEqual(heldout["wins"], 8)
        self.assertAlmostEqual(shuffled["mean"], -0.1)
        self.assertEqual(shuffled["wins"], 8)
        self.assertAlmostEqual(training["mean"], -2.0)
        self.assertEqual(training["wins"], 8)
        table = summary["capacity_table"]
        self.assertEqual(table["low-rank-r9"]["capacity_label"], "DOF9")
        self.assertEqual(
            table["low-rank-r9"]["relative_excess_percent_rounded_4dp"],
            1.1782,
        )
        self.assertEqual(
            table["direct-linear"]["relative_excess_percent_rounded_4dp"],
            4.5302,
        )
        self.assertEqual(summary["aggregates"]["inference_scope"], "descriptive_only")


if __name__ == "__main__":
    unittest.main()
