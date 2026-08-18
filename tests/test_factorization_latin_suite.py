"""Focused tests for the resumable Latin factorization experiment suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.run_factorization_latin_suite import (
    DATA_MASTER_SEED,
    EXPECTED_CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_MODEL_TYPE,
    METRIC_PATHS,
    MODELS,
    SEEDS,
    STATUS_SCHEMA_VERSION,
    _atomic_json,
    _expected_contexts,
    _result_expected_fields,
    _sha256_file,
    build_run_specs,
    build_suite_config,
    build_summary,
    execute_run,
    summary_statistics,
)

import torch


ROOT = Path(__file__).resolve().parents[1]


def _arguments(root: Path, executor: Path, *, max_workers: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        output=root,
        executor_checkpoint=executor,
        runner=ROOT / "scripts/run_expected_discrete_causal_coverage.py",
        python=Path(sys.executable),
        max_workers=max_workers,
        steps=600,
        tail_steps=100,
        learning_rate=1e-3,
        tail_learning_rate=5e-4,
        batch_size=8,
        train_pool_tasks=144,
        eval_tasks=48,
        eval_batch_size=16,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        validity_weight=0.10,
        diversity_weight=0.10,
        sharpening_weight_end=0.0,
        sharpening_start_fraction=0.80,
        proper_weight=1.0,
        balanced_weight=1.0,
        factor_temperature_start=1.0,
        factor_temperature_end=1.0,
        assignment_temperature=0.0,
        attention_layers=2,
        nll_threshold=0.05,
        log_every=100,
        device="cpu",
        train_split="expected-discrete-causal-train",
        eval_split="expected-discrete-causal-composition",
    )


def _evaluation(
    tasks: int,
    value: float,
    *,
    support_ablation: str,
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
    stdout_path.write_text("fake runner stdout\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    progress = output / "progress.jsonl"
    progress.write_text('{"step":600}\n', encoding="utf-8")
    expected = _result_expected_fields(suite_config, spec)
    checkpoint = output / "checkpoint_last.pt"
    checkpoint_payload = {
        **expected,
        "checkpoint_schema_version": EXPECTED_CHECKPOINT_SCHEMA_VERSION,
        "model_type": EXPECTED_MODEL_TYPE[spec["model"]],
        "source_sha256": suite_config["runner_source_sha256"],
        "model_state_dict": {"fixture": torch.tensor([1.0])},
        "latest_training_metrics": {"loss_total": 1.0},
    }
    torch.save(checkpoint_payload, checkpoint)
    train_contexts, eval_contexts = _expected_contexts(int(spec["context_fold"]))
    result: dict[str, object] = {
        **expected,
        "source_sha256": suite_config["runner_source_sha256"],
        "train_support_contexts": [list(value) for value in sorted(train_contexts)],
        "eval_support_contexts": [list(value) for value in sorted(eval_contexts)],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_parameters": 52_000,
        "trainable_parameters": (
            35_054 if spec["model"] == "factorized-3x4" else 35_079
        ),
        "training_seconds": 12.5,
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
    _atomic_json(output / "result.json", result)


def _success_record(spec: dict[str, object], metric: float) -> dict[str, object]:
    gate = metric >= 0.90
    metrics = {name: metric for name in METRIC_PATHS}
    metrics["training_seconds"] = 10.0
    return {
        "run_id": spec["run_id"],
        "model": spec["model"],
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
        "status": "succeeded",
        "execution_action": "executed",
        "attempt_index": 1,
        "static_gate_passed": gate,
        "model_parameters": 52_000,
        "trainable_parameters": (
            35_054 if spec["model"] == "factorized-3x4" else 35_079
        ),
        "head_kind": None if spec["model"] == "factorized-3x4" else "low-rank",
        "head_rank": None if spec["model"] == "factorized-3x4" else 5,
        "training_seconds": 10.0,
        "metrics": metrics,
        "artifacts": {},
    }


class FactorizationLatinSuiteTests(unittest.TestCase):
    def _config(self, root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        executor = root / "executor.pt"
        executor.write_bytes(b"audited executor fixture")
        config = build_suite_config(_arguments(root, executor))
        specs = build_run_specs(config, "a" * 64)
        return config, specs

    def test_matrix_contains_exactly_twelve_adjacent_model_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, specs = self._config(Path(directory))
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({spec["run_id"] for spec in specs}), 24)
        self.assertEqual({spec["model_seed"] for spec in specs}, set(SEEDS))
        for index in range(0, len(specs), 2):
            first, second = specs[index : index + 2]
            self.assertEqual((first["model"], second["model"]), MODELS)
            self.assertEqual(first["context_fold"], second["context_fold"])
            self.assertEqual(first["model_seed"], second["model_seed"])
            self.assertEqual(first["data_master_seed"], DATA_MASTER_SEED)

    def test_valid_attempt_is_skipped_and_corruption_creates_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)
            calls = 0

            def fake_runner(command, *, stdout_path, stderr_path, environment):
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

            first = execute_run(
                suite_config=config,
                spec=specs[0],
                suite_root=root,
                process_runner=fake_runner,
            )
            second = execute_run(
                suite_config=config,
                spec=specs[0],
                suite_root=root,
                process_runner=fake_runner,
            )
            self.assertEqual(calls, 1)
            self.assertEqual(first["execution_action"], "executed")
            self.assertEqual(second["execution_action"], "skipped_valid")
            self.assertIsNotNone(second["artifacts"]["attempt"])
            checkpoint_artifact = second["artifacts"]["checkpoint"]
            checkpoint = root / checkpoint_artifact["path"]
            checkpoint.write_bytes(b"corrupted")
            third = execute_run(
                suite_config=config,
                spec=specs[0],
                suite_root=root,
                process_runner=fake_runner,
            )
            self.assertEqual(calls, 2)
            self.assertEqual(third["attempt_index"], 2)
            self.assertEqual(third["status"], "succeeded")

    def test_nonzero_exit_is_retained_and_next_invocation_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)
            calls = 0

            def flaky_runner(command, *, stdout_path, stderr_path, environment):
                nonlocal calls
                del environment
                calls += 1
                stdout_path.write_text("partial\n", encoding="utf-8")
                stderr_path.write_text("boom\n", encoding="utf-8")
                if calls == 1:
                    return 7
                _write_valid_output(
                    list(command),
                    suite_config=config,
                    spec=specs[1],
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                return 0

            failed = execute_run(
                suite_config=config,
                spec=specs[1],
                suite_root=root,
                process_runner=flaky_runner,
            )
            status_path = root / "runs/unstructured-64/fold_0/seed_20260727/status.json"
            failed_status = json.loads(status_path.read_text(encoding="utf-8"))
            recovered = execute_run(
                suite_config=config,
                spec=specs[1],
                suite_root=root,
                process_runner=flaky_runner,
            )
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["failure"]["returncode"], 7)
            self.assertEqual(failed_status["schema_version"], STATUS_SCHEMA_VERSION)
            self.assertEqual(recovered["status"], "succeeded")
            self.assertEqual(recovered["attempt_index"], 2)
            attempts = sorted(
                (root / "runs/unstructured-64/fold_0/seed_20260727/attempts").iterdir()
            )
            self.assertEqual([path.name for path in attempts], ["001", "002"])

    def test_out_of_range_rate_and_wrong_control_identity_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)

            def bad_runner(command, *, stdout_path, stderr_path, environment):
                del environment
                _write_valid_output(
                    list(command),
                    suite_config=config,
                    spec=specs[2],
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                output = Path(command[command.index("--output") + 1])
                result_path = output / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["heldout_triple_coverage"]["map_exact_class_recall"] = 1.01
                result["shuffled_support_target_control"]["support_ablation"] = "none"
                _atomic_json(result_path, result)
                return 0

            record = execute_run(
                suite_config=config,
                spec=specs[2],
                suite_root=root,
                process_runner=bad_runner,
            )
            self.assertEqual(record["status"], "invalid")
            self.assertEqual(record["failure"]["kind"], "result_invalid")

    def test_checkpoint_internal_identity_is_verified_after_file_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)

            def bad_checkpoint_runner(
                command, *, stdout_path, stderr_path, environment
            ):
                del environment
                _write_valid_output(
                    list(command),
                    suite_config=config,
                    spec=specs[3],
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                output = Path(command[command.index("--output") + 1])
                checkpoint_path = output / "checkpoint_last.pt"
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                checkpoint["model_seed"] = int(specs[3]["model_seed"]) + 1
                torch.save(checkpoint, checkpoint_path)
                result_path = output / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                # Refresh the outer digest so only the checkpoint's internal
                # identity check can reject this artifact.
                result["checkpoint_sha256"] = _sha256_file(checkpoint_path)
                _atomic_json(result_path, result)
                return 0

            record = execute_run(
                suite_config=config,
                spec=specs[3],
                suite_root=root,
                process_runner=bad_checkpoint_runner,
            )
            self.assertEqual(record["status"], "invalid")
            self.assertEqual(record["failure"]["kind"], "identity_mismatch")

    def test_summary_uses_sample_std_and_strict_planned_gate_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, specs = self._config(root)
        records = []
        for spec in specs:
            metric = 1.0 if spec["model"] == "factorized-3x4" else 0.5
            record = _success_record(spec, metric)
            if spec["model"] == "factorized-3x4":
                record["metrics"]["shuffled.coverage_at_4_mass_weighted"] = 0.1
                record["metrics"]["training_seconds"] = 12.0
            else:
                record["metrics"]["shuffled.coverage_at_4_mass_weighted"] = 0.2
                record["metrics"]["training_seconds"] = 10.0
            records.append(record)
        # One missing factorized run must reduce the planned pass rate to 11/12
        # and remove exactly one paired observation.
        records[0] = {
            **records[0],
            "status": "invalid",
            "static_gate_passed": None,
            "metrics": None,
            "failure": {"kind": "fixture", "message": "missing", "returncode": None},
        }
        summary = build_summary(
            suite_config=config,
            suite_config_sha256="a" * 64,
            specs=specs,
            records=records,
        )
        factorized_gate = summary["aggregates"]["by_model"]["factorized-3x4"][
            "static_gate"
        ]
        self.assertEqual(factorized_gate["passed_count"], 11)
        self.assertEqual(factorized_gate["planned_count"], 12)
        self.assertAlmostEqual(factorized_gate["pass_rate_planned"], 11 / 12)
        self.assertEqual(summary["complete_pairs"], 11)
        delta = summary["paired_comparison"]["metrics"][
            "heldout.coverage_at_4_mass_weighted"
        ]
        self.assertEqual(delta["n"], 11)
        self.assertAlmostEqual(delta["mean"], 0.5)
        self.assertAlmostEqual(delta["std"], 0.0)
        self.assertEqual(delta["wins"], 11)
        shuffled_delta = summary["paired_comparison"]["metrics"][
            "shuffled.coverage_at_4_mass_weighted"
        ]
        self.assertEqual(shuffled_delta["direction"], "lower")
        self.assertAlmostEqual(shuffled_delta["mean"], -0.1)
        self.assertEqual(shuffled_delta["wins"], 11)
        training_delta = summary["paired_comparison"]["metrics"][
            "training_seconds"
        ]
        self.assertEqual(training_delta["direction"], "lower")
        self.assertEqual(training_delta["losses"], 11)
        self.assertEqual(summary["aggregates"]["inference_scope"], "descriptive_only")
        self.assertFalse(summary["integrity_gate"]["passed"])
        encoded = json.dumps(summary, allow_nan=False)
        self.assertNotIn("NaN", encoded)

    def test_statistics_are_sample_based_and_singletons_use_null_std(self) -> None:
        stats = summary_statistics([1.0, 2.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["std"], 1.0)
        singleton = summary_statistics([3.0])
        self.assertIsNone(singleton["std"])
        empty = summary_statistics([])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["mean"])
        self.assertTrue(math.isfinite(stats["max"]))


if __name__ == "__main__":
    unittest.main()
