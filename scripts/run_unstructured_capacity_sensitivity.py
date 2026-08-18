#!/usr/bin/env python3
"""Run the preregistered opaque-64 capacity sensitivity suite.

This suite is deliberately separate from the already-frozen primary Latin
factorization experiment.  It inherits the primary suite's runner, executor,
data, optimizer, schedule, and source identities byte-for-byte, then changes
only the opaque 64-way proposal head capacity:

* low-rank rank 9: four folds x seeds 20260727 and 20260728 (8 runs);
* unrestricted direct linear: four folds x seed 20260727 (4 runs).

Each output is stored in an immutable attempt directory and validated against
both its result JSON and the metadata inside its checkpoint.  Pairing against
the primary factorized runs is emitted when the primary summary is complete;
otherwise the sensitivity runs may finish with pairing explicitly pending.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_factorization_latin_suite as primary


SUITE_SCHEMA_VERSION = "prp-wm.unstructured-capacity-sensitivity.v1"
RUN_SPEC_SCHEMA_VERSION = "prp-wm.unstructured-capacity-run-spec.v1"
ATTEMPT_SCHEMA_VERSION = "prp-wm.unstructured-capacity-attempt.v1"
STATUS_SCHEMA_VERSION = "prp-wm.unstructured-capacity-status.v1"
PRIMARY_SUITE_SCHEMA_VERSION = "prp-wm.factorization-latin-suite.v1"
FOLDS = (0, 1, 2, 3)
RANK9_SEEDS = (20260727, 20260728)
DIRECT_SEEDS = (20260727,)
VARIANTS: dict[str, dict[str, Any]] = {
    "low-rank-r9": {
        "head_kind": "low-rank",
        "head_rank": 9,
        "seeds": list(RANK9_SEEDS),
        "expected_trainable_parameters": 35_467,
        "capacity_label": "DOF9",
    },
    "direct-linear": {
        "head_kind": "direct-linear",
        "head_rank": None,
        "seeds": list(DIRECT_SEEDS),
        "expected_trainable_parameters": 36_642,
        "capacity_label": "upper_capacity_direct_linear",
    },
}
FACTORIZED_TRAINABLE_PARAMETERS = 35_054
EXPECTED_MAIN_RUNNER_ARGS: dict[str, Any] = {
    "assignment_temperature": 0.0,
    "attention_layers": 2,
    "balanced_weight": 1.0,
    "batch_size": 8,
    "device": "cpu",
    "diversity_weight": 0.1,
    "eval_batch_size": 16,
    "eval_split": "expected-discrete-causal-composition",
    "eval_tasks": 48,
    "factor_temperature_end": 1.0,
    "factor_temperature_start": 1.0,
    "learning_rate": 0.001,
    "log_every": 100,
    "max_grad_norm": 1.0,
    "nll_threshold": 0.05,
    "proper_weight": 1.0,
    "sharpening_start_fraction": 0.8,
    "sharpening_weight_end": 0.0,
    "steps": 600,
    "tail_learning_rate": 0.0005,
    "tail_steps": 100,
    "train_pool_tasks": 144,
    "train_split": "expected-discrete-causal-train",
    "unstructured_head_kind": "low-rank",
    "unstructured_head_rank": None,
    "validity_weight": 0.1,
    "weight_decay": 0.0001,
}


class SensitivityError(RuntimeError):
    """Raised for an invalid sensitivity or primary-suite identity."""


class SensitivityValidationError(SensitivityError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-suite", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return primary._read_json(path)
    except primary.ResultValidationError as error:
        raise SensitivityValidationError(error.kind, str(error)) from error


def _require_equal(payload: Mapping[str, Any], field: str, expected: Any) -> None:
    if field not in payload or payload[field] != expected:
        raise SensitivityValidationError(
            "identity_mismatch",
            f"{field} mismatch: expected {expected!r}, got {payload.get(field, '<missing>')!r}",
        )


def _verify_file_identity(record: Mapping[str, Any], label: str) -> Path:
    path_value = record.get("path")
    sha_value = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        raise SensitivityError(f"primary suite has invalid {label} identity")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise SensitivityError(f"primary suite {label} is missing: {path}")
    if primary._sha256_file(path) != sha_value:
        raise SensitivityError(f"primary suite {label} SHA256 drifted: {path}")
    return path


def load_and_verify_primary_config(main_suite: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the frozen primary config and verify every runtime identity."""

    main_root = main_suite.resolve()
    config_path = main_root / "suite_config.json"
    if not config_path.is_file():
        raise SensitivityError(f"primary suite config is missing: {config_path}")
    config = _read_json(config_path)
    expected_top_level = {
        "schema_version": PRIMARY_SUITE_SCHEMA_VERSION,
        "suite_id": "factorization_latin_4fold_3seed_v1",
        "models": ["factorized-3x4", "unstructured-64"],
        "context_folds": list(FOLDS),
        "model_seeds": [20260727, 20260728, 20260729],
        "data_master_seed": primary.DATA_MASTER_SEED,
        "planned_runs": 24,
        "planned_pairs": 12,
    }
    for field, expected in expected_top_level.items():
        if config.get(field) != expected:
            raise SensitivityError(
                f"primary suite {field} mismatch: expected {expected!r}, "
                f"got {config.get(field, '<missing>')!r}"
            )
    if config.get("runner_args") != EXPECTED_MAIN_RUNNER_ARGS:
        raise SensitivityError("primary suite runner_args differ from preregistration")
    _verify_file_identity(config.get("runner", {}), "runner")
    _verify_file_identity(config.get("executor_checkpoint", {}), "executor checkpoint")
    _verify_file_identity(config.get("orchestrator", {}), "primary orchestrator")
    python_record = config.get("python")
    if not isinstance(python_record, Mapping) or not isinstance(
        python_record.get("path"), str
    ):
        raise SensitivityError("primary suite Python identity is invalid")
    if not Path(python_record["path"]).is_file():
        raise SensitivityError("primary suite Python interpreter is missing")
    source_manifest = config.get("runner_source_sha256")
    if not isinstance(source_manifest, Mapping):
        raise SensitivityError("primary suite source manifest is invalid")
    for relative, expected_sha in source_manifest.items():
        path = REPOSITORY_ROOT / str(relative)
        if not path.is_file() or primary._sha256_file(path) != expected_sha:
            raise SensitivityError(f"primary runtime source drifted: {relative}")
    hashes = {
        "artifact_sha256": primary._sha256_file(config_path),
        "payload_sha256": primary._payload_sha256(config),
    }
    return config, hashes


def build_suite_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_workers <= 0:
        raise SensitivityError("--max-workers must be positive")
    main_config, main_hashes = load_and_verify_primary_config(args.main_suite)
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": "unstructured_capacity_sensitivity_v1",
        "main_suite": {
            "path": str(args.main_suite.resolve()),
            "suite_config_artifact_sha256": main_hashes["artifact_sha256"],
            "suite_config_payload_sha256": main_hashes["payload_sha256"],
            "primary_orchestrator_sha256": main_config["orchestrator"]["sha256"],
        },
        "inherited_main_config": main_config,
        "max_workers": args.max_workers,
        "folds": list(FOLDS),
        "variants": VARIANTS,
        "planned_runs": 12,
        "planned_pairs": {"low-rank-r9": 8, "direct-linear": 4},
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": primary._sha256_file(Path(__file__).resolve()),
        },
        "metric_paths": {
            name: list(path) for name, path in primary.METRIC_PATHS.items()
        },
        "metric_directions": primary.METRIC_DIRECTIONS,
        "statistics": {
            "std": "sample_ddof_1",
            "paired_raw_delta": "factorized-3x4 minus capacity-control",
            "inference_scope": "descriptive_only",
            "dependence_note": main_config["statistics"]["dependence_note"],
            "no_confidence_intervals_or_hypothesis_tests": True,
        },
        "capacity_table": {
            "factorized_reference_trainable_parameters": FACTORIZED_TRAINABLE_PARAMETERS,
            "low-rank-r9": {
                "head_kind": "low-rank",
                "head_rank": 9,
                "capacity_label": "DOF9",
                "trainable_parameters": 35_467,
                "excess_parameters": 413,
                "relative_excess": 413 / FACTORIZED_TRAINABLE_PARAMETERS,
                "relative_excess_percent_rounded_4dp": 1.1782,
            },
            "direct-linear": {
                "head_kind": "direct-linear",
                "head_rank": None,
                "capacity_label": "upper_capacity_direct_linear",
                "trainable_parameters": 36_642,
                "excess_parameters": 1_588,
                "relative_excess": 1_588 / FACTORIZED_TRAINABLE_PARAMETERS,
                "relative_excess_percent_rounded_4dp": 4.5302,
            },
        },
    }


def build_run_specs(
    suite_config: Mapping[str, Any], suite_config_sha256: str
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # Pair the two seed-20260727 capacity settings adjacently within a fold;
    # rank 9's second seed follows them.
    for fold in FOLDS:
        for variant, seed in (
            ("low-rank-r9", 20260727),
            ("direct-linear", 20260727),
            ("low-rank-r9", 20260728),
        ):
            capacity = VARIANTS[variant]
            run_id = f"{variant}__fold{fold}__seed{seed}"
            specs.append(
                {
                    "schema_version": RUN_SPEC_SCHEMA_VERSION,
                    "suite_config_sha256": suite_config_sha256,
                    "run_id": run_id,
                    "variant": variant,
                    "model": "unstructured-64",
                    "context_fold": fold,
                    "model_seed": seed,
                    "data_master_seed": primary.DATA_MASTER_SEED,
                    "head_kind": capacity["head_kind"],
                    "head_rank": capacity["head_rank"],
                    "expected_trainable_parameters": capacity[
                        "expected_trainable_parameters"
                    ],
                    "capacity_label": capacity["capacity_label"],
                }
            )
    if len(specs) != 12 or len({spec["run_id"] for spec in specs}) != 12:
        raise SensitivityError("capacity matrix must contain 12 unique runs")
    return specs


def _run_root(suite_root: Path, spec: Mapping[str, Any]) -> Path:
    return (
        suite_root
        / "runs"
        / str(spec["variant"])
        / f"fold_{spec['context_fold']}"
        / f"seed_{spec['model_seed']}"
    )


def _base_primary_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": "unstructured-64",
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
    }


def _expected_result_fields(
    suite_config: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    main_config = suite_config["inherited_main_config"]
    expected = primary._result_expected_fields(
        main_config,
        _base_primary_spec(spec),
    )
    expected.update(
        {
            "head_kind": spec["head_kind"],
            "head_rank": spec["head_rank"],
            "requested_unstructured_head_kind": spec["head_kind"],
            "requested_unstructured_head_rank": spec["head_rank"],
        }
    )
    return expected


def _validate_checkpoint(
    *,
    checkpoint_path: Path,
    suite_config: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    try:
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:  # noqa: BLE001 - persisted as audit failure
        raise SensitivityValidationError(
            "artifact_mismatch",
            f"cannot load checkpoint: {type(error).__name__}: {error}",
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise SensitivityValidationError(
            "artifact_mismatch", "checkpoint root must be a mapping"
        )
    _require_equal(
        checkpoint,
        "checkpoint_schema_version",
        primary.EXPECTED_CHECKPOINT_SCHEMA_VERSION,
    )
    _require_equal(checkpoint, "model_type", "UnstructuredDiscreteCausalK4")
    expected = _expected_result_fields(suite_config, spec)
    fields = (
        "model",
        "posterior_parameterization",
        "experiment",
        "context_fold",
        "context_split_kind",
        "model_seed",
        "data_master_seed",
        "executor_checkpoint_sha256",
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "eval_batch_size",
        "learning_rate",
        "tail_steps",
        "tail_learning_rate",
        "tail_start_step",
        "weight_decay",
        "max_grad_norm",
        "validity_weight",
        "diversity_weight",
        "sharpening_weight_end",
        "sharpening_start_fraction",
        "proper_weight",
        "balanced_weight",
        "factor_temperature_start",
        "factor_temperature_end",
        "assignment_temperature",
        "head_kind",
        "head_rank",
        "requested_unstructured_head_kind",
        "requested_unstructured_head_rank",
        "attention_layers",
        "nll_threshold",
        "log_every",
        "train_split",
        "eval_split",
        "device",
        "unique_train_support_contexts",
        "unique_eval_support_contexts",
        "train_eval_contexts_disjoint",
        "initial_checkpoint",
        "initial_checkpoint_sha256",
        "initial_training_steps",
        "cumulative_training_steps",
    )
    for field in fields:
        _require_equal(checkpoint, field, expected[field])
    if checkpoint.get("source_sha256") != suite_config["inherited_main_config"][
        "runner_source_sha256"
    ]:
        raise SensitivityValidationError(
            "source_mismatch", "checkpoint source manifest differs from primary suite"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise SensitivityValidationError(
            "artifact_mismatch", "checkpoint model_state_dict is empty"
        )
    if not isinstance(checkpoint.get("latest_training_metrics"), Mapping):
        raise SensitivityValidationError(
            "artifact_mismatch", "checkpoint latest_training_metrics is missing"
        )


def validate_result(
    *,
    suite_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    output_dir: Path,
    suite_root: Path,
    stdout_path: Path,
    stderr_path: Path,
    run_spec_path: Path,
) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint_last.pt"
    progress_path = output_dir / "progress.jsonl"
    if not result_path.is_file():
        raise SensitivityValidationError("result_missing", f"missing {result_path}")
    result = _read_json(result_path)
    expected = _expected_result_fields(suite_config, spec)
    for field, expected_value in expected.items():
        _require_equal(result, field, expected_value)

    try:
        train_contexts = primary._context_set(
            result.get("train_support_contexts"), "train_support_contexts"
        )
        eval_contexts = primary._context_set(
            result.get("eval_support_contexts"), "eval_support_contexts"
        )
    except primary.ResultValidationError as error:
        raise SensitivityValidationError(error.kind, str(error)) from error
    expected_train, expected_eval = primary._expected_contexts(
        int(spec["context_fold"])
    )
    if train_contexts != expected_train or eval_contexts != expected_eval:
        raise SensitivityValidationError(
            "identity_mismatch", "support contexts do not match the Latin fold"
        )
    main_config = suite_config["inherited_main_config"]
    if result.get("source_sha256") != main_config["runner_source_sha256"]:
        raise SensitivityValidationError(
            "source_mismatch", "result source manifest differs from primary suite"
        )
    if not checkpoint_path.is_file():
        raise SensitivityValidationError(
            "artifact_mismatch", "checkpoint is missing"
        )
    if not isinstance(result.get("checkpoint_path"), str) or Path(
        result["checkpoint_path"]
    ).resolve() != checkpoint_path.resolve():
        raise SensitivityValidationError(
            "artifact_mismatch", "checkpoint_path escapes this attempt"
        )
    checkpoint_sha = primary._sha256_file(checkpoint_path)
    if result.get("checkpoint_sha256") != checkpoint_sha:
        raise SensitivityValidationError(
            "artifact_mismatch", "checkpoint SHA256 does not match bytes"
        )
    _validate_checkpoint(
        checkpoint_path=checkpoint_path,
        suite_config=suite_config,
        spec=spec,
    )
    if not progress_path.is_file() or progress_path.stat().st_size == 0:
        raise SensitivityValidationError(
            "artifact_mismatch", "progress.jsonl is missing or empty"
        )

    metrics: dict[str, float] = {}
    try:
        for name, path in primary.METRIC_PATHS.items():
            metrics[name] = primary._finite_number(
                primary._nested_value(result, path), name
            )
    except primary.ResultValidationError as error:
        raise SensitivityValidationError(error.kind, str(error)) from error
    for name in primary.RATE_METRICS:
        if not 0.0 <= metrics[name] <= 1.0:
            raise SensitivityValidationError(
                "result_invalid", f"{name} must lie in [0,1]"
            )
    if metrics["training_seconds"] < 0:
        raise SensitivityValidationError(
            "result_invalid", "training_seconds must be non-negative"
        )
    for section, tasks, ablation in (
        ("heldout_triple_coverage", main_config["runner_args"]["eval_tasks"], "none"),
        ("seen_context_triple_coverage", 144, "none"),
        (
            "shuffled_support_target_control",
            main_config["runner_args"]["eval_tasks"],
            "shuffle-targets",
        ),
    ):
        payload = result.get(section)
        if not isinstance(payload, Mapping):
            raise SensitivityValidationError("result_invalid", f"missing {section}")
        try:
            primary._validate_evaluation_section(
                payload,
                section=section,
                expected_tasks=tasks,
                expected_support_ablation=ablation,
                nll_threshold=main_config["runner_args"]["nll_threshold"],
            )
        except primary.ResultValidationError as error:
            raise SensitivityValidationError(error.kind, str(error)) from error

    gate_payload = result.get("static_gate")
    if not isinstance(gate_payload, Mapping):
        raise SensitivityValidationError("result_invalid", "static_gate is missing")
    for field in (
        "coverage_at_4_gte",
        "all_classes_covered_task_rate_gte",
        "factor_tuple_coverage_at_4_gte",
        "all_particles_support_exact_task_rate_gte",
    ):
        _require_equal(gate_payload, field, primary.STATIC_GATE_THRESHOLD)
    recomputed_gate = all(
        metrics[name] >= primary.STATIC_GATE_THRESHOLD
        for name in primary.STATIC_GATE_METRICS
    )
    if type(gate_payload.get("passed")) is not bool or gate_payload[
        "passed"
    ] != recomputed_gate:
        raise SensitivityValidationError(
            "result_invalid", "static gate disagrees with recomputed metrics"
        )
    _require_equal(
        result,
        "trainable_parameters",
        spec["expected_trainable_parameters"],
    )
    model_parameters = result.get("model_parameters")
    if type(model_parameters) is not int or model_parameters < result[
        "trainable_parameters"
    ]:
        raise SensitivityValidationError(
            "result_invalid", "model_parameters is invalid"
        )
    artifacts = {
        "run_spec": primary._artifact(run_spec_path, suite_root),
        "stdout": primary._artifact(stdout_path, suite_root),
        "stderr": primary._artifact(stderr_path, suite_root),
        "progress": primary._artifact(progress_path, suite_root),
        "checkpoint": primary._artifact(checkpoint_path, suite_root),
        "result": primary._artifact(result_path, suite_root),
    }
    return {
        "run_id": spec["run_id"],
        "variant": spec["variant"],
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
        "status": "succeeded",
        "static_gate_passed": recomputed_gate,
        "head_kind": spec["head_kind"],
        "head_rank": spec["head_rank"],
        "capacity_label": spec["capacity_label"],
        "model_parameters": model_parameters,
        "trainable_parameters": result["trainable_parameters"],
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _runner_command(
    suite_config: Mapping[str, Any], spec: Mapping[str, Any], output_dir: Path
) -> list[str]:
    main_config = suite_config["inherited_main_config"]
    command = primary._runner_command(
        main_config,
        _base_primary_spec(spec) | {"data_master_seed": primary.DATA_MASTER_SEED},
        output_dir,
    )
    head_flag = command.index("--unstructured-head-kind")
    command[head_flag + 1] = str(spec["head_kind"])
    if spec["head_rank"] is not None:
        command.extend(("--unstructured-head-rank", str(spec["head_rank"])))
    return command


ProcessRunner = Callable[..., int]


def _attempt_directories(run_root: Path) -> list[Path]:
    root = run_root / "attempts"
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def _failure_record(
    spec: Mapping[str, Any],
    *,
    status: str,
    attempt_index: int | None,
    kind: str,
    message: str,
    returncode: int | None,
    artifacts: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    return {
        "run_id": spec["run_id"],
        "variant": spec["variant"],
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
        "status": status,
        "execution_action": action,
        "attempt_index": attempt_index,
        "static_gate_passed": None,
        "head_kind": spec["head_kind"],
        "head_rank": spec["head_rank"],
        "capacity_label": spec["capacity_label"],
        "model_parameters": None,
        "trainable_parameters": None,
        "metrics": None,
        "failure": {"kind": kind, "message": message, "returncode": returncode},
        "artifacts": dict(artifacts),
    }


def _status_payload(
    spec: Mapping[str, Any],
    *,
    state: str,
    attempt_index: int,
    run_spec_sha256: str,
    artifacts: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "state": state,
        "run_id": spec["run_id"],
        "attempt_index": attempt_index,
        "run_spec_sha256": run_spec_sha256,
        "updated_at_utc": primary._utc_now(),
    }
    if artifacts is not None:
        payload["artifacts"] = artifacts
    if failure is not None:
        payload["failure"] = failure
    return payload


def execute_run(
    *,
    suite_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    suite_root: Path,
    process_runner: ProcessRunner = primary._invoke_subprocess,
) -> dict[str, Any]:
    run_root = _run_root(suite_root, spec)
    run_root.mkdir(parents=True, exist_ok=True)
    run_spec_path = run_root / "run_spec.json"
    try:
        primary._write_or_validate_json(run_spec_path, dict(spec))
    except primary.SuiteError as error:
        raise SensitivityError(str(error)) from error
    run_spec_sha = primary._sha256_file(run_spec_path)
    status_path = run_root / "status.json"
    for attempt_dir in reversed(_attempt_directories(run_root)):
        try:
            record = validate_result(
                suite_config=suite_config,
                spec=spec,
                output_dir=attempt_dir / "output",
                suite_root=suite_root,
                stdout_path=attempt_dir / "stdout.log",
                stderr_path=attempt_dir / "stderr.log",
                run_spec_path=run_spec_path,
            )
        except SensitivityValidationError:
            continue
        attempt_index = int(attempt_dir.name)
        record.update(
            {
                "attempt_index": attempt_index,
                "execution_action": "skipped_valid",
                "run_spec_sha256": run_spec_sha,
            }
        )
        record["artifacts"]["attempt"] = primary._artifact(
            attempt_dir / "attempt.json", suite_root
        )
        primary._atomic_json(
            status_path,
            _status_payload(
                spec,
                state="succeeded",
                attempt_index=attempt_index,
                run_spec_sha256=run_spec_sha,
                artifacts=record["artifacts"],
            ),
        )
        return record

    attempts = _attempt_directories(run_root)
    attempt_index = int(attempts[-1].name) + 1 if attempts else 1
    attempt_dir = run_root / "attempts" / f"{attempt_index:03d}"
    output_dir = attempt_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    attempt_path = attempt_dir / "attempt.json"
    command = _runner_command(suite_config, spec, output_dir)
    running = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "state": "running",
        "run_id": spec["run_id"],
        "attempt_index": attempt_index,
        "run_spec_sha256": run_spec_sha,
        "command": command,
        "started_at_utc": primary._utc_now(),
    }
    primary._atomic_json(attempt_path, running)
    primary._atomic_json(
        status_path,
        _status_payload(
            spec,
            state="running",
            attempt_index=attempt_index,
            run_spec_sha256=run_spec_sha,
        ),
    )
    environment = dict(os.environ)
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment["PYTHONHASHSEED"] = str(spec["model_seed"])
    started = time.monotonic()
    try:
        returncode = process_runner(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    except Exception as error:  # noqa: BLE001 - persisted failure
        returncode = None
        kind = "spawn_failed"
        message = f"{type(error).__name__}: {error}"
    else:
        kind = "process_failed"
        message = f"runner exited with status {returncode}"
    duration = time.monotonic() - started
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    if returncode != 0:
        final_attempt = running | {
            "state": "failed",
            "finished_at_utc": primary._utc_now(),
            "duration_seconds": duration,
            "returncode": returncode,
            "failure": {"kind": kind, "message": message},
        }
        primary._atomic_json(attempt_path, final_attempt)
        artifacts = {
            "run_spec": primary._artifact(run_spec_path, suite_root),
            "attempt": primary._artifact(attempt_path, suite_root),
            "stdout": primary._artifact(stdout_path, suite_root),
            "stderr": primary._artifact(stderr_path, suite_root),
        }
        primary._atomic_json(
            status_path,
            _status_payload(
                spec,
                state="failed",
                attempt_index=attempt_index,
                run_spec_sha256=run_spec_sha,
                artifacts=artifacts,
                failure=final_attempt["failure"],
            ),
        )
        record = _failure_record(
            spec,
            status="failed",
            attempt_index=attempt_index,
            kind=kind,
            message=message,
            returncode=returncode,
            artifacts=artifacts,
            action="executed",
        )
        record["run_spec_sha256"] = run_spec_sha
        return record

    try:
        record = validate_result(
            suite_config=suite_config,
            spec=spec,
            output_dir=output_dir,
            suite_root=suite_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            run_spec_path=run_spec_path,
        )
    except SensitivityValidationError as error:
        final_attempt = running | {
            "state": "invalid",
            "finished_at_utc": primary._utc_now(),
            "duration_seconds": duration,
            "returncode": 0,
            "failure": {"kind": error.kind, "message": str(error)},
        }
        primary._atomic_json(attempt_path, final_attempt)
        artifacts = {
            "run_spec": primary._artifact(run_spec_path, suite_root),
            "attempt": primary._artifact(attempt_path, suite_root),
            "stdout": primary._artifact(stdout_path, suite_root),
            "stderr": primary._artifact(stderr_path, suite_root),
            "progress": primary._artifact(output_dir / "progress.jsonl", suite_root),
            "checkpoint": primary._artifact(
                output_dir / "checkpoint_last.pt", suite_root
            ),
            "result": primary._artifact(output_dir / "result.json", suite_root),
        }
        primary._atomic_json(
            status_path,
            _status_payload(
                spec,
                state="invalid",
                attempt_index=attempt_index,
                run_spec_sha256=run_spec_sha,
                artifacts=artifacts,
                failure=final_attempt["failure"],
            ),
        )
        failure_record = _failure_record(
            spec,
            status="invalid",
            attempt_index=attempt_index,
            kind=error.kind,
            message=str(error),
            returncode=0,
            artifacts=artifacts,
            action="executed",
        )
        failure_record["run_spec_sha256"] = run_spec_sha
        return failure_record

    final_attempt = running | {
        "state": "succeeded",
        "finished_at_utc": primary._utc_now(),
        "duration_seconds": duration,
        "returncode": 0,
    }
    primary._atomic_json(attempt_path, final_attempt)
    record.update(
        {
            "attempt_index": attempt_index,
            "execution_action": "executed",
            "run_spec_sha256": run_spec_sha,
        }
    )
    record["artifacts"]["attempt"] = primary._artifact(attempt_path, suite_root)
    primary._atomic_json(
        status_path,
        _status_payload(
            spec,
            state="succeeded",
            attempt_index=attempt_index,
            run_spec_sha256=run_spec_sha,
            artifacts=record["artifacts"],
        ),
    )
    return record


def _group_summary(
    specs: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = [records_by_id[str(spec["run_id"])] for spec in specs]
    valid = [record for record in records if record["status"] == "succeeded"]
    passed = sum(record["static_gate_passed"] is True for record in valid)
    return {
        "planned_runs": len(specs),
        "valid_runs": len(valid),
        "metrics": {
            name: primary.summary_statistics(
                [record["metrics"][name] for record in valid]
            )
            for name in primary.METRIC_PATHS
        },
        "static_gate": {
            "passed_count": passed,
            "planned_count": len(specs),
            "pass_rate_planned": passed / len(specs) if specs else None,
            "evaluated_count": len(valid),
            "pass_rate_evaluated": passed / len(valid) if valid else None,
        },
    }


def _verify_artifact_from_record(
    root: Path, artifact: Any, label: str
) -> Path:
    if not isinstance(artifact, Mapping):
        raise SensitivityError(f"primary summary lacks {label} artifact")
    path_value = artifact.get("path")
    expected_sha = artifact.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        raise SensitivityError(f"primary {label} artifact identity is invalid")
    path = (root / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value).resolve()
    if not path.is_file() or primary._sha256_file(path) != expected_sha:
        raise SensitivityError(f"primary {label} artifact is missing or changed: {path}")
    return path


def load_primary_factorized_records(
    suite_config: Mapping[str, Any],
) -> tuple[str, dict[tuple[int, int], dict[str, Any]], str | None]:
    """Return validated primary records, or an explicit pending state."""

    main_root = Path(suite_config["main_suite"]["path"])
    summary_path = main_root / "summary.json"
    if not summary_path.is_file():
        return "pending_main_suite", {}, "primary summary.json is not present"
    summary_sha_path = main_root / "summary.sha256"
    if not summary_sha_path.is_file():
        raise SensitivityError("primary summary.sha256 is missing")
    expected_sha = summary_sha_path.read_text(encoding="utf-8").split()[0]
    if primary._sha256_file(summary_path) != expected_sha:
        raise SensitivityError("primary summary SHA256 does not match summary.sha256")
    summary = _read_json(summary_path)
    if (
        summary.get("complete") is not True
        or summary.get("integrity_gate", {}).get("passed") is not True
    ):
        return "pending_main_suite", {}, "primary suite is not complete and valid"
    main_config = suite_config["inherited_main_config"]
    main_config_sha = suite_config["main_suite"]["suite_config_payload_sha256"]
    if summary.get("suite_config_sha256") != main_config_sha:
        raise SensitivityError("primary summary references a different suite config")
    main_specs = primary.build_run_specs(main_config, main_config_sha)
    specs_by_id = {spec["run_id"]: spec for spec in main_specs}
    summary_records = {
        record["run_id"]: record
        for record in summary.get("runs", [])
        if isinstance(record, Mapping) and isinstance(record.get("run_id"), str)
    }
    validated: dict[tuple[int, int], dict[str, Any]] = {}
    for fold in FOLDS:
        for seed in RANK9_SEEDS:
            run_id = f"factorized-3x4__fold{fold}__seed{seed}"
            if run_id not in summary_records or run_id not in specs_by_id:
                raise SensitivityError(f"primary summary lacks {run_id}")
            record = summary_records[run_id]
            if record.get("status") != "succeeded":
                raise SensitivityError(f"primary factorized run is not valid: {run_id}")
            artifacts = record.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise SensitivityError(f"primary run lacks artifacts: {run_id}")
            result_path = _verify_artifact_from_record(
                main_root, artifacts.get("result"), f"{run_id} result"
            )
            run_spec_path = _verify_artifact_from_record(
                main_root, artifacts.get("run_spec"), f"{run_id} run_spec"
            )
            stdout_path = _verify_artifact_from_record(
                main_root, artifacts.get("stdout"), f"{run_id} stdout"
            )
            stderr_path = _verify_artifact_from_record(
                main_root, artifacts.get("stderr"), f"{run_id} stderr"
            )
            validated_record = primary.validate_result(
                suite_config=main_config,
                spec=specs_by_id[run_id],
                output_dir=result_path.parent,
                suite_root=main_root,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                run_spec_path=run_spec_path,
            )
            if validated_record["trainable_parameters"] != FACTORIZED_TRAINABLE_PARAMETERS:
                raise SensitivityError(
                    f"primary factorized parameter count changed: {run_id}"
                )
            validated[(fold, seed)] = validated_record
    return "ready", validated, None


def _paired_variant_summary(
    *,
    variant: str,
    specs: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    primary_state: str,
    primary_records: Mapping[tuple[int, int], Mapping[str, Any]],
    primary_reason: str | None,
) -> dict[str, Any]:
    deltas: dict[str, list[float]] = {name: [] for name in primary.METRIC_PATHS}
    details: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    gate = {
        "both_pass": 0,
        "factorized_only": 0,
        "capacity_only": 0,
        "neither": 0,
    }
    for spec in specs:
        control = records_by_id[str(spec["run_id"])]
        key = (int(spec["context_fold"]), int(spec["model_seed"]))
        reference = primary_records.get(key)
        if control["status"] != "succeeded" or reference is None:
            missing.append(
                {
                    "context_fold": key[0],
                    "model_seed": key[1],
                    "capacity_status": control["status"],
                    "factorized_status": (
                        reference["status"] if reference is not None else "pending"
                    ),
                }
            )
            continue
        pair_delta = {
            name: reference["metrics"][name] - control["metrics"][name]
            for name in primary.METRIC_PATHS
        }
        for name, value in pair_delta.items():
            deltas[name].append(value)
        factorized_pass = bool(reference["static_gate_passed"])
        capacity_pass = bool(control["static_gate_passed"])
        if factorized_pass and capacity_pass:
            gate["both_pass"] += 1
        elif factorized_pass:
            gate["factorized_only"] += 1
        elif capacity_pass:
            gate["capacity_only"] += 1
        else:
            gate["neither"] += 1
        details.append(
            {
                "context_fold": key[0],
                "model_seed": key[1],
                "factorized_run_id": f"factorized-3x4__fold{key[0]}__seed{key[1]}",
                "capacity_run_id": spec["run_id"],
                "raw_delta_factorized_minus_capacity": pair_delta,
            }
        )
    expected_pairs = len(specs)
    if len(details) == expected_pairs:
        state = "complete"
    elif primary_state != "ready":
        state = "pending_main_suite"
    else:
        state = "partial"
    return {
        "variant": variant,
        "status": state,
        "pending_reason": primary_reason if state == "pending_main_suite" else None,
        "raw_delta": "factorized-3x4 minus capacity-control",
        "metric_directions": primary.METRIC_DIRECTIONS,
        "inference_scope": "descriptive_only",
        "planned_pairs": expected_pairs,
        "complete_pairs": len(details),
        "missing_pairs": missing,
        "details": details,
        "metrics": {
            name: primary._paired_delta_summary(
                values,
                direction=primary.METRIC_DIRECTIONS[name],
            )
            for name, values in deltas.items()
        },
        "static_gate_contingency": gate,
    }


def build_summary(
    *,
    suite_config: Mapping[str, Any],
    suite_config_sha256: str,
    specs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    primary_state: str,
    primary_records: Mapping[tuple[int, int], Mapping[str, Any]],
    primary_reason: str | None,
) -> dict[str, Any]:
    records_by_id = {str(record["run_id"]): record for record in records}
    if set(records_by_id) != {str(spec["run_id"]) for spec in specs}:
        raise SensitivityError("summary requires one record per planned run")
    ordered = [records_by_id[str(spec["run_id"])] for spec in specs]
    valid = [record for record in ordered if record["status"] == "succeeded"]
    aggregates: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_specs = [spec for spec in specs if spec["variant"] == variant]
        aggregates[variant] = {
            "overall": _group_summary(variant_specs, records_by_id),
            "by_fold": {
                str(fold): _group_summary(
                    [spec for spec in variant_specs if spec["context_fold"] == fold],
                    records_by_id,
                )
                for fold in FOLDS
            },
            "by_seed": {
                str(seed): _group_summary(
                    [spec for spec in variant_specs if spec["model_seed"] == seed],
                    records_by_id,
                )
                for seed in VARIANTS[variant]["seeds"]
            },
        }
        paired[variant] = _paired_variant_summary(
            variant=variant,
            specs=variant_specs,
            records_by_id=records_by_id,
            primary_state=primary_state,
            primary_records=primary_records,
            primary_reason=primary_reason,
        )
    capacity_valid = all(
        record["status"] != "succeeded"
        or record["trainable_parameters"]
        == VARIANTS[record["variant"]]["expected_trainable_parameters"]
        for record in ordered
    )
    complete = len(valid) == len(specs)
    action_counts: dict[str, int] = {}
    for record in ordered:
        action = str(record.get("execution_action", "unknown"))
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_config_sha256": suite_config_sha256,
        "main_suite": suite_config["main_suite"],
        "complete": complete,
        "planned_runs": len(specs),
        "valid_runs": len(valid),
        "failed_runs": sum(record["status"] == "failed" for record in ordered),
        "invalid_runs": sum(record["status"] == "invalid" for record in ordered),
        "resume_actions": action_counts,
        "capacity_table": suite_config["capacity_table"],
        "runs": ordered,
        "aggregates": {
            "std_definition": "sample_ddof_1",
            "inference_scope": "descriptive_only",
            "dependence_note": suite_config["statistics"]["dependence_note"],
            "by_variant": aggregates,
        },
        "paired_comparison_to_primary_factorized": {
            "primary_state": primary_state,
            "primary_reason": primary_reason,
            "by_variant": paired,
        },
        "integrity_gate": {
            "all_12_capacity_runs_valid": complete,
            "capacity_parameter_counts_exact": capacity_valid and complete,
            "runner_executor_source_identity_inherited_from_primary": complete,
            "paired_results_ready": all(
                payload["status"] == "complete" for payload in paired.values()
            ),
            "passed": complete and capacity_valid,
            "note": (
                "The integrity gate may pass while primary pairing is pending; "
                "performance statistics are descriptive only."
            ),
        },
        "artifact_manifest": {
            record["run_id"]: record["artifacts"] for record in ordered
        },
    }


def _run_matrix(
    *,
    suite_config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    suite_root: Path,
    process_runner: ProcessRunner = primary._invoke_subprocess,
) -> tuple[list[dict[str, Any]], bool]:
    records: dict[str, dict[str, Any]] = {}
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=int(suite_config["max_workers"]))
    futures: dict[Future[dict[str, Any]], Mapping[str, Any]] = {
        executor.submit(
            execute_run,
            suite_config=suite_config,
            spec=spec,
            suite_root=suite_root,
            process_runner=process_runner,
        ): spec
        for spec in specs
    }
    try:
        for future in as_completed(futures):
            spec = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - keep remaining runs alive
                record = _failure_record(
                    spec,
                    status="failed",
                    attempt_index=None,
                    kind="orchestrator_error",
                    message=f"{type(error).__name__}: {error}",
                    returncode=None,
                    artifacts={},
                    action="orchestrator_error",
                )
            records[str(spec["run_id"])] = record
            print(
                json.dumps(
                    {
                        "run_id": record["run_id"],
                        "status": record["status"],
                        "action": record["execution_action"],
                        "completed": len(records),
                        "planned": len(specs),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        primary._terminate_active_processes()
        for future in futures:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    for spec in specs:
        run_id = str(spec["run_id"])
        if run_id not in records:
            records[run_id] = _failure_record(
                spec,
                status="interrupted" if interrupted else "failed",
                attempt_index=None,
                kind="interrupted" if interrupted else "orchestrator_error",
                message="suite stopped before this run completed",
                returncode=None,
                artifacts={},
                action="not_completed",
            )
    return [records[str(spec["run_id"])] for spec in specs], interrupted


def run_suite(
    args: argparse.Namespace,
    *,
    process_runner: ProcessRunner = primary._invoke_subprocess,
) -> tuple[dict[str, Any], bool]:
    suite_root = args.output.resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    try:
        lock = primary._suite_lock(suite_root)
        with lock:
            suite_config = build_suite_config(args)
            config_path = suite_root / "suite_config.json"
            primary._write_or_validate_json(config_path, suite_config)
            config_sha = primary._payload_sha256(suite_config)
            specs = build_run_specs(suite_config, config_sha)
            records, interrupted = _run_matrix(
                suite_config=suite_config,
                specs=specs,
                suite_root=suite_root,
                process_runner=process_runner,
            )
            primary_state, primary_records, primary_reason = (
                load_primary_factorized_records(suite_config)
            )
            summary = build_summary(
                suite_config=suite_config,
                suite_config_sha256=config_sha,
                specs=specs,
                records=records,
                primary_state=primary_state,
                primary_records=primary_records,
                primary_reason=primary_reason,
            )
            summary_path = suite_root / "summary.json"
            primary._atomic_json(summary_path, summary)
            summary_sha = primary._sha256_file(summary_path)
            temporary = suite_root / "summary.sha256.tmp"
            temporary.write_text(
                f"{summary_sha}  summary.json\n", encoding="utf-8"
            )
            temporary.replace(suite_root / "summary.sha256")
            print(
                json.dumps(
                    {
                        "summary_path": str(summary_path),
                        "summary_sha256": summary_sha,
                        "complete": summary["complete"],
                        "primary_pairing_state": primary_state,
                        "integrity_gate_passed": summary["integrity_gate"]["passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return summary, interrupted
    except primary.SuiteError as error:
        raise SensitivityError(str(error)) from error


def main() -> int:
    args = parse_args()
    try:
        summary, interrupted = run_suite(args)
    except SensitivityError as error:
        print(f"sensitivity suite error: {error}", file=sys.stderr)
        return 2
    if interrupted:
        return 130
    return 0 if summary["integrity_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
