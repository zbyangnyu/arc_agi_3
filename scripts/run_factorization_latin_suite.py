#!/usr/bin/env python3
"""Run and audit the 2-model x 4-fold x 3-seed Latin factorization suite.

Every training invocation is isolated in an immutable attempt directory.  A
run is considered complete only after its self-described identity, Latin
split, source manifest, metrics, and checkpoint digest have all been checked.
Consequently an interrupted invocation can be restarted safely: valid runs
are skipped, while failed or incomplete attempts are retained and a new
attempt is created.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SUITE_SCHEMA_VERSION = "prp-wm.factorization-latin-suite.v1"
RUN_SPEC_SCHEMA_VERSION = "prp-wm.factorization-latin-run-spec.v1"
ATTEMPT_SCHEMA_VERSION = "prp-wm.factorization-latin-attempt.v1"
STATUS_SCHEMA_VERSION = "prp-wm.factorization-latin-run-status.v1"
MODELS = ("factorized-3x4", "unstructured-64")
FOLDS = (0, 1, 2, 3)
SEEDS = (20260727, 20260728, 20260729)
DATA_MASTER_SEED = 2026071601
EXPECTED_HEAD_RANK = {
    "factorized-3x4": None,
    "unstructured-64": 5,
}
EXPECTED_HEAD_KIND = {
    "factorized-3x4": None,
    "unstructured-64": "low-rank",
}
EXPECTED_REQUESTED_HEAD_KIND = {
    "factorized-3x4": None,
    "unstructured-64": "low-rank",
}
EXPECTED_POSTERIOR = {
    "factorized-3x4": "independent_3_axes_x_4_values",
    "unstructured-64": "single_categorical_64_rules",
}
EXPECTED_EXPERIMENT = {
    "factorized-3x4": "expected_discrete_axis_structured_causal_k4",
    "unstructured-64": "expected_discrete_unstructured_64way_causal_k4",
}
EXPECTED_MODEL_TYPE = {
    "factorized-3x4": "ExpectedDiscreteCausalK4",
    "unstructured-64": "UnstructuredDiscreteCausalK4",
}
EXPECTED_CHECKPOINT_SCHEMA_VERSION = "prp-wm.expected-discrete-causal-k4.v1"
RUNNER_AUDITED_SOURCE_FILES = (
    "prp_wm/discrete_causal_rules.py",
    "prp_wm/unstructured_causal_rules.py",
    "prp_wm/causal_rules.py",
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


# The keys are stable identifiers used in the aggregate and paired tables.
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "heldout.coverage_at_4_mass_weighted": (
        "heldout_triple_coverage",
        "coverage_at_4_mass_weighted",
    ),
    "heldout.all_classes_covered_task_rate": (
        "heldout_triple_coverage",
        "all_classes_covered_task_rate",
    ),
    "heldout.factor_tuple_coverage_at_4": (
        "heldout_triple_coverage",
        "factor_tuple_coverage_at_4",
    ),
    "heldout.all_particles_support_exact_task_rate": (
        "heldout_triple_coverage",
        "all_particles_support_exact_task_rate",
    ),
    "heldout.map_exact_class_recall": (
        "heldout_triple_coverage",
        "map_exact_class_recall",
    ),
    "heldout.nll_threshold_class_recall": (
        "heldout_triple_coverage",
        "nll_threshold_class_recall",
    ),
    "heldout.support_exact_particle_rate": (
        "heldout_triple_coverage",
        "support_exact_particle_rate",
    ),
    "heldout.valid_particle_rate": (
        "heldout_triple_coverage",
        "valid_particle_rate",
    ),
    "heldout.mean_unique_factor_tuples": (
        "heldout_triple_coverage",
        "mean_unique_factor_tuples",
    ),
    "heldout.mean_unique_map_signatures": (
        "heldout_triple_coverage",
        "mean_unique_map_signatures",
    ),
    "seen.coverage_at_4_mass_weighted": (
        "seen_context_triple_coverage",
        "coverage_at_4_mass_weighted",
    ),
    "seen.all_classes_covered_task_rate": (
        "seen_context_triple_coverage",
        "all_classes_covered_task_rate",
    ),
    "seen.factor_tuple_coverage_at_4": (
        "seen_context_triple_coverage",
        "factor_tuple_coverage_at_4",
    ),
    "seen.all_particles_support_exact_task_rate": (
        "seen_context_triple_coverage",
        "all_particles_support_exact_task_rate",
    ),
    "shuffled.coverage_at_4_mass_weighted": (
        "shuffled_support_target_control",
        "coverage_at_4_mass_weighted",
    ),
    "shuffled.all_classes_covered_task_rate": (
        "shuffled_support_target_control",
        "all_classes_covered_task_rate",
    ),
    "shuffled.factor_tuple_coverage_at_4": (
        "shuffled_support_target_control",
        "factor_tuple_coverage_at_4",
    ),
    "shuffled.all_particles_support_exact_task_rate": (
        "shuffled_support_target_control",
        "all_particles_support_exact_task_rate",
    ),
    "training_seconds": ("training_seconds",),
}

# Raw paired deltas are always factorized minus unstructured.  This registry
# controls only the wins/ties/losses interpretation; it never flips the raw
# values reported in the artifact.
METRIC_DIRECTIONS: dict[str, str] = {
    name: (
        "lower"
        if name == "training_seconds" or name.startswith("shuffled.")
        else "higher"
    )
    for name in METRIC_PATHS
}
RATE_METRICS = {
    name
    for name in METRIC_PATHS
    if any(token in name for token in ("coverage", "rate", "recall"))
}

STATIC_GATE_METRICS = (
    "heldout.coverage_at_4_mass_weighted",
    "heldout.all_classes_covered_task_rate",
    "heldout.factor_tuple_coverage_at_4",
    "heldout.all_particles_support_exact_task_rate",
)
STATIC_GATE_THRESHOLD = 0.90
PARAMETER_MATCH_RELATIVE_TOLERANCE = 0.01
TIE_TOLERANCE = 1e-12


class SuiteError(RuntimeError):
    """Base error for an invalid or non-resumable suite."""


class ResultValidationError(SuiteError):
    """Raised when a purportedly complete run artifact is not auditable."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=REPOSITORY_ROOT / "scripts/run_expected_discrete_causal_coverage.py",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tail-learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-pool-tasks", type=int, default=144)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validity-weight", type=float, default=0.10)
    parser.add_argument("--diversity-weight", type=float, default=0.10)
    parser.add_argument("--sharpening-weight-end", type=float, default=0.0)
    parser.add_argument("--sharpening-start-fraction", type=float, default=0.80)
    parser.add_argument("--proper-weight", type=float, default=1.0)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--factor-temperature-start", type=float, default=1.0)
    parser.add_argument("--factor-temperature-end", type=float, default=1.0)
    parser.add_argument("--assignment-temperature", type=float, default=0.0)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--nll-threshold", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-split", default="expected-discrete-causal-train")
    parser.add_argument(
        "--eval-split", default="expected-discrete-causal-composition"
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ResultValidationError("result_invalid", f"cannot read {path}: {error}")
    if not isinstance(payload, dict):
        raise ResultValidationError("result_invalid", f"{path} is not a JSON object")
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_or_validate_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise SuiteError(
                f"immutable configuration differs from existing file: {path}"
            )
        return
    _atomic_json(path, payload)


def _artifact(path: Path, suite_root: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        relative = path.resolve().relative_to(suite_root.resolve())
        displayed_path = relative.as_posix()
    except ValueError:
        displayed_path = str(path.resolve())
    return {"path": displayed_path, "sha256": _sha256_file(path)}


def _python_version(python: Path) -> str:
    completed = subprocess.run(
        [str(python), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SuiteError(f"cannot execute Python interpreter: {python}")
    return (completed.stdout or completed.stderr).strip()


def _source_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in RUNNER_AUDITED_SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise SuiteError(f"audited runtime source is missing: {path}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _runner_args_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_pool_tasks": args.train_pool_tasks,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "tail_steps": args.tail_steps,
        "tail_learning_rate": args.tail_learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "validity_weight": args.validity_weight,
        "diversity_weight": args.diversity_weight,
        "sharpening_weight_end": args.sharpening_weight_end,
        "sharpening_start_fraction": args.sharpening_start_fraction,
        "proper_weight": args.proper_weight,
        "balanced_weight": args.balanced_weight,
        "factor_temperature_start": args.factor_temperature_start,
        "factor_temperature_end": args.factor_temperature_end,
        "assignment_temperature": args.assignment_temperature,
        "unstructured_head_kind": "low-rank",
        "unstructured_head_rank": None,
        "attention_layers": args.attention_layers,
        "nll_threshold": args.nll_threshold,
        "log_every": args.log_every,
        "device": args.device,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
    }


def _validate_cli_args(args: argparse.Namespace) -> None:
    for name in (
        "max_workers",
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "eval_batch_size",
        "attention_layers",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            raise SuiteError(f"--{name.replace('_', '-')} must be positive")
    if args.tail_steps < 0 or args.tail_steps > args.steps:
        raise SuiteError("--tail-steps must lie in [0, steps]")
    for name in (
        "learning_rate",
        "tail_learning_rate",
        "max_grad_norm",
        "factor_temperature_start",
        "factor_temperature_end",
        "nll_threshold",
    ):
        if getattr(args, name) <= 0:
            raise SuiteError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "validity_weight",
        "diversity_weight",
        "sharpening_weight_end",
        "proper_weight",
        "balanced_weight",
        "assignment_temperature",
    ):
        if getattr(args, name) < 0:
            raise SuiteError(f"--{name.replace('_', '-')} must be non-negative")
    if args.assignment_temperature != 0:
        raise SuiteError("the preregistered suite fixes --assignment-temperature=0")
    if not 0 <= args.sharpening_start_fraction < 1:
        raise SuiteError("--sharpening-start-fraction must lie in [0,1)")
    if args.batch_size > args.train_pool_tasks:
        raise SuiteError("--batch-size cannot exceed --train-pool-tasks")
    if args.train_split == args.eval_split:
        raise SuiteError("training and evaluation splits must differ")


def build_suite_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the immutable suite configuration from validated CLI arguments."""

    _validate_cli_args(args)
    runner = args.runner.resolve()
    python = args.python.resolve()
    executor = args.executor_checkpoint.resolve()
    for label, path in (
        ("runner", runner),
        ("Python interpreter", python),
        ("executor checkpoint", executor),
    ):
        if not path.is_file():
            raise SuiteError(f"{label} is missing: {path}")
    source_sha256 = _source_manifest()
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": "factorization_latin_4fold_3seed_v1",
        "models": list(MODELS),
        "context_folds": list(FOLDS),
        "model_seeds": list(SEEDS),
        "data_master_seed": DATA_MASTER_SEED,
        "planned_runs": len(MODELS) * len(FOLDS) * len(SEEDS),
        "planned_pairs": len(FOLDS) * len(SEEDS),
        "max_workers": args.max_workers,
        "python": {
            "path": str(python),
            "version": _python_version(python),
        },
        "runner": {"path": str(runner), "sha256": _sha256_file(runner)},
        "executor_checkpoint": {
            "path": str(executor),
            "sha256": _sha256_file(executor),
        },
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "runner_source_sha256": source_sha256,
        "runner_args": _runner_args_from_namespace(args),
        "latin_split": {
            "rule": "(observed_value_1 + observed_value_2) % 4 == context_fold",
            "unique_train_support_contexts": 36,
            "unique_eval_support_contexts": 12,
            "eval_contexts_per_axis": 4,
        },
        "static_gate": {
            "metric_names": list(STATIC_GATE_METRICS),
            "threshold_gte": STATIC_GATE_THRESHOLD,
        },
        "statistics": {
            "std": "sample_ddof_1",
            "paired_delta": "factorized-3x4 minus unstructured-64",
            "metric_directions": METRIC_DIRECTIONS,
            "tie_absolute_tolerance": TIE_TOLERANCE,
            "failed_runs_count_as_gate_failures_in_planned_rate": True,
            "inference_scope": "descriptive_only",
            "dependence_note": (
                "Latin folds share the same 48-context universe and data "
                "generator; fold/seed summaries are correlated repetitions, "
                "not independent samples. No confidence interval or "
                "hypothesis-test claim is made."
            ),
        },
        "parameter_matching": {
            "denominator": "factorized_trainable_parameters",
            "relative_tolerance_lte": PARAMETER_MATCH_RELATIVE_TOLERANCE,
        },
        "metric_paths": {
            name: list(path) for name, path in METRIC_PATHS.items()
        },
    }


def build_run_specs(
    suite_config: Mapping[str, Any], suite_config_sha256: str
) -> list[dict[str, Any]]:
    """Return the pair-adjacent, deterministic 24-run matrix."""

    specs: list[dict[str, Any]] = []
    for fold in suite_config["context_folds"]:
        for seed in suite_config["model_seeds"]:
            for model in suite_config["models"]:
                run_id = f"{model}__fold{fold}__seed{seed}"
                specs.append(
                    {
                        "schema_version": RUN_SPEC_SCHEMA_VERSION,
                        "suite_config_sha256": suite_config_sha256,
                        "run_id": run_id,
                        "model": model,
                        "context_fold": fold,
                        "model_seed": seed,
                        "data_master_seed": suite_config["data_master_seed"],
                        "runner_args": suite_config["runner_args"],
                        "expected": {
                            "context_split_kind": "latin_modulo_4",
                            "unique_train_support_contexts": 36,
                            "unique_eval_support_contexts": 12,
                            "head_kind": EXPECTED_HEAD_KIND[model],
                            "head_rank": EXPECTED_HEAD_RANK[model],
                        },
                    }
                )
    identities = {spec["run_id"] for spec in specs}
    if len(specs) != 24 or len(identities) != 24:
        raise SuiteError("the preregistered matrix must contain 24 unique runs")
    return specs


def _run_root(suite_root: Path, spec: Mapping[str, Any]) -> Path:
    return (
        suite_root
        / "runs"
        / str(spec["model"])
        / f"fold_{spec['context_fold']}"
        / f"seed_{spec['model_seed']}"
    )


def _expected_contexts(fold: int) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    all_contexts = {
        (axis, first, second)
        for axis in range(3)
        for first in range(4)
        for second in range(4)
    }
    heldout = {
        context for context in all_contexts if (context[1] + context[2]) % 4 == fold
    }
    return all_contexts - heldout, heldout


def _context_set(value: Any, field: str) -> set[tuple[int, int, int]]:
    if not isinstance(value, list):
        raise ResultValidationError("result_invalid", f"{field} must be a list")
    contexts: list[tuple[int, int, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or any(type(component) is not int for component in item)
        ):
            raise ResultValidationError(
                "result_invalid", f"{field} contains an invalid context"
            )
        contexts.append(tuple(item))
    if len(contexts) != len(set(contexts)):
        raise ResultValidationError(
            "result_invalid", f"{field} contains duplicate contexts"
        )
    return set(contexts)


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise ResultValidationError(
                "result_invalid", f"missing result field: {'.'.join(path)}"
            )
        value = value[component]
    return value


def _finite_number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ResultValidationError(
            "result_invalid", f"{field} must be a finite number"
        )
    return float(value)


def _require_equal(payload: Mapping[str, Any], field: str, expected: Any) -> None:
    if field not in payload or payload[field] != expected:
        actual = payload.get(field, "<missing>")
        raise ResultValidationError(
            "identity_mismatch",
            f"{field} mismatch: expected {expected!r}, got {actual!r}",
        )


def _result_expected_fields(
    suite_config: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    args = suite_config["runner_args"]
    fields = {
        "model": spec["model"],
        "posterior_parameterization": EXPECTED_POSTERIOR[spec["model"]],
        "experiment": EXPECTED_EXPERIMENT[spec["model"]],
        "context_fold": spec["context_fold"],
        "context_split_kind": "latin_modulo_4",
        "model_seed": spec["model_seed"],
        "data_master_seed": DATA_MASTER_SEED,
        "executor_checkpoint_sha256": suite_config["executor_checkpoint"]["sha256"],
        "steps": args["steps"],
        "batch_size": args["batch_size"],
        "train_pool_tasks": args["train_pool_tasks"],
        "eval_tasks": args["eval_tasks"],
        "eval_batch_size": args["eval_batch_size"],
        "learning_rate": args["learning_rate"],
        "tail_steps": args["tail_steps"],
        "tail_learning_rate": args["tail_learning_rate"],
        "tail_start_step": (
            args["steps"] - args["tail_steps"] if args["tail_steps"] else None
        ),
        "weight_decay": args["weight_decay"],
        "max_grad_norm": args["max_grad_norm"],
        "validity_weight": args["validity_weight"],
        "diversity_weight": args["diversity_weight"],
        "sharpening_weight_end": args["sharpening_weight_end"],
        "sharpening_start_fraction": args["sharpening_start_fraction"],
        "proper_weight": args["proper_weight"],
        "balanced_weight": args["balanced_weight"],
        "factor_temperature_start": args["factor_temperature_start"],
        "factor_temperature_end": args["factor_temperature_end"],
        "assignment_temperature": args["assignment_temperature"],
        "head_kind": EXPECTED_HEAD_KIND[spec["model"]],
        "requested_unstructured_head_kind": EXPECTED_REQUESTED_HEAD_KIND[
            spec["model"]
        ],
        "requested_unstructured_head_rank": args["unstructured_head_rank"],
        "attention_layers": args["attention_layers"],
        "nll_threshold": args["nll_threshold"],
        "log_every": args["log_every"],
        "train_split": args["train_split"],
        "eval_split": args["eval_split"],
        "device": args["device"],
        "head_rank": EXPECTED_HEAD_RANK[spec["model"]],
        "unique_train_support_contexts": 36,
        "unique_eval_support_contexts": 12,
        "train_eval_contexts_disjoint": True,
        "initial_checkpoint": None,
        "initial_checkpoint_sha256": None,
        "initial_training_steps": 0,
        "cumulative_training_steps": args["steps"],
        "tasks_seen": args["steps"] * args["batch_size"],
        "cumulative_tasks_seen": args["steps"] * args["batch_size"],
    }
    return fields


def _validate_rate_like_leaves(value: Any, path: tuple[str, ...]) -> None:
    """Reject non-finite or out-of-range rate/recall/coverage leaves."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_rate_like_leaves(child, path + (str(key),))
        return
    if not path:
        return
    leaf = path[-1]
    if any(token in leaf for token in ("rate", "recall", "coverage")):
        numeric = _finite_number(value, ".".join(path))
        if not 0.0 <= numeric <= 1.0:
            raise ResultValidationError(
                "result_invalid",
                f"{'.'.join(path)} must lie in [0,1], got {numeric}",
            )


def _validate_evaluation_section(
    payload: Mapping[str, Any],
    *,
    section: str,
    expected_tasks: int,
    expected_support_ablation: str,
    nll_threshold: float,
) -> None:
    _require_equal(payload, "tasks", expected_tasks)
    _require_equal(payload, "support_ablation", expected_support_ablation)
    _require_equal(payload, "behavior_classes", expected_tasks * 4)
    _require_equal(payload, "coverage_nll_threshold_per_cell", nll_threshold)
    covered = payload.get("covered_behavior_classes")
    if type(covered) is not int or not 0 <= covered <= expected_tasks * 4:
        raise ResultValidationError(
            "result_invalid",
            f"{section}.covered_behavior_classes must lie in [0,{expected_tasks * 4}]",
        )
    for field in ("mean_unique_factor_tuples", "mean_unique_map_signatures"):
        if field in payload:
            numeric = _finite_number(payload[field], f"{section}.{field}")
            if not 1.0 <= numeric <= 4.0:
                raise ResultValidationError(
                    "result_invalid", f"{section}.{field} must lie in [1,4]"
                )
    _validate_rate_like_leaves(payload, (section,))


def _validate_checkpoint_metadata(
    *,
    checkpoint_path: Path,
    suite_config: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    """Load the small checkpoint and validate its internal run identity."""

    try:
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:  # noqa: BLE001 - convert to artifact audit failure
        raise ResultValidationError(
            "artifact_mismatch",
            f"cannot load checkpoint metadata: {type(error).__name__}: {error}",
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise ResultValidationError(
            "artifact_mismatch", "checkpoint root must be a mapping"
        )
    _require_equal(
        checkpoint,
        "checkpoint_schema_version",
        EXPECTED_CHECKPOINT_SCHEMA_VERSION,
    )
    _require_equal(checkpoint, "model_type", EXPECTED_MODEL_TYPE[spec["model"]])
    result_expected = _result_expected_fields(suite_config, spec)
    checkpoint_fields = (
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
        "requested_unstructured_head_kind",
        "requested_unstructured_head_rank",
        "attention_layers",
        "nll_threshold",
        "log_every",
        "train_split",
        "eval_split",
        "device",
        "head_rank",
        "unique_train_support_contexts",
        "unique_eval_support_contexts",
        "train_eval_contexts_disjoint",
        "initial_checkpoint",
        "initial_checkpoint_sha256",
        "initial_training_steps",
        "cumulative_training_steps",
    )
    for field in checkpoint_fields:
        _require_equal(checkpoint, field, result_expected[field])
    if checkpoint.get("source_sha256") != suite_config["runner_source_sha256"]:
        raise ResultValidationError(
            "source_mismatch", "checkpoint source manifest differs from suite"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ResultValidationError(
            "artifact_mismatch", "checkpoint has no non-empty model_state_dict"
        )
    if not isinstance(checkpoint.get("latest_training_metrics"), Mapping):
        raise ResultValidationError(
            "artifact_mismatch", "checkpoint has no latest_training_metrics mapping"
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
    """Validate one completed runner output and return its summary record."""

    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint_last.pt"
    progress_path = output_dir / "progress.jsonl"
    if not result_path.is_file():
        raise ResultValidationError("result_missing", f"missing {result_path}")
    result = _read_json(result_path)
    for field, expected in _result_expected_fields(suite_config, spec).items():
        _require_equal(result, field, expected)

    train_contexts = _context_set(
        result.get("train_support_contexts"), "train_support_contexts"
    )
    eval_contexts = _context_set(
        result.get("eval_support_contexts"), "eval_support_contexts"
    )
    expected_train, expected_eval = _expected_contexts(int(spec["context_fold"]))
    if train_contexts != expected_train or eval_contexts != expected_eval:
        raise ResultValidationError(
            "identity_mismatch", "recorded support contexts do not match the Latin fold"
        )
    if train_contexts & eval_contexts:
        raise ResultValidationError(
            "result_invalid", "training and evaluation contexts overlap"
        )

    source_manifest = result.get("source_sha256")
    if source_manifest != suite_config["runner_source_sha256"]:
        raise ResultValidationError(
            "source_mismatch", "runner source manifest differs from suite configuration"
        )
    if not checkpoint_path.is_file():
        raise ResultValidationError(
            "artifact_mismatch", f"missing checkpoint: {checkpoint_path}"
        )
    recorded_checkpoint_path = result.get("checkpoint_path")
    if not isinstance(recorded_checkpoint_path, str) or (
        Path(recorded_checkpoint_path).resolve() != checkpoint_path.resolve()
    ):
        raise ResultValidationError(
            "artifact_mismatch", "checkpoint_path does not name this attempt's checkpoint"
        )
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ResultValidationError(
            "artifact_mismatch", "checkpoint SHA256 does not match checkpoint bytes"
        )
    _validate_checkpoint_metadata(
        checkpoint_path=checkpoint_path,
        suite_config=suite_config,
        spec=spec,
    )
    if not progress_path.is_file() or progress_path.stat().st_size == 0:
        raise ResultValidationError(
            "artifact_mismatch", "training progress artifact is missing or empty"
        )

    metrics: dict[str, float] = {}
    for name, path in METRIC_PATHS.items():
        metrics[name] = _finite_number(_nested_value(result, path), name)
    for name in RATE_METRICS:
        if not 0.0 <= metrics[name] <= 1.0:
            raise ResultValidationError(
                "result_invalid", f"{name} must lie in [0,1]"
            )
    if metrics["training_seconds"] < 0:
        raise ResultValidationError(
            "result_invalid", "training_seconds must be non-negative"
        )
    for section, expected_tasks, support_ablation in (
        (
            "heldout_triple_coverage",
            suite_config["runner_args"]["eval_tasks"],
            "none",
        ),
        ("seen_context_triple_coverage", 144, "none"),
        (
            "shuffled_support_target_control",
            suite_config["runner_args"]["eval_tasks"],
            "shuffle-targets",
        ),
    ):
        section_payload = result.get(section)
        if not isinstance(section_payload, Mapping):
            raise ResultValidationError("result_invalid", f"missing {section}")
        _validate_evaluation_section(
            section_payload,
            section=section,
            expected_tasks=expected_tasks,
            expected_support_ablation=support_ablation,
            nll_threshold=float(suite_config["runner_args"]["nll_threshold"]),
        )

    gate_payload = result.get("static_gate")
    if not isinstance(gate_payload, Mapping):
        raise ResultValidationError("result_invalid", "static_gate must be an object")
    for field in (
        "coverage_at_4_gte",
        "all_classes_covered_task_rate_gte",
        "factor_tuple_coverage_at_4_gte",
        "all_particles_support_exact_task_rate_gte",
    ):
        _require_equal(gate_payload, field, STATIC_GATE_THRESHOLD)
    recomputed_gate = all(metrics[name] >= STATIC_GATE_THRESHOLD for name in STATIC_GATE_METRICS)
    if type(gate_payload.get("passed")) is not bool:
        raise ResultValidationError("result_invalid", "static_gate.passed must be bool")
    if gate_payload["passed"] != recomputed_gate:
        raise ResultValidationError(
            "result_invalid", "static_gate.passed disagrees with recomputed metrics"
        )
    trainable_parameters = result.get("trainable_parameters")
    model_parameters = result.get("model_parameters")
    if type(trainable_parameters) is not int or trainable_parameters <= 0:
        raise ResultValidationError(
            "result_invalid", "trainable_parameters must be a positive integer"
        )
    if type(model_parameters) is not int or model_parameters < trainable_parameters:
        raise ResultValidationError(
            "result_invalid", "model_parameters must include trainable parameters"
        )

    artifacts = {
        "run_spec": _artifact(run_spec_path, suite_root),
        "stdout": _artifact(stdout_path, suite_root),
        "stderr": _artifact(stderr_path, suite_root),
        "progress": _artifact(progress_path, suite_root),
        "checkpoint": _artifact(checkpoint_path, suite_root),
        "result": _artifact(result_path, suite_root),
    }
    return {
        "run_id": spec["run_id"],
        "model": spec["model"],
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
        "status": "succeeded",
        "static_gate_passed": recomputed_gate,
        "model_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
        "head_kind": result["head_kind"],
        "head_rank": result["head_rank"],
        "training_seconds": metrics["training_seconds"],
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _runner_command(
    suite_config: Mapping[str, Any], spec: Mapping[str, Any], output_dir: Path
) -> list[str]:
    args = suite_config["runner_args"]
    command = [
        suite_config["python"]["path"],
        suite_config["runner"]["path"],
        "--output",
        str(output_dir),
        "--executor-checkpoint",
        suite_config["executor_checkpoint"]["path"],
        "--model",
        str(spec["model"]),
        "--context-fold",
        str(spec["context_fold"]),
        "--seed",
        str(spec["model_seed"]),
        "--data-master-seed",
        str(spec["data_master_seed"]),
    ]
    if spec["model"] == "unstructured-64":
        command.extend(
            (
                "--unstructured-head-kind",
                str(args["unstructured_head_kind"]),
            )
        )
    for name in (
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "eval_batch_size",
        "learning_rate",
        "tail_steps",
        "tail_learning_rate",
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
        "attention_layers",
        "nll_threshold",
        "log_every",
        "device",
        "train_split",
        "eval_split",
    ):
        command.extend((f"--{name.replace('_', '-')}", str(args[name])))
    return command


_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen[bytes]] = set()
_INTERRUPTED = threading.Event()


def _invoke_subprocess(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    environment: Mapping[str, str],
) -> int:
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(
            list(command),
            cwd=REPOSITORY_ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            env=dict(environment),
        )
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.add(process)
        try:
            return process.wait()
        finally:
            with _PROCESS_LOCK:
                _ACTIVE_PROCESSES.discard(process)


ProcessRunner = Callable[..., int]


def _terminate_active_processes() -> None:
    _INTERRUPTED.set()
    with _PROCESS_LOCK:
        processes = tuple(_ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def _attempt_directories(run_root: Path) -> list[Path]:
    attempts_root = run_root / "attempts"
    if not attempts_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in attempts_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


def _failure_record(
    *,
    spec: Mapping[str, Any],
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
        "model": spec["model"],
        "context_fold": spec["context_fold"],
        "model_seed": spec["model_seed"],
        "status": status,
        "execution_action": action,
        "attempt_index": attempt_index,
        "static_gate_passed": None,
        "model_parameters": None,
        "trainable_parameters": None,
        "head_kind": None,
        "head_rank": None,
        "training_seconds": None,
        "metrics": None,
        "failure": {
            "kind": kind,
            "message": message,
            "returncode": returncode,
        },
        "artifacts": dict(artifacts),
    }


def execute_run(
    *,
    suite_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    suite_root: Path,
    process_runner: ProcessRunner = _invoke_subprocess,
) -> dict[str, Any]:
    """Resume or execute one run, retaining every failed attempt."""

    run_root = _run_root(suite_root, spec)
    run_root.mkdir(parents=True, exist_ok=True)
    run_spec_path = run_root / "run_spec.json"
    _write_or_validate_json(run_spec_path, dict(spec))
    run_spec_sha256 = _sha256_file(run_spec_path)
    status_path = run_root / "status.json"

    # Adopt any valid prior output, even if the orchestrator died before it
    # could finalize status.json.
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
        except ResultValidationError:
            continue
        attempt_index = int(attempt_dir.name)
        record |= {
            "attempt_index": attempt_index,
            "execution_action": "skipped_valid",
            "run_spec_sha256": run_spec_sha256,
        }
        record["artifacts"]["attempt"] = _artifact(
            attempt_dir / "attempt.json", suite_root
        )
        _atomic_json(
            status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "state": "succeeded",
                "run_id": spec["run_id"],
                "attempt_index": attempt_index,
                "run_spec_sha256": run_spec_sha256,
                "last_action": "skipped_valid",
                "updated_at_utc": _utc_now(),
            },
        )
        return record

    attempts = _attempt_directories(run_root)
    attempt_index = int(attempts[-1].name) + 1 if attempts else 1
    attempt_dir = run_root / "attempts" / f"{attempt_index:03d}"
    output_dir = attempt_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    command = _runner_command(suite_config, spec, output_dir)
    started_at = _utc_now()
    attempt_path = attempt_dir / "attempt.json"
    running_attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "state": "running",
        "attempt_index": attempt_index,
        "run_id": spec["run_id"],
        "run_spec_sha256": run_spec_sha256,
        "command": command,
        "started_at_utc": started_at,
    }
    _atomic_json(attempt_path, running_attempt)
    _atomic_json(
        status_path,
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "state": "running",
            "run_id": spec["run_id"],
            "attempt_index": attempt_index,
            "run_spec_sha256": run_spec_sha256,
            "updated_at_utc": started_at,
        },
    )
    environment = dict(os.environ)
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment["PYTHONHASHSEED"] = str(spec["model_seed"])
    start = time.monotonic()
    try:
        returncode = process_runner(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    except Exception as error:  # noqa: BLE001 - persisted as an audited failure
        duration = time.monotonic() - start
        stdout_path.touch(exist_ok=True)
        stderr_path.touch(exist_ok=True)
        failure_kind = "spawn_failed"
        message = f"{type(error).__name__}: {error}"
        final_attempt = running_attempt | {
            "state": "failed",
            "finished_at_utc": _utc_now(),
            "duration_seconds": duration,
            "returncode": None,
            "failure": {"kind": failure_kind, "message": message},
        }
        _atomic_json(attempt_path, final_attempt)
        artifacts = {
            "run_spec": _artifact(run_spec_path, suite_root),
            "attempt": _artifact(attempt_path, suite_root),
            "stdout": _artifact(stdout_path, suite_root),
            "stderr": _artifact(stderr_path, suite_root),
        }
        record = _failure_record(
            spec=spec,
            status="failed",
            attempt_index=attempt_index,
            kind=failure_kind,
            message=message,
            returncode=None,
            artifacts=artifacts,
            action="executed",
        )
        record["run_spec_sha256"] = run_spec_sha256
        _atomic_json(
            status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "state": "failed",
                "run_id": spec["run_id"],
                "attempt_index": attempt_index,
                "run_spec_sha256": run_spec_sha256,
                "last_action": "executed",
                "updated_at_utc": _utc_now(),
                "returncode": None,
                "failure": final_attempt["failure"],
                "artifacts": artifacts,
            },
        )
        return record

    duration = time.monotonic() - start
    if returncode != 0:
        failure_kind = "interrupted" if _INTERRUPTED.is_set() else "process_failed"
        message = f"runner exited with status {returncode}"
        final_attempt = running_attempt | {
            "state": "interrupted" if _INTERRUPTED.is_set() else "failed",
            "finished_at_utc": _utc_now(),
            "duration_seconds": duration,
            "returncode": returncode,
            "failure": {"kind": failure_kind, "message": message},
        }
        _atomic_json(attempt_path, final_attempt)
        artifacts = {
            "run_spec": _artifact(run_spec_path, suite_root),
            "attempt": _artifact(attempt_path, suite_root),
            "stdout": _artifact(stdout_path, suite_root),
            "stderr": _artifact(stderr_path, suite_root),
        }
        record = _failure_record(
            spec=spec,
            status=final_attempt["state"],
            attempt_index=attempt_index,
            kind=failure_kind,
            message=message,
            returncode=returncode,
            artifacts=artifacts,
            action="executed",
        )
        record["run_spec_sha256"] = run_spec_sha256
        _atomic_json(
            status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "state": final_attempt["state"],
                "run_id": spec["run_id"],
                "attempt_index": attempt_index,
                "run_spec_sha256": run_spec_sha256,
                "last_action": "executed",
                "updated_at_utc": _utc_now(),
                "returncode": returncode,
                "failure": final_attempt["failure"],
                "artifacts": artifacts,
            },
        )
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
    except ResultValidationError as error:
        final_attempt = running_attempt | {
            "state": "invalid",
            "finished_at_utc": _utc_now(),
            "duration_seconds": duration,
            "returncode": returncode,
            "failure": {"kind": error.kind, "message": str(error)},
        }
        _atomic_json(attempt_path, final_attempt)
        artifacts = {
            "run_spec": _artifact(run_spec_path, suite_root),
            "attempt": _artifact(attempt_path, suite_root),
            "stdout": _artifact(stdout_path, suite_root),
            "stderr": _artifact(stderr_path, suite_root),
            "progress": _artifact(output_dir / "progress.jsonl", suite_root),
            "checkpoint": _artifact(output_dir / "checkpoint_last.pt", suite_root),
            "result": _artifact(output_dir / "result.json", suite_root),
        }
        failure_record = _failure_record(
            spec=spec,
            status="invalid",
            attempt_index=attempt_index,
            kind=error.kind,
            message=str(error),
            returncode=returncode,
            artifacts=artifacts,
            action="executed",
        )
        failure_record["run_spec_sha256"] = run_spec_sha256
        _atomic_json(
            status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "state": "invalid",
                "run_id": spec["run_id"],
                "attempt_index": attempt_index,
                "run_spec_sha256": run_spec_sha256,
                "last_action": "executed",
                "updated_at_utc": _utc_now(),
                "returncode": returncode,
                "failure": final_attempt["failure"],
                "artifacts": artifacts,
            },
        )
        return failure_record

    final_attempt = running_attempt | {
        "state": "succeeded",
        "finished_at_utc": _utc_now(),
        "duration_seconds": duration,
        "returncode": returncode,
    }
    _atomic_json(attempt_path, final_attempt)
    record |= {
        "attempt_index": attempt_index,
        "execution_action": "executed",
        "run_spec_sha256": run_spec_sha256,
    }
    record["artifacts"]["attempt"] = _artifact(attempt_path, suite_root)
    _atomic_json(
        status_path,
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "state": "succeeded",
            "run_id": spec["run_id"],
            "attempt_index": attempt_index,
            "run_spec_sha256": run_spec_sha256,
            "last_action": "executed",
            "updated_at_utc": _utc_now(),
            "artifacts": record["artifacts"],
        },
    )
    return record


def summary_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return explicit sample statistics without emitting JSON NaN values."""

    numeric = [float(value) for value in values]
    if not numeric:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) >= 2 else None,
        "min": min(numeric),
        "max": max(numeric),
    }


def _group_summary(
    planned_specs: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = [records_by_id[spec["run_id"]] for spec in planned_specs]
    valid = [record for record in records if record["status"] == "succeeded"]
    metrics = {
        name: summary_statistics(
            [record["metrics"][name] for record in valid]
        )
        for name in METRIC_PATHS
    }
    passed = sum(record["static_gate_passed"] is True for record in valid)
    planned = len(records)
    evaluated = len(valid)
    return {
        "planned_runs": planned,
        "valid_runs": evaluated,
        "metrics": metrics,
        "static_gate": {
            "passed_count": passed,
            "planned_count": planned,
            "pass_rate_planned": passed / planned if planned else None,
            "evaluated_count": evaluated,
            "pass_rate_evaluated": passed / evaluated if evaluated else None,
        },
    }


def _paired_delta_summary(
    values: Sequence[float], *, direction: str
) -> dict[str, Any]:
    if direction not in ("higher", "lower"):
        raise ValueError(f"unknown metric direction: {direction}")
    stats = summary_statistics(values)
    oriented = list(values) if direction == "higher" else [-value for value in values]
    wins = sum(value > TIE_TOLERANCE for value in oriented)
    losses = sum(value < -TIE_TOLERANCE for value in oriented)
    ties = len(values) - wins - losses
    return stats | {
        "direction": direction,
        "raw_delta": "factorized-3x4 minus unstructured-64",
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(values) if values else None,
    }


def build_summary(
    *,
    suite_config: Mapping[str, Any],
    suite_config_sha256: str,
    specs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate valid and failed runs without hiding missing observations."""

    records_by_id = {str(record["run_id"]): record for record in records}
    if set(records_by_id) != {str(spec["run_id"]) for spec in specs}:
        raise SuiteError("summary requires exactly one record for every planned run")
    ordered_records = [records_by_id[str(spec["run_id"])] for spec in specs]
    valid = [record for record in ordered_records if record["status"] == "succeeded"]
    failed = [record for record in ordered_records if record["status"] == "failed"]
    invalid = [record for record in ordered_records if record["status"] == "invalid"]
    interrupted = [
        record for record in ordered_records if record["status"] == "interrupted"
    ]

    by_model: dict[str, Any] = {}
    by_model_fold: dict[str, Any] = {}
    by_model_seed: dict[str, Any] = {}
    for model in MODELS:
        model_specs = [spec for spec in specs if spec["model"] == model]
        by_model[model] = _group_summary(model_specs, records_by_id)
        by_model_fold[model] = {
            str(fold): _group_summary(
                [spec for spec in model_specs if spec["context_fold"] == fold],
                records_by_id,
            )
            for fold in FOLDS
        }
        by_model_seed[model] = {
            str(seed): _group_summary(
                [spec for spec in model_specs if spec["model_seed"] == seed],
                records_by_id,
            )
            for seed in SEEDS
        }

    pair_details: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    paired_values: dict[str, list[float]] = {name: [] for name in METRIC_PATHS}
    gate_contingency = {
        "both_pass": 0,
        "factorized_only": 0,
        "unstructured_only": 0,
        "neither": 0,
    }
    parameter_pairs: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            factorized_id = f"factorized-3x4__fold{fold}__seed{seed}"
            unstructured_id = f"unstructured-64__fold{fold}__seed{seed}"
            factorized = records_by_id[factorized_id]
            unstructured = records_by_id[unstructured_id]
            if factorized["status"] != "succeeded" or unstructured["status"] != "succeeded":
                missing_pairs.append(
                    {
                        "context_fold": fold,
                        "model_seed": seed,
                        "factorized_status": factorized["status"],
                        "unstructured_status": unstructured["status"],
                    }
                )
                continue
            deltas = {
                name: factorized["metrics"][name] - unstructured["metrics"][name]
                for name in METRIC_PATHS
            }
            for name, value in deltas.items():
                paired_values[name].append(value)
            factorized_pass = bool(factorized["static_gate_passed"])
            unstructured_pass = bool(unstructured["static_gate_passed"])
            if factorized_pass and unstructured_pass:
                gate_contingency["both_pass"] += 1
            elif factorized_pass:
                gate_contingency["factorized_only"] += 1
            elif unstructured_pass:
                gate_contingency["unstructured_only"] += 1
            else:
                gate_contingency["neither"] += 1
            factorized_parameters = int(factorized["trainable_parameters"])
            unstructured_parameters = int(unstructured["trainable_parameters"])
            absolute_delta = unstructured_parameters - factorized_parameters
            relative_gap = abs(absolute_delta) / factorized_parameters
            parameter_pair = {
                "context_fold": fold,
                "model_seed": seed,
                "factorized_trainable_parameters": factorized_parameters,
                "unstructured_trainable_parameters": unstructured_parameters,
                "unstructured_minus_factorized": absolute_delta,
                "relative_absolute_gap_to_factorized": relative_gap,
                "within_tolerance": relative_gap
                <= PARAMETER_MATCH_RELATIVE_TOLERANCE,
            }
            parameter_pairs.append(parameter_pair)
            pair_details.append(
                {
                    "context_fold": fold,
                    "model_seed": seed,
                    "factorized_run_id": factorized_id,
                    "unstructured_run_id": unstructured_id,
                    "factorized_static_gate_passed": factorized_pass,
                    "unstructured_static_gate_passed": unstructured_pass,
                    "delta_factorized_minus_unstructured": deltas,
                }
            )

    complete_pairs = len(pair_details)
    parameter_evaluable = complete_pairs == len(FOLDS) * len(SEEDS)
    parameter_passed = parameter_evaluable and all(
        pair["within_tolerance"] for pair in parameter_pairs
    )
    complete = len(valid) == len(specs)
    integrity_passed = complete and complete_pairs == 12 and parameter_passed
    action_counts: dict[str, int] = {}
    for record in ordered_records:
        action = str(record.get("execution_action", "unknown"))
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_config_sha256": suite_config_sha256,
        "complete": complete,
        "planned_runs": len(specs),
        "valid_runs": len(valid),
        "failed_runs": len(failed),
        "invalid_runs": len(invalid),
        "interrupted_runs": len(interrupted),
        "planned_pairs": len(FOLDS) * len(SEEDS),
        "complete_pairs": complete_pairs,
        "executor_checkpoint": suite_config["executor_checkpoint"],
        "runner_source_sha256": suite_config["runner_source_sha256"],
        "resume_actions": action_counts,
        "runs": ordered_records,
        "aggregates": {
            "std_definition": "sample_ddof_1",
            "inference_scope": "descriptive_only",
            "dependence_note": suite_config["statistics"]["dependence_note"],
            "by_model": by_model,
            "by_model_and_fold": by_model_fold,
            "by_model_and_seed": by_model_seed,
        },
        "paired_comparison": {
            "direction": "factorized-3x4 minus unstructured-64",
            "metric_directions": METRIC_DIRECTIONS,
            "inference_scope": "descriptive_only",
            "dependence_note": suite_config["statistics"]["dependence_note"],
            "tie_absolute_tolerance": TIE_TOLERANCE,
            "planned_pairs": len(FOLDS) * len(SEEDS),
            "complete_pairs": complete_pairs,
            "missing_pairs": missing_pairs,
            "details": pair_details,
            "metrics": {
                name: _paired_delta_summary(
                    values,
                    direction=METRIC_DIRECTIONS[name],
                )
                for name, values in paired_values.items()
            },
            "static_gate_contingency": gate_contingency,
        },
        "parameter_matching": {
            "relative_tolerance_lte": PARAMETER_MATCH_RELATIVE_TOLERANCE,
            "denominator": "factorized_trainable_parameters",
            "evaluable": parameter_evaluable,
            "passed": parameter_passed,
            "pairs": parameter_pairs,
            "max_relative_absolute_gap": (
                max(
                    pair["relative_absolute_gap_to_factorized"]
                    for pair in parameter_pairs
                )
                if parameter_pairs
                else None
            ),
        },
        "integrity_gate": {
            "all_24_runs_valid": complete,
            "all_12_pairs_complete": complete_pairs == 12,
            "source_and_executor_identity_validated_per_run": complete,
            "parameter_matching_within_one_percent": parameter_passed,
            "passed": integrity_passed,
            "note": (
                "This is an audit-completeness gate, not a post-hoc scientific "
                "factorization-advantage threshold."
            ),
        },
        "artifact_manifest": {
            record["run_id"]: record["artifacts"] for record in ordered_records
        },
    }


@contextmanager
def _suite_lock(suite_root: Path) -> Iterator[None]:
    lock_path = suite_root / ".suite.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SuiteError(f"another orchestrator holds {lock_path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired={_utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_matrix(
    *,
    suite_config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    suite_root: Path,
    process_runner: ProcessRunner = _invoke_subprocess,
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
            except Exception as error:  # noqa: BLE001 - preserve matrix progress
                record = _failure_record(
                    spec=spec,
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
        _terminate_active_processes()
        for future in futures:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    # Every planned identity receives a record, even after Ctrl-C, so partial
    # summaries retain the planned denominator.
    for spec in specs:
        run_id = str(spec["run_id"])
        if run_id not in records:
            records[run_id] = _failure_record(
                spec=spec,
                status="interrupted" if interrupted else "failed",
                attempt_index=None,
                kind="interrupted" if interrupted else "orchestrator_error",
                message=(
                    "suite interrupted before this run completed"
                    if interrupted
                    else "run produced no orchestration record"
                ),
                returncode=None,
                artifacts={},
                action="not_completed",
            )
    return [records[str(spec["run_id"])] for spec in specs], interrupted


def run_suite(
    args: argparse.Namespace,
    *,
    process_runner: ProcessRunner = _invoke_subprocess,
) -> tuple[dict[str, Any], bool]:
    suite_root = args.output.resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    with _suite_lock(suite_root):
        suite_config = build_suite_config(args)
        suite_config_path = suite_root / "suite_config.json"
        _write_or_validate_json(suite_config_path, suite_config)
        suite_config_sha256 = _payload_sha256(suite_config)
        specs = build_run_specs(suite_config, suite_config_sha256)
        records, interrupted = _run_matrix(
            suite_config=suite_config,
            specs=specs,
            suite_root=suite_root,
            process_runner=process_runner,
        )
        summary = build_summary(
            suite_config=suite_config,
            suite_config_sha256=suite_config_sha256,
            specs=specs,
            records=records,
        )
        summary_path = suite_root / "summary.json"
        _atomic_json(summary_path, summary)
        summary_sha256 = _sha256_file(summary_path)
        temporary = suite_root / "summary.sha256.tmp"
        temporary.write_text(f"{summary_sha256}  summary.json\n", encoding="utf-8")
        temporary.replace(suite_root / "summary.sha256")
        print(
            json.dumps(
                {
                    "summary_path": str(summary_path),
                    "summary_sha256": summary_sha256,
                    "complete": summary["complete"],
                    "integrity_gate_passed": summary["integrity_gate"]["passed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return summary, interrupted


def main() -> int:
    args = parse_args()
    try:
        summary, interrupted = run_suite(args)
    except SuiteError as error:
        print(f"suite error: {error}", file=sys.stderr)
        return 2
    if interrupted:
        return 130
    return 0 if summary["integrity_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
