#!/usr/bin/env python3
"""Train an amortized K=4 rule posterior with detached 64-code costs.

The encoder reads public support transitions.  Training-only unordered
behavior panels are evaluated under every integer mechanism tuple by a frozen,
support-calibrated executor.  The resulting cost table is detached; gradients
reach the encoder through the exact expectation under the selected categorical
posterior, never through a straight-through decoder surrogate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


CHECKPOINT_SCHEMA_VERSION = "prp-wm.expected-discrete-causal-k4.v1"
_AUDITED_SOURCE_FILES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--model",
        choices=("factorized-3x4", "unstructured-64"),
        default="factorized-3x4",
    )
    parser.add_argument(
        "--unstructured-head-kind",
        choices=("low-rank", "direct-linear"),
        default="low-rank",
        help=(
            "opaque 64-way proposer head; direct-linear is the unrestricted "
            "LayerNorm-to-64 capacity sensitivity"
        ),
    )
    parser.add_argument(
        "--unstructured-head-rank",
        type=int,
        help=(
            "override the opaque 64-way head bottleneck rank; by default the "
            "nearest parameter-matched rank is selected"
        ),
    )
    parser.add_argument("--context-fold", type=int, choices=range(4))
    parser.add_argument("--continuous-result", type=Path)
    parser.add_argument("--straight-through-result", type=Path)
    parser.add_argument("--explicit-filter-result", type=Path)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-pool-tasks", type=int, default=144)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tail-steps", type=int, default=0)
    parser.add_argument("--tail-learning-rate", type=float, default=5e-4)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="expected-discrete-causal-train")
    parser.add_argument("--eval-split", default="expected-discrete-causal-composition")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    return {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in _AUDITED_SOURCE_FILES
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _linear_schedule(start: float, end: float, step: int, steps: int) -> float:
    if steps <= 1:
        return end
    return start + (step / (steps - 1)) * (end - start)


def _sharpening_schedule(end: float, start_fraction: float, step: int, steps: int) -> float:
    fraction = step / max(steps - 1, 1)
    if fraction <= start_fraction:
        return 0.0
    return end * (fraction - start_fraction) / max(1.0 - start_fraction, 1e-12)


def _load_audited_executor(torch: Any, path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    from scripts.run_causal_mechanism_coverage import _load_executor

    executor, checkpoint = _load_executor(torch, path, device)
    result_path = path.parent / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("causal_filter_executor_gate", {}).get("passed") is not True:
        raise SystemExit("executor did not pass the support-calibrated causal filter gate")
    support = result.get("heldout_support_evaluation", {})
    if (
        support.get("all_four_compatible_six_frame_exact_task_rate") != 1.0
        or support.get("top4_selection", {})
        .get("map_then_balanced_nll", {})
        .get("exact_version_space_task_rate")
        != 1.0
    ):
        raise SystemExit("executor support audit is not exact")
    return executor, checkpoint


def _comparison(path: Path, coverage: float) -> dict[str, object]:
    resolved = path.resolve()
    result = json.loads(resolved.read_text(encoding="utf-8"))
    baseline = result["heldout_triple_coverage"]["coverage_at_4_mass_weighted"]
    return {
        "result_path": str(resolved),
        "result_sha256": _sha256_file(resolved),
        "coverage_at_4": baseline,
        "absolute_gain": coverage - baseline,
    }


def _support_context_key(
    task: Any,
    *,
    factor_ids_for_program: Any,
    version_space: Any,
) -> tuple[int, int, int]:
    """Return ``(heldout_axis, observed_value_0, observed_value_1)``."""

    compatible = {
        factor_ids_for_program(program)
        for program in version_space(
            task.inference.support[:6],
            task.privileged.palette,
        )
    }
    varying = [
        axis
        for axis in range(3)
        if len({code[axis] for code in compatible}) == 4
    ]
    if len(compatible) != 4 or len(varying) != 1:
        raise AssertionError("expected one four-valued held-out mechanism axis")
    heldout_axis = varying[0]
    representative = min(compatible)
    observed = tuple(
        representative[axis]
        for axis in range(3)
        if axis != heldout_axis
    )
    return heldout_axis, observed[0], observed[1]


def _is_diagonal_holdout(context: tuple[int, int, int]) -> bool:
    """Hold out four of sixteen observed-value pairs for every axis."""

    return context[1] == context[2]


def _is_latin_holdout(
    context: tuple[int, int, int],
    context_fold: int,
) -> bool:
    """Hold out one Latin-square fold of observed mechanism-value pairs."""

    if context_fold not in range(4):
        raise ValueError("context_fold must be an integer in [0,3]")
    return (context[1] + context[2]) % 4 == context_fold


def _build_context_pool(
    *,
    make_pilot_tasks: Any,
    split: str,
    master_seed: int,
    diagnostic_indices: tuple[int, ...],
    count: int,
    heldout: bool,
    factor_ids_for_program: Any,
    version_space: Any,
    context_fold: int | None = None,
) -> tuple[Any, ...]:
    """Materialize a deterministic train or unseen-pair context pool.

    ``context_fold=None`` preserves the historical diagonal split used by the
    inference-speed benchmark.  Supplying a fold selects one of four disjoint
    Latin-square test folds via ``(value_1 + value_2) % 4``.
    """

    if context_fold is not None and context_fold not in range(4):
        raise ValueError("context_fold must be an integer in [0,3]")

    selected: list[Any] = []
    cursor = 0
    chunk_size = 192
    while len(selected) < count:
        candidates = make_pilot_tasks(
            split=split,
            master_seed=master_seed,
            start=cursor,
            count=chunk_size,
            diagnostic_indices=diagnostic_indices,
        )
        cursor += chunk_size
        for task in candidates:
            context = _support_context_key(
                task,
                factor_ids_for_program=factor_ids_for_program,
                version_space=version_space,
            )
            is_holdout = (
                _is_diagonal_holdout(context)
                if context_fold is None
                else _is_latin_holdout(context, context_fold)
            )
            if is_holdout == heldout:
                selected.append(task)
                if len(selected) == count:
                    break
        if cursor > max(count * 32, 6144) and len(selected) < count:
            raise RuntimeError("could not materialize the requested context split")
    return tuple(selected)


def _cached_task_factory(tasks: tuple[Any, ...]):
    """Adapt a pre-audited task tuple to the pilot factory call boundary."""

    def factory(
        *,
        split: str,
        master_seed: int,
        start: int,
        count: int,
        diagnostic_indices: tuple[int, ...],
    ) -> tuple[Any, ...]:
        del split, master_seed, diagnostic_indices
        selected = tasks[start : start + count]
        if len(selected) != count:
            raise ValueError("cached task request exceeds the audited pool")
        return selected

    return factory


def main() -> None:
    args = parse_args()
    for name in (
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "eval_batch_size",
        "attention_layers",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.tail_steps < 0:
        raise SystemExit("--tail-steps must be non-negative")
    if args.tail_steps > args.steps:
        raise SystemExit("--tail-steps cannot exceed --steps")
    for name in (
        "learning_rate",
        "tail_learning_rate",
        "max_grad_norm",
        "factor_temperature_start",
        "factor_temperature_end",
        "nll_threshold",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
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
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.sharpening_start_fraction < 1:
        raise SystemExit("--sharpening-start-fraction must lie in [0,1)")
    if args.batch_size > args.train_pool_tasks:
        raise SystemExit("--batch-size cannot exceed --train-pool-tasks")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    if args.model != "unstructured-64":
        if args.unstructured_head_kind != "low-rank":
            raise SystemExit(
                "--unstructured-head-kind is only configurable with "
                "--model unstructured-64"
            )
        if args.unstructured_head_rank is not None:
            raise SystemExit(
                "--unstructured-head-rank is only valid with --model unstructured-64"
            )
    elif args.unstructured_head_rank is not None:
        if args.unstructured_head_rank <= 0:
            raise SystemExit("--unstructured-head-rank must be positive")
        if args.unstructured_head_kind != "low-rank":
            raise SystemExit(
                "--unstructured-head-rank is only valid with "
                "--unstructured-head-kind low-rank"
            )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if args.model == "factorized-3x4":
        from prp_wm.discrete_causal_rules import ExpectedDiscreteCausalK4

        model_class = ExpectedDiscreteCausalK4
    else:
        from prp_wm.unstructured_causal_rules import UnstructuredDiscreteCausalK4

        model_class = UnstructuredDiscreteCausalK4
    from prp_wm.latent_rules import (
        outcome_map,
        rule_program_factor_ids,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _evaluate,
        _resolve_device,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    executor_path = args.executor_checkpoint.resolve()
    executor, executor_checkpoint = _load_audited_executor(
        torch,
        executor_path,
        device,
    )
    model_kwargs: dict[str, object] = {
        "attention_layers": args.attention_layers,
        "temperature": args.factor_temperature_start,
    }
    if args.model == "unstructured-64":
        model_kwargs["head_kind"] = args.unstructured_head_kind
        model_kwargs["head_rank"] = args.unstructured_head_rank
    model = model_class(executor, **model_kwargs).to(device)
    initial_checkpoint: dict[str, Any] | None = None
    initial_checkpoint_path: Path | None = None
    if args.initial_checkpoint is not None:
        initial_checkpoint_path = args.initial_checkpoint.resolve()
        initial_checkpoint = torch.load(
            initial_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if initial_checkpoint.get("model_type") != type(model).__name__:
            raise SystemExit("initial checkpoint has an incompatible model type")
        checkpoint_model = initial_checkpoint.get("model")
        if checkpoint_model is None:
            checkpoint_model = {
                "ExpectedDiscreteCausalK4": "factorized-3x4",
                "UnstructuredDiscreteCausalK4": "unstructured-64",
            }.get(initial_checkpoint.get("model_type"))
        if checkpoint_model != args.model:
            raise SystemExit("initial checkpoint used a different model family")
        if args.model == "unstructured-64":
            checkpoint_head_kind = initial_checkpoint.get("head_kind")
            if checkpoint_head_kind is None:
                # Checkpoints produced before explicit head-kind metadata used
                # the low-rank head exclusively.
                checkpoint_head_kind = "low-rank"
            if checkpoint_head_kind != model.head_kind:
                raise SystemExit(
                    "initial checkpoint used a different unstructured head kind"
                )
            checkpoint_head_rank = initial_checkpoint.get("head_rank")
            if checkpoint_head_rank is None and checkpoint_head_kind == "low-rank":
                state = initial_checkpoint.get("model_state_dict", {})
                bottleneck = state.get("rule_head.layers.1.weight")
                if bottleneck is not None:
                    checkpoint_head_rank = int(bottleneck.shape[0])
            if checkpoint_head_rank != model.head_rank:
                raise SystemExit(
                    "initial checkpoint used a different unstructured head rank"
                )
        if initial_checkpoint.get("context_fold") != args.context_fold:
            raise SystemExit("initial checkpoint used a different context fold")
        if (
            initial_checkpoint.get("executor_checkpoint_sha256")
            != _sha256_file(executor_path)
        ):
            raise SystemExit("initial checkpoint used a different executor")
        model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_pool = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split=args.train_split,
        master_seed=args.data_master_seed,
        diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
        count=args.train_pool_tasks,
        heldout=False,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
        context_fold=args.context_fold,
    )
    eval_pool = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split=args.eval_split,
        master_seed=args.data_master_seed,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.eval_tasks,
        heldout=True,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
        context_fold=args.context_fold,
    )
    seen_eval_pool = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split=args.eval_split,
        master_seed=args.data_master_seed,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=144,
        heldout=False,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
        context_fold=args.context_fold,
    )
    train_contexts = {
        _support_context_key(
            task,
            factor_ids_for_program=rule_program_factor_ids,
            version_space=version_space,
        )
        for task in train_pool
    }
    eval_contexts = {
        _support_context_key(
            task,
            factor_ids_for_program=rule_program_factor_ids,
            version_space=version_space,
        )
        for task in eval_pool
    }
    if train_contexts.intersection(eval_contexts):
        raise AssertionError("train and evaluation support contexts must be disjoint")
    if len(train_contexts) != 36 or len(eval_contexts) != 12:
        raise AssertionError("context split must contain 36 train and 12 eval contexts")
    eval_task_factory = _cached_task_factory(eval_pool)
    seen_eval_task_factory = _cached_task_factory(seen_eval_pool)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(args.seed + 1)
    sample_order = torch.randperm(len(train_pool), generator=sampler).tolist()
    sample_cursor = 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    run_config: dict[str, object] = {
        "experiment": (
            "expected_discrete_axis_structured_causal_k4"
            if args.model == "factorized-3x4"
            else "expected_discrete_unstructured_64way_causal_k4"
        ),
        "result_kind": "privileged_amortized_discrete_mechanism_ceiling",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": args.model,
        "posterior_parameterization": (
            "independent_3_axes_x_4_values"
            if args.model == "factorized-3x4"
            else "single_categorical_64_rules"
        ),
        "mechanism_axes_given": ["collision", "trigger", "relation"],
        "mechanism_value_labels_used_for_training": False,
        "program_labels_used_for_training": False,
        "true_query_targets_used_for_training": False,
        "support_derived_unordered_behavior_set_supervision": True,
        "all_64_integer_codes_evaluated_for_training": True,
        "discrete_cost_table_detached": True,
        "straight_through_decoder_gradient_used": False,
        "full_panel_bijective_assignment": True,
        "privileged_palette_canonicalization": True,
        "executor_frozen_and_eval": True,
        "executor_support_calibrated": True,
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_schema_version": executor_checkpoint[
            "checkpoint_schema_version"
        ],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_pool_tasks": args.train_pool_tasks,
        "unique_train_support_contexts": len(train_contexts),
        "unique_eval_support_contexts": len(eval_contexts),
        "context_fold": args.context_fold,
        "context_split_kind": (
            "legacy_diagonal" if args.context_fold is None else "latin_modulo_4"
        ),
        "train_support_contexts": [list(context) for context in sorted(train_contexts)],
        "eval_support_contexts": [list(context) for context in sorted(eval_contexts)],
        "context_generalization_split": (
            (
                "hold-out observed-axis value pairs with equal ordinal IDs"
                if args.context_fold is None
                else (
                    "hold-out observed-axis value pairs satisfying "
                    f"(value_1 + value_2) % 4 == {args.context_fold}"
                )
            )
            + "; all individual axis values remain in training"
        ),
        "train_eval_contexts_disjoint": True,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "tail_steps": args.tail_steps,
        "tail_learning_rate": args.tail_learning_rate,
        "tail_start_step": args.steps - args.tail_steps if args.tail_steps else None,
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
        "attention_layers": args.attention_layers,
        "nll_threshold": args.nll_threshold,
        "log_every": args.log_every,
        "train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "heldout_triple_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "model_config": asdict(model.config),
        "head_kind": getattr(model, "head_kind", None),
        "head_rank": getattr(model, "head_rank", None),
        "requested_unstructured_head_kind": (
            args.unstructured_head_kind
            if args.model == "unstructured-64"
            else None
        ),
        "requested_unstructured_head_rank": (
            args.unstructured_head_rank
            if args.model == "unstructured-64"
            else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "source_sha256": _source_sha256(),
        "initial_checkpoint": (
            str(initial_checkpoint_path) if initial_checkpoint_path is not None else None
        ),
        "initial_checkpoint_sha256": (
            _sha256_file(initial_checkpoint_path)
            if initial_checkpoint_path is not None
            else None
        ),
        "initial_training_steps": (
            int(initial_checkpoint["steps"])
            if initial_checkpoint is not None
            else 0
        ),
        "cumulative_training_steps": args.steps
        + (
            int(initial_checkpoint["steps"])
            if initial_checkpoint is not None
            else 0
        ),
    }

    latest: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            learning_rate = (
                args.tail_learning_rate
                if args.tail_steps and step >= args.steps - args.tail_steps
                else args.learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            factor_temperature = _linear_schedule(
                args.factor_temperature_start,
                args.factor_temperature_end,
                step,
                args.steps,
            )
            sharpening_weight = _sharpening_schedule(
                args.sharpening_weight_end,
                args.sharpening_start_fraction,
                step,
                args.steps,
            )
            if sample_cursor + args.batch_size > len(sample_order):
                sample_order = torch.randperm(
                    len(train_pool), generator=sampler
                ).tolist()
                sample_cursor = 0
            task_indices = sample_order[sample_cursor : sample_cursor + args.batch_size]
            sample_cursor += args.batch_size
            tasks = tuple(train_pool[index] for index in task_indices)
            batch = rulegrid_tasks_to_canonical_behavior_batch(
                tasks,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.losses(
                batch,
                validity_weight=args.validity_weight,
                diversity_weight=args.diversity_weight,
                sharpening_weight=sharpening_weight,
                proper_weight=args.proper_weight,
                balanced_weight=args.balanced_weight,
                assignment_temperature=args.assignment_temperature,
                temperature=factor_temperature,
            )
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            unique_codes = sum(
                int(torch.unique(ids, dim=0).shape[0])
                for ids in loss.inference.factor_ids
            ) / batch.batch_size
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "factor_temperature": factor_temperature,
                "sharpening_weight": sharpening_weight,
                "learning_rate": learning_rate,
                "mean_unique_factor_tuples": unique_codes,
            }
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record: dict[str, object] = {
                    "step": completed,
                    "tasks_seen": completed * args.batch_size,
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    training_seconds = time.perf_counter() - started
    checkpoint: dict[str, object] = {
        **run_config,
        "model_type": type(model).__name__,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "latest_training_metrics": latest,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    common_evaluation = {
        "torch": torch,
        "model": model,
        "device": device,
        "split": args.eval_split,
        "data_master_seed": args.data_master_seed,
        "task_count": args.eval_tasks,
        "batch_size": args.eval_batch_size,
        "nll_threshold": args.nll_threshold,
        "factor_temperature": args.factor_temperature_end,
        "make_pilot_tasks": eval_task_factory,
        "make_behavior_batch": rulegrid_tasks_to_canonical_behavior_batch,
        "outcome_map": outcome_map,
        "triple_indices": TRIPLE_DIAGNOSTIC_INDICES,
        "rule_program_factor_ids": rule_program_factor_ids,
        "version_space": version_space,
    }
    evaluation = _evaluate(**common_evaluation, support_ablation="none")
    seen_evaluation = _evaluate(
        **(
            common_evaluation
            | {
                "task_count": len(seen_eval_pool),
                "make_pilot_tasks": seen_eval_task_factory,
            }
        ),
        support_ablation="none",
    )
    shuffled = _evaluate(
        **common_evaluation,
        support_ablation="shuffle-targets",
    )
    coverage = evaluation["coverage_at_4_mass_weighted"]
    comparisons: dict[str, object] = {}
    for name, path in (
        ("continuous_latent_k4", args.continuous_result),
        ("straight_through_causal_k4", args.straight_through_result),
        ("explicit_64_rule_filter", args.explicit_filter_result),
    ):
        if path is not None:
            comparisons[name] = _comparison(path, coverage)
    gate = bool(
        coverage >= 0.90
        and evaluation["all_classes_covered_task_rate"] >= 0.90
        and evaluation["factor_tuple_coverage_at_4"] >= 0.90
        and evaluation["all_particles_support_exact_task_rate"] >= 0.90
    )
    result: dict[str, object] = {
        **run_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "tasks_seen": args.steps * args.batch_size,
        "cumulative_tasks_seen": (
            args.steps
            + (
                int(initial_checkpoint["steps"])
                if initial_checkpoint is not None
                else 0
            )
        )
        * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "heldout_triple_coverage": evaluation,
        "seen_context_triple_coverage": seen_evaluation,
        "shuffled_support_target_control": shuffled,
        "comparisons": comparisons,
        "comparison_note": (
            "Historical baselines used all 48 canonical contexts during "
            "training; their coverage is contextual reference only, not an "
            "apples-to-apples unseen-context result."
        ),
        "static_gate": {
            "coverage_at_4_gte": 0.90,
            "all_classes_covered_task_rate_gte": 0.90,
            "factor_tuple_coverage_at_4_gte": 0.90,
            "all_particles_support_exact_task_rate_gte": 0.90,
            "passed": gate,
        },
        "interpretation": (
            "A pass shows that the explicit discrete rule search can be "
            "amortized into support-only inference when credit assignment uses "
            "detached costs at exact mechanism codes. It does not establish "
            "autonomous discovery of the mechanism axes or palette roles."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
