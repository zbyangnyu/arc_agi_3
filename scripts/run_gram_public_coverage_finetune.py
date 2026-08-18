#!/usr/bin/env python3
"""Public-support-only GRAM version-space coverage continuation.

This runner warm-starts an existing GRAM checkpoint and directly fine-tunes
its public prior.  For every t0 support history, a frozen executor evaluates
all 64 integer factor codes; the four codes whose MAP grids exactly equal the
observed public outcomes form an unordered target set.  No true program,
diagnostic target, behavior panel, or training posterior enters the loss.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


ARCHITECTURE_CHECKPOINT_SCHEMA_VERSION = "prp-wm.gram-factorized-causal-k4.v1"
CONTINUATION_SCHEMA_VERSION = "prp-wm.gram-public-version-coverage.v1"
_AUDITED_SOURCE_FILES = (
    "prp_wm/gram_causal_rules.py",
    "prp_wm/discrete_causal_rules.py",
    "prp_wm/causal_rules.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_gram_public_coverage_finetune.py",
    "scripts/run_gram_causal_screen.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-gram-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--executor-checkpoint",
        type=Path,
        help="override the path, but not the audited SHA256, recorded by GRAM",
    )
    parser.add_argument("--context-fold", type=int, choices=range(4))
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--train-pool-tasks", type=int, default=144)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--tail-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "prior-head-only"),
        default="all",
        help=(
            "update the full public-prior proposer or only its Gaussian prior_head; "
            "the executor and behavior-conditioned posterior always remain frozen"
        ),
    )
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--axis-balance-weight", type=float, default=0.10)
    parser.add_argument("--validity-weight", type=float, default=0.10)
    parser.add_argument("--assignment-temperature", type=float, default=0.05)
    parser.add_argument("--deep-supervision-decay", type=float, default=0.5)
    parser.add_argument("--factor-temperature", type=float, default=1.0)
    parser.add_argument(
        "--inference-widths",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32),
        metavar="W",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="gram-causal-train")
    parser.add_argument("--eval-split", default="gram-causal-composition")
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


def _json_cli(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.resolve())
        elif isinstance(value, (tuple, list)):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _pad_public_actions(
    torch: Any,
    action_rows: Sequence[Sequence[Any]],
    *,
    device: Any,
) -> tuple[Any, Any | None]:
    if not action_rows or not action_rows[0]:
        raise ValueError("public action panel cannot be empty")
    count = len(action_rows[0])
    if any(len(row) != count for row in action_rows):
        raise ValueError("all tasks must have the same support length")
    max_atoms = max(action.shape[0] for row in action_rows for action in row)
    padded = torch.zeros(
        len(action_rows),
        count,
        max_atoms,
        4,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        len(action_rows),
        count,
        max_atoms,
        dtype=torch.bool,
        device=device,
    )
    for task_index, row in enumerate(action_rows):
        for step, action in enumerate(row):
            atoms = action.shape[0]
            padded[task_index, step, :atoms] = action.to(device)
            mask[task_index, step, :atoms] = True
    if max_atoms == 1:
        return padded[:, :, 0], None
    return padded, mask


def _raw_public_history_batch(
    torch: Any,
    histories: Sequence[Sequence[Any]],
    *,
    device: Any,
) -> Any:
    """Materialize equal-length observed histories without privileged fields."""

    from prp_wm.neural import RuleGridTensorBatch, encode_public_action

    materialized = tuple(tuple(history) for history in histories)
    if not materialized:
        raise ValueError("at least one history is required")
    steps = len(materialized[0])
    if steps <= 0 or any(len(history) != steps for history in materialized):
        raise ValueError("all public histories must have the same positive length")
    supports = materialized
    action_rows = tuple(
        tuple(encode_public_action(transition.action) for transition in support)
        for support in supports
    )
    actions, action_mask = _pad_public_actions(
        torch,
        action_rows,
        device=device,
    )
    return RuleGridTensorBatch(
        support_states=torch.tensor(
            [[transition.state for transition in support] for support in supports],
            dtype=torch.long,
            device=device,
        ),
        support_actions=actions,
        support_targets=torch.tensor(
            [
                [transition.next_state for transition in support]
                for support in supports
            ],
            dtype=torch.long,
            device=device,
        ),
        support_mask=torch.ones(
            len(materialized),
            steps,
            dtype=torch.bool,
            device=device,
        ),
        support_action_mask=action_mask,
    )


def _raw_public_support_batch_from_views(
    torch: Any,
    inference_views: Sequence[Any],
    *,
    device: Any,
) -> Any:
    """Materialize the fixed six-transition t0 prefix from controller views."""

    materialized = tuple(inference_views)
    supports = tuple(view.support[:6] for view in materialized)
    if any(len(support) != 6 for support in supports):
        raise ValueError("public coverage requires all six t0 support transitions")
    return _raw_public_history_batch(torch, supports, device=device)


def _public_support_batch(
    torch: Any,
    tasks: Sequence[Any],
    *,
    device: Any,
) -> Any:
    """Materialize support, then apply the explicit oracle palette transform."""

    from prp_wm.latent_rules import canonicalize_rulegrid_tensor_batch

    materialized = tuple(tasks)
    raw = _raw_public_support_batch_from_views(
        torch,
        tuple(task.inference for task in materialized),
        device=device,
    )
    return canonicalize_rulegrid_tensor_batch(raw, materialized)


def _context_from_codes(codes: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    normalized = tuple(tuple(int(value) for value in code) for code in codes)
    if len(normalized) != 4 or len(set(normalized)) != 4:
        raise ValueError("a t0 public version space must contain four distinct codes")
    varying = [
        axis
        for axis in range(3)
        if {code[axis] for code in normalized} == set(range(4))
    ]
    if len(varying) != 1:
        raise ValueError("compatible codes must vary on exactly one four-valued axis")
    heldout_axis = varying[0]
    representative = min(normalized)
    fixed = tuple(
        representative[axis] for axis in range(3) if axis != heldout_axis
    )
    if any(
        code[axis] != representative[axis]
        for code in normalized
        for axis in range(3)
        if axis != heldout_axis
    ):
        raise ValueError("non-heldout axes must be fixed by public support")
    return heldout_axis, fixed[0], fixed[1]


def _mask_balance_audit(torch: Any, factor_bank: Any, masks: Any) -> dict[str, object]:
    if masks.ndim != 2 or masks.shape[1] != factor_bank.shape[0]:
        raise ValueError("masks must have [N,64] shape")
    if not torch.all(masks.sum(dim=-1) == 4):
        raise ValueError("every public compatibility mask must contain four codes")
    contexts: Counter[tuple[int, int, int]] = Counter()
    joint_counts: Counter[tuple[int, int, int]] = Counter()
    axis_value_counts = [[0 for _ in range(4)] for _ in range(3)]
    bank_cpu = factor_bank.detach().cpu()
    for row in masks.detach().cpu():
        codes = [
            tuple(int(value) for value in code)
            for code in bank_cpu[row].tolist()
        ]
        contexts[_context_from_codes(codes)] += 1
        for code in codes:
            joint_counts[code] += 1
            for axis, value in enumerate(code):
                axis_value_counts[axis][value] += 1
    flat_axis_counts = [value for row in axis_value_counts for value in row]
    return {
        "tasks": int(masks.shape[0]),
        "compatible_codes_per_task": 4,
        "unique_public_contexts": len(contexts),
        "context_multiplicity_histogram": {
            str(key): value
            for key, value in sorted(Counter(contexts.values()).items())
        },
        "contexts": [list(context) for context in sorted(contexts)],
        "axis_value_counts": axis_value_counts,
        "axis_value_exactly_balanced": min(flat_axis_counts) == max(flat_axis_counts),
        "joint_code_union_size": len(joint_counts),
        "joint_frequency_histogram": {
            str(key): value
            for key, value in sorted(Counter(joint_counts.values()).items())
        },
        "missing_joint_codes": [
            list(code)
            for code in sorted(
                set(tuple(int(v) for v in row) for row in bank_cpu.tolist())
                - set(joint_counts)
            )
        ],
    }


def _audit_pool_masks(
    torch: Any,
    model: Any,
    tasks: Sequence[Any],
    *,
    batch_size: int,
    device: Any,
) -> tuple[Any, dict[str, object]]:
    masks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            batch = _public_support_batch(
                torch,
                tasks[start : start + batch_size],
                device=device,
            )
            masks.append(model.public_support_exact_mask(batch).cpu())
    combined = torch.cat(masks, dim=0)
    return combined, _mask_balance_audit(torch, model.factor_bank.cpu(), combined)


def _load_warm_start(torch: Any, args: argparse.Namespace, device: Any):
    from prp_wm.gram_causal_rules import GRAMFactorizedCausalK4
    from scripts.run_expected_discrete_causal_coverage import _load_audited_executor

    initial_path = args.initial_gram_checkpoint.resolve()
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if initial.get("checkpoint_schema_version") != ARCHITECTURE_CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("unexpected initial GRAM checkpoint schema")
    if initial.get("model_type") != "GRAMFactorizedCausalK4":
        raise SystemExit("initial checkpoint is not GRAMFactorizedCausalK4")
    checkpoint_fold = initial.get("context_fold")
    if checkpoint_fold is None:
        raise SystemExit("coverage continuation requires a Latin context-fold checkpoint")
    if args.context_fold is not None and args.context_fold != checkpoint_fold:
        raise SystemExit("--context-fold must match the initial GRAM checkpoint")
    executor_path = (
        Path(initial["executor_checkpoint"])
        if args.executor_checkpoint is None
        else args.executor_checkpoint.resolve()
    )
    executor_path = executor_path.resolve()
    expected_executor_sha = initial.get("executor_checkpoint_sha256")
    if expected_executor_sha is None or _sha256_file(executor_path) != expected_executor_sha:
        raise SystemExit("executor SHA256 does not match the initial GRAM checkpoint")
    executor, executor_metadata = _load_audited_executor(
        torch,
        executor_path,
        device,
    )
    cli = initial.get("cli_arguments", {})
    bounds = initial.get("guidance_log_variance_bounds", (-8.0, 4.0))
    model = GRAMFactorizedCausalK4(
        executor,
        recursive_steps=int(initial["recursive_steps"]),
        guidance_dim=int(initial["guidance_dim"]),
        attention_layers=int(cli.get("attention_layers", 2)),
        temperature=float(cli.get("factor_temperature_end", 1.0)),
        minimum_log_variance=float(bounds[0]),
        maximum_log_variance=float(bounds[1]),
        initial_log_variance=float(
            initial.get("initial_guidance_log_variance", -2.0)
        ),
        truncate_between_steps=bool(
            initial.get("truncate_between_recursive_steps", True)
        ),
    ).to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    # These modules belong exclusively to the behavior-conditioned posterior;
    # freezing them makes the public-only optimization boundary mechanical.
    for module in (
        model.behavior_item_context,
        model.posterior_head,
        model.posterior_behavior_to_guidance,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, initial, initial_path, executor_path, executor_metadata


def _configure_trainable_scope(
    model: Any,
    scope: str,
) -> tuple[list[Any], list[str]]:
    """Apply and audit the requested continuation parameter boundary."""

    if scope not in {"all", "prior-head-only"}:
        raise ValueError(f"unknown trainable scope: {scope!r}")
    if scope == "prior-head-only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.prior_head.parameters():
            parameter.requires_grad_(True)
    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named:
        raise RuntimeError("trainable scope selected no model parameters")
    if scope == "prior-head-only" and any(
        not name.startswith("prior_head.") for name, _ in trainable_named
    ):
        raise AssertionError("prior-head-only scope exposed a non-prior parameter")
    if any(name.startswith("executor.") for name, _ in trainable_named):
        raise AssertionError("the frozen executor entered the optimizer scope")
    if any(
        name.startswith("posterior_head.")
        or name.startswith("posterior_behavior_to_guidance.")
        or name.startswith("behavior_item_context.")
        for name, _ in trainable_named
    ):
        raise AssertionError("the behavior-conditioned posterior entered the optimizer scope")
    return (
        [parameter for _, parameter in trainable_named],
        [name for name, _ in trainable_named],
    )


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    for name in (
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "eval_batch_size",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.tail_steps < 0 or args.tail_steps > args.steps:
        raise SystemExit("--tail-steps must lie in [0,steps]")
    for name in (
        "learning_rate",
        "tail_learning_rate",
        "max_grad_norm",
        "deep_supervision_decay",
        "factor_temperature",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "coverage_weight",
        "axis_balance_weight",
        "validity_weight",
        "assignment_temperature",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size > args.train_pool_tasks:
        raise SystemExit("--batch-size cannot exceed --train-pool-tasks")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    if any(width <= 0 for width in args.inference_widths):
        raise SystemExit("--inference-widths must be positive")
    return tuple(sorted(set(args.inference_widths)))


def main() -> None:
    args = parse_args()
    widths = _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from prp_wm.latent_rules import (
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
        _resolve_device,
    )
    from scripts.run_expected_discrete_causal_coverage import _build_context_pool
    from scripts.run_gram_causal_screen import _evaluate_inference_width_curve

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    model, initial, initial_path, executor_path, executor_metadata = _load_warm_start(
        torch,
        args,
        device,
    )
    context_fold = int(initial["context_fold"])
    pool_arguments = {
        "make_pilot_tasks": make_pilot_tasks,
        "master_seed": args.data_master_seed,
        "factor_ids_for_program": rule_program_factor_ids,
        "version_space": version_space,
        "context_fold": context_fold,
    }
    train_pool = _build_context_pool(
        **pool_arguments,
        split=args.train_split,
        diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
        count=args.train_pool_tasks,
        heldout=False,
    )
    eval_pool = _build_context_pool(
        **pool_arguments,
        split=args.eval_split,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.eval_tasks,
        heldout=True,
    )
    _, train_balance = _audit_pool_masks(
        torch,
        model,
        train_pool,
        batch_size=args.eval_batch_size,
        device=device,
    )
    _, eval_balance = _audit_pool_masks(
        torch,
        model,
        eval_pool,
        batch_size=args.eval_batch_size,
        device=device,
    )
    train_contexts = {tuple(row) for row in train_balance["contexts"]}
    eval_contexts = {tuple(row) for row in eval_balance["contexts"]}
    if train_contexts.intersection(eval_contexts):
        raise AssertionError("train and held-out public support contexts overlap")
    if len(train_contexts) != 36 or len(eval_contexts) != 12:
        raise AssertionError("expected 36 train and 12 held-out public contexts")
    if not train_balance["axis_value_exactly_balanced"]:
        raise AssertionError("training public version spaces are not axis/value balanced")

    evaluation_arguments = {
        "torch": torch,
        "model": model,
        "device": device,
        "tasks": eval_pool,
        "batch_size": args.eval_batch_size,
        "widths": widths,
        "recursive_steps": model.recursive_steps,
        "factor_temperature": args.factor_temperature,
        "sample_noise": True,
        "inference_seed": args.seed + 20_000,
        "proper_weight": 1.0,
        "balanced_weight": 1.0,
        "make_behavior_batch": rulegrid_tasks_to_canonical_behavior_batch,
        "triple_indices": TRIPLE_DIAGNOSTIC_INDICES,
        "rule_program_factor_ids": rule_program_factor_ids,
        "version_space": version_space,
    }
    before = _evaluate_inference_width_curve(**evaluation_arguments)

    trainable, trainable_names = _configure_trainable_scope(
        model,
        args.trainable_scope,
    )
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(args.seed + 1)
    sample_order = torch.randperm(len(train_pool), generator=sampler).tolist()
    sample_cursor = 0
    latest: dict[str, object] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            learning_rate = (
                args.tail_learning_rate
                if args.tail_steps and step >= args.steps - args.tail_steps
                else args.learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if sample_cursor + args.batch_size > len(sample_order):
                sample_order = torch.randperm(
                    len(train_pool),
                    generator=sampler,
                ).tolist()
                sample_cursor = 0
            indices = sample_order[sample_cursor : sample_cursor + args.batch_size]
            sample_cursor += args.batch_size
            tasks = tuple(train_pool[index] for index in indices)
            batch = _public_support_batch(torch, tasks, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.coverage_losses(
                batch,
                coverage_weight=args.coverage_weight,
                axis_balance_weight=args.axis_balance_weight,
                validity_weight=args.validity_weight,
                assignment_temperature=args.assignment_temperature,
                deep_supervision_decay=args.deep_supervision_decay,
                temperature=args.factor_temperature,
                sample_noise=True,
            )
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite public coverage loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.max_grad_norm,
                ).detach().cpu()
            )
            optimizer.step()
            final_ids = loss.trajectories.factor_ids[-1]
            final_bank_indices = (
                16 * final_ids[..., 0]
                + 4 * final_ids[..., 1]
                + final_ids[..., 2]
            )
            sampled_compatible = loss.compatible_mask.gather(
                1,
                final_bank_indices,
            )
            mean_unique = sum(
                int(torch.unique(row, dim=0).shape[0]) for row in final_ids
            ) / batch.batch_size
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
                "factor_temperature": args.factor_temperature,
                "mean_unique_factor_tuples": mean_unique,
                "sampled_compatible_trajectory_rate": float(
                    sampled_compatible.float().mean().detach().cpu()
                ),
                "recursive_step_objectives": [
                    float(value) for value in loss.step_objectives.detach().cpu()
                ],
                "recursive_step_coverage": [
                    float(value) for value in loss.step_coverage.detach().cpu()
                ],
                "recursive_step_axis_balance": [
                    float(value) for value in loss.step_axis_balance.detach().cpu()
                ],
                "recursive_step_invalid_mass": [
                    float(value) for value in loss.step_invalid_mass.detach().cpu()
                ],
                "deep_supervision_weights": [
                    float(value)
                    for value in loss.deep_supervision_weights.detach().cpu()
                ],
            }
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record = {
                    "step": completed,
                    "tasks_seen": completed * args.batch_size,
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    training_seconds = time.perf_counter() - started
    model.eval()
    after = _evaluate_inference_width_curve(**evaluation_arguments)
    cli_arguments = _json_cli(args) | {
        # Retain the architecture-loader fields used by downstream screens.
        "attention_layers": int(
            initial.get("cli_arguments", {}).get("attention_layers", 2)
        ),
        "factor_temperature_end": args.factor_temperature,
    }
    run_config: dict[str, object] = {
        "experiment": "gram_public_version_space_coverage_finetune",
        "result_kind": "public_support_only_prior_coverage_continuation",
        # Architecture is unchanged, so existing GRAM inference loaders remain valid.
        "checkpoint_schema_version": ARCHITECTURE_CHECKPOINT_SCHEMA_VERSION,
        "continuation_schema_version": CONTINUATION_SCHEMA_VERSION,
        "model_type": type(model).__name__,
        "cli_arguments": cli_arguments,
        "training_objective": "public_t0_exact_four_code_permutation_coverage",
        "coverage_target_source": (
            "all-64 frozen-executor MAP equality with observed public t0 support"
        ),
        "true_program_labels_used_for_training": False,
        "behavior_or_query_targets_used_for_training": False,
        "training_posterior_used": False,
        "public_prior_trained_directly": True,
        "trainable_scope": args.trainable_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_tensor_count": len(trainable_names),
        "trainable_parameter_count": trainable_parameter_count,
        "training_trajectory_width": 4,
        "training_noise": "iid Gaussian public-prior trajectories",
        "iid_w4_all_four_coupon_collector_ceiling": 0.09375,
        "privileged_palette_canonicalization": True,
        "all_64_integer_codes_evaluated_for_training": True,
        "compatibility_uses_nll_threshold": False,
        "executor_frozen_and_eval": True,
        "unused_posterior_modules_frozen": True,
        "context_fold": context_fold,
        "context_split_kind": "latin_modulo_4",
        "train_eval_contexts_disjoint": True,
        "train_public_version_space_audit": train_balance,
        "eval_public_version_space_audit": eval_balance,
        "recursive_steps": model.recursive_steps,
        "guidance_dim": model.guidance_dim,
        "truncate_between_recursive_steps": model.truncate_between_steps,
        "guidance_log_variance_bounds": [
            model.minimum_log_variance,
            model.maximum_log_variance,
        ],
        "initial_guidance_log_variance": model.initial_log_variance,
        # Preserve the original model seed for paired fixed-K4 compatibility.
        "model_seed": initial.get("model_seed"),
        "continuation_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "model_config": asdict(model.config),
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_schema_version": executor_metadata.get(
            "checkpoint_schema_version"
        ),
        "initial_gram_checkpoint": str(initial_path),
        "initial_gram_checkpoint_sha256": _sha256_file(initial_path),
        "initial_gram_steps": int(initial.get("steps", 0)),
        "continuation_steps": args.steps,
        "source_sha256": _source_sha256(),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    checkpoint: dict[str, object] = {
        **run_config,
        "steps": int(initial.get("steps", 0)) + args.steps,
        "tasks_seen": int(initial.get("tasks_seen", 0)) + args.steps * args.batch_size,
        "continuation_tasks_seen": args.steps * args.batch_size,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "latest_training_metrics": latest,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    result: dict[str, object] = {
        **run_config,
        "steps": checkpoint["steps"],
        "continuation_tasks_seen": checkpoint["continuation_tasks_seen"],
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "inference_width_curve_before": before,
        "inference_width_curve_after": after,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": trainable_parameter_count,
        "interpretation": (
            "This isolates whether direct public-prior full-version-space coverage "
            "repairs GRAM proposal support. W=4 trajectories remain exchangeable iid; "
            "deterministic all-four W4 coverage is not an architectural expectation."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
