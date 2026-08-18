#!/usr/bin/env python3
"""Train a persistent stratified GRAM adapter from public version spaces.

The legacy GRAM checkpoint and executor are frozen.  The only optimized
parameters are a small support-conditioned logit correction, three ambiguity
gates, and three positive anchor gains.  Training targets are the complete set
of four codes that exactly reproduce public t0 support under the frozen
executor; no selected program, query target, or behavior panel is consumed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


CHECKPOINT_SCHEMA_VERSION = "prp-wm.persistent-stratified-gram-k4.v1"
RESULT_SCHEMA_VERSION = "prp-wm.persistent-stratified-gram-public-coverage.v1"
DEFAULT_GRAM = REPOSITORY_ROOT / (
    "runs/gram_causal_screen600_fold0_seed20260728/checkpoint_last.pt"
)
_AUDITED_SOURCE_FILES = (
    "prp_wm/stratified_gram.py",
    "prp_wm/gram_causal_rules.py",
    "prp_wm/discrete_causal_rules.py",
    "prp_wm/causal_rules.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_stratified_gram_public_coverage.py",
    "scripts/run_gram_public_coverage_finetune.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-gram-checkpoint", type=Path, default=DEFAULT_GRAM)
    parser.add_argument("--executor-checkpoint", type=Path)
    parser.add_argument("--context-fold", type=int, choices=range(4))
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-pool-tasks", type=int, default=144)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--tail-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--initial-anchor-gain", type=float, default=4.0)
    parser.add_argument(
        "--legacy-logit-mode",
        choices=("residual", "replace"),
        default="residual",
        help=(
            "add the adapter to frozen GRAM factor logits, or replace those logits "
            "while retaining the frozen public-support representation"
        ),
    )
    parser.add_argument("--margin-weight", type=float, default=0.10)
    parser.add_argument("--ambiguity-weight", type=float, default=0.10)
    parser.add_argument("--validity-weight", type=float, default=0.0)
    parser.add_argument("--joint-margin", type=float, default=1.0)
    parser.add_argument("--deep-supervision-decay", type=float, default=1.0)
    parser.add_argument("--factor-temperature", type=float, default=1.0)
    parser.add_argument(
        "--inference-widths",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32),
    )
    parser.add_argument("--log-every", type=int, default=50)
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
    encoded: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            encoded[key] = str(value.resolve())
        elif isinstance(value, (tuple, list)):
            encoded[key] = list(value)
        else:
            encoded[key] = value
    return encoded


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
        "initial_anchor_gain",
        "joint_margin",
        "deep_supervision_decay",
        "factor_temperature",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "margin_weight",
        "ambiguity_weight",
        "validity_weight",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size > args.train_pool_tasks:
        raise SystemExit("--batch-size cannot exceed --train-pool-tasks")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    widths = tuple(sorted(set(args.inference_widths)))
    if not widths or any(width <= 0 or width > 32 for width in widths):
        raise SystemExit("--inference-widths must lie in [1,32]")
    if 4 not in widths:
        raise SystemExit("deterministic W4 is the mandatory primary gate")
    return widths


def _load_initial_proposal(torch: Any, args: Any, device: Any):
    from prp_wm.stratified_gram import PersistentStratifiedGRAMProposal
    from scripts.run_gram_public_coverage_finetune import _load_warm_start

    legacy, initial, initial_path, executor_path, executor_metadata = _load_warm_start(
        torch,
        args,
        device,
    )
    proposal = PersistentStratifiedGRAMProposal(
        legacy,
        initial_anchor_gain=float(args.initial_anchor_gain),
        legacy_logit_mode=str(args.legacy_logit_mode),
    ).to(device)
    names = [name for name, _ in proposal.adapter_named_parameters()]
    if not names or any(name.startswith("legacy.") for name in names):
        raise AssertionError("trainable boundary must contain only the adapter")
    return proposal, initial, initial_path, executor_path, executor_metadata


def load_stratified_checkpoint(
    torch: Any,
    checkpoint_path: Path,
    *,
    device: Any,
    executor_checkpoint: Path | None = None,
):
    """Load a completed adapter while re-verifying its frozen base identity."""

    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("unexpected stratified GRAM checkpoint schema")
    base_path = Path(checkpoint["initial_gram_checkpoint"]).resolve()
    if _sha256_file(base_path) != checkpoint["initial_gram_checkpoint_sha256"]:
        raise SystemExit("initial GRAM checkpoint SHA256 drifted")
    args = SimpleNamespace(
        initial_gram_checkpoint=base_path,
        executor_checkpoint=executor_checkpoint,
        context_fold=int(checkpoint["context_fold"]),
        initial_anchor_gain=float(checkpoint["initial_anchor_gain"]),
        legacy_logit_mode=str(checkpoint.get("legacy_logit_mode", "residual")),
    )
    proposal, initial, _, executor_path, executor_metadata = _load_initial_proposal(
        torch,
        args,
        device,
    )
    proposal.load_state_dict(checkpoint["model_state_dict"], strict=True)
    proposal.eval()
    return proposal, checkpoint, initial, executor_path, executor_metadata


def _static_public_audit(
    *,
    torch: Any,
    model: Any,
    tasks: Sequence[Any],
    batch_size: int,
    widths: tuple[int, ...],
    device: Any,
    make_public_batch: Any,
    compatible_mask_for_tasks: Any | None = None,
) -> dict[str, object]:
    """Measure fresh proposals before any verifier ranking or true-code lookup."""

    factor_value_counts = {
        width: [[0] * 4 for _ in range(3)] for width in widths
    }
    totals = {
        width: {
            "recall": 0.0,
            "all_four": 0,
            "valid_particles": 0,
            "particles": 0,
            "unique": 0,
            "axis_recall": [0.0, 0.0, 0.0],
            "axis_tasks": [0, 0, 0],
            "context_records": {},
        }
        for width in widths
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[start : start + batch_size]
            batch = make_public_batch(torch, batch_tasks, device=device)
            if compatible_mask_for_tasks is None:
                compatible_mask = model.public_support_exact_mask(batch).cpu()
            else:
                compatible_mask = compatible_mask_for_tasks(
                    torch,
                    model,
                    batch_tasks,
                    device=device,
                ).detach().cpu()
            if not torch.all(compatible_mask.sum(dim=-1) == 4):
                raise AssertionError("static audit requires four compatible codes")
            compatible_codes = [
                {
                    tuple(int(value) for value in row)
                    for row in model.factor_bank[mask.to(model.factor_bank.device)]
                    .detach()
                    .cpu()
                    .tolist()
                }
                for mask in compatible_mask
            ]
            varying_axes = []
            for codes in compatible_codes:
                varying = [
                    axis
                    for axis in range(3)
                    if len({code[axis] for code in codes}) == 4
                ]
                if len(varying) != 1:
                    raise AssertionError("public K4 set must have one varying axis")
                varying_axes.append(varying[0])
            for width in widths:
                inference = model.sample_width_candidates(
                    batch,
                    width=width,
                    recursive_steps=model.recursive_steps,
                    temperature=1.0,
                    sample_noise=False,
                )
                for task_index, row in enumerate(inference.factor_ids.cpu().tolist()):
                    proposals = [tuple(int(value) for value in code) for code in row]
                    unique = set(proposals)
                    compatible = compatible_codes[task_index]
                    covered = len(unique.intersection(compatible))
                    axis = varying_axes[task_index]
                    record = totals[width]
                    record["recall"] += covered / 4
                    record["all_four"] += int(covered == 4)
                    record["valid_particles"] += sum(
                        proposal in compatible for proposal in proposals
                    )
                    record["particles"] += width
                    record["unique"] += len(unique)
                    record["axis_recall"][axis] += covered / 4
                    record["axis_tasks"][axis] += 1
                    context_key = (axis, tuple(sorted(compatible)))
                    context_record = record["context_records"].setdefault(
                        context_key,
                        {
                            "varying_axis": axis,
                            "compatible_codes": [list(code) for code in sorted(compatible)],
                            "tasks": 0,
                            "recall_sum": 0.0,
                            "all_four_count": 0,
                        },
                    )
                    context_record["tasks"] += 1
                    context_record["recall_sum"] += covered / 4
                    context_record["all_four_count"] += int(covered == 4)
                    for code in proposals:
                        for factor_axis, value in enumerate(code):
                            factor_value_counts[width][factor_axis][value] += 1

    points: list[dict[str, object]] = []
    for width in widths:
        record = totals[width]
        axis_recall = [
            record["axis_recall"][axis] / record["axis_tasks"][axis]
            for axis in range(3)
        ]
        point = {
            "width": width,
            "tasks": len(tasks),
            "deterministic_persistent_proposals": True,
            "fixed_anchor_bank": hasattr(model, "anchor_bank"),
            "version_space_recall": record["recall"] / len(tasks),
            "all_four_task_rate": record["all_four"] / len(tasks),
            "valid_particle_rate": record["valid_particles"] / record["particles"],
            "mean_unique_joint_codes": record["unique"] / len(tasks),
            "recall_by_varying_axis": axis_recall,
            "worst_axis_recall": min(axis_recall),
            "factor_value_counts": factor_value_counts[width],
            "any_empirical_zero_factor_value": any(
                count == 0
                for axis_counts in factor_value_counts[width]
                for count in axis_counts
            ),
            "context_records": [
                {
                    "varying_axis": context["varying_axis"],
                    "compatible_codes": context["compatible_codes"],
                    "tasks": context["tasks"],
                    "mean_recall": context["recall_sum"] / context["tasks"],
                    "all_four_task_rate": (
                        context["all_four_count"] / context["tasks"]
                    ),
                }
                for _, context in sorted(record["context_records"].items())
            ],
        }
        points.append(point)
    w4 = next(point for point in points if point["width"] == 4)
    checks = {
        "version_space_recall_gte_0_90": w4["version_space_recall"] >= 0.90,
        "all_four_task_rate_gte_0_75": w4["all_four_task_rate"] >= 0.75,
        "valid_particle_rate_gte_0_90": w4["valid_particle_rate"] >= 0.90,
        "mean_unique_joint_codes_gte_3_8": w4["mean_unique_joint_codes"] >= 3.8,
        "worst_axis_recall_gte_0_85": w4["worst_axis_recall"] >= 0.85,
        "no_empirical_zero_factor_value": not w4["any_empirical_zero_factor_value"],
    }
    return {
        "primary_gate_width": 4,
        "compatible_set_source": (
            "model_public_support_exact_mask"
            if compatible_mask_for_tasks is None
            else getattr(
                compatible_mask_for_tasks,
                "__name__",
                type(compatible_mask_for_tasks).__name__,
            )
        ),
        "proposal_uses_true_code_or_version_space": False,
        "version_space_used_after_proposal_for_metrics_only": True,
        "points": points,
        "w4_gate": {"checks": checks, "passed": all(checks.values())},
    }


def main() -> None:
    args = parse_args()
    widths = _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_expected_discrete_causal_coverage import _build_context_pool
    from scripts.run_gram_public_coverage_finetune import (
        _audit_pool_masks,
        _public_support_batch,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    model, initial, initial_path, executor_path, executor_metadata = (
        _load_initial_proposal(torch, args, device)
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
        raise AssertionError("train and held-out contexts overlap")
    if len(train_contexts) != 36 or len(eval_contexts) != 12:
        raise AssertionError("expected 36 train and 12 held-out contexts")

    before = _static_public_audit(
        torch=torch,
        model=model,
        tasks=eval_pool,
        batch_size=args.eval_batch_size,
        widths=widths,
        device=device,
        make_public_batch=_public_support_batch,
    )
    trainable_named = model.adapter_named_parameters()
    trainable = [parameter for _, parameter in trainable_named]
    trainable_names = [name for name, _ in trainable_named]
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
                    len(train_pool), generator=sampler
                ).tolist()
                sample_cursor = 0
            indices = sample_order[sample_cursor : sample_cursor + args.batch_size]
            sample_cursor += args.batch_size
            tasks = tuple(train_pool[index] for index in indices)
            batch = _public_support_batch(torch, tasks, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.hard_public_version_space_loss(
                batch,
                margin_weight=args.margin_weight,
                ambiguity_weight=args.ambiguity_weight,
                validity_weight=args.validity_weight,
                joint_margin=args.joint_margin,
                deep_supervision_decay=args.deep_supervision_decay,
                temperature=args.factor_temperature,
                sample_noise=False,
            )
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.max_grad_norm,
                ).detach().cpu()
            )
            optimizer.step()
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
                "anchor_gain": [
                    float(value) for value in model.anchor_gain.detach().cpu().tolist()
                ],
                "mean_ambiguity_probability": [
                    float(value)
                    for value in loss.ambiguity_probabilities.detach().mean(dim=0).cpu().tolist()
                ],
            }
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record = {"step": completed, "tasks_seen": completed * args.batch_size, **latest}
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    training_seconds = time.perf_counter() - started
    after = _static_public_audit(
        torch=torch,
        model=model,
        tasks=eval_pool,
        batch_size=args.eval_batch_size,
        widths=widths,
        device=device,
        make_public_batch=_public_support_batch,
    )
    checkpoint: dict[str, object] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": type(model).__name__,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "initial_gram_checkpoint": str(initial_path),
        "initial_gram_checkpoint_sha256": _sha256_file(initial_path),
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "context_fold": context_fold,
        "initial_anchor_gain": args.initial_anchor_gain,
        "legacy_logit_mode": model.legacy_logit_mode,
        "anchor_bank": model.anchor_bank.detach().cpu().tolist(),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "latest_training_metrics": latest,
        "cli_arguments": _json_cli(args),
        "source_sha256": _source_sha256(),
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment": "persistent_stratified_gram_public_version_space",
        "status": "complete",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "cli_arguments": _json_cli(args),
        "context_fold": context_fold,
        "training_seed": args.seed,
        "training_steps": args.steps,
        "training_tasks_seen": args.steps * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "device": str(device),
        "legacy_gram_frozen": True,
        "legacy_logit_mode": model.legacy_logit_mode,
        "executor_frozen": True,
        "posterior_or_behavior_supervision_used": False,
        "query_or_true_program_used_for_training": False,
        "public_version_space_teacher": "all-64 frozen-executor MAP exact equality",
        "proposal_sees_version_space_at_inference": False,
        "training_sample_noise": False,
        "persistent_anchor_scope": "same anchors across recursion, tasks, and stages",
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "initial_gram_checkpoint": str(initial_path),
        "initial_gram_checkpoint_sha256": _sha256_file(initial_path),
        "initial_gram_checkpoint_schema": initial.get("checkpoint_schema_version"),
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_schema": executor_metadata.get("checkpoint_schema_version"),
        "train_public_version_space_audit": train_balance,
        "eval_public_version_space_audit": eval_balance,
        "static_audit_before": before,
        "static_audit_after": after,
        "latest_training_metrics": latest,
        "source_sha256": _source_sha256(),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
