#!/usr/bin/env python3
"""Delta-calibrate the privileged factor executor on active evidence prefixes.

The original support-calibrated executor is an exact verifier on the six
public support transitions, but the sequential assimilation screen appends
three new transition types that were not part of that calibration domain:

``support(6) -> partial(7) -> neutral(8) -> strong(9)``.

This runner warm-starts the audited support-calibrated checkpoint and trains a
new, independent checkpoint.  Factor codes remain privileged, fixed inputs.
For each public state/action panel, simulator targets may be generated under
any of the 64 codes.  The default balanced schedule covers all 64 codes across
an eight-task training batch without expanding every task under every code.

The held-out audit is deliberately exhaustive.  At t0, t1, t2, and t3 it
scores all 64 codes, takes the neural MAP-exact bank, and compares that set to
the symbolic ``version_space``.  This is a privileged verifier ceiling, not a
claim about learning factor axes, palette roles, or an active query policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
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


DEFAULT_INITIAL_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs/support_calibrated_executor_seed20260724/checkpoint_last.pt"
)
CHECKPOINT_SCHEMA_VERSION = (
    "prp-wm.active-support-calibrated-factor-executor.v1"
)
STAGE_NAMES = ("t0_support", "t1_partial", "t2_neutral", "t3_strong")
_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_rulegrid_executor_ceiling.py",
    "scripts/run_support_calibrated_executor.py",
    "scripts/run_gram_smc_active_screen.py",
    "scripts/run_causal_mechanism_coverage.py",
    "scripts/run_active_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--codes-per-task",
        type=int,
        default=8,
        help=(
            "Uniformly rotated factor codes per training task. With the default "
            "8x8 batch schedule every step covers all 64 codes."
        ),
    )
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--tail-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--diagnostic-loss-weight", type=float, default=0.40)
    parser.add_argument(
        "--stage-loss-weights",
        type=float,
        nargs=4,
        default=(0.10, 0.15, 0.10, 0.25),
        metavar=("T0", "T1", "T2", "T3"),
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="active-executor-delta-train")
    parser.add_argument("--eval-split", default="active-executor-prefix-heldout")
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


def _validate_args(args: argparse.Namespace) -> tuple[float, tuple[float, ...]]:
    for name in (
        "steps",
        "batch_size",
        "codes_per_task",
        "eval_tasks",
        "eval_batch_size",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.codes_per_task > 64:
        raise SystemExit("--codes-per-task cannot exceed 64")
    if args.tail_steps < 0 or args.tail_steps > args.steps:
        raise SystemExit("--tail-steps must lie in [0, steps]")
    for name in ("learning_rate", "tail_learning_rate", "max_grad_norm"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("weight_decay", "balanced_weight", "diagnostic_loss_weight"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    stage_weights = tuple(float(value) for value in args.stage_loss_weights)
    if len(stage_weights) != 4 or min(stage_weights) < 0:
        raise SystemExit("--stage-loss-weights needs four non-negative values")
    total = float(args.diagnostic_loss_weight) + sum(stage_weights)
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("diagnostic and stage loss weights must sum to one")
    if args.seed < 0 or args.data_master_seed < 0:
        raise SystemExit("seeds must be non-negative")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    if any(not split or "/" in split for split in (args.train_split, args.eval_split)):
        raise SystemExit("split names must be non-empty and slash-free")
    return float(args.diagnostic_loss_weight), stage_weights


def _active_histories(tasks: Sequence[Any]) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    """Materialize true factual histories for the fixed partial/neutral/strong path."""

    from prp_wm.rulegrid import RuleGridTransition, version_space
    from scripts.run_gram_smc_active_screen import _select_fixed_evidence

    materialized = tuple(tasks)
    if not materialized:
        raise ValueError("tasks cannot be empty")
    current = [list(task.inference.support) for task in materialized]
    stages: list[tuple[tuple[Any, ...], ...]] = [
        tuple(tuple(history) for history in current)
    ]
    selected = tuple(_select_fixed_evidence(task) for task in materialized)
    for step_index in range(3):
        for task_index, task in enumerate(materialized):
            step = selected[task_index][step_index]
            target = task.privileged.active_targets[step.candidate_index]
            current[task_index].append(
                RuleGridTransition(step.probe.state, step.probe.action, target)
            )
        stages.append(tuple(tuple(history) for history in current))

    for task_index, task in enumerate(materialized):
        spaces = tuple(
            version_space(stage[task_index], task.privileged.palette)
            for stage in stages
        )
        if len(spaces[0]) != 4:
            raise AssertionError("t0 support must leave exactly four rules")
        if len(spaces[1]) not in (1, 3):
            raise AssertionError("t1 partial evidence must create a 1+3 split")
        if spaces[2] != spaces[1]:
            raise AssertionError("t2 neutral evidence must preserve the version space")
        if spaces[3] != (task.privileged.true_program,):
            raise AssertionError("t3 strong evidence must identify the true rule")
    return tuple(stages)


def _diagnostic_panels(
    tasks: Sequence[Any], indices: Sequence[int]
) -> tuple[tuple[Any, ...], ...]:
    normalized = tuple(indices)
    if not normalized:
        raise ValueError("diagnostic indices cannot be empty")
    return tuple(
        tuple(task.inference.diagnostics[index] for index in normalized)
        for task in tasks
    )


def _scheduled_program_rows(
    tasks: Sequence[Any],
    *,
    step: int,
    codes_per_task: int,
) -> tuple[tuple[Any, ...], ...]:
    """Return a deterministic balanced code schedule for one task batch.

    Contiguous, disjoint code blocks cover all 64 codes whenever
    ``len(tasks) * codes_per_task >= 64``.  The coprime step rotation changes
    the task/code pairing without changing global balance.
    """

    from prp_wm.rulegrid import ALL_PROGRAMS

    materialized = tuple(tasks)
    if not materialized:
        raise ValueError("tasks cannot be empty")
    if step < 0:
        raise ValueError("step must be non-negative")
    if not 1 <= codes_per_task <= len(ALL_PROGRAMS):
        raise ValueError("codes_per_task must lie in 1..64")
    offset = (step * 17) % len(ALL_PROGRAMS)
    return tuple(
        tuple(
            ALL_PROGRAMS[
                (offset + task_index * codes_per_task + code_index)
                % len(ALL_PROGRAMS)
            ]
            for code_index in range(codes_per_task)
        )
        for task_index in range(len(materialized))
    )


def _canonicalize_grid_tensor(
    torch: Any, raw: Any, tasks: Sequence[Any]
) -> Any:
    """Apply the same privileged palette-role lookup as the legacy runner."""

    from prp_wm.rulegrid import NUM_COLORS

    materialized = tuple(tasks)
    if raw.shape[0] != len(materialized):
        raise ValueError("grid tensor and task batch axes must match")
    rows: list[list[int]] = []
    for task in materialized:
        lookup = list(range(NUM_COLORS))
        for canonical_id, field in enumerate(fields(task.privileged.palette), start=1):
            lookup[getattr(task.privileged.palette, field.name)] = canonical_id
        rows.append(lookup)
    lookup_tensor = torch.tensor(rows, dtype=torch.long, device=raw.device)
    return lookup_tensor.gather(1, raw.reshape(len(materialized), -1)).reshape_as(raw)


def _program_conditioned_panel_batch(
    *,
    torch: Any,
    tasks: Sequence[Any],
    panels: Sequence[Sequence[Any]],
    program_rows: Sequence[Sequence[Any]],
    device: Any,
) -> Any:
    """Build a privileged batch with code-specific simulator supervision.

    ``panels`` may contain either transitions or probes; only their public
    ``state`` and ``action`` fields are consumed.  Existing observed targets
    are intentionally ignored and regenerated for each supplied factor code.
    """

    from prp_wm.latent_rules import OracleFactorBatch, rule_program_factor_ids
    from prp_wm.rulegrid import RuleGridTransition, simulate
    from scripts.run_gram_smc_active_screen import _support_batch
    from scripts.run_support_calibrated_executor import _repeat_batch_axis

    materialized_tasks = tuple(tasks)
    materialized_panels = tuple(tuple(panel) for panel in panels)
    materialized_rows = tuple(tuple(row) for row in program_rows)
    if not materialized_tasks or len(materialized_tasks) != len(materialized_panels):
        raise ValueError("tasks and panels must be non-empty and equally sized")
    if len(materialized_rows) != len(materialized_tasks):
        raise ValueError("program rows must match the task batch")
    panel_counts = {len(panel) for panel in materialized_panels}
    code_counts = {len(row) for row in materialized_rows}
    if len(panel_counts) != 1 or not next(iter(panel_counts)):
        raise ValueError("all public panels must have the same positive length")
    if len(code_counts) != 1 or not next(iter(code_counts)):
        raise ValueError("all program rows must have the same positive length")
    codes_per_task = next(iter(code_counts))

    # Reuse the existing arbitrary-history tensorizer for canonical states and
    # correctly padded atomic/composite public actions.  The placeholder target
    # is used only to satisfy the transition container and is replaced below.
    placeholder_histories: list[tuple[Any, ...]] = []
    for task, panel in zip(materialized_tasks, materialized_panels, strict=True):
        true_program = task.privileged.true_program
        placeholder_histories.append(
            tuple(
                RuleGridTransition(
                    item.state,
                    item.action,
                    simulate(
                        item.state,
                        item.action,
                        true_program,
                        task.privileged.palette,
                    ),
                )
                for item in panel
            )
        )
    public = _support_batch(
        torch,
        materialized_tasks,
        tuple(placeholder_histories),
        device=device,
    )

    raw_targets = torch.tensor(
        [
            [
                [
                    simulate(
                        item.state,
                        item.action,
                        program,
                        task.privileged.palette,
                    )
                    for item in panel
                ]
                for program in row
            ]
            for task, panel, row in zip(
                materialized_tasks,
                materialized_panels,
                materialized_rows,
                strict=True,
            )
        ],
        dtype=torch.long,
        device=device,
    )
    targets = _canonicalize_grid_tensor(torch, raw_targets, materialized_tasks)
    factor_ids = torch.tensor(
        [
            [rule_program_factor_ids(program) for program in row]
            for row in materialized_rows
        ],
        dtype=torch.long,
        device=device,
    ).reshape(len(materialized_tasks) * codes_per_task, 3)
    return OracleFactorBatch(
        states=_repeat_batch_axis(public.support_states, codes_per_task),
        actions=_repeat_batch_axis(public.support_actions, codes_per_task),
        targets=targets.reshape(
            len(materialized_tasks) * codes_per_task,
            next(iter(panel_counts)),
            *targets.shape[-2:],
        ),
        factor_ids=factor_ids,
        action_mask=_repeat_batch_axis(public.support_action_mask, codes_per_task),
        palette_canonicalized=True,
    )


def _bank_set_counts(
    *,
    scores: Any,
    tasks: Sequence[Any],
    histories: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Compare a neural MAP-exact factor bank with the symbolic version space."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    if scores.batch_size != len(tasks) or len(tasks) != len(histories):
        raise ValueError("scores, tasks, and histories must have equal batch size")
    factor_codes = tuple(
        tuple(int(value) for value in row)
        for row in scores.factor_ids.detach().cpu().tolist()
    )
    exact_mask = scores.map_exact.detach().cpu()
    set_equal = 0
    true_exact = 0
    false_positives = 0
    false_negatives = 0
    neural_sizes: Counter[int] = Counter()
    symbolic_sizes: Counter[int] = Counter()
    for task_index, (task, history) in enumerate(
        zip(tasks, histories, strict=True)
    ):
        neural = {
            factor_codes[index]
            for index in exact_mask[task_index].nonzero(as_tuple=False).flatten().tolist()
        }
        symbolic = {
            rule_program_factor_ids(program)
            for program in version_space(history, task.privileged.palette)
        }
        true_code = rule_program_factor_ids(task.privileged.true_program)
        set_equal += int(neural == symbolic)
        true_exact += int(true_code in neural)
        false_positives += len(neural - symbolic)
        false_negatives += len(symbolic - neural)
        neural_sizes[len(neural)] += 1
        symbolic_sizes[len(symbolic)] += 1
    return {
        "set_equal": set_equal,
        "true_exact": true_exact,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "neural_sizes": neural_sizes,
        "symbolic_sizes": symbolic_sizes,
    }


def _active_prefix_evaluation(
    *,
    torch: Any,
    model: Any,
    device: Any,
    split: str,
    data_master_seed: int,
    task_count: int,
    batch_size: int,
    make_pilot_tasks: Any,
    score_hypothesis_bank: Any,
) -> dict[str, object]:
    totals = {
        name: {
            "set_equal": 0,
            "true_exact": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "neural_sizes": Counter(),
            "symbolic_sizes": Counter(),
        }
        for name in STAGE_NAMES
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, task_count, batch_size):
            count = min(batch_size, task_count - start)
            tasks = make_pilot_tasks(
                split=split,
                master_seed=data_master_seed,
                start=start,
                count=count,
                diagnostic_indices=(0,),
            )
            stages = _active_histories(tasks)
            from scripts.run_gram_smc_active_screen import _support_batch

            for stage_name, histories in zip(STAGE_NAMES, stages, strict=True):
                public = _support_batch(torch, tasks, histories, device=device)
                scores = score_hypothesis_bank(
                    model,
                    public.support_states,
                    public.support_actions,
                    public.support_targets,
                    public.support_mask,
                    public.support_action_mask,
                )
                counts = _bank_set_counts(
                    scores=scores,
                    tasks=tasks,
                    histories=histories,
                )
                row = totals[stage_name]
                for key in (
                    "set_equal",
                    "true_exact",
                    "false_positives",
                    "false_negatives",
                ):
                    row[key] += counts[key]
                row["neural_sizes"].update(counts["neural_sizes"])
                row["symbolic_sizes"].update(counts["symbolic_sizes"])

    stages_result: dict[str, object] = {}
    for stage_index, name in enumerate(STAGE_NAMES):
        row = totals[name]
        stages_result[name] = {
            "tasks": task_count,
            "observed_transitions": 6 + stage_index,
            "neural_map_exact_bank_equals_symbolic_version_space_task_rate": (
                row["set_equal"] / task_count
            ),
            "true_rule_map_exact_task_rate": row["true_exact"] / task_count,
            "mean_false_positive_codes": row["false_positives"] / task_count,
            "mean_false_negative_codes": row["false_negatives"] / task_count,
            "neural_map_exact_bank_size_histogram": {
                str(size): count for size, count in sorted(row["neural_sizes"].items())
            },
            "symbolic_version_space_size_histogram": {
                str(size): count
                for size, count in sorted(row["symbolic_sizes"].items())
            },
        }
    return {
        "tasks": task_count,
        "all_64_factor_codes_scored_per_task": True,
        "stages": stages_result,
    }


def _active_executor_gate(
    prefix_evaluation: dict[str, object],
    diagnostic_evaluations: Sequence[dict[str, object]],
) -> dict[str, object]:
    stages = prefix_evaluation["stages"]
    stage_checks = {
        name: {
            "set_equality_gte_0_99": (
                stages[name][
                    "neural_map_exact_bank_equals_symbolic_version_space_task_rate"
                ]
                >= 0.99
            ),
            "true_rule_exact_eq_1": (
                stages[name]["true_rule_map_exact_task_rate"] == 1.0
            ),
        }
        for name in STAGE_NAMES
    }
    diagnostic_exact = all(
        evaluation["exact_task_accuracy"] == 1.0
        for evaluation in diagnostic_evaluations
    )
    passed = diagnostic_exact and all(
        all(checks.values()) for checks in stage_checks.values()
    )
    return {
        "requirements": {
            "each_stage_neural_symbolic_set_equality_task_rate_gte": 0.99,
            "each_stage_true_rule_map_exact_task_rate_eq": 1.0,
            "single_pair_heldout_triple_diagnostic_exact_task_accuracy_eq": 1.0,
        },
        "stage_checks": stage_checks,
        "diagnostic_exact": diagnostic_exact,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    diagnostic_weight, stage_weights = _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from prp_wm.causal_filter import score_hypothesis_bank
    from prp_wm.latent_rules import outcome_map
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from scripts.run_causal_mechanism_coverage import _load_executor
    from scripts.run_rulegrid_executor_ceiling import (
        _configure_determinism,
        _evaluate,
        _resolve_device,
        _runtime_identity,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    initial_path = args.initial_checkpoint.resolve()
    model, initial_checkpoint = _load_executor(torch, initial_path, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    panel_lengths = {
        "diagnostic_nontriple": len(NONTRIPLE_DIAGNOSTIC_INDICES),
        **{name: 6 + index for index, name in enumerate(STAGE_NAMES)},
    }
    factor_examples_per_step = (
        args.batch_size * args.codes_per_task * sum(panel_lengths.values())
    )
    run_config: dict[str, object] = {
        "experiment": "active_prefix_support_calibrated_oracle_factor_executor",
        "result_kind": "privileged_active_history_verifier_ceiling",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": "OracleFactorExecutor",
        "privileged_true_rule_factors_are_model_inputs": True,
        "privileged_palette_canonicalization": True,
        "palette_input": "oracle-canonical",
        "factor_code_is_fixed_per_transition_panel": True,
        "all_64_factor_codes_available_for_supervision": True,
        "training_code_schedule": {
            "kind": "deterministic-coprime-rotated-contiguous-blocks",
            "codes_per_task": args.codes_per_task,
            "codes_per_step": args.batch_size * args.codes_per_task,
            "all_64_codes_covered_each_step": (
                args.batch_size * args.codes_per_task >= 64
            ),
        },
        "train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "heldout_triple_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "active_prefix_sequence": list(STAGE_NAMES),
        "active_prefix_lengths": [6, 7, 8, 9],
        "active_candidate_selection": [
            "first-partial",
            "first-neutral-large-change",
            "first-strong",
        ],
        "delta_training": {
            "initial_checkpoint_path": str(initial_path),
            "initial_checkpoint_sha256": _sha256_file(initial_path),
            "initial_checkpoint_schema_version": initial_checkpoint.get(
                "checkpoint_schema_version"
            ),
            "steps": args.steps,
            "task_draws": args.steps * args.batch_size,
            "program_conditioned_transition_examples_per_step": (
                factor_examples_per_step
            ),
            "program_conditioned_transition_examples_total": (
                args.steps * factor_examples_per_step
            ),
        },
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "tail_steps": args.tail_steps,
        "tail_learning_rate": args.tail_learning_rate,
        "weight_decay": args.weight_decay,
        "balanced_weight": args.balanced_weight,
        "max_grad_norm": args.max_grad_norm,
        "domain_loss_weights": {
            "diagnostic_nontriple": diagnostic_weight,
            **{
                name: weight
                for name, weight in zip(STAGE_NAMES, stage_weights, strict=True)
            },
        },
        "model_config": initial_checkpoint["model_config"],
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
    }

    latest: dict[str, float] = {}
    started = time.perf_counter()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            learning_rate = (
                args.tail_learning_rate
                if args.tail_steps and step >= args.steps - args.tail_steps
                else args.learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            tasks = make_pilot_tasks(
                split=args.train_split,
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
            )
            histories = _active_histories(tasks)
            programs = _scheduled_program_rows(
                tasks,
                step=step,
                codes_per_task=args.codes_per_task,
            )
            domains = (
                (
                    "diagnostic_nontriple",
                    _diagnostic_panels(tasks, NONTRIPLE_DIAGNOSTIC_INDICES),
                    diagnostic_weight,
                ),
                *tuple(
                    (name, panel, weight)
                    for name, panel, weight in zip(
                        STAGE_NAMES,
                        histories,
                        stage_weights,
                        strict=True,
                    )
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            domain_losses: dict[str, float] = {}
            total_value = 0.0
            for name, panel, weight in domains:
                batch = _program_conditioned_panel_batch(
                    torch=torch,
                    tasks=tasks,
                    panels=panel,
                    program_rows=programs,
                    device=device,
                )
                losses = model.losses(batch, balanced_weight=args.balanced_weight)
                weighted = weight * losses.total
                if not bool(torch.isfinite(weighted).item()):
                    raise RuntimeError(f"non-finite {name} loss at step {step + 1}")
                weighted.backward()
                value = float(losses.total.detach().cpu())
                domain_losses[f"loss_{name}"] = value
                total_value += weight * value
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                .detach()
                .cpu()
            )
            optimizer.step()
            latest = {
                "loss_total": total_value,
                **domain_losses,
                "gradient_norm": gradient_norm,
            }
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record = {
                    "step": completed,
                    "learning_rate": learning_rate,
                    "tasks_seen": completed * args.batch_size,
                    "program_conditioned_transition_examples_seen": (
                        completed * factor_examples_per_step
                    ),
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    checkpoint: dict[str, object] = {
        **run_config,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "latest_training_metrics": latest,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    common_diagnostic_evaluation = {
        "torch": torch,
        "model": model,
        "device": device,
        "split": args.eval_split,
        "data_master_seed": args.data_master_seed,
        "task_count": args.eval_tasks,
        "batch_size": args.eval_batch_size,
        "make_pilot_tasks": make_pilot_tasks,
        "make_factor_batch": __import__(
            "prp_wm.latent_rules", fromlist=["rulegrid_tasks_to_oracle_factor_batch"]
        ).rulegrid_tasks_to_oracle_factor_batch,
        "outcome_map": outcome_map,
        "canonicalize_palette": True,
    }
    single = _evaluate(
        **common_diagnostic_evaluation, diagnostic_indices=tuple(range(12))
    )
    pair = _evaluate(
        **common_diagnostic_evaluation, diagnostic_indices=tuple(range(12, 21))
    )
    triple = _evaluate(
        **common_diagnostic_evaluation,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
    )
    prefix = _active_prefix_evaluation(
        torch=torch,
        model=model,
        device=device,
        split=args.eval_split,
        data_master_seed=args.data_master_seed,
        task_count=args.eval_tasks,
        batch_size=args.eval_batch_size,
        make_pilot_tasks=make_pilot_tasks,
        score_hypothesis_bank=score_hypothesis_bank,
    )
    gate = _active_executor_gate(prefix, (single, pair, triple))
    result: dict[str, object] = {
        **run_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tasks_seen": args.steps * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "single_evaluation": single,
        "pair_evaluation": pair,
        "heldout_triple_evaluation": triple,
        "heldout_active_prefix_evaluation": prefix,
        "active_prefix_executor_gate": gate,
        "interpretation": (
            "A pass establishes only an oracle-code, oracle-palette neural verifier "
            "for the fixed support/partial/neutral/strong protocol. The new checkpoint "
            "is independent of the GRAM proposal checkpoint and does not establish "
            "autonomous rule discovery or active action selection."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
