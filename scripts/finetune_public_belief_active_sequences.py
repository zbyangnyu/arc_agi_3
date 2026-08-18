#!/usr/bin/env python3
"""Fine-tune a public K4 rule belief on online active-candidate sequences.

The simulator-side candidate kind is used only to construct a balanced training
curriculum.  Model inference receives public ``(state, action, next_state)``
history and the controller-owned ``is_agent_probe_result`` bit; candidate kind,
palette, active target, and true program never enter the model input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


RESULT_SCHEMA_VERSION = "prp-wm.public-belief-active-sequence-finetune-run.v1"
FINETUNE_SCHEMA_VERSION = "prp-wm.public-belief-active-sequence-finetune.v1"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / (
    "runs/raw_palette_invariant_atom_matched_hard600_fold0_seed20260863/"
    "checkpoint_last.pt"
)
TRAIN_POOL_TASKS = 144
HELDOUT_POOL_TASKS = 48
EXPECTED_TRAIN_CONTEXTS = 36
EXPECTED_HELDOUT_CONTEXTS = 12
CURRICULUM = (
    "t0",
    "single_strong",
    "single_partial",
    "single_neutral",
    "random_length_2",
    "public_symmetry_break",
    "random_length_4",
    "public_symmetry_break_replay",
    "random_length_8",
    "strong_partial_neutral",
)
LEGACY_PUBLIC_PHASES = frozenset(
    {"public_symmetry_break", "public_symmetry_break_replay"}
)
SOURCE_FILES = (
    "prp_wm/public_version_k4.py",
    "prp_wm/rulegrid.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "scripts/finetune_public_belief_active_sequences.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_gram_public_coverage_finetune.py",
    "scripts/run_public_version_space_k4.py",
)


@dataclass(frozen=True)
class SampledControllerBatch:
    """Public histories plus evaluator-only sampling provenance."""

    controller_histories: tuple[Any, ...]
    candidate_kind_sequences: tuple[tuple[str, ...], ...]

    @property
    def histories(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(item.transitions for item in self.controller_histories)

    @property
    def candidate_kind_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for sequence in self.candidate_kind_sequences:
            counts.update(sequence)
        return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--tail-steps", type=int, default=80)
    parser.add_argument("--tail-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--task-consistency-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260871)
    parser.add_argument("--data-master-seed", type=int)
    parser.add_argument("--train-split")
    parser.add_argument("--eval-split")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-tasks-per-audit",
        type=int,
        default=HELDOUT_POOL_TASKS,
        help=(
            "held-out tasks used per sequence audit; the complete 48-task, "
            "12-context pool is still constructed and verified"
        ),
    )
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--color-permutation-augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: inherit the source checkpoint setting",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not 0 < args.batch_size <= TRAIN_POOL_TASKS:
        raise SystemExit(f"--batch-size must be in [1,{TRAIN_POOL_TASKS}]")
    if args.learning_rate <= 0 or args.tail_learning_rate <= 0:
        raise SystemExit("learning rates must be positive")
    if args.tail_steps < 0:
        raise SystemExit("--tail-steps must be non-negative")
    if args.tail_steps > args.steps:
        raise SystemExit("--tail-steps cannot exceed --steps")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative")
    if args.max_grad_norm <= 0:
        raise SystemExit("--max-grad-norm must be positive")
    if args.task_consistency_weight < 0:
        raise SystemExit("--task-consistency-weight must be non-negative")
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive")
    if not 1 <= args.eval_tasks_per_audit <= HELDOUT_POOL_TASKS:
        raise SystemExit(
            f"--eval-tasks-per-audit must be in [1,{HELDOUT_POOL_TASKS}]"
        )
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    return {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in SOURCE_FILES
    }


def _json_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _choice(torch: Any, indices: Sequence[int], *, generator: Any) -> int:
    materialized = tuple(int(index) for index in indices)
    if not materialized:
        raise AssertionError("candidate-kind partition was unexpectedly empty")
    offset = int(
        torch.randint(len(materialized), (1,), generator=generator).item()
    )
    return materialized[offset]


def _candidate_indices_for_phase(
    torch: Any,
    task: Any,
    phase: str,
    *,
    generator: Any,
) -> tuple[int, ...]:
    """Select candidates using private kind labels without returning labels."""

    kinds = task.privileged.candidate_kinds
    candidate_count = len(task.inference.active_candidates)
    if candidate_count != 8 or len(kinds) != candidate_count:
        raise AssertionError("the active curriculum expects an eight-probe bank")
    partitions = {
        "strong": tuple(index for index, kind in enumerate(kinds) if kind == "strong"),
        "partial": tuple(index for index, kind in enumerate(kinds) if kind == "partial"),
        "neutral-large-change": tuple(
            index
            for index, kind in enumerate(kinds)
            if kind == "neutral-large-change"
        ),
    }
    if {key: len(value) for key, value in partitions.items()} != {
        "strong": 2,
        "partial": 2,
        "neutral-large-change": 4,
    }:
        raise AssertionError("unexpected RuleGrid active candidate partition")
    if phase == "t0":
        return ()
    if phase == "single_strong":
        return (_choice(torch, partitions["strong"], generator=generator),)
    if phase == "single_partial":
        return (_choice(torch, partitions["partial"], generator=generator),)
    if phase == "single_neutral":
        return (
            _choice(
                torch,
                partitions["neutral-large-change"],
                generator=generator,
            ),
        )
    if phase.startswith("random_length_"):
        length = int(phase.rsplit("_", 1)[1])
        order = torch.randperm(candidate_count, generator=generator).tolist()
        return tuple(int(index) for index in order[:length])
    if phase == "strong_partial_neutral":
        return (
            _choice(torch, partitions["strong"], generator=generator),
            _choice(torch, partitions["partial"], generator=generator),
            _choice(
                torch,
                partitions["neutral-large-change"],
                generator=generator,
            ),
        )
    raise ValueError(f"unknown curriculum phase: {phase}")


def _sample_controller_batch(
    torch: Any,
    tasks: Sequence[Any],
    phase: str,
    *,
    generator: Any,
) -> SampledControllerBatch:
    """Execute a resettable-probe sequence and retain only public outcomes."""

    from prp_wm.rulegrid import RuleGridTransition
    from scripts.run_public_version_space_k4 import (
        ControllerHistory,
        _active_break_controller_history,
        _informative_then_replay_controller_history,
    )

    if phase in LEGACY_PUBLIC_PHASES:
        builder = (
            _active_break_controller_history
            if phase == "public_symmetry_break"
            else _informative_then_replay_controller_history
        )
        controller_histories = tuple(builder(task) for task in tasks)
        marker = (
            "public-constructed-symmetry-break"
            if phase == "public_symmetry_break"
            else "public-constructed-symmetry-break-replay"
        )
        return SampledControllerBatch(
            controller_histories,
            tuple((marker,) for _ in controller_histories),
        )

    controller_histories = []
    kind_sequences = []
    for task in tasks:
        selected = _candidate_indices_for_phase(
            torch,
            task,
            phase,
            generator=generator,
        )
        if len(set(selected)) != len(selected):
            raise AssertionError("an online sequence repeated a candidate")
        transitions = list(task.inference.support[:6])
        for candidate_index in selected:
            probe = task.inference.active_candidates[candidate_index]
            # The target is an environment result.  It is copied into the
            # public observation and is never supplied as a separate field.
            transitions.append(
                RuleGridTransition(
                    probe.state,
                    probe.action,
                    task.privileged.active_targets[candidate_index],
                )
            )
        controller_histories.append(
            ControllerHistory(
                tuple(transitions),
                (False,) * 6 + (True,) * len(selected),
            )
        )
        kind_sequences.append(
            tuple(task.privileged.candidate_kinds[index] for index in selected)
        )
    return SampledControllerBatch(
        tuple(controller_histories),
        tuple(kind_sequences),
    )


def _make_public_t0_batch(
    torch: Any,
    tasks: Sequence[Any],
    *,
    device: Any,
) -> Any:
    from scripts.run_gram_public_coverage_finetune import _raw_public_history_batch

    return _raw_public_history_batch(
        torch,
        tuple(tuple(task.inference.support[:6]) for task in tasks),
        device=device,
    )


def _audit_sequences(
    *,
    torch: Any,
    model: Any,
    tasks_by_phase: dict[str, Sequence[Any]],
    sampled: dict[str, SampledControllerBatch],
    batch_size: int,
    device: Any,
) -> dict[str, object]:
    from scripts.run_public_version_space_k4 import _static_factor_belief_audit

    reports: dict[str, object] = {}
    for phase in CURRICULUM:
        tasks = tuple(tasks_by_phase[phase])
        sequence = sampled[phase]
        report = _static_factor_belief_audit(
            torch=torch,
            model=model,
            tasks=tuple(tasks),
            controller_histories=sequence.controller_histories,
            batch_size=batch_size,
            device=device,
            make_support_batch=_make_public_t0_batch,
        )
        report["candidate_kind_used_for_evaluator_sampling_only"] = True
        report["candidate_kind_in_model_input"] = False
        report["sampled_candidate_kind_counts"] = sequence.candidate_kind_counts
        reports[phase] = report
    return reports


def _build_pools(
    *,
    torch: Any,
    model: Any,
    context_fold: int,
    data_master_seed: int,
    train_split: str,
    eval_split: str,
    device: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, object]]:
    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.rulegrid import version_space
    from scripts.run_expected_discrete_causal_coverage import _build_context_pool
    from scripts.run_gram_public_coverage_finetune import _mask_balance_audit
    from scripts.run_public_version_space_k4 import _symbolic_version_space_mask

    common = {
        "make_pilot_tasks": make_pilot_tasks,
        "master_seed": data_master_seed,
        "factor_ids_for_program": rule_program_factor_ids,
        "version_space": version_space,
        "context_fold": context_fold,
    }
    train_pool = _build_context_pool(
        **common,
        split=train_split,
        diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
        count=TRAIN_POOL_TASKS,
        heldout=False,
    )
    heldout_pool = _build_context_pool(
        **common,
        split=eval_split,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=HELDOUT_POOL_TASKS,
        heldout=True,
    )
    train_mask = _symbolic_version_space_mask(
        torch,
        model,
        train_pool,
        device=device,
    )
    heldout_mask = _symbolic_version_space_mask(
        torch,
        model,
        heldout_pool,
        device=device,
    )
    train_audit = _mask_balance_audit(
        torch,
        model.factor_bank.detach().cpu(),
        train_mask.detach().cpu(),
    )
    heldout_audit = _mask_balance_audit(
        torch,
        model.factor_bank.detach().cpu(),
        heldout_mask.detach().cpu(),
    )
    train_contexts = {tuple(row) for row in train_audit["contexts"]}
    heldout_contexts = {tuple(row) for row in heldout_audit["contexts"]}
    if len(train_contexts) != EXPECTED_TRAIN_CONTEXTS:
        raise AssertionError("training pool did not contain 36 contexts")
    if len(heldout_contexts) != EXPECTED_HELDOUT_CONTEXTS:
        raise AssertionError("held-out pool did not contain 12 contexts")
    if train_contexts.intersection(heldout_contexts):
        raise AssertionError("train and held-out contexts overlap")
    return train_pool, heldout_pool, {
        "context_fold": context_fold,
        "train": train_audit,
        "heldout": heldout_audit,
        "train_heldout_contexts_disjoint": True,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch

    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_gram_public_coverage_finetune import _raw_public_history_batch
    from scripts.run_public_version_space_k4 import (
        FACTOR_BELIEF_HEADS,
        _conditional_probe_innovation_targets,
        _controller_probe_result_mask,
        _permute_raw_support_colors,
        _symmetry_expanded_version_space_mask,
        load_public_version_k4_checkpoint,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    checkpoint_path = args.checkpoint.resolve()
    source_checkpoint_sha256 = _sha256_file(checkpoint_path)
    model, source_checkpoint, executor_path, executor_metadata = (
        load_public_version_k4_checkpoint(
            torch,
            checkpoint_path,
            device=device,
        )
    )
    if source_checkpoint.get("support_input") != "raw":
        raise SystemExit("active-sequence fine-tuning requires a raw-input checkpoint")
    version_head = source_checkpoint.get("version_head", "slot-joint")
    if version_head not in FACTOR_BELIEF_HEADS:
        raise SystemExit("active-sequence fine-tuning requires a factor-belief head")
    if not bool(getattr(model, "supports_agent_probe_result_context", False)):
        raise SystemExit("checkpoint does not support the controller probe-result bit")
    if source_checkpoint.get("controller_context_schema") != (
        "agent-probe-result-bool.v1"
    ):
        raise SystemExit("unexpected controller context schema")
    if source_checkpoint.get("controller_input_reads_privileged_palette") is True:
        raise SystemExit("checkpoint inference reads a privileged palette")

    checkpoint_cli = source_checkpoint.get("cli_arguments", {})
    previous_finetune = source_checkpoint.get("finetune", {})
    context_fold = int(source_checkpoint["context_fold"])
    data_master_seed = (
        int(args.data_master_seed)
        if args.data_master_seed is not None
        else int(
            previous_finetune.get("data_master_seed")
            or checkpoint_cli.get("data_master_seed")
            or 2026071601
        )
    )
    train_split = str(
        args.train_split
        or previous_finetune.get("train_split")
        or checkpoint_cli.get("train_split")
        or "gram-causal-train"
    )
    eval_split = str(
        args.eval_split
        or previous_finetune.get("eval_split")
        or checkpoint_cli.get("eval_split")
        or "gram-causal-composition"
    )
    color_permutation_augmentation = (
        bool(source_checkpoint.get("color_permutation_augmentation", False))
        if args.color_permutation_augmentation is None
        else bool(args.color_permutation_augmentation)
    )

    train_pool, heldout_pool, pool_audit = _build_pools(
        torch=torch,
        model=model,
        context_fold=context_fold,
        data_master_seed=data_master_seed,
        train_split=train_split,
        eval_split=eval_split,
        device=device,
    )
    eval_tasks = heldout_pool[: args.eval_tasks_per_audit]
    train_expanded = _symmetry_expanded_version_space_mask(
        torch,
        model,
        train_pool,
        device=device,
    )
    heldout_expanded = _symmetry_expanded_version_space_mask(
        torch,
        model,
        heldout_pool,
        device=device,
    )
    active_train_pool = tuple(
        task
        for task, mask in zip(train_pool, train_expanded, strict=True)
        if int(mask.sum().item()) > 4
    )
    active_heldout_pool = tuple(
        task
        for task, mask in zip(heldout_pool, heldout_expanded, strict=True)
        if int(mask.sum().item()) > 4
    )
    if len(active_train_pool) < args.batch_size or not active_heldout_pool:
        raise AssertionError("public symmetry-break curriculum has too few tasks")
    active_eval_tasks = active_heldout_pool[: args.eval_tasks_per_audit]
    eval_tasks_by_phase = {
        phase: active_eval_tasks if phase in LEGACY_PUBLIC_PHASES else eval_tasks
        for phase in CURRICULUM
    }
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(args.seed + 10_000)
    eval_sequences = {
        phase: _sample_controller_batch(
            torch,
            eval_tasks_by_phase[phase],
            phase,
            generator=eval_generator,
        )
        for phase in CURRICULUM
    }
    audits_before = _audit_sequences(
        torch=torch,
        model=model,
        tasks_by_phase=eval_tasks_by_phase,
        sampled=eval_sequences,
        batch_size=args.eval_batch_size,
        device=device,
    )

    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named:
        raise AssertionError("checkpoint exposes no trainable parameters")
    if any(name.startswith("executor.") for name, _ in trainable_named):
        raise AssertionError("frozen executor entered the fine-tune optimizer")
    trainable_names = [name for name, _ in trainable_named]
    trainable = [parameter for _, parameter in trainable_named]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output = args.output.resolve()
    if output == checkpoint_path.parent:
        raise SystemExit("refusing to overwrite the source checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    output_checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"

    task_generator = torch.Generator(device="cpu")
    task_generator.manual_seed(args.seed + 1)
    candidate_generator = torch.Generator(device="cpu")
    candidate_generator.manual_seed(args.seed + 2)
    task_order = torch.randperm(len(train_pool), generator=task_generator).tolist()
    task_cursor = 0
    active_task_order = torch.randperm(
        len(active_train_pool),
        generator=task_generator,
    ).tolist()
    active_task_cursor = 0
    cycle_tasks: tuple[Any, ...] = ()
    cycle_index = -1
    cumulative_kind_counts: Counter[str] = Counter()
    phase_updates: Counter[str] = Counter()
    latest: dict[str, object] = {}
    started = time.perf_counter()

    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            phase = CURRICULUM[step % len(CURRICULUM)]
            if step % len(CURRICULUM) == 0:
                cycle_index += 1
                if task_cursor + args.batch_size > len(task_order):
                    task_order = torch.randperm(
                        len(train_pool),
                        generator=task_generator,
                    ).tolist()
                    task_cursor = 0
                selected_indices = task_order[
                    task_cursor : task_cursor + args.batch_size
                ]
                task_cursor += args.batch_size
                cycle_tasks = tuple(train_pool[index] for index in selected_indices)
            if len(cycle_tasks) != args.batch_size:
                raise AssertionError("paired curriculum cycle lost its task batch")

            training_tasks = cycle_tasks
            if phase in LEGACY_PUBLIC_PHASES:
                if active_task_cursor + args.batch_size > len(active_task_order):
                    active_task_order = torch.randperm(
                        len(active_train_pool),
                        generator=task_generator,
                    ).tolist()
                    active_task_cursor = 0
                active_indices = active_task_order[
                    active_task_cursor : active_task_cursor + args.batch_size
                ]
                active_task_cursor += args.batch_size
                training_tasks = tuple(
                    active_train_pool[index] for index in active_indices
                )

            sampled = _sample_controller_batch(
                torch,
                training_tasks,
                phase,
                generator=candidate_generator,
            )
            batch = _raw_public_history_batch(
                torch,
                sampled.histories,
                device=device,
            )
            if color_permutation_augmentation:
                # Re-seeding per phase gives every task the same palette
                # permutation throughout one full contrastive curriculum cycle.
                color_generator = torch.Generator(device="cpu")
                color_generator.manual_seed(args.seed + 1_000_000 + cycle_index)
                batch = _permute_raw_support_colors(
                    torch,
                    batch,
                    generator=color_generator,
                    num_colors=model.config.num_colors,
                )

            compatible_mask = _symmetry_expanded_version_space_mask(
                torch,
                model,
                training_tasks,
                device=device,
                histories=sampled.histories,
            )
            evidence_axes, evidence_value_mask = (
                _conditional_probe_innovation_targets(
                    torch,
                    model,
                    training_tasks,
                    device=device,
                    controller_histories=sampled.controller_histories,
                )
            )
            probe_result_mask = _controller_probe_result_mask(
                torch,
                sampled.controller_histories,
                device=device,
            )
            learning_rate = (
                args.tail_learning_rate
                if args.tail_steps and step >= args.steps - args.tail_steps
                else args.learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss = model.symmetry_aware_factor_belief_loss(
                batch,
                compatible_mask=compatible_mask,
                evidence_axis_targets=evidence_axes,
                evidence_value_target_mask=evidence_value_mask,
                task_factor_weight=args.task_consistency_weight,
                is_agent_probe_result=probe_result_mask,
            )
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite fine-tune loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.max_grad_norm,
                ).detach().cpu()
            )
            optimizer.step()

            cumulative_kind_counts.update(sampled.candidate_kind_counts)
            phase_updates[phase] += 1
            posterior_sizes = compatible_mask.sum(dim=-1)
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
                "curriculum_phase": phase,
                "online_probe_results": len(
                    sampled.controller_histories[0].transitions
                )
                - 6,
                "mean_target_version_space_size": float(
                    posterior_sizes.float().mean().detach().cpu()
                ),
                "sampled_candidate_kind_counts": sampled.candidate_kind_counts,
                "sampled_candidate_kind_sequences": [
                    list(sequence)
                    for sequence in sampled.candidate_kind_sequences
                ],
                "candidate_kind_used_for_training_sampling_only": True,
                "candidate_kind_in_model_input": False,
            }
            completed = step + 1
            if (
                completed == 1
                or completed % args.log_every == 0
                or completed == args.steps
            ):
                record = {
                    "step": completed,
                    "tasks_seen": completed * args.batch_size,
                    "curriculum_cycles_started": cycle_index + 1,
                    **latest,
                }
                progress_file.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
                progress_file.flush()
                print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    training_seconds = time.perf_counter() - started
    audits_after = _audit_sequences(
        torch=torch,
        model=model,
        tasks_by_phase=eval_tasks_by_phase,
        sampled=eval_sequences,
        batch_size=args.eval_batch_size,
        device=device,
    )

    source_hashes = _source_sha256()
    finetune_metadata: dict[str, object] = {
        "finetune_schema_version": FINETUNE_SCHEMA_VERSION,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "context_fold_inherited_from_source": context_fold,
        "data_master_seed": data_master_seed,
        "train_split": train_split,
        "eval_split": eval_split,
        "train_pool_tasks": len(train_pool),
        "heldout_pool_tasks": len(heldout_pool),
        "public_symmetry_break_train_tasks": len(active_train_pool),
        "public_symmetry_break_heldout_tasks": len(active_heldout_pool),
        "train_contexts": EXPECTED_TRAIN_CONTEXTS,
        "heldout_contexts": EXPECTED_HELDOUT_CONTEXTS,
        "curriculum": list(CURRICULUM),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "phase_update_counts": dict(sorted(phase_updates.items())),
        "selected_candidate_kind_counts": dict(
            sorted(cumulative_kind_counts.items())
        ),
        "candidate_kind_used_for_training_sampling_only": True,
        "candidate_kind_in_model_input": False,
        "model_input": (
            "public transition history plus controller-owned "
            "is_agent_probe_result bit"
        ),
        "environment_target_exposed_only_as_observed_next_state": True,
        "color_permutation_augmentation": color_permutation_augmentation,
        "training_seconds": round(training_seconds, 6),
    }
    updated_checkpoint = dict(source_checkpoint)
    updated_checkpoint.update(
        {
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "latest_training_metrics": latest,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainable
            ),
            "color_permutation_augmentation": color_permutation_augmentation,
            "cli_arguments": _json_arguments(args),
            "source_sha256": source_hashes,
            "finetune_schema_version": FINETUNE_SCHEMA_VERSION,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "pre_finetune_source_sha256": source_checkpoint.get("source_sha256"),
            "finetune": finetune_metadata,
        }
    )
    temporary_checkpoint = output_checkpoint_path.with_suffix(".pt.tmp")
    torch.save(updated_checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(output_checkpoint_path)

    # Prove that the artifact still obeys the original strict loader contract.
    reloaded, reloaded_metadata, reloaded_executor, _ = (
        load_public_version_k4_checkpoint(
            torch,
            output_checkpoint_path,
            device=device,
        )
    )
    strict_reload = {
        "passed": True,
        "model_type": type(reloaded).__name__,
        "version_head": reloaded_metadata.get("version_head"),
        "executor_checkpoint": str(reloaded_executor),
    }
    del reloaded

    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "checkpoint_schema_version": updated_checkpoint[
            "checkpoint_schema_version"
        ],
        "checkpoint_path": str(output_checkpoint_path),
        "checkpoint_sha256": _sha256_file(output_checkpoint_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_schema": executor_metadata.get(
            "checkpoint_schema_version"
        ),
        "context_fold": context_fold,
        "device": str(device),
        "cli_arguments": _json_arguments(args),
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "finetune": finetune_metadata,
        "context_pool_audit": pool_audit,
        "eval_tasks_per_sequence_audit": len(eval_tasks),
        "eval_tasks_by_sequence_audit": {
            phase: len(eval_tasks_by_phase[phase]) for phase in CURRICULUM
        },
        "sequence_audits_before": audits_before,
        "sequence_audits_after": audits_after,
        "strict_checkpoint_reload": strict_reload,
        "source_sha256": source_hashes,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
