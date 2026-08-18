#!/usr/bin/env python3
"""Train a factor executor calibrated on both evidence and query domains.

The original executor ceiling learned diagnostic transitions only.  That is
enough to test a known rule on diagnostic queries, but not enough to rank rule
hypotheses from public support evidence.  This experiment additionally trains
every support transition under all four symbolically compatible factor tuples,
so an unobserved mechanism value cannot acquire a spurious likelihood bias.

This remains a privileged ceiling: factor axes, factor values, palette roles,
and the support version space are supplied during executor training.
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


CHECKPOINT_SCHEMA_VERSION = "prp-wm.support-calibrated-factor-executor.v1"
_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_rulegrid_executor_ceiling.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--diagnostic-loss-weight", type=float, default=0.50)
    parser.add_argument("--calibration-loss-weight", type=float, default=0.25)
    parser.add_argument("--neutral-loss-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--train-split", default="support-executor-train")
    parser.add_argument("--eval-split", default="support-executor-composition")
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


def _repeat_batch_axis(tensor: Any, repeats: int) -> Any:
    if tensor is None:
        return None
    return (
        tensor[:, None]
        .expand(-1, repeats, *([-1] * (tensor.ndim - 1)))
        .reshape(tensor.shape[0] * repeats, *tensor.shape[1:])
    )


def _compatible_support_batches(
    *,
    torch: Any,
    tasks: tuple[Any, ...],
    device: Any,
    make_tensor_batch: Any,
    oracle_batch_type: Any,
    factor_ids_for_program: Any,
    version_space: Any,
) -> tuple[Any, Any, Any, list[set[tuple[int, int, int]]]]:
    """Build two evidence domains, each replicated over the exact version space."""

    available_indices = tasks[0].privileged.diagnostic_target_indices
    if not available_indices:
        raise ValueError("tasks must materialize at least one diagnostic placeholder")
    public = make_tensor_batch(
        tasks,
        prefix_length=6,
        include_behavior_targets=False,
        diagnostic_indices=(available_indices[0],),
        device=device,
    )
    compatible_sets: list[set[tuple[int, int, int]]] = []
    ordered_codes: list[tuple[int, int, int]] = []
    for task in tasks:
        compatible = {
            factor_ids_for_program(program)
            for program in version_space(
                task.inference.support[:6], task.privileged.palette
            )
        }
        if len(compatible) != 4:
            raise AssertionError("six support transitions must leave four factor tuples")
        compatible_sets.append(compatible)
        ordered_codes.extend(sorted(compatible))
    factor_ids = torch.tensor(ordered_codes, dtype=torch.long, device=device)
    repeated_states = _repeat_batch_axis(public.support_states, 4)
    repeated_actions = _repeat_batch_axis(public.support_actions, 4)
    repeated_targets = _repeat_batch_axis(public.support_targets, 4)
    repeated_action_mask = _repeat_batch_axis(public.support_action_mask, 4)

    def materialize(indices: slice) -> Any:
        return oracle_batch_type(
            states=repeated_states[:, indices],
            actions=repeated_actions[:, indices],
            targets=repeated_targets[:, indices],
            factor_ids=factor_ids,
            action_mask=(
                repeated_action_mask[:, indices]
                if repeated_action_mask is not None
                else None
            ),
            palette_canonicalized=True,
        )

    return public, materialize(slice(0, 2)), materialize(slice(2, 6)), compatible_sets


def _support_evaluation(
    *,
    torch: Any,
    model: Any,
    device: Any,
    split: str,
    data_master_seed: int,
    task_count: int,
    batch_size: int,
    make_pilot_tasks: Any,
    make_tensor_batch: Any,
    oracle_batch_type: Any,
    factor_ids_for_program: Any,
    version_space: Any,
    outcome_map: Any,
    score_hypothesis_bank: Any,
    select_hypotheses: Any,
) -> dict[str, object]:
    totals = {
        "compatible_frames": 0,
        "compatible_exact_frames": 0,
        "compatible_particles": 0,
        "compatible_six_frame_exact": 0,
        "all_four_exact_tasks": 0,
        "proper_nll": 0.0,
        "proper_cells": 0,
        "likelihood_spread": 0.0,
        "bank_exact_codes": 0,
        "bank_exact_set_tasks": 0,
    }
    methods = ("proper_nll", "balanced_nll", "map_then_balanced_nll")
    selection = {
        method: {"covered": 0, "all": 0, "true_top4": 0}
        for method in methods
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
                diagnostic_indices=(21, 22, 23),
            )
            public, calibration, neutral, compatible_sets = _compatible_support_batches(
                torch=torch,
                tasks=tasks,
                device=device,
                make_tensor_batch=make_tensor_batch,
                oracle_batch_type=oracle_batch_type,
                factor_ids_for_program=factor_ids_for_program,
                version_space=version_space,
            )
            combined = oracle_batch_type(
                states=torch.cat((calibration.states, neutral.states), dim=1),
                actions=torch.cat((calibration.actions, neutral.actions), dim=1),
                targets=torch.cat((calibration.targets, neutral.targets), dim=1),
                factor_ids=calibration.factor_ids,
                action_mask=(
                    torch.cat((calibration.action_mask, neutral.action_mask), dim=1)
                    if calibration.action_mask is not None
                    else None
                ),
                palette_canonicalized=True,
            )
            prediction = model.predict_panel(combined)
            height, width = combined.states.shape[-2:]
            flat_targets = combined.targets.reshape(-1, height, width)
            cell_nll = -prediction.log_prob_cells(flat_targets).squeeze(1)
            maps = outcome_map(prediction)[:, 0]
            exact_frames = maps.eq(flat_targets).all(dim=(-2, -1)).reshape(count, 4, 6)
            exact_particles = exact_frames.all(dim=-1)
            totals["compatible_frames"] += count * 4 * 6
            totals["compatible_exact_frames"] += int(exact_frames.sum().cpu())
            totals["compatible_particles"] += count * 4
            totals["compatible_six_frame_exact"] += int(exact_particles.sum().cpu())
            totals["all_four_exact_tasks"] += int(exact_particles.all(dim=1).sum().cpu())
            totals["proper_nll"] += float(cell_nll.sum().cpu())
            totals["proper_cells"] += int(cell_nll.numel())
            per_particle_nll = cell_nll.reshape(count, 4, 6, height, width).mean(
                dim=(2, 3, 4)
            )
            totals["likelihood_spread"] += float(
                (per_particle_nll.amax(dim=1) - per_particle_nll.amin(dim=1)).sum().cpu()
            )

            scores = score_hypothesis_bank(
                model,
                public.support_states,
                public.support_actions,
                public.support_targets,
                public.support_mask,
                public.support_action_mask,
            )
            totals["bank_exact_codes"] += int(scores.map_exact.sum().cpu())
            for task_index, task in enumerate(tasks):
                exact_codes = {
                    tuple(int(value) for value in scores.factor_ids[index].tolist())
                    for index in scores.map_exact[task_index].nonzero(as_tuple=False).flatten()
                }
                totals["bank_exact_set_tasks"] += int(
                    exact_codes == compatible_sets[task_index]
                )
                true_code = factor_ids_for_program(task.privileged.true_program)
                for method in methods:
                    indices = select_hypotheses(
                        scores,
                        particles=4,
                        method=method,
                    )[task_index]
                    predicted = {
                        tuple(int(value) for value in scores.factor_ids[index].tolist())
                        for index in indices
                    }
                    overlap = predicted.intersection(compatible_sets[task_index])
                    selection[method]["covered"] += len(overlap)
                    selection[method]["all"] += int(
                        compatible_sets[task_index].issubset(predicted)
                    )
                    selection[method]["true_top4"] += int(true_code in predicted)

    return {
        "tasks": task_count,
        "compatible_support_proper_nll_per_cell": totals["proper_nll"]
        / totals["proper_cells"],
        "compatible_support_exact_frame_rate": totals["compatible_exact_frames"]
        / totals["compatible_frames"],
        "compatible_six_frame_exact_particle_rate": totals[
            "compatible_six_frame_exact"
        ]
        / totals["compatible_particles"],
        "all_four_compatible_six_frame_exact_task_rate": totals[
            "all_four_exact_tasks"
        ]
        / task_count,
        "mean_compatible_likelihood_spread_nats_per_cell": totals[
            "likelihood_spread"
        ]
        / task_count,
        "mean_neural_map_exact_bank_size": totals["bank_exact_codes"] / task_count,
        "neural_map_exact_bank_equals_symbolic_version_space_task_rate": totals[
            "bank_exact_set_tasks"
        ]
        / task_count,
        "top4_selection": {
            method: {
                "compatible_factor_coverage_at_4": values["covered"]
                / (4 * task_count),
                "exact_version_space_task_rate": values["all"] / task_count,
                "true_rule_top4_rate": values["true_top4"] / task_count,
            }
            for method, values in selection.items()
        },
    }


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.eval_tasks, args.eval_batch_size) <= 0:
        raise SystemExit("steps and batch sizes must be positive")
    weights = (
        args.diagnostic_loss_weight,
        args.calibration_loss_weight,
        args.neutral_loss_weight,
    )
    if min(weights) < 0 or abs(sum(weights) - 1.0) > 1e-9:
        raise SystemExit("the three domain loss weights must be non-negative and sum to one")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from prp_wm.causal_filter import score_hypothesis_bank, select_hypotheses
    from prp_wm.latent_rules import (
        OracleFactorBatch,
        OracleFactorExecutor,
        outcome_map,
        rule_program_factor_ids,
        rulegrid_tasks_to_canonical_tensor_batch,
        rulegrid_tasks_to_oracle_factor_batch,
    )
    from prp_wm.neural import NeuralPRPConfig
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.rulegrid import version_space
    from scripts.run_rulegrid_executor_ceiling import (
        _configure_determinism,
        _evaluate,
        _model_config,
        _resolve_device,
        _runtime_identity,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    config = _model_config(NeuralPRPConfig, args.model_profile)
    model = OracleFactorExecutor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    run_config: dict[str, object] = {
        "experiment": "support_calibrated_oracle_factor_executor_ceiling",
        "result_kind": "privileged_support_version_space_executor_ceiling",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": "OracleFactorExecutor",
        "privileged_true_rule_factors_are_diagnostic_inputs": True,
        "privileged_support_version_space_factors_are_training_inputs": True,
        "privileged_palette_canonicalization": True,
        "palette_input": "oracle-canonical",
        "support_factor_replication": "all-four-symbolically-compatible-tuples",
        "train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "heldout_triple_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "balanced_weight": args.balanced_weight,
        "domain_loss_weights": {
            "diagnostic": args.diagnostic_loss_weight,
            "calibration_support_first_two": args.calibration_loss_weight,
            "neutral_support_last_four": args.neutral_loss_weight,
        },
        "model_profile": args.model_profile,
        "model_config": asdict(config),
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
    }

    latest: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            tasks = make_pilot_tasks(
                split=args.train_split,
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
            )
            diagnostic = rulegrid_tasks_to_oracle_factor_batch(
                tasks,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
                device=device,
                canonicalize_palette=True,
            )
            _, calibration, neutral, _ = _compatible_support_batches(
                torch=torch,
                tasks=tasks,
                device=device,
                make_tensor_batch=rulegrid_tasks_to_canonical_tensor_batch,
                oracle_batch_type=OracleFactorBatch,
                factor_ids_for_program=rule_program_factor_ids,
                version_space=version_space,
            )
            optimizer.zero_grad(set_to_none=True)
            diagnostic_loss = model.losses(
                diagnostic, balanced_weight=args.balanced_weight
            )
            calibration_loss = model.losses(
                calibration, balanced_weight=args.balanced_weight
            )
            neutral_loss = model.losses(neutral, balanced_weight=args.balanced_weight)
            total = (
                args.diagnostic_loss_weight * diagnostic_loss.total
                + args.calibration_loss_weight * calibration_loss.total
                + args.neutral_loss_weight * neutral_loss.total
            )
            if not bool(torch.isfinite(total).item()):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            latest = {
                "loss_total": float(total.detach().cpu()),
                "loss_diagnostic": float(diagnostic_loss.total.detach().cpu()),
                "loss_calibration_support": float(calibration_loss.total.detach().cpu()),
                "loss_neutral_support": float(neutral_loss.total.detach().cpu()),
                "gradient_norm": gradient_norm,
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

    common_evaluation = {
        "torch": torch,
        "model": model,
        "device": device,
        "split": args.eval_split,
        "data_master_seed": args.data_master_seed,
        "task_count": args.eval_tasks,
        "batch_size": args.eval_batch_size,
        "make_pilot_tasks": make_pilot_tasks,
        "make_factor_batch": rulegrid_tasks_to_oracle_factor_batch,
        "outcome_map": outcome_map,
        "canonicalize_palette": True,
    }
    single = _evaluate(**common_evaluation, diagnostic_indices=tuple(range(12)))
    pair = _evaluate(**common_evaluation, diagnostic_indices=tuple(range(12, 21)))
    triple = _evaluate(
        **common_evaluation, diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES
    )
    support = _support_evaluation(
        torch=torch,
        model=model,
        device=device,
        split=args.eval_split,
        data_master_seed=args.data_master_seed,
        task_count=args.eval_tasks,
        batch_size=args.eval_batch_size,
        make_pilot_tasks=make_pilot_tasks,
        make_tensor_batch=rulegrid_tasks_to_canonical_tensor_batch,
        oracle_batch_type=OracleFactorBatch,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
        outcome_map=outcome_map,
        score_hypothesis_bank=score_hypothesis_bank,
        select_hypotheses=select_hypotheses,
    )
    gate = bool(
        single["exact_task_accuracy"] == 1.0
        and pair["exact_task_accuracy"] == 1.0
        and triple["exact_task_accuracy"] == 1.0
        and support["all_four_compatible_six_frame_exact_task_rate"] >= 0.99
        and support["top4_selection"]["map_then_balanced_nll"][
            "exact_version_space_task_rate"
        ]
        >= 0.99
    )
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
        "heldout_support_evaluation": support,
        "causal_filter_executor_gate": {
            "diagnostic_exact_task_accuracy_required": 1.0,
            "all_four_support_exact_task_rate_required": 0.99,
            "top4_exact_version_space_task_rate_required": 0.99,
            "passed": gate,
        },
        "interpretation": (
            "A pass only establishes a privileged world-model interface for "
            "latent hypothesize-and-test inference. It does not establish "
            "autonomous discovery of the factor axes or palette roles."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
