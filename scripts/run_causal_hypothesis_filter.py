#!/usr/bin/env python3
"""Audit explicit latent hypothesize-and-test inference on RuleGrid.

Inference enumerates 64 persistent mechanism tuples, scores each one only on
public support transitions with a frozen support-calibrated world model, and
retains four distinct hypotheses.  Held-out query outcomes are used only for
evaluation.  The mechanism axes, palette roles, and executor training remain
privileged, so this is a causal-abstraction ceiling rather than an ARC agent.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_causal_hypothesis_filter.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument("--continuous-result", type=Path)
    parser.add_argument("--slot-result", type=Path)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--selection-method", default="map_then_balanced_nll")
    parser.add_argument("--nll-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-split", default="causal-filter-composition")
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


def _unique_signatures(torch: Any, panels: Any) -> int:
    return int(torch.unique(panels.reshape(panels.shape[0], -1), dim=0).shape[0])


def _evaluate(
    *,
    torch: Any,
    executor: Any,
    device: Any,
    split: str,
    data_master_seed: int,
    task_count: int,
    batch_size: int,
    selection_method: str,
    nll_threshold: float,
    support_ablation: str,
    make_pilot_tasks: Any,
    make_behavior_batch: Any,
    score_hypothesis_bank: Any,
    select_hypotheses: Any,
    selected_factor_ids: Any,
    predict_factor_panel: Any,
    outcome_map: Any,
    factor_ids_for_program: Any,
    version_space: Any,
) -> dict[str, object]:
    axis_names = ("collision", "trigger", "relation")
    totals: dict[str, float] = {
        "classes": 0,
        "covered": 0,
        "weighted_covered": 0.0,
        "all_classes": 0,
        "factor_targets": 0,
        "factor_covered": 0,
        "all_factors": 0,
        "valid_particles": 0,
        "particles": 0,
        "support_exact": 0,
        "all_support": 0,
        "unique_codes": 0,
        "unique_signatures": 0,
        "posterior_compatible_mass": 0.0,
        "compatible_entropy_fraction": 0.0,
        "compatible_max_mass": 0.0,
        "balanced_gap_4_to_5": 0.0,
    }
    by_axis = {
        name: {"tasks": 0, "classes": 0, "covered": 0, "all": 0}
        for name in axis_names
    }
    factor_usage = torch.zeros((3, 4), dtype=torch.long)
    executor.eval()
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
            batch = make_behavior_batch(
                tasks,
                diagnostic_indices=(21, 22, 23),
                device=device,
            )
            evidence = batch
            if support_ablation == "shuffle-targets":
                evidence = replace(
                    batch,
                    support_targets=batch.support_targets.roll(1, dims=0),
                )
            elif support_ablation != "none":
                raise ValueError(f"unknown support ablation: {support_ablation}")
            scores = score_hypothesis_bank(
                executor,
                evidence.support_states,
                evidence.support_actions,
                evidence.support_targets,
                evidence.support_mask,
                evidence.support_action_mask,
            )
            selected_indices = select_hypotheses(
                scores,
                particles=4,
                method=selection_method,
            )
            codes = selected_factor_ids(scores, selected_indices)
            prediction = predict_factor_panel(
                executor,
                batch.query_states,
                batch.query_actions,
                codes,
                batch.query_action_mask,
            )
            batch_tasks, classes, queries, height, width = batch.behavior_targets.shape
            maps = outcome_map(prediction).reshape(
                batch_tasks, queries, 4, height, width
            )
            panel_exact = (
                maps[:, None] == batch.behavior_targets[:, :, :, None]
            ).all(dim=(2, 4, 5))
            nll = torch.empty(
                (batch_tasks, classes, 4),
                dtype=prediction.change_logits.dtype,
                device=device,
            )
            for class_index in range(classes):
                target = batch.behavior_targets[:, class_index].reshape(
                    batch_tasks * queries, height, width
                )
                nll[:, class_index] = -prediction.log_prob(target).reshape(
                    batch_tasks, queries, 4
                ).sum(dim=1) / float(queries * height * width)
            class_mask = batch.behavior_mass > 0
            class_covered = (
                panel_exact & (nll <= nll_threshold)
            ).any(dim=-1) & class_mask
            all_covered = (class_covered | ~class_mask).all(dim=1)
            totals["classes"] += int(class_mask.sum().cpu())
            totals["covered"] += int(class_covered.sum().cpu())
            totals["weighted_covered"] += float(
                (class_covered * batch.behavior_mass).sum().cpu()
            )
            totals["all_classes"] += int(all_covered.sum().cpu())

            selected_support_exact = scores.map_exact.gather(1, selected_indices)
            totals["support_exact"] += int(selected_support_exact.sum().cpu())
            totals["all_support"] += int(
                selected_support_exact.all(dim=1).sum().cpu()
            )
            totals["particles"] += batch_tasks * 4
            sorted_balanced = scores.balanced_nll_per_cell.sort(dim=1).values
            totals["balanced_gap_4_to_5"] += float(
                (sorted_balanced[:, 4] - sorted_balanced[:, 3]).sum().cpu()
            )

            valid_cells = batch.support_mask.sum(dim=1, keepdim=True) * height * width
            posterior = torch.softmax(
                -scores.proper_nll_per_cell * valid_cells, dim=1
            )
            for task_index, task in enumerate(tasks):
                compatible = {
                    factor_ids_for_program(program)
                    for program in version_space(
                        task.inference.support[:6], task.privileged.palette
                    )
                }
                predicted_rows = [
                    tuple(int(value) for value in row)
                    for row in codes[task_index].cpu().tolist()
                ]
                predicted = set(predicted_rows)
                overlap = predicted.intersection(compatible)
                totals["factor_targets"] += len(compatible)
                totals["factor_covered"] += len(overlap)
                totals["all_factors"] += int(compatible.issubset(predicted))
                totals["valid_particles"] += sum(
                    row in compatible for row in predicted_rows
                )
                totals["unique_codes"] += len(predicted)
                totals["unique_signatures"] += _unique_signatures(
                    torch, maps[task_index].permute(1, 0, 2, 3)
                )

                bank_rows = [
                    tuple(int(value) for value in row)
                    for row in scores.factor_ids.cpu().tolist()
                ]
                compatible_indices = torch.tensor(
                    [index for index, row in enumerate(bank_rows) if row in compatible],
                    dtype=torch.long,
                    device=device,
                )
                compatible_probabilities = posterior[task_index, compatible_indices]
                compatible_mass = compatible_probabilities.sum()
                normalized = compatible_probabilities / compatible_mass.clamp_min(1e-12)
                entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum()
                totals["posterior_compatible_mass"] += float(compatible_mass.cpu())
                totals["compatible_entropy_fraction"] += float(
                    (entropy / math.log(len(compatible))).cpu()
                )
                totals["compatible_max_mass"] += float(normalized.max().cpu())

                varying = [
                    axis
                    for axis in range(3)
                    if len({code[axis] for code in compatible}) == 4
                ]
                if len(varying) != 1:
                    raise AssertionError("expected exactly one unobserved mechanism axis")
                axis_name = axis_names[varying[0]]
                by_axis[axis_name]["tasks"] += 1
                by_axis[axis_name]["classes"] += int(class_mask[task_index].sum())
                by_axis[axis_name]["covered"] += int(class_covered[task_index].sum())
                by_axis[axis_name]["all"] += int(all_covered[task_index])
            for axis in range(3):
                factor_usage[axis] += torch.bincount(
                    codes[:, :, axis].reshape(-1).cpu(), minlength=4
                )

    return {
        "support_ablation": support_ablation,
        "tasks": task_count,
        "behavior_classes": int(totals["classes"]),
        "covered_behavior_classes": int(totals["covered"]),
        "coverage_at_4_mass_weighted": totals["weighted_covered"] / task_count,
        "coverage_at_4_unweighted": totals["covered"] / totals["classes"],
        "all_classes_covered_task_rate": totals["all_classes"] / task_count,
        "factor_tuple_coverage_at_4": totals["factor_covered"]
        / totals["factor_targets"],
        "all_four_factor_tuples_recovered_task_rate": totals["all_factors"]
        / task_count,
        "valid_particle_rate": totals["valid_particles"] / totals["particles"],
        "support_exact_particle_rate": totals["support_exact"] / totals["particles"],
        "all_particles_support_exact_task_rate": totals["all_support"] / task_count,
        "mean_unique_factor_tuples": totals["unique_codes"] / task_count,
        "mean_unique_map_signatures": totals["unique_signatures"] / task_count,
        "mean_posterior_mass_on_symbolic_version_space": totals[
            "posterior_compatible_mass"
        ]
        / task_count,
        "mean_normalized_entropy_within_compatible_set": totals[
            "compatible_entropy_fraction"
        ]
        / task_count,
        "mean_max_conditional_mass_within_compatible_set": totals[
            "compatible_max_mass"
        ]
        / task_count,
        "mean_balanced_nll_gap_fifth_minus_fourth": totals[
            "balanced_gap_4_to_5"
        ]
        / task_count,
        "factor_value_usage": factor_usage.tolist(),
        "coverage_nll_threshold_per_cell": nll_threshold,
        "by_heldout_axis": {
            name: {
                "tasks": values["tasks"],
                "coverage": values["covered"] / values["classes"],
                "all_classes_covered_task_rate": values["all"] / values["tasks"],
            }
            for name, values in by_axis.items()
        },
    }


def _prefix_curve(
    *,
    torch: Any,
    executor: Any,
    device: Any,
    split: str,
    data_master_seed: int,
    task_count: int,
    batch_size: int,
    selection_method: str,
    make_pilot_tasks: Any,
    make_behavior_batch: Any,
    score_hypothesis_bank: Any,
    select_hypotheses: Any,
    factor_ids_for_program: Any,
    version_space: Any,
) -> list[dict[str, object]]:
    rows = [
        {
            "tasks": 0,
            "version_space": 0,
            "compatible_covered": 0,
            "compatible_total": 0,
            "true_top4": 0,
            "exact_set": 0,
            "neural_exact_set": 0,
        }
        for _ in range(6)
    ]
    executor.eval()
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
            batch = make_behavior_batch(
                tasks,
                diagnostic_indices=(21, 22, 23),
                device=device,
            )
            for prefix in range(1, 7):
                action_mask = (
                    batch.support_action_mask[:, :prefix]
                    if batch.support_action_mask is not None
                    else None
                )
                scores = score_hypothesis_bank(
                    executor,
                    batch.support_states[:, :prefix],
                    batch.support_actions[:, :prefix],
                    batch.support_targets[:, :prefix],
                    batch.support_mask[:, :prefix],
                    action_mask,
                )
                selected = select_hypotheses(
                    scores, particles=4, method=selection_method
                )
                for task_index, task in enumerate(tasks):
                    compatible = {
                        factor_ids_for_program(program)
                        for program in version_space(
                            task.inference.support[:prefix], task.privileged.palette
                        )
                    }
                    predicted = {
                        tuple(int(value) for value in scores.factor_ids[index].tolist())
                        for index in selected[task_index]
                    }
                    exact_codes = {
                        tuple(int(value) for value in scores.factor_ids[index].tolist())
                        for index in scores.map_exact[task_index]
                        .nonzero(as_tuple=False)
                        .flatten()
                    }
                    true_code = factor_ids_for_program(task.privileged.true_program)
                    row = rows[prefix - 1]
                    row["tasks"] += 1
                    row["version_space"] += len(compatible)
                    row["compatible_covered"] += len(predicted.intersection(compatible))
                    row["compatible_total"] += len(compatible)
                    row["true_top4"] += int(true_code in predicted)
                    row["exact_set"] += int(
                        len(compatible) <= 4 and predicted == compatible
                    )
                    row["neural_exact_set"] += int(exact_codes == compatible)
    return [
        {
            "prefix_length": prefix,
            "tasks": int(row["tasks"]),
            "mean_symbolic_version_space_size": row["version_space"] / row["tasks"],
            "compatible_recall_at_4": row["compatible_covered"]
            / row["compatible_total"],
            "true_rule_top4_rate": row["true_top4"] / row["tasks"],
            "exact_version_space_task_rate_when_size_lte_4": row["exact_set"]
            / row["tasks"],
            "neural_map_exact_bank_equals_symbolic_task_rate": row[
                "neural_exact_set"
            ]
            / row["tasks"],
        }
        for prefix, row in enumerate(rows, start=1)
    ]


def main() -> None:
    args = parse_args()
    if args.eval_tasks <= 0 or args.eval_batch_size <= 1:
        raise SystemExit("eval tasks must be positive and batch size must exceed one")
    if args.nll_threshold <= 0:
        raise SystemExit("nll threshold must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from prp_wm.causal_filter import (
        predict_factor_panel,
        score_hypothesis_bank,
        select_hypotheses,
        selected_factor_ids,
    )
    from prp_wm.latent_rules import (
        outcome_map,
        rule_program_factor_ids,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.pilot import make_pilot_tasks
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _load_executor,
        _resolve_device,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    executor_path = args.executor_checkpoint.resolve()
    executor, executor_checkpoint = _load_executor(torch, executor_path, device)
    common = {
        "torch": torch,
        "executor": executor,
        "device": device,
        "split": args.eval_split,
        "data_master_seed": args.data_master_seed,
        "task_count": args.eval_tasks,
        "batch_size": args.eval_batch_size,
        "selection_method": args.selection_method,
        "nll_threshold": args.nll_threshold,
        "make_pilot_tasks": make_pilot_tasks,
        "make_behavior_batch": rulegrid_tasks_to_canonical_behavior_batch,
        "score_hypothesis_bank": score_hypothesis_bank,
        "select_hypotheses": select_hypotheses,
        "selected_factor_ids": selected_factor_ids,
        "predict_factor_panel": predict_factor_panel,
        "outcome_map": outcome_map,
        "factor_ids_for_program": rule_program_factor_ids,
        "version_space": version_space,
    }
    evaluation = _evaluate(**common, support_ablation="none")
    shuffled = _evaluate(**common, support_ablation="shuffle-targets")
    prefix = _prefix_curve(
        torch=torch,
        executor=executor,
        device=device,
        split=args.eval_split,
        data_master_seed=args.data_master_seed,
        task_count=args.eval_tasks,
        batch_size=args.eval_batch_size,
        selection_method=args.selection_method,
        make_pilot_tasks=make_pilot_tasks,
        make_behavior_batch=rulegrid_tasks_to_canonical_behavior_batch,
        score_hypothesis_bank=score_hypothesis_bank,
        select_hypotheses=select_hypotheses,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
    )
    comparisons: dict[str, object] = {}
    for name, raw_path in (
        ("continuous_latent_k4", args.continuous_result),
        ("amortized_causal_slots", args.slot_result),
    ):
        if raw_path is None:
            continue
        path = raw_path.resolve()
        baseline = json.loads(path.read_text(encoding="utf-8"))
        baseline_coverage = baseline["heldout_triple_coverage"][
            "coverage_at_4_mass_weighted"
        ]
        comparisons[name] = {
            "result_path": str(path),
            "result_sha256": _sha256_file(path),
            "coverage_at_4": baseline_coverage,
            "absolute_gain": evaluation["coverage_at_4_mass_weighted"]
            - baseline_coverage,
        }

    gate = bool(
        evaluation["coverage_at_4_mass_weighted"] >= 0.90
        and evaluation["all_classes_covered_task_rate"] >= 0.90
        and evaluation["factor_tuple_coverage_at_4"] >= 0.90
        and evaluation["all_particles_support_exact_task_rate"] >= 0.90
    )
    result: dict[str, object] = {
        "experiment": "explicit_latent_causal_hypothesis_filter_k4",
        "result_kind": "privileged_mechanism_bank_hypothesize_and_test_ceiling",
        "mechanism_axes_given": ["collision", "trigger", "relation"],
        "mechanism_values_enumerated": True,
        "hypothesis_bank_size": 64,
        "retained_hypotheses": 4,
        "selection_method": args.selection_method,
        "inference_evidence": "public-support-transitions-only",
        "query_targets_used_for_inference": False,
        "program_labels_used_for_inference": False,
        "privileged_palette_canonicalization": True,
        "executor_support_version_space_supervised_during_pretraining": True,
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_schema_version": executor_checkpoint[
            "checkpoint_schema_version"
        ],
        "eval_split": args.eval_split,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "source_sha256": _source_sha256(),
        "heldout_triple_coverage": evaluation,
        "shuffled_support_target_control": shuffled,
        "sequential_identification": prefix,
        "comparisons": comparisons,
        "static_gate": {
            "coverage_at_4_gte": 0.90,
            "all_classes_covered_task_rate_gte": 0.90,
            "factor_tuple_coverage_at_4_gte": 0.90,
            "all_particles_support_exact_task_rate_gte": 0.90,
            "passed": gate,
        },
        "interpretation": (
            "A pass shows that explicit latent hypothesis generation plus "
            "world-model consistency can recover an underdetermined mechanism "
            "set when the factor space and calibrated executor are given. It "
            "does not show autonomous discovery of that factor space."
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
