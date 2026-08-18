#!/usr/bin/env python3
"""Train and audit an axis-structured causal-mechanism K=4 ceiling.

The three RuleGrid mechanism axes and a factor-conditioned executor are fixed
privileged structure.  Factor values are not labels: a support-set encoder
must infer four hard mechanism tuples using only public observed transitions
and unordered full-panel behavior supervision.  Evaluation reports both
behavior Coverage@4 and recovery of the exact support-compatible tuple set.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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


CHECKPOINT_SCHEMA_VERSION = "prp-wm.causal-mechanism-k4.v1"
_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_rules.py",
    "prp_wm/latent_rules.py",
    "prp_wm/matched_executor.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/routed_executor.py",
    "prp_wm/rulegrid.py",
    "scripts/run_causal_mechanism_coverage.py",
    "scripts/run_rulegrid_executor_ceiling.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--support-weight", type=float, default=0.1)
    parser.add_argument("--proper-weight", type=float, default=1.0)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--duplicate-weight", type=float, default=0.05)
    parser.add_argument("--factor-temperature-start", type=float, default=1.0)
    parser.add_argument("--factor-temperature-end", type=float, default=0.25)
    parser.add_argument("--assignment-temperature-start", type=float, default=0.1)
    parser.add_argument("--assignment-temperature-end", type=float, default=0.0)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--nll-threshold", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="causal-mechanism-train")
    parser.add_argument("--eval-split", default="causal-mechanism-composition")
    return parser.parse_args()


def _positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"{name} must be {qualifier}")


def _resolve_device(torch: Any, raw: str) -> Any:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    return device


def _configure_determinism(torch: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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


def _load_executor(torch: Any, path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    from prp_wm.latent_rules import OracleFactorExecutor
    from prp_wm.matched_executor import (
        MatchedFactorLocalOracleFactorExecutor,
        MatchedWiderGlobalOracleFactorExecutor,
    )
    from prp_wm.neural import NeuralPRPConfig
    from prp_wm.routed_executor import CanonicalRoleRoutedOracleFactorExecutor

    if not path.is_file():
        raise SystemExit(f"executor checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type")
    executor_classes = {
        "OracleFactorExecutor": OracleFactorExecutor,
        "CanonicalRoleRoutedOracleFactorExecutor": (
            CanonicalRoleRoutedOracleFactorExecutor
        ),
        "MatchedWiderGlobalOracleFactorExecutor": (
            MatchedWiderGlobalOracleFactorExecutor
        ),
        "MatchedFactorLocalOracleFactorExecutor": (
            MatchedFactorLocalOracleFactorExecutor
        ),
    }
    if model_type not in executor_classes:
        raise SystemExit("causal ceiling requires a supported factor executor")
    if checkpoint.get("privileged_palette_canonicalization") is not True:
        raise SystemExit("executor must use oracle palette canonicalization")
    result_path = path.parent / "result.json"
    if not result_path.is_file():
        raise SystemExit("executor checkpoint needs its sibling result.json audit")
    executor_result = json.loads(result_path.read_text(encoding="utf-8"))
    if executor_result.get("checkpoint_sha256") != _sha256_file(path):
        raise SystemExit("executor result/checkpoint SHA256 provenance mismatch")
    if executor_result.get("model_type") != model_type:
        raise SystemExit("executor result/checkpoint model type mismatch")
    if (
        executor_result.get("heldout_triple_evaluation", {}).get(
            "exact_task_accuracy"
        )
        != 1.0
    ):
        raise SystemExit("executor checkpoint did not pass the held-out triple ceiling")
    config = NeuralPRPConfig(**checkpoint["model_config"])
    executor = executor_classes[model_type](config).to(device)
    executor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    return executor, checkpoint


def _linear_schedule(start: float, end: float, step: int, steps: int) -> float:
    if steps <= 1:
        return end
    fraction = step / (steps - 1)
    return start + fraction * (end - start)


def _unique_rows(torch: Any, values: Any) -> int:
    return int(torch.unique(values.reshape(values.shape[0], -1), dim=0).shape[0])


def _evaluate(
    *,
    torch: Any,
    model: Any,
    device: Any,
    split: str,
    data_master_seed: int,
    task_count: int,
    batch_size: int,
    nll_threshold: float,
    factor_temperature: float,
    support_ablation: str,
    make_pilot_tasks: Any,
    make_behavior_batch: Any,
    outcome_map: Any,
    triple_indices: tuple[int, ...],
    rule_program_factor_ids: Any,
    version_space: Any,
) -> dict[str, object]:
    axis_names = ("collision", "trigger", "relation")
    totals: dict[str, float] = {
        "weighted_covered": 0.0,
        "valid_classes": 0.0,
        "covered_classes": 0.0,
        "map_exact_classes": 0.0,
        "nll_classes": 0.0,
        "all_behavior_tasks": 0.0,
        "factor_targets": 0.0,
        "factor_targets_covered": 0.0,
        "all_factor_tasks": 0.0,
        "valid_particles": 0.0,
        "particles": 0.0,
        "support_exact_particles": 0.0,
        "all_support_tasks": 0.0,
        "worst_support_nll": 0.0,
        "unique_codes": 0.0,
        "unique_signatures": 0.0,
        "logit_margin": 0.0,
    }
    by_axis = {
        name: {"tasks": 0, "covered": 0, "classes": 0, "all": 0}
        for name in axis_names
    }
    factor_usage = torch.zeros((3, 4), dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for start in range(0, task_count, batch_size):
            count = min(batch_size, task_count - start)
            tasks = make_pilot_tasks(
                split=split,
                master_seed=data_master_seed,
                start=start,
                count=count,
                diagnostic_indices=triple_indices,
            )
            supervised = make_behavior_batch(
                tasks,
                diagnostic_indices=triple_indices,
                device=device,
            )
            inference_batch = supervised
            if support_ablation == "shuffle-targets":
                inference_batch = replace(
                    supervised,
                    support_targets=supervised.support_targets.roll(1, dims=0),
                )
            elif support_ablation != "none":
                raise ValueError(f"unknown support ablation: {support_ablation}")
            inference = model.infer_support(
                inference_batch, temperature=factor_temperature
            )
            prediction = model.predict_panel(supervised, inference)
            assert supervised.behavior_targets is not None
            assert supervised.behavior_mass is not None
            batch_tasks, classes, queries, height, width = (
                supervised.behavior_targets.shape
            )
            maps = outcome_map(prediction).reshape(
                batch_tasks,
                queries,
                model.config.particles,
                height,
                width,
            )
            panel_exact = (
                maps[:, None]
                == supervised.behavior_targets[:, :, :, None]
            ).all(dim=(2, 4, 5))
            nll_by_class = torch.empty(
                (batch_tasks, classes, model.config.particles),
                dtype=prediction.change_logits.dtype,
                device=device,
            )
            for class_index in range(classes):
                target = supervised.behavior_targets[:, class_index].reshape(
                    batch_tasks * queries, height, width
                )
                nll_by_class[:, class_index] = -prediction.log_prob(target).reshape(
                    batch_tasks, queries, model.config.particles
                ).sum(dim=1) / float(queries * height * width)
            class_mask = supervised.behavior_mass > 0
            map_covered = panel_exact.any(dim=-1) & class_mask
            nll_covered = (nll_by_class <= nll_threshold).any(dim=-1) & class_mask
            qualifying = (
                panel_exact & (nll_by_class <= nll_threshold)
            ) & class_mask[:, :, None]
            class_covered = qualifying.any(dim=-1) & class_mask
            totals["weighted_covered"] += float(
                (class_covered * supervised.behavior_mass).sum().cpu()
            )
            totals["valid_classes"] += int(class_mask.sum().cpu())
            totals["covered_classes"] += int(class_covered.sum().cpu())
            totals["map_exact_classes"] += int(map_covered.sum().cpu())
            totals["nll_classes"] += int(nll_covered.sum().cpu())
            all_behavior = (class_covered | ~class_mask).all(dim=1)
            totals["all_behavior_tasks"] += int(all_behavior.sum().cpu())

            support_prediction = model.predict_support(supervised, inference)
            support_targets = supervised.support_targets.reshape(
                batch_tasks * supervised.support_steps, height, width
            )
            support_maps = outcome_map(support_prediction).reshape(
                batch_tasks,
                supervised.support_steps,
                model.config.particles,
                height,
                width,
            )
            support_exact = (
                support_maps
                == supervised.support_targets[:, :, None]
            ).all(dim=(1, 3, 4))
            totals["support_exact_particles"] += int(support_exact.sum().cpu())
            totals["all_support_tasks"] += int(support_exact.all(dim=1).sum().cpu())
            support_nll = -support_prediction.log_prob(support_targets).reshape(
                batch_tasks, supervised.support_steps, model.config.particles
            ).sum(dim=1) / float(supervised.support_steps * height * width)
            totals["worst_support_nll"] += float(
                support_nll.amax(dim=-1).sum().cpu()
            )

            margins = inference.factor_logits.topk(2, dim=-1).values
            totals["logit_margin"] += float(
                (margins[..., 0] - margins[..., 1]).mean(dim=(1, 2)).sum().cpu()
            )
            for task_index, task in enumerate(tasks):
                compatible = version_space(
                    task.inference.support, task.privileged.palette
                )
                compatible_codes = {
                    rule_program_factor_ids(program) for program in compatible
                }
                if len(compatible_codes) != 4:
                    raise AssertionError("support must define exactly four factor tuples")
                predicted_codes = {
                    tuple(int(value) for value in row)
                    for row in inference.factor_ids[task_index].cpu().tolist()
                }
                covered_codes = compatible_codes.intersection(predicted_codes)
                totals["factor_targets"] += len(compatible_codes)
                totals["factor_targets_covered"] += len(covered_codes)
                totals["all_factor_tasks"] += int(
                    compatible_codes.issubset(predicted_codes)
                )
                totals["valid_particles"] += sum(
                    tuple(int(value) for value in row) in compatible_codes
                    for row in inference.factor_ids[task_index].cpu().tolist()
                )
                totals["particles"] += model.config.particles
                totals["unique_codes"] += len(predicted_codes)
                totals["unique_signatures"] += _unique_rows(
                    torch,
                    maps[task_index].permute(1, 0, 2, 3),
                )
                varying = [
                    axis
                    for axis in range(3)
                    if len({code[axis] for code in compatible_codes}) == 4
                ]
                if len(varying) != 1:
                    raise AssertionError("version space must vary along one axis")
                axis_name = axis_names[varying[0]]
                by_axis[axis_name]["tasks"] += 1
                by_axis[axis_name]["classes"] += int(class_mask[task_index].sum())
                by_axis[axis_name]["covered"] += int(class_covered[task_index].sum())
                by_axis[axis_name]["all"] += int(all_behavior[task_index])
            for axis in range(3):
                factor_usage[axis] += torch.bincount(
                    inference.factor_ids[:, :, axis].reshape(-1).cpu(),
                    minlength=4,
                )

    valid_classes = totals["valid_classes"]
    factor_targets = totals["factor_targets"]
    return {
        "support_ablation": support_ablation,
        "tasks": task_count,
        "behavior_classes": int(valid_classes),
        "covered_behavior_classes": int(totals["covered_classes"]),
        "coverage_at_4_mass_weighted": totals["weighted_covered"] / task_count,
        "coverage_at_4_unweighted": totals["covered_classes"] / valid_classes,
        "map_exact_class_recall": totals["map_exact_classes"] / valid_classes,
        "nll_threshold_class_recall": totals["nll_classes"] / valid_classes,
        "all_classes_covered_task_rate": totals["all_behavior_tasks"] / task_count,
        "factor_tuple_coverage_at_4": totals["factor_targets_covered"] / factor_targets,
        "all_four_factor_tuples_recovered_task_rate": totals["all_factor_tasks"] / task_count,
        "valid_particle_rate": totals["valid_particles"] / totals["particles"],
        "mean_unique_factor_tuples": totals["unique_codes"] / task_count,
        "mean_unique_map_signatures": totals["unique_signatures"] / task_count,
        "support_exact_particle_rate": totals["support_exact_particles"] / totals["particles"],
        "all_particles_support_exact_task_rate": totals["all_support_tasks"] / task_count,
        "mean_worst_particle_support_nll_per_cell": totals["worst_support_nll"] / task_count,
        "mean_factor_logit_margin": totals["logit_margin"] / task_count,
        "factor_value_usage": factor_usage.tolist(),
        "coverage_nll_threshold_per_cell": nll_threshold,
        "by_heldout_axis": {
            name: {
                "tasks": values["tasks"],
                "coverage": (
                    values["covered"] / values["classes"]
                    if values["classes"]
                    else None
                ),
                "all_classes_covered_task_rate": (
                    values["all"] / values["tasks"]
                    if values["tasks"]
                    else None
                ),
            }
            for name, values in by_axis.items()
        },
    }


def main() -> None:
    args = parse_args()
    for name in (
        "steps",
        "batch_size",
        "eval_tasks",
        "eval_batch_size",
        "attention_layers",
        "log_every",
    ):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name))
    for name in (
        "learning_rate",
        "max_grad_norm",
        "factor_temperature_start",
        "factor_temperature_end",
        "nll_threshold",
    ):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name))
    for name in (
        "weight_decay",
        "support_weight",
        "proper_weight",
        "balanced_weight",
        "duplicate_weight",
        "assignment_temperature_start",
        "assignment_temperature_end",
    ):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name), allow_zero=True)
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from prp_wm.causal_rules import AxisStructuredCausalK4
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

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    executor_path = args.executor_checkpoint.resolve()
    executor, executor_checkpoint = _load_executor(torch, executor_path, device)
    model = AxisStructuredCausalK4(
        executor,
        attention_layers=args.attention_layers,
        temperature=args.factor_temperature_start,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
    run_config: dict[str, object] = {
        "experiment": "axis_structured_causal_mechanism_k4_ceiling",
        "result_kind": "privileged_axis_and_palette_causal_abstraction_ceiling",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mechanism_axes_given": ["collision", "trigger", "relation"],
        "mechanism_value_labels_used_for_training": False,
        "program_labels_used_for_training": False,
        "true_query_targets_used_for_training": False,
        "support_derived_unordered_behavior_set_supervision": True,
        "full_panel_bijective_assignment": True,
        "straight_through_hard_factor_codes": True,
        "privileged_palette_canonicalization": True,
        "executor_frozen": True,
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_model_type": executor_checkpoint["model_type"],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "support_weight": args.support_weight,
        "proper_weight": args.proper_weight,
        "balanced_weight": args.balanced_weight,
        "duplicate_weight": args.duplicate_weight,
        "factor_temperature_start": args.factor_temperature_start,
        "factor_temperature_end": args.factor_temperature_end,
        "assignment_temperature_start": args.assignment_temperature_start,
        "assignment_temperature_end": args.assignment_temperature_end,
        "attention_layers": args.attention_layers,
        "train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "heldout_triple_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "model_config": asdict(model.config),
        "device": str(device),
        "torch_version": torch.__version__,
        "source_sha256": _source_sha256(),
    }

    latest: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            factor_temperature = _linear_schedule(
                args.factor_temperature_start,
                args.factor_temperature_end,
                step,
                args.steps,
            )
            assignment_temperature = _linear_schedule(
                args.assignment_temperature_start,
                args.assignment_temperature_end,
                step,
                args.steps,
            )
            tasks = make_pilot_tasks(
                split=args.train_split,
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
            )
            batch = rulegrid_tasks_to_canonical_behavior_batch(
                tasks,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.losses(
                batch,
                support_weight=args.support_weight,
                proper_weight=args.proper_weight,
                balanced_weight=args.balanced_weight,
                duplicate_weight=args.duplicate_weight,
                assignment_temperature=assignment_temperature,
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
                _unique_rows(torch, loss.inference.factor_ids[index])
                for index in range(batch.batch_size)
            ) / batch.batch_size
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "factor_temperature": factor_temperature,
                "assignment_temperature": assignment_temperature,
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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    checkpoint = {
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
        "make_pilot_tasks": make_pilot_tasks,
        "make_behavior_batch": rulegrid_tasks_to_canonical_behavior_batch,
        "outcome_map": outcome_map,
        "triple_indices": TRIPLE_DIAGNOSTIC_INDICES,
        "rule_program_factor_ids": rule_program_factor_ids,
        "version_space": version_space,
    }
    evaluation = _evaluate(
        **common_evaluation,
        support_ablation="none",
    )
    shuffled_evaluation = _evaluate(
        **common_evaluation,
        support_ablation="shuffle-targets",
    )
    comparison: dict[str, object] | None = None
    if args.baseline_result is not None:
        baseline_path = args.baseline_result.resolve()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_eval = baseline["heldout_triple_coverage"]
        comparison = {
            "baseline_result": str(baseline_path),
            "baseline_result_sha256": _sha256_file(baseline_path),
            "baseline_model": baseline["model"],
            "baseline_coverage_at_4": baseline_eval[
                "coverage_at_4_mass_weighted"
            ],
            "causal_coverage_at_4": evaluation[
                "coverage_at_4_mass_weighted"
            ],
            "absolute_coverage_gain": evaluation[
                "coverage_at_4_mass_weighted"
            ]
            - baseline_eval["coverage_at_4_mass_weighted"],
        }

    result: dict[str, object] = {
        **run_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "tasks_seen": args.steps * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "heldout_triple_coverage": evaluation,
        "shuffled_support_target_control": shuffled_evaluation,
        "continuous_k4_comparison": comparison,
        "static_gate": {
            "coverage_at_4_gte": 0.90,
            "all_classes_covered_task_rate_gte": 0.90,
            "factor_tuple_coverage_at_4_gte": 0.90,
            "passed": bool(
                evaluation["coverage_at_4_mass_weighted"] >= 0.90
                and evaluation["all_classes_covered_task_rate"] >= 0.90
                and evaluation["factor_tuple_coverage_at_4"] >= 0.90
            ),
        },
        "interpretation": (
            "A pass shows that unordered behavior supervision can recover a "
            "pre-structured discrete mechanism set inside a frozen verified "
            "executor. It does not show autonomous discovery of axes or a "
            "public-input ARC result. Active intervention remains out of scope "
            "until static candidate coverage passes."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
