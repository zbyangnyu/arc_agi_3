#!/usr/bin/env python3
"""Test whether public support can produce a coherent latent hypothesis set.

This is a deliberately privileged *abstraction ceiling*.  A separately
verified oracle-factor executor is reused and frozen, while every random task
palette is mapped to canonical role IDs.  The inference network sees only the
six public support transitions; its K=4 modes are trained against an unordered
set of compatible behavior panels, never a program/factor label.

The held-out score is Coverage@4 on triple-composition queries from a distinct
task stream.  ``tied-k1`` is a capacity-matched negative control whose four
tensor-interface slots are constrained to remain one identical hypothesis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


CHECKPOINT_SCHEMA_VERSION = "prp-wm.canonical-latent-coverage.v1"
_AUDITED_SOURCE_FILES = (
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_canonical_latent_coverage.py",
    "scripts/run_rulegrid_executor_ceiling.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("persistent-k4", "tied-k1"),
        default="persistent-k4",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--support-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--nll-threshold", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="canonical-latent-train")
    parser.add_argument("--eval-split", default="canonical-latent-composition")
    parser.add_argument(
        "--train-executor",
        action="store_true",
        help="Fine-tune the pretrained executor; frozen is the default ceiling.",
    )
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


def _load_pretrained_executor(
    torch: Any,
    model: Any,
    checkpoint_path: Path,
    device: Any,
) -> tuple[dict[str, object], list[str]]:
    if not checkpoint_path.is_file():
        raise SystemExit(f"executor checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit("executor checkpoint must contain a dictionary")
    if checkpoint.get("model_type") != "OracleFactorExecutor":
        raise SystemExit("latent ceiling currently requires the pooled oracle executor")
    if checkpoint.get("privileged_palette_canonicalization") is not True:
        raise SystemExit("executor checkpoint must use oracle palette canonicalization")
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise SystemExit("executor checkpoint has no model_state_dict")
    prefixes = ("grid_encoder.", "action_encoder.", "decoder.")
    target_state = model.state_dict()
    loaded: list[str] = []
    for name in target_state:
        if name.startswith(prefixes):
            source = source_state.get(name)
            if source is None or source.shape != target_state[name].shape:
                raise SystemExit(f"executor tensor is missing or incompatible: {name}")
            target_state[name] = source
            loaded.append(name)
    model.load_state_dict(target_state)
    return checkpoint, loaded


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
    make_pilot_tasks: Any,
    make_behavior_batch: Any,
    predict_panel: Any,
    outcome_map: Any,
    triple_indices: tuple[int, ...],
) -> dict[str, object]:
    weighted_covered = 0.0
    valid_classes = 0
    covered_classes = 0
    all_covered_tasks = 0
    unique_signatures = 0
    entropy_bits = 0.0
    latent_spread = 0.0

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
            assert supervised.behavior_targets is not None
            assert supervised.behavior_mass is not None
            public = replace(
                supervised,
                behavior_targets=None,
                behavior_mass=None,
            )
            inference = model.infer_support(public)
            prediction = predict_panel(model, public, inference)
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
            qualifying = (
                panel_exact & (nll_by_class <= nll_threshold)
            ) & class_mask[:, :, None]
            class_covered = qualifying.any(dim=-1) & class_mask
            weighted_covered += float(
                (class_covered * supervised.behavior_mass).sum().cpu()
            )
            valid_classes += int(class_mask.sum().cpu())
            covered_classes += int(class_covered.sum().cpu())
            all_covered_tasks += int(
                ((class_covered | ~class_mask).all(dim=1)).sum().cpu()
            )

            signatures = maps.permute(0, 2, 1, 3, 4).reshape(
                batch_tasks, model.config.particles, -1
            )
            unique_signatures += sum(
                int(torch.unique(signatures[index], dim=0).shape[0])
                for index in range(batch_tasks)
            )
            weights = inference.weights.clamp_min(1e-12)
            entropy_bits += float((-(weights * weights.log2()).sum(dim=1)).sum().cpu())
            latent_spread += sum(
                float(torch.pdist(inference.modes[index]).mean().cpu())
                for index in range(batch_tasks)
            )

    return {
        "tasks": task_count,
        "behavior_classes": valid_classes,
        "covered_behavior_classes": covered_classes,
        "coverage_at_4_mass_weighted": weighted_covered / task_count,
        "coverage_at_4_unweighted": covered_classes / valid_classes,
        "all_classes_covered_task_rate": all_covered_tasks / task_count,
        "mean_covered_classes_per_task": covered_classes / task_count,
        "mean_unique_map_signatures": unique_signatures / task_count,
        "mean_mode_weight_entropy_bits": entropy_bits / task_count,
        "mean_pairwise_latent_distance": latent_spread / task_count,
        "coverage_nll_threshold_per_cell": nll_threshold,
    }


def main() -> None:
    args = parse_args()
    for name in ("steps", "batch_size", "eval_tasks", "eval_batch_size", "log_every"):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("learning_rate", "max_grad_norm", "nll_threshold"):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("weight_decay", "balanced_weight", "support_weight"):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name), allow_zero=True)
    if args.seed < 0 or args.data_master_seed < 0:
        raise SystemExit("seeds must be non-negative")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from prp_wm.latent_rules import (
        TiedSingleBelief,
        balanced_behavior_assignment_loss,
        outcome_map,
        predict_persistent_panel,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.neural import NeuralPRPConfig, PersistentK4
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    raw_executor = torch.load(
        args.executor_checkpoint.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    config = NeuralPRPConfig(**raw_executor["model_config"])
    model_type = {
        "persistent-k4": PersistentK4,
        "tied-k1": TiedSingleBelief,
    }[args.model]
    model = model_type(config).to(device)
    executor_checkpoint, loaded_executor_tensors = _load_pretrained_executor(
        torch,
        model,
        args.executor_checkpoint.resolve(),
        device,
    )
    executor_prefixes = ("grid_encoder.", "action_encoder.", "decoder.")
    if not args.train_executor:
        for name, parameter in model.named_parameters():
            if name.startswith(executor_prefixes):
                parameter.requires_grad_(False)
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
    effective_hypotheses = 4 if args.model == "persistent-k4" else 1
    run_config: dict[str, object] = {
        "experiment": "canonical_palette_latent_hypothesis_coverage",
        "result_kind": "privileged_palette_abstraction_ceiling_not_agent_result",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": args.model,
        "effective_hypotheses": effective_hypotheses,
        "tensor_interface_slots": config.particles,
        "public_support_only_at_inference": True,
        "program_or_factor_labels_used_for_latent_training": False,
        "unordered_behavior_panel_supervision": True,
        "privileged_palette_canonicalization": True,
        "executor_frozen": not args.train_executor,
        "executor_checkpoint": str(args.executor_checkpoint.resolve()),
        "executor_checkpoint_sha256": _sha256_file(args.executor_checkpoint.resolve()),
        "executor_checkpoint_model_type": executor_checkpoint["model_type"],
        "loaded_executor_tensor_count": len(loaded_executor_tensors),
        "train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "heldout_triple_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "balanced_weight": args.balanced_weight,
        "support_weight": args.support_weight,
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "model_config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
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
            batch = rulegrid_tasks_to_canonical_behavior_batch(
                tasks,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            inference = model.infer_support(batch)
            height = width = model.config.grid_size
            support_denominator = batch.support_mask.sum().clamp_min(1).to(
                dtype=inference.prequential_mixture_log_prob.dtype
            ) * (height * width)
            support_loss = -(
                inference.prequential_mixture_log_prob
                * batch.support_mask.to(
                    dtype=inference.prequential_mixture_log_prob.dtype
                )
            ).sum() / support_denominator
            balanced = balanced_behavior_assignment_loss(
                model, batch, inference
            )
            total = (
                args.support_weight * support_loss
                + args.balanced_weight * balanced
            )
            if not bool(torch.isfinite(total).item()):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            latest = {
                "loss_total": float(total.detach().cpu()),
                "loss_support": float(support_loss.detach().cpu()),
                "loss_balanced_assignment": float(balanced.detach().cpu()),
                "gradient_norm": gradient_norm,
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
        "model_type": model_type.__name__,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "latest_training_metrics": latest,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    evaluation = _evaluate(
        torch=torch,
        model=model,
        device=device,
        split=args.eval_split,
        data_master_seed=args.data_master_seed,
        task_count=args.eval_tasks,
        batch_size=args.eval_batch_size,
        nll_threshold=args.nll_threshold,
        make_pilot_tasks=make_pilot_tasks,
        make_behavior_batch=rulegrid_tasks_to_canonical_behavior_batch,
        predict_panel=predict_persistent_panel,
        outcome_map=outcome_map,
        triple_indices=TRIPLE_DIAGNOSTIC_INDICES,
    )
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
        "interpretation": (
            "This isolates latent rule-set inference after privileged palette role "
            "canonicalization and a frozen verified executor. It is not a public-input "
            "ARC agent result. Persistent-K4 passes the static gate only if all-class "
            "coverage is high; tied-K1 is structurally unable to cover four distinct classes."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
