#!/usr/bin/env python3
"""Train and evaluate the privileged RuleGrid rule-executor ceiling.

This bounded diagnostic gives the shared neural executor the true
collision/trigger/relation factor IDs.  It trains only on diagnostic indices
0..20, then evaluates composition exclusively on held-out triple indices
21..23 from a different deterministic task stream.  A strong result says the
executor can use a correct latent rule; it does not demonstrate rule inference.
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


CHECKPOINT_SCHEMA_VERSION = "prp-wm.oracle-factor-executor.v1"
_AUDITED_SOURCE_FILES = (
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_rulegrid_executor_ceiling.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-profile",
        choices=("smoke", "pilot"),
        default="smoke",
    )
    parser.add_argument(
        "--executor",
        choices=("pooled", "spatial"),
        default="pooled",
        help="Controlled action representation A/B; all rule inputs remain identical.",
    )
    parser.add_argument(
        "--palette-input",
        choices=("raw", "oracle-canonical"),
        default="oracle-canonical",
        help=(
            "Use raw per-task colors or a privileged role-canonicalized palette. "
            "The latter isolates rule execution from unobserved palette bindings."
        ),
    )
    parser.add_argument("--train-split", default="executor-ceiling-train")
    parser.add_argument("--eval-split", default="executor-ceiling-composition")
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
    try:
        device = torch.device(raw)
    except (TypeError, RuntimeError) as error:
        raise SystemExit(f"invalid --device {raw!r}: {error}") from error
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


def _model_config(config_type: Any, profile: str) -> Any:
    if profile == "smoke":
        return config_type(
            color_embedding=16,
            position_embedding=16,
            encoder_channels=16,
            encoder_resblocks=1,
            normalization_groups=4,
            action_embedding=16,
            rule_dim=32,
            attention_ffn=64,
            decoder_resblocks=1,
        )
    if profile == "pilot":
        return config_type(
            color_embedding=32,
            position_embedding=32,
            encoder_channels=32,
            encoder_resblocks=2,
            normalization_groups=8,
            action_embedding=32,
            rule_dim=64,
            attention_ffn=128,
            decoder_resblocks=2,
        )
    raise ValueError(f"unknown model profile: {profile}")


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


def _runtime_identity(torch: Any, device: Any) -> dict[str, object]:
    value: dict[str, object] = {
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        value["cuda_runtime_version"] = torch.version.cuda
        value["cuda_device_name"] = torch.cuda.get_device_name(device)
    return value


def _evaluate(
    *,
    torch: Any,
    model: Any,
    device: Any,
    split: str,
    data_master_seed: int,
    task_count: int,
    batch_size: int,
    diagnostic_indices: tuple[int, ...],
    make_pilot_tasks: Any,
    make_factor_batch: Any,
    outcome_map: Any,
    canonicalize_palette: bool,
) -> dict[str, object]:
    total_nll = 0.0
    total_cells = 0
    total_frames = 0
    total_tasks = 0
    correct_cells = 0
    exact_frames = 0
    exact_tasks = 0
    changed_cells = 0
    correct_changed_cells = 0
    unchanged_cells = 0
    correct_unchanged_cells = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, task_count, batch_size):
            count = min(batch_size, task_count - start)
            tasks = make_pilot_tasks(
                split=split,
                master_seed=data_master_seed,
                start=start,
                count=count,
                diagnostic_indices=diagnostic_indices,
            )
            batch = make_factor_batch(
                tasks,
                diagnostic_indices=diagnostic_indices,
                device=device,
                canonicalize_palette=canonicalize_palette,
            )
            prediction = model.predict_panel(batch)
            height, width = batch.states.shape[-2:]
            flat_states = batch.states.reshape(-1, height, width)
            flat_targets = batch.targets.reshape(-1, height, width)
            cell_nll = -prediction.log_prob_cells(flat_targets).squeeze(1)
            maps = outcome_map(prediction)[:, 0]
            correct = maps.eq(flat_targets)
            changed = flat_targets.ne(flat_states)
            frame_exact = correct.all(dim=(-2, -1))
            task_exact = frame_exact.reshape(count, len(diagnostic_indices)).all(dim=1)

            total_nll += float(cell_nll.sum().detach().cpu())
            total_cells += int(cell_nll.numel())
            total_frames += int(flat_targets.shape[0])
            total_tasks += count
            correct_cells += int(correct.sum().detach().cpu())
            exact_frames += int(frame_exact.sum().detach().cpu())
            exact_tasks += int(task_exact.sum().detach().cpu())
            changed_cells += int(changed.sum().detach().cpu())
            correct_changed_cells += int((correct & changed).sum().detach().cpu())
            unchanged_cells += int((~changed).sum().detach().cpu())
            correct_unchanged_cells += int((correct & ~changed).sum().detach().cpu())

    proper_nll_per_cell = total_nll / total_cells
    exact_grid_accuracy = exact_frames / total_frames
    return {
        "tasks": total_tasks,
        "frames": total_frames,
        "cells": total_cells,
        "proper_nll_per_cell": proper_nll_per_cell,
        "cell_accuracy": correct_cells / total_cells,
        "exact_grid_accuracy": exact_grid_accuracy,
        "exact_task_accuracy": exact_tasks / total_tasks,
        "changed_cell_fraction": changed_cells / total_cells,
        "changed_cell_accuracy": (
            correct_changed_cells / changed_cells if changed_cells else None
        ),
        "unchanged_cell_accuracy": (
            correct_unchanged_cells / unchanged_cells if unchanged_cells else None
        ),
        "exploratory_near_exact_thresholds": {
            "proper_nll_per_cell_lte": 0.05,
            "exact_grid_accuracy_gte": 0.90,
        },
        "exploratory_near_exact": bool(
            proper_nll_per_cell <= 0.05 and exact_grid_accuracy >= 0.90
        ),
    }


def main() -> None:
    args = parse_args()
    for name in ("steps", "batch_size", "eval_tasks", "eval_batch_size", "log_every"):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("learning_rate", "max_grad_norm"):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("weight_decay", "balanced_weight"):
        _positive(f"--{name.replace('_', '-')}", getattr(args, name), allow_zero=True)
    if args.seed < 0 or args.data_master_seed < 0:
        raise SystemExit("seeds must be non-negative")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    if any(not split or "/" in split for split in (args.train_split, args.eval_split)):
        raise SystemExit("split names must be non-empty and slash-free")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from prp_wm.latent_rules import (
            OracleFactorExecutor,
            SpatialOracleFactorExecutor,
            outcome_map,
            rulegrid_tasks_to_oracle_factor_batch,
        )
        from prp_wm.neural import NeuralPRPConfig
        from prp_wm.pilot import (
            NONTRIPLE_DIAGNOSTIC_INDICES,
            PILOT_PROTOCOL_VERSION,
            TRIPLE_DIAGNOSTIC_INDICES,
            assert_nontriple_training_indices,
            make_pilot_tasks,
        )
        from prp_wm.rulegrid import BENCHMARK_VERSION
    except ImportError as error:
        raise SystemExit(f"executor ceiling requires PyTorch: {error}") from error

    train_indices = assert_nontriple_training_indices(NONTRIPLE_DIAGNOSTIC_INDICES)
    single_indices = tuple(range(12))
    pair_indices = tuple(range(12, 21))
    eval_indices = TRIPLE_DIAGNOSTIC_INDICES
    canonicalize_palette = args.palette_input == "oracle-canonical"
    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    config = _model_config(NeuralPRPConfig, args.model_profile)
    executor_type = {
        "pooled": OracleFactorExecutor,
        "spatial": SpatialOracleFactorExecutor,
    }[args.executor]
    model = executor_type(config).to(device)
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
    run_config: dict[str, object] = {
        "experiment": "oracle_factor_rule_executor_ceiling",
        "result_kind": "bounded_privileged_diagnostic_not_agent_result",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "pilot_protocol_version": PILOT_PROTOCOL_VERSION,
        "privileged_true_rule_factors_are_model_inputs": True,
        "palette_input": args.palette_input,
        "privileged_palette_canonicalization": canonicalize_palette,
        "full_program_lookup_embedding": False,
        "executor": args.executor,
        "composition_targets_materialized_for_training": False,
        "train_diagnostic_indices": list(train_indices),
        "eval_diagnostic_groups": {
            "single": list(single_indices),
            "pair": list(pair_indices),
            "heldout_triple": list(eval_indices),
        },
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "balanced_weight": args.balanced_weight,
        "max_grad_norm": args.max_grad_norm,
        "model_profile": args.model_profile,
        "model_config": asdict(config),
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
    }

    latest_metrics: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            tasks = make_pilot_tasks(
                split=args.train_split,
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=train_indices,
            )
            batch = rulegrid_tasks_to_oracle_factor_batch(
                tasks,
                diagnostic_indices=train_indices,
                device=device,
                canonicalize_palette=canonicalize_palette,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.losses(batch, balanced_weight=args.balanced_weight)
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            latest_metrics = loss.detached_metrics() | {"gradient_norm": gradient_norm}
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record: dict[str, object] = {
                    "step": completed,
                    "tasks_seen": completed * args.batch_size,
                    **latest_metrics,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    checkpoint: dict[str, object] = {
        **run_config,
        "model_type": executor_type.__name__,
        "model_state_dict": state_dict,
        "latest_training_metrics": latest_metrics,
    }
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)

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
        "canonicalize_palette": canonicalize_palette,
    }
    single_evaluation = _evaluate(
        **common_evaluation,
        diagnostic_indices=single_indices,
    )
    pair_evaluation = _evaluate(
        **common_evaluation,
        diagnostic_indices=pair_indices,
    )
    evaluation = _evaluate(
        **common_evaluation,
        diagnostic_indices=eval_indices,
    )
    result: dict[str, object] = {
        **run_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tasks_seen": args.steps * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest_metrics,
        "single_evaluation": single_evaluation,
        "pair_evaluation": pair_evaluation,
        "heldout_triple_evaluation": evaluation,
        "interpretation": (
            "A pass isolates rule inference as the remaining bottleneck. A failure in this "
            "bounded run does not prove the executor class is incapable; inspect optimization, "
            "action encoding, and state decoding before returning to learned particles."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
