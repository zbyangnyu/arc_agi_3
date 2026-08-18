#!/usr/bin/env python3
"""Train a deterministic, GPU-capable Persistent-K4 RuleGrid pilot.

This is intentionally a pilot, not the preregistered five-seed result.  Its
training loss is restricted to diagnostic queries 0..20 (single and pair
mechanisms).  It never materializes the three composition/triple targets at
indices 21..23; use ``eval_rulegrid_pilot.py`` to score those after training.

Example:

    python scripts/train_rulegrid_pilot.py \
      --device cuda --steps 2000 --batch-size 16 --seed 7 \
      --output runs/pilot_seed7
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


CHECKPOINT_SCHEMA_VERSION = "prp-wm.rulegrid-pilot-checkpoint.v2"
DEFAULT_DATA_MASTER_SEED = MASTER_SEED
_AUDITED_SOURCE_FILES = (
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/train_rulegrid_pilot.py",
    "scripts/eval_rulegrid_pilot.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--data-master-seed",
        type=int,
        default=DEFAULT_DATA_MASTER_SEED,
        help="Fixed RuleGrid nuisance/data seed; independent of --seed.",
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete torch device such as cuda:0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--model-profile",
        choices=("pilot", "architecture-default"),
        default="pilot",
        help=(
            "pilot is compact; architecture-default uses NeuralPRPConfig defaults. "
            "Both remain pilot runs."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Emit a JSON progress record every N updates.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write checkpoint_last.pt every N updates; 0 writes only at the end.",
    )
    parser.add_argument(
        "--split",
        default="pilot-train",
        help="Slash-free deterministic RuleGrid stream name.",
    )
    return parser.parse_args()


def _require_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    if value < 0 or (not allow_zero and value == 0):
        comparison = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"{name} must be {comparison}")


def _resolve_device(torch: Any, raw: str) -> Any:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(raw)
    except (TypeError, RuntimeError) as error:
        raise SystemExit(f"invalid --device {raw!r}: {error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    return device


def _configure_determinism(torch: Any, seed: int) -> None:
    """Set the reproducibility controls that are safe for this small pilot."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _model_config(NeuralPRPConfig: Any, profile: str) -> Any:
    if profile == "architecture-default":
        return NeuralPRPConfig()
    if profile != "pilot":  # argparse keeps this defensive branch unreachable.
        raise ValueError(f"unknown model profile: {profile!r}")
    # Retains the K=4 persistent architecture while keeping a GPU smoke/pilot
    # reasonably cheap.  This is intentionally recorded in the checkpoint and
    # is not represented as the formal preregistered study architecture.
    return NeuralPRPConfig(
        color_embedding=32,
        position_embedding=32,
        encoder_channels=32,
        encoder_resblocks=2,
        normalization_groups=8,
        action_embedding=32,
        rule_dim=64,
        attention_heads=4,
        attention_ffn=128,
        decoder_resblocks=2,
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _runtime_identity(torch: Any, device: Any) -> dict[str, object]:
    identity: dict[str, object] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        identity["cuda_runtime_version"] = torch.version.cuda
        identity["cuda_device_name"] = torch.cuda.get_device_name(device)
        identity["cuda_device_capability"] = list(torch.cuda.get_device_capability(device))
    return identity


def main() -> None:
    args = parse_args()
    _require_positive("--steps", args.steps)
    _require_positive("--batch-size", args.batch_size)
    _require_positive("--learning-rate", args.learning_rate)
    _require_positive("--weight-decay", args.weight_decay, allow_zero=True)
    _require_positive("--max-grad-norm", args.max_grad_norm)
    _require_positive("--log-every", args.log_every)
    _require_positive("--checkpoint-every", args.checkpoint_every, allow_zero=True)
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.data_master_seed < 0:
        raise SystemExit("--data-master-seed must be non-negative")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free string")

    # This must precede importing torch so CUDA libraries see the deterministic
    # cuBLAS workspace setting before their first CUDA operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from prp_wm.neural import NeuralPRPConfig, PersistentK4
        from prp_wm.pilot import (
            NONTRIPLE_DIAGNOSTIC_INDICES,
            PILOT_PROTOCOL_VERSION,
            assert_nontriple_training_indices,
            make_pilot_tensor_batch,
        )
        from prp_wm.rulegrid import BENCHMARK_VERSION
    except ImportError as error:
        raise SystemExit(f"RuleGrid pilot requires the optional PyTorch dependency: {error}") from error

    train_diagnostics = assert_nontriple_training_indices(NONTRIPLE_DIAGNOSTIC_INDICES)
    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    config = _model_config(NeuralPRPConfig, args.model_profile)
    model = PersistentK4(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    config_path = output / "run_config.json"
    summary_path = output / "train_summary.json"
    run_config: dict[str, object] = {
        "batch_size": args.batch_size,
        "benchmark_version": BENCHMARK_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "composition_targets_materialized_for_training": False,
        "data_master_seed": args.data_master_seed,
        "device": str(device),
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "materialized_diagnostic_target_indices": list(train_diagnostics),
        "model_config": asdict(config),
        "model_profile": args.model_profile,
        "pilot_protocol_version": PILOT_PROTOCOL_VERSION,
        "model_seed": args.seed,
        "split": args.split,
        "steps": args.steps,
        "train_diagnostic_indices": list(train_diagnostics),
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
        "weight_decay": args.weight_decay,
    }
    _atomic_json(config_path, run_config)

    def save_checkpoint(completed_steps: int, latest_metrics: dict[str, float]) -> str:
        # CPU tensors make the finished checkpoint evaluable on a CPU-only
        # machine as well as the originating CUDA device.
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        checkpoint: dict[str, object] = {
            "benchmark_version": BENCHMARK_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_config": asdict(config),
            "model_profile": args.model_profile,
            "model_state_dict": state_dict,
            "model_type": "PersistentK4",
            "pilot_protocol_version": PILOT_PROTOCOL_VERSION,
            "runtime_identity": _runtime_identity(torch, device),
            "source_sha256": _source_sha256(),
            "training": {
                "batch_size": args.batch_size,
                "composition_targets_materialized_for_training": False,
                "data_master_seed": args.data_master_seed,
                "learning_rate": args.learning_rate,
                "latest_metrics": latest_metrics,
                "materialized_diagnostic_target_indices": list(train_diagnostics),
                "model_seed": args.seed,
                "split": args.split,
                "steps_completed": completed_steps,
                "train_diagnostic_indices": list(train_diagnostics),
                "weight_decay": args.weight_decay,
            },
        }
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        return _sha256_file(checkpoint_path)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    latest_metrics: dict[str, float] = {}
    started = time.perf_counter()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            batch = make_pilot_tensor_batch(
                split=args.split,
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=train_diagnostics,
                include_behavior_targets=True,
                prefix_length=6,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.losses(batch)
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite total loss at step {step}: {loss.total}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            latest_metrics = loss.detached_metrics() | {"gradient_norm": gradient_norm}

            completed_steps = step + 1
            should_log = completed_steps == 1 or completed_steps % args.log_every == 0
            if should_log:
                record: dict[str, object] = {
                    "step": completed_steps,
                    "tasks_seen": completed_steps * args.batch_size,
                    **latest_metrics,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)
            if (
                args.checkpoint_every
                and completed_steps % args.checkpoint_every == 0
            ):
                save_checkpoint(completed_steps, latest_metrics)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    checkpoint_sha256 = save_checkpoint(args.steps, latest_metrics)
    elapsed_seconds = time.perf_counter() - started
    summary: dict[str, object] = {
        **run_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "final_metrics": latest_metrics,
        "model_parameters": parameter_count,
        "progress_path": str(progress_path),
        "summary_kind": "pilot_training_not_formal_preregistered_result",
        "tasks_seen": args.steps * args.batch_size,
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
