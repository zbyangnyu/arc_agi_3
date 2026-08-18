#!/usr/bin/env python3
"""Evaluate a RuleGrid pilot checkpoint on held-out triple composition queries.

The evaluator only materializes diagnostic targets 21..23.  It rejects a
checkpoint unless its audit metadata states that training used exactly 0..20,
so this command cannot accidentally turn a composition result into an
in-distribution report.

Example:

    python scripts/eval_rulegrid_pilot.py \
      --checkpoint runs/pilot_seed7/checkpoint_last.pt \
      --device cuda --tasks 192 --batch-size 16 \
      --output runs/pilot_seed7/composition_eval.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


DEFAULT_DATA_MASTER_SEED = MASTER_SEED
CHECKPOINT_SCHEMA_VERSION = "prp-wm.rulegrid-pilot-checkpoint.v2"
_AUDITED_SOURCE_FILES = (
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/train_rulegrid_pilot.py",
    "scripts/eval_rulegrid_pilot.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--data-master-seed",
        type=int,
        default=DEFAULT_DATA_MASTER_SEED,
        help=(
            "RuleGrid nuisance/data seed. Must match the checkpoint's training "
            "data master seed (default: MASTER_SEED)."
        ),
    )
    parser.add_argument("--tasks", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete torch device such as cuda:0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="pilot-composition",
        help="Slash-free evaluation stream name, separate from pilot-train.",
    )
    return parser.parse_args()


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


def _configure_deterministic_inference(torch: Any, seed: int) -> None:
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


def _load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"checkpoint does not exist: {path}")
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 lacks the explicit weights_only argument.
        loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, dict):
        raise SystemExit("checkpoint must contain a dictionary")
    return loaded


def _mode_cell_log_probabilities(torch: Any, prediction: Any) -> Any:
    """Return per-mode, factorized cell log probabilities ``[B,K,C,H,W]``."""

    batch_size, modes, colors, height, width = prediction.new_color_logits.shape
    original = prediction.input_colors[:, None, None, :, :]
    color_ids = torch.arange(colors, device=prediction.new_color_logits.device)[
        None, None, :, None, None
    ]
    masked_logits = prediction.new_color_logits.masked_fill(
        original == color_ids, torch.finfo(prediction.new_color_logits.dtype).min
    )
    changed_log_prob = (
        torch.nn.functional.logsigmoid(prediction.change_logits)[:, :, None]
        + torch.nn.functional.log_softmax(masked_logits, dim=2)
    )
    unchanged_log_prob = torch.nn.functional.logsigmoid(-prediction.change_logits)
    # The original color has precisely the no-change probability: its color
    # decoder logit is deliberately excluded by OutcomePrediction.
    return changed_log_prob.scatter(
        2,
        prediction.input_colors[:, None, None]
        .expand(batch_size, modes, 1, height, width),
        unchanged_log_prob[:, :, None],
    )


def _flat_query_prediction(torch: Any, model: Any, batch: Any, inference: Any) -> tuple[Any, Any]:
    """Predict all public query inputs and return prediction plus mode weights."""

    assert batch.query_states is not None
    assert batch.query_actions is not None
    batch_size, queries, height, width = batch.query_states.shape
    flat_states = batch.query_states.reshape(batch_size * queries, height, width)
    if batch.query_actions.ndim == 3:
        flat_actions = batch.query_actions.reshape(batch_size * queries, 4)
    else:
        atoms = batch.query_actions.shape[2]
        flat_actions = batch.query_actions.reshape(batch_size * queries, atoms, 4)
    flat_mask = (
        batch.query_action_mask.reshape(batch_size * queries, -1)
        if batch.query_action_mask is not None
        else None
    )
    flat_modes = (
        inference.modes[:, None]
        .expand(-1, queries, -1, -1)
        .reshape(batch_size * queries, model.config.particles, model.config.rule_dim)
    )
    flat_log_weights = (
        inference.log_weights[:, None]
        .expand(-1, queries, -1)
        .reshape(batch_size * queries, model.config.particles)
    )
    prediction = model.predict(flat_states, flat_actions, flat_modes, flat_mask)
    return prediction, flat_log_weights


def _support_posterior_mode_map(
    torch: Any,
    prediction: Any,
    flat_mode_indices: Any,
) -> Any:
    """Create one coherent per-query MAP frame from a support-selected mode.

    ``flat_mode_indices`` comes solely from the support posterior, before any
    triple target is scored.  All cells in a frame are then decoded under that
    *same* mode; unlike a marginal mixture argmax, this never splices cells
    from distinct particles into a Frankenstein frame.
    """

    by_mode = _mode_cell_log_probabilities(torch, prediction)
    batch_size, _, colors, height, width = by_mode.shape
    selected = by_mode.gather(
        1,
        flat_mode_indices[:, None, None, None, None].expand(
            batch_size, 1, colors, height, width
        ),
    ).squeeze(1)
    return selected.argmax(dim=1)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.data_master_seed < 0:
        raise SystemExit("--data-master-seed must be non-negative")
    if args.tasks <= 0 or args.batch_size <= 0:
        raise SystemExit("--tasks and --batch-size must be positive")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free string")
    # Must happen before importing torch so CUDA observes the cuBLAS setting.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from prp_wm.neural import NeuralPRPConfig, PersistentK4
        from prp_wm.pilot import (
            NONTRIPLE_DIAGNOSTIC_INDICES,
            PILOT_PROTOCOL_VERSION,
            TRIPLE_DIAGNOSTIC_INDICES,
            assert_nontriple_training_indices,
            make_pilot_tensor_batch,
        )
        from prp_wm.rulegrid import BENCHMARK_VERSION
    except ImportError as error:
        raise SystemExit(f"RuleGrid pilot evaluation requires PyTorch: {error}") from error

    device = _resolve_device(torch, args.device)
    _configure_deterministic_inference(torch, args.seed)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = _load_checkpoint(torch, checkpoint_path, device)
    if checkpoint.get("model_type") != "PersistentK4":
        raise SystemExit("checkpoint is not a PersistentK4 pilot checkpoint")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("checkpoint schema version is incompatible")
    if checkpoint.get("pilot_protocol_version") != PILOT_PROTOCOL_VERSION:
        raise SystemExit("checkpoint pilot protocol version is incompatible")
    if checkpoint.get("benchmark_version") != BENCHMARK_VERSION:
        raise SystemExit("checkpoint benchmark version is incompatible")
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        raise SystemExit("checkpoint has no training audit metadata")
    if training.get("composition_targets_materialized_for_training") is not False:
        raise SystemExit("checkpoint does not prove that triple targets were held out")
    checkpoint_data_master_seed = training.get("data_master_seed")
    if type(checkpoint_data_master_seed) is not int or checkpoint_data_master_seed < 0:
        raise SystemExit("checkpoint has no valid data_master_seed audit field")
    if args.data_master_seed != checkpoint_data_master_seed:
        raise SystemExit(
            "--data-master-seed must match checkpoint training data_master_seed "
            f"({checkpoint_data_master_seed})"
        )
    if training.get("split") == args.split:
        raise SystemExit(
            "evaluation --split must differ from the checkpoint training split "
            "to prevent task-instance overlap"
        )
    try:
        checkpoint_train_indices = tuple(training["train_diagnostic_indices"])
        assert_nontriple_training_indices(checkpoint_train_indices)
        checkpoint_materialized_indices = tuple(
            training["materialized_diagnostic_target_indices"]
        )
        assert_nontriple_training_indices(checkpoint_materialized_indices)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"checkpoint train-target audit failed: {error}") from error
    raw_config = checkpoint.get("model_config")
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(raw_state, dict):
        raise SystemExit("checkpoint model configuration or state dict is malformed")
    try:
        config = NeuralPRPConfig(**raw_config)
        model = PersistentK4(config).to(device)
        model.load_state_dict(raw_state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"could not restore PersistentK4 checkpoint: {error}") from error
    model.eval()

    grid_cells = config.grid_size * config.grid_size
    total_per_query_grid_nll = 0.0
    total_joint_triple_nll = 0.0
    total_cells = 0
    total_grids = 0
    correct_cells = 0
    exact_grids = 0
    entropy_sum = 0.0
    processed_tasks = 0
    with torch.no_grad():
        for start in range(0, args.tasks, args.batch_size):
            count = min(args.batch_size, args.tasks - start)
            batch = make_pilot_tensor_batch(
                split=args.split,
                master_seed=args.data_master_seed,
                start=start,
                count=count,
                diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
                include_behavior_targets=False,
                prefix_length=6,
                device=device,
            )
            output = model(batch)
            if output.query_log_prob_by_mode is None:
                raise RuntimeError("evaluation batch did not produce query scores")
            log_mixture = torch.logsumexp(
                output.inference.log_weights[:, None] + output.query_log_prob_by_mode,
                dim=-1,
            )
            total_per_query_grid_nll += float((-log_mixture).sum().detach().cpu())
            joint_log_mixture = torch.logsumexp(
                output.inference.log_weights
                + output.query_log_prob_by_mode.sum(dim=1),
                dim=-1,
            )
            total_joint_triple_nll += float((-joint_log_mixture).sum().detach().cpu())
            assert batch.query_targets is not None
            prediction, _ = _flat_query_prediction(torch, model, batch, output.inference)
            support_mode_indices = (
                output.inference.log_weights.argmax(dim=-1)[:, None]
                .expand(-1, len(TRIPLE_DIAGNOSTIC_INDICES))
                .reshape(-1)
            )
            coherent_mode_map = _support_posterior_mode_map(
                torch, prediction, support_mode_indices
            ).reshape_as(batch.query_targets)
            correct = coherent_mode_map.eq(batch.query_targets)
            correct_cells += int(correct.sum().detach().cpu())
            exact_grids += int(correct.all(dim=(-2, -1)).sum().detach().cpu())
            grids = count * len(TRIPLE_DIAGNOSTIC_INDICES)
            total_grids += grids
            total_cells += grids * grid_cells
            weights = output.inference.weights
            entropy_sum += float((-(weights * output.inference.log_weights).sum()).detach().cpu())
            processed_tasks += count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary: dict[str, object] = {
        "benchmark_version": BENCHMARK_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_runtime_identity": checkpoint.get("runtime_identity"),
        "checkpoint_source_sha256": checkpoint.get("source_sha256"),
        "checkpoint_train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "checkpoint_materialized_diagnostic_target_indices": list(
            NONTRIPLE_DIAGNOSTIC_INDICES
        ),
        "composition_targets_are_evaluation_only": True,
        "coherent_exact_grid_metric": (
            "support_posterior_argmax_mode_then_mode_conditioned_cellwise_map"
        ),
        "data_master_seed": args.data_master_seed,
        "diagnostic_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "device": str(device),
        "evaluation_rng_seed": args.seed,
        "exact_grid_accuracy_support_posterior_mode_map": exact_grids / total_grids,
        "mean_mode_entropy_nats": entropy_sum / processed_tasks,
        "model_config": asdict(config),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pilot_protocol_version": PILOT_PROTOCOL_VERSION,
        "joint_triple_predictive_nll_per_cell": total_joint_triple_nll / total_cells,
        "per_query_mixture_predictive_nll_per_cell": (
            total_per_query_grid_nll / total_cells
        ),
        "scored_grids": total_grids,
        "scored_tasks": processed_tasks,
        "split": args.split,
        "summary_kind": "pilot_composition_triple_evaluation_not_formal_preregistered_result",
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
        "cell_accuracy_support_posterior_mode_map_ancillary": correct_cells / total_cells,
    }
    output_path = args.output.resolve()
    _atomic_json(output_path, summary)
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
