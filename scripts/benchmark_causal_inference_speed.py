#!/usr/bin/env python3
"""Compare amortized support inference with cached exhaustive 64-code scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if min(args.tasks, args.batch_size, args.repeats) <= 0 or args.warmup < 0:
        raise SystemExit("task, batch, repeat counts must be valid")
    import torch
    from prp_wm.discrete_causal_rules import ExpectedDiscreteCausalK4
    from prp_wm.latent_rules import (
        rule_program_factor_ids,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.pilot import TRIPLE_DIAGNOSTIC_INDICES, make_pilot_tasks
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import _resolve_device
    from scripts.run_expected_discrete_causal_coverage import (
        _build_context_pool,
        _load_audited_executor,
    )

    device = _resolve_device(torch, args.device)
    executor_path = args.executor_checkpoint.resolve()
    executor, _ = _load_audited_executor(torch, executor_path, device)
    checkpoint_path = args.model_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ExpectedDiscreteCausalK4(
        executor,
        attention_layers=int(checkpoint["attention_layers"]),
        temperature=float(checkpoint["factor_temperature_end"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    tasks = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split="causal-speed-context-holdout",
        master_seed=int(checkpoint["data_master_seed"]),
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.tasks,
        heldout=True,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
    )
    batches = [
        rulegrid_tasks_to_canonical_behavior_batch(
            tasks[start : start + args.batch_size],
            diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
            device=device,
        )
        for start in range(0, len(tasks), args.batch_size)
    ]

    def amortized(batch):
        return model.infer_support(batch).factor_ids

    def exhaustive(batch):
        costs = model.discrete_support_costs(batch)
        indices = costs.argsort(dim=1, stable=True)[:, :4]
        return model.factor_bank[indices]

    with torch.no_grad():
        for _ in range(args.warmup):
            for batch in batches:
                amortized(batch)
                exhaustive(batch)
        _synchronize(torch, device)

        timings: dict[str, float] = {}
        predictions: dict[str, list] = {}
        for name, function in (("amortized", amortized), ("exhaustive_64", exhaustive)):
            started = time.perf_counter()
            for _ in range(args.repeats):
                for batch in batches:
                    function(batch)
            _synchronize(torch, device)
            elapsed = time.perf_counter() - started
            timings[name] = elapsed / (args.repeats * len(tasks))
            predictions[name] = [function(batch).cpu() for batch in batches]

    compatible_sets = [
        {
            rule_program_factor_ids(program)
            for program in version_space(
                task.inference.support[:6], task.privileged.palette
            )
        }
        for task in tasks
    ]
    accuracies: dict[str, float] = {}
    for name, chunks in predictions.items():
        codes = torch.cat(chunks, dim=0).tolist()
        exact = 0
        for rows, compatible in zip(codes, compatible_sets, strict=True):
            predicted = {tuple(int(value) for value in row) for row in rows}
            exact += int(predicted == compatible)
        accuracies[name] = exact / len(tasks)

    result = {
        "experiment": "causal_inference_runtime_benchmark",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "tasks": len(tasks),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "milliseconds_per_task": {
            name: seconds * 1000 for name, seconds in timings.items()
        },
        "amortized_speedup_over_cached_exhaustive": timings["exhaustive_64"]
        / timings["amortized"],
        "exact_version_space_task_rate": accuracies,
        "model_checkpoint": str(checkpoint_path),
        "model_checkpoint_sha256": _sha256_file(checkpoint_path),
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "source_sha256": {
            "prp_wm/discrete_causal_rules.py": _sha256_file(
                REPOSITORY_ROOT / "prp_wm/discrete_causal_rules.py"
            ),
            "scripts/benchmark_causal_inference_speed.py": _sha256_file(
                Path(__file__).resolve()
            ),
        },
        "note": (
            "This compares current CPU implementations with cached state/action "
            "encoding for exhaustive scoring; it is not a hardware-independent "
            "complexity claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
