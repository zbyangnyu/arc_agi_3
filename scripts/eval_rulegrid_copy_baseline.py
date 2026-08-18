#!/usr/bin/env python3
"""Evaluate the deterministic no-change/copy baseline on RuleGrid queries.

This evaluator is intentionally evaluation-only.  It uses the public query
input as its predicted next grid, then reads the requested privileged targets
only to score that prediction.  It is useful for checking whether sparse-grid
cell accuracy merely reflects copying the background.

Example:

    python scripts/eval_rulegrid_copy_baseline.py \
      --split pilot-composition --tasks 192 --diagnostic-indices 21 22 23 \
      --output results/pilot_v2_copy_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.pilot import make_pilot_tasks
from prp_wm.rulegrid import BENCHMARK_VERSION, MASTER_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="pilot-composition")
    parser.add_argument("--tasks", type=int, default=192)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--diagnostic-indices",
        type=int,
        nargs="+",
        default=(21, 22, 23),
        help="Canonical diagnostic-panel indices to evaluate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    if not indices:
        raise SystemExit("--diagnostic-indices cannot be empty")
    if len(set(indices)) != len(indices):
        raise SystemExit("--diagnostic-indices cannot contain duplicates")
    if any(index < 0 or index >= 24 for index in indices):
        raise SystemExit("--diagnostic-indices must lie in 0..23")
    return indices


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
    if args.tasks <= 0:
        raise SystemExit("--tasks must be positive")
    if args.data_master_seed < 0:
        raise SystemExit("--data-master-seed must be non-negative")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free string")
    diagnostic_indices = _validate_indices(tuple(args.diagnostic_indices))

    tasks = make_pilot_tasks(
        split=args.split,
        master_seed=args.data_master_seed,
        start=0,
        count=args.tasks,
        diagnostic_indices=diagnostic_indices,
    )
    total_cells = 0
    correct_cells = 0
    changed_cells = 0
    exact_grids = 0
    scored_grids = 0
    for task in tasks:
        for index in diagnostic_indices:
            state = task.inference.diagnostics[index].state
            target = task.privileged.diagnostic_target_for(index)
            grid_correct = True
            for state_row, target_row in zip(state, target, strict=True):
                for observed, expected in zip(state_row, target_row, strict=True):
                    total_cells += 1
                    changed_cells += int(observed != expected)
                    correct_cells += int(observed == expected)
                    grid_correct = grid_correct and observed == expected
            exact_grids += int(grid_correct)
            scored_grids += 1

    payload: dict[str, object] = {
        "baseline": "deterministic_copy_input_grid",
        "benchmark_version": BENCHMARK_VERSION,
        "cell_accuracy": correct_cells / total_cells,
        "changed_cell_fraction": changed_cells / total_cells,
        "data_master_seed": args.data_master_seed,
        "diagnostic_indices": list(diagnostic_indices),
        "exact_grid_accuracy": exact_grids / scored_grids,
        "scored_grids": scored_grids,
        "scored_tasks": len(tasks),
        "split": args.split,
        "summary_kind": "evaluation_only_sparse_grid_copy_baseline",
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
