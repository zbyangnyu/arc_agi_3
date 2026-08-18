#!/usr/bin/env python3
"""Materialize and audit the isolated randomized-geometry executor protocol.

This script generates no checkpoint and starts no training.  It writes only a
JSON manifest for singleton/pair training panels and triple-only evaluation
panels built by ``prp_wm.random_geometry_protocol``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SCHEMA_VERSION = "prp-wm.random-geometry-protocol-audit.v1"
AUDITED_SOURCES = (
    "prp_wm/random_geometry_protocol.py",
    "prp_wm/rulegrid.py",
    "scripts/audit_random_geometry_executor_protocol.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    if start < 0 or count <= 0:
        raise ValueError("geometry seed starts must be non-negative and counts positive")
    return tuple(range(start, start + count))


def _layout_coverage_gate(audit: dict[str, object]) -> dict[str, object]:
    expected_directions = {"E", "N", "S", "W"}
    expected_motion_shapes = {"single-cell", "domino-perpendicular"}
    split_results: dict[str, object] = {}
    passed = True
    for split in ("train", "eval"):
        coverage = audit[f"{split}_layout_coverage"]
        assert isinstance(coverage, dict)
        axis_results: dict[str, object] = {}
        for axis in ("collision", "trigger", "relation"):
            values = coverage[axis]
            assert isinstance(values, dict)
            directions = set(values["directions"])
            shapes = set(values["shapes"])
            anchors = int(values["unique_anchor_count"])
            if axis == "trigger":
                axis_passed = anchors >= 4 and shapes == {"trigger-payload-socket"}
            else:
                axis_passed = (
                    anchors >= 4
                    and directions == expected_directions
                    and shapes == expected_motion_shapes
                )
            axis_results[axis] = {
                "minimum_unique_anchor_count": 4,
                "observed_unique_anchor_count": anchors,
                "expected_directions": (
                    sorted(expected_directions) if axis != "trigger" else []
                ),
                "observed_directions": sorted(directions),
                "expected_shapes": (
                    sorted(expected_motion_shapes)
                    if axis != "trigger"
                    else ["trigger-payload-socket"]
                ),
                "observed_shapes": sorted(shapes),
                "passed": axis_passed,
            }
            passed = passed and axis_passed
        split_results[split] = axis_results
    return {"splits": split_results, "passed": passed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seed-start", type=int, default=100_000)
    parser.add_argument("--train-seed-count", type=int, default=64)
    parser.add_argument("--eval-seed-start", type=int, default=200_000)
    parser.add_argument("--eval-seed-count", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_seeds = _seed_range(args.train_seed_start, args.train_seed_count)
        eval_seeds = _seed_range(args.eval_seed_start, args.eval_seed_count)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if set(train_seeds).intersection(eval_seeds):
        raise SystemExit("train/eval geometry seed ranges must be disjoint")

    from prp_wm.random_geometry_protocol import (
        audit_random_geometry_dataset,
        build_random_geometry_dataset,
    )

    dataset = build_random_geometry_dataset(
        train_geometry_seeds=train_seeds,
        eval_geometry_seeds=eval_seeds,
    )
    audit = audit_random_geometry_dataset(dataset)
    layout_gate = _layout_coverage_gate(audit)
    protocol_gate = bool(
        audit["static_gates"]["passed"]  # type: ignore[index]
        and layout_gate["passed"]
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "geometry_randomized_executor_dataset_protocol_audit",
        "training_started": False,
        "checkpoint_written": False,
        "existing_audited_runtime_modified": False,
        "train_target_scope": "singleton_and_pair_mechanism_events_only",
        "eval_target_scope": "triple_composition_only",
        "palette_mode": "privileged_default_role_canonicalized",
        "factor_axes_and_value_codebook_given": True,
        "geometry_seed_metadata_exposed_to_model": False,
        "geometry_hash_metadata_exposed_to_model": False,
        "split_metadata_exposed_to_model": False,
        "train_geometry_seed_start": args.train_seed_start,
        "train_geometry_seed_count": args.train_seed_count,
        "eval_geometry_seed_start": args.eval_seed_start,
        "eval_geometry_seed_count": args.eval_seed_count,
        "source_sha256": {
            relative: _sha256_file(REPOSITORY_ROOT / relative)
            for relative in AUDITED_SOURCES
        },
        "dataset_audit": audit,
        "layout_coverage_gate": layout_gate,
        "overall_protocol_gate": {
            "requires": ["dataset_audit.static_gates", "layout_coverage_gate"],
            "passed": protocol_gate,
        },
        "interpretation": (
            "A pass certifies an ID-free, split-exclusive public geometry stream "
            "with singleton/pair train targets and triple-only evaluation targets. "
            "It does not certify executor learning or causal-variable discovery."
        ),
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": _sha256_file(output),
                "train_panel_count": audit["train_panel_count"],
                "eval_panel_count": audit["eval_panel_count"],
                "train_example_count": audit["train_example_count"],
                "eval_example_count": audit["eval_example_count"],
                "overall_protocol_gate": payload["overall_protocol_gate"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
