#!/usr/bin/env python3
"""Run the exact GF(2)-RuleProbe Stage 0 gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.evaluation import evaluate_gate0
from prp_wm.reproducibility import (
    ReproducibilityError,
    canonical_json_bytes,
    load_stage0a_config,
    sha256_bytes,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/stage0a.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Frozen Stage 0-A JSON config (default: configs/stage0a.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write canonical JSON result bytes.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fail unless the frozen configuration reproduces its expected SHA256.",
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gate-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_stage0a_config(args.config)
    except ReproducibilityError as error:
        raise SystemExit(f"invalid Stage 0-A config: {error}") from error

    overrides = {
        "trials": args.trials,
        "repeats": args.repeats,
        "budget": args.budget,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "gate_threshold": args.gate_threshold,
    }
    if args.verify and any(value is not None for value in overrides.values()):
        raise SystemExit("--verify cannot be combined with evaluation overrides")
    evaluation = dict(config.evaluation)
    evaluation.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    report = evaluate_gate0(
        **evaluation,
    )
    payload = canonical_json_bytes(report.to_dict())
    if args.verify:
        observed_hash = sha256_bytes(payload)
        if observed_hash != config.expected_result_sha256:
            raise SystemExit(
                "Stage 0-A result hash mismatch: "
                f"expected {config.expected_result_sha256}, got {observed_hash}"
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    main()
