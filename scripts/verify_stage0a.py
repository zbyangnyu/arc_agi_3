#!/usr/bin/env python3
"""Verify the frozen Stage 0-A result, interpreter, and source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.reproducibility import (
    ReproducibilityError,
    canonical_json_bytes,
    load_stage0a_config,
    verify_stage0a_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage0a.json",
        help="Frozen Stage 0-A JSON config (default: configs/stage0a.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_stage0a_config(args.config)
        verification = verify_stage0a_reference(REPOSITORY_ROOT, config)
    except ReproducibilityError as error:
        raise SystemExit(f"Stage 0-A verification failed: {error}") from error
    sys.stdout.buffer.write(canonical_json_bytes(verification))


if __name__ == "__main__":
    main()
