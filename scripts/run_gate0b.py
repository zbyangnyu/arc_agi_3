#!/usr/bin/env python3
"""Run the exact, pre-registered Stage 0-B RuleGrid headroom gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid_evaluation import evaluate_gate0b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026071701)
    parser.add_argument("--output", type=Path, default=Path("results/gate0b_seed0.json"))
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print JSON only instead of also materializing the report file",
    )
    args = parser.parse_args()
    report = evaluate_gate0b(
        repeats=args.repeats,
        budget=args.budget,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n"
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
