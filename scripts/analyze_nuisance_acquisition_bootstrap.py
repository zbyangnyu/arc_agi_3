#!/usr/bin/env python3
"""Cluster-bootstrap the paired nuisance-acquisition experiment.

The primary runner emits one row per seed, hidden query value, policy, and
budget.  Seeds and hidden values are repeated measurements of the same public
environment, so this analysis first averages them within
``(query_axis, group_index)`` and then resamples environment groups within
each query-axis stratum.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable


POLICIES = (
    "query-conditioned",
    "global-information-gain",
    "uniform",
)
PAIRS = (
    ("query-conditioned", "global-information-gain"),
    ("query-conditioned", "uniform"),
    ("global-information-gain", "uniform"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260876)
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return sum(materialized) / len(materialized)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty collection")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _interval(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "lower": _percentile(ordered, 0.025),
        "upper": _percentile(ordered, 0.975),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.replicates <= 0 or args.seed < 0:
        raise SystemExit("--replicates must be positive and --seed non-negative")
    input_path = args.input.resolve()
    source_path = Path(__file__).resolve()
    result = json.loads(input_path.read_text(encoding="utf-8"))
    rows = result.get("bootstrap_task_records")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("input has no bootstrap_task_records")
    budgets = tuple(int(value) for value in result["budgets"])
    axes = tuple(sorted({str(row["query_axis"]) for row in rows}))

    repeated: dict[
        tuple[str, int, int, str],
        list[float],
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["query_axis"]),
            int(row["group_index"]),
            int(row["budget"]),
            str(row["policy"]),
        )
        repeated[key].append(float(bool(row["won"])))

    groups_by_axis = {
        axis: tuple(
            sorted(
                {
                    group
                    for candidate_axis, group, _, _ in repeated
                    if candidate_axis == axis
                }
            )
        )
        for axis in axes
    }
    expected_repeated = (
        len(tuple(result["seeds"])) * int(result["programs_per_group"])
    )
    cluster_values: dict[tuple[str, int, int, str], float] = {}
    for key, values in repeated.items():
        if len(values) != expected_repeated:
            raise SystemExit(
                f"cluster {key} has {len(values)} rows, expected "
                f"{expected_repeated}"
            )
        cluster_values[key] = _mean(values)

    for axis, groups in groups_by_axis.items():
        for group in groups:
            for budget in budgets:
                for policy in POLICIES:
                    if (axis, group, budget, policy) not in cluster_values:
                        raise SystemExit(
                            f"missing cluster cell {(axis, group, budget, policy)}"
                        )

    estimates: dict[str, dict[str, object]] = {}
    bootstrap_samples: dict[tuple[int, str], list[float]] = {
        (budget, policy): []
        for budget in budgets
        for policy in POLICIES
    }
    paired_samples: dict[tuple[int, str, str], list[float]] = {
        (budget, left, right): []
        for budget in budgets
        for left, right in PAIRS
    }

    for budget in budgets:
        policy_estimates = {
            policy: _mean(
                cluster_values[(axis, group, budget, policy)]
                for axis in axes
                for group in groups_by_axis[axis]
            )
            for policy in POLICIES
        }
        estimates[str(budget)] = {
            "policies": policy_estimates,
            "paired_differences": {
                f"{left}_minus_{right}": (
                    policy_estimates[left] - policy_estimates[right]
                )
                for left, right in PAIRS
            },
        }

    generator = random.Random(args.seed)
    for _ in range(args.replicates):
        sampled_clusters = tuple(
            (axis, generator.choice(groups_by_axis[axis]))
            for axis in axes
            for _ in groups_by_axis[axis]
        )
        for budget in budgets:
            sampled_policy_means = {
                policy: _mean(
                    cluster_values[(axis, group, budget, policy)]
                    for axis, group in sampled_clusters
                )
                for policy in POLICIES
            }
            for policy, value in sampled_policy_means.items():
                bootstrap_samples[(budget, policy)].append(value)
            for left, right in PAIRS:
                paired_samples[(budget, left, right)].append(
                    sampled_policy_means[left] - sampled_policy_means[right]
                )

    for budget in budgets:
        row = estimates[str(budget)]
        row["policy_95_percentile_ci"] = {
            policy: _interval(bootstrap_samples[(budget, policy)])
            for policy in POLICIES
        }
        row["paired_difference_95_percentile_ci"] = {
            f"{left}_minus_{right}": _interval(
                paired_samples[(budget, left, right)]
            )
            for left, right in PAIRS
        }

    payload: dict[str, object] = {
        "analysis_schema_version": (
            "prp-wm.nuisance-acquisition-cluster-bootstrap.v1"
        ),
        "status": "complete",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "input_result_schema_version": result.get("result_schema_version"),
        "analysis_source": str(source_path),
        "analysis_source_sha256": _sha256(source_path),
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.replicates,
        "confidence_level": 0.95,
        "interval": "percentile",
        "resampling_unit": "(query_axis, group_index)",
        "stratification": "query_axis",
        "repeated_measure_aggregation": (
            "mean over all candidate-order seeds and hidden query slots "
            "before environment resampling"
        ),
        "query_axes": list(axes),
        "groups_per_axis": {
            axis: len(groups) for axis, groups in groups_by_axis.items()
        },
        "repeated_rows_per_environment_policy_budget": expected_repeated,
        "estimates": estimates,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
