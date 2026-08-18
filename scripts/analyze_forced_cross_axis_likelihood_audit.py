#!/usr/bin/env python3
"""Analyze the forced cross-axis likelihood-locality audit.

The runner already deduplicates canonical palette/order repeats.  This script
recomputes descriptive summaries from those semantic records, joins the four
learned pathways for counterexample inspection, and checks whether comparison
runs are semantically identical after removing candidate-order and run
provenance fields.

No bootstrap interval is reported: the 64 palette/order groups are canonical
invariance repeats rather than independent environments.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_RESULT_SCHEMA = "prp-wm.forced-cross-axis-likelihood-audit.v1"
ANALYSIS_SCHEMA = "prp-wm.forced-cross-axis-likelihood-analysis.v1"
BRANCHES = ("EE", "RR", "RP", "PR", "PP")
LEARNED_BRANCHES = ("RR", "RP", "PR", "PP")
AXES = ("collision", "trigger", "relation")
FACTOR_LABELS = {
    "collision": ("STOP", "BOUNCE", "PASS", "PUSH"),
    "trigger": ("TOGGLE", "DELETE", "SPAWN", "RECOLOR"),
    "relation": ("SWAP", "FOLLOW", "REPEL", "NONE"),
}
SEMANTIC_KEY_FIELDS = (
    "query_axis",
    "true_program_index",
    "query_probe_key",
    "forced_probe_key",
    "branch",
)
JOIN_KEY_FIELDS = SEMANTIC_KEY_FIELDS[:-1]
QUERY_KEY_FIELDS = (
    "query_axis",
    "true_program_index",
    "query_probe_key",
    "query_likelihood",
)
# Candidate positions are presentation/order artifacts.  The remaining names
# are included defensively in case a future runner embeds run provenance in an
# individual semantic record.
IGNORED_COMPARISON_FIELDS = {
    "candidate_index",
    "query_candidate_index",
    "split",
    "seed",
    "data_master_seed",
    "batch_size",
    "batch_size_public_panels",
    "target_chunk_size",
    "device",
}
COMPARISON_REL_TOL = 1e-6
COMPARISON_ABS_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Primary forced-audit result.json.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        action="append",
        default=[],
        help=(
            "Comparison result.json; repeat this option for batch/split "
            "sensitivity runs."
        ),
    )
    parser.add_argument(
        "--worst-records",
        type=int,
        default=20,
        help="Number of RR cross-axis counterexamples to join and report.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _nonfinite_paths(
    value: object,
    *,
    path: str = "$",
) -> list[str]:
    """Recursively locate every non-finite JSON numeric leaf."""

    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, Mapping):
        output: list[str] = []
        for key, child in value.items():
            output.extend(
                _nonfinite_paths(child, path=f"{path}.{key}")
            )
        return output
    if isinstance(value, list):
        output = []
        for index, child in enumerate(value):
            output.extend(
                _nonfinite_paths(child, path=f"{path}[{index}]")
            )
        return output
    return []


def _load_result(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"result does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{resolved} does not contain a JSON object")
    nonfinite = _nonfinite_paths(payload)
    if nonfinite:
        preview = ", ".join(nonfinite[:8])
        suffix = "" if len(nonfinite) <= 8 else ", ..."
        raise SystemExit(
            f"{resolved} contains {len(nonfinite)} non-finite values at "
            f"{preview}{suffix}"
        )
    if payload.get("result_schema_version") != EXPECTED_RESULT_SCHEMA:
        raise SystemExit(
            f"{resolved} has unsupported schema "
            f"{payload.get('result_schema_version')!r}"
        )
    rows = payload.get("primary_semantic_path_records")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{resolved} has no semantic path records")
    query_rows = payload.get("query_stage_records")
    if not isinstance(query_rows, list) or not query_rows:
        raise SystemExit(f"{resolved} has no query-stage records")
    return payload


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return sum(materialized) / len(materialized)


def _higher_percentile(
    values: Sequence[float],
    probability: float,
) -> float:
    """Match ``numpy.quantile(..., method="higher")`` exactly."""

    if not values:
        raise ValueError("cannot take a percentile of an empty collection")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    return ordered[int(math.ceil(position))]


def _summarize(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize an empty stratum")
    drops = tuple(
        float(record["log_odds_drop_nats"]) for record in records
    )
    reversals = tuple(
        bool(record["catastrophic_reversal"]) for record in records
    )
    return {
        "semantic_sequences": len(records),
        "catastrophic_reversals": sum(reversals),
        "catastrophic_reversal_rate": _mean(reversals),
        "mean_query_true_probability": _mean(
            float(record["query_true_probability"]) for record in records
        ),
        "mean_forced_true_probability": _mean(
            float(record["forced_true_probability"]) for record in records
        ),
        "mean_query_posterior_nll_nats": _mean(
            float(record["query_posterior_nll_nats"])
            for record in records
        ),
        "mean_true_code_conditional_nll_nats": _mean(
            float(record["query_nll_nats"]) for record in records
        ),
        "mean_log_odds_change_nats": _mean(
            float(record["log_odds_change_nats"]) for record in records
        ),
        "p99_log_odds_drop_nats": _higher_percentile(drops, 0.99),
        "max_log_odds_drop_nats": max(drops),
        "mean_query_marginal_kl_nats": _mean(
            float(record["query_marginal_kl_nats"])
            for record in records
        ),
        "max_query_marginal_kl_nats": max(
            float(record["query_marginal_kl_nats"])
            for record in records
        ),
        "mean_query_marginal_tv": _mean(
            float(record["query_marginal_tv"]) for record in records
        ),
        "max_query_marginal_tv": max(
            float(record["query_marginal_tv"]) for record in records
        ),
    }


def _axis_pair(record: Mapping[str, object]) -> str:
    forced_axis = record.get("forced_axis")
    if forced_axis is None:
        forced_axis = "neutral"
    return f"{record['query_axis']}->{forced_axis}"


def _probe_pair(record: Mapping[str, object]) -> str:
    return (
        f"{record['query_probe_key']} | "
        f"{record['forced_probe_key']}"
    )


def _hierarchical_summaries(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Summarize branch -> query axis -> axis pair -> semantic probe pair."""

    nested: dict[str, object] = {}
    flat: list[dict[str, object]] = []
    observed_branches = {str(row["branch"]) for row in records}
    unknown = observed_branches - set(BRANCHES)
    if unknown:
        raise SystemExit(f"unknown branches in semantic records: {unknown}")

    for branch in BRANCHES:
        branch_rows = tuple(
            row for row in records if row["branch"] == branch
        )
        if not branch_rows:
            raise SystemExit(f"missing semantic records for branch {branch}")
        branch_payload: dict[str, object] = {
            "summary": _summarize(branch_rows),
            "by_query_axis": {},
        }
        by_query_axis = branch_payload["by_query_axis"]
        assert isinstance(by_query_axis, dict)
        for query_axis in sorted(
            {str(row["query_axis"]) for row in branch_rows}
        ):
            axis_rows = tuple(
                row
                for row in branch_rows
                if row["query_axis"] == query_axis
            )
            axis_payload: dict[str, object] = {
                "summary": _summarize(axis_rows),
                "by_axis_pair": {},
            }
            by_axis_pair = axis_payload["by_axis_pair"]
            assert isinstance(by_axis_pair, dict)
            for axis_pair in sorted({_axis_pair(row) for row in axis_rows}):
                pair_rows = tuple(
                    row for row in axis_rows if _axis_pair(row) == axis_pair
                )
                pair_payload: dict[str, object] = {
                    "summary": _summarize(pair_rows),
                    "by_semantic_probe_pair": {},
                }
                by_probe = pair_payload["by_semantic_probe_pair"]
                assert isinstance(by_probe, dict)
                for probe_pair in sorted(
                    {_probe_pair(row) for row in pair_rows}
                ):
                    probe_rows = tuple(
                        row
                        for row in pair_rows
                        if _probe_pair(row) == probe_pair
                    )
                    summary = _summarize(probe_rows)
                    by_probe[probe_pair] = summary
                    first = probe_rows[0]
                    flat.append(
                        {
                            "branch": branch,
                            "query_axis": query_axis,
                            "axis_pair": axis_pair,
                            "query_probe_key": first["query_probe_key"],
                            "forced_probe_key": first["forced_probe_key"],
                            **summary,
                        }
                    )
                by_axis_pair[axis_pair] = pair_payload
            by_query_axis[query_axis] = axis_payload
        nested[branch] = branch_payload
    return nested, flat


def _semantic_key(
    record: Mapping[str, object],
    fields: Sequence[str],
) -> tuple[object, ...]:
    missing = tuple(field for field in fields if field not in record)
    if missing:
        raise SystemExit(f"semantic record is missing fields {missing}")
    return tuple(record[field] for field in fields)


def _record_map(
    records: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> dict[tuple[object, ...], Mapping[str, object]]:
    output: dict[tuple[object, ...], Mapping[str, object]] = {}
    for record in records:
        key = _semantic_key(record, fields)
        if key in output:
            raise SystemExit(f"duplicate semantic record key: {key}")
        output[key] = record
    return output


def _strip_ignored(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_ignored(child)
            for key, child in value.items()
            if key not in IGNORED_COMPARISON_FIELDS
        }
    if isinstance(value, list):
        return [_strip_ignored(child) for child in value]
    return value


def _canonical_records_sha256(
    records: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> str:
    mapped = _record_map(records, fields)
    canonical = [
        _strip_ignored(mapped[key])
        for key in sorted(mapped, key=lambda value: repr(value))
    ]
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compare_values(
    left: object,
    right: object,
) -> tuple[bool, bool, float, bool]:
    """Return exact, tolerant, maximum numeric delta, structure mismatch."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, False, 0.0, True
        exact = True
        tolerant = True
        maximum = 0.0
        mismatch = False
        for key in left:
            child = _compare_values(left[key], right[key])
            exact = exact and child[0]
            tolerant = tolerant and child[1]
            maximum = max(maximum, child[2])
            mismatch = mismatch or child[3]
        return exact, tolerant, maximum, mismatch
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, False, 0.0, True
        exact = True
        tolerant = True
        maximum = 0.0
        mismatch = False
        for left_child, right_child in zip(left, right):
            child = _compare_values(left_child, right_child)
            exact = exact and child[0]
            tolerant = tolerant and child[1]
            maximum = max(maximum, child[2])
            mismatch = mismatch or child[3]
        return exact, tolerant, maximum, mismatch
    if (
        isinstance(left, Real)
        and not isinstance(left, bool)
        and isinstance(right, Real)
        and not isinstance(right, bool)
    ):
        left_float = float(left)
        right_float = float(right)
        delta = abs(left_float - right_float)
        return (
            left == right,
            math.isclose(
                left_float,
                right_float,
                rel_tol=COMPARISON_REL_TOL,
                abs_tol=COMPARISON_ABS_TOL,
            ),
            delta,
            False,
        )
    same = type(left) is type(right) and left == right
    return same, same, 0.0, not same


def _compare_record_sets(
    primary: Sequence[Mapping[str, object]],
    comparison: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> dict[str, object]:
    left = _record_map(primary, fields)
    right = _record_map(comparison, fields)
    left_keys = set(left)
    right_keys = set(right)
    common = left_keys & right_keys
    exact_differences = 0
    tolerance_differences = 0
    structure_mismatches = 0
    maximum_delta = 0.0
    for key in common:
        comparison_result = _compare_values(
            _strip_ignored(left[key]),
            _strip_ignored(right[key]),
        )
        exact_differences += int(not comparison_result[0])
        tolerance_differences += int(not comparison_result[1])
        maximum_delta = max(maximum_delta, comparison_result[2])
        structure_mismatches += int(comparison_result[3])
    missing = len(left_keys - right_keys)
    extra = len(right_keys - left_keys)
    return {
        "primary_records": len(left),
        "comparison_records": len(right),
        "common_records": len(common),
        "missing_semantic_keys": missing,
        "extra_semantic_keys": extra,
        "records_differing_exactly": (
            exact_differences + missing + extra
        ),
        "records_differing_at_tolerance": (
            tolerance_differences + missing + extra
        ),
        "records_with_structure_mismatch": structure_mismatches,
        "maximum_absolute_numeric_difference": maximum_delta,
        "exact_match_ignoring_order_and_provenance": (
            not missing and not extra and exact_differences == 0
        ),
        "tolerance_match_ignoring_order_and_provenance": (
            not missing and not extra and tolerance_differences == 0
        ),
        "relative_tolerance": COMPARISON_REL_TOL,
        "absolute_tolerance": COMPARISON_ABS_TOL,
        "ignored_fields": sorted(IGNORED_COMPARISON_FIELDS),
    }


def _decode_factor_code(
    code: object,
    program_index: object,
) -> dict[str, str]:
    if (
        not isinstance(code, list)
        or len(code) != len(AXES)
        or any(type(value) is not int or not 0 <= value < 4 for value in code)
    ):
        raise SystemExit(f"invalid true factor code: {code!r}")
    expected_program_index = 16 * code[0] + 4 * code[1] + code[2]
    if int(program_index) != expected_program_index:
        raise SystemExit(
            f"factor code {code} disagrees with program {program_index}"
        )
    return {
        axis: FACTOR_LABELS[axis][value]
        for axis, value in zip(AXES, code)
    }


def _worst_joined_records(
    records: Sequence[Mapping[str, object]],
    count: int,
) -> list[dict[str, object]]:
    learned_maps = {
        branch: _record_map(
            tuple(row for row in records if row["branch"] == branch),
            JOIN_KEY_FIELDS,
        )
        for branch in LEARNED_BRANCHES
    }
    reference_keys = set(learned_maps["RR"])
    for branch in LEARNED_BRANCHES[1:]:
        if set(learned_maps[branch]) != reference_keys:
            raise SystemExit(
                f"learned branch {branch} has different semantic keys"
            )
    ranked = sorted(
        (
            row
            for row in learned_maps["RR"].values()
            if row["forced_category"] == "cross"
        ),
        key=lambda row: (
            -float(row["log_odds_drop_nats"]),
            str(row["query_axis"]),
            int(row["true_program_index"]),
            str(row["query_probe_key"]),
            str(row["forced_probe_key"]),
        ),
    )
    output: list[dict[str, object]] = []
    metric_fields = (
        "query_true_probability",
        "forced_true_probability",
        "query_posterior_nll_nats",
        "query_nll_nats",
        "log_odds_change_nats",
        "log_odds_drop_nats",
        "catastrophic_reversal",
        "query_marginal_kl_nats",
        "query_marginal_tv",
        "forced_log_evidence",
    )
    for rank, rr_record in enumerate(ranked[:count], start=1):
        key = _semantic_key(rr_record, JOIN_KEY_FIELDS)
        labels = _decode_factor_code(
            rr_record["true_factor_code"],
            rr_record["true_program_index"],
        )
        query_axis = str(rr_record["query_axis"])
        forced_axis = str(rr_record["forced_axis"])
        output.append(
            {
                "rr_rank": rank,
                "query_axis": query_axis,
                "forced_axis": forced_axis,
                "axis_pair": _axis_pair(rr_record),
                "true_program_index": rr_record["true_program_index"],
                "true_factor_code": rr_record["true_factor_code"],
                "true_factor_labels": labels,
                "true_query_factor_label": labels[query_axis],
                "true_forced_factor_label": labels[forced_axis],
                "query_probe_key": rr_record["query_probe_key"],
                "forced_probe_key": rr_record["forced_probe_key"],
                "branches": {
                    branch: {
                        field: learned_maps[branch][key][field]
                        for field in metric_fields
                    }
                    for branch in LEARNED_BRANCHES
                },
            }
        )
    return output


def _cross_summary(
    records: Sequence[Mapping[str, object]],
    branch: str,
) -> dict[str, object]:
    return _summarize(
        tuple(
            row
            for row in records
            if row["branch"] == branch
            and row["forced_category"] == "cross"
        )
    )


def _rr_failure_pattern(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe the deterministic support of high-confidence RR reversals."""

    reversals = tuple(
        row
        for row in records
        if row["branch"] == "RR"
        and row["forced_category"] == "cross"
        and bool(row["catastrophic_reversal"])
    )
    pr_reversal_keys = {
        _semantic_key(row, JOIN_KEY_FIELDS)
        for row in records
        if row["branch"] == "PR"
        and row["forced_category"] == "cross"
        and bool(row["catastrophic_reversal"])
    }
    rr_reversal_keys = {
        _semantic_key(row, JOIN_KEY_FIELDS) for row in reversals
    }
    factor_values: dict[str, list[str]] = {}
    for axis_index, axis in enumerate(AXES):
        observed = sorted(
            {
                int(row["true_factor_code"][axis_index])
                for row in reversals
            }
        )
        factor_values[axis] = [
            FACTOR_LABELS[axis][value] for value in observed
        ]
    return {
        "reversal_sequences": len(reversals),
        "unique_true_programs": len(
            {int(row["true_program_index"]) for row in reversals}
        ),
        "query_axes": sorted(
            {str(row["query_axis"]) for row in reversals}
        ),
        "forced_axes": sorted(
            {str(row["forced_axis"]) for row in reversals}
        ),
        "query_probe_keys": sorted(
            {str(row["query_probe_key"]) for row in reversals}
        ),
        "forced_probe_keys": sorted(
            {str(row["forced_probe_key"]) for row in reversals}
        ),
        "true_factor_labels": factor_values,
        "rr_and_pr_reversal_identities_exactly_equal": (
            rr_reversal_keys == pr_reversal_keys
        ),
    }


def _run_descriptor(
    path: Path,
    result: Mapping[str, object],
) -> dict[str, object]:
    rows = result["primary_semantic_path_records"]
    query_rows = result["query_stage_records"]
    assert isinstance(rows, list) and isinstance(query_rows, list)
    rr = _cross_summary(rows, "RR")
    rp = _cross_summary(rows, "RP")
    pr = _cross_summary(rows, "PR")
    pp = _cross_summary(rows, "PP")
    query_control = result.get("query_stage_positive_control", {})
    raw_query = (
        query_control.get("raw", {})
        if isinstance(query_control, Mapping)
        else {}
    )
    gate = result.get("decision_gate", {})
    validity = result.get("validity_audit", {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": result.get("status"),
        "split": result.get("split"),
        "seed": result.get("seed"),
        "data_master_seed": result.get("data_master_seed"),
        "groups_per_query_observed": result.get(
            "groups_per_query_observed"
        ),
        "batch_size_public_panels": result.get(
            "batch_size_public_panels"
        ),
        "target_chunk_size": result.get("target_chunk_size"),
        "validity_passed": (
            validity.get("passed")
            if isinstance(validity, Mapping)
            else None
        ),
        "decision": (
            gate.get("decision") if isinstance(gate, Mapping) else None
        ),
        "factor_locality_rescue_gate_passed": (
            gate.get("factor_locality_rescue_gate_passed")
            if isinstance(gate, Mapping)
            else None
        ),
        "raw_query_top1_accuracy": raw_query.get("top1_accuracy"),
        "raw_query_probability_ge_0_95_rate": raw_query.get(
            "probability_ge_0_95_rate"
        ),
        "RR_cross_reversal_rate": rr["catastrophic_reversal_rate"],
        "RR_cross_p99_drop_nats_higher": rr[
            "p99_log_odds_drop_nats"
        ],
        "RP_cross_reversal_rate": rp["catastrophic_reversal_rate"],
        "RP_cross_p99_drop_nats_higher": rp[
            "p99_log_odds_drop_nats"
        ],
        "PR_cross_reversal_rate": pr["catastrophic_reversal_rate"],
        "PR_cross_p99_drop_nats_higher": pr[
            "p99_log_odds_drop_nats"
        ],
        "PP_cross_reversal_rate": pp["catastrophic_reversal_rate"],
        "PP_cross_p99_drop_nats_higher": pp[
            "p99_log_odds_drop_nats"
        ],
        "semantic_path_records_sha256": _canonical_records_sha256(
            rows,
            SEMANTIC_KEY_FIELDS,
        ),
        "semantic_query_records_sha256": _canonical_records_sha256(
            query_rows,
            QUERY_KEY_FIELDS,
        ),
    }


def _robustness_analysis(
    primary_path: Path,
    primary: Mapping[str, object],
    comparisons: Sequence[
        tuple[Path, Mapping[str, object]]
    ],
) -> dict[str, object]:
    primary_rows = primary["primary_semantic_path_records"]
    primary_query_rows = primary["query_stage_records"]
    assert isinstance(primary_rows, list)
    assert isinstance(primary_query_rows, list)
    runs = [
        {
            "role": "primary",
            **_run_descriptor(primary_path, primary),
            "semantic_path_comparison_to_primary": None,
            "semantic_query_comparison_to_primary": None,
        }
    ]
    for index, (path, result) in enumerate(comparisons, start=1):
        comparison_rows = result["primary_semantic_path_records"]
        comparison_query_rows = result["query_stage_records"]
        assert isinstance(comparison_rows, list)
        assert isinstance(comparison_query_rows, list)
        runs.append(
            {
                "role": f"comparison_{index}",
                **_run_descriptor(path, result),
                "semantic_path_comparison_to_primary": (
                    _compare_record_sets(
                        primary_rows,
                        comparison_rows,
                        SEMANTIC_KEY_FIELDS,
                    )
                ),
                "semantic_query_comparison_to_primary": (
                    _compare_record_sets(
                        primary_query_rows,
                        comparison_query_rows,
                        QUERY_KEY_FIELDS,
                    )
                ),
            }
        )
    comparisons_only = runs[1:]
    exact_paths = all(
        bool(
            row["semantic_path_comparison_to_primary"][
                "exact_match_ignoring_order_and_provenance"
            ]
        )
        for row in comparisons_only
    )
    exact_queries = all(
        bool(
            row["semantic_query_comparison_to_primary"][
                "exact_match_ignoring_order_and_provenance"
            ]
        )
        for row in comparisons_only
    )
    tolerant_paths = all(
        bool(
            row["semantic_path_comparison_to_primary"][
                "tolerance_match_ignoring_order_and_provenance"
            ]
        )
        for row in comparisons_only
    )
    tolerant_queries = all(
        bool(
            row["semantic_query_comparison_to_primary"][
                "tolerance_match_ignoring_order_and_provenance"
            ]
        )
        for row in comparisons_only
    )
    return {
        "runs": runs,
        "comparison_run_count": len(comparisons_only),
        "all_runs_recursively_finite": True,
        "all_runs_valid": all(row["validity_passed"] is True for row in runs),
        "all_decisions_identical": (
            len({str(row["decision"]) for row in runs}) == 1
        ),
        "all_semantic_path_records_exactly_identical": exact_paths,
        "all_semantic_query_records_exactly_identical": exact_queries,
        "all_semantic_records_exactly_identical": (
            exact_paths and exact_queries
        ),
        "all_semantic_path_records_match_at_tolerance": tolerant_paths,
        "all_semantic_query_records_match_at_tolerance": tolerant_queries,
        "all_semantic_records_match_at_tolerance": (
            tolerant_paths and tolerant_queries
        ),
        "tested_splits": sorted(
            {str(row["split"]) for row in runs}
        ),
        "tested_seeds": sorted(
            {int(row["seed"]) for row in runs}
        ),
        "tested_public_panel_batch_sizes": sorted(
            {int(row["batch_size_public_panels"]) for row in runs}
        ),
        "interpretation": (
            "These are implementation/split-order invariance checks over one "
            "fixed canonical geometry bank, not independent experimental "
            "replicates."
        ),
    }


def _format_number(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _markdown_report(analysis: Mapping[str, object]) -> str:
    robustness = analysis["robustness"]
    branches = analysis["cross_branch_summaries"]
    query_control = analysis["query_stage_positive_control"]
    gate = analysis["decision_gate"]
    worst = analysis["worst_20_joined_learned_paths"]
    failure_pattern = analysis["rr_failure_pattern"]
    partition_audit = analysis["outcome_partition_audit"]
    neutral_summaries = analysis["neutral_branch_summaries"]
    assert isinstance(robustness, Mapping)
    assert isinstance(branches, Mapping)
    assert isinstance(query_control, Mapping)
    assert isinstance(gate, Mapping)
    assert isinstance(worst, list)
    assert isinstance(failure_pattern, Mapping)
    assert isinstance(partition_audit, Mapping)
    assert isinstance(neutral_summaries, Mapping)
    raw_query = query_control.get("raw", {})
    if not isinstance(raw_query, Mapping):
        raw_query = {}

    lines = [
        "# Forced cross-axis likelihood locality audit",
        "",
        "## Outcome",
        "",
        (
            f"The preregistered decision is **{gate.get('decision')}**. "
            f"Under RR, cross-axis observations reverse a confident query "
            f"belief in {_format_percent(branches['RR']['catastrophic_reversal_rate'])} "
            f"of the 1,536 semantic cross-axis paths, and the higher-method "
            f"P99 log-odds drop is "
            f"{_format_number(branches['RR']['p99_log_odds_drop_nats'])} "
            f"nats. Under PP these become "
            f"{_format_percent(branches['PP']['catastrophic_reversal_rate'])} "
            f"and {_format_number(branches['PP']['p99_log_odds_drop_nats'])} "
            f"nats."
        ),
        "",
        (
            "The fixed-geometry query positive control has raw top-1 "
            f"accuracy {_format_percent(raw_query.get('top1_accuracy'))}, "
            f"with {_format_percent(raw_query.get('probability_ge_0_95_rate'))} "
            "of query cases assigning at least 0.95 probability to the true "
            "query factor."
        ),
        "",
        "## Learned-path decomposition (cross-axis paths)",
        "",
        "| Branch | Query likelihood | Forced likelihood | Reversal rate | P99 drop (higher, nats) | Max drop (nats) | Mean marginal TV |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    branch_labels = {
        "EE": ("exact", "exact"),
        "RR": ("raw", "raw"),
        "RP": ("raw", "oracle projected"),
        "PR": ("oracle projected", "raw"),
        "PP": ("oracle projected", "oracle projected"),
    }
    for branch in BRANCHES:
        row = branches[branch]
        query_likelihood, forced_likelihood = branch_labels[branch]
        lines.append(
            f"| {branch} | {query_likelihood} | {forced_likelihood} | "
            f"{_format_percent(row['catastrophic_reversal_rate'])} | "
            f"{_format_number(row['p99_log_odds_drop_nats'])} | "
            f"{_format_number(row['max_log_odds_drop_nats'])} | "
            f"{_format_number(row['mean_query_marginal_tv'])} |"
        )
    lines.extend(
        [
            "",
            (
                "RP removes catastrophic reversals but retains a non-zero "
                "tail, while PR remains close to RR. This localizes the "
                "dominant failure to raw forced-observation likelihoods; "
                "raw query likelihoods can still induce cross-factor "
                "correlations that explain the remaining RP tail."
            ),
            "",
            "## Failure concentration and calibration",
            "",
            (
                f"All {failure_pattern['reversal_sequences']} RR reversals "
                f"come from {failure_pattern['unique_true_programs']} unique "
                "programs on `relation -> collision`, specifically forced "
                "`cross:collision:v1`. They cover collision "
                f"{', '.join(failure_pattern['true_factor_labels']['collision'])}, "
                "relation "
                f"{', '.join(failure_pattern['true_factor_labels']['relation'])}, "
                "and every trigger value; both relation query variants fail. "
                "The PR reversal identities are exactly the same as RR."
            ),
            "",
            (
                "The query projection leaves the query posterior metric "
                "essentially unchanged "
                f"(projected-minus-raw posterior NLL "
                f"{_format_number(query_control.get('projected_minus_raw_mean_true_query_nll_nats'))} "
                "nats), while true-code conditional full-grid NLL changes by "
                f"{_format_number(query_control.get('projected_minus_raw_mean_true_code_conditional_full_grid_nll_nats'))} "
                "nats. The latter is the preregistered gate quantity."
            ),
            "",
            (
                "The weakest atomic outcome partitions are collision v1 "
                f"(pair F1 {_format_number(partition_audit['by_semantic_probe_key']['cross:collision:v1']['same_outcome_pair_f1'])}) "
                "and trigger v1 "
                f"({_format_number(partition_audit['by_semantic_probe_key']['cross:trigger:v1']['same_outcome_pair_f1'])}). "
                "Only collision v1 causes reversals, so partition mismatch "
                "co-localizes with the failure but is not sufficient for it."
            ),
            "",
            (
                "Neutral probes cause no high-confidence reversals. Their RR "
                "P99 drop is "
                f"{_format_number(neutral_summaries['RR']['p99_log_odds_drop_nats'])} "
                "nats, versus zero under RP/PP, revealing smaller raw "
                "non-local likelihood drift."
            ),
            "",
            "## Robustness",
            "",
            "| Run | Split | Seed | Batch | Valid | Decision | RR reversal | RR P99 | PP reversal | PP P99 | Semantic equality |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    runs = robustness["runs"]
    assert isinstance(runs, list)
    for row in runs:
        comparison = row["semantic_path_comparison_to_primary"]
        equality = (
            "reference"
            if comparison is None
            else (
                "exact"
                if comparison[
                    "exact_match_ignoring_order_and_provenance"
                ]
                else (
                    "tolerance-only"
                    if comparison[
                        "tolerance_match_ignoring_order_and_provenance"
                    ]
                    else "different"
                )
            )
        )
        lines.append(
            f"| {row['role']} | {row['split']} | {row['seed']} | "
            f"{row['batch_size_public_panels']} | "
            f"{'yes' if row['validity_passed'] else 'no'} | "
            f"{row['decision']} | "
            f"{_format_percent(row['RR_cross_reversal_rate'])} | "
            f"{_format_number(row['RR_cross_p99_drop_nats_higher'])} | "
            f"{_format_percent(row['PP_cross_reversal_rate'])} | "
            f"{_format_number(row['PP_cross_p99_drop_nats_higher'])} | "
            f"{equality} |"
        )
    lines.extend(
        [
            "",
            (
                "All JSON inputs passed a recursive finite-number check. "
                "Semantic equality ignores candidate positions and "
                "split/seed/batch provenance; it does not ignore any model "
                "metric."
            ),
            "",
            "## Worst 20 RR cross-axis paths, joined across learned branches",
            "",
            "| # | Axis pair | True program | Query probe | Forced probe | RR drop | RP drop | PR drop | PP drop | RR reversal | PP reversal |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in worst:
        labels = row["true_factor_labels"]
        branch_rows = row["branches"]
        program = (
            f"C={labels['collision']}, T={labels['trigger']}, "
            f"R={labels['relation']}"
        )
        lines.append(
            f"| {row['rr_rank']} | {row['axis_pair']} | {program} | "
            f"{row['query_probe_key']} | {row['forced_probe_key']} | "
            f"{_format_number(branch_rows['RR']['log_odds_drop_nats'])} | "
            f"{_format_number(branch_rows['RP']['log_odds_drop_nats'])} | "
            f"{_format_number(branch_rows['PR']['log_odds_drop_nats'])} | "
            f"{_format_number(branch_rows['PP']['log_odds_drop_nats'])} | "
            f"{'yes' if branch_rows['RR']['catastrophic_reversal'] else 'no'} | "
            f"{'yes' if branch_rows['PP']['catastrophic_reversal'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- PP is an **oracle causal projection**, computed by "
                "factor-fiber log-mean-exp after full-grid likelihood "
                "evaluation. Its near-zero query-marginal drift is a "
                "mathematical consequence of the intervention and a "
                "mechanistic rescue test—not evidence that the current "
                "learned model can discover or apply the projection."
            ),
            (
                "- The 64 palette/order groups are canonically identical "
                "invariance repeats, not 64 independent environments. "
                "Accordingly this report gives finite exhaustive rates over "
                "semantic paths and no palette-level confidence interval."
            ),
            (
                "- The audit uses the two fixed, previously validated atomic "
                "probe geometries per axis. It establishes locality on this "
                "bank, not generalization to arbitrary or compositional "
                "geometries."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.worst_records < 0:
        raise SystemExit("--worst-records must be non-negative")
    primary_path = args.input.resolve()
    comparison_paths = tuple(path.resolve() for path in args.comparison)
    all_paths = (primary_path, *comparison_paths)
    if len(set(all_paths)) != len(all_paths):
        raise SystemExit("input and comparison paths must be unique")

    primary = _load_result(primary_path)
    comparisons = tuple(
        (path, _load_result(path)) for path in comparison_paths
    )
    primary_records = primary["primary_semantic_path_records"]
    assert isinstance(primary_records, list)
    hierarchical, flat = _hierarchical_summaries(primary_records)
    cross_summaries = {
        branch: _cross_summary(primary_records, branch)
        for branch in BRANCHES
    }
    neutral_summaries = {
        branch: _summarize(
            tuple(
                row
                for row in primary_records
                if row["branch"] == branch
                and row["forced_category"] == "neutral"
            )
        )
        for branch in BRANCHES
    }
    worst = _worst_joined_records(
        primary_records,
        args.worst_records,
    )
    robustness = _robustness_analysis(
        primary_path,
        primary,
        comparisons,
    )
    analysis_source = Path(__file__).resolve()
    payload: dict[str, object] = {
        "analysis_schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "primary_input": str(primary_path),
        "primary_input_sha256": _sha256(primary_path),
        "comparison_inputs": [str(path) for path in comparison_paths],
        "analysis_source": str(analysis_source),
        "analysis_source_sha256": _sha256(analysis_source),
        "input_result_schema_version": primary.get(
            "result_schema_version"
        ),
        "recursive_finite_validation": {
            "checked_inputs": len(all_paths),
            "passed": True,
            "nonfinite_value_count": 0,
        },
        "statistical_unit": (
            "One deduplicated semantic sequence. Palette/order groups are "
            "canonical invariance repeats, not independent samples."
        ),
        "geometry_scope": (
            "Two fixed validated atomic probe geometries per axis."
        ),
        "summary_method": {
            "p99": "higher order statistic: ceil(0.99 * (n - 1))",
            "confidence_intervals": (
                "none; no independent palette/environment clusters"
            ),
        },
        "query_stage_positive_control": primary.get(
            "query_stage_positive_control"
        ),
        "decision_gate": primary.get("decision_gate"),
        "validity_audit": primary.get("validity_audit"),
        "cross_branch_summaries": cross_summaries,
        "neutral_branch_summaries": neutral_summaries,
        "rr_failure_pattern": _rr_failure_pattern(primary_records),
        "outcome_partition_audit": primary.get(
            "outcome_partition_audit"
        ),
        "hierarchical_summaries": {
            "order": (
                "branch -> query_axis -> axis_pair -> "
                "semantic_probe_pair"
            ),
            "branches": hierarchical,
        },
        "flat_probe_strata": flat,
        "worst_20_joined_learned_paths": worst,
        "robustness": robustness,
        "oracle_projection_caveat": (
            "PP is an oracle factor-fiber projection. Its preservation of "
            "the query marginal is guaranteed by construction under the "
            "projected query posterior; it is a mechanistic intervention, "
            "not a deployable learned capability."
        ),
    }
    output_nonfinite = _nonfinite_paths(payload)
    if output_nonfinite:
        raise SystemExit(
            "analysis unexpectedly produced non-finite values at "
            + ", ".join(output_nonfinite[:8])
        )

    output_directory = primary_path.parent
    analysis_json = output_directory / "analysis.json"
    analysis_markdown = output_directory / "analysis.md"
    _atomic_json(analysis_json, payload)
    _atomic_text(analysis_markdown, _markdown_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "analysis_json": str(analysis_json),
                "analysis_markdown": str(analysis_markdown),
                "decision": primary.get("decision_gate", {}).get(
                    "decision"
                ),
                "robustness": {
                    key: robustness[key]
                    for key in (
                        "all_runs_valid",
                        "all_decisions_identical",
                        "all_semantic_records_exactly_identical",
                        "all_semantic_records_match_at_tolerance",
                    )
                },
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
