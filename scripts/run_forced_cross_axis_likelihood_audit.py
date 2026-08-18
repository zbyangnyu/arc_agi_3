#!/usr/bin/env python3
"""Audit whether irrelevant RuleGrid observations corrupt query beliefs.

This runner deliberately contains no acquisition selector.  For every query
axis and true 64-factor program it forces one of two strong query-axis probes,
then forks that posterior into six independent continuations: four atomic
probes of the other axes and two rule-independent neutral probes.

The learned executor is evaluated once per public canonical panel.  Its proper
full-grid likelihood is either used raw or projected onto the oracle factor
fiber acted on by the probe.  The learned controls are therefore:

* RR: raw query likelihood, raw forced likelihood;
* RP: raw query likelihood, projected forced likelihood;
* PR: projected query likelihood, raw forced likelihood; and
* PP: projected query likelihood, projected forced likelihood.

EE is the exact deterministic simulator control.  Palette groups are repeated
canonical/order/batch invariance checks, not independent environments.  Main
rates are deduplicated by semantic probe key and contain exactly
``3 * 64 * 2 * 6 = 2304`` records per branch.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.run_oracle_canonical_acquisition_ceiling import (  # noqa: E402
    DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    _load_active_executor,
    _normalise_log_weights,
    bayesian_log_likelihood_update,
)
from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (  # noqa: E402
    CANDIDATES_PER_TASK,
    _atomic_probe,
    _exact_candidate_outcome_maps,
    _neutral_support,
)
from scripts.run_nuisance_learned_bridge import (  # noqa: E402
    _finalise_partition_counter,
    _new_partition_counter,
    _update_partition_counter,
)


RESULT_SCHEMA_VERSION = "prp-wm.forced-cross-axis-likelihood-audit.v1"
DEFAULT_SPLIT = "forced-cross-axis-likelihood-audit-heldout-v1"
DEFAULT_GROUPS = 64
DEFAULT_SEED = 2026072401
PROGRAMS_PER_PUBLIC_PANEL = 64
QUERY_PROBES = 2
FORCED_PROBES = 6
BRANCHES = ("EE", "RR", "RP", "PR", "PP")
LEARNED_BRANCHES = ("RR", "RP", "PR", "PP")
REVERSAL_CONFIDENCE = 0.90
REVERSAL_RATE_GATE = 0.01
P99_DROP_GATE_NATS = 5.0
RESCUE_FRACTION_GATE = 0.90
QUERY_NLL_DEGRADATION_GATE_NATS = 0.05
_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/matched_executor.py",
    "prp_wm/neural.py",
    "prp_wm/routed_executor.py",
    "prp_wm/rulegrid.py",
    "scripts/run_active_support_calibrated_executor.py",
    "scripts/run_nuisance_learned_bridge.py",
    "scripts/run_oracle_canonical_acquisition_ceiling.py",
    "scripts/run_oracle_canonical_nuisance_acquisition_ceiling.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-executor-checkpoint",
        type=Path,
        default=DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-query", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-master-seed", type=int, default=2026072401)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--target-chunk-size", type=int, default=8)
    parser.add_argument("--worst-records", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.groups_per_query <= 0:
        raise SystemExit("--groups-per-query must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.target_chunk_size <= 0:
        raise SystemExit("--target-chunk-size must be positive")
    if args.worst_records < 0:
        raise SystemExit("--worst-records must be non-negative")
    if args.seed < 0 or args.data_master_seed < 0:
        raise SystemExit("seeds must be non-negative")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_sha256() -> dict[str, str]:
    return {
        relative: _sha256(REPOSITORY_ROOT / relative)
        for relative in _AUDITED_SOURCE_FILES
    }


def _mean(values: Iterable[float]) -> float:
    rows = tuple(float(value) for value in values)
    return sum(rows) / len(rows) if rows else 0.0


def percentile(values: Sequence[float], percentage: float) -> float:
    """Deterministic ``higher`` percentile used by the preregistration."""

    if not values:
        return 0.0
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must lie in [0,100]")
    ordered = sorted(float(value) for value in values)
    # Match numpy.quantile(..., method="higher"): the selected observation is
    # never interpolated downward, which is important for rare catastrophes.
    position = (len(ordered) - 1) * percentage / 100.0
    return ordered[int(math.ceil(position))]


def _semantic_candidate_menu(
    *,
    query_axis: Any,
    palette: Any,
    order_seed: int,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Return the frozen eight probes plus shuffle-invariant semantic keys."""

    from prp_wm.rulegrid import ALL_AXES, RuleGridProbe, _neutral_probe

    rows: list[tuple[str, Any]] = []
    for variant in range(QUERY_PROBES):
        rows.append(
            (
                f"query:{query_axis.value}:v{variant}",
                _atomic_probe(
                    query_axis,
                    f"private-query-{variant}",
                    palette,
                    variant=variant,
                ),
            )
        )
    for axis in ALL_AXES:
        if axis is query_axis:
            continue
        for variant in range(2):
            rows.append(
                (
                    f"cross:{axis.value}:v{variant}",
                    _atomic_probe(
                        axis,
                        f"private-cross-{axis.value}-{variant}",
                        palette,
                        variant=variant,
                    ),
                )
            )
    rows.extend(
        (
            (
                "neutral:v0",
                _neutral_probe("private-neutral-0", palette, row=2, col=3),
            ),
            (
                "neutral:v1",
                _neutral_probe("private-neutral-1", palette, row=4, col=3),
            ),
        )
    )
    if len(rows) != CANDIDATES_PER_TASK:
        raise AssertionError("forced audit menu must have eight probes")
    random.Random(order_seed).shuffle(rows)
    candidates = tuple(
        RuleGridProbe(f"C{index:02d}", probe.state, probe.action)
        for index, (_, probe) in enumerate(rows)
    )
    keys = tuple(key for key, _ in rows)
    if len(set(keys)) != CANDIDATES_PER_TASK:
        raise AssertionError("semantic probe keys must be unique")
    return candidates, keys


def _build_forced_audit_tasks(
    query_axis: Any,
    *,
    groups: int,
    split: str,
    master_seed: int,
    candidate_seed: int,
) -> tuple[Any, ...]:
    """Build 64 hidden programs for every shared public canonical panel."""

    from prp_wm.rulegrid import (
        ALL_PROGRAMS,
        RuleGridInferenceView,
        RuleGridPrivilegedTargets,
        RuleGridTask,
        derive_seed64,
        palette_from_seed,
        simulate,
    )

    if groups <= 0:
        raise ValueError("groups must be positive")
    tasks = []
    for group_index in range(groups):
        palette = palette_from_seed(
            derive_seed64(
                split,
                query_axis,
                group_index,
                "palette",
                master_seed=master_seed,
            )
        )
        candidates, semantic_keys = _semantic_candidate_menu(
            query_axis=query_axis,
            palette=palette,
            order_seed=(
                derive_seed64(
                    split,
                    query_axis,
                    group_index,
                    "candidate_order",
                    master_seed=master_seed,
                )
                ^ candidate_seed
            ),
        )
        shared_support = _neutral_support(ALL_PROGRAMS[0], palette)
        for program_index, program in enumerate(ALL_PROGRAMS):
            if _neutral_support(program, palette) != shared_support:
                raise AssertionError("neutral support leaked the hidden program")
            targets = tuple(
                simulate(probe.state, probe.action, program, palette)
                for probe in candidates
            )
            tasks.append(
                RuleGridTask(
                    inference=RuleGridInferenceView(
                        task_id=(
                            f"{split}/Q{query_axis.value}/"
                            f"G{group_index:04d}"
                        ),
                        support=shared_support,
                        active_candidates=candidates,
                        diagnostics=(),
                    ),
                    privileged=RuleGridPrivilegedTargets(
                        true_program=program,
                        palette=palette,
                        candidate_kinds=semantic_keys,
                        active_targets=targets,
                        diagnostic_targets=(),
                        diagnostic_target_indices=(),
                    ),
                )
            )
            if program_index >= PROGRAMS_PER_PUBLIC_PANEL:
                raise AssertionError("unexpected RuleGrid program count")
    expected = groups * PROGRAMS_PER_PUBLIC_PANEL
    if len(tasks) != expected:
        raise AssertionError(f"expected {expected} tasks, got {len(tasks)}")
    return tuple(tasks)


def _assert_public_program_copies(
    torch: Any,
    tensor: Any | None,
    *,
    name: str,
) -> None:
    """Assert all 64 hidden programs share one public input per group."""

    if tensor is None:
        return
    if tensor.ndim < 1 or tensor.shape[0] % PROGRAMS_PER_PUBLIC_PANEL:
        raise AssertionError(
            f"{name} batch must be divisible by "
            f"{PROGRAMS_PER_PUBLIC_PANEL}"
        )
    grouped = tensor.reshape(
        tensor.shape[0] // PROGRAMS_PER_PUBLIC_PANEL,
        PROGRAMS_PER_PUBLIC_PANEL,
        *tensor.shape[1:],
    )
    if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
        raise AssertionError(f"{name} leaked the hidden program")


def axis_project_log_likelihood(
    torch: Any,
    log_likelihood: Any,
    factor_bank: Any,
    axis_index: int | None,
) -> Any:
    """Project a full-grid likelihood onto one oracle factor fiber.

    Projection is performed *after* full-grid ``log_prob``.  For an axis probe
    it computes ``logsumexp - log(16)`` over the other two factors and
    broadcasts the value back to all codes in that fiber.  For a neutral probe
    it computes one ``logsumexp - log(64)`` constant, so Bayes leaves any prior
    exactly unchanged up to floating-point normalization.
    """

    if log_likelihood.ndim < 1:
        raise ValueError("likelihood needs a hypothesis axis")
    if factor_bank.ndim != 2 or factor_bank.shape[1] != 3:
        raise ValueError("factor bank must have [K,3] shape")
    if log_likelihood.shape[-1] != factor_bank.shape[0]:
        raise ValueError("likelihood and factor bank disagree")
    if not bool(torch.isfinite(log_likelihood).all().item()):
        raise ValueError("projection requires finite learned likelihoods")
    if axis_index is None:
        constant = torch.logsumexp(log_likelihood, dim=-1, keepdim=True)
        constant = constant - math.log(float(log_likelihood.shape[-1]))
        return constant.expand_as(log_likelihood)
    if type(axis_index) is not int or axis_index not in range(3):
        raise ValueError("axis_index must be None or an integer in [0,3)")
    bank = factor_bank.to(device=log_likelihood.device)
    projected = torch.empty_like(log_likelihood)
    for factor_value in range(4):
        mask = bank[:, axis_index].eq(factor_value)
        count = int(mask.sum().item())
        if count <= 0:
            raise AssertionError("factor fiber is empty")
        value = torch.logsumexp(log_likelihood[..., mask], dim=-1)
        value = value - math.log(float(count))
        projected[..., mask] = value[..., None]
    return projected


def exact_panel_log_likelihoods(
    torch: Any,
    exact_maps: Any,
    feedback: Any,
    *,
    target_chunk_size: int = 8,
) -> Any:
    """Score ``[G,T,P]`` feedback under exact ``[G,P,K]`` maps."""

    if exact_maps.ndim != 5 or feedback.ndim != 5:
        raise ValueError("maps and feedback need [G,P/K,H,W] panel shapes")
    groups, probes, hypotheses, height, width = exact_maps.shape
    if feedback.shape[0] != groups or feedback.shape[2:] != (
        probes,
        height,
        width,
    ):
        raise ValueError("feedback has an incompatible panel shape")
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be positive")
    outputs = []
    for start in range(0, feedback.shape[1], target_chunk_size):
        target = feedback[:, start : start + target_chunk_size]
        matches = exact_maps[:, None].eq(target[:, :, :, None])
        matches = matches.flatten(start_dim=-2).all(dim=-1)
        likelihood = torch.full(
            matches.shape,
            float("-inf"),
            dtype=torch.float32,
            device=matches.device,
        )
        likelihood[matches] = 0.0
        outputs.append(likelihood)
    return torch.cat(outputs, dim=1)


def score_public_prediction_against_program_feedback(
    torch: Any,
    prediction: Any,
    feedback: Any,
    *,
    probes: int,
    target_chunk_size: int = 8,
) -> Any:
    """Score one public prediction against all hidden-program feedback.

    ``prediction`` has flattened public rows ``[G*P,K,...]`` and is never
    copied across hidden programs.  The result is ``[G,T,P,K]``.
    """

    import torch.nn.functional as F

    if feedback.ndim != 5:
        raise ValueError("feedback must have [G,T,P,H,W] shape")
    groups, targets, feedback_probes, height, width = feedback.shape
    if probes != feedback_probes:
        raise ValueError("probe count disagrees with feedback")
    hypotheses = prediction.change_logits.shape[1]
    colors = prediction.new_color_logits.shape[2]
    if prediction.input_colors.shape != (groups * probes, height, width):
        raise ValueError("prediction input colors have the wrong shape")
    change = prediction.change_logits.reshape(
        groups,
        probes,
        hypotheses,
        height,
        width,
    )
    new_color = prediction.new_color_logits.reshape(
        groups,
        probes,
        hypotheses,
        colors,
        height,
        width,
    )
    inputs = prediction.input_colors.reshape(
        groups,
        probes,
        height,
        width,
    )
    color_ids = torch.arange(
        colors,
        device=new_color.device,
    )[None, None, None, :, None, None]
    masked = new_color.masked_fill(
        inputs[:, :, None, None].eq(color_ids),
        torch.finfo(new_color.dtype).min,
    )
    color_log_prob = F.log_softmax(masked, dim=3)
    unchanged = F.logsigmoid(-change)
    changed_base = F.logsigmoid(change)
    outputs = []
    feedback = feedback.to(device=new_color.device)
    for start in range(0, targets, target_chunk_size):
        target = feedback[:, start : start + target_chunk_size]
        chunk = target.shape[1]
        gather_index = target[:, :, :, None, None].expand(
            groups,
            chunk,
            probes,
            hypotheses,
            1,
            height,
            width,
        )
        selected_color = color_log_prob[:, None].expand(
            groups,
            chunk,
            probes,
            hypotheses,
            colors,
            height,
            width,
        ).gather(4, gather_index).squeeze(4)
        unchanged_cells = unchanged[:, None].expand(
            groups,
            chunk,
            probes,
            hypotheses,
            height,
            width,
        )
        changed_cells = (
            changed_base[:, None] + selected_color
        )
        same = target[:, :, :, None].eq(inputs[:, None, :, None])
        cells = torch.where(same, unchanged_cells, changed_cells)
        outputs.append(cells.sum(dim=(-2, -1)))
    result = torch.cat(outputs, dim=1)
    if result.shape != (groups, targets, probes, hypotheses):
        raise AssertionError("learned panel scorer returned a wrong shape")
    if not bool(torch.isfinite(result).all().item()):
        raise AssertionError("learned full-grid likelihood must be finite")
    return result


def forced_bayes_fork(
    torch: Any,
    query_log_posterior: Any,
    forced_log_likelihood: Any,
) -> tuple[Any, Any]:
    """Apply every forced observation independently to one query posterior."""

    if query_log_posterior.ndim != 2:
        raise ValueError("query posterior must have [B,K] shape")
    if (
        forced_log_likelihood.ndim != 3
        or forced_log_likelihood.shape[0] != query_log_posterior.shape[0]
        or forced_log_likelihood.shape[2] != query_log_posterior.shape[1]
    ):
        raise ValueError("forced likelihood must have [B,F,K] shape")
    expanded = query_log_posterior[:, None].expand_as(
        forced_log_likelihood
    )
    shape = expanded.shape
    posterior, evidence = bayesian_log_likelihood_update(
        torch,
        expanded.reshape(-1, shape[-1]),
        forced_log_likelihood.reshape(-1, shape[-1]),
    )
    return posterior.reshape(shape), evidence.reshape(shape[:-1])


def _query_marginals(
    torch: Any,
    log_weights: Any,
    factor_bank: Any,
    axis_index: int,
) -> Any:
    normalised = _normalise_log_weights(torch, log_weights)
    probability = normalised.exp()
    result = torch.zeros(
        (*probability.shape[:-1], 4),
        dtype=probability.dtype,
        device=probability.device,
    )
    indices = factor_bank[:, axis_index].to(
        device=probability.device,
    )
    indices = indices.reshape(
        *((1,) * (probability.ndim - 1)),
        probability.shape[-1],
    ).expand_as(probability)
    return result.scatter_add(-1, indices, probability)


def safe_true_query_log_odds(
    torch: Any,
    log_weights: Any,
    factor_bank: Any,
    axis_index: int,
    true_values: Any,
) -> Any:
    """Return true-vs-all-other log odds without probability clipping."""

    if log_weights.ndim != 2:
        raise ValueError("log_weights must have [B,K] shape")
    if true_values.shape != log_weights.shape[:1]:
        raise ValueError("true_values must have [B] shape")
    bank_values = factor_bank[:, axis_index].to(log_weights.device)
    true_mask = bank_values[None].eq(true_values[:, None])
    true_terms = log_weights.masked_fill(~true_mask, float("-inf"))
    false_terms = log_weights.masked_fill(true_mask, float("-inf"))
    return (
        torch.logsumexp(true_terms, dim=-1)
        - torch.logsumexp(false_terms, dim=-1)
    )


def deduplicate_semantic_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Deduplicate palette/order repeats and assert canonical invariance."""

    keys = (
        "query_axis",
        "true_program_index",
        "query_probe_key",
        "forced_probe_key",
        "branch",
    )
    ignored = {"group_index", "candidate_index", "query_candidate_index"}
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    output = []
    for semantic_key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        reference = {
            key: value
            for key, value in rows[0].items()
            if key not in ignored
        }
        for row in rows[1:]:
            comparison = {
                key: value
                for key, value in row.items()
                if key not in ignored
            }
            for metric_key, expected in reference.items():
                actual = comparison.get(metric_key)
                if isinstance(expected, float):
                    if not math.isclose(
                        expected,
                        float(actual),
                        rel_tol=1e-5,
                        abs_tol=1e-5,
                    ):
                        raise AssertionError(
                            "canonical palette repeat changed metric "
                            f"{metric_key} for {semantic_key}"
                        )
                elif actual != expected:
                    raise AssertionError(
                        "canonical palette repeat changed semantic result "
                        f"{metric_key} for {semantic_key}"
                    )
        representative = dict(rows[0])
        representative["canonical_repeat_count"] = len(rows)
        output.append(representative)
    return output


def _summarise_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    drops = tuple(float(record["log_odds_drop_nats"]) for record in records)
    return {
        "semantic_sequences": len(records),
        "mean_query_true_probability": _mean(
            float(record["query_true_probability"]) for record in records
        ),
        "mean_forced_true_probability": _mean(
            float(record["forced_true_probability"]) for record in records
        ),
        "mean_query_nll_nats": _mean(
            float(record["query_nll_nats"]) for record in records
        ),
        "mean_query_posterior_nll_nats": _mean(
            float(record["query_posterior_nll_nats"])
            for record in records
        ),
        "mean_log_odds_change_nats": _mean(
            float(record["log_odds_change_nats"]) for record in records
        ),
        "p99_log_odds_drop_nats": percentile(drops, 99.0),
        "max_log_odds_drop_nats": max(drops, default=0.0),
        "catastrophic_reversal_rate": _mean(
            int(bool(record["catastrophic_reversal"])) for record in records
        ),
        "mean_query_marginal_kl_nats": _mean(
            float(record["query_marginal_kl_nats"]) for record in records
        ),
        "max_query_marginal_kl_nats": max(
            (
                float(record["query_marginal_kl_nats"])
                for record in records
            ),
            default=0.0,
        ),
        "mean_query_marginal_tv": _mean(
            float(record["query_marginal_tv"]) for record in records
        ),
        "max_query_marginal_tv": max(
            (float(record["query_marginal_tv"]) for record in records),
            default=0.0,
        ),
    }


def evaluate_gates(
    summaries: Mapping[str, Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    *,
    validity_passed: bool = True,
    expected_semantic_sequences: int = 1536,
) -> dict[str, object]:
    """Apply the preregistered RR-to-PP factor-locality decision gate."""

    required_summary_fields = {
        "catastrophic_reversal_rate",
        "p99_log_odds_drop_nats",
        "mean_query_nll_nats",
        "semantic_sequences",
    }
    invalid_reasons = []
    for branch in LEARNED_BRANCHES:
        if branch not in summaries:
            invalid_reasons.append(f"missing summary branch {branch}")
            continue
        branch_summary = summaries[branch].get("cross")
        if not isinstance(branch_summary, Mapping):
            invalid_reasons.append(f"{branch} missing cross summary")
            continue
        missing = required_summary_fields - set(branch_summary)
        if missing:
            invalid_reasons.append(
                f"{branch} missing fields {sorted(missing)}"
            )
        if int(branch_summary.get("semantic_sequences", -1)) != (
            expected_semantic_sequences
        ):
            invalid_reasons.append(
                f"{branch} semantic sequence count is not "
                f"{expected_semantic_sequences}"
            )
        for field in required_summary_fields - {"semantic_sequences"}:
            value = branch_summary.get(field)
            if value is None or not math.isfinite(float(value)):
                invalid_reasons.append(f"{branch}.{field} is nonfinite")
    if not validity_passed:
        invalid_reasons.append("protocol validity audit failed")
    if invalid_reasons:
        return {
            "decision": "invalid-no-decision",
            "factor_locality_rescue_gate_passed": False,
            "invalid_reasons": invalid_reasons,
        }

    record_key_fields = (
        "query_axis",
        "true_program_index",
        "query_probe_key",
        "forced_probe_key",
    )
    record_maps: dict[str, dict[tuple[object, ...], bool]] = {
        "RR": {},
        "PP": {},
    }
    for row in records:
        branch = row.get("branch")
        if branch not in record_maps:
            continue
        if row.get("forced_category", "cross") != "cross":
            continue
        missing = tuple(field for field in record_key_fields if field not in row)
        if missing or "catastrophic_reversal" not in row:
            invalid_reasons.append(
                f"{branch} record missing fields "
                f"{list(missing) + ([] if 'catastrophic_reversal' in row else ['catastrophic_reversal'])}"
            )
            continue
        key = tuple(row[field] for field in record_key_fields)
        if key in record_maps[str(branch)]:
            invalid_reasons.append(f"duplicate {branch} semantic record {key}")
            continue
        record_maps[str(branch)][key] = bool(
            row["catastrophic_reversal"]
        )
    for branch, rows in record_maps.items():
        if len(rows) != expected_semantic_sequences:
            invalid_reasons.append(
                f"{branch} cross record count is not "
                f"{expected_semantic_sequences}"
            )
    if set(record_maps["RR"]) != set(record_maps["PP"]):
        invalid_reasons.append("RR and PP semantic record keys disagree")
    if invalid_reasons:
        return {
            "decision": "invalid-no-decision",
            "factor_locality_rescue_gate_passed": False,
            "invalid_reasons": invalid_reasons,
        }

    raw = summaries["RR"]["cross"]
    rescued = summaries["PP"]["cross"]
    raw_reversal = float(raw["catastrophic_reversal_rate"])
    projected_reversal = float(rescued["catastrophic_reversal_rate"])
    raw_p99 = float(raw["p99_log_odds_drop_nats"])
    projected_p99 = float(rescued["p99_log_odds_drop_nats"])
    reversal_triggered = raw_reversal >= REVERSAL_RATE_GATE
    p99_triggered = raw_p99 >= P99_DROP_GATE_NATS
    reversal_improvement = (
        (raw_reversal - projected_reversal) / raw_reversal
        if raw_reversal > 0.0
        else 1.0
    )
    p99_improvement = (
        (raw_p99 - projected_p99) / raw_p99
        if raw_p99 > 0.0
        else 1.0
    )
    new_projection_reversals = sum(
        int(projected and not record_maps["RR"][key])
        for key, projected in record_maps["PP"].items()
    )
    query_nll_degradation = (
        float(summaries["PR"]["cross"]["mean_query_nll_nats"])
        - float(summaries["RR"]["cross"]["mean_query_nll_nats"])
    )
    raw_failure = reversal_triggered or p99_triggered
    rescue_checks = (
        (not reversal_triggered or reversal_improvement >= RESCUE_FRACTION_GATE),
        (not p99_triggered or p99_improvement >= RESCUE_FRACTION_GATE),
        new_projection_reversals == 0,
        query_nll_degradation < QUERY_NLL_DEGRADATION_GATE_NATS,
    )
    locality_rescue = raw_failure and all(rescue_checks)
    if locality_rescue:
        decision = "support-factorized-jepa-executor"
    elif raw_failure:
        decision = "turn-to-within-axis-calibration"
    else:
        decision = "turn-to-harder-compositional-benchmark"
    return {
        "decision": decision,
        "raw_failure_gate_passed": raw_failure,
        "reversal_gate_triggered": reversal_triggered,
        "p99_drop_gate_triggered": p99_triggered,
        "rr_reversal_rate": raw_reversal,
        "pp_reversal_rate": projected_reversal,
        "rr_p99_drop_nats": raw_p99,
        "pp_p99_drop_nats": projected_p99,
        "reversal_improvement_fraction": reversal_improvement,
        "p99_drop_improvement_fraction": p99_improvement,
        "new_projection_reversals": new_projection_reversals,
        "projected_query_nll_degradation_nats": query_nll_degradation,
        "projected_query_true_code_conditional_full_grid_nll_degradation_nats": (
            query_nll_degradation
        ),
        "factor_locality_rescue_gate_passed": locality_rescue,
    }


def _semantic_panel_spec(query_axis: Any) -> tuple[tuple[str, int | None], ...]:
    """Return the stable semantic order and acted-on factor for one panel."""

    from prp_wm.rulegrid import ALL_AXES

    rows: list[tuple[str, int | None]] = [
        (f"query:{query_axis.value}:v{variant}", ALL_AXES.index(query_axis))
        for variant in range(2)
    ]
    for axis in ALL_AXES:
        if axis is query_axis:
            continue
        rows.extend(
            (
                (f"cross:{axis.value}:v0", ALL_AXES.index(axis)),
                (f"cross:{axis.value}:v1", ALL_AXES.index(axis)),
            )
        )
    rows.extend((("neutral:v0", None), ("neutral:v1", None)))
    if len(rows) != CANDIDATES_PER_TASK:
        raise AssertionError("semantic panel spec has the wrong length")
    return tuple(rows)


def _align_panel(
    torch: Any,
    tensor: Any,
    semantic_keys: Sequence[Sequence[str]],
    semantic_order: Sequence[str],
    *,
    panel_dim: int,
) -> tuple[Any, Any]:
    """Align shuffled public panels and return aligned tensor plus indices."""

    indices = torch.tensor(
        [
            [tuple(keys).index(key) for key in semantic_order]
            for keys in semantic_keys
        ],
        dtype=torch.long,
        device=tensor.device,
    )
    if tensor.shape[0] != indices.shape[0]:
        raise ValueError("tensor and semantic-key groups disagree")
    shape = list(tensor.shape)
    view = [indices.shape[0]] + [1] * (tensor.ndim - 1)
    view[panel_dim] = indices.shape[1]
    expanded = list(shape)
    expanded[panel_dim] = indices.shape[1]
    gather = indices.reshape(view).expand(expanded)
    return torch.gather(tensor, panel_dim, gather), indices


def _marginal_shift_metrics(torch: Any, before: Any, after: Any) -> tuple[Any, Any]:
    """Return KL(before||after) and total variation along the last axis."""

    tiny = torch.finfo(before.dtype).tiny
    terms = torch.where(
        before > 0,
        before * (
            before.clamp_min(tiny).log()
            - after.clamp_min(tiny).log()
        ),
        torch.zeros_like(before),
    )
    return terms.sum(dim=-1), 0.5 * (before - after).abs().sum(dim=-1)


def _canonical_group_records(
    *,
    torch: Any,
    query_axis: Any,
    factor_bank: Any,
    exact_likelihood: Any,
    raw_likelihood: Any,
    projected_likelihood: Any,
    semantic_spec: Sequence[tuple[str, int | None]],
    actual_indices: Sequence[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Materialize the five forced branches for one canonical group."""

    if exact_likelihood.shape != raw_likelihood.shape:
        raise ValueError("exact and learned likelihood panels must share shape")
    if projected_likelihood.shape != raw_likelihood.shape:
        raise ValueError("raw and projected panels must share shape")
    programs, probes, hypotheses = raw_likelihood.shape
    if (programs, probes, hypotheses) != (
        PROGRAMS_PER_PUBLIC_PANEL,
        CANDIDATES_PER_TASK,
        PROGRAMS_PER_PUBLIC_PANEL,
    ):
        raise ValueError("canonical panel must have [64,8,64] likelihoods")
    query_axis_index = next(
        axis for key, axis in semantic_spec if key.startswith("query:")
    )
    assert query_axis_index is not None
    true_values = factor_bank[:, query_axis_index].to(torch.long)
    prior = torch.full(
        (programs, hypotheses),
        -math.log(float(hypotheses)),
        dtype=raw_likelihood.dtype,
    )
    forced_keys = tuple(key for key, _ in semantic_spec[2:])
    path_records: list[dict[str, object]] = []
    query_records: list[dict[str, object]] = []
    exact_support_after: Counter[str] = Counter()
    max_raw_projected_query_marginal_difference = 0.0
    max_raw_projected_query_evidence_difference = 0.0

    for query_probe_index in range(QUERY_PROBES):
        query_key = semantic_spec[query_probe_index][0]
        exact_query, exact_query_evidence = bayesian_log_likelihood_update(
            torch,
            prior,
            exact_likelihood[:, query_probe_index],
        )
        raw_query, raw_query_evidence = bayesian_log_likelihood_update(
            torch,
            prior,
            raw_likelihood[:, query_probe_index],
        )
        projected_query, projected_query_evidence = (
            bayesian_log_likelihood_update(
                torch,
                prior,
                projected_likelihood[:, query_probe_index],
            )
        )
        exact_support = torch.isfinite(exact_query).sum(dim=-1)
        if not bool(exact_support.eq(16).all().item()):
            raise AssertionError("exact query probe must leave 16 hypotheses")
        exact_query_marginal = _query_marginals(
            torch,
            exact_query,
            factor_bank,
            query_axis_index,
        )
        raw_query_marginal = _query_marginals(
            torch,
            raw_query,
            factor_bank,
            query_axis_index,
        )
        projected_query_marginal = _query_marginals(
            torch,
            projected_query,
            factor_bank,
            query_axis_index,
        )
        true_column = true_values[:, None]
        exact_true = exact_query_marginal.gather(1, true_column).squeeze(1)
        if not torch.allclose(
            exact_true,
            torch.ones_like(exact_true),
            atol=0.0,
            rtol=0.0,
        ):
            raise AssertionError("exact query posterior did not identify query value")

        marginal_difference = (
            raw_query_marginal - projected_query_marginal
        ).abs().max()
        evidence_difference = (
            raw_query_evidence - projected_query_evidence
        ).abs().max()
        max_raw_projected_query_marginal_difference = max(
            max_raw_projected_query_marginal_difference,
            float(marginal_difference),
        )
        max_raw_projected_query_evidence_difference = max(
            max_raw_projected_query_evidence_difference,
            float(evidence_difference),
        )
        if not torch.allclose(
            raw_query_marginal,
            projected_query_marginal,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise AssertionError(
                "uniform-prior raw/projected query marginals disagree"
            )
        if not torch.allclose(
            raw_query_evidence,
            projected_query_evidence,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise AssertionError(
                "uniform-prior raw/projected query evidence disagrees"
            )

        for pathway, marginal, evidence in (
            ("raw", raw_query_marginal, raw_query_evidence),
            (
                "projected",
                projected_query_marginal,
                projected_query_evidence,
            ),
        ):
            pathway_likelihood = (
                raw_likelihood
                if pathway == "raw"
                else projected_likelihood
            )
            true_probability = marginal.gather(1, true_column).squeeze(1)
            maximum = marginal.max(dim=-1).values
            for program_index in range(programs):
                probability = float(true_probability[program_index])
                conditional_nll = -float(
                    pathway_likelihood[
                        program_index,
                        query_probe_index,
                        program_index,
                    ]
                )
                query_records.append(
                    {
                        "query_axis": query_axis.value,
                        "true_program_index": program_index,
                        "true_factor_code": [
                            int(value)
                            for value in factor_bank[program_index].tolist()
                        ],
                        "query_probe_key": query_key,
                        "query_likelihood": pathway,
                        "true_query_probability": probability,
                        "true_query_nll_nats": -math.log(
                            max(probability, 1e-300)
                        ),
                        "true_code_conditional_full_grid_nll_nats": (
                            conditional_nll
                        ),
                        "true_query_top1": bool(
                            true_probability[program_index]
                            >= maximum[program_index] - 1e-7
                        ),
                        "true_query_probability_ge_0_95": bool(
                            probability >= 0.95
                        ),
                        "query_log_evidence": float(evidence[program_index]),
                    }
                )

        query_posteriors = {
            "EE": exact_query,
            "RR": raw_query,
            "RP": raw_query,
            "PR": projected_query,
            "PP": projected_query,
        }
        forced_panels = {
            "EE": exact_likelihood[:, 2:],
            "RR": raw_likelihood[:, 2:],
            "RP": projected_likelihood[:, 2:],
            "PR": raw_likelihood[:, 2:],
            "PP": projected_likelihood[:, 2:],
        }
        query_conditional_nll = {
            "EE": -exact_likelihood[
                torch.arange(programs),
                query_probe_index,
                torch.arange(programs),
            ],
            "RR": -raw_likelihood[
                torch.arange(programs),
                query_probe_index,
                torch.arange(programs),
            ],
            "RP": -raw_likelihood[
                torch.arange(programs),
                query_probe_index,
                torch.arange(programs),
            ],
            "PR": -projected_likelihood[
                torch.arange(programs),
                query_probe_index,
                torch.arange(programs),
            ],
            "PP": -projected_likelihood[
                torch.arange(programs),
                query_probe_index,
                torch.arange(programs),
            ],
        }
        for branch in BRANCHES:
            query_posterior = query_posteriors[branch]
            forced_posterior, forced_evidence = forced_bayes_fork(
                torch,
                query_posterior,
                forced_panels[branch],
            )
            query_marginal = _query_marginals(
                torch,
                query_posterior,
                factor_bank,
                query_axis_index,
            )
            forced_marginal = _query_marginals(
                torch,
                forced_posterior,
                factor_bank,
                query_axis_index,
            )
            query_true = query_marginal.gather(
                1,
                true_column,
            ).squeeze(1)
            forced_true = forced_marginal.gather(
                2,
                true_values[:, None, None].expand(
                    programs,
                    FORCED_PROBES,
                    1,
                ),
            ).squeeze(2)
            kl, tv = _marginal_shift_metrics(
                torch,
                query_marginal[:, None].expand_as(forced_marginal),
                forced_marginal,
            )
            if branch == "EE":
                odds_change = torch.zeros_like(forced_true)
                odds_drop = torch.zeros_like(forced_true)
            else:
                query_odds = safe_true_query_log_odds(
                    torch,
                    query_posterior,
                    factor_bank,
                    query_axis_index,
                    true_values,
                )
                forced_odds = safe_true_query_log_odds(
                    torch,
                    forced_posterior.reshape(
                        programs * FORCED_PROBES,
                        hypotheses,
                    ),
                    factor_bank,
                    query_axis_index,
                    true_values[:, None].expand(
                        programs,
                        FORCED_PROBES,
                    ).reshape(-1),
                ).reshape(programs, FORCED_PROBES)
                odds_change = forced_odds - query_odds[:, None]
                odds_drop = (-odds_change).clamp_min(0.0)
                if not bool(
                    torch.isfinite(odds_change).all().item()
                    and torch.isfinite(odds_drop).all().item()
                ):
                    raise AssertionError("learned log odds must be finite")
            other = forced_marginal.clone()
            other.scatter_(
                2,
                true_values[:, None, None].expand(
                    programs,
                    FORCED_PROBES,
                    1,
                ),
                -1.0,
            )
            forced_maximum = forced_marginal.max(dim=-1).values
            reversal = (
                (query_true[:, None] >= REVERSAL_CONFIDENCE)
                & (forced_maximum >= REVERSAL_CONFIDENCE)
                & (forced_true < other.max(dim=-1).values - 1e-7)
            )

            if branch == "EE":
                support = torch.isfinite(forced_posterior).sum(dim=-1)
                for forced_index, key in enumerate(forced_keys):
                    expected = 16 if key.startswith("neutral:") else 4
                    if not bool(support[:, forced_index].eq(expected).all()):
                        raise AssertionError(
                            f"exact {key} must leave {expected} hypotheses"
                        )
                    exact_support_after[f"{key}:{expected}"] += programs
                if (
                    float(kl.max()) >= 1e-8
                    or float(tv.max()) >= 1e-8
                ):
                    raise AssertionError(
                        "exact irrelevant probes changed query marginal"
                    )

            for program_index in range(programs):
                query_probability = float(query_true[program_index])
                for forced_index, forced_key in enumerate(forced_keys):
                    path_records.append(
                        {
                            "query_axis": query_axis.value,
                            "true_program_index": program_index,
                            "true_factor_code": [
                                int(value)
                                for value in factor_bank[
                                    program_index
                                ].tolist()
                            ],
                            "query_probe_key": query_key,
                            "forced_probe_key": forced_key,
                            "forced_category": (
                                "neutral"
                                if forced_key.startswith("neutral:")
                                else "cross"
                            ),
                            "forced_axis": (
                                None
                                if forced_key.startswith("neutral:")
                                else forced_key.split(":")[1]
                            ),
                            "branch": branch,
                            "query_candidate_index": int(
                                actual_indices[query_probe_index]
                            ),
                            "candidate_index": int(
                                actual_indices[forced_index + 2]
                            ),
                            "query_true_probability": query_probability,
                            "forced_true_probability": float(
                                forced_true[
                                    program_index,
                                    forced_index,
                                ]
                            ),
                            "query_posterior_nll_nats": -math.log(
                                max(query_probability, 1e-300)
                            ),
                            "query_nll_nats": float(
                                query_conditional_nll[branch][program_index]
                            ),
                            "log_odds_change_nats": float(
                                odds_change[
                                    program_index,
                                    forced_index,
                                ]
                            ),
                            "log_odds_drop_nats": float(
                                odds_drop[
                                    program_index,
                                    forced_index,
                                ]
                            ),
                            "catastrophic_reversal": bool(
                                reversal[
                                    program_index,
                                    forced_index,
                                ]
                            ),
                            "query_marginal_kl_nats": float(
                                kl[program_index, forced_index]
                            ),
                            "query_marginal_tv": float(
                                tv[program_index, forced_index]
                            ),
                            "forced_log_evidence": float(
                                forced_evidence[
                                    program_index,
                                    forced_index,
                                ]
                            ),
                        }
                    )
    return (
        path_records,
        query_records,
        {
            "exact_support_after": dict(exact_support_after),
            "max_raw_projected_query_marginal_difference": (
                max_raw_projected_query_marginal_difference
            ),
            "max_raw_projected_query_evidence_difference": (
                max_raw_projected_query_evidence_difference
            ),
        },
    )


def _query_stage_summary(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result = {}
    for pathway in ("raw", "projected"):
        rows = tuple(
            row for row in records if row["query_likelihood"] == pathway
        )
        probabilities = tuple(
            float(row["true_query_probability"]) for row in rows
        )
        result[pathway] = {
            "semantic_query_cases": len(rows),
            "top1_accuracy": _mean(
                int(bool(row["true_query_top1"])) for row in rows
            ),
            "probability_ge_0_95_rate": _mean(
                int(bool(row["true_query_probability_ge_0_95"]))
                for row in rows
            ),
            "mean_true_query_probability": _mean(probabilities),
            "minimum_true_query_probability": min(
                probabilities,
                default=0.0,
            ),
            "mean_true_query_nll_nats": _mean(
                float(row["true_query_nll_nats"]) for row in rows
            ),
            "mean_true_code_conditional_full_grid_nll_nats": _mean(
                float(
                    row[
                        "true_code_conditional_full_grid_nll_nats"
                    ]
                )
                for row in rows
            ),
        }
    result["projected_minus_raw_mean_true_query_nll_nats"] = (
        float(result["projected"]["mean_true_query_nll_nats"])
        - float(result["raw"]["mean_true_query_nll_nats"])
    )
    result[
        "projected_minus_raw_mean_true_code_conditional_full_grid_nll_nats"
    ] = (
        float(
            result["projected"][
                "mean_true_code_conditional_full_grid_nll_nats"
            ]
        )
        - float(
            result["raw"][
                "mean_true_code_conditional_full_grid_nll_nats"
            ]
        )
    )
    return result


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch

    from prp_wm.causal_filter import enumerate_factor_codes
    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import ALL_AXES, ALL_PROGRAMS
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_nuisance_learned_bridge import (
        _canonical_candidate_panel,
    )
    from scripts.run_oracle_canonical_acquisition_ceiling import (
        _candidate_panel,
    )

    _configure_determinism(torch, args.seed)
    device = _resolve_device(torch, args.device)
    checkpoint_path = args.active_executor_checkpoint.resolve()
    executor, executor_checkpoint, executor_result = _load_active_executor(
        torch,
        checkpoint_path,
        device,
    )
    factor_bank = enumerate_factor_codes(device="cpu")
    if factor_bank.shape != (PROGRAMS_PER_PUBLIC_PANEL, 3):
        raise SystemExit("audit requires the stable Cartesian 64-code bank")
    for program_index, program in enumerate(ALL_PROGRAMS):
        if tuple(int(value) for value in factor_bank[program_index]) != (
            rule_program_factor_ids(program)
        ):
            raise SystemExit("ALL_PROGRAMS and factor bank order disagree")

    primary_records: list[dict[str, object]] = []
    query_stage_records: list[dict[str, object]] = []
    partition_aggregate = _new_partition_counter()
    partition_by_category: dict[str, Any] = defaultdict(
        _new_partition_counter
    )
    partition_by_semantic_key: dict[str, Any] = defaultdict(
        _new_partition_counter
    )
    invariance_by_axis: dict[str, object] = {}
    validity_by_axis: dict[str, object] = {}
    canonical_panel_hashes: dict[str, str] = {}

    for query_axis in ALL_AXES:
        tasks = _build_forced_audit_tasks(
            query_axis,
            groups=args.groups_per_query,
            split=args.split,
            master_seed=args.data_master_seed,
            candidate_seed=args.seed,
        )
        semantic_spec = _semantic_panel_spec(query_axis)
        semantic_order = tuple(key for key, _ in semantic_spec)
        reference: dict[str, Any] | None = None
        maximum_raw_difference = 0.0
        maximum_projected_difference = 0.0
        checked_repeats = 0

        for group_start in range(
            0,
            args.groups_per_query,
            args.batch_size,
        ):
            group_end = min(
                group_start + args.batch_size,
                args.groups_per_query,
            )
            selected_tasks = tasks[
                group_start * PROGRAMS_PER_PUBLIC_PANEL :
                group_end * PROGRAMS_PER_PUBLIC_PANEL
            ]
            representatives = selected_tasks[
                ::PROGRAMS_PER_PUBLIC_PANEL
            ]
            groups = len(representatives)
            if len(selected_tasks) != groups * PROGRAMS_PER_PUBLIC_PANEL:
                raise AssertionError("batch split a public 64-program group")

            (
                exact_maps,
                learned_maps,
                prediction,
                _,
            ) = _canonical_candidate_panel(
                torch=torch,
                tasks=representatives,
                factor_bank=factor_bank,
                executor=executor,
                device=device,
            )
            (
                public_states,
                public_actions,
                public_action_mask,
                feedback,
            ) = _candidate_panel(
                torch=torch,
                tasks=selected_tasks,
                device=device,
            )
            _assert_public_program_copies(
                torch,
                public_states,
                name="candidate states",
            )
            _assert_public_program_copies(
                torch,
                public_actions,
                name="candidate actions",
            )
            _assert_public_program_copies(
                torch,
                public_action_mask,
                name="candidate action masks",
            )
            feedback = feedback.reshape(
                groups,
                PROGRAMS_PER_PUBLIC_PANEL,
                CANDIDATES_PER_TASK,
                executor.config.grid_size,
                executor.config.grid_size,
            )
            semantic_keys = tuple(
                tuple(task.privileged.candidate_kinds)
                for task in representatives
            )
            aligned_exact_maps, actual_indices = _align_panel(
                torch,
                exact_maps,
                semantic_keys,
                semantic_order,
                panel_dim=1,
            )
            aligned_learned_maps, _ = _align_panel(
                torch,
                learned_maps,
                semantic_keys,
                semantic_order,
                panel_dim=1,
            )
            aligned_feedback, _ = _align_panel(
                torch,
                feedback,
                semantic_keys,
                semantic_order,
                panel_dim=2,
            )

            # This equality uses the independently materialized environment
            # sidecar, not the map itself as a substitute for feedback.
            for local_group in range(groups):
                for program_index in range(PROGRAMS_PER_PUBLIC_PANEL):
                    if not torch.equal(
                        aligned_exact_maps[
                            local_group,
                            :,
                            program_index,
                        ],
                        aligned_feedback[
                            local_group,
                            program_index,
                        ].cpu(),
                    ):
                        raise AssertionError(
                            "true-code exact map and feedback disagree"
                        )

            exact_likelihood = exact_panel_log_likelihoods(
                torch,
                aligned_exact_maps,
                aligned_feedback.cpu(),
                target_chunk_size=args.target_chunk_size,
            )
            raw_likelihood = (
                score_public_prediction_against_program_feedback(
                    torch,
                    prediction,
                    feedback,
                    probes=CANDIDATES_PER_TASK,
                    target_chunk_size=args.target_chunk_size,
                )
            )
            raw_likelihood, _ = _align_panel(
                torch,
                raw_likelihood,
                semantic_keys,
                semantic_order,
                panel_dim=2,
            )
            raw_likelihood = raw_likelihood.detach().cpu()
            projected_likelihood = torch.stack(
                tuple(
                    axis_project_log_likelihood(
                        torch,
                        raw_likelihood[:, :, probe_index],
                        factor_bank,
                        axis_index,
                    )
                    for probe_index, (_, axis_index) in enumerate(
                        semantic_spec
                    )
                ),
                dim=2,
            )

            for local_group in range(groups):
                global_group = group_start + local_group
                rows = {
                    "exact_likelihood": exact_likelihood[local_group],
                    "raw_likelihood": raw_likelihood[local_group],
                    "projected_likelihood": projected_likelihood[local_group],
                    "exact_maps": aligned_exact_maps[local_group],
                    "learned_maps": aligned_learned_maps[local_group],
                }
                if reference is None:
                    reference = {
                        key: value.clone() for key, value in rows.items()
                    }
                    canonical_panel_hashes[query_axis.value] = (
                        hashlib.sha256(
                            reference["exact_maps"]
                            .contiguous()
                            .numpy()
                            .tobytes()
                        ).hexdigest()
                    )
                else:
                    if not torch.equal(
                        rows["exact_likelihood"],
                        reference["exact_likelihood"],
                    ):
                        raise AssertionError(
                            "canonical exact likelihood changed by palette/order"
                        )
                    if not torch.equal(
                        rows["exact_maps"],
                        reference["exact_maps"],
                    ):
                        raise AssertionError(
                            "canonical exact maps changed by palette/order"
                        )
                    if not torch.equal(
                        rows["learned_maps"],
                        reference["learned_maps"],
                    ):
                        raise AssertionError(
                            "canonical learned MAP changed by palette/order"
                        )
                    raw_difference = float(
                        (
                            rows["raw_likelihood"]
                            - reference["raw_likelihood"]
                        ).abs().max()
                    )
                    projected_difference = float(
                        (
                            rows["projected_likelihood"]
                            - reference["projected_likelihood"]
                        ).abs().max()
                    )
                    maximum_raw_difference = max(
                        maximum_raw_difference,
                        raw_difference,
                    )
                    maximum_projected_difference = max(
                        maximum_projected_difference,
                        projected_difference,
                    )
                    if raw_difference > 2e-5 or projected_difference > 2e-5:
                        raise AssertionError(
                            "canonical learned likelihood changed by "
                            "palette/order"
                        )
                checked_repeats += 1

                # Only one canonical semantic representative contributes to
                # rates.  Other groups are invariance checks above.
                if global_group != 0:
                    continue
                records, query_rows, axis_validity = (
                    _canonical_group_records(
                        torch=torch,
                        query_axis=query_axis,
                        factor_bank=factor_bank,
                        exact_likelihood=rows["exact_likelihood"],
                        raw_likelihood=rows["raw_likelihood"],
                        projected_likelihood=rows[
                            "projected_likelihood"
                        ],
                        semantic_spec=semantic_spec,
                        actual_indices=[
                            int(value)
                            for value in actual_indices[
                                local_group
                            ].tolist()
                        ],
                    )
                )
                primary_records.extend(records)
                query_stage_records.extend(query_rows)
                validity_by_axis[query_axis.value] = axis_validity
                for probe_index, (semantic_key, _) in enumerate(
                    semantic_spec
                ):
                    category = (
                        "query"
                        if semantic_key.startswith("query:")
                        else (
                            "neutral"
                            if semantic_key.startswith("neutral:")
                            else "cross"
                        )
                    )
                    for counter in (
                        partition_aggregate,
                        partition_by_category[category],
                    ):
                        _update_partition_counter(
                            counter,
                            torch=torch,
                            exact_maps=rows["exact_maps"][probe_index],
                            learned_maps=rows["learned_maps"][probe_index],
                        )
                    # Neutral probes already have a dedicated category audit.
                    # Keeping the 12 role-aware atomic keys here exposes weak
                    # axis/variant combinations without pooling query and
                    # cross-axis uses.
                    if category != "neutral":
                        _update_partition_counter(
                            partition_by_semantic_key[semantic_key],
                            torch=torch,
                            exact_maps=rows["exact_maps"][probe_index],
                            learned_maps=rows["learned_maps"][probe_index],
                        )

        invariance_by_axis[query_axis.value] = {
            "observed_public_panel_repeats": checked_repeats,
            "unique_semantic_panels": 1,
            "maximum_raw_log_likelihood_difference": (
                maximum_raw_difference
            ),
            "maximum_projected_log_likelihood_difference": (
                maximum_projected_difference
            ),
            "passed": checked_repeats == args.groups_per_query,
        }

    summaries: dict[str, object] = {}
    for branch in BRANCHES:
        branch_rows = tuple(
            row for row in primary_records if row["branch"] == branch
        )
        summaries[branch] = {
            "all": _summarise_records(branch_rows),
            "cross": _summarise_records(
                tuple(
                    row
                    for row in branch_rows
                    if row["forced_category"] == "cross"
                )
            ),
            "neutral": _summarise_records(
                tuple(
                    row
                    for row in branch_rows
                    if row["forced_category"] == "neutral"
                )
            ),
        }

    expected_per_branch = (
        len(ALL_AXES)
        * PROGRAMS_PER_PUBLIC_PANEL
        * QUERY_PROBES
        * FORCED_PROBES
    )
    expected_cross_per_branch = (
        len(ALL_AXES) * PROGRAMS_PER_PUBLIC_PANEL * QUERY_PROBES * 4
    )
    expected_query_per_path = (
        len(ALL_AXES) * PROGRAMS_PER_PUBLIC_PANEL * QUERY_PROBES
    )
    count_checks = {
        "per_branch_total": all(
            int(summaries[branch]["all"]["semantic_sequences"])
            == expected_per_branch
            for branch in BRANCHES
        ),
        "per_branch_cross": all(
            int(summaries[branch]["cross"]["semantic_sequences"])
            == expected_cross_per_branch
            for branch in BRANCHES
        ),
        "per_branch_neutral": all(
            int(summaries[branch]["neutral"]["semantic_sequences"])
            == (
                len(ALL_AXES)
                * PROGRAMS_PER_PUBLIC_PANEL
                * QUERY_PROBES
                * 2
            )
            for branch in BRANCHES
        ),
        "query_stage_per_path": all(
            sum(
                int(row["query_likelihood"] == pathway)
                for row in query_stage_records
            )
            == expected_query_per_path
            for pathway in ("raw", "projected")
        ),
    }
    exact_cross = summaries["EE"]["cross"]
    exact_neutral = summaries["EE"]["neutral"]
    exact_control_passed = (
        float(exact_cross["max_query_marginal_kl_nats"]) < 1e-8
        and float(exact_cross["max_query_marginal_tv"]) < 1e-8
        and float(exact_neutral["max_query_marginal_kl_nats"]) < 1e-8
        and float(exact_neutral["max_query_marginal_tv"]) < 1e-8
    )
    invariance_passed = all(
        bool(row["passed"]) for row in invariance_by_axis.values()
    )
    validity_audit = {
        "expected_semantic_sequences_per_branch": expected_per_branch,
        "expected_cross_sequences_per_branch": expected_cross_per_branch,
        "observed_path_records": len(primary_records),
        "count_checks": count_checks,
        "exact_control_passed": exact_control_passed,
        "canonical_invariance_passed": invariance_passed,
        "by_axis": validity_by_axis,
        "passed": (
            all(count_checks.values())
            and exact_control_passed
            and invariance_passed
        ),
    }
    query_summary = _query_stage_summary(query_stage_records)
    gates = evaluate_gates(
        summaries,
        primary_records,
        validity_passed=bool(validity_audit["passed"]),
        expected_semantic_sequences=expected_cross_per_branch,
    )

    pp_by_key = {
        (
            row["query_axis"],
            row["true_program_index"],
            row["query_probe_key"],
            row["forced_probe_key"],
        ): row
        for row in primary_records
        if row["branch"] == "PP"
    }
    worst = []
    rr_cross = sorted(
        (
            row
            for row in primary_records
            if row["branch"] == "RR"
            and row["forced_category"] == "cross"
        ),
        key=lambda row: (
            -float(row["log_odds_drop_nats"]),
            str(row["query_axis"]),
            int(row["true_program_index"]),
            str(row["query_probe_key"]),
            str(row["forced_probe_key"]),
        ),
    )
    for row in rr_cross[: args.worst_records]:
        key = (
            row["query_axis"],
            row["true_program_index"],
            row["query_probe_key"],
            row["forced_probe_key"],
        )
        projected = pp_by_key[key]
        worst.append(
            {
                **dict(row),
                "pp_forced_true_probability": projected[
                    "forced_true_probability"
                ],
                "pp_log_odds_drop_nats": projected[
                    "log_odds_drop_nats"
                ],
                "pp_catastrophic_reversal": projected[
                    "catastrophic_reversal"
                ],
            }
        )

    partition_audit = {
        "aggregate": _finalise_partition_counter(partition_aggregate),
        "by_probe_category": {
            category: _finalise_partition_counter(counter)
            for category, counter in sorted(
                partition_by_category.items()
            )
        },
        "by_semantic_probe_key": {
            semantic_key: _finalise_partition_counter(counter)
            for semantic_key, counter in sorted(
                partition_by_semantic_key.items()
            )
        },
        "unique_semantic_panels": len(ALL_AXES) * CANDIDATES_PER_TASK,
        "observed_palette_order_repeats": (
            len(ALL_AXES)
            * args.groups_per_query
            * CANDIDATES_PER_TASK
        ),
        "note": (
            "Only one canonical representative per axis/probe enters "
            "partition rates; palette/order groups are invariance repeats."
        ),
    }
    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "complete"
            if validity_audit["passed"]
            else "invalid"
        ),
        "experiment": "forced_cross_axis_likelihood_locality",
        "protocol": {
            "branches": {
                "EE": "exact query / exact forced",
                "RR": "raw learned query / raw learned forced",
                "RP": "raw learned query / projected learned forced",
                "PR": "projected learned query / raw learned forced",
                "PP": "projected learned query / projected learned forced",
            },
            "selector_used": False,
            "prior": "uniform over the complete Cartesian 64-code bank",
            "forced_fork": (
                "Every forced probe starts independently from the same "
                "post-query posterior; forced observations never accumulate."
            ),
            "projection": (
                "Full-grid OutcomePrediction.log_prob first, then oracle "
                "factor-fiber logmeanexp; neutral is one all-code constant."
            ),
            "gate_query_nll_metric": (
                "negative full-grid learned log likelihood of the true "
                "factor code, conditional on the forced query observation; "
                "posterior true-query NLL is reported separately"
            ),
            "statistical_unit": (
                "canonical semantic sequence; palette/order groups are "
                "invariance repeats and never independent cases"
            ),
        },
        "groups_per_query_observed": args.groups_per_query,
        "unique_semantic_sequences_per_branch": expected_per_branch,
        "unique_cross_sequences_per_branch": expected_cross_per_branch,
        "query_stage_positive_control": query_summary,
        "summaries": summaries,
        "decision_gate": gates,
        "validity_audit": validity_audit,
        "canonical_repeat_invariance": invariance_by_axis,
        "canonical_panel_hashes": canonical_panel_hashes,
        "outcome_partition_audit": partition_audit,
        "worst_20_or_requested_rr_cross_counterexamples": worst,
        "primary_semantic_path_records": primary_records,
        "query_stage_records": query_stage_records,
        "active_executor_checkpoint": str(checkpoint_path),
        "active_executor_checkpoint_sha256": _sha256(checkpoint_path),
        "active_executor_checkpoint_schema": executor_checkpoint.get(
            "checkpoint_schema_version"
        ),
        "active_executor_original_gate_passed": executor_result.get(
            "active_prefix_executor_gate",
            {},
        ).get("passed"),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "audited_source_sha256": _source_sha256(),
        "split": args.split,
        "seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "batch_size_public_panels": args.batch_size,
        "target_chunk_size": args.target_chunk_size,
        "device": str(device),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    _atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "query_stage_positive_control": query_summary,
                "summaries": summaries,
                "decision_gate": gates,
                "validity_audit": validity_audit,
                "outcome_partition_audit": partition_audit,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
