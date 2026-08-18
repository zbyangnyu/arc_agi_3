#!/usr/bin/env python3
"""Run the 2x2 belief/outcome bridge on the RuleGrid nuisance protocol.

The four paired conditions cross an exact or learned initial rule belief with
an exact or learned candidate-outcome model:

* exact-belief/exact-outcome;
* learned-belief/exact-outcome;
* exact-belief/learned-outcome; and
* learned-belief/learned-outcome.

Every condition receives the same tasks, shuffled candidate panels, explicit
query axis, factor-code bank, environment feedback, seed, and exploration
budget.  Acquisition is always query-conditioned expected terminal-door gain.
The learned outcome model contributes only its MAP outcome map during
selection.  After all choices are fixed, its selected ``OutcomePrediction``
is scored against canonical environment feedback with a proper log
likelihood.  Exact-outcome conditions instead use a deterministic Bayes
filter.

This is still an oracle-canonical bridge experiment: factor axes, the query
axis, the 64-code bank, and palette canonicalization are privileged.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.run_oracle_canonical_acquisition_ceiling import (  # noqa: E402
    DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    DEFAULT_BELIEF_CHECKPOINT,
    PublicDoorQuery,
    _door_marginals,
    _finalise_task_records,
    _load_active_executor,
    _normalise_log_weights,
    _selected_prediction,
    _symbolic_door_identified,
    _task_metrics,
    bayesian_log_likelihood_update,
)
from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (  # noqa: E402
    CANDIDATES_PER_TASK,
    MAX_BUDGET,
    PROGRAMS_PER_GROUP,
    _build_nuisance_tasks,
    _exact_candidate_outcome_maps,
    _learned_marked_initial_joint_log_weights,
    _select_query_conditioned_candidate,
    _selector_boundary_audit,
    _symbolic_initial_joint_log_weights,
)


RESULT_SCHEMA_VERSION = "prp-wm.nuisance-learned-bridge.v1"
DEFAULT_BUDGETS = (0, 1, 2, 3)
DEFAULT_SEEDS = (20260873,)
EXACT_EXACT = "exact-belief/exact-outcome"
LEARNED_EXACT = "learned-belief/exact-outcome"
EXACT_LEARNED = "exact-belief/learned-outcome"
LEARNED_LEARNED = "learned-belief/learned-outcome"


@dataclass(frozen=True)
class BridgeCondition:
    """One cell in the paired belief-by-outcome experiment."""

    name: str
    belief: str
    outcome: str

    def __post_init__(self) -> None:
        if self.belief not in {"exact", "learned"}:
            raise ValueError("belief must be exact or learned")
        if self.outcome not in {"exact", "learned"}:
            raise ValueError("outcome must be exact or learned")


CONDITIONS = (
    BridgeCondition(EXACT_EXACT, "exact", "exact"),
    BridgeCondition(LEARNED_EXACT, "learned", "exact"),
    BridgeCondition(EXACT_LEARNED, "exact", "learned"),
    BridgeCondition(LEARNED_LEARNED, "learned", "learned"),
)
CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--belief-checkpoint",
        type=Path,
        default=DEFAULT_BELIEF_CHECKPOINT,
    )
    parser.add_argument(
        "--active-executor-checkpoint",
        type=Path,
        default=DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-query", type=int, default=64)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--data-master-seed", type=int, default=2026071601)
    parser.add_argument(
        "--split",
        default="oracle-canonical-nuisance-acquisition-ceiling",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if args.groups_per_query <= 0:
        raise SystemExit("--groups-per-query must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.data_master_seed < 0:
        raise SystemExit("--data-master-seed must be non-negative")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free name")
    budgets = tuple(int(value) for value in args.budgets)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(value < 0 or value > MAX_BUDGET for value in budgets)
    ):
        raise SystemExit(
            f"--budgets must be unique integers in [0,{MAX_BUDGET}]"
        )
    seeds = tuple(int(value) for value in args.seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(value < 0 for value in seeds)
    ):
        raise SystemExit("--seeds must be unique non-negative integers")
    return tuple(sorted(budgets)), seeds


def _sha256(path: Path) -> str:
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


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _repeat_public_groups(torch: Any, tensor: Any) -> Any:
    """Broadcast one public-environment row to its three hidden query modes."""

    if tensor.ndim < 1:
        raise ValueError("a public tensor must have a batch axis")
    return tensor.repeat_interleave(PROGRAMS_PER_GROUP, dim=0)


def _assert_public_group_copies(
    torch: Any,
    tensor: Any | None,
    *,
    name: str,
) -> None:
    """Require every three-row hidden-mode block to share public inputs."""

    if tensor is None:
        return
    if tensor.ndim < 1 or tensor.shape[0] % PROGRAMS_PER_GROUP:
        raise AssertionError(
            f"{name} must have a batch axis divisible by "
            f"{PROGRAMS_PER_GROUP}"
        )
    grouped = tensor.reshape(
        tensor.shape[0] // PROGRAMS_PER_GROUP,
        PROGRAMS_PER_GROUP,
        *tensor.shape[1:],
    )
    reference = grouped[:, :1].expand_as(grouped)
    if not torch.equal(grouped, reference):
        raise AssertionError(
            f"{name} differs between hidden modes of one public environment"
        )


def _broadcast_group_prediction(
    *,
    torch: Any,
    prediction: Any,
    groups: int,
    candidate_count: int,
) -> Any:
    """Broadcast a flattened [G*P,K,...] prediction over hidden-mode copies."""

    from prp_wm.neural import OutcomePrediction

    if groups <= 0 or candidate_count <= 0:
        raise ValueError("groups and candidates must be positive")

    def repeat(value: Any) -> Any:
        if value.shape[0] != groups * candidate_count:
            raise ValueError("prediction has an unexpected flattened batch")
        grouped = value.reshape(groups, candidate_count, *value.shape[1:])
        return _repeat_public_groups(torch, grouped).reshape(
            groups * PROGRAMS_PER_GROUP * candidate_count,
            *value.shape[1:],
        )

    return OutcomePrediction(
        input_colors=repeat(prediction.input_colors),
        change_logits=repeat(prediction.change_logits),
        new_color_logits=repeat(prediction.new_color_logits),
    )


def _assert_normalised_log_weights(torch: Any, log_weights: Any) -> None:
    """Reject NaNs, empty posteriors, or silently unnormalised belief rows."""

    if bool(torch.isnan(log_weights).any().item()):
        raise AssertionError("posterior contains NaN")
    normaliser = torch.logsumexp(log_weights, dim=-1)
    if not bool(torch.isfinite(normaliser).all().item()):
        raise AssertionError("posterior has an empty hypothesis row")
    if not torch.allclose(
        normaliser,
        torch.zeros_like(normaliser),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise AssertionError("posterior rows are not normalised")


def _canonical_candidate_panel(
    *,
    torch: Any,
    tasks: Sequence[Any],
    factor_bank: Any,
    executor: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return exact maps, learned MAPs, learned distribution, and feedback."""

    from prp_wm.causal_filter import predict_factor_panel
    from prp_wm.latent_rules import outcome_map
    from scripts.run_active_support_calibrated_executor import (
        _canonicalize_grid_tensor,
    )
    from scripts.run_oracle_canonical_acquisition_ceiling import (
        _candidate_panel,
    )

    materialized = tuple(tasks)
    states, actions, action_mask, canonical_feedback = _candidate_panel(
        torch=torch,
        tasks=materialized,
        device=device,
    )
    task_codes = factor_bank.to(device)[None].expand(
        len(materialized),
        -1,
        -1,
    )
    with torch.no_grad():
        prediction = predict_factor_panel(
            executor,
            states,
            actions,
            task_codes,
            action_mask,
        )
        learned_maps = outcome_map(prediction).reshape(
            len(materialized),
            CANDIDATES_PER_TASK,
            factor_bank.shape[0],
            executor.config.grid_size,
            executor.config.grid_size,
        ).cpu()
    exact_raw, _ = _exact_candidate_outcome_maps(
        torch=torch,
        tasks=materialized,
        factor_bank=factor_bank,
    )
    exact_maps = _canonicalize_grid_tensor(
        torch,
        exact_raw,
        materialized,
    ).cpu()
    return exact_maps, learned_maps, prediction, canonical_feedback


def _deterministic_log_likelihood(
    *,
    torch: Any,
    selected_outcome_maps: Any,
    observed_feedback: Any,
) -> Any:
    """Return zero or -inf likelihood for an exact deterministic executor."""

    if selected_outcome_maps.ndim != 4:
        raise ValueError("selected maps must have [B,H,H,W] shape")
    if observed_feedback.shape != (
        selected_outcome_maps.shape[0],
        *selected_outcome_maps.shape[-2:],
    ):
        raise ValueError("feedback must have [B,H,W] shape")
    matches = selected_outcome_maps.eq(
        observed_feedback[:, None],
    ).flatten(start_dim=2).all(dim=-1)
    result = torch.full(
        matches.shape,
        float("-inf"),
        dtype=torch.float32,
        device=matches.device,
    )
    result[matches] = 0.0
    return result


def _rollout_condition(
    *,
    torch: Any,
    condition: BridgeCondition,
    tasks: Sequence[Any],
    query: PublicDoorQuery,
    initial_log_weights: Any,
    factor_bank: Any,
    exact_outcome_maps: Any,
    learned_outcome_maps: Any,
    learned_prediction: Any,
    canonical_feedback: Any,
    budgets: Sequence[int],
) -> tuple[dict[int, list[dict[str, object]]], list[dict[str, object]]]:
    """Run one bridge condition with feedback hidden until after selection."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import RuleGridTransition

    materialized = tuple(tasks)
    batch_size, candidate_count, hypotheses = exact_outcome_maps.shape[:3]
    if batch_size != len(materialized):
        raise ValueError("outcome batch and tasks must match")
    if learned_outcome_maps.shape != exact_outcome_maps.shape:
        raise ValueError("exact and learned outcome maps must share shape")
    if initial_log_weights.shape != (batch_size, hypotheses):
        raise ValueError("initial posterior has the wrong shape")
    if candidate_count != CANDIDATES_PER_TASK:
        raise ValueError("unexpected nuisance candidate count")

    acquisition_maps = (
        exact_outcome_maps
        if condition.outcome == "exact"
        else learned_outcome_maps
    )
    log_weights = initial_log_weights.clone()
    available = torch.ones(
        (batch_size, candidate_count),
        dtype=torch.bool,
    )
    histories = [list(task.inference.support) for task in materialized]
    first_identified: list[int | None] = [None] * batch_size
    selected_scores: list[list[Any]] = [[] for _ in materialized]
    selected_indices: list[list[int]] = [[] for _ in materialized]
    predictive_log_probabilities: list[list[float]] = [
        [] for _ in materialized
    ]
    target_doors = torch.tensor(
        [
            rule_program_factor_ids(task.privileged.true_program)[
                query.axis_index
            ]
            for task in materialized
        ],
        dtype=torch.long,
    )
    query_values = factor_bank[:, query.axis_index]
    snapshots: dict[int, list[dict[str, object]]] = {}

    def snapshot(budget: int) -> None:
        normalised = _normalise_log_weights(torch, log_weights)
        posterior = normalised.exp()
        partial = _task_metrics(
            torch=torch,
            log_weights=normalised,
            factor_bank=factor_bank,
            target_doors=target_doors,
            first_identified_steps=first_identified,
            budget=budget,
            selected_scores=selected_scores,
            predictive_log_probabilities=predictive_log_probabilities,
        )
        records = _finalise_task_records(
            torch=torch,
            partial=partial,
            factor_bank=factor_bank,
            query=query,
        )
        for task_index, record in enumerate(records):
            indices = selected_indices[task_index][:budget]
            query_probabilities = _door_marginals(
                torch,
                posterior[task_index],
                query_values,
            )
            maximum = query_probabilities.max()
            tied = torch.isclose(
                query_probabilities,
                maximum,
                atol=1e-7,
                rtol=1e-6,
            )
            target = int(target_doors[task_index])
            tied_count = int(tied.sum())
            record["tie_aware_terminal_accuracy"] = (
                1.0 / tied_count if bool(tied[target]) else 0.0
            )
            record["terminal_decision_set_size"] = tied_count
            record["selected_candidate_indices"] = list(indices)
            record["selected_candidate_categories_audit"] = [
                materialized[task_index].privileged.candidate_kinds[index]
                for index in indices
            ]
        snapshots[budget] = records

    if 0 in budgets:
        snapshot(0)
    _assert_normalised_log_weights(torch, log_weights)
    for step in range(1, max(budgets) + 1):
        choices: list[int] = []
        scores: list[Any] = []
        for task_index in range(batch_size):
            score = _select_query_conditioned_candidate(
                torch,
                log_weights[task_index],
                query_values,
                acquisition_maps[task_index],
                available[task_index],
            )
            choices.append(int(score.candidate_index))
            scores.append(score)
        for task_index, candidate_index in enumerate(choices):
            available[task_index, candidate_index] = False
            selected_indices[task_index].append(candidate_index)
            selected_scores[task_index].append(scores[task_index])

        # The environment sidecar is first indexed after every choice is fixed.
        task_rows_device = torch.arange(
            batch_size,
            device=canonical_feedback.device,
        )
        choice_rows_device = torch.tensor(
            choices,
            dtype=torch.long,
            device=canonical_feedback.device,
        )
        feedback_rows = canonical_feedback[
            task_rows_device,
            choice_rows_device,
        ]
        if condition.outcome == "exact":
            selected_maps = exact_outcome_maps[
                torch.arange(batch_size),
                torch.tensor(choices, dtype=torch.long),
            ]
            likelihood = _deterministic_log_likelihood(
                torch=torch,
                selected_outcome_maps=selected_maps,
                observed_feedback=feedback_rows.cpu(),
            )
        else:
            selected_prediction = _selected_prediction(
                torch=torch,
                prediction=learned_prediction,
                candidate_indices=choices,
                candidate_count=candidate_count,
            )
            with torch.no_grad():
                likelihood = selected_prediction.log_prob(
                    feedback_rows,
                ).cpu()
            if not bool(torch.isfinite(likelihood).all().item()):
                raise AssertionError(
                    "learned proper likelihood must be finite for every code"
                )
        log_weights, log_evidence = bayesian_log_likelihood_update(
            torch,
            log_weights,
            likelihood,
        )
        if not bool(torch.isfinite(log_evidence).all().item()):
            raise AssertionError("predictive log evidence must be finite")
        _assert_normalised_log_weights(torch, log_weights)

        for task_index, candidate_index in enumerate(choices):
            predictive_log_probabilities[task_index].append(
                float(log_evidence[task_index])
            )
            probe = materialized[
                task_index
            ].inference.active_candidates[candidate_index]
            observed = materialized[
                task_index
            ].privileged.active_targets[candidate_index]
            histories[task_index].append(
                RuleGridTransition(probe.state, probe.action, observed)
            )
            if (
                first_identified[task_index] is None
                and _symbolic_door_identified(
                    materialized[task_index],
                    histories[task_index],
                    query,
                )
            ):
                first_identified[task_index] = step
        if step in budgets:
            snapshot(step)

    traces = []
    for task_index, indices in enumerate(selected_indices):
        traces.append(
            {
                "task_batch_index": task_index,
                "selected_candidate_indices": list(indices),
                "selected_candidate_categories_audit": [
                    materialized[task_index].privileged.candidate_kinds[index]
                    for index in indices
                ],
                "first_symbolic_query_identification_step": (
                    first_identified[task_index]
                ),
            }
        )
    return snapshots, traces


def _new_partition_counter() -> dict[str, Any]:
    return {
        "candidate_panels": 0,
        "partition_exact_panels": 0,
        "predicted_grids": 0,
        "exact_grids": 0,
        "same_true_positive": 0,
        "same_false_positive": 0,
        "same_false_negative": 0,
        "class_count_pairs": Counter(),
    }


def _update_partition_counter(
    counter: dict[str, Any],
    *,
    torch: Any,
    exact_maps: Any,
    learned_maps: Any,
) -> None:
    """Accumulate equality-partition and grid metrics for one candidate."""

    hypotheses = exact_maps.shape[0]
    if learned_maps.shape != exact_maps.shape or exact_maps.ndim != 3:
        raise ValueError("partition maps must share [H,H,W] shape")
    exact_flat = exact_maps.reshape(hypotheses, -1)
    learned_flat = learned_maps.reshape(hypotheses, -1)
    exact_inverse = torch.unique(
        exact_flat,
        dim=0,
        sorted=True,
        return_inverse=True,
    )[1]
    learned_inverse = torch.unique(
        learned_flat,
        dim=0,
        sorted=True,
        return_inverse=True,
    )[1]
    exact_same = exact_inverse[:, None].eq(exact_inverse[None, :])
    learned_same = learned_inverse[:, None].eq(learned_inverse[None, :])
    upper = torch.triu(
        torch.ones_like(exact_same, dtype=torch.bool),
        diagonal=1,
    )
    counter["candidate_panels"] += 1
    counter["partition_exact_panels"] += int(
        bool(exact_same.eq(learned_same).all())
    )
    grid_exact = exact_flat.eq(learned_flat).all(dim=-1)
    counter["predicted_grids"] += hypotheses
    counter["exact_grids"] += int(grid_exact.sum())
    counter["same_true_positive"] += int(
        (exact_same & learned_same & upper).sum()
    )
    counter["same_false_positive"] += int(
        (~exact_same & learned_same & upper).sum()
    )
    counter["same_false_negative"] += int(
        (exact_same & ~learned_same & upper).sum()
    )
    counter["class_count_pairs"][
        (
            int(exact_inverse.max()) + 1,
            int(learned_inverse.max()) + 1,
        )
    ] += 1


def _finalise_partition_counter(
    counter: dict[str, Any],
) -> dict[str, object]:
    panels = int(counter["candidate_panels"])
    grids = int(counter["predicted_grids"])
    true_positive = int(counter["same_true_positive"])
    false_positive = int(counter["same_false_positive"])
    false_negative = int(counter["same_false_negative"])
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 1.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 1.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "candidate_panels": panels,
        "exact_partition_rate": (
            int(counter["partition_exact_panels"]) / panels
            if panels
            else 0.0
        ),
        "predicted_grids": grids,
        "exact_grid_rate": (
            int(counter["exact_grids"]) / grids if grids else 0.0
        ),
        "same_outcome_pair_precision": precision,
        "same_outcome_pair_recall": recall,
        "same_outcome_pair_f1": f1,
        "exact_to_learned_class_count": {
            f"{exact}->{learned}": count
            for (exact, learned), count in sorted(
                counter["class_count_pairs"].items()
            )
        },
    }


def _condition_summary(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    first_categories = tuple(
        str(record["selected_candidate_categories_audit"][0])
        for record in records
        if record["selected_candidate_categories_audit"]
    )
    first_indices = tuple(
        int(record["selected_candidate_indices"][0])
        for record in records
        if record["selected_candidate_indices"]
    )
    observed = tuple(
        float(record["mean_observed_log_predictive_probability"])
        for record in records
        if record["mean_observed_log_predictive_probability"] is not None
    )
    categories = (
        "query-atomic",
        "nuisance-axis-0",
        "nuisance-axis-1",
        "neutral",
    )
    return {
        "tasks": len(records),
        "tie_aware_terminal_accuracy": _mean(
            float(record["tie_aware_terminal_accuracy"])
            for record in records
        ),
        "terminal_win_rate": _mean(
            int(bool(record["won"])) for record in records
        ),
        "mean_true_query_probability": _mean(
            float(record["true_query_probability"]) for record in records
        ),
        "mean_query_entropy_nats": _mean(
            float(record["query_entropy_nats"]) for record in records
        ),
        "mean_joint_effective_hypotheses": _mean(
            float(record["joint_effective_hypotheses"])
            for record in records
        ),
        "symbolic_query_identified_rate": _mean(
            int(
                record["first_symbolic_identification_step"] is not None
                and int(record["first_symbolic_identification_step"])
                <= int(record["budget"])
            )
            for record in records
        ),
        "mean_observed_log_predictive_probability": (
            _mean(observed) if observed else None
        ),
        "first_selection_category_rates_audit": (
            {
                category: first_categories.count(category)
                / len(first_categories)
                for category in categories
            }
            if first_categories
            else {}
        ),
        "distinct_first_candidate_indices": sorted(set(first_indices)),
        "mean_terminal_decision_set_size": _mean(
            int(record["terminal_decision_set_size"])
            for record in records
        ),
    }


def _first_choice_agreement(
    task_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Compare each condition's first action with the exact/exact baseline."""

    by_condition: dict[str, dict[tuple[object, ...], dict[str, object]]] = {
        condition.name: {} for condition in CONDITIONS
    }
    for record in task_records:
        if int(record["budget"]) != 1:
            continue
        key = (
            record["seed"],
            record["query_axis"],
            record["group_index"],
            record["hidden_query_slot"],
        )
        by_condition[str(record["condition"])][key] = record
    baseline = by_condition[EXACT_EXACT]
    result = {}
    for condition in CONDITIONS:
        rows = by_condition[condition.name]
        shared = sorted(set(baseline) & set(rows))
        index_matches = 0
        category_matches = 0
        for key in shared:
            base_index = baseline[key]["selected_candidate_indices"][0]
            row_index = rows[key]["selected_candidate_indices"][0]
            base_category = baseline[key][
                "selected_candidate_categories_audit"
            ][0]
            row_category = rows[key][
                "selected_candidate_categories_audit"
            ][0]
            index_matches += int(base_index == row_index)
            category_matches += int(base_category == row_category)
        result[condition.name] = {
            "paired_tasks": len(shared),
            "top1_index_agreement_with_exact_exact": (
                index_matches / len(shared) if shared else None
            ),
            "top1_category_agreement_with_exact_exact": (
                category_matches / len(shared) if shared else None
            ),
        }
    return result


def _public_first_choice_consistency(
    task_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Audit that hidden modes cannot alter the first public action."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(
        list
    )
    for record in task_records:
        if int(record["budget"]) != 1:
            continue
        key = (
            record["seed"],
            record["query_axis"],
            record["group_index"],
            record["condition"],
        )
        grouped[key].append(record)
    violations = 0
    for records in grouped.values():
        slots = {
            int(record["hidden_query_slot"]) for record in records
        }
        indices = {
            int(record["selected_candidate_indices"][0])
            for record in records
        }
        if (
            slots != set(range(PROGRAMS_PER_GROUP))
            or len(records) != PROGRAMS_PER_GROUP
            or len(indices) != 1
        ):
            violations += 1
    return {
        "available": bool(grouped),
        "public_condition_groups": len(grouped),
        "violations": violations,
        "passed": bool(grouped) and violations == 0,
    }


def main() -> None:
    args = parse_args()
    budgets, seeds = _validate_args(args)

    import torch

    from prp_wm.causal_filter import enumerate_factor_codes
    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import ALL_AXES
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_oracle_canonical_acquisition_ceiling import (
        _candidate_panel,
    )
    from scripts.run_public_version_space_k4 import (
        load_public_version_k4_checkpoint,
    )

    device = _resolve_device(torch, args.device)
    belief_path = args.belief_checkpoint.resolve()
    executor_path = args.active_executor_checkpoint.resolve()
    belief_model, belief_checkpoint, _, _ = (
        load_public_version_k4_checkpoint(
            torch,
            belief_path,
            device=device,
        )
    )
    if belief_checkpoint.get("support_input") != "raw":
        raise SystemExit("belief checkpoint must infer from raw public support")
    if not hasattr(belief_model, "infer_factor_belief"):
        raise SystemExit("belief checkpoint has no factor-belief interface")
    executor, executor_checkpoint, executor_result = _load_active_executor(
        torch,
        executor_path,
        device,
    )
    factor_bank = belief_model.factor_bank.detach().cpu()
    if factor_bank.shape != (64, 3):
        raise SystemExit("bridge requires the complete 64-code factor bank")
    if belief_model.config != executor.config:
        raise SystemExit("belief and executor model configs disagree")
    expected_bank = enumerate_factor_codes(device="cpu")
    if not torch.equal(factor_bank, expected_bank):
        raise SystemExit("factor bank is not the stable Cartesian 64-code bank")
    if int(torch.unique(factor_bank, dim=0).shape[0]) != 64:
        raise SystemExit("factor bank contains duplicate codes")
    selector_audit = _selector_boundary_audit()
    if not bool(selector_audit.get("passed")):
        raise SystemExit("selector boundary audit failed")

    compact_records: list[dict[str, object]] = []
    partition_aggregate = _new_partition_counter()
    partition_by_category = defaultdict(_new_partition_counter)
    belief_coverage: list[dict[str, object]] = []

    for seed in seeds:
        _configure_determinism(torch, seed)
        for query in (
            PublicDoorQuery(axis_index)
            for axis_index in range(len(ALL_AXES))
        ):
            query_axis = ALL_AXES[query.axis_index]
            tasks = _build_nuisance_tasks(
                query_axis,
                groups=args.groups_per_query,
                split=args.split,
                master_seed=args.data_master_seed,
                candidate_seed=seed,
            )
            for group_start in range(
                0,
                args.groups_per_query,
                args.batch_size,
            ):
                group_end = min(
                    group_start + args.batch_size,
                    args.groups_per_query,
                )
                task_start = group_start * PROGRAMS_PER_GROUP
                task_end = group_end * PROGRAMS_PER_GROUP
                selected_tasks = tasks[task_start:task_end]
                representative_tasks = selected_tasks[::PROGRAMS_PER_GROUP]
                group_count = len(representative_tasks)
                if len(selected_tasks) != group_count * PROGRAMS_PER_GROUP:
                    raise AssertionError(
                        "each public group must contain all hidden query modes"
                    )

                # The public support is identical for the three possible hidden
                # query values.  Infer each prior and executor panel once per
                # public environment, then broadcast bitwise-identical results.
                exact_group_initial = _symbolic_initial_joint_log_weights(
                    torch=torch,
                    tasks=representative_tasks,
                    factor_bank=factor_bank,
                )
                learned_group_initial = (
                    _learned_marked_initial_joint_log_weights(
                        torch=torch,
                        model=belief_model,
                        tasks=representative_tasks,
                        device=device,
                    )
                )
                exact_initial = _repeat_public_groups(
                    torch=torch,
                    tensor=exact_group_initial,
                )
                learned_initial = _repeat_public_groups(
                    torch=torch,
                    tensor=learned_group_initial,
                )
                _assert_public_group_copies(
                    torch,
                    exact_initial,
                    name="exact initial belief",
                )
                _assert_public_group_copies(
                    torch,
                    learned_initial,
                    name="learned initial belief",
                )
                _assert_normalised_log_weights(torch, exact_initial)
                _assert_normalised_log_weights(torch, learned_initial)
                exact_probability = _normalise_log_weights(
                    torch,
                    exact_initial,
                ).exp()
                learned_log = _normalise_log_weights(
                    torch,
                    learned_initial,
                )
                learned_probability = learned_log.exp()
                exact_mask = torch.isfinite(exact_initial)
                for local_index in range(len(selected_tasks)):
                    mask = exact_mask[local_index]
                    kl = (
                        exact_probability[local_index, mask]
                        * (
                            exact_initial[local_index, mask]
                            - learned_log[local_index, mask]
                        )
                    ).sum()
                    belief_coverage.append(
                        {
                            "seed": seed,
                            "query_axis": query_axis.value,
                            "group_index": (
                                group_start
                                + local_index // PROGRAMS_PER_GROUP
                            ),
                            "hidden_query_slot": (
                                local_index % PROGRAMS_PER_GROUP
                            ),
                            "learned_mass_on_exact_48_codes": float(
                                learned_probability[
                                    local_index,
                                    mask,
                                ].sum()
                            ),
                            "symbolic_to_learned_kl_nats": float(kl),
                        }
                    )
                (
                    exact_group_maps,
                    learned_group_maps,
                    group_prediction,
                    _,
                ) = _canonical_candidate_panel(
                    torch=torch,
                    tasks=representative_tasks,
                    factor_bank=factor_bank,
                    executor=executor,
                    device=device,
                )
                exact_maps = _repeat_public_groups(
                    torch,
                    exact_group_maps,
                )
                learned_maps = _repeat_public_groups(
                    torch,
                    learned_group_maps,
                )
                learned_prediction = _broadcast_group_prediction(
                    torch=torch,
                    prediction=group_prediction,
                    groups=group_count,
                    candidate_count=CANDIDATES_PER_TASK,
                )
                (
                    public_states,
                    public_actions,
                    public_action_mask,
                    canonical_feedback,
                ) = _candidate_panel(
                    torch=torch,
                    tasks=selected_tasks,
                    device=device,
                )
                _assert_public_group_copies(
                    torch,
                    public_states,
                    name="candidate states",
                )
                _assert_public_group_copies(
                    torch,
                    public_actions,
                    name="candidate actions",
                )
                _assert_public_group_copies(
                    torch,
                    public_action_mask,
                    name="candidate action mask",
                )
                _assert_public_group_copies(
                    torch,
                    exact_maps,
                    name="exact candidate maps",
                )
                _assert_public_group_copies(
                    torch,
                    learned_maps,
                    name="learned candidate maps",
                )

                # The simulator map for the true latent code must reproduce the
                # canonical environment sidecar exactly.
                for local_index, task in enumerate(selected_tasks):
                    true_code = torch.tensor(
                        rule_program_factor_ids(
                            task.privileged.true_program
                        ),
                        dtype=factor_bank.dtype,
                    )
                    matches = torch.nonzero(
                        factor_bank.eq(true_code).all(dim=-1),
                    ).flatten()
                    if len(matches) != 1:
                        raise AssertionError(
                            "true program does not have one stable factor code"
                        )
                    true_index = int(matches.item())
                    if not torch.equal(
                        exact_maps[local_index, :, true_index],
                        canonical_feedback[local_index].cpu(),
                    ):
                        raise AssertionError(
                            "exact outcome map and canonical feedback disagree"
                        )

                # Partition quality is a property of the public panel, so count
                # one representative rather than its three hidden-mode copies.
                for local_group, task in enumerate(representative_tasks):
                    for candidate_index, category in enumerate(
                        task.privileged.candidate_kinds
                    ):
                        for counter in (
                            partition_aggregate,
                            partition_by_category[str(category)],
                        ):
                            _update_partition_counter(
                                counter,
                                torch=torch,
                                exact_maps=exact_group_maps[
                                    local_group,
                                    candidate_index,
                                ],
                                learned_maps=learned_group_maps[
                                    local_group,
                                    candidate_index,
                                ],
                            )

                for condition in CONDITIONS:
                    initial = (
                        exact_initial
                        if condition.belief == "exact"
                        else learned_initial
                    )
                    snapshots, _ = _rollout_condition(
                        torch=torch,
                        condition=condition,
                        tasks=selected_tasks,
                        query=query,
                        initial_log_weights=initial,
                        factor_bank=factor_bank,
                        exact_outcome_maps=exact_maps,
                        learned_outcome_maps=learned_maps,
                        learned_prediction=learned_prediction,
                        canonical_feedback=canonical_feedback,
                        budgets=budgets,
                    )
                    for budget, records in snapshots.items():
                        for local_index, record in enumerate(records):
                            compact_records.append(
                                {
                                    "seed": seed,
                                    "query_axis": query_axis.value,
                                    "group_index": (
                                        group_start
                                        + local_index // PROGRAMS_PER_GROUP
                                    ),
                                    "hidden_query_slot": (
                                        local_index % PROGRAMS_PER_GROUP
                                    ),
                                    "condition": condition.name,
                                    "budget": budget,
                                    "won": bool(record["won"]),
                                    "tie_aware_terminal_accuracy": float(
                                        record[
                                            "tie_aware_terminal_accuracy"
                                        ]
                                    ),
                                    "terminal_decision_set_size": int(
                                        record[
                                            "terminal_decision_set_size"
                                        ]
                                    ),
                                    "true_query_probability": float(
                                        record["true_door_probability"]
                                    ),
                                    "query_entropy_nats": float(
                                        record["door_entropy_nats"]
                                    ),
                                    "joint_effective_hypotheses": float(
                                        record[
                                            "joint_effective_hypotheses"
                                        ]
                                    ),
                                    "first_symbolic_identification_step": (
                                        record[
                                            "first_symbolic_identification_step"
                                        ]
                                    ),
                                    "mean_observed_log_predictive_probability": (
                                        record[
                                            "mean_observed_log_predictive_probability"
                                        ]
                                    ),
                                    "selected_candidate_indices": record[
                                        "selected_candidate_indices"
                                    ],
                                    "selected_candidate_categories_audit": (
                                        record[
                                            "selected_candidate_categories_audit"
                                        ]
                                    ),
                                }
                            )

    evaluations: dict[str, object] = {}
    for budget in budgets:
        evaluations[str(budget)] = {}
        for condition in CONDITIONS:
            records = tuple(
                record
                for record in compact_records
                if record["condition"] == condition.name
                and int(record["budget"]) == budget
            )
            evaluations[str(budget)][condition.name] = _condition_summary(
                records
            )
    evaluations_by_axis: dict[str, object] = {}
    for axis in ALL_AXES:
        axis_rows: dict[str, object] = {}
        for budget in budgets:
            axis_rows[str(budget)] = {}
            for condition in CONDITIONS:
                records = tuple(
                    record
                    for record in compact_records
                    if record["query_axis"] == axis.value
                    and record["condition"] == condition.name
                    and int(record["budget"]) == budget
                )
                axis_rows[str(budget)][condition.name] = (
                    _condition_summary(records)
                )
        evaluations_by_axis[axis.value] = axis_rows

    if 0 in budgets:
        exact_b0 = evaluations["0"][EXACT_EXACT]
        if not math.isclose(
            float(exact_b0["tie_aware_terminal_accuracy"]),
            1.0 / 3.0,
            abs_tol=1e-6,
        ):
            raise AssertionError("exact/exact B0 control must equal 1/3")
    if 1 in budgets:
        exact_b1 = evaluations["1"][EXACT_EXACT]
        if (
            not math.isclose(
                float(exact_b1["tie_aware_terminal_accuracy"]),
                1.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(
                    exact_b1["first_selection_category_rates_audit"][
                        "query-atomic"
                    ]
                ),
                1.0,
                abs_tol=1e-6,
            )
        ):
            raise AssertionError(
                "exact/exact B1 control must select a query probe and solve"
            )

    first_choice_agreement = _first_choice_agreement(compact_records)
    public_choice_consistency = _public_first_choice_consistency(
        compact_records
    )
    if 1 in budgets and not bool(public_choice_consistency["passed"]):
        raise AssertionError(
            "hidden modes changed a condition's first public action"
        )
    belief_coverage_summary = {
        "tasks": len(belief_coverage),
        "public_environment_seed_instances": (
            len(seeds) * len(ALL_AXES) * args.groups_per_query
        ),
        "unique_base_public_environments": (
            len(ALL_AXES) * args.groups_per_query
        ),
        "mean_mass_on_exact_48_codes": _mean(
            record["learned_mass_on_exact_48_codes"]
            for record in belief_coverage
        ),
        "mean_symbolic_to_learned_kl_nats": _mean(
            record["symbolic_to_learned_kl_nats"]
            for record in belief_coverage
        ),
        "task_records": belief_coverage,
    }
    partition_audit = {
        "aggregate": _finalise_partition_counter(partition_aggregate),
        "by_candidate_category": {
            category: _finalise_partition_counter(counter)
            for category, counter in sorted(partition_by_category.items())
        },
        "note": (
            "One public panel per (seed, query axis, group) is counted; "
            "the three hidden query slots share that panel."
        ),
    }
    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "experiment": "query_conditioned_nuisance_learned_2x2_bridge",
        "interpretation": (
            "Paired fixed-budget oracle-canonical bridge isolating initial "
            "rule-belief "
            "quality from candidate-outcome model quality. The learned belief "
            "is used only as the t=0 initializer; every later update is an "
            "explicit Bayes update under the selected outcome model."
        ),
        "conditions": [
            {
                "name": condition.name,
                "initial_belief": condition.belief,
                "candidate_outcome_model": condition.outcome,
                "posterior_update": (
                    "deterministic exact outcome equality"
                    if condition.outcome == "exact"
                    else (
                        "selected learned OutcomePrediction.log_prob on "
                        "canonical feedback"
                    )
                ),
            }
            for condition in CONDITIONS
        ],
        "paired_protocol": (
            "All four conditions share tasks, candidate ordering, hidden "
            "programs, query axis, canonical feedback, seeds, and budgets."
        ),
        "acquisition": (
            "query-conditioned expected terminal-door gain over MAP outcome "
            "partitions; learned conditions use outcome_map(prediction)"
        ),
        "public_group_inference_cache": (
            "Belief and outcome models are evaluated once per public "
            "environment and broadcast bitwise over its three hidden modes."
        ),
        "feedback_read_only_after_candidate_selection": True,
        "budget_semantics": (
            "Exactly B probes are executed with no early stopping; later-budget "
            "regression diagnoses posterior-update instability."
        ),
        "structural_limitation": (
            "The current menu contains strong query-atomic probes that exactly "
            "identify the queried factor under the simulator. B1 therefore "
            "tests whether a condition finds and assimilates that probe, while "
            "B0 belief quality and later fixed-budget posterior stability are "
            "more discriminating; this is not yet a hard test of learned "
            "belief-driven multi-step acquisition."
        ),
        "selector_boundary_audit": selector_audit,
        "budgets": list(budgets),
        "seeds": list(seeds),
        "groups_per_query": args.groups_per_query,
        "programs_per_group": PROGRAMS_PER_GROUP,
        "batch_size_public_groups": args.batch_size,
        "total_hidden_task_seed_instances": (
            len(seeds)
            * len(ALL_AXES)
            * args.groups_per_query
            * PROGRAMS_PER_GROUP
        ),
        "unique_base_hidden_tasks": (
            len(ALL_AXES)
            * args.groups_per_query
            * PROGRAMS_PER_GROUP
        ),
        "evaluations": evaluations,
        "evaluations_by_axis": evaluations_by_axis,
        "first_choice_agreement_audit": first_choice_agreement,
        "public_first_choice_consistency_audit": (
            public_choice_consistency
        ),
        "learned_belief_initial_coverage": belief_coverage_summary,
        "learned_outcome_partition_audit": partition_audit,
        "compact_task_records": compact_records,
        "belief_checkpoint": str(belief_path),
        "belief_checkpoint_sha256": _sha256(belief_path),
        "belief_checkpoint_model_type": belief_checkpoint.get("model_type"),
        "active_executor_checkpoint": str(executor_path),
        "active_executor_checkpoint_sha256": _sha256(executor_path),
        "active_executor_checkpoint_schema": executor_checkpoint.get(
            "checkpoint_schema_version"
        ),
        "active_executor_original_gate_passed": executor_result.get(
            "active_prefix_executor_gate",
            {},
        ).get("passed"),
        "device": str(device),
        "split": args.split,
        "data_master_seed": args.data_master_seed,
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output / "result.json"),
                "evaluations": evaluations,
                "first_choice_agreement_audit": first_choice_agreement,
                "public_first_choice_consistency_audit": (
                    public_choice_consistency
                ),
                "learned_belief_initial_coverage": {
                    key: value
                    for key, value in belief_coverage_summary.items()
                    if key != "task_records"
                },
                "learned_outcome_partition_audit": partition_audit,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
