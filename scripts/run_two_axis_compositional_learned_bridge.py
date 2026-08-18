#!/usr/bin/env python3
"""Bridge the learned factor executor into the exact two-axis query game.

The exact two-axis runner supplies a balanced 16-hypothesis scenario, a
four-door compositional query, and exact RuleGrid outcome partitions.  This
runner reconstructs only the matching public palette/probe panel, evaluates a
frozen ``OracleFactorExecutor`` for all eight probes and all 64 factor codes,
and crosses two choices independently:

* exact or learned-MAP partitions for query-success action selection; and
* exact filtering or learned full-grid proper likelihood for Bayes updates.

All four conditions start from the same exact uniform posterior over the
scenario's 16 allowed hypotheses.  The learned executor never sees candidate
categories, the hidden truth, the query door, or simulator class IDs.
Canonical environment feedback is indexed only after a candidate is selected.

This remains an oracle-canonical diagnostic rather than an ARC-AGI-3 score:
the factor codebook, palette-role canonicalization, query axes, and initial
16-code restriction are privileged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, fields
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.run_nuisance_learned_bridge import (  # noqa: E402
    _finalise_partition_counter,
    _new_partition_counter,
    _update_partition_counter,
)
from scripts.run_forced_cross_axis_likelihood_audit import (  # noqa: E402
    axis_project_log_likelihood,
)
from scripts.run_oracle_canonical_acquisition_ceiling import (  # noqa: E402
    DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    _load_active_executor,
    _normalise_log_weights,
    _selected_prediction,
    bayesian_log_likelihood_update,
)
from scripts.run_two_axis_compositional_acquisition import (  # noqa: E402
    CANDIDATES_PER_SCENARIO,
    DEFAULT_BUDGETS,
    FACTOR_AXES,
    INITIAL_HYPOTHESES,
    RESULT_SCHEMA_VERSION as EXACT_RESULT_SCHEMA_VERSION,
    TwoAxisScenario,
    _build_scenario,
    _candidate_menu,
    _complementary_relevant_axes,
    _door_marginals,
    _entropy_bits,
    _factor_programs,
    _initial_log_weights,
    _posterior_update,
    _seed64,
    _select_candidate,
)


RESULT_SCHEMA_VERSION = "prp-wm.two-axis-compositional-learned-bridge.v2"
DEFAULT_SEEDS = (20260911,)
EXACT_EXACT = "exact-select/exact-update"
LEARNED_LEARNED = "learned-select/learned-update"
EXACT_LEARNED = "exact-select/learned-update"
LEARNED_EXACT = "learned-select/exact-update"
EXACT_PROJECTED = "exact-select/projected-learned-update"
LEARNED_PROJECTED = "learned-select/projected-learned-update"


@dataclass(frozen=True)
class BridgeCondition:
    """One independently controlled selection/update condition."""

    name: str
    selection: str
    update: str

    def __post_init__(self) -> None:
        if self.selection not in {"exact", "learned"}:
            raise ValueError("selection must be exact or learned")
        if self.update not in {"exact", "learned", "projected-learned"}:
            raise ValueError(
                "update must be exact, learned, or projected-learned"
            )


CONDITIONS = (
    BridgeCondition(EXACT_EXACT, "exact", "exact"),
    BridgeCondition(LEARNED_LEARNED, "learned", "learned"),
    BridgeCondition(EXACT_LEARNED, "exact", "learned"),
    BridgeCondition(LEARNED_EXACT, "learned", "exact"),
    BridgeCondition(EXACT_PROJECTED, "exact", "projected-learned"),
    BridgeCondition(LEARNED_PROJECTED, "learned", "projected-learned"),
)


@dataclass(frozen=True)
class LearnedScenarioPanel:
    """Frozen executor outputs aligned to one ``TwoAxisScenario``."""

    exact_grids: Any  # long [P,64,H,W], canonical environment outcomes
    learned_maps: Any  # long [P,64,H,W], canonical executor MAP outcomes
    learned_class_ids: Any  # long [P,64], equality partitions of learned_maps
    learned_prediction: Any  # OutcomePrediction with flattened [P,64,...]
    partition_alignment_passed: bool
    public_states_shape: tuple[int, ...]
    public_actions_shape: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--active-executor-checkpoint",
        type=Path,
        default=DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    )
    parser.add_argument("--groups-per-pair", type=int, default=16)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--data-master-seed", type=int, default=2026072401)
    parser.add_argument(
        "--split",
        default="two-axis-compositional-learned-bridge",
    )
    parser.add_argument("--trace-scenarios", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if args.groups_per_pair <= 0:
        raise SystemExit("--groups-per-pair must be positive")
    if args.trace_scenarios < 0:
        raise SystemExit("--trace-scenarios must be non-negative")
    if args.data_master_seed < 0:
        raise SystemExit("--data-master-seed must be non-negative")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free name")
    budgets = tuple(int(value) for value in args.budgets)
    if (
        len(set(budgets)) != len(budgets)
        or tuple(sorted(budgets)) != DEFAULT_BUDGETS
    ):
        raise SystemExit("--budgets must contain exactly 0 1 2 3")
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


def _canonicalize_with_palette(torch: Any, raw: Any, palette: Any) -> Any:
    """Map one arbitrary-shaped grid tensor to stable palette-role IDs."""

    from prp_wm.rulegrid import NUM_COLORS

    lookup = list(range(NUM_COLORS))
    for canonical_id, field in enumerate(fields(palette), start=1):
        lookup[getattr(palette, field.name)] = canonical_id
    table = torch.tensor(lookup, dtype=torch.long, device=raw.device)
    return table[raw]


def _maps_to_class_ids(torch: Any, maps: Any) -> Any:
    """Convert ``[P,K,H,W]`` grids to one equality-class row per candidate."""

    if maps.ndim != 4:
        raise ValueError("outcome maps must have [P,K,H,W] shape")
    rows = []
    for candidate in maps:
        _, inverse = torch.unique(
            candidate.flatten(start_dim=1),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        rows.append(inverse)
    return torch.stack(rows, dim=0)


def _same_partition(torch: Any, left: Any, right: Any) -> bool:
    """Compare class labels up to an arbitrary relabeling."""

    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("partition rows must be equally shaped vectors")
    return bool(
        left[:, None].eq(left[None]).eq(
            right[:, None].eq(right[None]),
        ).all()
    )


def _reconstruct_public_panel(
    torch: Any,
    *,
    scenario: TwoAxisScenario,
    query_axis_indices: tuple[int, int],
    group_index: int,
    split: str,
    master_seed: int,
    policy_seed: int,
) -> tuple[Any, tuple[Any, ...], Any]:
    """Rebuild only the public probes/palette used by the exact scenario."""

    from prp_wm.rulegrid import ALL_AXES, palette_from_seed

    pair_name = "-".join(
        ALL_AXES[index].value for index in query_axis_indices
    )
    scenario_seed = _seed64(
        EXACT_RESULT_SCHEMA_VERSION,
        split,
        master_seed,
        pair_name,
        group_index,
    )
    palette = palette_from_seed(_seed64(scenario_seed, "palette"))
    probes, categories, axes = _candidate_menu(
        palette=palette,
        query_axis_indices=query_axis_indices,
        order_seed=_seed64(
            scenario_seed,
            "candidate-order",
            policy_seed,
        ),
    )
    if categories != scenario.candidate_categories:
        raise AssertionError("reconstructed candidate categories changed")
    if axes != scenario.candidate_axis_indices:
        raise AssertionError("reconstructed candidate axes changed")
    if tuple(probe.probe_id for probe in probes) != scenario.candidate_probe_ids:
        raise AssertionError("reconstructed public probe IDs changed")
    return palette, probes, scenario_seed


def _build_learned_panel(
    torch: Any,
    *,
    executor: Any,
    scenario: TwoAxisScenario,
    query_axis_indices: tuple[int, int],
    group_index: int,
    split: str,
    master_seed: int,
    policy_seed: int,
    device: Any,
) -> LearnedScenarioPanel:
    """Evaluate all eight public probes under all 64 frozen factor codes."""

    from prp_wm.causal_filter import predict_factor_panel
    from prp_wm.latent_rules import outcome_map
    from prp_wm.neural import encode_public_action
    from prp_wm.rulegrid import simulate

    palette, probes, _ = _reconstruct_public_panel(
        torch,
        scenario=scenario,
        query_axis_indices=query_axis_indices,
        group_index=group_index,
        split=split,
        master_seed=master_seed,
        policy_seed=policy_seed,
    )
    programs = _factor_programs()
    raw_states = torch.tensor(
        [[probe.state for probe in probes]],
        dtype=torch.long,
        device=device,
    )
    states = _canonicalize_with_palette(torch, raw_states, palette)
    encoded = tuple(encode_public_action(probe.action) for probe in probes)
    if any(tuple(action.shape) != (1, 4) for action in encoded):
        raise AssertionError("the exact two-axis menu must use atomic actions")
    actions = torch.stack(
        tuple(action[0] for action in encoded),
        dim=0,
    )[None].to(device)
    raw_exact = torch.tensor(
        [
            [
                simulate(probe.state, probe.action, program, palette)
                for program in programs
            ]
            for probe in probes
        ],
        dtype=torch.long,
        device=device,
    )
    exact_grids_device = _canonicalize_with_palette(
        torch,
        raw_exact,
        palette,
    )
    factor_ids = scenario.factor_bank.to(device)[None]
    with torch.no_grad():
        prediction = predict_factor_panel(
            executor,
            states,
            actions,
            factor_ids,
        )
        learned_maps = outcome_map(prediction).reshape(
            CANDIDATES_PER_SCENARIO,
            scenario.factor_bank.shape[0],
            executor.config.grid_size,
            executor.config.grid_size,
        )

    exact_grids = exact_grids_device.detach().cpu()
    learned_maps_cpu = learned_maps.detach().cpu()
    exact_classes = _maps_to_class_ids(torch, exact_grids)
    alignment = all(
        _same_partition(
            torch,
            exact_classes[index],
            scenario.candidate_class_ids[index],
        )
        for index in range(CANDIDATES_PER_SCENARIO)
    )
    if not alignment:
        raise AssertionError(
            "reconstructed canonical grids changed exact scenario partitions"
        )
    return LearnedScenarioPanel(
        exact_grids=exact_grids,
        learned_maps=learned_maps_cpu,
        learned_class_ids=_maps_to_class_ids(torch, learned_maps_cpu),
        learned_prediction=prediction,
        partition_alignment_passed=alignment,
        public_states_shape=tuple(states.shape),
        public_actions_shape=tuple(actions.shape),
    )


def _assert_normalised(torch: Any, log_weights: Any) -> None:
    normalizer = torch.logsumexp(log_weights, dim=-1)
    if not bool(torch.isfinite(normalizer).item()):
        raise AssertionError("posterior has no finite probability mass")
    if not torch.allclose(
        normalizer,
        torch.zeros_like(normalizer),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise AssertionError("posterior is not normalized")


def _learned_update(
    torch: Any,
    *,
    log_weights: Any,
    log_likelihood: Any,
) -> tuple[Any, float]:
    posterior, evidence = bayesian_log_likelihood_update(
        torch,
        log_weights[None],
        log_likelihood[None],
    )
    result = posterior[0]
    _assert_normalised(torch, result)
    return result, float(evidence[0])


def _rollout_condition(
    torch: Any,
    *,
    scenario: TwoAxisScenario,
    panel: LearnedScenarioPanel,
    truth_index: int,
    condition: BridgeCondition,
    budgets: Sequence[int],
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    """Run one truth with selection and update models independently chosen."""

    log_weights = _initial_log_weights(torch, scenario)
    _assert_normalised(torch, log_weights)
    available = torch.ones(CANDIDATES_PER_SCENARIO, dtype=torch.bool)
    selection_classes = (
        scenario.candidate_class_ids
        if condition.selection == "exact"
        else panel.learned_class_ids
    )
    if not bool(
        scenario.hypothesis_indices.eq(int(truth_index)).any().item()
    ):
        raise ValueError("truth must be one of the 16 initial hypotheses")
    selected_indices: list[int] = []
    selected_scores: list[Any] = []
    log_evidences: list[float] = []
    snapshots: dict[int, dict[str, object]] = {}
    target_door = int(scenario.door_values[truth_index])
    task_id = f"{scenario.scenario_id}/H{truth_index:02d}"

    def snapshot(budget: int) -> None:
        probabilities = _normalise_log_weights(torch, log_weights).exp()
        door_marginals = _door_marginals(
            torch,
            log_weights,
            scenario.door_values,
        )
        maximum = door_marginals.max()
        tied = torch.isclose(
            door_marginals,
            maximum,
            atol=1e-7,
            rtol=1e-6,
        )
        tied_count = int(tied.sum())
        chosen_door = int(torch.argmax(door_marginals))
        snapshots[budget] = {
            "task_id": task_id,
            "scenario_id": scenario.scenario_id,
            "truth_factor_index": int(truth_index),
            "target_door": target_door,
            "chosen_door": chosen_door,
            "won": chosen_door == target_door,
            "tie_aware_terminal_accuracy": (
                1.0 / tied_count if bool(tied[target_door]) else 0.0
            ),
            "optimal_query_success_probability": float(maximum),
            "true_query_probability": float(door_marginals[target_door]),
            "query_entropy_bits": _entropy_bits(door_marginals),
            "joint_entropy_bits": _entropy_bits(probabilities),
            "posterior_effective_hypotheses": (
                2.0 ** _entropy_bits(probabilities)
            ),
            "query_identified": float(maximum) >= 1.0 - 1e-6,
            "selected_candidate_indices": selected_indices[:budget],
            "selected_candidate_categories_audit": [
                scenario.candidate_categories[index]
                for index in selected_indices[:budget]
            ],
            "b2_complementary_relevant_axes": (
                _complementary_relevant_axes(scenario, selected_indices)
                if budget >= 2
                else None
            ),
            "mean_observed_log_predictive_probability": (
                sum(log_evidences[:budget]) / len(log_evidences[:budget])
                if log_evidences[:budget]
                else None
            ),
        }

    if 0 in budgets:
        snapshot(0)
    for step in range(1, max(budgets) + 1):
        score = _select_candidate(
            torch,
            policy="query-success-greedy",
            log_weights=log_weights,
            door_values=scenario.door_values,
            candidate_class_ids=selection_classes,
            available=available,
            uniform_order=scenario.uniform_order,
            step=step,
        )
        candidate_index = int(score.candidate_index)
        available[candidate_index] = False
        selected_indices.append(candidate_index)
        selected_scores.append(score)

        # The exact feedback sidecar is not indexed until selection is fixed.
        if condition.update == "exact":
            observed_class = int(
                scenario.candidate_class_ids[candidate_index, truth_index]
            )
            log_weights = _posterior_update(
                torch,
                log_weights,
                scenario.candidate_class_ids[candidate_index],
                observed_class,
            )
        else:
            if panel.learned_prediction is None:
                raise ValueError("learned update requires an OutcomePrediction")
            selected_prediction = _selected_prediction(
                torch=torch,
                prediction=panel.learned_prediction,
                candidate_indices=(candidate_index,),
                candidate_count=CANDIDATES_PER_SCENARIO,
            )
            feedback = panel.exact_grids[
                candidate_index,
                truth_index,
            ].to(selected_prediction.change_logits.device)[None]
            with torch.no_grad():
                log_likelihood = selected_prediction.log_prob(
                    feedback,
                )[0].cpu()
            if not bool(torch.isfinite(log_likelihood).all().item()):
                raise AssertionError(
                    "learned proper likelihood must be finite"
                )
            if condition.update == "projected-learned":
                log_likelihood = axis_project_log_likelihood(
                    torch,
                    log_likelihood,
                    scenario.factor_bank,
                    scenario.candidate_axis_indices[candidate_index],
                )
            log_weights, evidence = _learned_update(
                torch,
                log_weights=log_weights,
                log_likelihood=log_likelihood,
            )
            log_evidences.append(evidence)
        _assert_normalised(torch, log_weights)
        if step in budgets:
            snapshot(step)
    trace = {
        "task_id": task_id,
        "condition": condition.name,
        "selected_candidate_indices": selected_indices,
        "selected_candidate_categories_audit": [
            scenario.candidate_categories[index]
            for index in selected_indices
        ],
        "step_scores": [
            {
                "candidate_index": int(score.candidate_index),
                "expected_query_success": float(
                    score.expected_query_success
                ),
                "expected_query_gain": float(score.expected_query_gain),
                "query_information_bits": float(
                    score.query_information_bits
                ),
                "global_information_bits": float(
                    score.global_information_bits
                ),
            }
            for score in selected_scores
        ],
        "observed_log_predictive_probabilities": log_evidences,
    }
    return snapshots, trace


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _summarise_condition(
    records: Sequence[dict[str, object]],
    *,
    budget: int,
) -> dict[str, object]:
    categories = (
        "relevant-axis-0",
        "relevant-axis-1",
        "nuisance-axis",
        "neutral",
    )
    step_rates: dict[str, object] = {}
    for step in range(1, budget + 1):
        selected = tuple(
            str(record["selected_candidate_categories_audit"][step - 1])
            for record in records
        )
        step_rates[str(step)] = {
            category: selected.count(category) / len(selected)
            for category in categories
        }
    evidences = tuple(
        float(record["mean_observed_log_predictive_probability"])
        for record in records
        if record["mean_observed_log_predictive_probability"] is not None
    )
    return {
        "tasks": len(records),
        "terminal_win_rate": _mean(
            int(bool(record["won"])) for record in records
        ),
        "tie_aware_terminal_accuracy": _mean(
            float(record["tie_aware_terminal_accuracy"])
            for record in records
        ),
        "mean_optimal_query_success_probability": _mean(
            float(record["optimal_query_success_probability"])
            for record in records
        ),
        "mean_true_query_probability": _mean(
            float(record["true_query_probability"]) for record in records
        ),
        "mean_query_entropy_bits": _mean(
            float(record["query_entropy_bits"]) for record in records
        ),
        "mean_joint_entropy_bits": _mean(
            float(record["joint_entropy_bits"]) for record in records
        ),
        "query_identified_rate": _mean(
            int(bool(record["query_identified"])) for record in records
        ),
        "b2_complementary_relevant_axes_rate": (
            _mean(
                int(bool(record["b2_complementary_relevant_axes"]))
                for record in records
            )
            if budget >= 2
            else None
        ),
        "selected_category_rates_by_step_audit": step_rates,
        "mean_observed_log_predictive_probability": (
            _mean(evidences) if evidences else None
        ),
    }


def _summarise_records(
    records: dict[str, dict[int, list[dict[str, object]]]],
    budgets: Sequence[int],
) -> dict[str, object]:
    return {
        str(budget): {
            condition.name: _summarise_condition(
                records[condition.name][budget],
                budget=budget,
            )
            for condition in CONDITIONS
        }
        for budget in budgets
    }


def _exact_control(summary: dict[str, object]) -> dict[str, object]:
    expected_success = {"0": 0.25, "1": 0.5, "2": 1.0}
    expected_entropy = {"0": 2.0, "1": 1.0, "2": 0.0}
    checks: dict[str, bool] = {}
    for budget, expected in expected_success.items():
        row = summary[budget][EXACT_EXACT]
        checks[f"b{budget}_success"] = math.isclose(
            float(row["tie_aware_terminal_accuracy"]),
            expected,
            abs_tol=1e-7,
        ) and math.isclose(
            float(row["mean_optimal_query_success_probability"]),
            expected,
            abs_tol=1e-7,
        )
        checks[f"b{budget}_query_entropy"] = math.isclose(
            float(row["mean_query_entropy_bits"]),
            expected_entropy[budget],
            abs_tol=1e-7,
        )
    checks["b2_complementary_axes"] = math.isclose(
        float(
            summary["2"][EXACT_EXACT][
                "b2_complementary_relevant_axes_rate"
            ]
        ),
        1.0,
        abs_tol=1e-7,
    )
    return {
        "expected_success": expected_success,
        "expected_query_entropy_bits": expected_entropy,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_experiment(
    *,
    torch: Any,
    executor: Any,
    groups_per_pair: int,
    seeds: Sequence[int],
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    split: str = "two-axis-compositional-learned-bridge",
    data_master_seed: int = 2026072401,
    device: Any = "cpu",
    trace_scenarios: int = 1,
) -> dict[str, object]:
    """Run the paired four-condition learned bridge."""

    from prp_wm.rulegrid import ALL_AXES

    budgets = tuple(sorted(int(value) for value in budgets))
    seeds = tuple(int(value) for value in seeds)
    query_pairs = tuple(itertools.combinations(range(FACTOR_AXES), 2))
    records = {
        condition.name: {budget: [] for budget in budgets}
        for condition in CONDITIONS
    }
    by_pair = {
        "-".join(ALL_AXES[index].value for index in pair): {
            condition.name: {budget: [] for budget in budgets}
            for condition in CONDITIONS
        }
        for pair in query_pairs
    }
    partition_aggregate = _new_partition_counter()
    partition_by_category = defaultdict(_new_partition_counter)
    compact_records: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    scenario_audits: list[dict[str, object]] = []
    traced_scenarios = 0

    executor.eval()
    for seed in seeds:
        for query_pair in query_pairs:
            pair_name = "-".join(
                ALL_AXES[index].value for index in query_pair
            )
            for group_index in range(groups_per_pair):
                scenario = _build_scenario(
                    torch,
                    query_axis_indices=query_pair,
                    group_index=group_index,
                    split=split,
                    master_seed=data_master_seed,
                    policy_seed=seed,
                )
                panel = _build_learned_panel(
                    torch,
                    executor=executor,
                    scenario=scenario,
                    query_axis_indices=query_pair,
                    group_index=group_index,
                    split=split,
                    master_seed=data_master_seed,
                    policy_seed=seed,
                    device=device,
                )
                for candidate_index, category in enumerate(
                    scenario.candidate_categories
                ):
                    for counter in (
                        partition_aggregate,
                        partition_by_category[str(category)],
                    ):
                        _update_partition_counter(
                            counter,
                            torch=torch,
                            exact_maps=panel.exact_grids[candidate_index],
                            learned_maps=panel.learned_maps[candidate_index],
                        )
                scenario_audits.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "query_axes": list(scenario.query_axis_names),
                        "nuisance_axis": scenario.nuisance_axis_name,
                        "allowed_query_values": [
                            list(values)
                            for values in scenario.allowed_query_values
                        ],
                        "initial_hypotheses": int(
                            scenario.hypothesis_indices.numel()
                        ),
                        "candidate_categories_audit": list(
                            scenario.candidate_categories
                        ),
                        "exact_partition_alignment_passed": (
                            panel.partition_alignment_passed
                        ),
                        "public_states_shape": list(panel.public_states_shape),
                        "public_actions_shape": list(panel.public_actions_shape),
                    }
                )
                keep_trace = traced_scenarios < trace_scenarios
                for condition in CONDITIONS:
                    for truth_index in scenario.hypothesis_indices.tolist():
                        snapshots, trace = _rollout_condition(
                            torch,
                            scenario=scenario,
                            panel=panel,
                            truth_index=int(truth_index),
                            condition=condition,
                            budgets=budgets,
                        )
                        for budget, record in snapshots.items():
                            records[condition.name][budget].append(record)
                            by_pair[pair_name][condition.name][budget].append(
                                record
                            )
                            compact_records.append(
                                {
                                    "seed": seed,
                                    "query_pair": pair_name,
                                    "group_index": group_index,
                                    "condition": condition.name,
                                    "budget": budget,
                                    **record,
                                }
                            )
                        if keep_trace and int(truth_index) == int(
                            scenario.hypothesis_indices[0]
                        ):
                            traces.append(trace)
                if keep_trace:
                    traced_scenarios += 1

    summary = _summarise_records(records, budgets)
    by_pair_summary = {
        pair_name: _summarise_records(pair_records, budgets)
        for pair_name, pair_records in by_pair.items()
    }
    exact_control = _exact_control(summary)
    if not exact_control["passed"]:
        raise AssertionError("exact-select/exact-update control failed")
    partition_audit = {
        "aggregate": _finalise_partition_counter(partition_aggregate),
        "by_candidate_category": {
            category: _finalise_partition_counter(counter)
            for category, counter in sorted(partition_by_category.items())
        },
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "experiment_kind": "two-axis-compositional-learned-2x2-bridge",
        "conditions": [
            {
                "name": condition.name,
                "selection_partition": condition.selection,
                "posterior_update": condition.update,
            }
            for condition in CONDITIONS
        ],
        "protocol": {
            "initial_posterior": (
                "exact uniform over the scenario's 16 allowed factor codes"
            ),
            "selection_policy": "query-success-greedy",
            "learned_selection": (
                "equality partitions of frozen executor coherent MAP grids"
            ),
            "learned_update": (
                "full canonical 8x8 OutcomePrediction.log_prob feedback; "
                "no MAP equality or axis projection"
            ),
            "projected_learned_update": (
                "the same full-grid learned log likelihood, oracle-projected "
                "by the selected candidate's acted factor axis using a "
                "log-mean-exp nuisance-fiber marginal"
            ),
            "feedback_read_only_after_candidate_selection": True,
            "factor_bank": "complete Cartesian 64-code bank",
            "candidate_count": CANDIDATES_PER_SCENARIO,
            "budgets": list(budgets),
            "paired": (
                "all six conditions share scenarios, initial posterior, "
                "candidate order, truth codes, feedback, and budgets"
            ),
        },
        "groups_per_pair": groups_per_pair,
        "seeds": list(seeds),
        "scenario_count": (
            len(query_pairs) * groups_per_pair * len(seeds)
        ),
        "truth_tasks_per_scenario": INITIAL_HYPOTHESES,
        "aggregate": summary,
        "by_query_pair": by_pair_summary,
        "b2_summary": {
            condition.name: {
                "tie_aware_terminal_accuracy": summary["2"][
                    condition.name
                ]["tie_aware_terminal_accuracy"],
                "terminal_win_rate": summary["2"][condition.name][
                    "terminal_win_rate"
                ],
                "complementary_relevant_axes_rate": summary["2"][
                    condition.name
                ]["b2_complementary_relevant_axes_rate"],
                "mean_query_entropy_bits": summary["2"][condition.name][
                    "mean_query_entropy_bits"
                ],
            }
            for condition in CONDITIONS
        },
        "exact_control": exact_control,
        "learned_partition_audit": partition_audit,
        "scenario_audits": scenario_audits,
        "traces": traces,
        "compact_task_records": compact_records,
        "limitations": [
            "known factor codebook and exact 16-code initial restriction",
            "oracle palette-role canonicalization",
            "fixed atomic RuleGrid probe family",
            "learned belief inference is not included",
            "projected-learned controls use the oracle acted-axis identity",
        ],
    }


def main() -> None:
    args = parse_args()
    budgets, seeds = _validate_args(args)

    import torch

    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, seeds[0])
    checkpoint_path = args.active_executor_checkpoint.resolve()
    executor, checkpoint, checkpoint_result = _load_active_executor(
        torch,
        checkpoint_path,
        device,
    )
    result = run_experiment(
        torch=torch,
        executor=executor,
        groups_per_pair=args.groups_per_pair,
        seeds=seeds,
        budgets=budgets,
        split=args.split,
        data_master_seed=args.data_master_seed,
        device=device,
        trace_scenarios=args.trace_scenarios,
    )
    result.update(
        {
            "active_executor_checkpoint": str(checkpoint_path),
            "active_executor_checkpoint_sha256": _sha256(checkpoint_path),
            "active_executor_checkpoint_schema": checkpoint.get(
                "checkpoint_schema_version"
            ),
            "active_executor_original_gate_passed": checkpoint_result.get(
                "active_prefix_executor_gate",
                {},
            ).get("passed"),
            "device": str(device),
            "split": args.split,
            "data_master_seed": args.data_master_seed,
        }
    )
    _atomic_json(args.output, result)
    b2 = result["b2_summary"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "scenario_count": result["scenario_count"],
                "exact_control_passed": result["exact_control"]["passed"],
                "b2_summary": b2,
                "partition_f1": result["learned_partition_audit"][
                    "aggregate"
                ]["same_outcome_pair_f1"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
