#!/usr/bin/env python3
"""Exact two-axis compositional acquisition in the RuleGrid simulator.

The experiment isolates a small but important active-reasoning question:
when a four-way query is the Cartesian product of two latent factors, does an
acquisition policy deliberately test both relevant factors?  Every scenario
uses the full 4x4x4 RuleGrid factor bank.  Two query axes are each restricted
to two values, while the third axis retains all four values, leaving exactly
2x2x4=16 initially possible hypotheses.  The two restricted values on each
query axis map compositionally to four terminal doors.

The public candidate menu contains two exact atomic probes per RuleGrid axis
and two rule-independent neutral probes.  Its order is shuffled without using
the hidden program.  Five paired policies share the same scenario, candidate
order, exact simulator partitions, feedback, and budgets:

* the existing greedy expected-query-success selector;
* one-step query mutual information;
* one-step global mutual information;
* a receding-horizon exact depth-two dynamic program; and
* a uniform order sampled without replacement.

This is a privileged symbolic control, not an ARC-AGI-3 score.  It assumes the
factor codebook, the two query axes, the restricted value sets, exact RuleGrid
outcome partitions, and exact Bayesian filtering.  Its purpose is to test the
acquisition objective independently of learned world-model calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.run_oracle_canonical_acquisition_ceiling import (  # noqa: E402
    _normalise_log_weights,
    _select_acquisition_candidate,
)
from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (  # noqa: E402
    _atomic_probe,
    _select_global_information_candidate,
)


RESULT_SCHEMA_VERSION = "prp-wm.two-axis-compositional-acquisition.v1"
POLICIES = (
    "query-success-greedy",
    "query-mi",
    "global-mi",
    "depth-2-dp",
    "uniform",
)
QUERY_AWARE_POLICIES = (
    "query-success-greedy",
    "query-mi",
    "depth-2-dp",
)
DEFAULT_BUDGETS = (0, 1, 2, 3)
DEFAULT_SEEDS = (20260911, 20260912, 20260913)
FACTOR_CARDINALITY = 4
FACTOR_AXES = 3
FACTOR_BANK_SIZE = FACTOR_CARDINALITY**FACTOR_AXES
INITIAL_HYPOTHESES = 2 * 2 * 4
CANDIDATES_PER_SCENARIO = 8
DOORS = 4
_LOG_2 = math.log(2.0)


@dataclass(frozen=True)
class TwoAxisScenario:
    """One public candidate bank and its exact simulator partitions."""

    scenario_id: str
    query_axis_indices: tuple[int, int]
    query_axis_names: tuple[str, str]
    nuisance_axis_index: int
    nuisance_axis_name: str
    allowed_query_values: tuple[tuple[int, int], tuple[int, int]]
    factor_bank: Any
    hypothesis_indices: Any
    door_values: Any
    candidate_class_ids: Any
    candidate_categories: tuple[str, ...]
    candidate_axis_indices: tuple[int | None, ...]
    candidate_probe_ids: tuple[str, ...]
    partition_validation: dict[str, object]
    uniform_order: tuple[int, ...]


@dataclass(frozen=True)
class CandidateScore:
    """Common trace representation for all acquisition policies."""

    candidate_index: int
    expected_query_success: float
    expected_query_gain: float
    query_information_bits: float
    global_information_bits: float
    depth_two_value: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
        default="two-axis-compositional-acquisition",
    )
    parser.add_argument("--trace-scenarios", type=int, default=3)
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


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed64(*parts: object) -> int:
    source = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "little")


def _entropy_bits(probabilities: Any) -> float:
    nonzero = probabilities > 0
    if not bool(nonzero.any().item()):
        return 0.0
    entropy_nats = -(
        probabilities[nonzero] * probabilities[nonzero].log()
    ).sum()
    return float(entropy_nats) / _LOG_2


def _normalised_probabilities(torch: Any, log_weights: Any) -> Any:
    return _normalise_log_weights(torch, log_weights).exp()


def _door_marginals(torch: Any, log_weights: Any, door_values: Any) -> Any:
    weights = _normalised_probabilities(torch, log_weights)
    result = torch.zeros(
        DOORS,
        dtype=weights.dtype,
        device=weights.device,
    )
    result.scatter_add_(0, door_values, weights)
    return result / result.sum()


def _partition_inverse(torch: Any, class_ids: Any) -> tuple[Any, int]:
    flattened = class_ids.reshape(class_ids.shape[0], -1)
    _, inverse = torch.unique(
        flattened,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    classes = int(inverse.max().item()) + 1
    return inverse, classes


def _information_scores(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    class_ids: Any,
) -> tuple[float, float]:
    """Return I(door; outcome) and I(hypothesis; outcome), in bits."""

    weights = _normalised_probabilities(torch, log_weights)
    inverse, classes = _partition_inverse(torch, class_ids)
    joint = torch.zeros(
        (classes, DOORS),
        dtype=weights.dtype,
        device=weights.device,
    )
    joint.view(-1).scatter_add_(0, inverse * DOORS + door_values, weights)
    class_mass = joint.sum(dim=-1)
    door_mass = joint.sum(dim=0)
    query_entropy = _entropy_bits(door_mass)
    conditional_query_entropy = 0.0
    for class_index in range(classes):
        mass = float(class_mass[class_index])
        if mass <= 0.0:
            continue
        conditional_query_entropy += mass * _entropy_bits(
            joint[class_index] / class_mass[class_index]
        )
    global_information = _entropy_bits(class_mass)
    return (
        max(0.0, query_entropy - conditional_query_entropy),
        max(0.0, global_information),
    )


def _posterior_branches(
    torch: Any,
    log_weights: Any,
    class_ids: Any,
) -> tuple[tuple[float, Any], ...]:
    """Enumerate non-zero exact observation branches."""

    weights = _normalised_probabilities(torch, log_weights)
    inverse, classes = _partition_inverse(torch, class_ids)
    branches = []
    for class_index in range(classes):
        mask = inverse == class_index
        probability = float(weights[mask].sum())
        if probability <= 0.0:
            continue
        posterior = torch.full_like(log_weights, float("-inf"))
        posterior[mask] = log_weights[mask]
        posterior = _normalise_log_weights(torch, posterior)
        branches.append((probability, posterior))
    return tuple(branches)


def _expected_query_success_after_probe(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    class_ids: Any,
) -> float:
    return sum(
        probability
        * float(_door_marginals(torch, posterior, door_values).max())
        for probability, posterior in _posterior_branches(
            torch,
            log_weights,
            class_ids,
        )
    )


def _select_query_mi_candidate(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    candidate_class_ids: Any,
    available: Any,
) -> CandidateScore:
    """Select maximum exact I(query door; outcome), with public tie breaks."""

    current_success = float(
        _door_marginals(torch, log_weights, door_values).max()
    )
    scored = []
    for candidate_index in range(candidate_class_ids.shape[0]):
        if not bool(available[candidate_index].item()):
            continue
        query_bits, global_bits = _information_scores(
            torch,
            log_weights,
            door_values,
            candidate_class_ids[candidate_index],
        )
        expected_success = _expected_query_success_after_probe(
            torch,
            log_weights,
            door_values,
            candidate_class_ids[candidate_index],
        )
        scored.append(
            CandidateScore(
                candidate_index=candidate_index,
                expected_query_success=expected_success,
                expected_query_gain=max(
                    0.0,
                    expected_success - current_success,
                ),
                query_information_bits=query_bits,
                global_information_bits=global_bits,
                depth_two_value=None,
            )
        )
    if not scored:
        raise ValueError("at least one candidate must remain available")
    return min(
        scored,
        key=lambda item: (
            -item.query_information_bits,
            -item.expected_query_success,
            -item.global_information_bits,
            item.candidate_index,
        ),
    )


def _one_step_optimal_value(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    candidate_class_ids: Any,
    available: Any,
) -> float:
    indices = tuple(
        index
        for index in range(candidate_class_ids.shape[0])
        if bool(available[index].item())
    )
    if not indices:
        return float(_door_marginals(torch, log_weights, door_values).max())
    return max(
        _expected_query_success_after_probe(
            torch,
            log_weights,
            door_values,
            candidate_class_ids[index],
        )
        for index in indices
    )


def _select_depth_two_candidate(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    candidate_class_ids: Any,
    available: Any,
) -> CandidateScore:
    """Exact two-action dynamic program with observation-contingent recourse."""

    current_success = float(
        _door_marginals(torch, log_weights, door_values).max()
    )
    scored = []
    for candidate_index in range(candidate_class_ids.shape[0]):
        if not bool(available[candidate_index].item()):
            continue
        next_available = available.clone()
        next_available[candidate_index] = False
        depth_two_value = sum(
            branch_probability
            * _one_step_optimal_value(
                torch,
                posterior,
                door_values,
                candidate_class_ids,
                next_available,
            )
            for branch_probability, posterior in _posterior_branches(
                torch,
                log_weights,
                candidate_class_ids[candidate_index],
            )
        )
        expected_success = _expected_query_success_after_probe(
            torch,
            log_weights,
            door_values,
            candidate_class_ids[candidate_index],
        )
        query_bits, global_bits = _information_scores(
            torch,
            log_weights,
            door_values,
            candidate_class_ids[candidate_index],
        )
        scored.append(
            CandidateScore(
                candidate_index=candidate_index,
                expected_query_success=expected_success,
                expected_query_gain=max(
                    0.0,
                    expected_success - current_success,
                ),
                query_information_bits=query_bits,
                global_information_bits=global_bits,
                depth_two_value=depth_two_value,
            )
        )
    if not scored:
        raise ValueError("at least one candidate must remain available")
    return min(
        scored,
        key=lambda item: (
            -float(item.depth_two_value),
            -item.query_information_bits,
            -item.expected_query_success,
            -item.global_information_bits,
            item.candidate_index,
        ),
    )


def _common_score(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    candidate_class_ids: Any,
    candidate_index: int,
) -> CandidateScore:
    current_success = float(
        _door_marginals(torch, log_weights, door_values).max()
    )
    expected_success = _expected_query_success_after_probe(
        torch,
        log_weights,
        door_values,
        candidate_class_ids[candidate_index],
    )
    query_bits, global_bits = _information_scores(
        torch,
        log_weights,
        door_values,
        candidate_class_ids[candidate_index],
    )
    return CandidateScore(
        candidate_index=candidate_index,
        expected_query_success=expected_success,
        expected_query_gain=max(0.0, expected_success - current_success),
        query_information_bits=query_bits,
        global_information_bits=global_bits,
        depth_two_value=None,
    )


def _select_candidate(
    torch: Any,
    *,
    policy: str,
    log_weights: Any,
    door_values: Any,
    candidate_class_ids: Any,
    available: Any,
    uniform_order: Sequence[int],
    step: int,
) -> CandidateScore:
    if policy == "query-success-greedy":
        # The singleton spatial dimensions adapt exact partition class IDs to
        # the existing selector's [candidate, hypothesis, height, width] API.
        existing = _select_acquisition_candidate(
            torch,
            log_weights,
            door_values,
            candidate_class_ids[..., None, None],
            available,
        )
        result = _common_score(
            torch,
            log_weights,
            door_values,
            candidate_class_ids,
            existing.candidate_index,
        )
        if not math.isclose(
            result.expected_query_gain,
            existing.expected_door_gain,
            abs_tol=1e-6,
        ):
            raise AssertionError("existing query-success score changed")
        return result
    if policy == "query-mi":
        return _select_query_mi_candidate(
            torch,
            log_weights,
            door_values,
            candidate_class_ids,
            available,
        )
    if policy == "global-mi":
        existing = _select_global_information_candidate(
            torch,
            log_weights,
            candidate_class_ids[..., None, None],
            available,
        )
        return _common_score(
            torch,
            log_weights,
            door_values,
            candidate_class_ids,
            existing.candidate_index,
        )
    if policy == "depth-2-dp":
        return _select_depth_two_candidate(
            torch,
            log_weights,
            door_values,
            candidate_class_ids,
            available,
        )
    if policy == "uniform":
        candidate_index = int(uniform_order[step - 1])
        if not bool(available[candidate_index].item()):
            raise AssertionError("uniform order repeated a candidate")
        return _common_score(
            torch,
            log_weights,
            door_values,
            candidate_class_ids,
            candidate_index,
        )
    raise ValueError(f"unknown policy: {policy}")


def _factor_programs() -> tuple[Any, ...]:
    from prp_wm.rulegrid import (
        ALL_COLLISIONS,
        ALL_RELATIONS,
        ALL_TRIGGERS,
        RuleProgram,
    )

    return tuple(
        RuleProgram(
            ALL_COLLISIONS[collision],
            ALL_TRIGGERS[trigger],
            ALL_RELATIONS[relation],
        )
        for collision, trigger, relation in itertools.product(
            range(FACTOR_CARDINALITY),
            repeat=FACTOR_AXES,
        )
    )


def _candidate_menu(
    *,
    palette: Any,
    query_axis_indices: tuple[int, int],
    order_seed: int,
) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[int | None, ...]]:
    from prp_wm.rulegrid import ALL_AXES, RuleGridProbe, _neutral_probe

    nuisance_axis_index = next(
        index
        for index in range(FACTOR_AXES)
        if index not in query_axis_indices
    )
    category_by_axis = {
        query_axis_indices[0]: "relevant-axis-0",
        query_axis_indices[1]: "relevant-axis-1",
        nuisance_axis_index: "nuisance-axis",
    }
    rows: list[tuple[str, int | None, Any]] = []
    for axis_index, axis in enumerate(ALL_AXES):
        for variant in range(2):
            rows.append(
                (
                    category_by_axis[axis_index],
                    axis_index,
                    _atomic_probe(
                        axis,
                        f"private-{axis.value}-{variant}",
                        palette,
                        variant=variant,
                    ),
                )
            )
    rows.extend(
        (
            (
                "neutral",
                None,
                _neutral_probe("private-neutral-0", palette, row=2, col=3),
            ),
            (
                "neutral",
                None,
                _neutral_probe("private-neutral-1", palette, row=4, col=3),
            ),
        )
    )
    if len(rows) != CANDIDATES_PER_SCENARIO:
        raise AssertionError("two-axis menu must contain exactly eight probes")
    random.Random(order_seed).shuffle(rows)
    probes = tuple(
        RuleGridProbe(f"C{index:02d}", probe.state, probe.action)
        for index, (_, _, probe) in enumerate(rows)
    )
    return (
        probes,
        tuple(category for category, _, _ in rows),
        tuple(axis_index for _, axis_index, _ in rows),
    )


def _simulator_partitions(
    torch: Any,
    *,
    probes: Sequence[Any],
    programs: Sequence[Any],
    palette: Any,
    candidate_axis_indices: Sequence[int | None],
) -> tuple[Any, dict[str, object]]:
    """Construct class IDs from real RuleGrid grids and verify locality."""

    from prp_wm.rulegrid import simulate

    raw = torch.tensor(
        [
            [
                simulate(probe.state, probe.action, program, palette)
                for program in programs
            ]
            for probe in probes
        ],
        dtype=torch.long,
    )
    factor_bank = torch.cartesian_prod(
        *(torch.arange(FACTOR_CARDINALITY) for _ in range(FACTOR_AXES))
    )
    class_rows = []
    candidate_records = []
    passed = True
    for candidate_index, axis_index in enumerate(candidate_axis_indices):
        flattened = raw[candidate_index].flatten(start_dim=1)
        _, inverse = torch.unique(
            flattened,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        classes = int(inverse.max().item()) + 1
        if axis_index is None:
            class_count_expected = classes == 1
            axis_local = True
        else:
            class_count_expected = classes == FACTOR_CARDINALITY
            axis_local = all(
                int(torch.unique(inverse[factor_bank[:, axis_index] == value]).numel())
                == 1
                for value in range(FACTOR_CARDINALITY)
            )
        candidate_passed = class_count_expected and axis_local
        passed = passed and candidate_passed
        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "axis_index": axis_index,
                "outcome_classes": classes,
                "expected_class_count": (
                    1 if axis_index is None else FACTOR_CARDINALITY
                ),
                "outcome_depends_only_on_target_axis": axis_local,
                "passed": candidate_passed,
            }
        )
        class_rows.append(inverse)
    result = torch.stack(class_rows, dim=0)
    return result, {
        "constructed_from_rulegrid_simulator_grids": True,
        "factor_bank_size": int(raw.shape[1]),
        "grid_shape": list(raw.shape[-2:]),
        "candidates": candidate_records,
        "passed": passed,
    }


def _build_scenario(
    torch: Any,
    *,
    query_axis_indices: tuple[int, int],
    group_index: int,
    split: str,
    master_seed: int,
    policy_seed: int,
) -> TwoAxisScenario:
    from prp_wm.rulegrid import ALL_AXES, palette_from_seed

    if (
        len(set(query_axis_indices)) != 2
        or any(index < 0 or index >= FACTOR_AXES for index in query_axis_indices)
    ):
        raise ValueError("query axes must be two distinct factor indices")
    pair_name = "-".join(ALL_AXES[index].value for index in query_axis_indices)
    scenario_seed = _seed64(
        RESULT_SCHEMA_VERSION,
        split,
        master_seed,
        pair_name,
        group_index,
    )
    palette = palette_from_seed(_seed64(scenario_seed, "palette"))
    value_pairs = tuple(itertools.combinations(range(FACTOR_CARDINALITY), 2))
    allowed = (
        value_pairs[
            _seed64(scenario_seed, "allowed", query_axis_indices[0])
            % len(value_pairs)
        ],
        value_pairs[
            _seed64(scenario_seed, "allowed", query_axis_indices[1])
            % len(value_pairs)
        ],
    )
    probes, categories, candidate_axes = _candidate_menu(
        palette=palette,
        query_axis_indices=query_axis_indices,
        order_seed=_seed64(scenario_seed, "candidate-order", policy_seed),
    )
    factor_bank = torch.cartesian_prod(
        *(torch.arange(FACTOR_CARDINALITY) for _ in range(FACTOR_AXES))
    )
    programs = _factor_programs()
    candidate_classes, validation = _simulator_partitions(
        torch,
        probes=probes,
        programs=programs,
        palette=palette,
        candidate_axis_indices=candidate_axes,
    )
    if not validation["passed"]:
        raise AssertionError("RuleGrid atomic-partition validation failed")
    allowed_mask = torch.ones(FACTOR_BANK_SIZE, dtype=torch.bool)
    for pair_position, axis_index in enumerate(query_axis_indices):
        values = torch.tensor(allowed[pair_position], dtype=torch.long)
        allowed_mask &= (factor_bank[:, axis_index, None] == values).any(dim=1)
    hypothesis_indices = torch.nonzero(
        allowed_mask,
        as_tuple=False,
    ).flatten()
    if int(hypothesis_indices.numel()) != INITIAL_HYPOTHESES:
        raise AssertionError("two-axis restriction must leave 16 hypotheses")
    local_value_maps = []
    for pair_position in range(2):
        mapping = torch.zeros(FACTOR_CARDINALITY, dtype=torch.long)
        mapping[allowed[pair_position][0]] = 0
        mapping[allowed[pair_position][1]] = 1
        local_value_maps.append(mapping)
    door_values = (
        2 * local_value_maps[0][factor_bank[:, query_axis_indices[0]]]
        + local_value_maps[1][factor_bank[:, query_axis_indices[1]]]
    )
    door_counts = torch.bincount(
        door_values[hypothesis_indices],
        minlength=DOORS,
    )
    if tuple(int(value) for value in door_counts) != (4, 4, 4, 4):
        raise AssertionError("the 16 hypotheses must balance four doors")
    uniform_generator = torch.Generator(device="cpu")
    uniform_generator.manual_seed(
        _seed64(scenario_seed, "uniform-order", policy_seed)
        % (2**63 - 1)
    )
    uniform_order = tuple(
        int(value)
        for value in torch.randperm(
            CANDIDATES_PER_SCENARIO,
            generator=uniform_generator,
        ).tolist()
    )
    nuisance_axis_index = next(
        index
        for index in range(FACTOR_AXES)
        if index not in query_axis_indices
    )
    return TwoAxisScenario(
        scenario_id=(
            f"{split}/S{policy_seed}/Q{pair_name}/G{group_index:04d}"
        ),
        query_axis_indices=query_axis_indices,
        query_axis_names=tuple(
            ALL_AXES[index].value for index in query_axis_indices
        ),
        nuisance_axis_index=nuisance_axis_index,
        nuisance_axis_name=ALL_AXES[nuisance_axis_index].value,
        allowed_query_values=allowed,
        factor_bank=factor_bank,
        hypothesis_indices=hypothesis_indices,
        door_values=door_values,
        candidate_class_ids=candidate_classes,
        candidate_categories=categories,
        candidate_axis_indices=candidate_axes,
        candidate_probe_ids=tuple(probe.probe_id for probe in probes),
        partition_validation=validation,
        uniform_order=uniform_order,
    )


def _initial_log_weights(torch: Any, scenario: TwoAxisScenario) -> Any:
    result = torch.full(
        (FACTOR_BANK_SIZE,),
        float("-inf"),
        dtype=torch.float32,
    )
    result[scenario.hypothesis_indices] = -math.log(INITIAL_HYPOTHESES)
    return result


def _posterior_update(
    torch: Any,
    log_weights: Any,
    candidate_class_ids: Any,
    observed_class: int,
) -> Any:
    mask = candidate_class_ids == observed_class
    result = torch.full_like(log_weights, float("-inf"))
    result[mask] = log_weights[mask]
    if not bool(torch.isfinite(result).any().item()):
        raise AssertionError("exact simulator feedback eliminated the truth")
    return _normalise_log_weights(torch, result)


def _complementary_relevant_axes(
    scenario: TwoAxisScenario,
    selected_indices: Sequence[int],
) -> bool:
    if len(selected_indices) < 2:
        return False
    selected_axes = {
        scenario.candidate_axis_indices[index]
        for index in selected_indices[:2]
    }
    return selected_axes == set(scenario.query_axis_indices)


def _rollout_one(
    torch: Any,
    *,
    scenario: TwoAxisScenario,
    truth_index: int,
    policy: str,
    budgets: Sequence[int],
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    log_weights = _initial_log_weights(torch, scenario)
    initial_joint_entropy_bits = _entropy_bits(log_weights.exp())
    available = torch.ones(CANDIDATES_PER_SCENARIO, dtype=torch.bool)
    selected_indices: list[int] = []
    selected_scores: list[CandidateScore] = []
    first_identified_step: int | None = None
    snapshots: dict[int, dict[str, object]] = {}
    target_door = int(scenario.door_values[truth_index])
    task_id = f"{scenario.scenario_id}/H{truth_index:02d}"

    def snapshot(budget: int) -> None:
        probabilities = _normalised_probabilities(torch, log_weights)
        door_marginals = _door_marginals(
            torch,
            log_weights,
            scenario.door_values,
        )
        query_entropy_bits = _entropy_bits(door_marginals)
        joint_entropy_bits = _entropy_bits(probabilities)
        chosen_door = int(torch.argmax(door_marginals))
        snapshots[budget] = {
            "task_id": task_id,
            "scenario_id": scenario.scenario_id,
            "truth_factor_index": truth_index,
            "target_door": target_door,
            "chosen_door": chosen_door,
            "won": chosen_door == target_door,
            "optimal_query_success_probability": float(
                door_marginals.max()
            ),
            "true_query_probability": float(door_marginals[target_door]),
            "query_entropy_bits": query_entropy_bits,
            "joint_entropy_bits": joint_entropy_bits,
            "global_information_acquired_bits": max(
                0.0,
                initial_joint_entropy_bits - joint_entropy_bits,
            ),
            "posterior_effective_hypotheses": 2.0**joint_entropy_bits,
            "query_identified": query_entropy_bits < 1e-7,
            "first_query_identification_step": first_identified_step,
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
        }

    if 0 in budgets:
        snapshot(0)
    for step in range(1, max(budgets) + 1):
        score = _select_candidate(
            torch,
            policy=policy,
            log_weights=log_weights,
            door_values=scenario.door_values,
            candidate_class_ids=scenario.candidate_class_ids,
            available=available,
            uniform_order=scenario.uniform_order,
            step=step,
        )
        candidate_index = score.candidate_index
        available[candidate_index] = False
        selected_indices.append(candidate_index)
        selected_scores.append(score)
        observed_class = int(
            scenario.candidate_class_ids[candidate_index, truth_index]
        )
        log_weights = _posterior_update(
            torch,
            log_weights,
            scenario.candidate_class_ids[candidate_index],
            observed_class,
        )
        if (
            first_identified_step is None
            and _entropy_bits(
                _door_marginals(
                    torch,
                    log_weights,
                    scenario.door_values,
                )
            )
            < 1e-7
        ):
            first_identified_step = step
        if step in budgets:
            snapshot(step)
    trace = {
        "task_id": task_id,
        "policy": policy,
        "selected_candidate_indices": selected_indices,
        "selected_candidate_categories_audit": [
            scenario.candidate_categories[index]
            for index in selected_indices
        ],
        "step_scores": [
            {
                "candidate_index": score.candidate_index,
                "expected_query_success": score.expected_query_success,
                "expected_query_gain": score.expected_query_gain,
                "query_information_bits": score.query_information_bits,
                "global_information_bits": score.global_information_bits,
                "depth_two_value": score.depth_two_value,
            }
            for score in selected_scores
        ],
    }
    return snapshots, trace


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _summarise_policy(
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
    return {
        "tasks": len(records),
        "terminal_win_rate": _mean(
            int(bool(record["won"])) for record in records
        ),
        "mean_optimal_query_success_probability": _mean(
            float(record["optimal_query_success_probability"])
            for record in records
        ),
        "mean_true_query_probability": _mean(
            float(record["true_query_probability"]) for record in records
        ),
        "query_identified_rate": _mean(
            int(bool(record["query_identified"])) for record in records
        ),
        "mean_query_entropy_bits": _mean(
            float(record["query_entropy_bits"]) for record in records
        ),
        "mean_joint_entropy_bits": _mean(
            float(record["joint_entropy_bits"]) for record in records
        ),
        "mean_global_information_acquired_bits": _mean(
            float(record["global_information_acquired_bits"])
            for record in records
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
        "mean_selected_category_counts_audit": {
            category: _mean(
                record["selected_candidate_categories_audit"].count(category)
                for record in records
            )
            for category in categories
        },
    }


def _paired_summary(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("paired policies must have the same number of tasks")
    for left_record, right_record in zip(left, right, strict=True):
        if left_record["task_id"] != right_record["task_id"]:
            raise ValueError("paired policy task order changed")
    left_wins = tuple(bool(record["won"]) for record in left)
    right_wins = tuple(bool(record["won"]) for record in right)
    return {
        "tasks": len(left),
        "terminal_win_rate_delta_left_minus_right": (
            _mean(left_wins) - _mean(right_wins)
        ),
        "optimal_query_success_delta_left_minus_right": _mean(
            float(left_record["optimal_query_success_probability"])
            - float(right_record["optimal_query_success_probability"])
            for left_record, right_record in zip(left, right, strict=True)
        ),
        "query_entropy_bits_delta_left_minus_right": _mean(
            float(left_record["query_entropy_bits"])
            - float(right_record["query_entropy_bits"])
            for left_record, right_record in zip(left, right, strict=True)
        ),
        "global_information_bits_delta_left_minus_right": _mean(
            float(left_record["global_information_acquired_bits"])
            - float(right_record["global_information_acquired_bits"])
            for left_record, right_record in zip(left, right, strict=True)
        ),
        "left_only_wins": sum(
            left_win and not right_win
            for left_win, right_win in zip(left_wins, right_wins, strict=True)
        ),
        "right_only_wins": sum(
            right_win and not left_win
            for left_win, right_win in zip(left_wins, right_wins, strict=True)
        ),
        "both_win": sum(
            left_win and right_win
            for left_win, right_win in zip(left_wins, right_wins, strict=True)
        ),
        "neither_wins": sum(
            not left_win and not right_win
            for left_win, right_win in zip(left_wins, right_wins, strict=True)
        ),
    }


def _summarise_records(
    records: dict[str, dict[int, list[dict[str, object]]]],
    budgets: Sequence[int],
) -> dict[str, object]:
    result = {}
    for budget in budgets:
        result[str(budget)] = {
            "policies": {
                policy: _summarise_policy(
                    records[policy][budget],
                    budget=budget,
                )
                for policy in POLICIES
            },
            "paired": {
                f"{left}_minus_{right}": _paired_summary(
                    records[left][budget],
                    records[right][budget],
                )
                for left, right in itertools.combinations(POLICIES, 2)
            },
        }
    return result


def _exact_control(
    summary: dict[str, object],
) -> dict[str, object]:
    expected = {"0": 0.25, "1": 0.5, "2": 1.0}
    observed: dict[str, object] = {}
    passed = True
    for policy in QUERY_AWARE_POLICIES:
        policy_values = {
            budget: float(
                summary[budget]["policies"][policy][
                    "mean_optimal_query_success_probability"
                ]
            )
            for budget in expected
        }
        terminal_values = {
            budget: float(
                summary[budget]["policies"][policy]["terminal_win_rate"]
            )
            for budget in expected
        }
        checks = {
            budget: (
                math.isclose(
                    policy_values[budget],
                    expected[budget],
                    abs_tol=1e-7,
                )
                and math.isclose(
                    terminal_values[budget],
                    expected[budget],
                    abs_tol=1e-7,
                )
            )
            for budget in expected
        }
        complement_rate = float(
            summary["2"]["policies"][policy][
                "b2_complementary_relevant_axes_rate"
            ]
        )
        checks["b2_complementary_relevant_axes_rate_is_one"] = math.isclose(
            complement_rate,
            1.0,
            abs_tol=1e-7,
        )
        policy_passed = all(checks.values())
        passed = passed and policy_passed
        observed[policy] = {
            "mean_optimal_query_success_probability": policy_values,
            "terminal_win_rate": terminal_values,
            "b2_complementary_relevant_axes_rate": complement_rate,
            "checks": checks,
            "passed": policy_passed,
        }
    return {
        "expected_balanced_compositional_query_success": expected,
        "interpretation": (
            "Before probing there are four equiprobable doors; one exact "
            "relevant-axis probe leaves two; complementary relevant-axis "
            "probes identify the Cartesian-product door."
        ),
        "observed_query_aware_policies": observed,
        "passed": passed,
    }


def run_experiment(
    *,
    torch: Any,
    groups_per_pair: int,
    seeds: Sequence[int],
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    split: str = "two-axis-compositional-acquisition",
    data_master_seed: int = 2026072401,
    trace_scenarios: int = 3,
) -> dict[str, object]:
    from prp_wm.rulegrid import ALL_AXES

    budgets = tuple(sorted(int(value) for value in budgets))
    seeds = tuple(int(value) for value in seeds)
    query_pairs = tuple(itertools.combinations(range(FACTOR_AXES), 2))
    records = {
        policy: {budget: [] for budget in budgets}
        for policy in POLICIES
    }
    by_pair_records = {
        "-".join(ALL_AXES[index].value for index in pair): {
            policy: {budget: [] for budget in budgets}
            for policy in POLICIES
        }
        for pair in query_pairs
    }
    traces: list[dict[str, object]] = []
    scenario_audits: list[dict[str, object]] = []
    scenario_count = 0
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
                scenario_count += 1
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
                        "candidate_probe_ids": list(
                            scenario.candidate_probe_ids
                        ),
                        "uniform_order": list(scenario.uniform_order),
                        "partition_validation": (
                            scenario.partition_validation
                        ),
                    }
                )
                for policy in POLICIES:
                    for truth_index in scenario.hypothesis_indices.tolist():
                        snapshots, trace = _rollout_one(
                            torch,
                            scenario=scenario,
                            truth_index=int(truth_index),
                            policy=policy,
                            budgets=budgets,
                        )
                        for budget, record in snapshots.items():
                            records[policy][budget].append(record)
                            by_pair_records[pair_name][policy][budget].append(
                                record
                            )
                        if (
                            len(traces)
                            < trace_scenarios
                            * len(POLICIES)
                            * INITIAL_HYPOTHESES
                        ):
                            traces.append(trace)
    summary = _summarise_records(records, budgets)
    exact_control = _exact_control(summary)
    if not exact_control["passed"]:
        raise AssertionError("balanced two-axis exact control failed")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_kind": "privileged-symbolic-acquisition-control",
        "protocol": {
            "factor_bank": "4x4x4=64 exact RuleGrid programs",
            "query": (
                "two axes restricted to two values each; their Cartesian "
                "product maps to four equiprobable doors"
            ),
            "initial_hypotheses": INITIAL_HYPOTHESES,
            "candidate_menu": (
                "two exact atomic probes per axis plus two neutral probes; "
                "public candidate order shuffled per scenario"
            ),
            "candidate_count": CANDIDATES_PER_SCENARIO,
            "budgets": list(budgets),
            "policies": list(POLICIES),
            "depth_two_policy": (
                "exact observation-contingent two-action dynamic program, "
                "receding horizon at each acquisition step"
            ),
            "posterior_update": (
                "exact filtering by RuleGrid simulator outcome class ID"
            ),
            "paired_protocol": (
                "all policies share query, restrictions, palette, candidate "
                "menu/order, truth programs, feedback, and budgets"
            ),
            "privileged_assumptions": [
                "known three-axis four-value factor codebook",
                "known query axes and two-value restrictions",
                "exact RuleGrid outcome partitions",
                "exact Bayesian filtering",
            ],
        },
        "configuration": {
            "groups_per_pair": groups_per_pair,
            "query_axis_pairs": [
                [ALL_AXES[index].value for index in pair]
                for pair in query_pairs
            ],
            "seeds": list(seeds),
            "data_master_seed": data_master_seed,
            "split": split,
            "scenarios": scenario_count,
            "tasks_per_policy": (
                scenario_count * INITIAL_HYPOTHESES
            ),
        },
        "partition_validation": {
            "scenarios": len(scenario_audits),
            "all_passed": all(
                bool(record["partition_validation"]["passed"])
                for record in scenario_audits
            ),
            "expected_atomic_classes": FACTOR_CARDINALITY,
            "expected_neutral_classes": 1,
            "scenario_audits": scenario_audits,
        },
        "exact_control": exact_control,
        "aggregate": summary,
        "by_query_axis_pair": {
            pair_name: _summarise_records(pair_records, budgets)
            for pair_name, pair_records in by_pair_records.items()
        },
        "traces": traces,
    }


def main() -> None:
    args = parse_args()
    budgets, seeds = _validate_args(args)
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise SystemExit("PyTorch is required for this experiment") from error
    result = run_experiment(
        torch=torch,
        groups_per_pair=args.groups_per_pair,
        seeds=seeds,
        budgets=budgets,
        split=args.split,
        data_master_seed=args.data_master_seed,
        trace_scenarios=args.trace_scenarios,
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scenarios": result["configuration"]["scenarios"],
                "tasks_per_policy": result["configuration"][
                    "tasks_per_policy"
                ],
                "exact_control_passed": result["exact_control"]["passed"],
                "budget_2": result["aggregate"]["2"]["policies"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
