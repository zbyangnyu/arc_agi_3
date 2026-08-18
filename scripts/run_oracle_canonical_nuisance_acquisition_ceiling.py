#!/usr/bin/env python3
"""Privileged nuisance-rule ceiling for query-conditioned acquisition.

This controlled experiment asks whether a controller that values information
for the current door query behaves differently from a controller that values
all predictable information equally.  Every episode starts with a marked
partial no-op observation that leaves three query values and all four values
of both nuisance factors possible (48 joint codes).  Its public candidate
menu contains:

* two strong atomic probes of the queried factor;
* two strong atomic probes for each irrelevant factor; and
* two rule-independent neutral probes.

The three paired policies are query-conditioned expected door gain, global
exact-partition information gain, and uniform sampling without replacement.
They share tasks, initial posterior, exact outcome partitions, environment
feedback, and budgets.

This remains a privileged symbolic ceiling.  It uses a known 3x4 factor
codebook, an explicit query axis, oracle palettes, exact simulator partitions,
and exact filtering.  A learned public-belief checkpoint is evaluated only as
a coverage audit and never initializes the primary controller.  Candidate
kinds and simulator targets are audit-only sidecars and are never passed to
either acquisition selector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
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
    DEFAULT_BELIEF_CHECKPOINT,
    PublicDoorQuery,
    _acquisition_score,
    _door_marginals,
    _finalise_task_records,
    _normalise_log_weights,
    _symbolic_door_identified,
    _task_metrics,
    _select_acquisition_candidate,
    bayesian_log_likelihood_update,
    factor_marginals_to_joint_log_weights,
)


RESULT_SCHEMA_VERSION = (
    "prp-wm.oracle-canonical-nuisance-acquisition-ceiling.v1"
)
DEFAULT_BUDGETS = (0, 1, 2, 3, 4, 8)
DEFAULT_SEEDS = (20260873, 20260874, 20260875)
POLICIES = (
    "query-conditioned",
    "global-information-gain",
    "uniform",
)
CANDIDATES_PER_TASK = 8
PROGRAMS_PER_GROUP = 3
MAX_BUDGET = 8
_AUDITED_SOURCE_FILES = (
    "prp_wm/latent_rules.py",
    "prp_wm/public_version_k4.py",
    "prp_wm/rulegrid.py",
    "scripts/run_gram_public_coverage_finetune.py",
    "scripts/run_oracle_canonical_acquisition_ceiling.py",
    "scripts/run_oracle_canonical_nuisance_acquisition_ceiling.py",
    "scripts/run_public_version_space_k4.py",
)


@dataclass(frozen=True)
class GlobalInformationScore:
    """One deterministic MAP-outcome partition score."""

    candidate_index: int
    information_gain_nats: float
    predicted_outcome_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--belief-checkpoint",
        type=Path,
        default=DEFAULT_BELIEF_CHECKPOINT,
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
    parser.add_argument("--trace-tasks-per-query", type=int, default=2)
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
    if args.trace_tasks_per_query < 0:
        raise SystemExit("--trace-tasks-per-query must be non-negative")
    seeds = tuple(int(value) for value in args.seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(value < 0 for value in seeds)
        or args.data_master_seed < 0
    ):
        raise SystemExit("seeds must be non-negative")
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
    return tuple(sorted(budgets)), seeds


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    return {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in _AUDITED_SOURCE_FILES
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _global_information_score(
    torch: Any,
    log_weights: Any,
    predicted_outcomes: Any,
    *,
    candidate_index: int,
) -> GlobalInformationScore:
    """Compute deterministic-outcome mutual information for one candidate."""

    if log_weights.ndim != 1:
        raise ValueError("one posterior row is required")
    if predicted_outcomes.ndim != 3:
        raise ValueError("candidate outcomes must have [H,H,W] shape")
    if predicted_outcomes.shape[0] != log_weights.shape[0]:
        raise ValueError("candidate outcomes must cover every hypothesis")
    weights = _normalise_log_weights(torch, log_weights).exp()
    flattened = predicted_outcomes.reshape(predicted_outcomes.shape[0], -1)
    _, inverse = torch.unique(
        flattened,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    classes = int(inverse.max().item()) + 1
    class_mass = torch.zeros(
        classes,
        dtype=weights.dtype,
        device=weights.device,
    )
    class_mass.scatter_add_(0, inverse, weights)
    nonzero = class_mass > 0
    information_gain = -(
        class_mass[nonzero] * class_mass[nonzero].log()
    ).sum()
    return GlobalInformationScore(
        candidate_index=candidate_index,
        information_gain_nats=float(information_gain.clamp_min(0.0)),
        predicted_outcome_classes=classes,
    )


def _select_query_conditioned_candidate(
    torch: Any,
    log_weights: Any,
    query_values: Any,
    candidate_outcomes: Any,
    available: Any,
) -> Any:
    """Select expected query-door gain without any task metadata."""

    return _select_acquisition_candidate(
        torch,
        log_weights,
        query_values,
        candidate_outcomes,
        available,
    )


def _select_global_information_candidate(
    torch: Any,
    log_weights: Any,
    candidate_outcomes: Any,
    available: Any,
) -> GlobalInformationScore:
    """Select global partition information without receiving the query."""

    if candidate_outcomes.ndim != 4:
        raise ValueError("candidate outcomes must have [P,H,H,W] shape")
    if (
        available.shape != candidate_outcomes.shape[:1]
        or available.dtype != torch.bool
    ):
        raise ValueError("available must be a boolean candidate mask")
    scores = tuple(
        _global_information_score(
            torch,
            log_weights,
            candidate_outcomes[candidate_index],
            candidate_index=candidate_index,
        )
        for candidate_index in range(candidate_outcomes.shape[0])
        if bool(available[candidate_index].item())
    )
    if not scores:
        raise ValueError("at least one candidate must remain available")
    return min(
        scores,
        key=lambda item: (
            -item.information_gain_nats,
            item.candidate_index,
        ),
    )


def _selector_boundary_audit() -> dict[str, object]:
    """Record selector inputs and reject common privileged field names."""

    selectors = {
        "query-conditioned": (
            _select_query_conditioned_candidate,
            (
                "torch",
                "log_weights",
                "query_values",
                "candidate_outcomes",
                "available",
            ),
        ),
        "global-information-gain": (
            _select_global_information_candidate,
            (
                "torch",
                "log_weights",
                "candidate_outcomes",
                "available",
            ),
        ),
    }
    forbidden = (
        "task_id",
        "probe_id",
        "candidate_kind",
        "active_target",
        "true_program",
        "feedback",
    )
    records: dict[str, object] = {}
    passed = True
    for name, (selector, expected) in selectors.items():
        parameters = tuple(inspect.signature(selector).parameters)
        source = inspect.getsource(selector)
        checks = {
            "expected_signature": parameters == expected,
            "forbidden_names_absent": all(
                forbidden_name not in source
                for forbidden_name in forbidden
            ),
        }
        records[name] = {
            "parameters": list(parameters),
            "checks": checks,
            "passed": all(checks.values()),
        }
        passed = passed and all(checks.values())
    return {
        "forbidden_names": list(forbidden),
        "selectors": records,
        "passed": passed,
    }


def _atomic_probe(axis: Any, probe_id: str, palette: Any, *, variant: int) -> Any:
    from prp_wm.rulegrid import (
        Axis,
        Direction,
        _collision_probe,
        _relation_probe,
        _trigger_probe,
    )

    if variant not in (0, 1):
        raise ValueError("atomic probe variant must be zero or one")
    if variant == 0:
        if axis is Axis.COLLISION:
            return _collision_probe(probe_id, palette, row=1, col=1)
        if axis is Axis.TRIGGER:
            return _trigger_probe(probe_id, palette, row=3, col=1)
        if axis is Axis.RELATION:
            return _relation_probe(probe_id, palette, row=5, col=1)
    else:
        if axis is Axis.COLLISION:
            return _collision_probe(
                probe_id,
                palette,
                row=6,
                col=5,
                direction=Direction.WEST,
            )
        if axis is Axis.TRIGGER:
            return _trigger_probe(probe_id, palette, row=6, col=2)
        if axis is Axis.RELATION:
            return _relation_probe(
                probe_id,
                palette,
                row=6,
                col=5,
                direction=Direction.WEST,
            )
    raise ValueError(f"unknown RuleGrid axis: {axis!r}")


def _candidate_menu(
    *,
    query_axis: Any,
    palette: Any,
    order_seed: int,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Build and nuisance-shuffle an eight-probe public candidate menu."""

    from prp_wm.rulegrid import (
        ALL_AXES,
        RuleGridProbe,
        _neutral_probe,
    )

    irrelevant_axes = tuple(axis for axis in ALL_AXES if axis is not query_axis)
    if len(irrelevant_axes) != 2:
        raise AssertionError("a RuleGrid query must have two irrelevant axes")
    rows: list[tuple[str, Any]] = [
        (
            "query-atomic",
            _atomic_probe(query_axis, "private-query-0", palette, variant=0),
        ),
        (
            "query-atomic",
            _atomic_probe(query_axis, "private-query-1", palette, variant=1),
        ),
    ]
    for nuisance_index, axis in enumerate(irrelevant_axes):
        rows.extend(
            (
                (
                    f"nuisance-axis-{nuisance_index}",
                    _atomic_probe(
                        axis,
                        f"private-nuisance-{nuisance_index}-0",
                        palette,
                        variant=0,
                    ),
                ),
                (
                    f"nuisance-axis-{nuisance_index}",
                    _atomic_probe(
                        axis,
                        f"private-nuisance-{nuisance_index}-1",
                        palette,
                        variant=1,
                    ),
                ),
            ),
        )
    rows.extend(
        (
        (
            "neutral",
            _neutral_probe("private-neutral-0", palette, row=2, col=3),
        ),
        (
            "neutral",
            _neutral_probe("private-neutral-1", palette, row=4, col=3),
        ),
        )
    )
    if len(rows) != CANDIDATES_PER_TASK:
        raise AssertionError("atomic nuisance menu must contain eight probes")
    random.Random(order_seed).shuffle(rows)
    # Public IDs are deliberately generic after shuffling.  They are retained
    # by the environment serialization but neither selector receives them.
    candidates = tuple(
        RuleGridProbe(f"C{index:02d}", probe.state, probe.action)
        for index, (_, probe) in enumerate(rows)
    )
    kinds = tuple(kind for kind, _ in rows)
    return candidates, kinds


def _neutral_support(program: Any, palette: Any) -> tuple[Any, ...]:
    """Four rule-independent observations, leaving all factors unobserved."""

    from prp_wm.rulegrid import RuleGridTransition, _neutral_probe, simulate

    positions = ((1, 1), (1, 5), (4, 1), (4, 5))
    probes = tuple(
        _neutral_probe(f"S{index:02d}", palette, row=row, col=col)
        for index, (row, col) in enumerate(positions)
    )
    return tuple(
        RuleGridTransition(
            probe.state,
            probe.action,
            simulate(probe.state, probe.action, program, palette),
        )
        for probe in probes
    )


def _marked_query_partial_support(
    program: Any,
    palette: Any,
    query_axis: Any,
) -> tuple[Any, ...]:
    """Four neutral observations plus a no-op 1+3 query partition."""

    from prp_wm.rulegrid import (
        RuleGridTransition,
        _partial_probe,
        simulate,
    )

    neutral = _neutral_support(program, palette)
    partial = _partial_probe(
        query_axis,
        "S04",
        palette,
        variant=0,
    )
    target = simulate(partial.state, partial.action, program, palette)
    if target != partial.state:
        raise ValueError("selected hidden query mode is not in the 3-way no-op set")
    return neutral + (
        RuleGridTransition(partial.state, partial.action, target),
    )


def _build_nuisance_tasks(
    query_axis: Any,
    *,
    groups: int,
    split: str,
    master_seed: int,
    candidate_seed: int,
) -> tuple[Any, ...]:
    """Build three no-op query values per palette with shared public data."""

    from prp_wm.rulegrid import (
        ALL_AXES,
        ALL_COLLISIONS,
        ALL_RELATIONS,
        ALL_TRIGGERS,
        Axis,
        Collision,
        Relation,
        RuleProgram,
        RuleGridInferenceView,
        RuleGridPrivilegedTargets,
        RuleGridTask,
        Trigger,
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
        candidates, kinds = _candidate_menu(
            query_axis=query_axis,
            palette=palette,
            order_seed=derive_seed64(
                split,
                query_axis,
                group_index,
                "candidate_order",
                master_seed=master_seed,
            ) ^ candidate_seed,
        )
        modes = (ALL_COLLISIONS, ALL_TRIGGERS, ALL_RELATIONS)
        query_index = ALL_AXES.index(query_axis)
        factor_ids = [
            group_index % 4,
            (group_index // 4) % 4,
            (group_index // 16) % 4,
        ]
        singled_out = {
            Axis.COLLISION: ALL_COLLISIONS.index(Collision.BOUNCE),
            Axis.TRIGGER: ALL_TRIGGERS.index(Trigger.TOGGLE),
            Axis.RELATION: ALL_RELATIONS.index(Relation.SWAP),
        }[query_axis]
        programs = []
        for query_value in (
            value for value in range(4) if value != singled_out
        ):
            selected = list(factor_ids)
            selected[query_index] = query_value
            programs.append(
                RuleProgram(
                    modes[0][selected[0]],
                    modes[1][selected[1]],
                    modes[2][selected[2]],
                )
            )
        # Neutral outcomes are exactly program independent; build one public
        # support tuple and verify every hidden query value produces it.
        shared_support = _marked_query_partial_support(
            programs[0],
            palette,
            query_axis,
        )
        for program in programs:
            support = _marked_query_partial_support(
                program,
                palette,
                query_axis,
            )
            if support != shared_support:
                raise AssertionError("neutral support unexpectedly leaked a rule")
            targets = tuple(
                simulate(probe.state, probe.action, program, palette)
                for probe in candidates
            )
            tasks.append(
                RuleGridTask(
                    inference=RuleGridInferenceView(
                        task_id=(
                            f"{split}/Q{query_axis.value}/G{group_index:04d}"
                        ),
                        support=shared_support,
                        active_candidates=candidates,
                        diagnostics=(),
                    ),
                    privileged=RuleGridPrivilegedTargets(
                        true_program=program,
                        palette=palette,
                        candidate_kinds=kinds,
                        active_targets=targets,
                        diagnostic_targets=(),
                        diagnostic_target_indices=(),
                    ),
                )
            )
    if len(tasks) != groups * PROGRAMS_PER_GROUP:
        raise AssertionError("nuisance task bank must balance three query values")
    return tuple(tasks)


def _symbolic_initial_joint_log_weights(
    *,
    torch: Any,
    tasks: Sequence[Any],
    factor_bank: Any,
) -> Any:
    """Uniform exact posterior over the 48 support-consistent factor codes."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    rows = []
    bank_rows = tuple(tuple(int(value) for value in row) for row in factor_bank)
    for task in tasks:
        allowed = {
            rule_program_factor_ids(program)
            for program in version_space(
                task.inference.support,
                task.privileged.palette,
            )
        }
        if len(allowed) != 48:
            raise AssertionError(
                f"marked partial support must leave 48 codes, got {len(allowed)}"
            )
        mask = torch.tensor(
            [code in allowed for code in bank_rows],
            dtype=torch.bool,
        )
        row = torch.full(
            (len(bank_rows),),
            float("-inf"),
            dtype=torch.float32,
        )
        row[mask] = -math.log(float(mask.sum()))
        rows.append(row)
    return torch.stack(rows, dim=0)


def _learned_marked_initial_joint_log_weights(
    *,
    torch: Any,
    model: Any,
    tasks: Sequence[Any],
    device: Any,
) -> Any:
    """Coverage audit only: infer belief with the final partial result marked."""

    from scripts.run_gram_public_coverage_finetune import (
        _raw_public_history_batch,
    )

    histories = tuple(task.inference.support for task in tasks)
    public = _raw_public_history_batch(torch, histories, device=device)
    marked = torch.zeros_like(public.support_mask)
    marked[:, -1] = True
    with torch.no_grad():
        belief = model.infer_factor_belief(
            public,
            is_agent_probe_result=marked,
        )
    return factor_marginals_to_joint_log_weights(
        torch,
        belief.factor_probabilities,
        model.factor_bank,
    ).detach().cpu()


def _exact_candidate_outcome_maps(
    *,
    torch: Any,
    tasks: Sequence[Any],
    factor_bank: Any,
) -> tuple[Any, Any]:
    """Materialize exact simulator outcomes and observed environment feedback."""

    from prp_wm.rulegrid import (
        ALL_COLLISIONS,
        ALL_RELATIONS,
        ALL_TRIGGERS,
        RuleProgram,
        simulate,
    )

    programs = tuple(
        RuleProgram(
            ALL_COLLISIONS[int(code[0])],
            ALL_TRIGGERS[int(code[1])],
            ALL_RELATIONS[int(code[2])],
        )
        for code in factor_bank
    )
    maps = []
    feedback = []
    for task in tasks:
        task_maps = []
        for probe in task.inference.active_candidates:
            task_maps.append(
                [
                    simulate(
                        probe.state,
                        probe.action,
                        program,
                        task.privileged.palette,
                    )
                    for program in programs
                ]
            )
        maps.append(task_maps)
        feedback.append(task.privileged.active_targets)
    return (
        torch.tensor(maps, dtype=torch.long),
        torch.tensor(feedback, dtype=torch.long),
    )


def _symbolic_to_learned_audit(
    *,
    torch: Any,
    symbolic_log_weights: Any,
    learned_log_weights: Any,
    factor_bank: Any,
    query: PublicDoorQuery,
) -> list[dict[str, object]]:
    """Measure learned coverage without feeding it into the main controller."""

    if symbolic_log_weights.shape != learned_log_weights.shape:
        raise ValueError("symbolic and learned posterior shapes must match")
    symbolic = _normalise_log_weights(torch, symbolic_log_weights)
    learned = _normalise_log_weights(torch, learned_log_weights)
    exact_mask = torch.isfinite(symbolic)
    symbolic_probabilities = symbolic.exp()
    learned_probabilities = learned.exp()
    query_values = factor_bank[:, query.axis_index]
    records = []
    for task_index in range(symbolic.shape[0]):
        mask = exact_mask[task_index]
        symbolic_to_learned_kl = (
            symbolic_probabilities[task_index, mask]
            * (
                symbolic[task_index, mask]
                - learned[task_index, mask]
            )
        ).sum()
        learned_query_marginal = _door_marginals(
            torch,
            learned_probabilities[task_index],
            query_values,
        )
        records.append(
            {
                "symbolic_to_learned_kl_nats": float(
                    symbolic_to_learned_kl
                ),
                "learned_mass_on_exact_48_codes": float(
                    learned_probabilities[task_index, mask].sum()
                ),
                "learned_query_marginal": [
                    float(value) for value in learned_query_marginal
                ],
            }
        )
    return records


def _uniform_orders(
    *,
    torch: Any,
    query_index: int,
    groups: int,
    candidates: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """One public random order per palette, shared by three hidden query values."""

    if groups <= 0 or candidates <= 0:
        raise ValueError("groups and candidates must be positive")
    orders = []
    for group_index in range(groups):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            seed + 1_000_003 * query_index + 10_007 * group_index
        )
        order = tuple(
            int(value)
            for value in torch.randperm(
                candidates,
                generator=generator,
            ).tolist()
        )
        orders.extend((order,) * PROGRAMS_PER_GROUP)
    return tuple(orders)


def _rollout_policy(
    *,
    torch: Any,
    policy: str,
    tasks: Sequence[Any],
    query: PublicDoorQuery,
    initial_log_weights: Any,
    factor_bank: Any,
    outcome_maps: Any,
    environment_feedback: Any,
    uniform_orders: Sequence[Sequence[int]],
    budgets: Sequence[int],
) -> tuple[dict[int, list[dict[str, object]]], list[dict[str, object]]]:
    """Run a fixed-budget paired policy; reveal feedback only after selection."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import RuleGridTransition

    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    materialized = tuple(tasks)
    batch_size, candidate_count, hypotheses = outcome_maps.shape[:3]
    if batch_size != len(materialized):
        raise ValueError("prediction batch must match tasks")
    if initial_log_weights.shape != (batch_size, hypotheses):
        raise ValueError("initial posterior has the wrong shape")
    if len(uniform_orders) != batch_size:
        raise ValueError("uniform order batch must match tasks")
    max_budget = max(budgets)
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
        partial = _task_metrics(
            torch=torch,
            log_weights=log_weights,
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
            # Kinds are attached only to completed traces/metrics.  They never
            # affect selection, posterior updates, stopping, or door choice.
            record["selected_candidate_categories_audit"] = [
                materialized[task_index].privileged.candidate_kinds[index]
                for index in selected_indices[task_index][:budget]
            ]
        snapshots[budget] = records

    if 0 in budgets:
        snapshot(0)
    for step in range(1, max_budget + 1):
        choices: list[int] = []
        step_scores: list[Any] = []
        for task_index in range(batch_size):
            if policy == "query-conditioned":
                score = _select_query_conditioned_candidate(
                    torch,
                    log_weights[task_index],
                    query_values,
                    outcome_maps[task_index],
                    available[task_index],
                )
            elif policy == "global-information-gain":
                global_score = _select_global_information_candidate(
                    torch,
                    log_weights[task_index],
                    outcome_maps[task_index],
                    available[task_index],
                )
                score = _acquisition_score(
                    torch,
                    log_weights[task_index],
                    query_values,
                    outcome_maps[
                        task_index,
                        global_score.candidate_index,
                    ],
                    candidate_index=global_score.candidate_index,
                )
            else:
                candidate_index = int(uniform_orders[task_index][step - 1])
                if not bool(available[task_index, candidate_index].item()):
                    raise AssertionError("uniform order repeated a candidate")
                score = _acquisition_score(
                    torch,
                    log_weights[task_index],
                    query_values,
                    outcome_maps[task_index, candidate_index],
                    candidate_index=candidate_index,
                )
            choices.append(score.candidate_index)
            step_scores.append(score)
        for task_index, candidate_index in enumerate(choices):
            available[task_index, candidate_index] = False
            selected_indices[task_index].append(candidate_index)
            selected_scores[task_index].append(step_scores[task_index])

        # The environment sidecar is first indexed after all choices are fixed.
        feedback_rows = environment_feedback[
            torch.arange(batch_size, device=environment_feedback.device),
            torch.tensor(
                choices,
                dtype=torch.long,
                device=environment_feedback.device,
            ),
        ]
        selected_maps = outcome_maps[
            torch.arange(batch_size),
            torch.tensor(choices, dtype=torch.long),
        ]
        matches = selected_maps.eq(
            feedback_rows[:, None],
        ).flatten(start_dim=2).all(dim=-1)
        log_likelihood = torch.full_like(
            log_weights,
            float("-inf"),
        )
        log_likelihood[matches] = 0.0
        log_weights, log_evidence = bayesian_log_likelihood_update(
            torch,
            log_weights,
            log_likelihood,
        )
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

    traces = [
        {
            "task_batch_index": task_index,
            "selected_candidate_indices": selected_indices[task_index],
            "selected_candidate_categories_audit": [
                materialized[task_index].privileged.candidate_kinds[index]
                for index in selected_indices[task_index]
            ],
            "first_symbolic_query_identification_step": first_identified[
                task_index
            ],
            "step_scores": [
                {
                    "candidate_index": score.candidate_index,
                    "expected_query_door_gain": score.expected_door_gain,
                    "global_outcome_information_gain_nats": (
                        score.outcome_information_gain_nats
                    ),
                    "predicted_outcome_classes": (
                        score.predicted_outcome_classes
                    ),
                }
                for score in selected_scores[task_index]
            ],
        }
        for task_index in range(batch_size)
    ]
    return snapshots, traces


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _summarise_policy(
    records: Sequence[dict[str, object]],
    *,
    budget: int,
) -> dict[str, object]:
    identified_steps = tuple(
        int(item["first_symbolic_identification_step"])
        for item in records
        if item["first_symbolic_identification_step"] is not None
        and int(item["first_symbolic_identification_step"]) <= budget
    )
    observed_log_probabilities = tuple(
        float(item["mean_observed_log_predictive_probability"])
        for item in records
        if item["mean_observed_log_predictive_probability"] is not None
    )
    first_categories = tuple(
        str(item["selected_candidate_categories_audit"][0])
        for item in records
        if item["selected_candidate_categories_audit"]
    )
    category_counts = {
        category: sum(category in item["selected_candidate_categories_audit"]
                      for item in records)
        for category in (
            "query-atomic",
            "nuisance-axis-0",
            "nuisance-axis-1",
            "neutral",
        )
    }
    return {
        "tasks": len(records),
        "terminal_win_rate": _mean(int(bool(item["won"])) for item in records),
        "mean_true_query_probability": _mean(
            float(item["true_door_probability"]) for item in records
        ),
        "mean_query_entropy_nats": _mean(
            float(item["door_entropy_nats"]) for item in records
        ),
        "mean_joint_effective_hypotheses": _mean(
            float(item["joint_effective_hypotheses"]) for item in records
        ),
        "symbolic_query_identified_rate": len(identified_steps) / len(records),
        "failure_penalized_mean_probes_to_query_identification": _mean(
            (
                int(item["first_symbolic_identification_step"])
                if item["first_symbolic_identification_step"] is not None
                and int(item["first_symbolic_identification_step"]) <= budget
                else budget + 1
            )
            for item in records
        ),
        "mean_cumulative_expected_query_gain": _mean(
            float(item["cumulative_expected_door_gain"]) for item in records
        ),
        "mean_cumulative_global_information_gain_nats": _mean(
            float(item["cumulative_outcome_information_gain_nats"])
            for item in records
        ),
        "mean_observed_log_predictive_probability": (
            _mean(observed_log_probabilities)
            if observed_log_probabilities
            else None
        ),
        "first_selection_category_rates_audit": {
            category: first_categories.count(category) / len(first_categories)
            for category in (
                "query-atomic",
                "nuisance-axis-0",
                "nuisance-axis-1",
                "neutral",
            )
        } if first_categories else {},
        "mean_selected_category_counts_audit": {
            category: count / len(records)
            for category, count in category_counts.items()
        },
    }


def _paired_summary(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
    *,
    budget: int,
) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("paired policies must contain the same tasks")
    left_wins = tuple(bool(item["won"]) for item in left)
    right_wins = tuple(bool(item["won"]) for item in right)

    def penalized(item: dict[str, object]) -> int:
        step = item["first_symbolic_identification_step"]
        return (
            int(step)
            if step is not None and int(step) <= budget
            else budget + 1
        )

    return {
        "tasks": len(left),
        "paired_terminal_win_rate_delta_left_minus_right": (
            _mean(left_wins) - _mean(right_wins)
        ),
        "left_only_wins": sum(
            a and not b for a, b in zip(left_wins, right_wins, strict=True)
        ),
        "right_only_wins": sum(
            b and not a for a, b in zip(left_wins, right_wins, strict=True)
        ),
        "both_win": sum(
            a and b for a, b in zip(left_wins, right_wins, strict=True)
        ),
        "neither_wins": sum(
            not a and not b for a, b in zip(left_wins, right_wins, strict=True)
        ),
        "paired_failure_penalized_probe_delta_left_minus_right": _mean(
            penalized(a) - penalized(b)
            for a, b in zip(left, right, strict=True)
        ),
    }


def _summarise_scope(
    records: dict[str, dict[int, list[dict[str, object]]]],
    budgets: Sequence[int],
) -> dict[str, object]:
    result: dict[str, object] = {}
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
                "query-conditioned_minus_global-information-gain": (
                    _paired_summary(
                        records["query-conditioned"][budget],
                        records["global-information-gain"][budget],
                        budget=budget,
                    )
                ),
                "query-conditioned_minus_uniform": _paired_summary(
                    records["query-conditioned"][budget],
                    records["uniform"][budget],
                    budget=budget,
                ),
                "global-information-gain_minus_uniform": _paired_summary(
                    records["global-information-gain"][budget],
                    records["uniform"][budget],
                    budget=budget,
                ),
            },
        }
    return result


def _empty_policy_records(
    budgets: Sequence[int],
) -> dict[str, dict[int, list[dict[str, object]]]]:
    return {
        policy: {budget: [] for budget in budgets}
        for policy in POLICIES
    }


def _summarise_belief_coverage(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "tasks": len(records),
        "mean_symbolic_to_learned_kl_nats": _mean(
            float(item["symbolic_to_learned_kl_nats"])
            for item in records
        ),
        "mean_learned_mass_on_exact_48_codes": _mean(
            float(item["learned_mass_on_exact_48_codes"])
            for item in records
        ),
        "mean_learned_query_marginal": [
            _mean(
                float(item["learned_query_marginal"][value])
                for item in records
            )
            for value in range(4)
        ],
    }
    return result


def main() -> None:
    args = parse_args()
    budgets, seeds = _validate_args(args)

    import torch

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import ALL_AXES
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_public_version_space_k4 import (
        load_public_version_k4_checkpoint,
    )

    _configure_determinism(torch, seeds[0])
    device = _resolve_device(torch, args.device)
    belief_path = args.belief_checkpoint.resolve()
    belief_model, belief_checkpoint, _, _ = load_public_version_k4_checkpoint(
        torch,
        belief_path,
        device=device,
    )
    if belief_checkpoint.get("support_input") != "raw":
        raise SystemExit("belief checkpoint must infer from raw public support")
    if not hasattr(belief_model, "infer_factor_belief"):
        raise SystemExit("belief checkpoint has no factor-belief interface")
    factor_bank = belief_model.factor_bank.detach().cpu()
    if factor_bank.shape != (64, 3):
        raise SystemExit("nuisance ceiling requires the complete 64-code bank")

    by_query_records: dict[
        str,
        dict[str, dict[int, list[dict[str, object]]]],
    ] = {
        axis.value: _empty_policy_records(budgets)
        for axis in ALL_AXES
    }
    aggregate_records = _empty_policy_records(budgets)
    by_seed_records = {
        str(seed): _empty_policy_records(budgets)
        for seed in seeds
    }
    traces: dict[
        str,
        dict[str, dict[str, list[dict[str, object]]]],
    ] = {}
    belief_coverage_records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []

    for seed in seeds:
        _configure_determinism(torch, seed)
        seed_traces: dict[
            str,
            dict[str, list[dict[str, object]]],
        ] = {}
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
            candidate_count = len(tasks[0].inference.active_candidates)
            if candidate_count != CANDIDATES_PER_TASK:
                raise AssertionError("nuisance menu size changed unexpectedly")
            uniform_orders = _uniform_orders(
                torch=torch,
                query_index=query.axis_index,
                groups=args.groups_per_query,
                candidates=candidate_count,
                seed=seed,
            )
            query_traces = {policy: [] for policy in POLICIES}
            for start in range(0, len(tasks), args.batch_size):
                selected_tasks = tasks[start : start + args.batch_size]
                selected_orders = uniform_orders[
                    start : start + args.batch_size
                ]
                symbolic_initial = _symbolic_initial_joint_log_weights(
                    torch=torch,
                    tasks=selected_tasks,
                    factor_bank=factor_bank,
                )
                learned_initial = (
                    _learned_marked_initial_joint_log_weights(
                        torch=torch,
                        model=belief_model,
                        tasks=selected_tasks,
                        device=device,
                    )
                )
                batch_coverage = _symbolic_to_learned_audit(
                    torch=torch,
                    symbolic_log_weights=symbolic_initial,
                    learned_log_weights=learned_initial,
                    factor_bank=factor_bank,
                    query=query,
                )
                for local_index, audit_record in enumerate(batch_coverage):
                    global_task_index = start + local_index
                    belief_coverage_records.append(
                        {
                            "seed": seed,
                            "query_axis": query_axis.value,
                            "group_index": (
                                global_task_index // PROGRAMS_PER_GROUP
                            ),
                            "environment_cluster_id": (
                                f"{query_axis.value}:"
                                f"{global_task_index // PROGRAMS_PER_GROUP}"
                            ),
                            "hidden_query_slot": (
                                global_task_index % PROGRAMS_PER_GROUP
                            ),
                            **audit_record,
                        }
                    )
                maps, feedback = _exact_candidate_outcome_maps(
                    torch=torch,
                    tasks=selected_tasks,
                    factor_bank=factor_bank,
                )
                for policy in POLICIES:
                    snapshots, batch_traces = _rollout_policy(
                        torch=torch,
                        policy=policy,
                        tasks=selected_tasks,
                        query=query,
                        initial_log_weights=symbolic_initial,
                        factor_bank=factor_bank,
                        outcome_maps=maps,
                        environment_feedback=feedback,
                        uniform_orders=selected_orders,
                        budgets=budgets,
                    )
                    for budget, records in snapshots.items():
                        by_query_records[
                            query_axis.value
                        ][policy][budget].extend(records)
                        by_seed_records[str(seed)][policy][budget].extend(
                            records
                        )
                        aggregate_records[policy][budget].extend(records)
                        for local_index, record in enumerate(records):
                            global_task_index = start + local_index
                            task = selected_tasks[local_index]
                            hidden_query_value = rule_program_factor_ids(
                                task.privileged.true_program
                            )[query.axis_index]
                            bootstrap_records.append(
                                {
                                    "seed": seed,
                                    "query_axis": query_axis.value,
                                    "group_index": (
                                        global_task_index
                                        // PROGRAMS_PER_GROUP
                                    ),
                                    "environment_cluster_id": (
                                        f"{query_axis.value}:"
                                        f"{global_task_index // PROGRAMS_PER_GROUP}"
                                    ),
                                    "hidden_query_slot": (
                                        global_task_index
                                        % PROGRAMS_PER_GROUP
                                    ),
                                    "hidden_query_value_audit": (
                                        hidden_query_value
                                    ),
                                    "policy": policy,
                                    "budget": budget,
                                    "won": bool(record["won"]),
                                    "true_query_probability": float(
                                        record["true_door_probability"]
                                    ),
                                    "query_entropy_nats": float(
                                        record["door_entropy_nats"]
                                    ),
                                    "first_symbolic_identification_step": (
                                        record[
                                            "first_symbolic_identification_step"
                                        ]
                                    ),
                                    "selected_categories_audit": record[
                                        "selected_candidate_categories_audit"
                                    ],
                                }
                            )
                    remaining = max(
                        0,
                        args.trace_tasks_per_query
                        - len(query_traces[policy]),
                    )
                    for trace in batch_traces[:remaining]:
                        query_traces[policy].append(
                            {
                                "global_task_index": (
                                    start
                                    + int(trace["task_batch_index"])
                                ),
                                "query_axis": query_axis.value,
                                **{
                                    key: value
                                    for key, value in trace.items()
                                    if key != "task_batch_index"
                                },
                            }
                        )
            seed_traces[query_axis.value] = query_traces
        traces[str(seed)] = seed_traces

    selector_audit = _selector_boundary_audit()
    if not selector_audit["passed"]:
        raise AssertionError("selector boundary audit failed")
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "experiment": (
            "oracle_canonical_query_conditioned_nuisance_acquisition_ceiling"
        ),
        "status": "complete",
        "interpretation": (
            "Privileged symbolic ceiling: a marked query-partial no-op leaves "
            "3x4x4=48 exact codes; equal-cost atomic candidates diagnose "
            "either the query or one nuisance axis."
        ),
        "controller_reads_explicit_query_axis": True,
        "controller_infers_query_from_task_id": False,
        "candidate_selection_reads_task_or_probe_id": False,
        "candidate_selection_reads_candidate_kind": False,
        "candidate_selection_reads_true_program_or_feedback": False,
        "feedback_read_only_after_candidate_selection": True,
        "candidate_kind_used_only_for_posthoc_audit_metrics": True,
        "query_conditioned_objective": (
            "expected terminal query-door success gain; global deterministic "
            "outcome information as secondary key"
        ),
        "global_information_objective": (
            "entropy in nats of the candidate's exact deterministic outcome "
            "partition over the joint posterior; query not supplied"
        ),
        "posterior_initialization": (
            "uniform symbolic exact mask over the 48 factor codes consistent "
            "with four neutral transitions and one marked query-partial no-op"
        ),
        "posterior_update": (
            "exact deterministic Bayes filter retaining hypotheses whose "
            "simulator outcome equals the observed next state"
        ),
        "candidate_outcome_model": (
            "privileged exact RuleGrid simulator partition; no learned "
            "executor is used in the primary acquisition result"
        ),
        "paired_protocol": (
            "same three hidden query values per palette, exact t0 posterior, "
            "shuffled atomic candidate menu, feedback, and budgets"
        ),
        "bootstrap_protocol": {
            "recommended_environment_cluster": (
                "(query_axis, group_index), exposed as "
                "environment_cluster_id"
            ),
            "candidate_order_seeds_are_repeated_measures": True,
            "hidden_query_slots_are_repeated_measures": True,
            "warning": (
                "Do not treat seed x hidden-slot rows as independent "
                "environments; cluster or use a hierarchical bootstrap."
            ),
        },
        "candidate_menu": {
            "query_atomic": 2,
            "first_nuisance_axis_atomic": 2,
            "second_nuisance_axis_atomic": 2,
            "neutral": 2,
        },
        "exact_48_code_theoretical_reference": {
            "query_strong": {
                "expected_query_door_gain": 2.0 / 3.0,
                "global_information_gain_nats": math.log(3.0),
            },
            "one_nuisance_axis_strong": {
                "expected_query_door_gain": 0.0,
                "global_information_gain_nats": math.log(4.0),
            },
            "neutral": {
                "expected_query_door_gain": 0.0,
                "global_information_gain_nats": 0.0,
            },
        },
        "learned_belief_coverage_audit": {
            "used_by_primary_controller": False,
            "aggregate": {
                **_summarise_belief_coverage(belief_coverage_records),
            },
            "by_query_axis": {
                axis.value: _summarise_belief_coverage(
                    tuple(
                        record
                        for record in belief_coverage_records
                        if record["query_axis"] == axis.value
                    )
                )
                for axis in ALL_AXES
            },
            "task_records": belief_coverage_records,
        },
        "budgets": list(budgets),
        "seeds": list(seeds),
        "groups_per_query": args.groups_per_query,
        "programs_per_group": PROGRAMS_PER_GROUP,
        "tasks_per_query": PROGRAMS_PER_GROUP * args.groups_per_query,
        "total_tasks": (
            len(ALL_AXES)
            * PROGRAMS_PER_GROUP
            * args.groups_per_query
            * len(seeds)
        ),
        "data_master_seed": args.data_master_seed,
        "split": args.split,
        "device": str(device),
        "belief_checkpoint": str(belief_path),
        "belief_checkpoint_sha256": _sha256_file(belief_path),
        "belief_checkpoint_model_type": belief_checkpoint.get("model_type"),
        "selector_boundary_audit": selector_audit,
        "evaluations": {
            "by_query_axis": {
                axis: _summarise_scope(records, budgets)
                for axis, records in by_query_records.items()
            },
            "aggregate": _summarise_scope(
                aggregate_records,
                budgets,
            ),
            "by_seed": {
                seed: _summarise_scope(records, budgets)
                for seed, records in by_seed_records.items()
            },
        },
        "bootstrap_task_records": bootstrap_records,
        "episode_traces": traces,
        "source_sha256": _source_sha256(),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
