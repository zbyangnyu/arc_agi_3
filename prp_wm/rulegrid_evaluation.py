"""Exact Stage 0-B oracle evaluation for :mod:`prp_wm.rulegrid`.

The evaluator deliberately has no learned policy.  It establishes whether the
RuleGrid construction has the pre-registered active-identification headroom
before any neural model is trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import random
import statistics
from typing import Iterable

from .rulegrid import (
    ALL_AXES,
    ALL_PROGRAMS,
    Axis,
    RuleGridProbe,
    RuleGridTask,
    RuleProgram,
    behavior_classes,
    behavior_signature,
    count_changed_cells,
    expected_heldout_version_space,
    grid_bytes,
    iter_split_tasks,
    partition_sizes,
    simulate,
    version_space,
)


@dataclass(frozen=True)
class Gate0BTaskScore:
    task_id: str
    heldout_axis: str
    true_mode_id: int
    oracle_rmst4: float
    uniform_exact_rmst4: float


@dataclass(frozen=True)
class Gate0BReport:
    benchmark_version: str
    tasks: int
    repeats_per_program_axis: int
    budget: int
    bootstrap_resamples: int
    oracle_rmst4: float
    uniform_exact_rmst4: float
    uniform_minus_oracle_rmst4_ci95: tuple[float, float]
    relative_rmst_reduction: float
    all_calibration_version_spaces_are_four: bool
    all_neutral_supports_preserve_version_space: bool
    candidate_partition_counts: dict[str, int]
    all_neutral_change_requirements_hold: bool
    all_diagnostic_signatures_unique: bool
    gate_threshold: float
    gate_eligible: bool
    passes: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _PreparedTask:
    task: RuleGridTask
    versions: tuple[RuleProgram, ...]
    diagnostic_signatures: dict[RuleProgram, tuple[bytes, ...]]
    candidate_outcomes: tuple[dict[RuleProgram, bytes], ...]


def _mode_id(program: RuleProgram, axis: Axis) -> int:
    mode = program.mode_for_axis(axis)
    return list(type(mode)).index(mode)


def _entropy_from_signature_groups(
    programs: Iterable[RuleProgram], signatures: dict[RuleProgram, tuple[bytes, ...]]
) -> float:
    buckets: dict[tuple[bytes, ...], int] = {}
    for program in programs:
        signature = signatures[program]
        buckets[signature] = buckets.get(signature, 0) + 1
    total = sum(buckets.values())
    if total == 0:
        raise ValueError("cannot calculate entropy of an empty version space")
    return -sum(
        (count / total) * math.log2(count / total) for count in buckets.values()
    )


def _identified(
    programs: Iterable[RuleProgram], signatures: dict[RuleProgram, tuple[bytes, ...]]
) -> bool:
    return len({signatures[program] for program in programs}) <= 1


def _candidate_eig(
    versions: tuple[RuleProgram, ...],
    outcomes: dict[RuleProgram, bytes],
    signatures: dict[RuleProgram, tuple[bytes, ...]],
) -> float:
    prior_entropy = _entropy_from_signature_groups(versions, signatures)
    partitions: dict[bytes, list[RuleProgram]] = {}
    for program in versions:
        partitions.setdefault(outcomes[program], []).append(program)
    expected_posterior_entropy = sum(
        (len(programs) / len(versions))
        * _entropy_from_signature_groups(programs, signatures)
        for programs in partitions.values()
    )
    return max(0.0, prior_entropy - expected_posterior_entropy)


def _posterior_after_indices(
    prepared: _PreparedTask, indices: Iterable[int]
) -> tuple[RuleProgram, ...]:
    true_program = prepared.task.privileged.true_program
    candidates = prepared.candidate_outcomes
    versions = prepared.versions
    for index in indices:
        observed = candidates[index][true_program]
        versions = tuple(
            program for program in versions if candidates[index][program] == observed
        )
    return versions


def _oracle_rmst(prepared: _PreparedTask, budget: int) -> float:
    versions = prepared.versions
    unused = set(range(len(prepared.candidate_outcomes)))
    true_program = prepared.task.privileged.true_program
    for step in range(budget + 1):
        if _identified(versions, prepared.diagnostic_signatures):
            return float(step)
        if step == budget:
            return float(budget)
        # Tie-break by public candidate ID, not a sidecar kind or target.
        selected = min(
            unused,
            key=lambda index: (
                -_candidate_eig(
                    versions,
                    prepared.candidate_outcomes[index],
                    prepared.diagnostic_signatures,
                ),
                prepared.task.inference.active_candidates[index].probe_id,
            ),
        )
        unused.remove(selected)
        observed = prepared.candidate_outcomes[selected][true_program]
        versions = tuple(
            program
            for program in versions
            if prepared.candidate_outcomes[selected][program] == observed
        )
    raise AssertionError("unreachable oracle loop exit")


def _uniform_without_replacement_exact_rmst(prepared: _PreparedTask, budget: int) -> float:
    """Exact RMST by uniform subsets; no Monte Carlo policy rollout occurs."""

    candidate_count = len(prepared.candidate_outcomes)
    if budget > candidate_count:
        raise ValueError("budget cannot exceed the number of candidates")
    rmst = 0.0
    # After t uniform draws without replacement every subset of size t has the
    # same probability.  This is equivalent to the <= 2^8 subset DP required
    # by the protocol but simpler to audit for this eight-candidate bank.
    for step in range(budget):
        subsets = tuple(itertools.combinations(range(candidate_count), step))
        survival = sum(
            not _identified(
                _posterior_after_indices(prepared, selected),
                prepared.diagnostic_signatures,
            )
            for selected in subsets
        ) / len(subsets)
        rmst += survival
    return rmst


def _validate_task(prepared: _PreparedTask) -> tuple[dict[str, int], bool, bool, bool]:
    """Hard-fail generator checks relevant to Gate 0-B."""

    task = prepared.task
    palette = task.privileged.palette
    expected = expected_heldout_version_space(
        task.privileged.true_program,
        _axis_from_task_id(task.inference.task_id),
    )
    exact_calibration = version_space(task.inference.support[:2], palette)
    exact_full_support = version_space(task.inference.support, palette)
    if exact_calibration != expected:
        raise AssertionError(
            f"calibration version space mismatch in {task.inference.task_id}: "
            f"{len(exact_calibration)} candidates"
        )
    if exact_full_support != expected:
        raise AssertionError(f"neutral support changed the version space in {task.inference.task_id}")
    if prepared.versions != expected:
        raise AssertionError("prepared version space must agree with exact enumeration")

    counts = {"strong": 0, "partial": 0, "neutral-large-change": 0}
    strong_change_counts: list[int] = []
    neutral_change_counts: list[int] = []
    for kind, probe in zip(
        task.privileged.candidate_kinds,
        task.inference.active_candidates,
        strict=True,
    ):
        counts[kind] = counts.get(kind, 0) + 1
        sizes = partition_sizes(prepared.versions, probe, palette)
        if kind == "strong":
            if sizes != (1, 1, 1, 1):
                raise AssertionError(f"strong probe is not 1+1+1+1: {sizes}")
            strong_change_counts.extend(
                count_changed_cells(probe.state, simulate(probe.state, probe.action, program, palette))
                for program in prepared.versions
            )
        elif kind == "partial":
            if sizes != (1, 3):
                raise AssertionError(f"partial probe is not 1+3: {sizes}")
        elif kind == "neutral-large-change":
            if sizes != (4,):
                raise AssertionError(f"neutral probe is not behavior-neutral: {sizes}")
            neutral_change_counts.extend(
                count_changed_cells(probe.state, simulate(probe.state, probe.action, program, palette))
                for program in prepared.versions
            )
        else:
            raise AssertionError(f"unknown private candidate kind: {kind!r}")
    if counts != {"strong": 2, "partial": 2, "neutral-large-change": 4}:
        raise AssertionError(f"wrong active candidate composition: {counts!r}")
    neutral_large_enough = min(neutral_change_counts) >= statistics.median(strong_change_counts)
    if not neutral_large_enough:
        raise AssertionError("neutral-large-change probes are too visually small")
    all_signature_classes = behavior_classes(ALL_PROGRAMS, task.inference.diagnostics, palette)
    signature_unique = len(all_signature_classes) == len(ALL_PROGRAMS)
    if not signature_unique:
        raise AssertionError("full diagnostic panel does not identify all 64 programs")
    return counts, True, True, signature_unique


def _axis_from_task_id(value: str) -> Axis:
    # Stable ID form ends in .../H<axis-id>/N<replicate>.  This parser is used
    # only by the oracle evaluator and intentionally has no hidden metadata.
    pieces = value.split("/")
    if len(pieces) != 4 or not pieces[2].startswith("H"):
        raise ValueError(f"malformed task ID: {value!r}")
    try:
        axis_id = int(pieces[2][1:])
    except ValueError as error:
        raise ValueError(f"malformed heldout axis in task ID: {value!r}") from error
    try:
        return ALL_AXES[axis_id]
    except IndexError as error:
        raise ValueError(f"unknown heldout axis in task ID: {value!r}") from error


def _prepare_task(task: RuleGridTask) -> _PreparedTask:
    axis = _axis_from_task_id(task.inference.task_id)
    versions = expected_heldout_version_space(task.privileged.true_program, axis)
    signatures = {
        program: behavior_signature(program, task.inference.diagnostics, task.privileged.palette)
        for program in versions
    }
    outcomes = tuple(
        {
            program: grid_bytes(
                simulate(probe.state, probe.action, program, task.privileged.palette)
            )
            for program in versions
        }
        for probe in task.inference.active_candidates
    )
    return _PreparedTask(task, versions, signatures, outcomes)


def _stratified_bootstrap_ci95(
    scores: Iterable[Gate0BTaskScore], *, resamples: int, seed: int
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    strata: dict[tuple[str, int], list[float]] = {}
    for score in scores:
        strata.setdefault((score.heldout_axis, score.true_mode_id), []).append(
            score.uniform_exact_rmst4 - score.oracle_rmst4
        )
    if not strata or any(not values for values in strata.values()):
        raise ValueError("bootstrap requires non-empty strata")
    total = sum(len(values) for values in strata.values())
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled_sum = 0.0
        for values in strata.values():
            sampled_sum += sum(values[rng.randrange(len(values))] for _ in values)
        estimates.append(sampled_sum / total)
    estimates.sort()
    lower_index = max(0, math.ceil(0.025 * resamples) - 1)
    upper_index = min(resamples - 1, math.ceil(0.975 * resamples) - 1)
    return estimates[lower_index], estimates[upper_index]


def evaluate_gate0b(
    *,
    repeats: int = 25,
    budget: int = 4,
    bootstrap_resamples: int = 2_000,
    seed: int = 2026071701,
    gate_threshold: float = 0.25,
) -> Gate0BReport:
    """Run the fully enumerated Stage 0-B headroom gate.

    With defaults this evaluates 64 programs x 3 held-out axes x 25 nuisance
    repeats = 4,800 paired tasks.  The uniform comparator is exact over action
    subsets, so its result does not depend on policy rollout seeds.
    """

    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if not 0.0 < gate_threshold < 1.0:
        raise ValueError("gate_threshold must lie strictly between 0 and 1")

    scores: list[Gate0BTaskScore] = []
    partition_counts = {"strong": 0, "partial": 0, "neutral-large-change": 0}
    calibration_ok = True
    neutral_support_ok = True
    neutral_change_ok = True
    signature_ok = True
    for task in iter_split_tasks("gate0b", repeats=repeats):
        prepared = _prepare_task(task)
        counts, calibration, neutral_support, signatures = _validate_task(prepared)
        for key, value in counts.items():
            partition_counts[key] += value
        calibration_ok = calibration_ok and calibration
        neutral_support_ok = neutral_support_ok and neutral_support
        signature_ok = signature_ok and signatures
        # _validate_task raises on an invalid neutral size; retained as an
        # explicit report field for a self-contained machine-readable gate.
        neutral_change_ok = neutral_change_ok and True
        axis = _axis_from_task_id(task.inference.task_id)
        scores.append(
            Gate0BTaskScore(
                task_id=task.inference.task_id,
                heldout_axis=axis.value,
                true_mode_id=_mode_id(task.privileged.true_program, axis),
                oracle_rmst4=_oracle_rmst(prepared, budget),
                uniform_exact_rmst4=_uniform_without_replacement_exact_rmst(
                    prepared, budget
                ),
            )
        )

    oracle = statistics.fmean(item.oracle_rmst4 for item in scores)
    uniform = statistics.fmean(item.uniform_exact_rmst4 for item in scores)
    relative_reduction = (uniform - oracle) / uniform
    ci95 = _stratified_bootstrap_ci95(
        scores, resamples=bootstrap_resamples, seed=seed
    )
    expected_tasks = len(ALL_PROGRAMS) * len(ALL_AXES) * 25
    gate_eligible = (
        len(scores) >= expected_tasks
        and repeats == 25
        and budget == 4
        and bootstrap_resamples >= 2_000
    )
    passes = (
        gate_eligible
        and calibration_ok
        and neutral_support_ok
        and neutral_change_ok
        and signature_ok
        and partition_counts
        == {
            "strong": 2 * len(scores),
            "partial": 2 * len(scores),
            "neutral-large-change": 4 * len(scores),
        }
        and relative_reduction >= gate_threshold
        and ci95[0] > 0.0
    )
    return Gate0BReport(
        benchmark_version="prp-rulegrid-v0.2.0",
        tasks=len(scores),
        repeats_per_program_axis=repeats,
        budget=budget,
        bootstrap_resamples=bootstrap_resamples,
        oracle_rmst4=oracle,
        uniform_exact_rmst4=uniform,
        uniform_minus_oracle_rmst4_ci95=ci95,
        relative_rmst_reduction=relative_reduction,
        all_calibration_version_spaces_are_four=calibration_ok,
        all_neutral_supports_preserve_version_space=neutral_support_ok,
        candidate_partition_counts=partition_counts,
        all_neutral_change_requirements_hold=neutral_change_ok,
        all_diagnostic_signatures_unique=signature_ok,
        gate_threshold=gate_threshold,
        gate_eligible=gate_eligible,
        passes=passes,
    )

