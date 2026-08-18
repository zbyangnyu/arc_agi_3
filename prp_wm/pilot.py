"""Deterministic task streams for the RuleGrid Stage-1 pilot.

The pilot is deliberately narrower than the preregistered five-seed study.  It
exists to exercise the real RuleGrid-to-PersistentK4 path on a GPU and to make
one leakage boundary mechanically obvious: composition (triple) diagnostics
are never materialized for the training loss.

This module does not import PyTorch at import time.  Stage 0 remains usable on
a dependency-free installation, and the pure task-stream invariants can be
tested independently of the optional neural extra.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .rulegrid import ALL_AXES, BENCHMARK_VERSION, Axis, RuleProgram, RuleGridTask, make_rulegrid_task


PILOT_PROTOCOL_VERSION = "prp-wm-rulegrid-pilot-v2"
"""Version of the deterministic pilot stream and its target split."""

NONTRIPLE_DIAGNOSTIC_INDICES: tuple[int, ...] = tuple(range(21))
"""Canonical single/pair queries.  These are the only pilot train targets."""

TRIPLE_DIAGNOSTIC_INDICES: tuple[int, ...] = (21, 22, 23)
"""Canonical composition diagnostic queries, evaluation-only for the pilot."""

_TASKS_PER_NUISANCE_GROUP = 64 * len(ALL_AXES)
_COPRIME_STRIDE = 137


def _stable_uint64(*parts: object) -> int:
    """Return a process-stable integer without relying on Python's hash salt."""

    source = "|".join(str(part) for part in (PILOT_PROTOCOL_VERSION, *parts))
    return int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:8], "little")


def assert_nontriple_training_indices(indices: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Reject a training target selection that contains a triple diagnostic.

    The pilot keeps its train selection intentionally fixed rather than making
    it a permissive CLI option.  Requiring the exact canonical set prevents a
    later convenience edit from silently converting the composition holdout
    into an in-distribution target.
    """

    normalized = tuple(indices)
    if normalized != NONTRIPLE_DIAGNOSTIC_INDICES:
        triple_overlap = sorted(set(normalized).intersection(TRIPLE_DIAGNOSTIC_INDICES))
        detail = (
            f"; forbidden triple indices present: {triple_overlap}"
            if triple_overlap
            else ""
        )
        raise ValueError(
            "pilot training must use exactly diagnostic indices 0..20" + detail
        )
    return normalized


def _validate_stream_arguments(
    *, split: str, master_seed: int, start: int, count: int
) -> None:
    if type(split) is not str or not split or "/" in split:
        raise ValueError("split must be a non-empty slash-free string")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative integer")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")


def pilot_task_specs(
    *,
    split: str,
    master_seed: int,
    start: int,
    count: int,
) -> tuple[tuple[RuleProgram, Axis, int], ...]:
    """Return deterministic ``(program, heldout_axis, replicate)`` task specs.

    Every consecutive 192-task nuisance group contains every program/axis
    combination exactly once.  Crucially, the replicate is generated from the
    group index *before* the program is selected, so all 64 programs for an
    axis share the same palette and geometry.  A model therefore cannot infer
    the hidden program from a nuisance layout in this stream.
    """

    _validate_stream_arguments(
        split=split, master_seed=master_seed, start=start, count=count
    )
    offset = _stable_uint64(BENCHMARK_VERSION, split, master_seed, "order") % _TASKS_PER_NUISANCE_GROUP
    specs: list[tuple[RuleProgram, Axis, int]] = []
    for sample_index in range(start, start + count):
        group_index, position = divmod(sample_index, _TASKS_PER_NUISANCE_GROUP)
        # 137 is coprime with 192, so this is a permutation within each group.
        rank = (position * _COPRIME_STRIDE + offset) % _TASKS_PER_NUISANCE_GROUP
        axis = ALL_AXES[rank // 64]
        program = RuleProgram.from_program_id(rank % 64)
        # No hidden program or axis appears in this nuisance seed.
        replicate = _stable_uint64(
            BENCHMARK_VERSION, split, master_seed, group_index, "replicate"
        )
        specs.append((program, axis, replicate))
    return tuple(specs)


def make_pilot_tasks(
    *,
    split: str,
    master_seed: int,
    start: int,
    count: int,
    diagnostic_indices: tuple[int, ...] | list[int] | None = None,
) -> tuple[RuleGridTask, ...]:
    """Materialize a deterministic, balanced batch of RuleGrid tasks.

    When ``diagnostic_indices`` is supplied, the lower-level RuleGrid builder
    simulates only those privileged diagnostic targets.  In particular, pilot
    training passes ``0..20`` and never constructs triple targets.
    """

    return tuple(
        make_rulegrid_task(
            program,
            heldout_axis,
            replicate,
            split=split,
            master_seed=master_seed,
            diagnostic_indices=diagnostic_indices,
        )
        for program, heldout_axis, replicate in pilot_task_specs(
            split=split,
            master_seed=master_seed,
            start=start,
            count=count,
        )
    )


def make_pilot_tensor_batch(
    *,
    split: str,
    master_seed: int,
    start: int,
    count: int,
    diagnostic_indices: tuple[int, ...] | list[int],
    include_behavior_targets: bool,
    prefix_length: int = 6,
    device: Any | None = None,
) -> Any:
    """Materialize a neural batch from the pilot stream.

    ``diagnostic_indices`` is deliberately explicit at this boundary.  The
    training script passes the fixed non-triple set; composition evaluation
    passes only ``21..23``.  The adapter then materializes no other diagnostic
    target tensors.
    """

    # Local import preserves the dependency-free Stage 0 package boundary.
    from .neural import rulegrid_tasks_to_tensor_batch

    return rulegrid_tasks_to_tensor_batch(
        make_pilot_tasks(
            split=split,
            master_seed=master_seed,
            start=start,
            count=count,
            diagnostic_indices=diagnostic_indices,
        ),
        prefix_length=prefix_length,
        include_behavior_targets=include_behavior_targets,
        diagnostic_indices=diagnostic_indices,
        device=device,
    )


__all__ = [
    "NONTRIPLE_DIAGNOSTIC_INDICES",
    "PILOT_PROTOCOL_VERSION",
    "TRIPLE_DIAGNOSTIC_INDICES",
    "assert_nontriple_training_indices",
    "make_pilot_tasks",
    "make_pilot_tensor_batch",
    "pilot_task_specs",
]
