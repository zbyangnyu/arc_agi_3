"""Exact, deterministic RuleGrid v0.2.0 environment.

This module is deliberately symbolic.  It is the Stage 0-B testbed for the
PRP-WM protocol, not a learned world model.  It provides three things which
are easy to accidentally conflate in a neural experiment:

* an executable transition DSL for collision, trigger, and relation rules;
* an enumerable 64-program version space and behavior-class oracle; and
* public task inputs separated from simulator-only targets.

The implementation uses immutable tuple grids instead of NumPy so the exact
oracle and golden tests have no third-party runtime dependency.  ``Grid`` is
always an 8 by 8 tuple of integer palette values in ``0..15``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import itertools
import json
import math
import random
from typing import Iterable, Iterator, Mapping, Sequence, TypeAlias


GRID_SIZE = 8
NUM_COLORS = 16
BENCHMARK_VERSION = "prp-rulegrid-v0.2.0"
MASTER_SEED = 2026071601

Coord: TypeAlias = tuple[int, int]
Grid: TypeAlias = tuple[tuple[int, ...], ...]


class Collision(str, Enum):
    STOP = "STOP"
    BOUNCE = "BOUNCE"
    PASS = "PASS"
    PUSH = "PUSH"


class Trigger(str, Enum):
    TOGGLE = "TOGGLE"
    DELETE = "DELETE"
    SPAWN = "SPAWN"
    RECOLOR = "RECOLOR"


class Relation(str, Enum):
    SWAP = "SWAP"
    FOLLOW = "FOLLOW"
    REPEL = "REPEL"
    NONE = "NONE"


class Axis(str, Enum):
    COLLISION = "collision"
    TRIGGER = "trigger"
    RELATION = "relation"


class Direction(str, Enum):
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"


class ActionKind(str, Enum):
    MOVE = "MOVE"
    ACTIVATE = "ACTIVATE"


ALL_COLLISIONS: tuple[Collision, ...] = tuple(Collision)
ALL_TRIGGERS: tuple[Trigger, ...] = tuple(Trigger)
ALL_RELATIONS: tuple[Relation, ...] = tuple(Relation)
ALL_AXES: tuple[Axis, ...] = tuple(Axis)

_DIRECTION_VECTOR: Mapping[Direction, Coord] = {
    Direction.NORTH: (-1, 0),
    Direction.EAST: (0, 1),
    Direction.SOUTH: (1, 0),
    Direction.WEST: (0, -1),
}


def direction_vector(direction: Direction) -> Coord:
    """Return the row/column vector for a public direction token."""

    if not isinstance(direction, Direction):
        raise ValueError(f"unknown direction: {direction!r}")
    return _DIRECTION_VECTOR[direction]


def add_coord(coord: Coord, vector: Coord, scale: int = 1) -> Coord:
    return (coord[0] + scale * vector[0], coord[1] + scale * vector[1])


def in_bounds(coord: Coord) -> bool:
    return 0 <= coord[0] < GRID_SIZE and 0 <= coord[1] < GRID_SIZE


def _require_color(value: int, name: str = "color") -> None:
    if type(value) is not int or not 0 <= value < NUM_COLORS:
        raise ValueError(f"{name} must be an integer in 0..15, got {value!r}")


def validate_grid(grid: Grid) -> None:
    if not isinstance(grid, tuple) or len(grid) != GRID_SIZE:
        raise ValueError(f"grid must have exactly {GRID_SIZE} rows")
    for row_index, row in enumerate(grid):
        if not isinstance(row, tuple) or len(row) != GRID_SIZE:
            raise ValueError(f"grid row {row_index} must have exactly {GRID_SIZE} cells")
        for column_index, value in enumerate(row):
            _require_color(value, f"grid[{row_index}][{column_index}]")


def blank_grid() -> Grid:
    return ((0,) * GRID_SIZE,) * GRID_SIZE


def grid_with_cells(cells: Mapping[Coord, int]) -> Grid:
    """Construct a valid immutable grid, rejecting duplicate/out-of-bounds cells."""

    rows = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for coord, color in cells.items():
        if not in_bounds(coord):
            raise ValueError(f"cell coordinate is outside the grid: {coord!r}")
        _require_color(color)
        rows[coord[0]][coord[1]] = color
    return tuple(tuple(row) for row in rows)


def grid_cell(grid: Grid, coord: Coord) -> int:
    validate_grid(grid)
    if not in_bounds(coord):
        raise ValueError(f"cell coordinate is outside the grid: {coord!r}")
    return grid[coord[0]][coord[1]]


def grid_bytes(grid: Grid) -> bytes:
    validate_grid(grid)
    return bytes(value for row in grid for value in row)


def grid_to_jsonable(grid: Grid) -> list[list[int]]:
    validate_grid(grid)
    return [list(row) for row in grid]


def grid_from_jsonable(value: object) -> Grid:
    if not isinstance(value, list):
        raise ValueError("serialized grid must be a JSON list")
    try:
        grid = tuple(tuple(row) for row in value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("serialized grid rows must be iterable") from error
    validate_grid(grid)
    return grid


@dataclass(frozen=True, order=True)
class Shape:
    """A normalized connected object shape, retained for future Shape-OOD data."""

    cells: tuple[Coord, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("a shape must contain at least one cell")
        normalized = _normalize_cells(self.cells)
        if normalized != self.cells:
            raise ValueError("shape cells must be normalized and row/column sorted")
        if len(set(self.cells)) != len(self.cells):
            raise ValueError("shape cells must be unique")

    def translated(self, anchor: Coord) -> tuple[Coord, ...]:
        return tuple((anchor[0] + row, anchor[1] + col) for row, col in self.cells)


def _normalize_cells(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    materialized = tuple(cells)
    if not materialized:
        raise ValueError("cannot normalize an empty shape")
    min_row = min(coord[0] for coord in materialized)
    min_col = min(coord[1] for coord in materialized)
    return tuple(sorted((row - min_row, col - min_col) for row, col in materialized))


S0 = Shape(((0, 0),))
S1 = Shape(((0, 0), (0, 1)))
S2 = Shape(((0, 0), (1, 0), (1, 1)))
SHAPES_ID: tuple[Shape, ...] = (S0, S1)
SHAPE_OOD: Shape = S2


def d4_transforms(shape: Shape) -> tuple[Shape, ...]:
    """Return the unique normalized members of the D4 orbit in stable order."""

    if not isinstance(shape, Shape):
        raise ValueError("shape must be a Shape")
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    seen: set[Shape] = set()
    result: list[Shape] = []
    for transform in transforms:
        candidate = Shape(_normalize_cells(transform(row, col) for row, col in shape.cells))
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return tuple(result)


@dataclass(frozen=True, order=True)
class RuleProgram:
    """One of the 64 product programs in the protocol-defined order."""

    collision: Collision
    trigger: Trigger
    relation: Relation

    def __post_init__(self) -> None:
        if not isinstance(self.collision, Collision):
            raise ValueError("collision must be a Collision enum member")
        if not isinstance(self.trigger, Trigger):
            raise ValueError("trigger must be a Trigger enum member")
        if not isinstance(self.relation, Relation):
            raise ValueError("relation must be a Relation enum member")

    @property
    def program_id(self) -> int:
        return (
            16 * ALL_COLLISIONS.index(self.collision)
            + 4 * ALL_TRIGGERS.index(self.trigger)
            + ALL_RELATIONS.index(self.relation)
        )

    @classmethod
    def from_program_id(cls, program_id: int) -> "RuleProgram":
        if type(program_id) is not int or not 0 <= program_id < 64:
            raise ValueError("program_id must be an integer in 0..63")
        collision_id, remainder = divmod(program_id, 16)
        trigger_id, relation_id = divmod(remainder, 4)
        return cls(
            ALL_COLLISIONS[collision_id],
            ALL_TRIGGERS[trigger_id],
            ALL_RELATIONS[relation_id],
        )

    def mode_for_axis(self, axis: Axis) -> Enum:
        if axis is Axis.COLLISION:
            return self.collision
        if axis is Axis.TRIGGER:
            return self.trigger
        if axis is Axis.RELATION:
            return self.relation
        raise ValueError(f"unknown axis: {axis!r}")

    def replace_axis(self, axis: Axis, mode: Enum) -> "RuleProgram":
        if axis is Axis.COLLISION and isinstance(mode, Collision):
            return RuleProgram(mode, self.trigger, self.relation)
        if axis is Axis.TRIGGER and isinstance(mode, Trigger):
            return RuleProgram(self.collision, mode, self.relation)
        if axis is Axis.RELATION and isinstance(mode, Relation):
            return RuleProgram(self.collision, self.trigger, mode)
        raise ValueError(f"mode {mode!r} is incompatible with axis {axis.value}")


ALL_PROGRAMS: tuple[RuleProgram, ...] = tuple(
    RuleProgram(collision, trigger, relation)
    for collision in ALL_COLLISIONS
    for trigger in ALL_TRIGGERS
    for relation in ALL_RELATIONS
)


def modes_for_axis(axis: Axis) -> tuple[Enum, ...]:
    if axis is Axis.COLLISION:
        return ALL_COLLISIONS
    if axis is Axis.TRIGGER:
        return ALL_TRIGGERS
    if axis is Axis.RELATION:
        return ALL_RELATIONS
    raise ValueError(f"unknown axis: {axis!r}")


@dataclass(frozen=True)
class Palette:
    """Per-task color role permutation.  Values must be unique non-backgrounds."""

    actor: int = 1
    blocker: int = 2
    object_a: int = 3
    object_b: int = 4
    trigger: int = 5
    payload_p0: int = 6
    payload_p1: int = 7
    payload_p2: int = 8
    socket: int = 9
    pulse_d0: int = 10
    pulse_d1: int = 11
    distractor: int = 12

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if len(set(values)) != len(values):
            raise ValueError("all non-background palette roles must have distinct colors")
        for name, value in self.__dict__.items():
            if type(value) is not int or not 1 <= value < NUM_COLORS:
                raise ValueError(f"palette role {name} must be an integer in 1..15")

    @property
    def role_values(self) -> tuple[int, ...]:
        return tuple(self.__dict__.values())


DEFAULT_PALETTE = Palette()


def palette_from_seed(seed: int) -> Palette:
    """Create a deterministic palette permutation without consulting a program."""

    if type(seed) is not int:
        raise ValueError("palette seed must be an integer")
    colors = list(range(1, NUM_COLORS))
    random.Random(seed).shuffle(colors)
    return Palette(*colors[:12])


@dataclass(frozen=True, order=True)
class GridAction:
    """A public atomic action: only type, coordinate, and optional direction."""

    kind: ActionKind
    coord: Coord
    direction: Direction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise ValueError("action kind must be an ActionKind")
        if (
            not isinstance(self.coord, tuple)
            or len(self.coord) != 2
            or any(type(value) is not int for value in self.coord)
            or not in_bounds(self.coord)
        ):
            raise ValueError("action coordinate must be an in-bounds integer (row, column)")
        if self.kind is ActionKind.MOVE and not isinstance(self.direction, Direction):
            raise ValueError("MOVE requires a direction")
        if self.kind is ActionKind.ACTIVATE and self.direction is not None:
            raise ValueError("ACTIVATE must not carry a direction")

    @property
    def canonical_id(self) -> str:
        direction = "-" if self.direction is None else self.direction.value
        return f"{self.kind.value}:{self.coord[0]}:{self.coord[1]}:{direction}"


@dataclass(frozen=True)
class CompositeAction:
    """A simultaneous set of non-overlapping public atomic actions.

    A diagnostic query may combine independent local mechanisms.  The action
    remains public: it is only a tuple of the same coordinate/type/direction
    tokens accepted by :class:`GridAction`.  The simulator evaluates every
    local event against the original frame and rejects overlapping write sets.
    """

    actions: tuple[GridAction, ...]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("a composite action must contain at least one atomic action")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("a composite action cannot repeat an atomic action")

    @property
    def canonical_id(self) -> str:
        return "+".join(action.canonical_id for action in self.actions)


AnyAction: TypeAlias = GridAction | CompositeAction


def atomic_actions(action: AnyAction) -> tuple[GridAction, ...]:
    if isinstance(action, GridAction):
        return (action,)
    if isinstance(action, CompositeAction):
        return action.actions
    raise ValueError(f"unsupported action: {action!r}")


def action_to_jsonable(action: AnyAction) -> dict[str, object]:
    if isinstance(action, GridAction):
        return {
            "type": action.kind.value,
            "coord": [action.coord[0], action.coord[1]],
            "direction": None if action.direction is None else action.direction.value,
        }
    return {
        "type": "COMPOSITE",
        "actions": [action_to_jsonable(item) for item in action.actions],
    }


def action_bytes(action: AnyAction) -> bytes:
    return json.dumps(
        action_to_jsonable(action), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class OverlappingWritesError(ValueError):
    """Raised instead of assigning an arbitrary order to overlapping events."""


@dataclass(frozen=True)
class _Delta:
    writes: tuple[tuple[Coord, int], ...]

    @property
    def write_coords(self) -> frozenset[Coord]:
        return frozenset(coord for coord, _ in self.writes)


def _component(grid: Grid, start: Coord, color: int) -> frozenset[Coord]:
    """Return the 4-connected component of ``color`` containing ``start``."""

    if not in_bounds(start) or grid[start[0]][start[1]] != color:
        return frozenset()
    pending = [start]
    visited: set[Coord] = set()
    while pending:
        coord = pending.pop()
        if coord in visited:
            continue
        if not in_bounds(coord) or grid[coord[0]][coord[1]] != color:
            continue
        visited.add(coord)
        for vector in _DIRECTION_VECTOR.values():
            pending.append(add_coord(coord, vector))
    return frozenset(visited)


def _translated(cells: Iterable[Coord], vector: Coord, scale: int = 1) -> frozenset[Coord]:
    return frozenset(add_coord(coord, vector, scale) for coord in cells)


def _write_delta(before: Mapping[Coord, int], after: Mapping[Coord, int]) -> _Delta:
    keys = set(before) | set(after)
    writes = tuple(
        sorted(
            (coord, after.get(coord, 0))
            for coord in keys
            if before.get(coord, 0) != after.get(coord, 0)
        )
    )
    return _Delta(writes)


def _all_allowed(
    grid: Grid,
    cells: Iterable[Coord],
    allowed_occupied: frozenset[Coord],
) -> bool:
    for coord in cells:
        if not in_bounds(coord):
            return False
        if grid[coord[0]][coord[1]] != 0 and coord not in allowed_occupied:
            return False
    return True


def _collision_delta(grid: Grid, action: GridAction, rule: Collision, palette: Palette) -> _Delta:
    if action.kind is not ActionKind.MOVE or action.direction is None:
        return _Delta(())
    actor = _component(grid, action.coord, palette.actor)
    if not actor:
        return _Delta(())
    vector = direction_vector(action.direction)
    first_target = _translated(actor, vector)
    # A collision is present only when the actor's next footprint is entirely
    # occupied by one contiguous blocker component.  This makes malformed or
    # partial layouts a defined no-op rather than a partial write.
    if not first_target or any(
        not in_bounds(coord) or grid[coord[0]][coord[1]] != palette.blocker
        for coord in first_target
    ):
        return _Delta(())
    blocker_anchor = next(iter(first_target))
    blocker = _component(grid, blocker_anchor, palette.blocker)
    if blocker != first_target:
        return _Delta(())
    original = {coord: grid[coord[0]][coord[1]] for coord in actor | blocker}

    if rule is Collision.STOP:
        return _Delta(())
    if rule is Collision.BOUNCE:
        target_actor = _translated(actor, vector, -1)
        if not _all_allowed(grid, target_actor, actor):
            return _Delta(())
        after = dict(original)
        for coord in actor:
            after[coord] = 0
        for coord in target_actor:
            after[coord] = palette.actor
        return _write_delta(original, after)
    if rule is Collision.PASS:
        target_actor = _translated(actor, vector, 2)
        if not _all_allowed(grid, target_actor, actor):
            return _Delta(())
        after = dict(original)
        for coord in actor:
            after[coord] = 0
        for coord in target_actor:
            after[coord] = palette.actor
        return _write_delta(original, after)
    if rule is Collision.PUSH:
        target_blocker = _translated(blocker, vector)
        if not _all_allowed(grid, target_blocker, actor | blocker):
            return _Delta(())
        # The standard local layout has equal actor/blocker footprints.  Do
        # not silently execute a malformed non-rigid push with unmatched shapes.
        if blocker != first_target:
            return _Delta(())
        after = dict(original)
        for coord in actor | blocker:
            after[coord] = 0
        for coord in blocker:
            after[coord] = palette.actor
        for coord in target_blocker:
            after[coord] = palette.blocker
        return _write_delta(original, after)
    raise AssertionError(f"unhandled collision rule: {rule!r}")


def _trigger_delta(grid: Grid, action: GridAction, rule: Trigger, palette: Palette) -> _Delta:
    if action.kind is not ActionKind.ACTIVATE:
        return _Delta(())
    trigger_coord = action.coord
    if grid[trigger_coord[0]][trigger_coord[1]] != palette.trigger:
        return _Delta(())
    # Trigger layouts have a fixed, public eastward local convention:
    # trigger | payload | socket.  The trigger itself is never written.
    payload_coord = add_coord(trigger_coord, (0, 1))
    socket_coord = add_coord(trigger_coord, (0, 2))
    if not in_bounds(payload_coord) or not in_bounds(socket_coord):
        return _Delta(())
    payload = grid[payload_coord[0]][payload_coord[1]]
    socket = grid[socket_coord[0]][socket_coord[1]]
    before = {
        payload_coord: payload,
        socket_coord: socket,
    }
    after = dict(before)
    if rule is Trigger.TOGGLE:
        if payload == palette.payload_p0:
            after[payload_coord] = palette.payload_p1
        elif payload == palette.payload_p1:
            after[payload_coord] = palette.payload_p0
        else:
            return _Delta(())
    elif rule is Trigger.DELETE:
        if payload != palette.payload_p0:
            return _Delta(())
        after[payload_coord] = 0
    elif rule is Trigger.SPAWN:
        if payload != palette.payload_p0 or socket != palette.socket:
            return _Delta(())
        after[socket_coord] = palette.payload_p0
    elif rule is Trigger.RECOLOR:
        if payload != palette.payload_p0:
            return _Delta(())
        after[payload_coord] = palette.payload_p2
    else:
        raise AssertionError(f"unhandled trigger rule: {rule!r}")
    return _write_delta(before, after)


def _relation_delta(grid: Grid, action: GridAction, rule: Relation, palette: Palette) -> _Delta:
    if action.kind is not ActionKind.MOVE or action.direction is None:
        return _Delta(())
    object_a = _component(grid, action.coord, palette.object_a)
    if not object_a:
        return _Delta(())
    vector = direction_vector(action.direction)
    b_footprint = _translated(object_a, vector)
    if not b_footprint or any(
        not in_bounds(coord) or grid[coord[0]][coord[1]] != palette.object_b
        for coord in b_footprint
    ):
        return _Delta(())
    object_b = _component(grid, next(iter(b_footprint)), palette.object_b)
    if object_b != b_footprint:
        return _Delta(())
    original = {coord: grid[coord[0]][coord[1]] for coord in object_a | object_b}
    if rule is Relation.NONE:
        return _Delta(())
    if rule is Relation.SWAP:
        after = dict(original)
        for coord in object_a:
            after[coord] = palette.object_b
        for coord in object_b:
            after[coord] = palette.object_a
        return _write_delta(original, after)
    target_b = _translated(object_b, vector)
    if not _all_allowed(grid, target_b, object_a | object_b):
        return _Delta(())
    after = dict(original)
    if rule is Relation.FOLLOW:
        for coord in object_a | object_b:
            after[coord] = 0
        for coord in object_b:
            after[coord] = palette.object_a
        for coord in target_b:
            after[coord] = palette.object_b
        return _write_delta(original, after)
    if rule is Relation.REPEL:
        for coord in object_b:
            after[coord] = 0
        for coord in target_b:
            after[coord] = palette.object_b
        return _write_delta(original, after)
    raise AssertionError(f"unhandled relation rule: {rule!r}")


def _atomic_delta(grid: Grid, action: GridAction, program: RuleProgram, palette: Palette) -> _Delta:
    """Evaluate all semantics compatible with an atomic action.

    A MOVE can be a collision event *or* a relation event according to the
    visible color at its public coordinate.  Distinct role colors make these
    cases mutually exclusive in a valid local layout.
    """

    if action.kind is ActionKind.ACTIVATE:
        return _trigger_delta(grid, action, program.trigger, palette)
    collision = _collision_delta(grid, action, program.collision, palette)
    relation = _relation_delta(grid, action, program.relation, palette)
    if collision.writes and relation.writes:
        raise OverlappingWritesError("one atomic MOVE matched two local rule events")
    return collision if collision.writes else relation


def _pulse_delta(grid: Grid, palette: Palette) -> _Delta:
    before: dict[Coord, int] = {}
    after: dict[Coord, int] = {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = grid[row][col]
            if value == palette.pulse_d0:
                before[(row, col)] = value
                after[(row, col)] = palette.pulse_d1
            elif value == palette.pulse_d1:
                before[(row, col)] = value
                after[(row, col)] = palette.pulse_d0
    return _write_delta(before, after)


@lru_cache(maxsize=262_144)
def simulate(grid: Grid, action: AnyAction, program: RuleProgram, palette: Palette = DEFAULT_PALETTE) -> Grid:
    """Apply one deterministic RuleGrid transition.

    Every local event is evaluated on the same pre-transition grid.  Its write
    set is collected first, then all disjoint writes are committed together.
    This is intentionally stricter than a sequential renderer: overlapping
    layouts are invalid rather than resolved by a hidden execution order.
    """

    validate_grid(grid)
    if not isinstance(program, RuleProgram):
        raise ValueError("program must be a RuleProgram")
    if not isinstance(palette, Palette):
        raise ValueError("palette must be a Palette")
    deltas = [_atomic_delta(grid, item, program, palette) for item in atomic_actions(action)]
    pulse = _pulse_delta(grid, palette)
    if pulse.writes:
        deltas.append(pulse)
    writes: dict[Coord, int] = {}
    for delta in deltas:
        for coord, value in delta.writes:
            if coord in writes:
                raise OverlappingWritesError(
                    f"independent events overlap at {coord!r}; layout is invalid"
                )
            writes[coord] = value
    rows = [list(row) for row in grid]
    for (row, col), value in writes.items():
        rows[row][col] = value
    return tuple(tuple(row) for row in rows)


def count_changed_cells(before: Grid, after: Grid) -> int:
    validate_grid(before)
    validate_grid(after)
    return sum(
        before[row][col] != after[row][col]
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    )


@dataclass(frozen=True)
class RuleGridTransition:
    state: Grid
    action: AnyAction
    next_state: Grid

    def __post_init__(self) -> None:
        validate_grid(self.state)
        validate_grid(self.next_state)
        atomic_actions(self.action)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "state": grid_to_jsonable(self.state),
            "action": action_to_jsonable(self.action),
            "next_state": grid_to_jsonable(self.next_state),
        }


@dataclass(frozen=True)
class RuleGridProbe:
    """A public counterfactual input.  Its true target is deliberately absent."""

    probe_id: str
    state: Grid
    action: AnyAction

    def __post_init__(self) -> None:
        if not self.probe_id or not isinstance(self.probe_id, str):
            raise ValueError("probe_id must be a non-empty string")
        validate_grid(self.state)
        atomic_actions(self.action)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "state": grid_to_jsonable(self.state),
            "action": action_to_jsonable(self.action),
        }


@dataclass(frozen=True)
class RuleGridInferenceView:
    """Controller-safe task data, with all target/oracle fields excluded."""

    task_id: str
    support: tuple[RuleGridTransition, ...]
    active_candidates: tuple[RuleGridProbe, ...]
    diagnostics: tuple[RuleGridProbe, ...]

    def __getattr__(self, name: str) -> object:
        if name in {
            "program",
            "program_id",
            "true_rule",
            "probe_kind",
            "candidate_kind",
            "target",
            "targets",
            "oracle_eig",
            "version_space",
        }:
            raise PermissionError(f"{name} is privileged simulator-only data")
        raise AttributeError(name)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "support": [item.to_jsonable() for item in self.support],
            "active_candidates": [item.to_jsonable() for item in self.active_candidates],
            "diagnostics": [item.to_jsonable() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class RuleGridPrivilegedTargets:
    """Simulator/training-only sidecar.  Never pass this to a controller.

    ``diagnostic_targets`` may be a strict ordered subset of the public
    diagnostic panel.  ``diagnostic_target_indices`` maps its entries back to
    their public panel indices.  The default ``None`` preserves the original
    API: a caller passing a full target tuple gets the positional mapping
    ``0..len(diagnostic_targets)-1`` automatically.

    This explicit mapping lets a training run materialize only its permitted
    privileged targets instead of eagerly simulating an evaluation holdout.
    """

    true_program: RuleProgram
    palette: Palette
    candidate_kinds: tuple[str, ...]
    active_targets: tuple[Grid, ...]
    diagnostic_targets: tuple[Grid, ...]
    diagnostic_target_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostic_targets, tuple):
            raise TypeError("diagnostic_targets must be a tuple")
        indices = self.diagnostic_target_indices
        if indices is None:
            indices = tuple(range(len(self.diagnostic_targets)))
            object.__setattr__(self, "diagnostic_target_indices", indices)
        if not isinstance(indices, tuple):
            raise TypeError("diagnostic_target_indices must be a tuple or None")
        if len(indices) != len(self.diagnostic_targets):
            raise ValueError(
                "diagnostic_target_indices must have one entry per diagnostic target"
            )
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("diagnostic_target_indices must contain non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("diagnostic_target_indices must not contain duplicates")

    def diagnostic_target_for(self, diagnostic_index: int) -> Grid:
        """Return one materialized target, rejecting an unavailable index."""

        if type(diagnostic_index) is not int or diagnostic_index < 0:
            raise ValueError("diagnostic_index must be a non-negative integer")
        assert self.diagnostic_target_indices is not None
        try:
            target_position = self.diagnostic_target_indices.index(diagnostic_index)
        except ValueError as error:
            raise ValueError(
                f"diagnostic target {diagnostic_index} was not materialized for this task"
            ) from error
        return self.diagnostic_targets[target_position]


@dataclass(frozen=True)
class RuleGridTask:
    inference: RuleGridInferenceView
    privileged: RuleGridPrivilegedTargets

    def __post_init__(self) -> None:
        assert self.privileged.diagnostic_target_indices is not None
        if any(
            index >= len(self.inference.diagnostics)
            for index in self.privileged.diagnostic_target_indices
        ):
            raise ValueError("diagnostic target index is outside the public panel")


def transition_is_consistent(
    transition: RuleGridTransition, program: RuleProgram, palette: Palette
) -> bool:
    return simulate(transition.state, transition.action, program, palette) == transition.next_state


def version_space(
    history: Iterable[RuleGridTransition], palette: Palette = DEFAULT_PALETTE
) -> tuple[RuleProgram, ...]:
    """Enumerate all of the 64 programs consistent with observed transitions."""

    transitions = tuple(history)
    return tuple(
        program
        for program in ALL_PROGRAMS
        if all(transition_is_consistent(item, program, palette) for item in transitions)
    )


def expected_heldout_version_space(program: RuleProgram, axis: Axis) -> tuple[RuleProgram, ...]:
    """The four variants left after strong evidence on the other two axes."""

    return tuple(program.replace_axis(axis, mode) for mode in modes_for_axis(axis))


def outcome_partition(
    programs: Iterable[RuleProgram], probe: RuleGridProbe, palette: Palette
) -> dict[bytes, tuple[RuleProgram, ...]]:
    buckets: dict[bytes, list[RuleProgram]] = {}
    for program in programs:
        outcome = simulate(probe.state, probe.action, program, palette)
        buckets.setdefault(grid_bytes(outcome), []).append(program)
    return {outcome: tuple(bucket) for outcome, bucket in buckets.items()}


def partition_sizes(
    programs: Iterable[RuleProgram], probe: RuleGridProbe, palette: Palette
) -> tuple[int, ...]:
    return tuple(sorted(len(bucket) for bucket in outcome_partition(programs, probe, palette).values()))


def behavior_signature(
    program: RuleProgram,
    diagnostics: Iterable[RuleGridProbe],
    palette: Palette,
) -> tuple[bytes, ...]:
    return tuple(
        grid_bytes(simulate(probe.state, probe.action, program, palette))
        for probe in diagnostics
    )


def behavior_classes(
    programs: Iterable[RuleProgram],
    diagnostics: Iterable[RuleGridProbe],
    palette: Palette,
) -> dict[tuple[bytes, ...], tuple[RuleProgram, ...]]:
    diagnostics_tuple = tuple(diagnostics)
    classes: dict[tuple[bytes, ...], list[RuleProgram]] = {}
    for program in programs:
        signature = behavior_signature(program, diagnostics_tuple, palette)
        classes.setdefault(signature, []).append(program)
    return {signature: tuple(members) for signature, members in classes.items()}


def behavior_identified(
    programs: Iterable[RuleProgram], diagnostics: Iterable[RuleGridProbe], palette: Palette
) -> bool:
    return len(behavior_classes(tuple(programs), tuple(diagnostics), palette)) <= 1


def _entropy_bits(group_sizes: Iterable[int]) -> float:
    sizes = tuple(size for size in group_sizes if size)
    total = sum(sizes)
    if total == 0:
        raise ValueError("cannot calculate entropy of an empty version space")
    return -sum((size / total) * math.log2(size / total) for size in sizes)


def exact_behavior_eig(
    programs: Iterable[RuleProgram],
    probe: RuleGridProbe,
    diagnostics: Iterable[RuleGridProbe],
    palette: Palette,
) -> float:
    """Exact behavior-class information gain in bits under uniform programs."""

    programs_tuple = tuple(programs)
    if not programs_tuple:
        raise ValueError("version space cannot be empty")
    diagnostics_tuple = tuple(diagnostics)
    prior = behavior_classes(programs_tuple, diagnostics_tuple, palette)
    prior_entropy = _entropy_bits(len(members) for members in prior.values())
    outcomes = outcome_partition(programs_tuple, probe, palette)
    expected_entropy = 0.0
    for members in outcomes.values():
        probability = len(members) / len(programs_tuple)
        posterior = behavior_classes(members, diagnostics_tuple, palette)
        expected_entropy += probability * _entropy_bits(len(group) for group in posterior.values())
    return max(0.0, prior_entropy - expected_entropy)


@dataclass(frozen=True)
class OracleProbeChoice:
    probe_id: str
    eig_bits: float


def select_exact_oracle_probe(
    programs: Iterable[RuleProgram],
    candidates: Iterable[RuleGridProbe],
    diagnostics: Iterable[RuleGridProbe],
    palette: Palette,
) -> OracleProbeChoice:
    """Select maximum behavior EIG, breaking ties by public candidate ID."""

    candidates_tuple = tuple(candidates)
    if not candidates_tuple:
        raise ValueError("at least one candidate is required")
    scored = tuple(
        OracleProbeChoice(
            probe.probe_id,
            exact_behavior_eig(programs, probe, diagnostics, palette),
        )
        for probe in candidates_tuple
    )
    return min(scored, key=lambda item: (-item.eig_bits, item.probe_id))


def derive_seed64(
    split: str, heldout_axis: Axis, replicate: int, stream_name: str,
    *, master_seed: int = MASTER_SEED,
) -> int:
    """Protocol seed derivation; notably no program ID appears in the input."""

    if stream_name not in {
        "palette",
        "geometry",
        "support_order",
        "candidate_order",
        "diagnostic",
        "rollout",
    }:
        raise ValueError(f"unsupported nuisance stream: {stream_name!r}")
    if type(replicate) is not int or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    source = (
        f"{BENCHMARK_VERSION}|{master_seed}|{split}|{heldout_axis.value}|"
        f"{replicate}|{stream_name}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "little")


def task_id(split: str, program: RuleProgram, heldout_axis: Axis, replicate: int) -> str:
    if not split:
        raise ValueError("split cannot be empty")
    if type(replicate) is not int or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    axis_id = ALL_AXES.index(heldout_axis)
    return f"{split}/P{program.program_id:02d}/H{axis_id}/N{replicate:04d}"


def _set_cells(target: dict[Coord, int], additions: Mapping[Coord, int]) -> None:
    for coord, color in additions.items():
        if not in_bounds(coord):
            raise ValueError(f"fixture requires out-of-bounds cell {coord!r}")
        if coord in target:
            raise ValueError(f"fixture cells overlap at {coord!r}")
        target[coord] = color


def _collision_probe(probe_id: str, palette: Palette, *, row: int, col: int, direction: Direction = Direction.EAST) -> RuleGridProbe:
    vector = direction_vector(direction)
    actor = (row, col)
    blocker = add_coord(actor, vector)
    cells: dict[Coord, int] = {}
    _set_cells(cells, {actor: palette.actor, blocker: palette.blocker})
    return RuleGridProbe(probe_id, grid_with_cells(cells), GridAction(ActionKind.MOVE, actor, direction))


def _trigger_probe(probe_id: str, palette: Palette, *, row: int, col: int, payload: int | None = None, socket: int | None = None) -> RuleGridProbe:
    payload_value = palette.payload_p0 if payload is None else payload
    socket_value = palette.socket if socket is None else socket
    trigger_coord = (row, col)
    cells: dict[Coord, int] = {}
    _set_cells(
        cells,
        {
            trigger_coord: palette.trigger,
            (row, col + 1): payload_value,
            (row, col + 2): socket_value,
        },
    )
    return RuleGridProbe(probe_id, grid_with_cells(cells), GridAction(ActionKind.ACTIVATE, trigger_coord))


def _relation_probe(probe_id: str, palette: Palette, *, row: int, col: int, direction: Direction = Direction.EAST, block_destination: bool = False) -> RuleGridProbe:
    vector = direction_vector(direction)
    object_a = (row, col)
    object_b = add_coord(object_a, vector)
    cells: dict[Coord, int] = {}
    _set_cells(cells, {object_a: palette.object_a, object_b: palette.object_b})
    if block_destination:
        _set_cells(cells, {add_coord(object_b, vector): palette.distractor})
    return RuleGridProbe(probe_id, grid_with_cells(cells), GridAction(ActionKind.MOVE, object_a, direction))


def _pulse_grid(palette: Palette, *, row: int = 2, col: int = 2) -> Grid:
    return grid_with_cells(
        {
            (row, col): palette.pulse_d0,
            (row, col + 1): palette.pulse_d1,
            (row + 1, col): palette.pulse_d1,
            (row + 1, col + 1): palette.pulse_d0,
        }
    )


def _neutral_probe(probe_id: str, palette: Palette, *, row: int, col: int) -> RuleGridProbe:
    # No actor/trigger/A exists, so MOVE is a rule-independent no-op.  The
    # visible 2x2 pulse makes this deliberately a large-change distractor.
    return RuleGridProbe(
        probe_id,
        _pulse_grid(palette, row=row, col=col),
        GridAction(ActionKind.MOVE, (0, 0), Direction.EAST),
    )


def _partial_probe(axis: Axis, probe_id: str, palette: Palette, *, variant: int) -> RuleGridProbe:
    """Construct a 1+3 partition using legal no-op preconditions.

    The selected effectful mode is BOUNCE / TOGGLE / SWAP respectively.  The
    public layout is independent of the true mode; the private ``partial``
    label is attached only by the task generator.
    """

    if axis is Axis.COLLISION:
        row = 2 + variant
        # p-d is free for BOUNCE; p+2d is occupied, so PASS and PUSH must make
        # no partial writes.  STOP is intrinsically a no-op.
        cells = {
            (row, 1): palette.actor,
            (row, 2): palette.blocker,
            (row, 3): palette.distractor,
        }
        return RuleGridProbe(
            probe_id,
            grid_with_cells(cells),
            GridAction(ActionKind.MOVE, (row, 1), Direction.EAST),
        )
    if axis is Axis.TRIGGER:
        row = 2 + variant
        # P1 is valid only for TOGGLE under the canonical trigger convention.
        return _trigger_probe(
            probe_id,
            palette,
            row=row,
            col=2,
            payload=palette.payload_p1,
            socket=palette.socket,
        )
    if axis is Axis.RELATION:
        return _relation_probe(
            probe_id,
            palette,
            row=2 + variant,
            col=1,
            block_destination=True,
        )
    raise AssertionError(f"unhandled axis: {axis!r}")


def _overlay_probes(probe_id: str, probes: Sequence[RuleGridProbe]) -> RuleGridProbe:
    """Combine disjoint fixture grids and atomic actions into a diagnostic query."""

    cells: dict[Coord, int] = {}
    actions: list[GridAction] = []
    for probe in probes:
        if isinstance(probe.action, CompositeAction):
            raise ValueError("diagnostic fixture must start from atomic probes")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = probe.state[row][col]
                if value:
                    _set_cells(cells, {(row, col): value})
        actions.append(probe.action)
    action: AnyAction = actions[0] if len(actions) == 1 else CompositeAction(tuple(actions))
    return RuleGridProbe(probe_id, grid_with_cells(cells), action)


@lru_cache(maxsize=256)
def _diagnostic_panel(palette: Palette) -> tuple[RuleGridProbe, ...]:
    """Build the protocol's canonical 12 single / 9 pair / 3 triple panel."""

    singles: list[RuleGridProbe] = []
    # Each quartet is geometrically distinct but has the same causal purpose:
    # one fully discriminating event for that axis.
    collision_specs = ((1, 1, Direction.EAST), (1, 5, Direction.WEST), (2, 1, Direction.EAST), (2, 5, Direction.WEST))
    trigger_specs = ((3, 1), (3, 4), (4, 1), (4, 4))
    relation_specs = ((5, 1, Direction.EAST), (5, 5, Direction.WEST), (6, 1, Direction.EAST), (6, 5, Direction.WEST))
    for index, (row, col, direction) in enumerate(collision_specs):
        singles.append(_collision_probe(f"D{index:02d}", palette, row=row, col=col, direction=direction))
    for index, (row, col) in enumerate(trigger_specs, start=4):
        singles.append(_trigger_probe(f"D{index:02d}", palette, row=row, col=col))
    for index, (row, col, direction) in enumerate(relation_specs, start=8):
        singles.append(_relation_probe(f"D{index:02d}", palette, row=row, col=col, direction=direction))

    # Pair/triple probes use dedicated non-overlapping rows, preserving the
    # same public action semantics.  The panel order is fixed by the protocol.
    def c(row: int) -> RuleGridProbe:
        return _collision_probe("fixture-c", palette, row=row, col=1)

    def t(row: int) -> RuleGridProbe:
        return _trigger_probe("fixture-t", palette, row=row, col=3)

    def r(row: int) -> RuleGridProbe:
        return _relation_probe("fixture-r", palette, row=row, col=5, direction=Direction.WEST)

    pairs: list[RuleGridProbe] = []
    for index in range(3):
        pairs.append(_overlay_probes(f"D{12 + index:02d}", (c(1), t(3))))
    for index in range(3):
        pairs.append(_overlay_probes(f"D{15 + index:02d}", (c(1), r(5))))
    for index in range(3):
        pairs.append(_overlay_probes(f"D{18 + index:02d}", (t(3), r(5))))
    triples = tuple(
        _overlay_probes(f"D{21 + index:02d}", (c(1), t(3), r(5)))
        for index in range(3)
    )
    panel = tuple(singles + pairs) + triples
    if len(panel) != 24:
        raise AssertionError("diagnostic panel must contain exactly 24 probes")
    return panel


def _normalize_diagnostic_target_indices(
    diagnostic_indices: Sequence[int] | None, diagnostic_count: int
) -> tuple[int, ...]:
    """Validate an explicit target-materialization subset for one task."""

    if diagnostic_count <= 0:
        raise ValueError("diagnostic_count must be positive")
    if diagnostic_indices is None:
        return tuple(range(diagnostic_count))
    indices = tuple(diagnostic_indices)
    if not indices:
        raise ValueError("diagnostic_indices cannot be empty")
    if any(type(index) is not int for index in indices):
        raise TypeError("diagnostic_indices must contain plain integers")
    if len(set(indices)) != len(indices):
        raise ValueError("diagnostic_indices cannot contain duplicates")
    if any(index < 0 or index >= diagnostic_count for index in indices):
        raise ValueError(
            f"diagnostic_indices must lie in [0, {diagnostic_count}), got {indices!r}"
        )
    return indices


def make_rulegrid_task(
    program: RuleProgram,
    heldout_axis: Axis,
    replicate: int,
    *,
    split: str = "gate0b",
    master_seed: int = MASTER_SEED,
    diagnostic_indices: Sequence[int] | None = None,
) -> RuleGridTask:
    """Generate one deterministic Stage 0-B task skeleton.

    The construction intentionally never branches on ``program`` until target
    frames are simulated.  Therefore all nuisance layout/palette/order choices
    are structurally independent of hidden rule modes.  By default all 24
    privileged diagnostic targets are materialized, preserving the historical
    API.  Passing ``diagnostic_indices`` instead simulates only that exact
    ordered subset; this is required for a clean composition holdout.
    """

    if not isinstance(program, RuleProgram):
        raise ValueError("program must be a RuleProgram")
    if not isinstance(heldout_axis, Axis):
        raise ValueError("heldout_axis must be an Axis")
    palette = palette_from_seed(
        derive_seed64(split, heldout_axis, replicate, "palette", master_seed=master_seed)
    )
    geometry_rng = random.Random(
        derive_seed64(split, heldout_axis, replicate, "geometry", master_seed=master_seed)
    )

    calibration: list[RuleGridProbe] = []
    if heldout_axis is not Axis.COLLISION:
        calibration.append(_collision_probe("S00", palette, row=1, col=2))
    if heldout_axis is not Axis.TRIGGER:
        calibration.append(_trigger_probe("S01", palette, row=3, col=2))
    if heldout_axis is not Axis.RELATION:
        calibration.append(_relation_probe("S02", palette, row=5, col=2))
    if len(calibration) != 2:
        raise AssertionError("exactly two non-heldout calibration probes are required")

    # Four rule-independent support transitions.  Their private labels retain
    # the C/T/R/heldout provenance without leaking it into public inputs.
    neutral_support = [
        _neutral_probe(f"S{2 + index:02d}", palette, row=1 + (index % 2) * 3, col=1 + (index // 2) * 3)
        for index in range(4)
    ]
    order_rng = random.Random(
        derive_seed64(split, heldout_axis, replicate, "support_order", master_seed=master_seed)
    )
    order_rng.shuffle(neutral_support)
    support_probes = tuple(calibration + neutral_support)
    support = tuple(
        RuleGridTransition(probe.state, probe.action, simulate(probe.state, probe.action, program, palette))
        for probe in support_probes
    )

    candidates: list[tuple[str, RuleGridProbe]] = []
    if heldout_axis is Axis.COLLISION:
        candidates.extend(
            (
                ("strong", _collision_probe("A00", palette, row=2, col=2)),
                ("strong", _collision_probe("A01", palette, row=5, col=5, direction=Direction.WEST)),
            )
        )
    elif heldout_axis is Axis.TRIGGER:
        candidates.extend(
            (
                ("strong", _trigger_probe("A00", palette, row=2, col=2)),
                ("strong", _trigger_probe("A01", palette, row=5, col=2)),
            )
        )
    else:
        candidates.extend(
            (
                ("strong", _relation_probe("A00", palette, row=2, col=2)),
                ("strong", _relation_probe("A01", palette, row=5, col=5, direction=Direction.WEST)),
            )
        )
    candidates.extend(
        ("partial", _partial_probe(heldout_axis, f"A{2 + index:02d}", palette, variant=index))
        for index in range(2)
    )
    candidates.extend(
        ("neutral-large-change", _neutral_probe(f"A{4 + index:02d}", palette, row=2 + (index % 2), col=1 + 3 * (index // 2)))
        for index in range(4)
    )
    candidate_rng = random.Random(
        derive_seed64(split, heldout_axis, replicate, "candidate_order", master_seed=master_seed)
    )
    candidate_rng.shuffle(candidates)

    # Keep a geometry stream draw so future fixture variants can be extended
    # without ever using program ID.  It currently only fixes a deterministic
    # public ordering-neutral metadata-free branch.
    _ = geometry_rng.randrange(8)
    diagnostics = _diagnostic_panel(palette)
    selected_diagnostic_indices = _normalize_diagnostic_target_indices(
        diagnostic_indices, len(diagnostics)
    )
    active_candidates = tuple(item[1] for item in candidates)
    candidate_kinds = tuple(item[0] for item in candidates)
    active_targets = tuple(
        simulate(probe.state, probe.action, program, palette) for probe in active_candidates
    )
    diagnostic_targets = tuple(
        simulate(diagnostics[index].state, diagnostics[index].action, program, palette)
        for index in selected_diagnostic_indices
    )
    return RuleGridTask(
        inference=RuleGridInferenceView(
            task_id=task_id(split, program, heldout_axis, replicate),
            support=support,
            active_candidates=active_candidates,
            diagnostics=diagnostics,
        ),
        privileged=RuleGridPrivilegedTargets(
            true_program=program,
            palette=palette,
            candidate_kinds=candidate_kinds,
            active_targets=active_targets,
            diagnostic_targets=diagnostic_targets,
            diagnostic_target_indices=selected_diagnostic_indices,
        ),
    )


def inference_sha256(task: RuleGridTask) -> str:
    payload = json.dumps(
        task.inference.to_jsonable(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_public_json(task: RuleGridTask) -> str:
    return json.dumps(task.inference.to_jsonable(), sort_keys=True, separators=(",", ":"))


def task_privileged_json(task: RuleGridTask) -> str:
    """Serialize sidecar targets for materialization, never for controllers."""

    payload = {
        "task_id": task.inference.task_id,
        "true_program_id": task.privileged.true_program.program_id,
        "palette": dict(task.privileged.palette.__dict__),
        "candidate_kinds": list(task.privileged.candidate_kinds),
        "active_targets": [grid_to_jsonable(grid) for grid in task.privileged.active_targets],
        "diagnostic_targets": [
            grid_to_jsonable(grid) for grid in task.privileged.diagnostic_targets
        ],
    }
    assert task.privileged.diagnostic_target_indices is not None
    default_indices = tuple(range(len(task.privileged.diagnostic_targets)))
    if task.privileged.diagnostic_target_indices != default_indices:
        payload["diagnostic_target_indices"] = list(
            task.privileged.diagnostic_target_indices
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def iter_split_tasks(
    split: str,
    *,
    repeats: int,
    master_seed: int = MASTER_SEED,
) -> Iterator[RuleGridTask]:
    """Yield the full factorial split in stable program/axis/replicate order."""

    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    for program in ALL_PROGRAMS:
        for heldout_axis in ALL_AXES:
            for replicate in range(repeats):
                yield make_rulegrid_task(
                    program,
                    heldout_axis,
                    replicate,
                    split=split,
                    master_seed=master_seed,
                )
