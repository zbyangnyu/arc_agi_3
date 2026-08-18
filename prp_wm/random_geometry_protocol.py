"""Isolated randomized-geometry protocol for the privileged RuleGrid executor.

The existing RuleGrid benchmark and audited training runtime are intentionally
left untouched.  This module only consumes their public grid/action DSL and
simulator.  Training panels contain singleton or pair mechanism events;
evaluation panels contain triple events from a disjoint geometry-seed stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
from typing import Any, Iterable, Iterator, Sequence

from .rulegrid import (
    ALL_COLLISIONS,
    ALL_RELATIONS,
    ALL_TRIGGERS,
    ActionKind,
    AnyAction,
    Axis,
    CompositeAction,
    Coord,
    DEFAULT_PALETTE,
    Direction,
    Grid,
    GridAction,
    Palette,
    RuleProgram,
    action_to_jsonable,
    add_coord,
    direction_vector,
    grid_to_jsonable,
    grid_with_cells,
    in_bounds,
    simulate,
)


PROTOCOL_VERSION = "prp-wm-random-geometry-executor-v1"
AXES: tuple[Axis, ...] = (Axis.COLLISION, Axis.TRIGGER, Axis.RELATION)
TRAIN_AXIS_SETS: tuple[tuple[Axis, ...], ...] = (
    (Axis.COLLISION,),
    (Axis.TRIGGER,),
    (Axis.RELATION,),
    (Axis.COLLISION, Axis.TRIGGER),
    (Axis.COLLISION, Axis.RELATION),
    (Axis.TRIGGER, Axis.RELATION),
)
EVAL_AXIS_SETS: tuple[tuple[Axis, ...], ...] = (AXES,)
FACTOR_CODES: tuple[tuple[int, int, int], ...] = tuple(
    itertools.product(range(4), repeat=3)
)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in (PROTOCOL_VERSION, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def factor_code_to_program(code: Sequence[int]) -> RuleProgram:
    normalized = tuple(code)
    if len(normalized) != 3 or any(
        type(value) is not int or value not in range(4) for value in normalized
    ):
        raise ValueError("factor code must contain three integers in [0,4)")
    return RuleProgram(
        ALL_COLLISIONS[normalized[0]],
        ALL_TRIGGERS[normalized[1]],
        ALL_RELATIONS[normalized[2]],
    )


def _action_set_jsonable(action: AnyAction) -> list[dict[str, object]]:
    atoms = action.actions if isinstance(action, CompositeAction) else (action,)
    return [
        action_to_jsonable(atom)
        for atom in sorted(atoms, key=lambda item: item.canonical_id)
    ]


def geometry_sha256(state: Grid, action: AnyAction) -> str:
    """Hash semantic public geometry without split, seed, ID, or atom order."""

    payload = {
        "state": grid_to_jsonable(state),
        "action_set": _action_set_jsonable(action),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FixtureLayout:
    axis: Axis
    anchor: Coord
    direction: Direction | None
    shape: str
    occupied_cells: frozenset[Coord]
    safety_envelope: frozenset[Coord]
    action: GridAction
    cells: tuple[tuple[Coord, int], ...]


@dataclass(frozen=True)
class GeometryPanel:
    """One private-audit panel; only ``state`` and ``action`` are public."""

    split: str
    geometry_seed: int
    geometry_variant: int
    axes: tuple[Axis, ...]
    state: Grid
    action: AnyAction
    fixtures: tuple[FixtureLayout, ...]
    nuisance_cells: frozenset[Coord]
    geometry_sha256: str

    @property
    def panel_kind(self) -> str:
        return {1: "singleton", 2: "pair", 3: "triple"}[len(self.axes)]

    @property
    def action_atom_count(self) -> int:
        return len(self.action.actions) if isinstance(self.action, CompositeAction) else 1

    def model_inputs_jsonable(self, factor_code: Sequence[int]) -> dict[str, object]:
        """Return the exact executor inputs, excluding every audit identifier."""

        code = tuple(factor_code)
        factor_code_to_program(code)
        return {
            "state": grid_to_jsonable(self.state),
            "action": action_to_jsonable(self.action),
            "factor_code": list(code),
        }


@dataclass(frozen=True)
class ExecutorExample:
    panel: GeometryPanel
    factor_code: tuple[int, int, int]
    target: Grid

    def model_record_jsonable(self) -> dict[str, object]:
        """Serialize model I/O only; split/seed/hash/axis names stay private."""

        return {
            "inputs": self.panel.model_inputs_jsonable(self.factor_code),
            "target": grid_to_jsonable(self.target),
        }


@dataclass(frozen=True)
class RandomGeometryDataset:
    train_panels: tuple[GeometryPanel, ...]
    eval_panels: tuple[GeometryPanel, ...]

    def iter_examples(self, split: str) -> Iterator[ExecutorExample]:
        panels = {
            "train": self.train_panels,
            "eval": self.eval_panels,
        }.get(split)
        if panels is None:
            raise ValueError("split must be 'train' or 'eval'")
        for panel in panels:
            for code in FACTOR_CODES:
                yield ExecutorExample(
                    panel=panel,
                    factor_code=code,
                    target=simulate(
                        panel.state,
                        panel.action,
                        factor_code_to_program(code),
                        DEFAULT_PALETTE,
                    ),
                )


def _translated(cells: Iterable[Coord], vector: Coord, scale: int) -> set[Coord]:
    return {add_coord(cell, vector, scale) for cell in cells}


def _motion_shape(direction: Direction, domino: bool) -> tuple[Coord, ...]:
    if not domino:
        return ((0, 0),)
    # A component elongated along the motion vector would overlap its one-step
    # translated counterpart.  The perpendicular domino remains connected and
    # gives genuine shape variation while preserving a valid local event.
    if direction in (Direction.EAST, Direction.WEST):
        return ((0, 0), (1, 0))
    return ((0, 0), (0, 1))


def _sample_motion_fixture(
    axis: Axis,
    rng: random.Random,
    palette: Palette,
) -> FixtureLayout:
    if axis not in (Axis.COLLISION, Axis.RELATION):
        raise ValueError("motion fixtures are defined for collision/relation")
    for _ in range(2_000):
        direction = rng.choice(tuple(Direction))
        vector = direction_vector(direction)
        domino = bool(rng.randrange(2))
        shape = _motion_shape(direction, domino)
        anchor = (rng.randrange(8), rng.randrange(8))
        object_a = {(anchor[0] + row, anchor[1] + col) for row, col in shape}
        shifts = (-1, 0, 1, 2) if axis is Axis.COLLISION else (0, 1, 2)
        envelope = set().union(
            *(_translated(object_a, vector, shift) for shift in shifts)
        )
        if not envelope or not all(in_bounds(cell) for cell in envelope):
            continue
        object_b = _translated(object_a, vector, 1)
        if object_a.intersection(object_b):
            continue
        if axis is Axis.COLLISION:
            colors = {cell: palette.actor for cell in object_a}
            colors.update({cell: palette.blocker for cell in object_b})
        else:
            colors = {cell: palette.object_a for cell in object_a}
            colors.update({cell: palette.object_b for cell in object_b})
        return FixtureLayout(
            axis=axis,
            anchor=anchor,
            direction=direction,
            shape="domino-perpendicular" if domino else "single-cell",
            occupied_cells=frozenset(object_a.union(object_b)),
            safety_envelope=frozenset(envelope),
            action=GridAction(ActionKind.MOVE, anchor, direction),
            cells=tuple(sorted(colors.items())),
        )
    raise RuntimeError(f"could not place a valid {axis.value} fixture")


def _sample_trigger_fixture(
    rng: random.Random,
    palette: Palette,
) -> FixtureLayout:
    row = rng.randrange(8)
    column = rng.randrange(6)
    anchor = (row, column)
    cells = {
        anchor: palette.trigger,
        (row, column + 1): palette.payload_p0,
        (row, column + 2): palette.socket,
    }
    occupied = frozenset(cells)
    return FixtureLayout(
        axis=Axis.TRIGGER,
        anchor=anchor,
        direction=None,
        shape="trigger-payload-socket",
        occupied_cells=occupied,
        safety_envelope=occupied,
        action=GridAction(ActionKind.ACTIVATE, anchor),
        cells=tuple(sorted(cells.items())),
    )


def _sample_fixture(axis: Axis, rng: random.Random, palette: Palette) -> FixtureLayout:
    if axis is Axis.TRIGGER:
        return _sample_trigger_fixture(rng, palette)
    return _sample_motion_fixture(axis, rng, palette)


def _changed_cells(before: Grid, after: Grid) -> frozenset[Coord]:
    return frozenset(
        (row, column)
        for row in range(8)
        for column in range(8)
        if before[row][column] != after[row][column]
    )


def _axis_index(axis: Axis) -> int:
    return AXES.index(axis)


def _validate_panel(panel: GeometryPanel) -> dict[str, object]:
    if panel.geometry_sha256 != geometry_sha256(panel.state, panel.action):
        raise AssertionError("geometry hash is not derived solely from public input")
    if panel.action_atom_count != len(panel.axes):
        raise AssertionError("one public action atom is required per selected axis")
    if tuple(sorted(panel.axes, key=_axis_index)) != panel.axes:
        raise AssertionError("panel axes must use canonical order")
    if len(set(panel.axes)) != len(panel.axes):
        raise AssertionError("panel axes cannot repeat")

    write_sets: dict[Axis, frozenset[Coord]] = {}
    for fixture in panel.fixtures:
        changed_union: set[Coord] = set()
        axis = fixture.axis
        for value in range(4):
            code = [0, 0, 0]
            code[_axis_index(axis)] = value
            target = simulate(
                panel.state,
                fixture.action,
                factor_code_to_program(code),
                DEFAULT_PALETTE,
            )
            changed_union.update(_changed_cells(panel.state, target))
        write_sets[axis] = frozenset(changed_union)
    for left, right in itertools.combinations(panel.axes, 2):
        if write_sets[left].intersection(write_sets[right]):
            raise AssertionError("selected mechanism write sets are not disjoint")

    targets = {
        code: simulate(
            panel.state,
            panel.action,
            factor_code_to_program(code),
            DEFAULT_PALETTE,
        )
        for code in FACTOR_CODES
    }
    expected_behavior_count = 4 ** len(panel.axes)
    behavior_count = len(set(targets.values()))
    if behavior_count != expected_behavior_count:
        raise AssertionError(
            "panel does not expose the full Cartesian behavior product: "
            f"expected {expected_behavior_count}, got {behavior_count}"
        )
    for axis in panel.axes:
        axis_index = _axis_index(axis)
        other_axes = [item for item in panel.axes if item is not axis]
        for other_values in itertools.product(range(4), repeat=len(other_axes)):
            base = [0, 0, 0]
            for other_axis, value in zip(other_axes, other_values, strict=True):
                base[_axis_index(other_axis)] = value
            signatures = set()
            for value in range(4):
                code = list(base)
                code[axis_index] = value
                signatures.add(targets[tuple(code)])
            if len(signatures) != 4:
                raise AssertionError(
                    f"{axis.value} values are not conditionally distinguishable"
                )
    return {
        "behavior_class_count": behavior_count,
        "expected_behavior_class_count": expected_behavior_count,
        "write_sets_pairwise_disjoint": True,
        "selected_axis_values_conditionally_distinguishable": True,
        "write_cell_counts": {
            axis.value: len(write_sets[axis]) for axis in panel.axes
        },
    }


def make_geometry_panel(
    *,
    split: str,
    geometry_seed: int,
    axes: Sequence[Axis],
    geometry_variant: int = 0,
) -> GeometryPanel:
    """Generate one deterministic panel and mechanically validate semantics."""

    normalized_axes = tuple(sorted(tuple(axes), key=_axis_index))
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    if type(geometry_seed) is not int or geometry_seed < 0:
        raise ValueError("geometry_seed must be a non-negative integer")
    if type(geometry_variant) is not int or geometry_variant < 0:
        raise ValueError("geometry_variant must be a non-negative integer")
    allowed = TRAIN_AXIS_SETS if split == "train" else EVAL_AXIS_SETS
    if normalized_axes not in allowed:
        raise ValueError(
            "train permits singleton/pair axes; eval permits triple axes only"
        )
    palette = DEFAULT_PALETTE
    rng = random.Random(
        _stable_seed(
            split,
            geometry_seed,
            geometry_variant,
            *(axis.value for axis in normalized_axes),
        )
    )
    for _ in range(5_000):
        fixtures = tuple(_sample_fixture(axis, rng, palette) for axis in normalized_axes)
        if any(
            left.safety_envelope.intersection(right.safety_envelope)
            for left, right in itertools.combinations(fixtures, 2)
        ):
            continue
        cells: dict[Coord, int] = {}
        for fixture in fixtures:
            for coord, color in fixture.cells:
                if coord in cells:
                    raise AssertionError("fixture occupied cells overlap")
                cells[coord] = color
        protected = set().union(
            *(set(fixture.safety_envelope) for fixture in fixtures)
        )
        nuisance_candidates = [
            (row, column)
            for row in range(8)
            for column in range(8)
            if (row, column) not in protected and (row, column) not in cells
        ]
        nuisance_count = min(rng.randrange(5), len(nuisance_candidates))
        nuisance_cells = frozenset(rng.sample(nuisance_candidates, nuisance_count))
        for coord in nuisance_cells:
            cells[coord] = palette.distractor
        actions = [fixture.action for fixture in fixtures]
        rng.shuffle(actions)
        action: AnyAction = (
            actions[0]
            if len(actions) == 1
            else CompositeAction(tuple(actions))
        )
        state = grid_with_cells(cells)
        panel = GeometryPanel(
            split=split,
            geometry_seed=geometry_seed,
            geometry_variant=geometry_variant,
            axes=normalized_axes,
            state=state,
            action=action,
            fixtures=fixtures,
            nuisance_cells=nuisance_cells,
            geometry_sha256=geometry_sha256(state, action),
        )
        _validate_panel(panel)
        return panel
    raise RuntimeError("could not pack a write-disjoint randomized panel")


def build_random_geometry_dataset(
    *,
    train_geometry_seeds: Sequence[int],
    eval_geometry_seeds: Sequence[int],
) -> RandomGeometryDataset:
    """Build split-exclusive singleton/pair train and triple-only eval panels."""

    train_seeds = tuple(train_geometry_seeds)
    eval_seeds = tuple(eval_geometry_seeds)
    if not train_seeds or not eval_seeds:
        raise ValueError("train and eval geometry seed lists must be non-empty")
    if len(set(train_seeds)) != len(train_seeds) or len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("geometry seeds cannot repeat within a split")
    if set(train_seeds).intersection(eval_seeds):
        raise ValueError("train and eval geometry seeds must be disjoint")
    used_hashes: set[str] = set()

    def unique_panel(split: str, seed: int, axes: tuple[Axis, ...]) -> GeometryPanel:
        for variant in range(10_000):
            panel = make_geometry_panel(
                split=split,
                geometry_seed=seed,
                geometry_variant=variant,
                axes=axes,
            )
            if panel.geometry_sha256 not in used_hashes:
                used_hashes.add(panel.geometry_sha256)
                return panel
        raise RuntimeError("could not generate a unique public geometry")

    train_panels = tuple(
        unique_panel("train", seed, axes)
        for seed in train_seeds
        for axes in TRAIN_AXIS_SETS
    )
    eval_panels = tuple(
        unique_panel("eval", seed, axes)
        for seed in eval_seeds
        for axes in EVAL_AXIS_SETS
    )
    dataset = RandomGeometryDataset(train_panels, eval_panels)
    audit = audit_random_geometry_dataset(dataset)
    if audit["static_gates"]["passed"] is not True:  # type: ignore[index]
        raise AssertionError("generated randomized-geometry dataset failed its audit")
    return dataset


def _layout_coverage(panels: Sequence[GeometryPanel]) -> dict[str, object]:
    directions: dict[str, set[str]] = {axis.value: set() for axis in AXES}
    shapes: dict[str, set[str]] = {axis.value: set() for axis in AXES}
    anchors: dict[str, set[Coord]] = {axis.value: set() for axis in AXES}
    for panel in panels:
        for fixture in panel.fixtures:
            name = fixture.axis.value
            anchors[name].add(fixture.anchor)
            shapes[name].add(fixture.shape)
            if fixture.direction is not None:
                directions[name].add(fixture.direction.value)
    return {
        axis.value: {
            "unique_anchor_count": len(anchors[axis.value]),
            "directions": sorted(directions[axis.value]),
            "shapes": sorted(shapes[axis.value]),
        }
        for axis in AXES
    }


def audit_random_geometry_dataset(dataset: RandomGeometryDataset) -> dict[str, object]:
    """Return a JSON-safe manifest and fail-closed protocol invariants."""

    train_hashes = {panel.geometry_sha256 for panel in dataset.train_panels}
    eval_hashes = {panel.geometry_sha256 for panel in dataset.eval_panels}
    train_panel_audits = [_validate_panel(panel) for panel in dataset.train_panels]
    eval_panel_audits = [_validate_panel(panel) for panel in dataset.eval_panels]
    public_keys = {
        key
        for panel in (*dataset.train_panels, *dataset.eval_panels)
        for key in panel.model_inputs_jsonable((0, 0, 0))
    }
    forbidden_public_keys = {
        "split",
        "geometry_seed",
        "geometry_variant",
        "geometry_sha256",
        "panel_kind",
        "axes",
        "task_id",
        "probe_id",
    }
    train_scope_valid = all(
        panel.split == "train"
        and panel.panel_kind in ("singleton", "pair")
        and panel.action_atom_count in (1, 2)
        for panel in dataset.train_panels
    )
    eval_scope_valid = all(
        panel.split == "eval"
        and panel.panel_kind == "triple"
        and panel.action_atom_count == 3
        for panel in dataset.eval_panels
    )
    train_axes = {axis for panel in dataset.train_panels for axis in panel.axes}
    eval_axes = {axis for panel in dataset.eval_panels for axis in panel.axes}
    gates = {
        "train_contains_only_singletons_and_pairs": train_scope_valid,
        "eval_contains_only_triples": eval_scope_valid,
        "all_axes_covered_in_both_splits": train_axes == set(AXES)
        and eval_axes == set(AXES),
        "train_geometry_hashes_unique": len(train_hashes)
        == len(dataset.train_panels),
        "eval_geometry_hashes_unique": len(eval_hashes) == len(dataset.eval_panels),
        "train_eval_geometry_hash_intersection_empty": not train_hashes.intersection(
            eval_hashes
        ),
        "all_panels_write_disjoint": all(
            item["write_sets_pairwise_disjoint"] for item in (*train_panel_audits, *eval_panel_audits)
        ),
        "all_selected_values_distinguishable": all(
            item["selected_axis_values_conditionally_distinguishable"]
            for item in (*train_panel_audits, *eval_panel_audits)
        ),
        "model_input_has_no_explicit_identifier_field": not public_keys.intersection(
            forbidden_public_keys
        ),
        "all_64_factor_codes_materialized_per_panel": len(FACTOR_CODES) == 64,
    }
    gates["passed"] = all(gates.values())
    return {
        "schema_version": PROTOCOL_VERSION,
        "train_panel_count": len(dataset.train_panels),
        "eval_panel_count": len(dataset.eval_panels),
        "train_example_count": len(dataset.train_panels) * len(FACTOR_CODES),
        "eval_example_count": len(dataset.eval_panels) * len(FACTOR_CODES),
        "train_panel_kind_counts": {
            kind: sum(panel.panel_kind == kind for panel in dataset.train_panels)
            for kind in ("singleton", "pair", "triple")
        },
        "eval_panel_kind_counts": {
            kind: sum(panel.panel_kind == kind for panel in dataset.eval_panels)
            for kind in ("singleton", "pair", "triple")
        },
        "train_unique_geometry_hash_count": len(train_hashes),
        "eval_unique_geometry_hash_count": len(eval_hashes),
        "train_eval_geometry_hash_intersection": sorted(
            train_hashes.intersection(eval_hashes)
        ),
        "model_input_keys": sorted(public_keys),
        "explicit_identifier_fields_in_model_input": sorted(
            public_keys.intersection(forbidden_public_keys)
        ),
        "train_layout_coverage": _layout_coverage(dataset.train_panels),
        "eval_layout_coverage": _layout_coverage(dataset.eval_panels),
        "train_behavior_class_count_histogram": {
            str(count): sum(
                item["behavior_class_count"] == count for item in train_panel_audits
            )
            for count in (4, 16, 64)
        },
        "eval_behavior_class_count_histogram": {
            str(count): sum(
                item["behavior_class_count"] == count for item in eval_panel_audits
            )
            for count in (4, 16, 64)
        },
        "static_gates": gates,
        "train_panels": [
            {
                "geometry_seed": panel.geometry_seed,
                "geometry_variant": panel.geometry_variant,
                "geometry_sha256": panel.geometry_sha256,
                "panel_kind": panel.panel_kind,
                "axes": [axis.value for axis in panel.axes],
                "action_atom_count": panel.action_atom_count,
            }
            for panel in dataset.train_panels
        ],
        "eval_panels": [
            {
                "geometry_seed": panel.geometry_seed,
                "geometry_variant": panel.geometry_variant,
                "geometry_sha256": panel.geometry_sha256,
                "panel_kind": panel.panel_kind,
                "axes": [axis.value for axis in panel.axes],
                "action_atom_count": panel.action_atom_count,
            }
            for panel in dataset.eval_panels
        ],
    }


__all__ = [
    "AXES",
    "EVAL_AXIS_SETS",
    "ExecutorExample",
    "FACTOR_CODES",
    "FixtureLayout",
    "GeometryPanel",
    "PROTOCOL_VERSION",
    "RandomGeometryDataset",
    "TRAIN_AXIS_SETS",
    "audit_random_geometry_dataset",
    "build_random_geometry_dataset",
    "factor_code_to_program",
    "geometry_sha256",
    "make_geometry_panel",
]
