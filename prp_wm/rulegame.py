"""Minimal win-only game whose optimal terminal action requires visual history.

This is intentionally separate from RuleGrid: RuleGrid diagnoses rule beliefs,
whereas RuleGame asks whether RL can acquire the useful behavior from terminal
success alone.  The terminal decision frame is pixel-identical for all four
hidden collision modes sharing a nuisance seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from .rulegrid import (
    ALL_COLLISIONS,
    ActionKind,
    Collision,
    Direction,
    Grid,
    GridAction,
    Relation,
    RuleProgram,
    Trigger,
    add_coord,
    direction_vector,
    grid_with_cells,
    palette_from_seed,
    simulate,
)


PROBE = 0
CONTINUE = 1
FIRST_DOOR = 2
NUM_ACTIONS = 6
HORIZON = 3


@dataclass(frozen=True)
class RuleGameSpec:
    mode: Collision
    nuisance_seed: int
    task_id: str


@dataclass(frozen=True)
class RuleGameObservation:
    grid: Grid
    legal_actions: tuple[bool, ...]


@dataclass(frozen=True)
class RuleGameStep:
    observation: RuleGameObservation
    reward: float
    done: bool
    won: bool


def _seed(split: str, master_seed: int, group: int) -> int:
    source = f"rulegame-v1|{split}|{master_seed}|{group}".encode()
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "little")


def make_rulegame_specs(
    *, split: str, master_seed: int, start: int, count: int
) -> tuple[RuleGameSpec, ...]:
    """Return balanced four-mode groups with nuisance independent of mode."""

    if not split or "/" in split:
        raise ValueError("split must be a non-empty slash-free string")
    if any(type(value) is not int or value < 0 for value in (master_seed, start)):
        raise ValueError("master_seed and start must be non-negative integers")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    result = []
    for index in range(start, start + count):
        group, mode_index = divmod(index, len(ALL_COLLISIONS))
        result.append(
            RuleGameSpec(
                mode=ALL_COLLISIONS[mode_index],
                nuisance_seed=_seed(split, master_seed, group),
                task_id=f"{split}/G{group:06d}/M{mode_index}",
            )
        )
    return tuple(result)


class RuleGame:
    """Three-turn game: probe or skip, continue, then choose one of four doors."""

    def __init__(self, spec: RuleGameSpec) -> None:
        self.spec = spec
        self._palette = palette_from_seed(spec.nuisance_seed)
        rng = random.Random(spec.nuisance_seed)
        self._direction = tuple(Direction)[rng.randrange(4)]
        vector = direction_vector(self._direction)
        actor = (3, 3)
        blocker = add_coord(actor, vector)
        self._initial = grid_with_cells(
            {actor: self._palette.actor, blocker: self._palette.blocker}
        )
        self._probe_action = GridAction(ActionKind.MOVE, actor, self._direction)
        self._evidence = simulate(
            self._initial,
            self._probe_action,
            RuleProgram(spec.mode, Trigger.TOGGLE, Relation.NONE),
            self._palette,
        )
        spare = tuple(
            color for color in range(1, 16) if color not in self._palette.role_values
        )
        door_color, player_color = spare[:2]
        self._decision = grid_with_cells(
            {
                (1, 1): door_color,
                (1, 3): door_color,
                (1, 5): door_color,
                (1, 7): door_color,
                (6, 4): player_color,
            }
        )
        self._phase = 0
        self._probed = False
        self._done = False

    @property
    def observation(self) -> RuleGameObservation:
        if self._phase == 0:
            return RuleGameObservation(
                self._initial,
                (True, True, False, False, False, False),
            )
        if self._phase == 1:
            return RuleGameObservation(
                self._evidence if self._probed else self._initial,
                (False, True, False, False, False, False),
            )
        return RuleGameObservation(
            self._decision,
            (False, False, True, True, True, True),
        )

    def step(self, action: int) -> RuleGameStep:
        if self._done:
            raise RuntimeError("cannot step a finished RuleGame")
        if type(action) is not int or not 0 <= action < NUM_ACTIONS:
            raise ValueError("action must be an integer in the public action space")
        if not self.observation.legal_actions[action]:
            raise ValueError("action is illegal in the current public phase")
        if self._phase == 0:
            self._probed = action == PROBE
            self._phase = 1
            return RuleGameStep(self.observation, 0.0, False, False)
        if self._phase == 1:
            self._phase = 2
            return RuleGameStep(self.observation, 0.0, False, False)

        selected_mode = action - FIRST_DOOR
        won = self._probed and selected_mode == ALL_COLLISIONS.index(self.spec.mode)
        self._done = True
        return RuleGameStep(self.observation, float(won), True, won)


__all__ = [
    "CONTINUE",
    "FIRST_DOOR",
    "HORIZON",
    "NUM_ACTIONS",
    "PROBE",
    "RuleGame",
    "RuleGameObservation",
    "RuleGameSpec",
    "RuleGameStep",
    "make_rulegame_specs",
]
