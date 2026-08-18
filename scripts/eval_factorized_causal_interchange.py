#!/usr/bin/env python3
"""Audit single-axis symbolic code interchange in the factorized RuleGrid ceiling.

This is deliberately a *privileged* diagnostic.  The collision/trigger/
relation axes, their four-value codebooks, palette-role canonicalization, and
the exact RuleGrid simulator are supplied by the benchmark.  The amortizer
still has to infer its K=4 integer codes from public support transitions, but
the experiment does not test autonomous discovery of causal variables.

For every selected held-out-context task and every axis, the script chooses a
source and a different donor code from the model's support-only hypotheses,
then replaces exactly one coordinate *after argmax*.  This is an intervention
on an explicit privileged integer code, not on a learned hidden activation.

The canonical audit is retained for continuity but de-duplicated by executor,
patched code, and public input.  A second audit exhausts all 64 patched codes
and all 36 directed single-axis value substitutions on deterministic randomized
legal geometries.  Both use the benchmark's shared deterministic simulator for
targets.  The targets are independent of network output and stored query
labels, but they are not produced by an independently implemented simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SCHEMA_VERSION = "prp-wm.factorized-causal-interchange.v2"
AXIS_NAMES = ("collision", "trigger", "relation")
_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_rules.py",
    "prp_wm/discrete_causal_rules.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/eval_factorized_causal_interchange.py",
    "scripts/run_causal_mechanism_coverage.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256() -> dict[str, str]:
    return {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in _AUDITED_SOURCE_FILES
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_factor_code(code: Sequence[int]) -> tuple[int, int, int]:
    normalized = tuple(code)
    if len(normalized) != 3:
        raise ValueError("a factor code must contain exactly three axes")
    if any(type(value) is not int or not 0 <= value < 4 for value in normalized):
        raise ValueError("factor-code values must be plain integers in [0,4)")
    return normalized  # type: ignore[return-value]


def patch_factor_code(
    source: Sequence[int], donor: Sequence[int], axis: int
) -> tuple[int, int, int]:
    """Replace exactly one coordinate of ``source`` with ``donor[axis]``."""

    source_code = _validate_factor_code(source)
    donor_code = _validate_factor_code(donor)
    if type(axis) is not int or axis not in range(3):
        raise ValueError("axis must be an integer in [0,3)")
    if source_code[axis] == donor_code[axis]:
        raise ValueError("donor must change the selected source axis")
    patched = list(source_code)
    patched[axis] = donor_code[axis]
    result = _validate_factor_code(patched)
    if sum(left != right for left, right in zip(source_code, result)) != 1:
        raise AssertionError("interchange did not change exactly one axis")
    return result


def factor_code_to_rule_program(code: Sequence[int]) -> Any:
    """Decode the benchmark-supplied integer codebook into a RuleProgram."""

    from prp_wm.rulegrid import (
        ALL_COLLISIONS,
        ALL_RELATIONS,
        ALL_TRIGGERS,
        RuleProgram,
    )

    collision, trigger, relation = _validate_factor_code(code)
    return RuleProgram(
        ALL_COLLISIONS[collision],
        ALL_TRIGGERS[trigger],
        ALL_RELATIONS[relation],
    )


def _program_json(code: Sequence[int]) -> dict[str, object]:
    program = factor_code_to_rule_program(code)
    return {
        "factor_code": list(_validate_factor_code(code)),
        "program_id": program.program_id,
        "privileged_mode_names": {
            "collision": program.collision.value,
            "trigger": program.trigger.value,
            "relation": program.relation.value,
        },
    }


@dataclass(frozen=True)
class _InterchangeChoice:
    source_task: int
    donor_task: int
    axis: int
    source_particle: int
    donor_particle: int
    source_code: tuple[int, int, int]
    donor_code: tuple[int, int, int]
    patched_code: tuple[int, int, int]


@dataclass(frozen=True)
class _RandomizedGeometry:
    index: int
    state: Any
    action: Any
    layout: dict[str, object]


def _build_randomized_geometry(
    *,
    index: int,
    collision_row: int,
    collision_col: int,
    collision_direction: Any,
    trigger_row: int,
    trigger_col: int,
    relation_row: int,
    relation_col: int,
    relation_direction: Any,
    action_order: Sequence[str],
) -> _RandomizedGeometry:
    """Build one legal three-event panel using only the public RuleGrid DSL."""

    from prp_wm.rulegrid import (
        ActionKind,
        CompositeAction,
        DEFAULT_PALETTE,
        GridAction,
        add_coord,
        direction_vector,
        grid_with_cells,
    )

    if len({collision_row, trigger_row, relation_row}) != 3:
        raise ValueError("randomized local events must occupy distinct rows")
    if tuple(sorted(action_order)) != ("collision", "relation", "trigger"):
        raise ValueError("action_order must be a permutation of the three axes")
    palette = DEFAULT_PALETTE
    collision_vector = direction_vector(collision_direction)
    collision_actor = (collision_row, collision_col)
    collision_blocker = add_coord(collision_actor, collision_vector)
    trigger_coord = (trigger_row, trigger_col)
    relation_vector = direction_vector(relation_direction)
    relation_a = (relation_row, relation_col)
    relation_b = add_coord(relation_a, relation_vector)
    cells = {
        collision_actor: palette.actor,
        collision_blocker: palette.blocker,
        trigger_coord: palette.trigger,
        (trigger_row, trigger_col + 1): palette.payload_p0,
        (trigger_row, trigger_col + 2): palette.socket,
        relation_a: palette.object_a,
        relation_b: palette.object_b,
    }
    if len(cells) != 7:
        raise AssertionError("randomized fixture cells unexpectedly overlap")
    actions = {
        "collision": GridAction(
            ActionKind.MOVE, collision_actor, collision_direction
        ),
        "trigger": GridAction(ActionKind.ACTIVATE, trigger_coord),
        "relation": GridAction(ActionKind.MOVE, relation_a, relation_direction),
    }
    return _RandomizedGeometry(
        index=index,
        state=grid_with_cells(cells),
        action=CompositeAction(tuple(actions[name] for name in action_order)),
        layout={
            "collision": {
                "row": collision_row,
                "column": collision_col,
                "direction": collision_direction.value,
            },
            "trigger": {"row": trigger_row, "column": trigger_col},
            "relation": {
                "row": relation_row,
                "column": relation_col,
                "direction": relation_direction.value,
            },
            "action_order": list(action_order),
        },
    )


def _make_randomized_geometries(
    count: int, *, seed: int
) -> tuple[_RandomizedGeometry, ...]:
    """Recombine singleton geometries seen in training into novel triples.

    This deliberately holds local-geometry marginals fixed: every collision,
    trigger, and relation fixture occurs in diagnostic indices 0..11 used by
    executor training.  Only the cross-axis triple combination and atom order
    are randomized, isolating composition from a harsher coordinate-OOD test.
    """

    from prp_wm.rulegrid import Direction

    if type(count) is not int or count <= 0:
        raise ValueError("randomized geometry count must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("randomized geometry seed must be non-negative")
    canonical = _build_randomized_geometry(
        index=-1,
        collision_row=1,
        collision_col=1,
        collision_direction=Direction.EAST,
        trigger_row=3,
        trigger_col=3,
        relation_row=5,
        relation_col=5,
        relation_direction=Direction.WEST,
        action_order=("collision", "trigger", "relation"),
    )
    rng = random.Random(seed)
    collision_specs = (
        (1, 1, Direction.EAST),
        (1, 5, Direction.WEST),
        (2, 1, Direction.EAST),
        (2, 5, Direction.WEST),
    )
    trigger_specs = ((3, 1), (3, 4), (4, 1), (4, 4))
    relation_specs = (
        (5, 1, Direction.EAST),
        (5, 5, Direction.WEST),
        (6, 1, Direction.EAST),
        (6, 5, Direction.WEST),
    )
    geometries: list[_RandomizedGeometry] = []
    seen: set[tuple[Any, str]] = set()
    attempts = 0
    while len(geometries) < count:
        attempts += 1
        if attempts > max(1_000, 100 * count):
            raise RuntimeError("could not generate enough unique randomized geometries")
        collision_row, collision_col, collision_direction = rng.choice(
            collision_specs
        )
        trigger_row, trigger_col = rng.choice(trigger_specs)
        relation_row, relation_col, relation_direction = rng.choice(
            relation_specs
        )
        order = ["collision", "trigger", "relation"]
        rng.shuffle(order)
        candidate = _build_randomized_geometry(
            index=len(geometries),
            collision_row=collision_row,
            collision_col=collision_col,
            collision_direction=collision_direction,
            trigger_row=trigger_row,
            trigger_col=trigger_col,
            relation_row=relation_row,
            relation_col=relation_col,
            relation_direction=relation_direction,
            action_order=order,
        )
        # A changed action tuple alone is not sufficient geometry variation.
        if candidate.state == canonical.state:
            continue
        signature = (candidate.state, candidate.action.canonical_id)
        if signature in seen:
            continue
        seen.add(signature)
        geometries.append(candidate)
    return tuple(geometries)


def _unique_particle_codes(
    codes: Sequence[Sequence[int]],
) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    seen: set[tuple[int, int, int]] = set()
    unique: list[tuple[int, tuple[int, int, int]]] = []
    for particle, raw_code in enumerate(codes):
        code = _validate_factor_code(raw_code)
        if code not in seen:
            seen.add(code)
            unique.append((particle, code))
    return tuple(sorted(unique, key=lambda item: (item[1], item[0])))


def _select_interchanges(
    inferred_codes: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[_InterchangeChoice, ...], tuple[dict[str, object], ...]]:
    """Choose source/donor hypotheses using model outputs only.

    Oracle support compatibility is intentionally absent from this selection
    rule.  A poor amortizer therefore cannot be made to look better by using
    privileged labels to select a convenient particle.
    """

    task_codes = tuple(_unique_particle_codes(codes) for codes in inferred_codes)
    if len(task_codes) < 2:
        raise ValueError("at least two tasks are required for donor interchange")
    if any(not codes for codes in task_codes):
        raise ValueError("every task must expose at least one inferred code")
    selected: list[_InterchangeChoice] = []
    skipped: list[dict[str, object]] = []
    for source_task, candidates in enumerate(task_codes):
        source_particle, source_code = candidates[0]
        for axis in range(3):
            donor_choice: tuple[int, int, tuple[int, int, int]] | None = None
            for offset in range(1, len(task_codes)):
                donor_task = (source_task + offset) % len(task_codes)
                for donor_particle, donor_code in task_codes[donor_task]:
                    if donor_code[axis] != source_code[axis]:
                        donor_choice = donor_task, donor_particle, donor_code
                        break
                if donor_choice is not None:
                    break
            if donor_choice is None:
                skipped.append(
                    {
                        "source_task_index": source_task,
                        "axis_index": axis,
                        "axis_name": AXIS_NAMES[axis],
                        "reason": "no_other_task_inferred_a_different_axis_value",
                    }
                )
                continue
            donor_task, donor_particle, donor_code = donor_choice
            selected.append(
                _InterchangeChoice(
                    source_task=source_task,
                    donor_task=donor_task,
                    axis=axis,
                    source_particle=source_particle,
                    donor_particle=donor_particle,
                    source_code=source_code,
                    donor_code=donor_code,
                    patched_code=patch_factor_code(
                        source_code, donor_code, axis
                    ),
                )
            )
    return tuple(selected), tuple(skipped)


def _tensor_json(tensor: Any) -> object:
    return tensor.detach().cpu().tolist()


def _tensor_panel_sha256(*tensors: Any) -> str:
    return _json_sha256(
        [
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "values": _tensor_json(tensor),
            }
            for tensor in tensors
            if tensor is not None
        ]
    )


def _evaluate_randomized_geometry_executor(
    *,
    torch: Any,
    executor: Any,
    executor_path: Path,
    geometry_count: int,
    seed: int,
) -> dict[str, object]:
    """Exhaust all codes and directed axis substitutions on novel layouts."""

    from prp_wm.causal_filter import enumerate_factor_codes
    from prp_wm.latent_rules import outcome_map
    from prp_wm.neural import encode_public_action
    from prp_wm.rulegrid import DEFAULT_PALETTE, simulate

    geometries = _make_randomized_geometries(geometry_count, seed=seed)
    device = next(executor.parameters()).device
    states = torch.tensor(
        [geometry.state for geometry in geometries],
        dtype=torch.long,
        device=device,
    )
    actions = torch.stack(
        [encode_public_action(geometry.action) for geometry in geometries]
    ).to(device)
    if actions.ndim != 3 or actions.shape[1] != 3:
        raise AssertionError("randomized triple actions must contain three atoms")
    action_mask = torch.ones(
        actions.shape[:2], dtype=torch.bool, device=device
    )
    factor_bank = enumerate_factor_codes().to(device)
    code_list = [
        _validate_factor_code(code)
        for code in factor_bank.detach().cpu().tolist()
    ]
    if {
        factor_code_to_rule_program(code).program_id for code in code_list
    } != set(range(64)):
        raise AssertionError("factor bank must enumerate all 64 programs")
    programs = [factor_code_to_rule_program(code) for code in code_list]
    geometry_total = len(geometries)
    rule_total = len(code_list)
    flat_states = (
        states[:, None]
        .expand(-1, rule_total, -1, -1)
        .reshape(geometry_total * rule_total, *states.shape[-2:])
    )
    flat_actions = (
        actions[:, None]
        .expand(-1, rule_total, -1, -1)
        .reshape(geometry_total * rule_total, *actions.shape[-2:])
    )
    flat_action_mask = (
        action_mask[:, None]
        .expand(-1, rule_total, -1)
        .reshape(geometry_total * rule_total, action_mask.shape[-1])
    )
    flat_factors = (
        factor_bank[None]
        .expand(geometry_total, -1, -1)
        .reshape(geometry_total * rule_total, 3)
    )
    target_values = [
        simulate(geometry.state, geometry.action, program, DEFAULT_PALETTE)
        for geometry in geometries
        for program in programs
    ]
    targets = torch.tensor(target_values, dtype=torch.long, device=device)
    with torch.no_grad():
        prediction = executor.predict(
            flat_states,
            flat_actions,
            flat_factors,
            flat_action_mask,
        )
        maps = outcome_map(prediction)[:, 0]
        cell_nll = -prediction.log_prob_cells(targets)[:, 0]
    exact = maps.eq(targets).all(dim=(-2, -1))
    mean_nll = cell_nll.mean(dim=(-2, -1))

    input_hashes = [
        _tensor_panel_sha256(states[index], actions[index], action_mask[index])
        for index in range(geometry_total)
    ]
    if len(set(input_hashes)) != geometry_total:
        raise AssertionError("randomized public geometry inputs must be unique")
    geometry_records = [
        {
            "geometry_index": geometry.index,
            "public_input_sha256": input_hashes[index],
            "layout": geometry.layout,
        }
        for index, geometry in enumerate(geometries)
    ]
    execution_records: list[dict[str, object]] = []
    for geometry_index in range(geometry_total):
        for code_index, code in enumerate(code_list):
            flat_index = geometry_index * rule_total + code_index
            execution_records.append(
                {
                    "geometry_index": geometry_index,
                    "public_input_sha256": input_hashes[geometry_index],
                    "factor_code": list(code),
                    "program_id": programs[code_index].program_id,
                    "simulator_target_sha256": _tensor_panel_sha256(
                        targets[flat_index]
                    ),
                    "executor_map_sha256": _tensor_panel_sha256(maps[flat_index]),
                    "map_grid_exact": bool(exact[flat_index].item()),
                    "proper_mean_cell_nll": float(mean_nll[flat_index].cpu()),
                }
            )

    program_index = {code: index for index, code in enumerate(code_list)}
    directed_transitions: set[tuple[int, int, int]] = set()
    patched_programs: set[tuple[int, int, int]] = set()
    intervention_count = 0
    effective_count = 0
    per_axis: dict[str, dict[str, object]] = {}
    for axis, axis_name in enumerate(AXIS_NAMES):
        axis_count = 0
        axis_effective = 0
        for geometry_index in range(geometry_total):
            panel = targets[
                geometry_index * rule_total : (geometry_index + 1) * rule_total
            ]
            for source_code in code_list:
                source_target = panel[program_index[source_code]]
                for donor_value in range(4):
                    if donor_value == source_code[axis]:
                        continue
                    patched = list(source_code)
                    patched[axis] = donor_value
                    patched_code = _validate_factor_code(patched)
                    if sum(
                        left != right
                        for left, right in zip(source_code, patched_code)
                    ) != 1:
                        raise AssertionError("randomized intervention is not single-axis")
                    patched_target = panel[program_index[patched_code]]
                    is_effective = bool(patched_target.ne(source_target).any().item())
                    axis_count += 1
                    axis_effective += int(is_effective)
                    directed_transitions.add(
                        (axis, source_code[axis], donor_value)
                    )
                    patched_programs.add(patched_code)
        intervention_count += axis_count
        effective_count += axis_effective
        per_axis[axis_name] = {
            "intervention_case_count": axis_count,
            "effective_intervention_count": axis_effective,
            "effective_intervention_rate": axis_effective / axis_count,
            "directed_value_transition_count": len(
                {item for item in directed_transitions if item[0] == axis}
            ),
        }
    nll_values = [float(record["proper_mean_cell_nll"]) for record in execution_records]
    exact_count = sum(bool(record["map_grid_exact"]) for record in execution_records)
    return {
        "executor_checkpoint_path": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "geometry_generator": (
            "randomly recombine collision/trigger/relation singleton geometries "
            "from training indices 0..11 into non-canonical triples; randomize "
            "composite-action atom order"
        ),
        "local_geometry_marginals_seen_during_executor_training": True,
        "triple_geometry_combinations_used_during_executor_training": False,
        "canonical_geometry_excluded": True,
        "geometry_count": geometry_total,
        "unique_public_geometry_count": len(set(input_hashes)),
        "program_code_count": len(patched_programs),
        "directed_axis_value_transition_count": len(directed_transitions),
        "unique_execution_case_count": len(execution_records),
        "expected_unique_execution_case_count": geometry_total * 64,
        "intervention_case_count": intervention_count,
        "effective_intervention_count": effective_count,
        "effective_intervention_rate": effective_count / intervention_count,
        "map_grid_exact_count": exact_count,
        "map_grid_exact_rate": exact_count / len(execution_records),
        "proper_mean_cell_nll": _mean(nll_values),
        "proper_max_case_mean_cell_nll": max(nll_values),
        "per_axis": per_axis,
        "geometries": geometry_records,
        "execution_cases": execution_records,
    }


def _make_support_only_batch(
    torch: Any,
    tasks: Sequence[Any],
    *,
    diagnostic_indices: Sequence[int],
    device: Any,
) -> Any:
    """Create the exact model input without reading stored query targets."""

    from prp_wm.latent_rules import (
        _pad_public_action_panel,
        canonicalize_rulegrid_tensor_batch,
    )
    from prp_wm.neural import RuleGridTensorBatch, encode_public_action

    materialized = tuple(tasks)
    indices = tuple(diagnostic_indices)
    support_states: list[list[Any]] = []
    support_targets: list[list[Any]] = []
    support_actions: list[list[Any]] = []
    query_states: list[list[Any]] = []
    query_actions: list[list[Any]] = []
    for task in materialized:
        support = task.inference.support[:6]
        if len(support) != 6:
            raise ValueError("interchange evaluation requires six support transitions")
        support_states.append([transition.state for transition in support])
        support_targets.append([transition.next_state for transition in support])
        support_actions.append(
            [encode_public_action(transition.action) for transition in support]
        )
        diagnostics = tuple(task.inference.diagnostics[index] for index in indices)
        query_states.append([probe.state for probe in diagnostics])
        query_actions.append([encode_public_action(probe.action) for probe in diagnostics])
    support_action_tensor, support_action_mask = _pad_public_action_panel(
        support_actions, device
    )
    query_action_tensor, query_action_mask = _pad_public_action_panel(
        query_actions, device
    )
    raw = RuleGridTensorBatch(
        support_states=torch.tensor(
            support_states, dtype=torch.long, device=device
        ),
        support_actions=support_action_tensor,
        support_targets=torch.tensor(
            support_targets, dtype=torch.long, device=device
        ),
        support_mask=torch.ones(
            len(materialized), 6, dtype=torch.bool, device=device
        ),
        query_states=torch.tensor(query_states, dtype=torch.long, device=device),
        query_actions=query_action_tensor,
        query_targets=None,
        behavior_targets=None,
        behavior_mass=None,
        support_action_mask=support_action_mask,
        query_action_mask=query_action_mask,
    )
    # This privileged transform reads palette role bindings, never a rule or
    # query target.  It matches how the audited amortizer and executor trained.
    return canonicalize_rulegrid_tensor_batch(raw, materialized)


def _resolve_device(torch: Any, raw: str) -> Any:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    return device


def _configure_determinism(torch: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class _ResolvedArtifact:
    result_path: Path
    result: dict[str, Any]
    checkpoint_path: Path
    executor_path: Path


def _resolve_recorded_path(raw: object, *, relative_to: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SystemExit(f"artifact has no valid {name}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _resolve_artifact(raw_path: Path) -> _ResolvedArtifact:
    supplied = raw_path.expanduser().resolve()
    if supplied.is_dir():
        result_path = supplied / "result.json"
        supplied_checkpoint: Path | None = None
    elif supplied.suffix == ".json":
        result_path = supplied
        supplied_checkpoint = None
    else:
        result_path = supplied.parent / "result.json"
        supplied_checkpoint = supplied
    if not result_path.is_file():
        raise SystemExit(f"artifact result.json does not exist: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    is_factorized = result.get("model") == "factorized-3x4" or (
        result.get("model") is None
        and result.get("experiment")
        == "expected_discrete_axis_structured_causal_k4"
    )
    if not is_factorized:
        raise SystemExit(
            f"causal interchange requires model=factorized-3x4: {result_path}"
        )
    checkpoint_path = _resolve_recorded_path(
        result.get("checkpoint_path"),
        relative_to=result_path.parent,
        name="checkpoint_path",
    )
    if supplied_checkpoint is not None and supplied_checkpoint != checkpoint_path:
        raise SystemExit("supplied checkpoint differs from result checkpoint_path")
    if not checkpoint_path.is_file():
        raise SystemExit(f"artifact checkpoint does not exist: {checkpoint_path}")
    if result.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise SystemExit("artifact result/checkpoint SHA256 provenance mismatch")
    executor_path = _resolve_recorded_path(
        result.get("executor_checkpoint"),
        relative_to=result_path.parent,
        name="executor_checkpoint",
    )
    if not executor_path.is_file():
        raise SystemExit(f"executor checkpoint does not exist: {executor_path}")
    if result.get("executor_checkpoint_sha256") != _sha256_file(executor_path):
        raise SystemExit("artifact executor SHA256 provenance mismatch")
    return _ResolvedArtifact(
        result_path=result_path,
        result=result,
        checkpoint_path=checkpoint_path,
        executor_path=executor_path,
    )


def _current_source_matches(
    recorded: object, current: dict[str, str]
) -> dict[str, bool | None]:
    if not isinstance(recorded, dict):
        return {relative: None for relative in current}
    return {
        relative: (
            recorded.get(relative) == digest
            if relative in recorded
            else None
        )
        for relative, digest in current.items()
    }


def _canonical_grid(tensor: Any) -> tuple[tuple[int, ...], ...]:
    values = tensor.detach().cpu().tolist()
    return tuple(tuple(int(value) for value in row) for row in values)


def _evaluate_choice(
    *,
    torch: Any,
    executor: Any,
    batch: Any,
    tasks: Sequence[Any],
    diagnostic_indices: Sequence[int],
    choice: _InterchangeChoice,
    compatible_sets: Sequence[set[tuple[int, int, int]]],
    probabilities: Any,
) -> dict[str, object]:
    from prp_wm.latent_rules import outcome_map
    from prp_wm.rulegrid import DEFAULT_PALETTE, simulate

    source_index = choice.source_task
    states = batch.query_states[source_index]
    actions = batch.query_actions[source_index]
    action_mask = (
        batch.query_action_mask[source_index]
        if batch.query_action_mask is not None
        else None
    )
    factors = torch.tensor(
        [choice.patched_code] * len(diagnostic_indices),
        dtype=torch.long,
        device=states.device,
    )
    patched_program = factor_code_to_rule_program(choice.patched_code)
    source_program = factor_code_to_rule_program(choice.source_code)
    patched_targets: list[Any] = []
    source_targets: list[Any] = []
    for local_index, diagnostic_index in enumerate(diagnostic_indices):
        action = tasks[source_index].inference.diagnostics[diagnostic_index].action
        canonical_state = _canonical_grid(states[local_index])
        # This is independent of network output and stored task labels.  It is
        # deliberately *not* implementation-independent: executor training
        # labels and this audit both use the benchmark's RuleGrid simulator.
        patched_targets.append(
            simulate(canonical_state, action, patched_program, DEFAULT_PALETTE)
        )
        source_targets.append(
            simulate(canonical_state, action, source_program, DEFAULT_PALETTE)
        )
    targets = torch.tensor(
        patched_targets, dtype=torch.long, device=states.device
    )
    source_target_tensor = torch.tensor(
        source_targets, dtype=torch.long, device=states.device
    )
    with torch.no_grad():
        prediction = executor.predict(states, actions, factors, action_mask)
        maps = outcome_map(prediction)[:, 0]
        cell_nll = -prediction.log_prob_cells(targets)[:, 0]

    diagnostics: list[dict[str, object]] = []
    for local_index, diagnostic_index in enumerate(diagnostic_indices):
        changed = targets[local_index].ne(states[local_index])
        unchanged = ~changed
        intervention_delta = targets[local_index].ne(
            source_target_tensor[local_index]
        )
        proper_nll = float(cell_nll[local_index].mean().detach().cpu())
        changed_nll = (
            float(cell_nll[local_index][changed].mean().detach().cpu())
            if bool(changed.any().item())
            else 0.0
        )
        unchanged_nll = (
            float(cell_nll[local_index][unchanged].mean().detach().cpu())
            if bool(unchanged.any().item())
            else 0.0
        )
        diagnostics.append(
            {
                "diagnostic_index": diagnostic_index,
                "public_input_sha256": _tensor_panel_sha256(
                    states[local_index],
                    actions[local_index],
                    action_mask[local_index] if action_mask is not None else None,
                ),
                "simulator_target_sha256": _tensor_panel_sha256(
                    targets[local_index]
                ),
                "executor_map_sha256": _tensor_panel_sha256(maps[local_index]),
                "map_grid_exact": bool(
                    maps[local_index].eq(targets[local_index]).all().item()
                ),
                "proper_mean_cell_nll": proper_nll,
                "changed_mean_cell_nll": changed_nll,
                "unchanged_mean_cell_nll": unchanged_nll,
                "target_changed_cell_count": int(changed.sum().item()),
                "intervention_changed_cell_count": int(
                    intervention_delta.sum().item()
                ),
                "intervention_effective": bool(intervention_delta.any().item()),
            }
        )

    hamming = sum(
        source != patched
        for source, patched in zip(choice.source_code, choice.patched_code)
    )
    source_confidence = probabilities[
        choice.source_task, choice.source_particle
    ].detach().cpu().tolist()
    donor_confidence = probabilities[
        choice.donor_task, choice.donor_particle
    ].detach().cpu().tolist()
    return {
        "source_task_index": choice.source_task,
        "donor_task_index": choice.donor_task,
        "axis_index": choice.axis,
        "axis_name": AXIS_NAMES[choice.axis],
        "source_particle_index": choice.source_particle,
        "donor_particle_index": choice.donor_particle,
        "source": _program_json(choice.source_code),
        "donor": _program_json(choice.donor_code),
        "patched": _program_json(choice.patched_code),
        "source_particle_axis_probabilities": source_confidence,
        "donor_particle_axis_probabilities": donor_confidence,
        "source_code_support_compatible": (
            choice.source_code in compatible_sets[choice.source_task]
        ),
        "donor_code_support_compatible": (
            choice.donor_code in compatible_sets[choice.donor_task]
        ),
        "selected_axis_value_changed": (
            choice.source_code[choice.axis] != choice.patched_code[choice.axis]
        ),
        "nonselected_axes_preserved": all(
            choice.source_code[axis] == choice.patched_code[axis]
            for axis in range(3)
            if axis != choice.axis
        ),
        "source_to_patched_hamming_distance": hamming,
        "all_diagnostics_map_exact": all(
            bool(item["map_grid_exact"]) for item in diagnostics
        ),
        "all_diagnostics_intervention_effective": all(
            bool(item["intervention_effective"]) for item in diagnostics
        ),
        "diagnostics": diagnostics,
    }


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _summarize(
    artifact_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    support_audits = [
        audit
        for artifact in artifact_records
        for audit in artifact["support_inference_audit"]  # type: ignore[index]
    ]
    interchanges = [
        record
        for artifact in artifact_records
        for record in artifact["interchanges"]  # type: ignore[index]
    ]
    raw_diagnostics = [
        diagnostic
        for record in interchanges
        for diagnostic in record["diagnostics"]  # type: ignore[index]
    ]
    skipped = [
        record
        for artifact in artifact_records
        for record in artifact["skipped_interchanges"]  # type: ignore[index]
    ]
    # The same frozen executor, integer code, and canonical public input recur
    # across triple IDs, tasks, folds, and model seeds.  Those are not
    # independent execution trials.  Collapse them before computing any
    # executor-performance denominator.
    execution_cases: dict[tuple[object, ...], dict[str, object]] = {}
    intervention_cases: dict[tuple[object, ...], dict[str, object]] = {}

    def equivalent_case(
        left: dict[str, object], right: dict[str, object]
    ) -> bool:
        nll_key = "proper_mean_cell_nll"
        if any(
            left.get(key) != right.get(key)
            for key in set(left).union(right)
            if key != nll_key
        ):
            return False
        return math.isclose(
            float(left[nll_key]),
            float(right[nll_key]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )

    for artifact in artifact_records:
        executor_sha = str(artifact["executor_checkpoint_sha256"])
        for record in artifact["interchanges"]:  # type: ignore[index]
            source_code = tuple(record["source"]["factor_code"])  # type: ignore[index]
            patched_code = tuple(record["patched"]["factor_code"])  # type: ignore[index]
            axis = int(record["axis_index"])  # type: ignore[arg-type]
            for diagnostic in record["diagnostics"]:  # type: ignore[index]
                public_hash = str(diagnostic["public_input_sha256"])
                execution_key = (executor_sha, patched_code, public_hash)
                execution_value = {
                    "map_grid_exact": bool(diagnostic["map_grid_exact"]),
                    "proper_mean_cell_nll": float(
                        diagnostic["proper_mean_cell_nll"]
                    ),
                    "simulator_target_sha256": str(
                        diagnostic["simulator_target_sha256"]
                    ),
                    "executor_map_sha256": str(
                        diagnostic["executor_map_sha256"]
                    ),
                }
                prior = execution_cases.get(execution_key)
                if prior is not None and not equivalent_case(
                    prior, execution_value
                ):
                    raise AssertionError(
                        "duplicate canonical execution case produced different results"
                    )
                execution_cases[execution_key] = execution_value
                intervention_key = (
                    executor_sha,
                    source_code,
                    patched_code,
                    axis,
                    public_hash,
                )
                intervention_value = {
                    **execution_value,
                    "axis_index": axis,
                    "intervention_effective": bool(
                        diagnostic["intervention_effective"]
                    ),
                }
                prior_intervention = intervention_cases.get(intervention_key)
                if (
                    prior_intervention is not None
                    and not equivalent_case(
                        prior_intervention, intervention_value
                    )
                ):
                    raise AssertionError(
                        "duplicate canonical intervention produced different results"
                    )
                intervention_cases[intervention_key] = intervention_value
    executions = list(execution_cases.values())
    interventions = list(intervention_cases.values())
    proper_nlls = [float(item["proper_mean_cell_nll"]) for item in executions]
    effective = [item for item in interventions if item["intervention_effective"]]
    exact = [item for item in executions if item["map_grid_exact"]]
    exact_effective = [item for item in effective if item["map_grid_exact"]]
    public_inputs = {
        str(item["public_input_sha256"]) for item in raw_diagnostics
    }
    simulator_targets = {
        str(item["simulator_target_sha256"]) for item in executions
    }
    support_inputs = {
        str(item["public_canonical_support_sha256"]) for item in support_audits
    }
    per_axis: dict[str, dict[str, object]] = {}
    for axis, name in enumerate(AXIS_NAMES):
        axis_records = [record for record in interchanges if record["axis_index"] == axis]
        axis_interventions = [
            item for item in interventions if item["axis_index"] == axis
        ]
        per_axis[name] = {
            "interchange_count": len(axis_records),
            "all_diagnostics_map_exact_rate": _mean(
                float(bool(item["map_grid_exact"]))
                for item in axis_interventions
            ),
            "effective_intervention_rate": _mean(
                float(bool(item["intervention_effective"]))
                for item in axis_interventions
            ),
            "unique_intervention_case_count": len(axis_interventions),
        }
    context_exact = [
        bool(artifact.get("heldout_context_coverage_exact", False))
        for artifact in artifact_records
    ]
    return {
        "artifact_count": len(artifact_records),
        "support_task_count": len(support_audits),
        "heldout_context_coverage_exact_artifact_rate": _mean(
            float(value) for value in context_exact
        ),
        "minimum_unique_heldout_context_count": min(
            (
                int(artifact.get("unique_heldout_context_count", 0))
                for artifact in artifact_records
            ),
            default=0,
        ),
        "support_inferred_exact_version_space_rate": _mean(
            float(bool(item["predicted_set_exact"])) for item in support_audits
        ),
        "support_inferred_code_recall": _mean(
            float(item["compatible_code_recall"]) for item in support_audits
        ),
        "expected_interchange_count": len(support_audits) * 3,
        "evaluated_interchange_count": len(interchanges),
        "skipped_interchange_count": len(skipped),
        "single_axis_structural_invariant_rate": _mean(
            float(
                record["source_to_patched_hamming_distance"] == 1
                and record["selected_axis_value_changed"]
                and record["nonselected_axes_preserved"]
            )
            for record in interchanges
        ),
        "source_code_support_compatible_rate": _mean(
            float(bool(record["source_code_support_compatible"]))
            for record in interchanges
        ),
        "donor_code_support_compatible_rate": _mean(
            float(bool(record["donor_code_support_compatible"]))
            for record in interchanges
        ),
        "raw_diagnostic_prediction_count": len(raw_diagnostics),
        "diagnostic_prediction_count": len(executions),
        "unique_execution_case_count": len(executions),
        "unique_intervention_case_count": len(interventions),
        "unique_canonical_support_input_count": len(support_inputs),
        "unique_public_diagnostic_input_count": len(public_inputs),
        "unique_patched_program_public_input_case_count": len(executions),
        "unique_simulator_target_count": len(simulator_targets),
        "interchange_all_diagnostics_map_exact_rate": _mean(
            float(bool(item["map_grid_exact"])) for item in interventions
        ),
        "diagnostic_map_grid_exact_rate": len(exact) / max(len(executions), 1),
        "effective_intervention_diagnostic_count": len(effective),
        "effective_intervention_rate": len(effective)
        / max(len(interventions), 1),
        "effective_intervention_map_grid_exact_rate": (
            len(exact_effective) / max(len(effective), 1)
        ),
        "proper_mean_cell_nll": _mean(proper_nlls),
        "proper_max_diagnostic_mean_cell_nll": max(proper_nlls, default=0.0),
        "per_axis": per_axis,
    }


def _evaluate_artifact(
    *,
    torch: Any,
    artifact: _ResolvedArtifact,
    artifact_index: int,
    tasks_per_artifact: int,
    device: Any,
    current_sources: dict[str, str],
) -> dict[str, object]:
    from prp_wm.discrete_causal_rules import ExpectedDiscreteCausalK4
    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.pilot import TRIPLE_DIAGNOSTIC_INDICES, make_pilot_tasks
    from prp_wm.rulegrid import version_space
    from scripts.run_expected_discrete_causal_coverage import (
        _build_context_pool,
        _load_audited_executor,
        _support_context_key,
    )

    result = artifact.result
    executor, executor_checkpoint = _load_audited_executor(
        torch, artifact.executor_path, device
    )
    checkpoint = torch.load(
        artifact.checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("model_type") != "ExpectedDiscreteCausalK4":
        raise SystemExit("factorized artifact has an incompatible model_type")
    for field in ("model", "context_fold", "data_master_seed", "eval_split"):
        if checkpoint.get(field) != result.get(field):
            raise SystemExit(f"result/checkpoint mismatch for {field}")
    if checkpoint.get("executor_checkpoint_sha256") != _sha256_file(
        artifact.executor_path
    ):
        raise SystemExit("amortizer checkpoint references a different executor")
    executor_state = executor_checkpoint.get("model_state_dict")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(executor_state, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint has no auditable model_state_dict")
    embedded_executor_exact = all(
        f"executor.{name}" in model_state
        and torch.equal(
            model_state[f"executor.{name}"].detach().cpu(),
            value.detach().cpu(),
        )
        for name, value in executor_state.items()
    )
    if not embedded_executor_exact:
        raise SystemExit(
            "amortizer embeds executor weights different from the audited checkpoint"
        )
    attention_layers = int(result.get("attention_layers", 2))
    temperature = float(result.get("factor_temperature_end", 1.0))
    model = ExpectedDiscreteCausalK4(
        executor,
        attention_layers=attention_layers,
        temperature=temperature,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    executor.eval()
    if any(parameter.requires_grad for parameter in executor.parameters()):
        raise AssertionError("audited executor was not frozen")

    context_fold = result.get("context_fold")
    if context_fold is not None:
        context_fold = int(context_fold)
    tasks = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split=str(result["eval_split"]),
        master_seed=int(result["data_master_seed"]),
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=tasks_per_artifact,
        heldout=True,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
        context_fold=context_fold,
    )
    batch = _make_support_only_batch(
        torch,
        tasks,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        device=device,
    )
    batch.validate(model.config)
    with torch.no_grad():
        inference = model.infer_support(batch, temperature=temperature)
    inferred_codes = inference.factor_ids.detach().cpu().tolist()
    choices, skipped = _select_interchanges(inferred_codes)
    compatible_sets = [
        {
            rule_program_factor_ids(program)
            for program in version_space(
                task.inference.support[:6], task.privileged.palette
            )
        }
        for task in tasks
    ]
    heldout_context_counts: dict[tuple[int, int, int], int] = {}
    for task in tasks:
        context = _support_context_key(
            task,
            factor_ids_for_program=rule_program_factor_ids,
            version_space=version_space,
        )
        heldout_context_counts[context] = heldout_context_counts.get(context, 0) + 1
    recorded_contexts = result.get("eval_support_contexts")
    if isinstance(recorded_contexts, list):
        try:
            expected_contexts = {
                tuple(int(value) for value in context)
                for context in recorded_contexts
            }
        except (TypeError, ValueError) as error:
            raise SystemExit("artifact eval_support_contexts are malformed") from error
        expected_context_source = "artifact_result"
    else:
        expected_contexts = {
            (axis, left, right)
            for axis in range(3)
            for left in range(4)
            for right in range(4)
            if (
                left == right
                if context_fold is None
                else (left + right) % 4 == context_fold
            )
        }
        expected_context_source = "reconstructed_from_recorded_context_fold"
    if any(len(context) != 3 for context in expected_contexts):
        raise SystemExit("artifact eval_support_contexts must be three-tuples")
    context_coverage_exact = (
        len(expected_contexts) == 12
        and set(heldout_context_counts) == expected_contexts
    )
    support_audit: list[dict[str, object]] = []
    for task_index, (raw_codes, compatible) in enumerate(
        zip(inferred_codes, compatible_sets, strict=True)
    ):
        predicted_list = [
            _validate_factor_code(code) for code in raw_codes
        ]
        predicted = set(predicted_list)
        support_audit.append(
            {
                "task_index": task_index,
                "public_canonical_support_sha256": _tensor_panel_sha256(
                    batch.support_states[task_index],
                    batch.support_actions[task_index],
                    batch.support_targets[task_index],
                    batch.support_mask[task_index],
                    (
                        batch.support_action_mask[task_index]
                        if batch.support_action_mask is not None
                        else None
                    ),
                ),
                "inferred_particle_codes": [list(code) for code in predicted_list],
                "inferred_unique_codes": [list(code) for code in sorted(predicted)],
                "support_compatible_codes": [
                    list(code) for code in sorted(compatible)
                ],
                "predicted_unique_code_count": len(predicted),
                "predicted_set_exact": predicted == compatible,
                "compatible_code_recall": len(predicted.intersection(compatible))
                / len(compatible),
            }
        )
    interchanges = [
        _evaluate_choice(
            torch=torch,
            executor=executor,
            batch=batch,
            tasks=tasks,
            diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
            choice=choice,
            compatible_sets=compatible_sets,
            probabilities=inference.factor_probabilities,
        )
        for choice in choices
    ]
    executor_result_path = artifact.executor_path.parent / "result.json"
    return {
        "artifact_index": artifact_index,
        "artifact_id": (
            f"fold-{context_fold}-seed-{result.get('model_seed')}-"
            f"{_sha256_file(artifact.checkpoint_path)[:12]}"
        ),
        "model": "factorized-3x4",
        "context_fold": context_fold,
        "model_seed": result.get("model_seed"),
        "data_master_seed": result.get("data_master_seed"),
        "eval_split": result.get("eval_split"),
        "result_path": str(artifact.result_path),
        "result_sha256": _sha256_file(artifact.result_path),
        "checkpoint_path": str(artifact.checkpoint_path),
        "checkpoint_sha256": _sha256_file(artifact.checkpoint_path),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "executor_checkpoint_path": str(artifact.executor_path),
        "executor_checkpoint_sha256": _sha256_file(artifact.executor_path),
        "executor_checkpoint_schema_version": executor_checkpoint.get(
            "checkpoint_schema_version"
        ),
        "embedded_executor_state_exactly_matches_audited_checkpoint": (
            embedded_executor_exact
        ),
        "executor_result_path": str(executor_result_path),
        "executor_result_sha256": _sha256_file(executor_result_path),
        "recorded_source_sha256": result.get("source_sha256"),
        "recorded_source_matches_current": _current_source_matches(
            result.get("source_sha256"), current_sources
        ),
        "tasks_evaluated": len(tasks),
        "expected_heldout_contexts": [
            list(context) for context in sorted(expected_contexts)
        ],
        "expected_heldout_context_source": expected_context_source,
        "heldout_context_task_counts": [
            {"context": list(context), "task_count": count}
            for context, count in sorted(heldout_context_counts.items())
        ],
        "unique_heldout_context_count": len(heldout_context_counts),
        "heldout_context_coverage_exact": context_coverage_exact,
        "triple_diagnostic_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "support_inference_audit": support_audit,
        "interchanges": interchanges,
        "skipped_interchanges": list(skipped),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        required=True,
        help="Repeat for each factorized result.json, checkpoint, or run directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks-per-artifact",
        type=int,
        default=48,
        help="48 covers all 12 held-out contexts four times in the pilot stream.",
    )
    parser.add_argument(
        "--random-geometries",
        type=int,
        default=8,
        help="Number of deterministic non-canonical public layouts per executor.",
    )
    parser.add_argument("--nll-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tasks_per_artifact < 2:
        raise SystemExit("--tasks-per-artifact must be at least 2")
    if args.random_geometries <= 0:
        raise SystemExit("--random-geometries must be positive")
    if args.nll_threshold <= 0 or not math.isfinite(args.nll_threshold):
        raise SystemExit("--nll-threshold must be a finite positive number")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    import torch

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    current_sources = _source_sha256()
    resolved = [_resolve_artifact(path) for path in args.artifact]
    identities = {
        (artifact.result_path, artifact.checkpoint_path) for artifact in resolved
    }
    if len(identities) != len(resolved):
        raise SystemExit("duplicate --artifact inputs are not allowed")
    artifact_records = [
        _evaluate_artifact(
            torch=torch,
            artifact=artifact,
            artifact_index=index,
            tasks_per_artifact=args.tasks_per_artifact,
            device=device,
            current_sources=current_sources,
        )
        for index, artifact in enumerate(resolved)
    ]
    summary = _summarize(artifact_records)
    from scripts.run_expected_discrete_causal_coverage import _load_audited_executor

    unique_executors: dict[str, _ResolvedArtifact] = {}
    for artifact in resolved:
        executor_sha = _sha256_file(artifact.executor_path)
        unique_executors.setdefault(executor_sha, artifact)
    randomized_geometry_records: list[dict[str, object]] = []
    for executor_sha, artifact in sorted(unique_executors.items()):
        executor, _ = _load_audited_executor(
            torch, artifact.executor_path, device
        )
        executor.eval()
        record = _evaluate_randomized_geometry_executor(
            torch=torch,
            executor=executor,
            executor_path=artifact.executor_path,
            geometry_count=args.random_geometries,
            seed=args.seed,
        )
        if record["executor_checkpoint_sha256"] != executor_sha:
            raise AssertionError("randomized executor identity changed during audit")
        randomized_geometry_records.append(record)
    randomized_case_count = sum(
        int(record["unique_execution_case_count"])
        for record in randomized_geometry_records
    )
    randomized_summary: dict[str, object] = {
        "unique_executor_count": len(randomized_geometry_records),
        "geometry_count_per_executor": args.random_geometries,
        "unique_execution_case_count": randomized_case_count,
        "expected_unique_execution_case_count": sum(
            int(record["expected_unique_execution_case_count"])
            for record in randomized_geometry_records
        ),
        "minimum_unique_public_geometry_count": min(
            (
                int(record["unique_public_geometry_count"])
                for record in randomized_geometry_records
            ),
            default=0,
        ),
        "minimum_program_code_count": min(
            (int(record["program_code_count"]) for record in randomized_geometry_records),
            default=0,
        ),
        "minimum_directed_axis_value_transition_count": min(
            (
                int(record["directed_axis_value_transition_count"])
                for record in randomized_geometry_records
            ),
            default=0,
        ),
        "effective_intervention_rate": (
            sum(
                int(record["effective_intervention_count"])
                for record in randomized_geometry_records
            )
            / max(
                sum(
                    int(record["intervention_case_count"])
                    for record in randomized_geometry_records
                ),
                1,
            )
        ),
        "map_grid_exact_rate": (
            sum(
                int(record["map_grid_exact_count"])
                for record in randomized_geometry_records
            )
            / max(randomized_case_count, 1)
        ),
        "proper_mean_cell_nll": (
            sum(
                float(record["proper_mean_cell_nll"])
                * int(record["unique_execution_case_count"])
                for record in randomized_geometry_records
            )
            / max(randomized_case_count, 1)
        ),
        "proper_max_case_mean_cell_nll": max(
            (
                float(record["proper_max_case_mean_cell_nll"])
                for record in randomized_geometry_records
            ),
            default=0.0,
        ),
    }
    interchange_gate = bool(
        summary["skipped_interchange_count"] == 0
        and summary["single_axis_structural_invariant_rate"] == 1.0
        and summary["source_code_support_compatible_rate"] == 1.0
        and summary["donor_code_support_compatible_rate"] == 1.0
        and summary["effective_intervention_rate"] == 1.0
        and summary["diagnostic_map_grid_exact_rate"] == 1.0
        and summary["proper_mean_cell_nll"] <= args.nll_threshold
    )
    inference_gate = bool(
        summary["heldout_context_coverage_exact_artifact_rate"] == 1.0
        and summary["support_inferred_exact_version_space_rate"] >= 0.90
        and summary["support_inferred_code_recall"] >= 0.90
    )
    randomized_geometry_gate = bool(
        randomized_summary["unique_executor_count"] > 0
        and randomized_summary["minimum_unique_public_geometry_count"]
        == args.random_geometries
        and randomized_summary["minimum_program_code_count"] == 64
        and randomized_summary[
            "minimum_directed_axis_value_transition_count"
        ]
        == 36
        and randomized_summary["unique_execution_case_count"]
        == randomized_summary["expected_unique_execution_case_count"]
        and randomized_summary["effective_intervention_rate"] == 1.0
        and randomized_summary["map_grid_exact_rate"] == 1.0
        and randomized_summary["proper_max_case_mean_cell_nll"]
        <= args.nll_threshold
    )
    overall_gate = bool(
        inference_gate and interchange_gate and randomized_geometry_gate
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "privileged_single_axis_symbolic_code_interchange",
        "result_kind": "privileged_symbolic_mechanism_execution_diagnostic",
        "interchange_level": "post_argmax_explicit_integer_factor_code",
        "learned_hidden_activation_interchange_tested": False,
        "mechanism_axes_and_value_codebook_given": True,
        "palette_role_canonicalization_given": True,
        "ground_truth_simulator_given": True,
        "ground_truth_independent_of_network_output": True,
        "ground_truth_uses_training_simulator_implementation": True,
        "ground_truth_implementation_independent": False,
        "autonomous_causal_variable_discovery_tested": False,
        "model_inference_reads_public_support_only": True,
        "stored_query_targets_read_for_ground_truth": False,
        "task_true_program_read_for_ground_truth": False,
        "ground_truth_construction": (
            "decode patched integer tuple through the privileged RuleGrid "
            "codebook, then call deterministic prp_wm.rulegrid.simulate on "
            "canonical or randomized public diagnostic inputs"
        ),
        "interchange_selection": (
            "lexicographically first unique source hypothesis; cyclically "
            "first other-task donor hypothesis with a different selected-axis "
            "value; selection never consults support compatibility or labels"
        ),
        "heldout_diagnostic_scope": (
            "triple-composition indices 21..23, withheld from amortizer "
            "training targets; these three canonical inputs are redundant and "
            "are de-duplicated for executor metrics"
        ),
        "tasks_per_artifact": args.tasks_per_artifact,
        "random_geometries_per_executor": args.random_geometries,
        "evaluation_seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "nll_threshold": args.nll_threshold,
        "source_sha256": current_sources,
        "artifacts": artifact_records,
        "summary": summary,
        "randomized_geometry_audits": randomized_geometry_records,
        "randomized_geometry_summary": randomized_summary,
        "overall_passed": overall_gate,
        "static_gates": {
            "support_inference_gate": {
                "complete_heldout_context_coverage_eq": True,
                "exact_version_space_rate_gte": 0.90,
                "compatible_code_recall_gte": 0.90,
                "passed": inference_gate,
            },
            "frozen_executor_interchange_gate": {
                "no_skipped_interchanges": True,
                "single_axis_structural_invariant_rate_eq": 1.0,
                "source_code_support_compatible_rate_eq": 1.0,
                "donor_code_support_compatible_rate_eq": 1.0,
                "effective_intervention_rate_eq": 1.0,
                "diagnostic_map_grid_exact_rate_eq": 1.0,
                "proper_mean_cell_nll_lte": args.nll_threshold,
                "passed": interchange_gate,
            },
            "randomized_geometry_executor_gate": {
                "unique_noncanonical_geometries_per_executor_eq": (
                    args.random_geometries
                ),
                "program_code_count_eq": 64,
                "directed_axis_value_transition_count_eq": 36,
                "effective_intervention_rate_eq": 1.0,
                "map_grid_exact_rate_eq": 1.0,
                "proper_max_case_mean_cell_nll_lte": args.nll_threshold,
                "passed": randomized_geometry_gate,
            },
            "overall_conjunctive_gate": {
                "requires": [
                    "support_inference_gate",
                    "frozen_executor_interchange_gate",
                    "randomized_geometry_executor_gate",
                ],
                "passed": overall_gate,
            },
        },
        "interpretation": (
            "An overall pass supports support-conditioned recovery of the "
            "privileged discrete version space plus symbolic code-conditioned "
            "execution and single-axis substitution on randomized legal "
            "geometries. It is not a learned hidden-state interchange test, an "
            "independent simulator validation, or evidence that axes, value "
            "semantics, palette roles, or a causal model were discovered."
        ),
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": _sha256_file(output),
                "summary": summary,
                "randomized_geometry_summary": randomized_summary,
                "static_gates": payload["static_gates"],
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
