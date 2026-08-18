#!/usr/bin/env python3
"""Train a public-only persistent K4 support encoder on exact version spaces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


CHECKPOINT_SCHEMA_VERSION = "prp-wm.public-version-space-causal-k4.v1"
RESULT_SCHEMA_VERSION = "prp-wm.public-version-space-causal-k4-run.v1"
HISTORY_CONTEXT_HEADS = frozenset(
    {
        "atom-matched-composite-event-history-probe-factor-belief",
        "composite-relative-event-history-probe-factor-belief",
        "history-conditioned-probe-factor-belief",
        "palette-invariant-atom-matched-composite-event-history-probe-factor-belief",
        "relational-composite-event-history-probe-factor-belief",
        "relative-event-history-probe-factor-belief",
        "translation-invariant-history-probe-factor-belief",
    }
)
PROBE_CONTEXT_HEADS = frozenset(
    {
        "probe-aware-symmetry-factor-belief",
    }
) | HISTORY_CONTEXT_HEADS
FACTOR_BELIEF_HEADS = frozenset({"symmetry-factor-belief"}) | PROBE_CONTEXT_HEADS


@dataclass(frozen=True)
class ControllerHistory:
    """Public transitions paired with controller-owned observation provenance."""

    transitions: tuple[Any, ...]
    is_agent_probe_result: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("controller history cannot be empty")
        if len(self.transitions) != len(self.is_agent_probe_result):
            raise ValueError("controller provenance must match transition count")
        if any(type(flag) is not bool for flag in self.is_agent_probe_result):
            raise TypeError("controller provenance flags must be bool")


DEFAULT_EXECUTOR = REPOSITORY_ROOT / (
    "runs/support_calibrated_executor_seed20260724/checkpoint_last.pt"
)
_AUDITED_SOURCE_FILES = (
    "prp_wm/public_version_k4.py",
    "prp_wm/discrete_causal_rules.py",
    "prp_wm/causal_rules.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_public_version_space_k4.py",
    "scripts/run_stratified_gram_public_coverage.py",
    "scripts/run_gram_public_coverage_finetune.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, default=DEFAULT_EXECUTOR)
    parser.add_argument("--context-fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-pool-tasks", type=int, default=144)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--tail-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.10)
    parser.add_argument("--varying-axis-weight", type=float, default=1.0)
    parser.add_argument("--validity-weight", type=float, default=0.0)
    parser.add_argument("--joint-margin", type=float, default=1.0)
    parser.add_argument("--factor-temperature", type=float, default=1.0)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument(
        "--support-input",
        choices=("oracle-canonical", "raw"),
        default="oracle-canonical",
        help=(
            "oracle-canonical applies the simulator palette-role mapping; raw "
            "passes public color IDs unchanged"
        ),
    )
    parser.add_argument(
        "--student-support-encoders",
        action="store_true",
        help="use trainable support encoders separate from the frozen teacher executor",
    )
    parser.add_argument(
        "--version-head",
        choices=(
            "slot-joint",
            "factorized-set",
            "transition-evidence",
            "symmetry-factor-belief",
            "probe-aware-symmetry-factor-belief",
            "history-conditioned-probe-factor-belief",
            "relative-event-history-probe-factor-belief",
            "composite-relative-event-history-probe-factor-belief",
            "relational-composite-event-history-probe-factor-belief",
            "atom-matched-composite-event-history-probe-factor-belief",
            "palette-invariant-atom-matched-composite-event-history-probe-factor-belief",
            "translation-invariant-history-probe-factor-belief",
        ),
        default="slot-joint",
    )
    parser.add_argument("--task-consistency-weight", type=float, default=0.5)
    parser.add_argument(
        "--active-prefix-curriculum",
        action="store_true",
        help="alternate t0 batches with t0 plus one public symmetry-breaking result",
    )
    parser.add_argument(
        "--neutral-probe-curriculum",
        action="store_true",
        help=(
            "cycle t0, informative probe, and replay probe batches so an "
            "agent-result marker does not imply information gain"
        ),
    )
    parser.add_argument(
        "--semantic-composite-curriculum",
        action="store_true",
        help=(
            "add a neutral probe that composes two previously observed local "
            "events without repeating either raw transition"
        ),
    )
    parser.add_argument(
        "--informative-composite-curriculum",
        action="store_true",
        help=(
            "pair neutral semantic composites with a composite containing a "
            "genuinely belief-reducing public probe result"
        ),
    )
    parser.add_argument(
        "--color-permutation-augmentation",
        action="store_true",
        help="randomly relabel non-background raw colors per training example",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="gram-causal-train")
    parser.add_argument("--eval-split", default="gram-causal-composition")
    return parser.parse_args()


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


def _json_cli(args: argparse.Namespace) -> dict[str, object]:
    encoded: dict[str, object] = {}
    for key, value in vars(args).items():
        encoded[key] = str(value.resolve()) if isinstance(value, Path) else value
    return encoded


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "eval_batch_size",
        "attention_layers",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.tail_steps < 0 or args.tail_steps > args.steps:
        raise SystemExit("--tail-steps must lie in [0,steps]")
    for name in (
        "learning_rate",
        "tail_learning_rate",
        "max_grad_norm",
        "joint_margin",
        "factor_temperature",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "margin_weight",
        "validity_weight",
        "varying_axis_weight",
        "task_consistency_weight",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size > args.train_pool_tasks:
        raise SystemExit("--batch-size cannot exceed --train-pool-tasks")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    if args.student_support_encoders and args.support_input != "raw":
        raise SystemExit("--student-support-encoders requires --support-input raw")
    if args.color_permutation_augmentation and args.support_input != "raw":
        raise SystemExit("--color-permutation-augmentation requires --support-input raw")
    if args.version_head in FACTOR_BELIEF_HEADS and args.support_input != "raw":
        raise SystemExit("symmetry-aware factor belief requires --support-input raw")
    if (
        args.version_head == "translation-invariant-history-probe-factor-belief"
        and not args.student_support_encoders
    ):
        raise SystemExit(
            "translation-invariant history head requires student support encoders"
        )
    if args.active_prefix_curriculum and args.version_head not in FACTOR_BELIEF_HEADS:
        raise SystemExit(
            "active prefix curriculum requires a symmetry-aware factor belief head"
        )
    if args.neutral_probe_curriculum and not args.active_prefix_curriculum:
        raise SystemExit("neutral probe curriculum requires active prefix curriculum")
    if args.neutral_probe_curriculum and args.version_head not in PROBE_CONTEXT_HEADS:
        raise SystemExit("neutral probe curriculum requires a probe-context head")
    if args.semantic_composite_curriculum and not args.neutral_probe_curriculum:
        raise SystemExit(
            "semantic composite curriculum requires neutral probe curriculum"
        )
    if (
        args.informative_composite_curriculum
        and not args.semantic_composite_curriculum
    ):
        raise SystemExit(
            "informative composite curriculum requires semantic composite curriculum"
        )


def _permute_raw_support_colors(
    torch: Any,
    batch: Any,
    *,
    generator: Any,
    num_colors: int,
):
    """Apply one consistent non-background color permutation per example."""

    lookup = torch.arange(num_colors, dtype=torch.long)[None].repeat(
        batch.batch_size,
        1,
    )
    for task_index in range(batch.batch_size):
        lookup[task_index, 1:] = (
            torch.randperm(num_colors - 1, generator=generator) + 1
        )
    lookup = lookup.to(batch.support_states.device)

    def transform(grids: Any):
        flattened = grids.reshape(grids.shape[0], -1)
        return lookup.gather(1, flattened).reshape_as(grids)

    return replace(
        batch,
        support_states=transform(batch.support_states),
        support_targets=transform(batch.support_targets),
    )


def _symbolic_version_space_mask(
    torch: Any,
    model: Any,
    tasks: Any,
    *,
    device: Any,
):
    """Build an independent simulator-label mask; never feed it to inference."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    bank = [
        tuple(int(value) for value in row)
        for row in model.factor_bank.detach().cpu().tolist()
    ]
    code_to_index = {code: index for index, code in enumerate(bank)}
    mask = torch.zeros(len(tasks), len(bank), dtype=torch.bool, device=device)
    for task_index, task in enumerate(tasks):
        compatible = version_space(
            task.inference.support[:6],
            task.privileged.palette,
        )
        for program in compatible:
            mask[
                task_index,
                code_to_index[tuple(rule_program_factor_ids(program))],
            ] = True
    return mask.detach()


def _symbolic_transition_evidence_targets(
    torch: Any,
    tasks: Any,
    *,
    device: Any,
):
    """Label each observed transition as one-axis evidence or neutral."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    axis_targets = torch.full(
        (len(tasks), 6),
        3,
        dtype=torch.long,
        device=device,
    )
    value_targets = torch.full(
        (len(tasks), 6),
        -1,
        dtype=torch.long,
        device=device,
    )
    for task_index, task in enumerate(tasks):
        for step, transition in enumerate(task.inference.support[:6]):
            compatible = version_space(
                (transition,),
                task.privileged.palette,
            )
            codes = tuple(rule_program_factor_ids(program) for program in compatible)
            fixed = [
                axis
                for axis in range(3)
                if len({code[axis] for code in codes}) == 1
            ]
            if not fixed:
                if len(compatible) != 64:
                    raise AssertionError("neutral evidence must preserve all 64 rules")
                continue
            if len(fixed) != 1 or len(compatible) != 16:
                raise AssertionError("one transition must identify at most one factor")
            axis = fixed[0]
            axis_targets[task_index, step] = axis
            value_targets[task_index, step] = codes[0][axis]
    return axis_targets.detach(), value_targets.detach()


def _symmetry_expanded_version_space_mask(
    torch: Any,
    model: Any,
    tasks: Any,
    *,
    device: Any,
    histories: Any | None = None,
):
    """Union named programs across the public p1/p2 role-name symmetry."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    bank = [tuple(int(value) for value in row) for row in model.factor_bank.cpu().tolist()]
    lookup = {code: index for index, code in enumerate(bank)}
    mask = torch.zeros(len(tasks), 64, dtype=torch.bool, device=device)
    selected_histories = (
        tuple(task.inference.support[:6] for task in tasks)
        if histories is None
        else tuple(tuple(history) for history in histories)
    )
    if len(selected_histories) != len(tasks):
        raise ValueError("histories must match tasks")
    for task_index, (task, history) in enumerate(zip(tasks, selected_histories)):
        palette = task.privileged.palette
        swapped = replace(
            palette,
            payload_p1=palette.payload_p2,
            payload_p2=palette.payload_p1,
        )
        programs = set(version_space(history, palette))
        programs.update(version_space(history, swapped))
        for program in programs:
            mask[task_index, lookup[tuple(rule_program_factor_ids(program))]] = True
    return mask.detach()


def _symmetry_transition_evidence_targets(
    torch: Any,
    tasks: Any,
    *,
    device: Any,
    histories: Any | None = None,
    ignored_steps: Any | None = None,
):
    """Return one evidence-axis label and a symmetry-aware value set per step."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    selected_histories = (
        tuple(task.inference.support[:6] for task in tasks)
        if histories is None
        else tuple(tuple(history) for history in histories)
    )
    if len(selected_histories) != len(tasks):
        raise ValueError("histories must match tasks")
    steps = len(selected_histories[0])
    if any(len(history) != steps for history in selected_histories):
        raise ValueError("evidence histories must have equal length")
    if ignored_steps is not None:
        if ignored_steps.shape != (len(tasks), steps):
            raise ValueError("ignored_steps must have shape [tasks,steps]")
        if ignored_steps.dtype != torch.bool:
            raise ValueError("ignored_steps must be boolean")
    axes = torch.full((len(tasks), steps), 3, dtype=torch.long, device=device)
    values = torch.zeros(len(tasks), steps, 3, 4, dtype=torch.bool, device=device)
    for task_index, (task, history) in enumerate(zip(tasks, selected_histories)):
        palette = task.privileged.palette
        swapped = replace(
            palette,
            payload_p1=palette.payload_p2,
            payload_p2=palette.payload_p1,
        )
        for step, transition in enumerate(history):
            if ignored_steps is not None and bool(
                ignored_steps[task_index, step].item()
            ):
                continue
            programs = set(version_space((transition,), palette))
            programs.update(version_space((transition,), swapped))
            codes = tuple(rule_program_factor_ids(program) for program in programs)
            constrained = [
                axis
                for axis in range(3)
                if len({code[axis] for code in codes}) < 4
            ]
            if not constrained:
                continue
            if len(constrained) != 1:
                raise AssertionError("one transition constrained multiple causal axes")
            axis = constrained[0]
            axes[task_index, step] = axis
            for value in {code[axis] for code in codes}:
                values[task_index, step, axis, value] = True
    return axes.detach(), values.detach()


def _conditional_probe_innovation_targets(
    torch: Any,
    model: Any,
    tasks: Any,
    *,
    device: Any,
    controller_histories: Any,
):
    """Label each probe result by the public belief reduction it causes.

    A transition can be uninformative in isolation yet decisive given the
    preceding history.  In particular, a no-op on the observed trigger output
    distinguishes RECOLOR from TOGGLE only after the earlier recoloring event
    established those two hypotheses.  Preserve the standalone labels for the
    other observations and replace each informative probe-result label with
    the surviving posterior factor set from ``V(H[:t]) -> V(H[:t+1])``.
    """

    selected = tuple(controller_histories)
    if not selected:
        raise ValueError("controller histories cannot be empty")
    if len(selected) != len(tasks):
        raise ValueError("controller histories must match tasks")
    selected_histories = tuple(item.transitions for item in selected)
    steps = len(selected_histories[0])
    if any(len(history) != steps for history in selected_histories):
        raise ValueError("controller histories must have equal length")

    probe_result_mask = torch.tensor(
        [item.is_agent_probe_result for item in selected],
        dtype=torch.bool,
        device=device,
    )
    axes, values = _symmetry_transition_evidence_targets(
        torch,
        tasks,
        device=device,
        histories=selected_histories,
        ignored_steps=probe_result_mask,
    )
    axes = axes.clone()
    values = values.clone()
    for task_index, (task, item) in enumerate(zip(tasks, selected)):
        for step, is_probe_result in enumerate(item.is_agent_probe_result):
            if not is_probe_result:
                continue
            prefix_compatible = _symmetry_expanded_version_space_mask(
                torch,
                model,
                (task,),
                device=device,
                histories=(item.transitions[:step],),
            )
            posterior_compatible = _symmetry_expanded_version_space_mask(
                torch,
                model,
                (task,),
                device=device,
                histories=(item.transitions[: step + 1],),
            )
            if bool(
                (posterior_compatible & ~prefix_compatible).any().item()
            ):
                raise AssertionError("an observed result enlarged the version space")
            if not bool(posterior_compatible.any().item()):
                raise AssertionError("an observed result emptied the version space")
            if torch.equal(prefix_compatible, posterior_compatible):
                axes[task_index, step] = 3
                values[task_index, step].zero_()
                continue
            prefix_factor_sets = model._factor_value_masks(
                model.factor_bank,
                prefix_compatible,
            )[0]
            posterior_factor_sets = model._factor_value_masks(
                model.factor_bank,
                posterior_compatible,
            )[0]
            if bool((posterior_factor_sets.sum(dim=-1) == 0).any().item()):
                raise AssertionError("an observed result produced an empty factor set")
            reduced = (prefix_factor_sets != posterior_factor_sets).any(dim=-1)
            reduction_count = int(reduced.sum().item())
            if reduction_count == 0:
                raise AssertionError(
                    "joint belief shrank without a factor-marginal reduction"
                )
            if reduction_count != 1:
                raise AssertionError(
                    "one probe result reduced multiple factor marginals"
                )
            innovation_axis = int(reduced.to(dtype=torch.long).argmax().item())
            axes[task_index, step] = innovation_axis
            values[task_index, step].zero_()
            values[task_index, step, innovation_axis] = posterior_factor_sets[
                innovation_axis
            ]
    return axes.detach(), values.detach()


def _conditional_active_innovation_targets(
    torch: Any,
    model: Any,
    tasks: Any,
    *,
    device: Any,
    histories: Any,
):
    """Backward-compatible one-final-probe wrapper for focused tests."""

    controller_histories = tuple(
        ControllerHistory(
            tuple(history),
            (False,) * (len(history) - 1) + (True,),
        )
        for history in histories
    )
    return _conditional_probe_innovation_targets(
        torch,
        model,
        tasks,
        device=device,
        controller_histories=controller_histories,
    )


def _public_trigger_symmetry_break_probe(inference_view: Any):
    """Choose a probe using only an observed trigger transition and raw colors."""

    from prp_wm.rulegrid import (
        ActionKind,
        GridAction,
        RuleGridProbe,
        grid_with_cells,
    )

    activate = next(
        transition
        for transition in inference_view.support[:6]
        if transition.action.kind is ActionKind.ACTIVATE
    )
    row, column = activate.action.coord
    payload_coord = (row, column + 1)
    socket_coord = (row, column + 2)
    observed_output_color = activate.next_state[payload_coord[0]][payload_coord[1]]
    if observed_output_color in (0, activate.state[payload_coord[0]][payload_coord[1]]):
        raise ValueError("trigger symmetry break requires a changed non-background output")
    return RuleGridProbe(
        "public-trigger-symmetry-break",
        grid_with_cells(
            {
                (row, column): activate.state[row][column],
                payload_coord: observed_output_color,
                socket_coord: activate.state[socket_coord[0]][socket_coord[1]],
            }
        ),
        GridAction(ActionKind.ACTIVATE, (row, column)),
    )


def _active_break_controller_history(task: Any) -> ControllerHistory:
    """Append feedback and its controller-owned provenance in one operation."""

    from prp_wm.rulegrid import RuleGridTransition, simulate

    probe = _public_trigger_symmetry_break_probe(task.inference)
    target = simulate(
        probe.state,
        probe.action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    prefix = tuple(task.inference.support[:6])
    return ControllerHistory(
        prefix + (RuleGridTransition(probe.state, probe.action, target),),
        (False,) * len(prefix) + (True,),
    )


def _active_break_history(task: Any):
    """Compatibility view containing only the public transitions."""

    return _active_break_controller_history(task).transitions


def _neutral_replay_controller_history(task: Any) -> ControllerHistory:
    """Replay an observed trigger probe whose result adds no public evidence."""

    from prp_wm.rulegrid import ActionKind, RuleGridTransition, simulate

    prefix = tuple(task.inference.support[:6])
    observed = next(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.ACTIVATE
    )
    target = simulate(
        observed.state,
        observed.action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    if target != observed.next_state:
        raise AssertionError("replayed public transition changed its outcome")
    return ControllerHistory(
        prefix + (
            RuleGridTransition(observed.state, observed.action, target),
        ),
        (False,) * len(prefix) + (True,),
    )


def _informative_then_replay_controller_history(task: Any) -> ControllerHistory:
    """Apply the symmetry break once, then repeat it without new information."""

    from prp_wm.rulegrid import RuleGridTransition, simulate

    informative = _active_break_controller_history(task)
    observed = informative.transitions[-1]
    repeated_target = simulate(
        observed.state,
        observed.action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    if repeated_target != observed.next_state:
        raise AssertionError("repeated informative probe changed its outcome")
    return ControllerHistory(
        informative.transitions
        + (
            RuleGridTransition(
                observed.state,
                observed.action,
                repeated_target,
            ),
        ),
        informative.is_agent_probe_result + (True,),
    )


def _overlay_public_grids(grids: Any):
    """Overlay disjoint non-background public fixtures into one grid."""

    from prp_wm.rulegrid import grid_with_cells

    cells: dict[tuple[int, int], int] = {}
    for grid in grids:
        for row, values in enumerate(grid):
            for column, value in enumerate(values):
                if value == 0:
                    continue
                coord = (row, column)
                if coord in cells:
                    raise ValueError("public fixtures overlap")
                cells[coord] = value
    return grid_with_cells(cells)


def _public_semantic_composite_replay_probe(inference_view: Any):
    """Compose two observed local events using public states and actions only.

    The selected ACTIVATE and effectful MOVE fixtures were already observed
    separately.  Their disjoint overlay is therefore a semantic replay, while
    its composite action and raw tensors differ from every prior transition.
    """

    from prp_wm.rulegrid import (
        ActionKind,
        CompositeAction,
        RuleGridProbe,
    )

    prefix = tuple(inference_view.support[:6])
    activate = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.ACTIVATE
    )
    effectful_move = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.MOVE
        and transition.action.coord != (0, 0)
    )
    if len(activate) != 1 or len(effectful_move) != 1:
        raise ValueError(
            "semantic composite replay requires one ACTIVATE and one "
            "non-neutral MOVE observation"
        )
    components = (activate[0], effectful_move[0])
    return RuleGridProbe(
        "public-semantic-composite-replay",
        _overlay_public_grids(item.state for item in components),
        CompositeAction(tuple(item.action for item in components)),
    )


def _semantic_composite_replay_controller_history(task: Any) -> ControllerHistory:
    """Append a novel raw transition entailed by two public observations."""

    from prp_wm.rulegrid import ActionKind, RuleGridTransition, simulate

    prefix = tuple(task.inference.support[:6])
    components = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.ACTIVATE
        or (
            transition.action.kind is ActionKind.MOVE
            and transition.action.coord != (0, 0)
        )
    )
    if len(components) != 2:
        raise ValueError("semantic composite replay lost its public components")
    probe = _public_semantic_composite_replay_probe(task.inference)
    target = simulate(
        probe.state,
        probe.action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    expected = _overlay_public_grids(item.next_state for item in components)
    if target != expected:
        raise AssertionError(
            "semantic composite result was not entailed by public outcomes"
        )
    transition = RuleGridTransition(probe.state, probe.action, target)
    if transition in prefix:
        raise AssertionError("semantic composite replay repeated a raw transition")
    return ControllerHistory(
        prefix + (transition,),
        (False,) * len(prefix) + (True,),
    )


def _translate_public_grid_rows(grid: Any, row_delta: int):
    """Translate public non-background cells without reading role metadata."""

    from prp_wm.rulegrid import grid_with_cells

    return grid_with_cells(
        {
            (row + row_delta, column): value
            for row, values in enumerate(grid)
            for column, value in enumerate(values)
            if value != 0
        }
    )


def _translated_move_semantic_composite_replay_controller_history(
    task: Any,
    *,
    target_row: int = 6,
) -> ControllerHistory:
    """Hold out a semantic composition after relocating its MOVE fixture."""

    from prp_wm.rulegrid import (
        ActionKind,
        CompositeAction,
        GridAction,
        RuleGridTransition,
        simulate,
    )

    prefix = tuple(task.inference.support[:6])
    activate = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.ACTIVATE
    )
    effectful_move = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.MOVE
        and transition.action.coord != (0, 0)
    )
    if len(activate) != 1 or len(effectful_move) != 1:
        raise ValueError(
            "translated semantic replay requires one ACTIVATE and one MOVE"
        )
    move = effectful_move[0]
    row_delta = target_row - move.action.coord[0]
    moved_state = _translate_public_grid_rows(move.state, row_delta)
    moved_next_state = _translate_public_grid_rows(move.next_state, row_delta)
    moved_action = GridAction(
        move.action.kind,
        (target_row, move.action.coord[1]),
        move.action.direction,
    )
    state = _overlay_public_grids((activate[0].state, moved_state))
    action = CompositeAction((activate[0].action, moved_action))
    target = simulate(
        state,
        action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    expected = _overlay_public_grids(
        (activate[0].next_state, moved_next_state)
    )
    if target != expected:
        raise AssertionError(
            "translated semantic composite was not entailed by public outcomes"
        )
    transition = RuleGridTransition(state, action, target)
    if transition in prefix:
        raise AssertionError("translated composite repeated a raw transition")
    return ControllerHistory(
        prefix + (transition,),
        (False,) * len(prefix) + (True,),
    )


def _informative_semantic_composite_controller_history(task: Any) -> ControllerHistory:
    """Compose a new informative trigger experiment with one known MOVE."""

    from prp_wm.rulegrid import (
        ActionKind,
        CompositeAction,
        RuleGridTransition,
        simulate,
    )

    prefix = tuple(task.inference.support[:6])
    trigger_probe = _public_trigger_symmetry_break_probe(task.inference)
    moves = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.MOVE
        and transition.action.coord != (0, 0)
    )
    if len(moves) != 1:
        raise ValueError("informative composite requires one observed MOVE")
    state = _overlay_public_grids((trigger_probe.state, moves[0].state))
    action = CompositeAction((trigger_probe.action, moves[0].action))
    target = simulate(
        state,
        action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    transition = RuleGridTransition(state, action, target)
    if transition in prefix:
        raise AssertionError("informative composite repeated a raw transition")
    return ControllerHistory(
        prefix + (transition,),
        (False,) * len(prefix) + (True,),
    )


def _translated_move_informative_composite_controller_history(
    task: Any,
    *,
    target_row: int = 6,
) -> ControllerHistory:
    """Relocate the known MOVE inside an informative composite holdout."""

    from prp_wm.rulegrid import (
        ActionKind,
        CompositeAction,
        GridAction,
        RuleGridTransition,
        simulate,
    )

    prefix = tuple(task.inference.support[:6])
    trigger_probe = _public_trigger_symmetry_break_probe(task.inference)
    moves = tuple(
        transition
        for transition in prefix
        if transition.action.kind is ActionKind.MOVE
        and transition.action.coord != (0, 0)
    )
    if len(moves) != 1:
        raise ValueError("translated informative composite requires one MOVE")
    move = moves[0]
    row_delta = target_row - move.action.coord[0]
    moved_state = _translate_public_grid_rows(move.state, row_delta)
    moved_action = GridAction(
        move.action.kind,
        (target_row, move.action.coord[1]),
        move.action.direction,
    )
    state = _overlay_public_grids((trigger_probe.state, moved_state))
    action = CompositeAction((trigger_probe.action, moved_action))
    target = simulate(
        state,
        action,
        task.privileged.true_program,
        task.privileged.palette,
    )
    transition = RuleGridTransition(state, action, target)
    if transition in prefix:
        raise AssertionError(
            "translated informative composite repeated a raw transition"
        )
    return ControllerHistory(
        prefix + (transition,),
        (False,) * len(prefix) + (True,),
    )


def _controller_probe_result_mask(
    torch: Any,
    controller_histories: Any,
    *,
    device: Any,
):
    selected = tuple(controller_histories)
    if not selected:
        raise ValueError("controller histories cannot be empty")
    steps = len(selected[0].transitions)
    if any(len(item.transitions) != steps for item in selected):
        raise ValueError("controller histories must have equal length")
    return torch.tensor(
        [item.is_agent_probe_result for item in selected],
        dtype=torch.bool,
        device=device,
    )


def _static_factor_belief_audit(
    *,
    torch: Any,
    model: Any,
    tasks: Any,
    batch_size: int,
    device: Any,
    make_support_batch: Any,
    histories: Any | None = None,
    controller_histories: Any | None = None,
) -> dict[str, object]:
    if histories is not None and controller_histories is not None:
        raise ValueError("pass histories or controller_histories, not both")
    selected_controller_histories = (
        None if controller_histories is None else tuple(controller_histories)
    )
    if selected_controller_histories is not None:
        if len(selected_controller_histories) != len(tasks):
            raise ValueError("controller histories must match tasks")
        histories = tuple(
            item.transitions for item in selected_controller_histories
        )
    if histories is not None and len(histories) != len(tasks):
        raise ValueError("histories must match tasks")
    totals = {
        "intersection": 0,
        "target": 0,
        "predicted": 0,
        "target_mass": 0.0,
        "factor_instances": 0,
        "exact_tasks": 0,
        "ambiguous_tasks": 0,
        "target_joint_hypotheses": 0,
        "predicted_joint_hypotheses": 0,
    }
    axis_intersection = [0, 0, 0]
    axis_target = [0, 0, 0]
    axis_predicted = [0, 0, 0]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[start : start + batch_size]
            batch_histories = (
                None
                if histories is None
                else histories[start : start + batch_size]
            )
            batch_controller_histories = (
                None
                if selected_controller_histories is None
                else selected_controller_histories[start : start + batch_size]
            )
            if batch_histories is None:
                batch = make_support_batch(torch, batch_tasks, device=device)
            else:
                from scripts.run_gram_public_coverage_finetune import (
                    _raw_public_history_batch,
                )

                batch = _raw_public_history_batch(
                    torch,
                    batch_histories,
                    device=device,
                )
            is_agent_probe_result = None
            if batch_controller_histories is not None and getattr(
                model,
                "supports_agent_probe_result_context",
                False,
            ):
                is_agent_probe_result = _controller_probe_result_mask(
                    torch,
                    batch_controller_histories,
                    device=device,
                )
            belief = model.infer_factor_belief(
                batch,
                is_agent_probe_result=is_agent_probe_result,
            )
            # Ground truth is deliberately materialized only after inference.
            compatible = _symmetry_expanded_version_space_mask(
                torch,
                model,
                batch_tasks,
                device=device,
                histories=batch_histories,
            )
            targets = model._factor_value_masks(model.factor_bank, compatible)
            predicted = model._threshold_factor_sets(belief.factor_probabilities)
            intersection = predicted & targets
            totals["intersection"] += int(intersection.sum().cpu())
            totals["target"] += int(targets.sum().cpu())
            totals["predicted"] += int(predicted.sum().cpu())
            totals["target_mass"] += float(
                (
                    belief.factor_probabilities
                    * targets.to(dtype=belief.factor_probabilities.dtype)
                ).sum().cpu()
            )
            totals["factor_instances"] += int(targets.shape[0] * 3)
            totals["exact_tasks"] += int(
                (predicted == targets).all(dim=(1, 2)).sum().cpu()
            )
            target_counts = targets.sum(dim=-1)
            predicted_counts = predicted.sum(dim=-1)
            target_products = target_counts.prod(dim=-1)
            predicted_products = predicted_counts.prod(dim=-1)
            totals["ambiguous_tasks"] += int((target_products > 4).sum().cpu())
            totals["target_joint_hypotheses"] += int(target_products.sum().cpu())
            totals["predicted_joint_hypotheses"] += int(
                predicted_products.sum().cpu()
            )
            for axis in range(3):
                axis_intersection[axis] += int(intersection[:, axis].sum().cpu())
                axis_target[axis] += int(targets[:, axis].sum().cpu())
                axis_predicted[axis] += int(predicted[:, axis].sum().cpu())
    recall_by_axis = [
        axis_intersection[axis] / axis_target[axis] for axis in range(3)
    ]
    precision_by_axis = [
        axis_intersection[axis] / axis_predicted[axis] for axis in range(3)
    ]
    metrics = {
        "tasks": len(tasks),
        "selection_rule": f"factor probability >= {model.factor_set_threshold:.2f}",
        "factor_set_recall": totals["intersection"] / totals["target"],
        "factor_set_precision": totals["intersection"] / totals["predicted"],
        "exact_task_factor_set_rate": totals["exact_tasks"] / len(tasks),
        "target_probability_mass": (
            totals["target_mass"] / totals["factor_instances"]
        ),
        "recall_by_axis": recall_by_axis,
        "precision_by_axis": precision_by_axis,
        "worst_axis_recall": min(recall_by_axis),
        "worst_axis_precision": min(precision_by_axis),
        "symmetry_ambiguous_tasks": totals["ambiguous_tasks"],
        "mean_target_joint_hypotheses": (
            totals["target_joint_hypotheses"] / len(tasks)
        ),
        "mean_predicted_joint_hypotheses": (
            totals["predicted_joint_hypotheses"] / len(tasks)
        ),
    }
    checks = {
        "factor_set_recall_gte_0_95": metrics["factor_set_recall"] >= 0.95,
        "factor_set_precision_gte_0_95": metrics["factor_set_precision"] >= 0.95,
        "exact_task_factor_set_rate_gte_0_90": (
            metrics["exact_task_factor_set_rate"] >= 0.90
        ),
        "target_probability_mass_gte_0_90": (
            metrics["target_probability_mass"] >= 0.90
        ),
        "worst_axis_recall_gte_0_90": metrics["worst_axis_recall"] >= 0.90,
        "worst_axis_precision_gte_0_90": (
            metrics["worst_axis_precision"] >= 0.90
        ),
    }
    return {
        "proposal_uses_privileged_palette_or_true_program": False,
        "controller_supplies_agent_probe_result_mask": (
            selected_controller_histories is not None
            and bool(getattr(model, "supports_agent_probe_result_context", False))
        ),
        "symmetry_ground_truth_materialized_after_inference": True,
        "metrics": metrics,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def load_public_version_k4_checkpoint(
    torch: Any,
    checkpoint_path: Path,
    *,
    device: Any,
    executor_checkpoint: Path | None = None,
):
    from prp_wm.public_version_k4 import (
        AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4,
        CompositeRelativeEventHistoryProbeFactorBeliefCausalK4,
        FactorizedPublicVersionSpaceCausalK4,
        HistoryConditionedProbeFactorBeliefCausalK4,
        PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4,
        ProbeAwareSymmetryFactorBeliefCausalK4,
        PublicVersionSpaceCausalK4,
        RelationalCompositeEventHistoryProbeFactorBeliefCausalK4,
        RelativeEventHistoryProbeFactorBeliefCausalK4,
        SymmetryAwareFactorBeliefCausalK4,
        TransitionEvidencePublicVersionSpaceCausalK4,
        TranslationInvariantHistoryProbeFactorBeliefCausalK4,
    )
    from scripts.run_expected_discrete_causal_coverage import _load_audited_executor

    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("unexpected public-version K4 checkpoint schema")
    executor_path = (
        Path(checkpoint["executor_checkpoint"])
        if executor_checkpoint is None
        else executor_checkpoint
    ).resolve()
    if _sha256_file(executor_path) != checkpoint["executor_checkpoint_sha256"]:
        raise SystemExit("executor checkpoint SHA256 drifted")
    executor, executor_metadata = _load_audited_executor(
        torch,
        executor_path,
        device,
    )
    model_classes = {
        "slot-joint": PublicVersionSpaceCausalK4,
        "factorized-set": FactorizedPublicVersionSpaceCausalK4,
        "transition-evidence": TransitionEvidencePublicVersionSpaceCausalK4,
        "symmetry-factor-belief": SymmetryAwareFactorBeliefCausalK4,
        "probe-aware-symmetry-factor-belief": (
            ProbeAwareSymmetryFactorBeliefCausalK4
        ),
        "history-conditioned-probe-factor-belief": (
            HistoryConditionedProbeFactorBeliefCausalK4
        ),
        "relative-event-history-probe-factor-belief": (
            RelativeEventHistoryProbeFactorBeliefCausalK4
        ),
        "composite-relative-event-history-probe-factor-belief": (
            CompositeRelativeEventHistoryProbeFactorBeliefCausalK4
        ),
        "relational-composite-event-history-probe-factor-belief": (
            RelationalCompositeEventHistoryProbeFactorBeliefCausalK4
        ),
        "atom-matched-composite-event-history-probe-factor-belief": (
            AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4
        ),
        "palette-invariant-atom-matched-composite-event-history-probe-factor-belief": (
            PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4
        ),
        "translation-invariant-history-probe-factor-belief": (
            TranslationInvariantHistoryProbeFactorBeliefCausalK4
        ),
    }
    version_head = checkpoint.get("version_head", "slot-joint")
    model_class = model_classes[version_head]
    recorded_model_type = checkpoint.get("model_type")
    if recorded_model_type not in (None, model_class.__name__):
        raise SystemExit("checkpoint model_type disagrees with version_head")
    if (
        version_head in PROBE_CONTEXT_HEADS
        and checkpoint.get("controller_context_schema")
        != "agent-probe-result-bool.v1"
    ):
        raise SystemExit("unexpected controller context schema")
    model = model_class(
        executor,
        attention_layers=int(checkpoint["attention_layers"]),
        temperature=float(checkpoint["factor_temperature"]),
        independent_support_encoders=bool(
            checkpoint.get("independent_support_encoders", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, executor_path, executor_metadata


def main() -> None:
    args = parse_args()
    _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.public_version_k4 import (
        AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4,
        CompositeRelativeEventHistoryProbeFactorBeliefCausalK4,
        FactorizedPublicVersionSpaceCausalK4,
        HistoryConditionedProbeFactorBeliefCausalK4,
        PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4,
        ProbeAwareSymmetryFactorBeliefCausalK4,
        PublicVersionSpaceCausalK4,
        RelationalCompositeEventHistoryProbeFactorBeliefCausalK4,
        RelativeEventHistoryProbeFactorBeliefCausalK4,
        SymmetryAwareFactorBeliefCausalK4,
        TransitionEvidencePublicVersionSpaceCausalK4,
        TranslationInvariantHistoryProbeFactorBeliefCausalK4,
    )
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_expected_discrete_causal_coverage import (
        _build_context_pool,
        _load_audited_executor,
    )
    from scripts.run_gram_public_coverage_finetune import (
        _public_support_batch,
        _raw_public_history_batch,
        _raw_public_support_batch_from_views,
        _mask_balance_audit,
    )
    from scripts.run_stratified_gram_public_coverage import _static_public_audit

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    executor_path = args.executor_checkpoint.resolve()
    executor, executor_metadata = _load_audited_executor(
        torch,
        executor_path,
        device,
    )
    model_classes = {
        "slot-joint": PublicVersionSpaceCausalK4,
        "factorized-set": FactorizedPublicVersionSpaceCausalK4,
        "transition-evidence": TransitionEvidencePublicVersionSpaceCausalK4,
        "symmetry-factor-belief": SymmetryAwareFactorBeliefCausalK4,
        "probe-aware-symmetry-factor-belief": (
            ProbeAwareSymmetryFactorBeliefCausalK4
        ),
        "history-conditioned-probe-factor-belief": (
            HistoryConditionedProbeFactorBeliefCausalK4
        ),
        "relative-event-history-probe-factor-belief": (
            RelativeEventHistoryProbeFactorBeliefCausalK4
        ),
        "composite-relative-event-history-probe-factor-belief": (
            CompositeRelativeEventHistoryProbeFactorBeliefCausalK4
        ),
        "relational-composite-event-history-probe-factor-belief": (
            RelationalCompositeEventHistoryProbeFactorBeliefCausalK4
        ),
        "atom-matched-composite-event-history-probe-factor-belief": (
            AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4
        ),
        "palette-invariant-atom-matched-composite-event-history-probe-factor-belief": (
            PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4
        ),
        "translation-invariant-history-probe-factor-belief": (
            TranslationInvariantHistoryProbeFactorBeliefCausalK4
        ),
    }
    model_class = model_classes[args.version_head]
    model = model_class(
        executor,
        attention_layers=args.attention_layers,
        temperature=args.factor_temperature,
        independent_support_encoders=args.student_support_encoders,
    ).to(device)
    if args.support_input == "raw":
        def make_support_batch(torch_module: Any, tasks: Any, *, device: Any):
            return _raw_public_support_batch_from_views(
                torch_module,
                tuple(task.inference for task in tasks),
                device=device,
            )

        audit_compatible_mask = _symbolic_version_space_mask
    else:
        make_support_batch = _public_support_batch
        audit_compatible_mask = None
    pool_arguments = {
        "make_pilot_tasks": make_pilot_tasks,
        "master_seed": args.data_master_seed,
        "factor_ids_for_program": rule_program_factor_ids,
        "version_space": version_space,
        "context_fold": args.context_fold,
    }
    train_pool = _build_context_pool(
        **pool_arguments,
        split=args.train_split,
        diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
        count=args.train_pool_tasks,
        heldout=False,
    )
    eval_pool = _build_context_pool(
        **pool_arguments,
        split=args.eval_split,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.eval_tasks,
        heldout=True,
    )
    train_symbolic_mask = _symbolic_version_space_mask(
        torch, model, train_pool, device=device
    )
    eval_symbolic_mask = _symbolic_version_space_mask(
        torch, model, eval_pool, device=device
    )
    train_balance = _mask_balance_audit(
        torch, model.factor_bank.cpu(), train_symbolic_mask.cpu()
    )
    eval_balance = _mask_balance_audit(
        torch, model.factor_bank.cpu(), eval_symbolic_mask.cpu()
    )
    train_contexts = {tuple(row) for row in train_balance["contexts"]}
    eval_contexts = {tuple(row) for row in eval_balance["contexts"]}
    if train_contexts.intersection(eval_contexts):
        raise AssertionError("train and held-out public contexts overlap")
    if len(train_contexts) != 36 or len(eval_contexts) != 12:
        raise AssertionError("expected 36 train and 12 held-out contexts")
    active_train_pool: tuple[Any, ...] = ()
    active_eval_pool: tuple[Any, ...] = ()
    active_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    neutral_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    hard_replay_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    semantic_composite_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    heldout_geometry_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    informative_composite_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    informative_geometry_eval_controller_histories: tuple[ControllerHistory, ...] = ()
    active_before: dict[str, object] | None = None
    neutral_before: dict[str, object] | None = None
    hard_replay_before: dict[str, object] | None = None
    semantic_composite_before: dict[str, object] | None = None
    heldout_geometry_before: dict[str, object] | None = None
    informative_composite_before: dict[str, object] | None = None
    informative_geometry_before: dict[str, object] | None = None
    if args.active_prefix_curriculum:
        train_expanded = _symmetry_expanded_version_space_mask(
            torch,
            model,
            train_pool,
            device=device,
        )
        eval_expanded = _symmetry_expanded_version_space_mask(
            torch,
            model,
            eval_pool,
            device=device,
        )
        active_train_pool = tuple(
            task
            for task, mask in zip(train_pool, train_expanded)
            if int(mask.sum().item()) > 4
        )
        active_eval_pool = tuple(
            task
            for task, mask in zip(eval_pool, eval_expanded)
            if int(mask.sum().item()) > 4
        )
        if len(active_train_pool) < args.batch_size or not active_eval_pool:
            raise AssertionError("active curriculum has too few ambiguous tasks")
        active_eval_controller_histories = tuple(
            _active_break_controller_history(task) for task in active_eval_pool
        )
        active_before = _static_factor_belief_audit(
            torch=torch,
            model=model,
            tasks=active_eval_pool,
            controller_histories=active_eval_controller_histories,
            batch_size=args.eval_batch_size,
            device=device,
            make_support_batch=make_support_batch,
        )
        if args.neutral_probe_curriculum:
            neutral_eval_controller_histories = tuple(
                _neutral_replay_controller_history(task)
                for task in active_eval_pool
            )
            neutral_before = _static_factor_belief_audit(
                torch=torch,
                model=model,
                tasks=active_eval_pool,
                controller_histories=neutral_eval_controller_histories,
                batch_size=args.eval_batch_size,
                device=device,
                make_support_batch=make_support_batch,
            )
            hard_replay_eval_controller_histories = tuple(
                _informative_then_replay_controller_history(task)
                for task in active_eval_pool
            )
            hard_replay_before = _static_factor_belief_audit(
                torch=torch,
                model=model,
                tasks=active_eval_pool,
                controller_histories=hard_replay_eval_controller_histories,
                batch_size=args.eval_batch_size,
                device=device,
                make_support_batch=make_support_batch,
            )
            if args.semantic_composite_curriculum:
                semantic_composite_eval_controller_histories = tuple(
                    _semantic_composite_replay_controller_history(task)
                    for task in active_eval_pool
                )
                semantic_prefix = _symmetry_expanded_version_space_mask(
                    torch,
                    model,
                    active_eval_pool,
                    device=device,
                )
                semantic_posterior = _symmetry_expanded_version_space_mask(
                    torch,
                    model,
                    active_eval_pool,
                    device=device,
                    histories=tuple(
                        item.transitions
                        for item in semantic_composite_eval_controller_histories
                    ),
                )
                if not torch.equal(semantic_prefix, semantic_posterior):
                    raise AssertionError(
                        "semantic composite replay changed the version space"
                    )
                semantic_composite_before = _static_factor_belief_audit(
                    torch=torch,
                    model=model,
                    tasks=active_eval_pool,
                    controller_histories=(
                        semantic_composite_eval_controller_histories
                    ),
                    batch_size=args.eval_batch_size,
                    device=device,
                    make_support_batch=make_support_batch,
                )
                heldout_geometry_eval_controller_histories = tuple(
                    _translated_move_semantic_composite_replay_controller_history(
                        task
                    )
                    for task in active_eval_pool
                )
                heldout_geometry_posterior = (
                    _symmetry_expanded_version_space_mask(
                        torch,
                        model,
                        active_eval_pool,
                        device=device,
                        histories=tuple(
                            item.transitions
                            for item in heldout_geometry_eval_controller_histories
                        ),
                    )
                )
                if not torch.equal(
                    semantic_prefix,
                    heldout_geometry_posterior,
                ):
                    raise AssertionError(
                        "held-out geometry replay changed the version space"
                    )
                heldout_geometry_before = _static_factor_belief_audit(
                    torch=torch,
                    model=model,
                    tasks=active_eval_pool,
                    controller_histories=(
                        heldout_geometry_eval_controller_histories
                    ),
                    batch_size=args.eval_batch_size,
                    device=device,
                    make_support_batch=make_support_batch,
                )
                if args.informative_composite_curriculum:
                    informative_composite_eval_controller_histories = tuple(
                        _informative_semantic_composite_controller_history(task)
                        for task in active_eval_pool
                    )
                    informative_posterior = _symmetry_expanded_version_space_mask(
                        torch,
                        model,
                        active_eval_pool,
                        device=device,
                        histories=tuple(
                            item.transitions
                            for item in informative_composite_eval_controller_histories
                        ),
                    )
                    if bool(
                        (informative_posterior & ~semantic_prefix).any().item()
                    ) or not bool(
                        informative_posterior.sum(dim=-1).eq(4).all().item()
                    ):
                        raise AssertionError(
                            "informative composite did not reduce 8 hypotheses to 4"
                        )
                    informative_composite_before = _static_factor_belief_audit(
                        torch=torch,
                        model=model,
                        tasks=active_eval_pool,
                        controller_histories=(
                            informative_composite_eval_controller_histories
                        ),
                        batch_size=args.eval_batch_size,
                        device=device,
                        make_support_batch=make_support_batch,
                    )
                    informative_geometry_eval_controller_histories = tuple(
                        _translated_move_informative_composite_controller_history(
                            task
                        )
                        for task in active_eval_pool
                    )
                    informative_geometry_posterior = (
                        _symmetry_expanded_version_space_mask(
                            torch,
                            model,
                            active_eval_pool,
                            device=device,
                            histories=tuple(
                                item.transitions
                                for item in informative_geometry_eval_controller_histories
                            ),
                        )
                    )
                    if not torch.equal(
                        informative_posterior,
                        informative_geometry_posterior,
                    ):
                        raise AssertionError(
                            "translated informative composite changed its posterior"
                        )
                    informative_geometry_before = _static_factor_belief_audit(
                        torch=torch,
                        model=model,
                        tasks=active_eval_pool,
                        controller_histories=(
                            informative_geometry_eval_controller_histories
                        ),
                        batch_size=args.eval_batch_size,
                        device=device,
                        make_support_batch=make_support_batch,
                    )
    if args.version_head in FACTOR_BELIEF_HEADS:
        before = _static_factor_belief_audit(
            torch=torch,
            model=model,
            tasks=eval_pool,
            batch_size=args.eval_batch_size,
            device=device,
            make_support_batch=make_support_batch,
        )
    else:
        before = _static_public_audit(
            torch=torch,
            model=model,
            tasks=eval_pool,
            batch_size=args.eval_batch_size,
            widths=(4,),
            device=device,
            make_public_batch=make_support_batch,
            compatible_mask_for_tasks=audit_compatible_mask,
        )
    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named or any(
        name.startswith("executor.") for name, _ in trainable_named
    ):
        raise AssertionError("executor entered the public K4 optimizer")
    trainable = [parameter for _, parameter in trainable_named]
    trainable_names = [name for name, _ in trainable_named]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(args.seed + 1)
    color_generator = torch.Generator(device="cpu")
    color_generator.manual_seed(args.seed + 2)
    sample_order = torch.randperm(len(train_pool), generator=sampler).tolist()
    sample_cursor = 0
    active_order = (
        torch.randperm(len(active_train_pool), generator=sampler).tolist()
        if active_train_pool
        else []
    )
    active_cursor = 0
    probe_cycle_tasks: tuple[Any, ...] | None = None
    probe_cycle_color_seed = args.seed + 1_000_000
    probe_cycle_index = 0
    latest: dict[str, object] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            learning_rate = (
                args.tail_learning_rate
                if args.tail_steps and step >= args.steps - args.tail_steps
                else args.learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if args.neutral_probe_curriculum:
                curriculum_phases = (
                    6
                    if args.informative_composite_curriculum
                    else 5
                    if args.semantic_composite_curriculum
                    else 4
                )
                curriculum_phase = step % curriculum_phases
                use_informative_probe = curriculum_phase == 1
                use_neutral_probe = curriculum_phase == 2
                use_hard_replay = curriculum_phase == 3
                use_semantic_composite = (
                    args.semantic_composite_curriculum
                    and curriculum_phase == 4
                )
                use_informative_composite = (
                    args.informative_composite_curriculum
                    and curriculum_phase == 5
                )
            else:
                use_informative_probe = (
                    args.active_prefix_curriculum and step % 2 == 1
                )
                use_neutral_probe = False
                use_hard_replay = False
                use_semantic_composite = False
                use_informative_composite = False
            use_probe_prefix = (
                use_informative_probe
                or use_neutral_probe
                or use_hard_replay
                or use_semantic_composite
                or use_informative_composite
            )
            training_histories: tuple[Any, ...] | None = None
            training_controller_histories: tuple[ControllerHistory, ...] | None = None
            if use_probe_prefix:
                if args.neutral_probe_curriculum and not use_informative_probe:
                    if probe_cycle_tasks is None:
                        raise AssertionError("paired probe cycle lost its tasks")
                    tasks = probe_cycle_tasks
                else:
                    if active_cursor + args.batch_size > len(active_order):
                        active_order = torch.randperm(
                            len(active_train_pool), generator=sampler
                        ).tolist()
                        active_cursor = 0
                    indices = active_order[
                        active_cursor : active_cursor + args.batch_size
                    ]
                    active_cursor += args.batch_size
                    tasks = tuple(active_train_pool[index] for index in indices)
                    if args.neutral_probe_curriculum:
                        probe_cycle_tasks = tasks
                        probe_cycle_color_seed = (
                            args.seed + 1_000_000 + probe_cycle_index
                        )
                        probe_cycle_index += 1
                training_controller_histories = tuple(
                    (
                        _informative_semantic_composite_controller_history(task)
                        if use_informative_composite
                        else _semantic_composite_replay_controller_history(task)
                        if use_semantic_composite
                        else _informative_then_replay_controller_history(task)
                        if use_hard_replay
                        else _neutral_replay_controller_history(task)
                        if use_neutral_probe
                        else _active_break_controller_history(task)
                    )
                    for task in tasks
                )
                training_histories = tuple(
                    item.transitions for item in training_controller_histories
                )
                batch = _raw_public_history_batch(
                    torch,
                    training_histories,
                    device=device,
                )
            else:
                if sample_cursor + args.batch_size > len(sample_order):
                    sample_order = torch.randperm(
                        len(train_pool), generator=sampler
                    ).tolist()
                    sample_cursor = 0
                indices = sample_order[
                    sample_cursor : sample_cursor + args.batch_size
                ]
                sample_cursor += args.batch_size
                tasks = tuple(train_pool[index] for index in indices)
                batch = make_support_batch(torch, tasks, device=device)
            if args.color_permutation_augmentation:
                augmentation_generator = color_generator
                if args.neutral_probe_curriculum and use_probe_prefix:
                    augmentation_generator = torch.Generator(device="cpu")
                    augmentation_generator.manual_seed(probe_cycle_color_seed)
                batch = _permute_raw_support_colors(
                    torch,
                    batch,
                    generator=augmentation_generator,
                    num_colors=model.config.num_colors,
                )
            optimizer.zero_grad(set_to_none=True)
            if args.version_head in FACTOR_BELIEF_HEADS:
                compatible_mask = _symmetry_expanded_version_space_mask(
                    torch,
                    model,
                    tasks,
                    device=device,
                    histories=training_histories,
                )
                evidence_axes, evidence_value_mask = (
                    _conditional_probe_innovation_targets(
                        torch,
                        model,
                        tasks,
                        device=device,
                        controller_histories=training_controller_histories,
                    )
                    if use_probe_prefix
                    else _symmetry_transition_evidence_targets(
                        torch,
                        tasks,
                        device=device,
                    )
                )
                loss = model.symmetry_aware_factor_belief_loss(
                    batch,
                    compatible_mask=compatible_mask,
                    evidence_axis_targets=evidence_axes,
                    evidence_value_target_mask=evidence_value_mask,
                    task_factor_weight=args.task_consistency_weight,
                    is_agent_probe_result=(
                        _controller_probe_result_mask(
                            torch,
                            training_controller_histories,
                            device=device,
                        )
                        if training_controller_histories is not None
                        and bool(
                            getattr(
                                model,
                                "supports_agent_probe_result_context",
                                False,
                            )
                        )
                        else None
                    ),
                )
            else:
                compatible_mask = (
                    _symbolic_version_space_mask(
                        torch,
                        model,
                        tasks,
                        device=device,
                    )
                    if args.support_input == "raw"
                    else None
                )
                loss_arguments = {
                    "compatible_mask": compatible_mask,
                    "margin_weight": args.margin_weight,
                    "validity_weight": args.validity_weight,
                    "joint_margin": args.joint_margin,
                    "temperature": args.factor_temperature,
                }
                if args.version_head == "factorized-set":
                    loss_arguments["varying_axis_weight"] = args.varying_axis_weight
                elif args.version_head == "transition-evidence":
                    evidence_axis_targets, evidence_value_targets = (
                        _symbolic_transition_evidence_targets(
                            torch,
                            tasks,
                            device=device,
                        )
                    )
                    loss_arguments.update(
                        {
                            "evidence_axis_targets": evidence_axis_targets,
                            "evidence_value_targets": evidence_value_targets,
                            "task_consistency_weight": args.task_consistency_weight,
                        }
                    )
                loss = model.hard_public_version_space_loss(
                    batch,
                    **loss_arguments,
                )
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite public K4 loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.max_grad_norm,
                ).detach().cpu()
            )
            optimizer.step()
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
                "training_prefix": (
                    "t0_plus_informative_semantic_composite"
                    if use_informative_composite
                    else "t0_plus_semantic_composite_replay"
                    if use_semantic_composite
                    else "t0_plus_informative_then_replay"
                    if use_hard_replay
                    else "t0_plus_neutral_probe_result"
                    if use_neutral_probe
                    else "t0_plus_informative_probe_result"
                    if use_informative_probe
                    else "t0"
                ),
            }
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record = {"step": completed, "tasks_seen": completed * args.batch_size, **latest}
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    training_seconds = time.perf_counter() - started
    if args.version_head in FACTOR_BELIEF_HEADS:
        after = _static_factor_belief_audit(
            torch=torch,
            model=model,
            tasks=eval_pool,
            batch_size=args.eval_batch_size,
            device=device,
            make_support_batch=make_support_batch,
        )
    else:
        after = _static_public_audit(
            torch=torch,
            model=model,
            tasks=eval_pool,
            batch_size=args.eval_batch_size,
            widths=(4,),
            device=device,
            make_public_batch=make_support_batch,
            compatible_mask_for_tasks=audit_compatible_mask,
        )
    active_after: dict[str, object] | None = None
    neutral_after: dict[str, object] | None = None
    hard_replay_after: dict[str, object] | None = None
    semantic_composite_after: dict[str, object] | None = None
    heldout_geometry_after: dict[str, object] | None = None
    informative_composite_after: dict[str, object] | None = None
    informative_geometry_after: dict[str, object] | None = None
    if args.active_prefix_curriculum:
        active_after = _static_factor_belief_audit(
            torch=torch,
            model=model,
            tasks=active_eval_pool,
            controller_histories=active_eval_controller_histories,
            batch_size=args.eval_batch_size,
            device=device,
            make_support_batch=make_support_batch,
        )
        if args.neutral_probe_curriculum:
            neutral_after = _static_factor_belief_audit(
                torch=torch,
                model=model,
                tasks=active_eval_pool,
                controller_histories=neutral_eval_controller_histories,
                batch_size=args.eval_batch_size,
                device=device,
                make_support_batch=make_support_batch,
            )
            hard_replay_after = _static_factor_belief_audit(
                torch=torch,
                model=model,
                tasks=active_eval_pool,
                controller_histories=hard_replay_eval_controller_histories,
                batch_size=args.eval_batch_size,
                device=device,
                make_support_batch=make_support_batch,
            )
            if args.semantic_composite_curriculum:
                semantic_composite_after = _static_factor_belief_audit(
                    torch=torch,
                    model=model,
                    tasks=active_eval_pool,
                    controller_histories=(
                        semantic_composite_eval_controller_histories
                    ),
                    batch_size=args.eval_batch_size,
                    device=device,
                    make_support_batch=make_support_batch,
                )
                heldout_geometry_after = _static_factor_belief_audit(
                    torch=torch,
                    model=model,
                    tasks=active_eval_pool,
                    controller_histories=(
                        heldout_geometry_eval_controller_histories
                    ),
                    batch_size=args.eval_batch_size,
                    device=device,
                    make_support_batch=make_support_batch,
                )
                if args.informative_composite_curriculum:
                    informative_composite_after = _static_factor_belief_audit(
                        torch=torch,
                        model=model,
                        tasks=active_eval_pool,
                        controller_histories=(
                            informative_composite_eval_controller_histories
                        ),
                        batch_size=args.eval_batch_size,
                        device=device,
                        make_support_batch=make_support_batch,
                    )
                    informative_geometry_after = _static_factor_belief_audit(
                        torch=torch,
                        model=model,
                        tasks=active_eval_pool,
                        controller_histories=(
                            informative_geometry_eval_controller_histories
                        ),
                        batch_size=args.eval_batch_size,
                        device=device,
                        make_support_batch=make_support_batch,
                    )
    checkpoint: dict[str, object] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": type(model).__name__,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "context_fold": args.context_fold,
        "attention_layers": args.attention_layers,
        "factor_temperature": args.factor_temperature,
        "support_input": args.support_input,
        "version_head": args.version_head,
        "independent_support_encoders": args.student_support_encoders,
        "color_permutation_augmentation": args.color_permutation_augmentation,
        "active_prefix_curriculum": args.active_prefix_curriculum,
        "neutral_probe_curriculum": args.neutral_probe_curriculum,
        "semantic_composite_curriculum": args.semantic_composite_curriculum,
        "informative_composite_curriculum": (
            args.informative_composite_curriculum
        ),
        "controller_context_schema": (
            "agent-probe-result-bool.v1"
            if args.version_head in PROBE_CONTEXT_HEADS
            else None
        ),
        "probe_relational_feature": (
            "has_identical_prior_public_transition"
            if args.version_head in HISTORY_CONTEXT_HEADS
            else None
        ),
        "support_coordinate_encoding": (
            "zero_frozen_absolute_row_and_column_embeddings"
            if args.version_head
            == "translation-invariant-history-probe-factor-belief"
            else "learned_absolute_row_and_column_embeddings"
        ),
        "probe_event_encoding": (
            "palette-invariant-causal-atom-matched-relative-event-set-composite-only.v5"
            if args.version_head
            == "palette-invariant-atom-matched-composite-event-history-probe-factor-belief"
            else "causal-atom-matched-relative-event-set-composite-only.v4"
            if args.version_head
            == "atom-matched-composite-event-history-probe-factor-belief"
            else "shared-history-nearest-action-relative-cell-set-composite-only.v3"
            if args.version_head
            == "relational-composite-event-history-probe-factor-belief"
            else "nearest-action-relative-cell-set-composite-only.v2"
            if args.version_head
            == "composite-relative-event-history-probe-factor-belief"
            else "nearest-action-relative-cell-set.v1"
            if args.version_head == "relative-event-history-probe-factor-belief"
            else None
        ),
        "controller_input_reads_privileged_palette": (
            args.support_input == "oracle-canonical"
        ),
        "model_config": asdict(model.config),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "latest_training_metrics": latest,
        "cli_arguments": _json_cli(args),
        "source_sha256": _source_sha256(),
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment": (
            "raw_controller_input_persistent_k4_version_space_abstraction"
            if args.support_input == "raw"
            else "oracle_palette_canonical_persistent_k4_version_space_abstraction"
        ),
        "status": "complete",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "cli_arguments": _json_cli(args),
        "context_fold": args.context_fold,
        "training_seed": args.seed,
        "training_steps": args.steps,
        "training_tasks_seen": args.steps * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "device": str(device),
        "persistent_slot_count": 4,
        "primary_belief_representation": (
            "three categorical factor value sets; Cartesian expansion is deferred"
            if args.version_head in FACTOR_BELIEF_HEADS
            else "four joint latent rule proposals"
        ),
        "fixed_k4_is_primary_representation": (
            args.version_head not in FACTOR_BELIEF_HEADS
        ),
        "support_input": args.support_input,
        "version_head": args.version_head,
        "independent_support_encoders": args.student_support_encoders,
        "color_permutation_augmentation": args.color_permutation_augmentation,
        "active_prefix_curriculum": args.active_prefix_curriculum,
        "neutral_probe_curriculum": args.neutral_probe_curriculum,
        "semantic_composite_curriculum": args.semantic_composite_curriculum,
        "informative_composite_curriculum": (
            args.informative_composite_curriculum
        ),
        "controller_context_schema": (
            "agent-probe-result-bool.v1"
            if args.version_head in PROBE_CONTEXT_HEADS
            else None
        ),
        "probe_relational_feature": (
            "has_identical_prior_public_transition"
            if args.version_head in HISTORY_CONTEXT_HEADS
            else None
        ),
        "support_coordinate_encoding": (
            "zero_frozen_absolute_row_and_column_embeddings"
            if args.version_head
            == "translation-invariant-history-probe-factor-belief"
            else "learned_absolute_row_and_column_embeddings"
        ),
        "probe_event_encoding": (
            "palette-invariant-causal-atom-matched-relative-event-set-composite-only.v5"
            if args.version_head
            == "palette-invariant-atom-matched-composite-event-history-probe-factor-belief"
            else "causal-atom-matched-relative-event-set-composite-only.v4"
            if args.version_head
            == "atom-matched-composite-event-history-probe-factor-belief"
            else "shared-history-nearest-action-relative-cell-set-composite-only.v3"
            if args.version_head
            == "relational-composite-event-history-probe-factor-belief"
            else "nearest-action-relative-cell-set-composite-only.v2"
            if args.version_head
            == "composite-relative-event-history-probe-factor-belief"
            else "nearest-action-relative-cell-set.v1"
            if args.version_head == "relative-event-history-probe-factor-belief"
            else None
        ),
        "controller_input_reads_privileged_palette": (
            args.support_input == "oracle-canonical"
        ),
        "controller_input_fields": (
            (
                "inference.support(state, action, observed next_state) plus "
                "controller-owned is_agent_probe_result bit"
            )
            if args.version_head in PROBE_CONTEXT_HEADS
            else "inference.support(state, action, observed next_state) only"
            if args.support_input == "raw"
            else (
                "inference.support plus simulator privileged.palette role "
                "canonicalization"
            )
        ),
        "training_teacher_reads_simulator_palette": True,
        "training_teacher_reads_true_program": False,
        "frozen_executor_pretrained_on_oracle_canonical_roles": True,
        "environment_true_program_used_only_to_generate_observed_active_result": (
            args.active_prefix_curriculum
        ),
        "support_encoder_trainable": True,
        "executor_frozen": True,
        "query_behavior_or_true_program_used_for_training": False,
        "controller_probe_phase_signal": (
            args.version_head in PROBE_CONTEXT_HEADS
        ),
        "probe_evidence_history_conditioned": (
            args.version_head in HISTORY_CONTEXT_HEADS
        ),
        "controller_probe_mask_built_before_version_space_teacher": (
            args.active_prefix_curriculum
            and args.version_head in PROBE_CONTEXT_HEADS
        ),
        "public_version_space_teacher": (
            (
                "union of symbolic version spaces under payload_p1/payload_p2 "
                "role-name symmetry; labels only"
            )
            if args.version_head in FACTOR_BELIEF_HEADS
            else (
                "independent symbolic enumeration using simulator palette; labels only"
                if args.support_input == "raw"
                else "all-64 frozen-executor MAP exact equality after oracle canonicalization"
            )
        ),
        "active_innovation_teacher": (
            (
                "each controller-marked probe result is supervised by the "
                "factor-set reduction from its preceding public history to "
                "the updated public history"
            )
            if args.active_prefix_curriculum
            else None
        ),
        "canonical_assignment": (
            None
            if args.version_head in FACTOR_BELIEF_HEADS
            else "slot k receives compatible code whose varying-axis value is k"
        ),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_schema": executor_metadata.get("checkpoint_schema_version"),
        "train_public_version_space_audit": train_balance,
        "eval_public_version_space_audit": eval_balance,
        "static_audit_before": before,
        "static_audit_after": after,
        "active_prefix_eval_tasks": len(active_eval_pool),
        "active_prefix_audit_before": active_before,
        "active_prefix_audit_after": active_after,
        "neutral_probe_eval_tasks": (
            len(active_eval_pool) if args.neutral_probe_curriculum else 0
        ),
        "neutral_probe_audit_before": neutral_before,
        "neutral_probe_audit_after": neutral_after,
        "hard_replay_audit_before": hard_replay_before,
        "hard_replay_audit_after": hard_replay_after,
        "semantic_composite_eval_tasks": (
            len(active_eval_pool) if args.semantic_composite_curriculum else 0
        ),
        "semantic_composite_audit_before": semantic_composite_before,
        "semantic_composite_audit_after": semantic_composite_after,
        "heldout_geometry_composite_target_row": (
            6 if args.semantic_composite_curriculum else None
        ),
        "heldout_geometry_composite_used_for_training": False,
        "heldout_geometry_composite_audit_before": heldout_geometry_before,
        "heldout_geometry_composite_audit_after": heldout_geometry_after,
        "informative_composite_eval_tasks": (
            len(active_eval_pool)
            if args.informative_composite_curriculum
            else 0
        ),
        "informative_composite_audit_before": informative_composite_before,
        "informative_composite_audit_after": informative_composite_after,
        "informative_geometry_composite_target_row": (
            6 if args.informative_composite_curriculum else None
        ),
        "informative_geometry_composite_used_for_training": False,
        "informative_geometry_composite_audit_before": (
            informative_geometry_before
        ),
        "informative_geometry_composite_audit_after": informative_geometry_after,
        "latest_training_metrics": latest,
        "source_sha256": _source_sha256(),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
