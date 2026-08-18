#!/usr/bin/env python3
"""Continue an active-prefix executor with randomized singleton locality.

This is a privileged GeomSup/LocReg diagnostic, not an ARC agent.  It starts
from an audited active-support-calibrated ``OracleFactorExecutor`` and keeps
the original five active-prefix replay domains.  At every continuation step it
also draws one randomized-geometry singleton for each known mechanism axis
and materializes all 64 privileged factor codes.

GeomSup is the executor's proper plus sparse-change-balanced NLL on those
singletons.  LocReg exploits a singleton's known acted axis during training:
for each of its four acted-axis values, the true target is identical over the
16 nuisance-code settings.  The regularizer is a SmoothL1/Huber dispersion of
the corresponding full-grid true-target log likelihoods around their fiber
mean.  A zero ``--locality-weight`` is therefore the matched GeomSup control.

An optional frozen-teacher term distils the initial executor's complete
categorical next-cell distribution on the original replay domains.  This is a
trust-region diagnostic: it does not constrain the new random-geometry
singletons and it never reads the teacher at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from prp_wm.rulegrid import MASTER_SEED


DEFAULT_INITIAL_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs/active_support_calibrated_executor_cont300_seed20260731"
    / "checkpoint_last.pt"
)
CHECKPOINT_SCHEMA_VERSION = "prp-wm.counterfactual-locality-finetune.v3"
EXPECTED_INITIAL_SCHEMA_VERSION = (
    "prp-wm.active-support-calibrated-factor-executor.v1"
)
_AUDITED_SOURCE_FILES = (
    "prp_wm/latent_rules.py",
    "prp_wm/matched_executor.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/random_geometry_protocol.py",
    "prp_wm/routed_executor.py",
    "prp_wm/rulegrid.py",
    "scripts/run_active_support_calibrated_executor.py",
    "scripts/run_causal_mechanism_coverage.py",
    "scripts/run_counterfactual_locality_finetune.py",
    "scripts/run_rulegrid_executor_ceiling.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=DEFAULT_INITIAL_CHECKPOINT,
    )
    parser.add_argument(
        "--executor-architecture",
        choices=(
            "global",
            "canonical-role-routed",
            "matched-wider-global",
            "matched-factor-local",
        ),
        default="global",
        help=(
            "Choose the original global executor, the 17k routed diagnostic, "
            "or one of the parameter-identical four-branch P1 conditions."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026072402)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--codes-per-task", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--diagnostic-loss-weight", type=float, default=0.40)
    parser.add_argument(
        "--stage-loss-weights",
        type=float,
        nargs=4,
        default=(0.10, 0.15, 0.10, 0.25),
        metavar=("T0", "T1", "T2", "T3"),
    )
    parser.add_argument(
        "--geometry-weight",
        type=float,
        default=0.10,
        help="Weight on singleton proper+balanced geometry supervision.",
    )
    parser.add_argument(
        "--geometry-axis-scope",
        choices=("singleton", "singleton-pair"),
        default="singleton",
        help=(
            "Use the historical three singleton panels per update or the P1 "
            "matched protocol with three singleton and three pair panels."
        ),
    )
    parser.add_argument(
        "--geometry-train-seed-count",
        type=int,
        default=0,
        help=(
            "Number of randomized geometry seeds to cycle through in a "
            "seed-specific fixed permutation; zero uses one new seed per step."
        ),
    )
    parser.add_argument(
        "--active-task-start-offset",
        type=int,
        default=0,
        help="Starting pilot-task index for this continuation replicate.",
    )
    parser.add_argument(
        "--locality-weight",
        type=float,
        default=0.10,
        help="Final locality weight; use 0 for the matched GeomSup control.",
    )
    parser.add_argument(
        "--locality-ramp-steps",
        type=int,
        default=50,
        help="Linear ramp length for locality weight; zero applies it immediately.",
    )
    parser.add_argument("--locality-huber-beta", type=float, default=1.0)
    parser.add_argument(
        "--teacher-distillation-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on frozen-initial-teacher categorical KL over the original "
            "active replay domains; zero disables distillation."
        ),
    )
    parser.add_argument("--geometry-train-seed-base", type=int, default=20_000_000)
    parser.add_argument("--geometry-eval-seed-base", type=int, default=30_000_000)
    parser.add_argument("--heldout-singleton-seeds", type=int, default=8)
    parser.add_argument("--heldout-geometry-batch-panels", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train-split", default="counterfactual-locality-active-replay-train"
    )
    parser.add_argument(
        "--eval-split", default="counterfactual-locality-active-prefix-heldout"
    )
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


def _validate_args(args: argparse.Namespace) -> tuple[float, tuple[float, ...]]:
    positive_ints = (
        "steps",
        "batch_size",
        "codes_per_task",
        "eval_tasks",
        "eval_batch_size",
        "heldout_singleton_seeds",
        "heldout_geometry_batch_panels",
        "log_every",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.codes_per_task > 64:
        raise SystemExit("--codes-per-task cannot exceed 64")
    if args.locality_ramp_steps < 0:
        raise SystemExit("--locality-ramp-steps must be non-negative")
    if args.geometry_train_seed_count < 0:
        raise SystemExit("--geometry-train-seed-count must be non-negative")
    for name in ("learning_rate", "max_grad_norm", "locality_huber_beta"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "balanced_weight",
        "geometry_weight",
        "locality_weight",
        "teacher_distillation_weight",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    stage_weights = tuple(float(value) for value in args.stage_loss_weights)
    if len(stage_weights) != 4 or min(stage_weights) < 0:
        raise SystemExit("--stage-loss-weights needs four non-negative values")
    diagnostic_weight = float(args.diagnostic_loss_weight)
    if diagnostic_weight < 0 or abs(diagnostic_weight + sum(stage_weights) - 1.0) > 1e-9:
        raise SystemExit("diagnostic and stage replay loss weights must sum to one")
    for name in (
        "seed",
        "data_master_seed",
        "geometry_train_seed_base",
        "geometry_eval_seed_base",
        "active_task_start_offset",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    geometry_seed_count = args.geometry_train_seed_count or args.steps
    train_seeds = set(
        range(
            args.geometry_train_seed_base,
            args.geometry_train_seed_base + geometry_seed_count,
        )
    )
    eval_seeds = set(
        range(
            args.geometry_eval_seed_base,
            args.geometry_eval_seed_base + args.heldout_singleton_seeds,
        )
    )
    if train_seeds.intersection(eval_seeds):
        raise SystemExit("train and held-out geometry seed streams must be disjoint")
    if args.train_split == args.eval_split:
        raise SystemExit("active replay train and evaluation splits must differ")
    return diagnostic_weight, stage_weights


def _locality_ramp(final_weight: float, completed_step: int, ramp_steps: int) -> float:
    """Return a deterministic linear warm-up that reaches the final weight."""

    if final_weight < 0:
        raise ValueError("final_weight must be non-negative")
    if completed_step <= 0:
        raise ValueError("completed_step must be positive")
    if ramp_steps < 0:
        raise ValueError("ramp_steps must be non-negative")
    if ramp_steps == 0:
        return float(final_weight)
    return float(final_weight) * min(1.0, completed_step / ramp_steps)


def _singleton_panels(geometry_seed: int) -> tuple[Any, ...]:
    """Draw exactly one train-scope singleton for each known mechanism axis."""

    from prp_wm.random_geometry_protocol import AXES, make_geometry_panel

    return tuple(
        make_geometry_panel(
            split="train",
            geometry_seed=geometry_seed,
            axes=(axis,),
        )
        for axis in AXES
    )


def _build_singleton_batch(
    *,
    torch: Any,
    panels: Sequence[Any],
    device: Any,
) -> Any:
    """Materialize each singleton under all 64 privileged factor codes."""

    from prp_wm.latent_rules import OracleFactorBatch
    from prp_wm.neural import encode_public_action
    from prp_wm.random_geometry_protocol import (
        AXES,
        FACTOR_CODES,
        factor_code_to_program,
    )
    from prp_wm.rulegrid import DEFAULT_PALETTE, simulate

    materialized = tuple(panels)
    if not materialized:
        raise ValueError("panels cannot be empty")
    axis_to_index = {axis: index for index, axis in enumerate(AXES)}
    acted_axes: list[int] = []
    for panel in materialized:
        if panel.panel_kind != "singleton" or len(panel.axes) != 1:
            raise ValueError("locality batches require singleton panels")
        acted_axes.append(axis_to_index[panel.axes[0]])

    code_count = len(FACTOR_CODES)
    panel_count = len(materialized)
    states = torch.tensor(
        [panel.state for panel in materialized],
        dtype=torch.long,
        device=device,
    )
    actions = torch.stack(
        [encode_public_action(panel.action) for panel in materialized]
    ).to(device)
    if actions.shape[1:] != (1, 4):
        raise AssertionError("singleton actions must encode as one atom")
    factors = torch.tensor(FACTOR_CODES, dtype=torch.long, device=device)
    targets = torch.tensor(
        [
            simulate(
                panel.state,
                panel.action,
                factor_code_to_program(code),
                DEFAULT_PALETTE,
            )
            for panel in materialized
            for code in FACTOR_CODES
        ],
        dtype=torch.long,
        device=device,
    )
    batch = OracleFactorBatch(
        states=states[:, None, None]
        .expand(-1, code_count, 1, -1, -1)
        .reshape(panel_count * code_count, 1, *states.shape[-2:]),
        actions=actions[:, None]
        .expand(-1, code_count, -1, -1)
        .reshape(panel_count * code_count, 1, 4),
        targets=targets[:, None],
        factor_ids=factors[None]
        .expand(panel_count, -1, -1)
        .reshape(panel_count * code_count, 3),
        action_mask=None,
        palette_canonicalized=True,
    )

    # The simulator, not a learned prediction, certifies that every nuisance
    # setting in a singleton fiber has the same supervision target.
    reshaped_targets = batch.targets.reshape(
        panel_count, code_count, *batch.targets.shape[1:]
    )
    for panel_index, axis_index in enumerate(acted_axes):
        for value in range(4):
            mask = factors[:, axis_index].eq(value)
            fiber = reshaped_targets[panel_index, mask]
            if fiber.shape[0] != 16:
                raise AssertionError("a singleton factor fiber must contain 16 codes")
            if not bool(fiber.eq(fiber[:1]).all().item()):
                raise AssertionError("singleton nuisance factors changed the target")
    return batch


def _geometry_panels(geometry_seed: int) -> tuple[Any, ...]:
    """Draw the P1 train set: all three singleton and three pair axis sets."""

    from prp_wm.random_geometry_protocol import TRAIN_AXIS_SETS, make_geometry_panel

    return tuple(
        make_geometry_panel(
            split="train",
            geometry_seed=geometry_seed,
            axes=axes,
        )
        for axes in TRAIN_AXIS_SETS
    )


def _build_geometry_batch(
    *,
    torch: Any,
    panels: Sequence[Any],
    device: Any,
) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    """Materialize singleton/pair panels under the same ordered 64-code bank."""

    from prp_wm.latent_rules import OracleFactorBatch
    from prp_wm.neural import encode_public_action
    from prp_wm.random_geometry_protocol import (
        AXES,
        FACTOR_CODES,
        factor_code_to_program,
    )
    from prp_wm.rulegrid import DEFAULT_PALETTE, simulate

    materialized = tuple(panels)
    if not materialized:
        raise ValueError("panels cannot be empty")
    axis_to_index = {axis: index for index, axis in enumerate(AXES)}
    active_axis_sets: list[tuple[int, ...]] = []
    encoded_actions = []
    for panel in materialized:
        if panel.panel_kind not in ("singleton", "pair"):
            raise ValueError("P1 geometry batches require singleton/pair panels")
        active_axis_sets.append(
            tuple(axis_to_index[axis] for axis in panel.axes)
        )
        encoded_actions.append(encode_public_action(panel.action))

    code_count = len(FACTOR_CODES)
    panel_count = len(materialized)
    states = torch.tensor(
        [panel.state for panel in materialized],
        dtype=torch.long,
        device=device,
    )
    factors = torch.tensor(FACTOR_CODES, dtype=torch.long, device=device)
    targets = torch.tensor(
        [
            simulate(
                panel.state,
                panel.action,
                factor_code_to_program(code),
                DEFAULT_PALETTE,
            )
            for panel in materialized
            for code in FACTOR_CODES
        ],
        dtype=torch.long,
        device=device,
    )
    max_atoms = max(action.shape[0] for action in encoded_actions)
    padded_actions = torch.zeros(
        panel_count,
        max_atoms,
        4,
        dtype=torch.long,
        device=device,
    )
    panel_action_mask = torch.zeros(
        panel_count,
        max_atoms,
        dtype=torch.bool,
        device=device,
    )
    for panel_index, action in enumerate(encoded_actions):
        atom_count = action.shape[0]
        padded_actions[panel_index, :atom_count] = action.to(device)
        panel_action_mask[panel_index, :atom_count] = True
    expanded_actions = padded_actions[:, None].expand(
        -1,
        code_count,
        -1,
        -1,
    ).reshape(panel_count * code_count, 1, max_atoms, 4)
    expanded_mask = panel_action_mask[:, None].expand(
        -1,
        code_count,
        -1,
    ).reshape(panel_count * code_count, 1, max_atoms)
    if max_atoms == 1:
        expanded_actions = expanded_actions[:, :, 0]
        action_mask = None
    else:
        action_mask = expanded_mask
    batch = OracleFactorBatch(
        states=states[:, None, None]
        .expand(-1, code_count, 1, -1, -1)
        .reshape(panel_count * code_count, 1, *states.shape[-2:]),
        actions=expanded_actions,
        targets=targets[:, None],
        factor_ids=factors[None]
        .expand(panel_count, -1, -1)
        .reshape(panel_count * code_count, 3),
        action_mask=action_mask,
        palette_canonicalized=True,
    )

    reshaped_targets = batch.targets.reshape(
        panel_count,
        code_count,
        *batch.targets.shape[1:],
    )
    for panel_index, active_axes in enumerate(active_axis_sets):
        for active_values in itertools.product(
            range(4),
            repeat=len(active_axes),
        ):
            fiber_mask = torch.ones(
                code_count,
                dtype=torch.bool,
                device=device,
            )
            for axis, value in zip(active_axes, active_values, strict=True):
                fiber_mask &= factors[:, axis].eq(value)
            fiber = reshaped_targets[panel_index, fiber_mask]
            expected = 4 ** (3 - len(active_axes))
            if fiber.shape[0] != expected:
                raise AssertionError("active-axis fiber has the wrong size")
            if not bool(fiber.eq(fiber[:1]).all().item()):
                raise AssertionError("nuisance factors changed a geometry target")
    return batch, tuple(active_axis_sets)


def _fiber_deviations(
    *,
    torch: Any,
    full_grid_log_likelihood: Any,
    factor_codes: Any,
    acted_axis_indices: Sequence[int],
) -> tuple[Any, tuple[Any, ...]]:
    """Center singleton scores inside each four-by-sixteen factor fiber."""

    if full_grid_log_likelihood.ndim != 2:
        raise ValueError("full_grid_log_likelihood must have [P,64] shape")
    if factor_codes.shape != (full_grid_log_likelihood.shape[1], 3):
        raise ValueError("factor_codes must have [64,3] matching the score bank")
    if len(acted_axis_indices) != full_grid_log_likelihood.shape[0]:
        raise ValueError("one acted axis is required per panel")
    centered: list[Any] = []
    ranges: list[Any] = []
    for panel_index, raw_axis in enumerate(acted_axis_indices):
        axis = int(raw_axis)
        if axis not in range(3):
            raise ValueError("acted axes must be integer indices in [0,3)")
        for value in range(4):
            fiber = full_grid_log_likelihood[
                panel_index, factor_codes[:, axis].eq(value)
            ]
            if fiber.numel() != 16:
                raise AssertionError("each acted-value fiber must contain 16 codes")
            centered.append(fiber - fiber.mean())
            ranges.append(fiber.max() - fiber.min())
    return torch.cat(centered), tuple(ranges)


def _fiber_huber_dispersion(
    *,
    torch: Any,
    full_grid_log_likelihood: Any,
    factor_codes: Any,
    acted_axis_indices: Sequence[int],
    beta: float,
) -> Any:
    """SmoothL1/Huber dispersion of true-target log likelihood within fibers."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    deviations, _ = _fiber_deviations(
        torch=torch,
        full_grid_log_likelihood=full_grid_log_likelihood,
        factor_codes=factor_codes,
        acted_axis_indices=acted_axis_indices,
    )
    return torch.nn.functional.smooth_l1_loss(
        deviations,
        torch.zeros_like(deviations),
        beta=beta,
        reduction="mean",
    )


def _active_set_fiber_deviations(
    *,
    torch: Any,
    full_grid_log_likelihood: Any,
    factor_codes: Any,
    active_axis_sets: Sequence[Sequence[int]],
) -> tuple[Any, tuple[Any, ...]]:
    """Center scores after holding every acted-axis tuple fixed."""

    if full_grid_log_likelihood.ndim != 2:
        raise ValueError("full_grid_log_likelihood must have [P,64] shape")
    if factor_codes.shape != (full_grid_log_likelihood.shape[1], 3):
        raise ValueError("factor_codes must have [64,3] matching the score bank")
    if len(active_axis_sets) != full_grid_log_likelihood.shape[0]:
        raise ValueError("one active-axis set is required per panel")
    centered: list[Any] = []
    ranges: list[Any] = []
    for panel_index, raw_axes in enumerate(active_axis_sets):
        axes = tuple(int(axis) for axis in raw_axes)
        if (
            not axes
            or len(set(axes)) != len(axes)
            or any(axis not in range(3) for axis in axes)
        ):
            raise ValueError("active-axis sets must be unique non-empty subsets")
        for active_values in itertools.product(range(4), repeat=len(axes)):
            fiber_mask = torch.ones(
                factor_codes.shape[0],
                dtype=torch.bool,
                device=factor_codes.device,
            )
            for axis, value in zip(axes, active_values, strict=True):
                fiber_mask &= factor_codes[:, axis].eq(value)
            fiber = full_grid_log_likelihood[panel_index, fiber_mask]
            expected = 4 ** (3 - len(axes))
            if fiber.numel() != expected:
                raise AssertionError("active-axis likelihood fiber has wrong size")
            centered.append(fiber - fiber.mean())
            ranges.append(fiber.max() - fiber.min())
    return torch.cat(centered), tuple(ranges)


def _active_set_fiber_huber_dispersion(
    *,
    torch: Any,
    full_grid_log_likelihood: Any,
    factor_codes: Any,
    active_axis_sets: Sequence[Sequence[int]],
    beta: float,
) -> Any:
    if beta <= 0:
        raise ValueError("beta must be positive")
    deviations, _ = _active_set_fiber_deviations(
        torch=torch,
        full_grid_log_likelihood=full_grid_log_likelihood,
        factor_codes=factor_codes,
        active_axis_sets=active_axis_sets,
    )
    return torch.nn.functional.smooth_l1_loss(
        deviations,
        torch.zeros_like(deviations),
        beta=beta,
        reduction="mean",
    )


def _outcome_log_probabilities(*, torch: Any, prediction: Any) -> Any:
    """Return normalized categorical cell outcomes as ``[B,K,C,H,W]``."""

    if prediction.new_color_logits.ndim != 5:
        raise ValueError("new_color_logits must have [B,K,C,H,W] shape")
    batch, modes, colors, height, width = prediction.new_color_logits.shape
    if prediction.change_logits.shape != (batch, modes, height, width):
        raise ValueError("change_logits must match [B,K,H,W]")
    if prediction.input_colors.shape != (batch, height, width):
        raise ValueError("input_colors must match [B,H,W]")
    original = prediction.input_colors[:, None, None]
    color_ids = torch.arange(
        colors,
        device=prediction.new_color_logits.device,
    )[None, None, :, None, None]
    masked_color_logits = prediction.new_color_logits.masked_fill(
        original == color_ids,
        torch.finfo(prediction.new_color_logits.dtype).min,
    )
    changed = torch.nn.functional.logsigmoid(
        prediction.change_logits
    ).unsqueeze(2) + torch.nn.functional.log_softmax(
        masked_color_logits,
        dim=2,
    )
    unchanged = torch.nn.functional.logsigmoid(
        -prediction.change_logits
    ).unsqueeze(2)
    return changed.scatter(
        2,
        original.expand(-1, modes, -1, -1, -1),
        unchanged,
    )


def _teacher_categorical_kl(
    *,
    torch: Any,
    student_prediction: Any,
    teacher_prediction: Any,
) -> Any:
    """Mean ``KL(teacher || student)`` in nats per mode and grid cell."""

    if not bool(
        student_prediction.input_colors.eq(teacher_prediction.input_colors).all().item()
    ):
        raise ValueError("student and teacher predictions must share input colors")
    student_log = _outcome_log_probabilities(
        torch=torch,
        prediction=student_prediction,
    )
    teacher_log = _outcome_log_probabilities(
        torch=torch,
        prediction=teacher_prediction,
    ).detach()
    if student_log.shape != teacher_log.shape:
        raise ValueError("student and teacher outcome spaces must match")
    return (
        teacher_log.exp() * (teacher_log - student_log)
    ).sum(dim=2).mean()


def _singleton_objective(
    *,
    torch: Any,
    model: Any,
    batch: Any,
    acted_axis_indices: Sequence[int],
    balanced_weight: float,
    huber_beta: float,
) -> dict[str, Any]:
    """Compute GeomSup and LocReg from a single shared model prediction."""

    if balanced_weight < 0:
        raise ValueError("balanced_weight must be non-negative")
    batch.validate(model.config)
    prediction = model.predict_panel(batch)
    height, width = batch.targets.shape[-2:]
    flat_states = batch.states.reshape(-1, height, width)
    flat_targets = batch.targets.reshape(-1, height, width)
    cell_nll = -prediction.log_prob_cells(flat_targets).squeeze(1)
    changed = flat_targets.ne(flat_states)

    def masked_mean(values: Any, mask: Any) -> Any:
        return (
            values.masked_select(mask).mean()
            if bool(mask.any().item())
            else values.new_zeros(())
        )

    changed_nll = masked_mean(cell_nll, changed)
    unchanged_nll = masked_mean(cell_nll, ~changed)
    balanced_nll = (
        0.5 * (changed_nll + unchanged_nll)
        if bool(changed.any().item()) and bool((~changed).any().item())
        else changed_nll + unchanged_nll
    )
    proper_nll = cell_nll.mean()
    geometry_supervision = proper_nll + balanced_weight * balanced_nll
    panel_count = len(acted_axis_indices)
    if batch.batch_size != panel_count * 64:
        raise ValueError("singleton objective expects all 64 codes per panel")
    full_grid_log_likelihood = prediction.log_prob(flat_targets).squeeze(1).reshape(
        panel_count, 64
    )
    factor_codes = batch.factor_ids.reshape(panel_count, 64, 3)[0]
    if not bool(
        batch.factor_ids.reshape(panel_count, 64, 3)
        .eq(factor_codes[None])
        .all()
        .item()
    ):
        raise ValueError("every singleton panel must use the same factor bank order")
    locality = _fiber_huber_dispersion(
        torch=torch,
        full_grid_log_likelihood=full_grid_log_likelihood,
        factor_codes=factor_codes,
        acted_axis_indices=acted_axis_indices,
        beta=huber_beta,
    )
    return {
        "prediction": prediction,
        "cell_nll": cell_nll,
        "full_grid_log_likelihood": full_grid_log_likelihood,
        "proper_nll": proper_nll,
        "balanced_nll": balanced_nll,
        "changed_nll": changed_nll,
        "unchanged_nll": unchanged_nll,
        "geometry_supervision": geometry_supervision,
        "locality": locality,
    }


def _geometry_objective(
    *,
    torch: Any,
    model: Any,
    batch: Any,
    active_axis_sets: Sequence[Sequence[int]],
    balanced_weight: float,
    huber_beta: float,
) -> dict[str, Any]:
    """GeomSup/LocReg for matched singleton and pair panels."""

    if balanced_weight < 0:
        raise ValueError("balanced_weight must be non-negative")
    batch.validate(model.config)
    prediction = model.predict_panel(batch)
    height, width = batch.targets.shape[-2:]
    flat_states = batch.states.reshape(-1, height, width)
    flat_targets = batch.targets.reshape(-1, height, width)
    cell_nll = -prediction.log_prob_cells(flat_targets).squeeze(1)
    changed = flat_targets.ne(flat_states)

    def masked_mean(values: Any, mask: Any) -> Any:
        return (
            values.masked_select(mask).mean()
            if bool(mask.any().item())
            else values.new_zeros(())
        )

    changed_nll = masked_mean(cell_nll, changed)
    unchanged_nll = masked_mean(cell_nll, ~changed)
    balanced_nll = (
        0.5 * (changed_nll + unchanged_nll)
        if bool(changed.any().item()) and bool((~changed).any().item())
        else changed_nll + unchanged_nll
    )
    proper_nll = cell_nll.mean()
    geometry_supervision = proper_nll + balanced_weight * balanced_nll
    panel_count = len(active_axis_sets)
    if batch.batch_size != panel_count * 64:
        raise ValueError("geometry objective expects all 64 codes per panel")
    full_grid_log_likelihood = prediction.log_prob(flat_targets).squeeze(1).reshape(
        panel_count,
        64,
    )
    reshaped_codes = batch.factor_ids.reshape(panel_count, 64, 3)
    factor_codes = reshaped_codes[0]
    if not bool(reshaped_codes.eq(factor_codes[None]).all().item()):
        raise ValueError("every geometry panel must use the same factor bank order")
    locality = _active_set_fiber_huber_dispersion(
        torch=torch,
        full_grid_log_likelihood=full_grid_log_likelihood,
        factor_codes=factor_codes,
        active_axis_sets=active_axis_sets,
        beta=huber_beta,
    )
    return {
        "prediction": prediction,
        "cell_nll": cell_nll,
        "full_grid_log_likelihood": full_grid_log_likelihood,
        "proper_nll": proper_nll,
        "balanced_nll": balanced_nll,
        "changed_nll": changed_nll,
        "unchanged_nll": unchanged_nll,
        "geometry_supervision": geometry_supervision,
        "locality": locality,
    }


def _heldout_singleton_evaluation(
    *,
    torch: Any,
    model: Any,
    device: Any,
    geometry_seed_base: int,
    geometry_seed_count: int,
    batch_panels: int,
    balanced_weight: float,
    huber_beta: float,
) -> dict[str, object]:
    """Evaluate singleton accuracy and likelihood locality on a disjoint stream."""

    from prp_wm.latent_rules import outcome_map
    from prp_wm.random_geometry_protocol import AXES

    panels = tuple(
        panel
        for offset in range(geometry_seed_count)
        for panel in _singleton_panels(geometry_seed_base + offset)
    )
    axis_to_index = {axis: index for index, axis in enumerate(AXES)}
    totals: dict[str, float | int] = {
        "cells": 0,
        "cell_nll": 0.0,
        "examples": 0,
        "exact": 0,
        "panels": 0,
        "panels_all_64_exact": 0,
        "huber_sum": 0.0,
        "abs_sum": 0.0,
        "square_sum": 0.0,
        "deviations": 0,
        "range_sum": 0.0,
        "ranges": 0,
        "range_max": 0.0,
    }
    per_axis = {
        axis.value: {"examples": 0, "exact": 0, "panels": 0, "all_64_exact": 0}
        for axis in AXES
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(panels), batch_panels):
            panel_batch = panels[start : start + batch_panels]
            acted_axes = [axis_to_index[panel.axes[0]] for panel in panel_batch]
            batch = _build_singleton_batch(
                torch=torch,
                panels=panel_batch,
                device=device,
            )
            objective = _singleton_objective(
                torch=torch,
                model=model,
                batch=batch,
                acted_axis_indices=acted_axes,
                balanced_weight=balanced_weight,
                huber_beta=huber_beta,
            )
            height, width = batch.targets.shape[-2:]
            targets = batch.targets.reshape(-1, height, width)
            maps = outcome_map(objective["prediction"])[:, 0]
            exact = maps.eq(targets).all(dim=(-2, -1)).reshape(len(panel_batch), 64)
            deviations, ranges = _fiber_deviations(
                torch=torch,
                full_grid_log_likelihood=objective["full_grid_log_likelihood"],
                factor_codes=batch.factor_ids.reshape(len(panel_batch), 64, 3)[0],
                acted_axis_indices=acted_axes,
            )
            huber_values = torch.nn.functional.smooth_l1_loss(
                deviations,
                torch.zeros_like(deviations),
                beta=huber_beta,
                reduction="none",
            )
            totals["cells"] += int(objective["cell_nll"].numel())
            totals["cell_nll"] += float(objective["cell_nll"].sum().cpu())
            totals["examples"] += int(exact.numel())
            totals["exact"] += int(exact.sum().cpu())
            totals["panels"] += len(panel_batch)
            totals["panels_all_64_exact"] += int(exact.all(dim=1).sum().cpu())
            totals["huber_sum"] += float(huber_values.sum().cpu())
            totals["abs_sum"] += float(deviations.abs().sum().cpu())
            totals["square_sum"] += float(deviations.square().sum().cpu())
            totals["deviations"] += int(deviations.numel())
            range_values = [float(value.cpu()) for value in ranges]
            totals["range_sum"] += sum(range_values)
            totals["ranges"] += len(range_values)
            totals["range_max"] = max(float(totals["range_max"]), *range_values)
            for panel_index, panel in enumerate(panel_batch):
                row = per_axis[panel.axes[0].value]
                row["examples"] += 64
                row["exact"] += int(exact[panel_index].sum().cpu())
                row["panels"] += 1
                row["all_64_exact"] += int(bool(exact[panel_index].all().item()))

    deviations_count = int(totals["deviations"])
    result_per_axis = {
        name: {
            "panels": row["panels"],
            "examples": row["examples"],
            "exact_map_grid_accuracy": row["exact"] / row["examples"],
            "all_64_codes_exact_panel_rate": row["all_64_exact"] / row["panels"],
        }
        for name, row in per_axis.items()
    }
    return {
        "geometry_scope": "heldout-randomized-singletons",
        "geometry_seed_base": geometry_seed_base,
        "geometry_seed_count": geometry_seed_count,
        "panels": totals["panels"],
        "all_64_factor_codes_per_panel": True,
        "examples": totals["examples"],
        "proper_nll_per_cell": float(totals["cell_nll"]) / int(totals["cells"]),
        "exact_map_grid_accuracy": int(totals["exact"]) / int(totals["examples"]),
        "all_64_codes_exact_panel_rate": int(totals["panels_all_64_exact"])
        / int(totals["panels"]),
        "fiber_dispersion": {
            "definition": (
                "full-grid true-target log likelihood centered within each "
                "same-acted-axis-value 16-code nuisance fiber"
            ),
            "huber_beta": huber_beta,
            "mean_huber": float(totals["huber_sum"]) / deviations_count,
            "mean_absolute_deviation": float(totals["abs_sum"]) / deviations_count,
            "root_mean_square_deviation": (
                float(totals["square_sum"]) / deviations_count
            )
            ** 0.5,
            "mean_within_fiber_range": float(totals["range_sum"])
            / int(totals["ranges"]),
            "max_within_fiber_range": float(totals["range_max"]),
            "fiber_count": int(totals["ranges"]),
            "scores_per_fiber": 16,
        },
        "per_axis": result_per_axis,
    }


def main() -> None:
    args = parse_args()
    diagnostic_weight, stage_weights = _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from prp_wm.causal_filter import score_hypothesis_bank
    from prp_wm.latent_rules import outcome_map
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.random_geometry_protocol import AXES, PROTOCOL_VERSION
    from scripts.run_active_support_calibrated_executor import (
        STAGE_NAMES,
        _active_executor_gate,
        _active_histories,
        _active_prefix_evaluation,
        _diagnostic_panels,
        _program_conditioned_panel_batch,
        _scheduled_program_rows,
    )
    from scripts.run_causal_mechanism_coverage import _load_executor
    from scripts.run_rulegrid_executor_ceiling import (
        _configure_determinism,
        _evaluate,
        _resolve_device,
        _runtime_identity,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    initial_path = args.initial_checkpoint.resolve()
    model, initial_checkpoint = _load_executor(torch, initial_path, device)
    if (
        initial_checkpoint.get("checkpoint_schema_version")
        != EXPECTED_INITIAL_SCHEMA_VERSION
    ):
        raise SystemExit(
            "counterfactual locality continuation requires an audited "
            "active-support-calibrated checkpoint"
        )
    if args.executor_architecture == "canonical-role-routed":
        from prp_wm.routed_executor import (
            CanonicalRoleRoutedOracleFactorExecutor,
        )

        routed = CanonicalRoleRoutedOracleFactorExecutor(
            model.config
        ).to(device)
        routed.load_state_dict(model.state_dict(), strict=True)
        model = routed
        model_type = "CanonicalRoleRoutedOracleFactorExecutor"
    elif args.executor_architecture in {
        "matched-wider-global",
        "matched-factor-local",
    }:
        from prp_wm.matched_executor import (
            MatchedFactorLocalOracleFactorExecutor,
            MatchedWiderGlobalOracleFactorExecutor,
        )

        matched_class = {
            "matched-wider-global": MatchedWiderGlobalOracleFactorExecutor,
            "matched-factor-local": MatchedFactorLocalOracleFactorExecutor,
        }[args.executor_architecture]
        matched = matched_class(model.config).to(device)
        matched.initialize_from_oracle_state_dict(model.state_dict())
        model = matched
        model_type = matched_class.__name__
    else:
        model_type = "OracleFactorExecutor"
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    teacher = None
    if args.teacher_distillation_weight > 0:
        teacher, teacher_checkpoint = _load_executor(torch, initial_path, device)
        if (
            teacher_checkpoint.get("checkpoint_schema_version")
            != EXPECTED_INITIAL_SCHEMA_VERSION
        ):
            raise SystemExit("teacher checkpoint schema changed after student load")
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    geometry_train_seed_count = args.geometry_train_seed_count or args.steps
    train_geometry_seeds = range(
        args.geometry_train_seed_base,
        args.geometry_train_seed_base + geometry_train_seed_count,
    )
    geometry_seed_offsets = list(range(geometry_train_seed_count))
    random.Random(args.seed).shuffle(geometry_seed_offsets)
    geometry_seed_schedule = tuple(
        args.geometry_train_seed_base + offset for offset in geometry_seed_offsets
    )
    eval_geometry_seeds = range(
        args.geometry_eval_seed_base,
        args.geometry_eval_seed_base + args.heldout_singleton_seeds,
    )
    run_config: dict[str, object] = {
        "experiment": "counterfactual_singleton_locality_finetune",
        "result_kind": "privileged_geometry_supervision_and_locality_ablation",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model_type,
        "executor_architecture": args.executor_architecture,
        "privileged_true_rule_factors_are_model_inputs": True,
        "privileged_palette_canonicalization": True,
        "palette_input": "oracle-canonical",
        "random_geometry_protocol_version": PROTOCOL_VERSION,
        "initial_checkpoint_provenance": {
            "path": str(initial_path),
            "sha256": _sha256_file(initial_path),
            "schema_version": initial_checkpoint.get("checkpoint_schema_version"),
        },
        "objective": {
            "formula": (
                "active_prefix_replay + geometry_weight * "
                "(proper_nll + balanced_weight * balanced_nll) + "
                "ramped_locality_weight * fiber_huber + "
                "teacher_distillation_weight * replay_teacher_categorical_kl"
            ),
            "active_prefix_replay_domains": [
                "diagnostic_nontriple",
                *STAGE_NAMES,
            ],
            "geometry_scope": (
                "three randomized singletons per step, one per axis"
                if args.geometry_axis_scope == "singleton"
                else (
                    "six randomized panels per step: all three singleton "
                    "and all three pair axis sets"
                )
            ),
            "geometry_factor_codes_per_panel": 64,
            "locality_scope": (
                "full-grid true-target log-likelihood dispersion after "
                "holding the complete active-axis tuple fixed"
            ),
            "teacher_distillation_scope": (
                "frozen initial executor categorical next-cell distribution "
                "on the five original active replay domains only"
            ),
            "teacher_reads_new_geometry": False,
            "teacher_used_at_inference": False,
        },
        "steps": args.steps,
        "batch_size": args.batch_size,
        "codes_per_task": args.codes_per_task,
        "eval_tasks": args.eval_tasks,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "balanced_weight": args.balanced_weight,
        "max_grad_norm": args.max_grad_norm,
        "active_replay_loss_weights": {
            "diagnostic_nontriple": diagnostic_weight,
            **{
                name: weight
                for name, weight in zip(STAGE_NAMES, stage_weights, strict=True)
            },
        },
        "geometry_weight": args.geometry_weight,
        "geometry_axis_scope": args.geometry_axis_scope,
        "locality_weight": args.locality_weight,
        "locality_ramp_steps": args.locality_ramp_steps,
        "locality_huber_beta": args.locality_huber_beta,
        "teacher_distillation_weight": args.teacher_distillation_weight,
        "geometry_train_seed_stream": {
            "base": args.geometry_train_seed_base,
            "count": geometry_train_seed_count,
            "end_exclusive": (
                args.geometry_train_seed_base + geometry_train_seed_count
            ),
            "schedule": list(geometry_seed_schedule),
            "completed_cycles": args.steps // geometry_train_seed_count,
            "partial_cycle_steps": args.steps % geometry_train_seed_count,
        },
        "heldout_singleton_seed_stream": {
            "base": args.geometry_eval_seed_base,
            "count": args.heldout_singleton_seeds,
            "end_exclusive": (
                args.geometry_eval_seed_base + args.heldout_singleton_seeds
            ),
        },
        "geometry_seed_streams_disjoint": not bool(
            set(train_geometry_seeds).intersection(eval_geometry_seeds)
        ),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "active_task_start_offset": args.active_task_start_offset,
        "model_config": initial_checkpoint["model_config"],
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "runtime_identity": _runtime_identity(torch, device),
        "source_sha256": _source_sha256(),
    }

    latest: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            tasks = make_pilot_tasks(
                split=args.train_split,
                master_seed=args.data_master_seed,
                start=args.active_task_start_offset + step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
            )
            histories = _active_histories(tasks)
            programs = _scheduled_program_rows(
                tasks,
                step=step,
                codes_per_task=args.codes_per_task,
            )
            domains = (
                (
                    "diagnostic_nontriple",
                    _diagnostic_panels(tasks, NONTRIPLE_DIAGNOSTIC_INDICES),
                    diagnostic_weight,
                ),
                *tuple(
                    (name, panel, weight)
                    for name, panel, weight in zip(
                        STAGE_NAMES,
                        histories,
                        stage_weights,
                        strict=True,
                    )
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            replay_value = 0.0
            teacher_distillation_value = 0.0
            replay_metrics: dict[str, float] = {}
            teacher_metrics: dict[str, float] = {}
            for name, panel, weight in domains:
                batch = _program_conditioned_panel_batch(
                    torch=torch,
                    tasks=tasks,
                    panels=panel,
                    program_rows=programs,
                    device=device,
                )
                losses = model.losses(batch, balanced_weight=args.balanced_weight)
                weighted = weight * losses.total
                if not bool(torch.isfinite(weighted).item()):
                    raise RuntimeError(f"non-finite {name} loss at step {step + 1}")
                weighted.backward()
                value = float(losses.total.detach().cpu())
                replay_metrics[f"loss_replay_{name}"] = value
                replay_value += weight * value
                if teacher is not None:
                    student_prediction = model.predict_panel(batch)
                    with torch.no_grad():
                        teacher_prediction = teacher.predict_panel(batch)
                    teacher_kl = _teacher_categorical_kl(
                        torch=torch,
                        student_prediction=student_prediction,
                        teacher_prediction=teacher_prediction,
                    )
                    weighted_teacher = (
                        args.teacher_distillation_weight * weight * teacher_kl
                    )
                    if not bool(torch.isfinite(weighted_teacher).item()):
                        raise RuntimeError(
                            f"non-finite {name} teacher KL at step {step + 1}"
                        )
                    weighted_teacher.backward()
                    teacher_value = float(teacher_kl.detach().cpu())
                    teacher_metrics[f"teacher_kl_{name}"] = teacher_value
                    teacher_distillation_value += weight * teacher_value

            geometry_seed = geometry_seed_schedule[
                step % geometry_train_seed_count
            ]
            if args.geometry_axis_scope == "singleton-pair":
                geometry_panels = _geometry_panels(geometry_seed)
                geometry_batch, active_axis_sets = _build_geometry_batch(
                    torch=torch,
                    panels=geometry_panels,
                    device=device,
                )
                geometry_metrics = _geometry_objective(
                    torch=torch,
                    model=model,
                    batch=geometry_batch,
                    active_axis_sets=active_axis_sets,
                    balanced_weight=args.balanced_weight,
                    huber_beta=args.locality_huber_beta,
                )
                geometry_panels_per_step = 6
                pair_panels_per_step = 3
            else:
                geometry_panels = _singleton_panels(geometry_seed)
                geometry_batch = _build_singleton_batch(
                    torch=torch,
                    panels=geometry_panels,
                    device=device,
                )
                geometry_metrics = _singleton_objective(
                    torch=torch,
                    model=model,
                    batch=geometry_batch,
                    acted_axis_indices=tuple(range(len(AXES))),
                    balanced_weight=args.balanced_weight,
                    huber_beta=args.locality_huber_beta,
                )
                geometry_panels_per_step = 3
                pair_panels_per_step = 0
            completed = step + 1
            locality_weight = _locality_ramp(
                args.locality_weight,
                completed,
                args.locality_ramp_steps,
            )
            geometry_term = (
                args.geometry_weight * geometry_metrics["geometry_supervision"]
            )
            locality_term = locality_weight * geometry_metrics["locality"]
            added = geometry_term + locality_term
            if not bool(torch.isfinite(added).item()):
                raise RuntimeError(f"non-finite geometry loss at step {completed}")
            added.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                .detach()
                .cpu()
            )
            optimizer.step()
            geometry_value = float(
                geometry_metrics["geometry_supervision"].detach().cpu()
            )
            locality_value = float(geometry_metrics["locality"].detach().cpu())
            latest = {
                "loss_total": (
                    replay_value
                    + args.geometry_weight * geometry_value
                    + locality_weight * locality_value
                    + args.teacher_distillation_weight
                    * teacher_distillation_value
                ),
                "loss_active_prefix_replay": replay_value,
                **replay_metrics,
                "teacher_categorical_kl_replay": teacher_distillation_value,
                **teacher_metrics,
                "loss_geometry_supervision": geometry_value,
                "loss_geometry_proper_nll": float(
                    geometry_metrics["proper_nll"].detach().cpu()
                ),
                "loss_geometry_balanced_nll": float(
                    geometry_metrics["balanced_nll"].detach().cpu()
                ),
                "loss_locality_fiber_huber": locality_value,
                "effective_locality_weight": locality_weight,
                "gradient_norm": gradient_norm,
            }
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record: dict[str, object] = {
                    "step": completed,
                    "active_replay_tasks_seen": completed * args.batch_size,
                    "singleton_panels_seen": completed * 3,
                    "singleton_factor_examples_seen": completed * 3 * 64,
                    "pair_panels_seen": completed * pair_panels_per_step,
                    "pair_factor_examples_seen": (
                        completed * pair_panels_per_step * 64
                    ),
                    "geometry_panels_seen": (
                        completed * geometry_panels_per_step
                    ),
                    "geometry_factor_examples_seen": (
                        completed * geometry_panels_per_step * 64
                    ),
                    "geometry_seed": geometry_seed,
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    common_diagnostic_evaluation = {
        "torch": torch,
        "model": model,
        "device": device,
        "split": args.eval_split,
        "data_master_seed": args.data_master_seed,
        "task_count": args.eval_tasks,
        "batch_size": args.eval_batch_size,
        "make_pilot_tasks": make_pilot_tasks,
        "make_factor_batch": __import__(
            "prp_wm.latent_rules", fromlist=["rulegrid_tasks_to_oracle_factor_batch"]
        ).rulegrid_tasks_to_oracle_factor_batch,
        "outcome_map": outcome_map,
        "canonicalize_palette": True,
    }
    single = _evaluate(
        **common_diagnostic_evaluation,
        diagnostic_indices=tuple(range(12)),
    )
    pair = _evaluate(
        **common_diagnostic_evaluation,
        diagnostic_indices=tuple(range(12, 21)),
    )
    triple = _evaluate(
        **common_diagnostic_evaluation,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
    )
    prefix = _active_prefix_evaluation(
        torch=torch,
        model=model,
        device=device,
        split=args.eval_split,
        data_master_seed=args.data_master_seed,
        task_count=args.eval_tasks,
        batch_size=args.eval_batch_size,
        make_pilot_tasks=make_pilot_tasks,
        score_hypothesis_bank=score_hypothesis_bank,
    )
    gate = _active_executor_gate(prefix, (single, pair, triple))
    heldout_singleton = _heldout_singleton_evaluation(
        torch=torch,
        model=model,
        device=device,
        geometry_seed_base=args.geometry_eval_seed_base,
        geometry_seed_count=args.heldout_singleton_seeds,
        batch_panels=args.heldout_geometry_batch_panels,
        balanced_weight=args.balanced_weight,
        huber_beta=args.locality_huber_beta,
    )

    checkpoint: dict[str, object] = {
        **run_config,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "latest_training_metrics": latest,
        "heldout_singleton_evaluation": heldout_singleton,
        "single_evaluation": single,
        "pair_evaluation": pair,
        "heldout_triple_evaluation": triple,
        "heldout_active_prefix_evaluation": prefix,
        "active_prefix_executor_gate": gate,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    result: dict[str, object] = {
        **run_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "active_replay_tasks_seen": args.steps * args.batch_size,
        "singleton_panels_seen": args.steps * 3,
        "singleton_factor_examples_seen": args.steps * 3 * 64,
        "pair_panels_seen": (
            args.steps * 3
            if args.geometry_axis_scope == "singleton-pair"
            else 0
        ),
        "pair_factor_examples_seen": (
            args.steps * 3 * 64
            if args.geometry_axis_scope == "singleton-pair"
            else 0
        ),
        "geometry_panels_seen": (
            args.steps
            * (6 if args.geometry_axis_scope == "singleton-pair" else 3)
        ),
        "geometry_factor_examples_seen": (
            args.steps
            * (6 if args.geometry_axis_scope == "singleton-pair" else 3)
            * 64
        ),
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "heldout_singleton_evaluation": heldout_singleton,
        "single_evaluation": single,
        "pair_evaluation": pair,
        "heldout_triple_evaluation": triple,
        "heldout_active_prefix_evaluation": prefix,
        "active_prefix_executor_gate": gate,
        "interpretation": (
            "Compare executor conditions under the same replay, geometry, "
            "locality, and teacher-distillation continuation. The matched "
            "four-branch wider-global and factor-local conditions have "
            "identical parameters, public canonical router/masks, branch "
            "compute, and initialization; only their factor-conditioning "
            "graph differs. This remains an oracle-code, oracle-palette "
            "executor diagnostic."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
