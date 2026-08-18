#!/usr/bin/env python3
"""Evaluation-only Coverage@4 audit on held-out RuleGrid triple diagnostics.

For each evaluation task, this script derives the programs compatible with the
*observed public support*, groups them by their three public triple outcomes,
and constructs those alternative target panels only after loading a frozen
checkpoint.  A behavior class is covered when at least one *single* particle
mode predicts all three frames exactly by its mode-conditioned MAP decoder and
has panel-mean NLL per cell no greater than the frozen threshold.

The hidden true program and the task's true diagnostic target sidecar are never
read for inference or mode selection.  Class/mode matching is post-hoc
evaluation scoring, not a controller decision.

Example:

    python scripts/eval_rulegrid_coverage_audit.py \
      --checkpoint runs/pilot_seed7/checkpoint_last.pt \
      --device cuda --tasks 192 --batch-size 16 \
      --output runs/pilot_seed7/triple_coverage_audit.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.pilot import (
    NONTRIPLE_DIAGNOSTIC_INDICES,
    PILOT_PROTOCOL_VERSION,
    TRIPLE_DIAGNOSTIC_INDICES,
    assert_nontriple_training_indices,
    make_pilot_tasks,
)
from prp_wm.rulegrid import (
    BENCHMARK_VERSION,
    MASTER_SEED,
    Grid,
    RuleGridTask,
    behavior_classes,
    simulate,
    version_space,
)


CHECKPOINT_SCHEMA_VERSION = "prp-wm.rulegrid-pilot-checkpoint.v2"
DEFAULT_DATA_MASTER_SEED = MASTER_SEED
DEFAULT_NLL_THRESHOLD_PER_CELL = 0.05
CANONICAL_TASK_BLOCK = 192
_AUDITED_SOURCE_FILES = (
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/train_rulegrid_pilot.py",
    "scripts/eval_rulegrid_coverage_audit.py",
)
_CHECKPOINT_COMPATIBLE_SOURCE_FILES = (
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/train_rulegrid_pilot.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AlternativeBehaviorPanels:
    """Evaluation-only class panels derived from observed support evidence.

    ``targets[task][class][query]`` is one coherent triple panel for a
    behavior-equivalence class.  The associated mass is the exact fraction of
    support-compatible programs in that class.  Neither structure contains a
    true-program label or is used to choose a model mode.
    """

    targets: tuple[tuple[tuple[Grid, ...], ...], ...]
    masses: tuple[tuple[float, ...], ...]
    compatible_program_counts: tuple[int, ...]


@dataclass(frozen=True)
class TensorAlternativeBehaviorPanels:
    """Padded GPU/CPU representation of evaluation-only class panels."""

    targets: Any  # [B,M,Q,H,W] long
    masses: Any  # [B,M] float
    mask: Any  # [B,M] bool
    compatible_program_counts: tuple[int, ...]


@dataclass(frozen=True)
class CoverageBatchScore:
    """Per-task/class Coverage@4 scoring result."""

    class_covered: Any  # [B,M] bool
    class_mask: Any  # [B,M] bool
    class_mass: Any  # [B,M] float
    map_exact_by_mode: Any  # [B,M,K] bool, ancillary: ignores NLL cutoff
    nll_threshold_by_mode: Any  # [B,M,K] bool, ancillary: ignores MAP exactness
    qualifying_modes: Any  # [B,M,K] bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260718,
        help="Evaluation RNG/determinism seed; it does not choose RuleGrid tasks.",
    )
    parser.add_argument(
        "--data-master-seed",
        type=int,
        default=DEFAULT_DATA_MASTER_SEED,
        help="Must match the checkpoint's fixed RuleGrid data master seed.",
    )
    parser.add_argument("--tasks", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete torch device such as cuda:0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="pilot-composition",
        help="Canonical held-out composition stream; no alternate split is accepted.",
    )
    return parser.parse_args()


def _resolve_device(torch: Any, raw: str) -> Any:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(raw)
    except (TypeError, RuntimeError) as error:
        raise SystemExit(f"invalid --device {raw!r}: {error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    return device


def _configure_deterministic_inference(torch: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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


def _runtime_identity(torch: Any, device: Any) -> dict[str, object]:
    identity: dict[str, object] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        identity["cuda_runtime_version"] = torch.version.cuda
        identity["cuda_device_name"] = torch.cuda.get_device_name(device)
        identity["cuda_device_capability"] = list(torch.cuda.get_device_capability(device))
    return identity


def _load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"checkpoint does not exist: {path}")
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 lacks the explicit weights_only argument.
        loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, dict):
        raise SystemExit("checkpoint must contain a dictionary")
    return loaded


def _validate_checkpoint_source_manifest(checkpoint: dict[str, object]) -> None:
    """Reject a checkpoint produced by nonmatching core pilot source files."""

    manifest = checkpoint.get("source_sha256")
    if not isinstance(manifest, dict):
        raise SystemExit("checkpoint has no source_sha256 provenance manifest")
    for relative in _CHECKPOINT_COMPATIBLE_SOURCE_FILES:
        expected = manifest.get(relative)
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise SystemExit(
                f"checkpoint source manifest lacks a valid SHA256 for {relative}"
            )
        actual = _sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise SystemExit(
                f"checkpoint source hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )


def _validate_checkpoint(
    checkpoint: dict[str, object], *, data_master_seed: int, split: str
) -> dict[str, object]:
    """Validate provenance and return the audited training metadata."""

    if checkpoint.get("model_type") != "PersistentK4":
        raise SystemExit("checkpoint is not a PersistentK4 pilot checkpoint")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("checkpoint schema version is incompatible")
    if checkpoint.get("pilot_protocol_version") != PILOT_PROTOCOL_VERSION:
        raise SystemExit("checkpoint pilot protocol version is incompatible")
    if checkpoint.get("benchmark_version") != BENCHMARK_VERSION:
        raise SystemExit("checkpoint benchmark version is incompatible")
    _validate_checkpoint_source_manifest(checkpoint)
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        raise SystemExit("checkpoint has no training audit metadata")
    if training.get("composition_targets_materialized_for_training") is not False:
        raise SystemExit("checkpoint does not prove that triple targets were held out")
    checkpoint_data_master_seed = training.get("data_master_seed")
    if type(checkpoint_data_master_seed) is not int or checkpoint_data_master_seed < 0:
        raise SystemExit("checkpoint has no valid data_master_seed audit field")
    if data_master_seed != checkpoint_data_master_seed:
        raise SystemExit(
            "--data-master-seed must match checkpoint training data_master_seed "
            f"({checkpoint_data_master_seed})"
        )
    if training.get("split") == split:
        raise SystemExit(
            "coverage evaluation --split must differ from checkpoint training split "
            "to prevent task-instance overlap"
        )
    try:
        assert_nontriple_training_indices(tuple(training["train_diagnostic_indices"]))
        assert_nontriple_training_indices(
            tuple(training["materialized_diagnostic_target_indices"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"checkpoint train-target audit failed: {error}") from error
    return training


def construct_alternative_behavior_panels(
    tasks: Sequence[RuleGridTask],
    *,
    diagnostic_indices: Sequence[int] = TRIPLE_DIAGNOSTIC_INDICES,
) -> AlternativeBehaviorPanels:
    """Build alternate triple panels from support-compatible programs only.

    This deliberately derives ``compatible`` with ``version_space`` from the
    observed support.  It never reads ``task.privileged.true_program`` or the
    true diagnostic target sidecar.  The simulator is used solely to produce
    evaluation labels for the behavior classes it just derived.
    """

    indices = tuple(diagnostic_indices)
    if indices != TRIPLE_DIAGNOSTIC_INDICES:
        raise ValueError("coverage audit is defined only for canonical triple indices 21..23")
    all_targets: list[tuple[tuple[Grid, ...], ...]] = []
    all_masses: list[tuple[float, ...]] = []
    compatible_counts: list[int] = []
    for task in tasks:
        diagnostics = tuple(task.inference.diagnostics[index] for index in indices)
        # Palette encodes a nuisance color permutation, not a rule label.
        compatible = version_space(task.inference.support, task.privileged.palette)
        if len(compatible) != 4:
            raise ValueError(
                "canonical Coverage@4 requires exactly four programs compatible "
                f"with observed support, got {len(compatible)}"
            )
        classes = behavior_classes(compatible, diagnostics, task.privileged.palette)
        ordered_classes = tuple(sorted(classes.items(), key=lambda item: item[0]))
        if len(ordered_classes) != 4:
            raise ValueError(
                "canonical triple Coverage@4 requires exactly four behavior classes, "
                f"got {len(ordered_classes)}"
            )
        panels: list[tuple[Grid, ...]] = []
        masses: list[float] = []
        for _, programs in ordered_classes:
            representative = programs[0]
            panels.append(
                tuple(
                    simulate(probe.state, probe.action, representative, task.privileged.palette)
                    for probe in diagnostics
                )
            )
            masses.append(len(programs) / len(compatible))
        if abs(sum(masses) - 1.0) > 1e-12:
            raise AssertionError("behavior-class masses must sum to one")
        all_targets.append(tuple(panels))
        all_masses.append(tuple(masses))
        compatible_counts.append(len(compatible))
    return AlternativeBehaviorPanels(
        targets=tuple(all_targets),
        masses=tuple(all_masses),
        compatible_program_counts=tuple(compatible_counts),
    )


def _pad_public_action_grid(
    torch: Any, action_grid: list[list[Any]], device: Any
) -> tuple[Any, Any | None]:
    """Pad public action tensors while preserving composite-action masks."""

    batch_size = len(action_grid)
    if not batch_size or not action_grid[0]:
        raise ValueError("action grid cannot be empty")
    count = len(action_grid[0])
    if any(len(actions) != count for actions in action_grid):
        raise ValueError("all tasks must have equal action counts")
    max_atoms = max(action.shape[0] for actions in action_grid for action in actions)
    if max_atoms <= 0:
        raise ValueError("public actions must contain at least one atom")
    padded = torch.zeros((batch_size, count, max_atoms, 4), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, count, max_atoms), dtype=torch.bool, device=device)
    for task_index, actions in enumerate(action_grid):
        for action_index, action in enumerate(actions):
            atoms = action.shape[0]
            padded[task_index, action_index, :atoms] = action.to(device)
            mask[task_index, action_index, :atoms] = True
    if max_atoms == 1:
        return padded[:, :, 0], None
    return padded, mask


def make_public_triple_input_batch(
    torch: Any, tasks: Sequence[RuleGridTask], *, device: Any
) -> Any:
    """Create a model batch using only public support and triple query inputs.

    Query targets are intentionally absent.  This makes it mechanically clear
    that target panels cannot participate in support inference or mode choice.
    """

    from prp_wm.neural import RuleGridTensorBatch, encode_public_action

    if not tasks:
        raise ValueError("at least one task is required")
    support_count = len(tasks[0].inference.support)
    if support_count != 6:
        raise ValueError("coverage audit requires the fixed six-transition support prefix")
    support_states: list[list[Grid]] = []
    support_targets: list[list[Grid]] = []
    support_actions: list[list[Any]] = []
    query_states: list[list[Grid]] = []
    query_actions: list[list[Any]] = []
    for task in tasks:
        if len(task.inference.support) != support_count:
            raise ValueError("all tasks need equal support lengths")
        probes = tuple(task.inference.diagnostics[index] for index in TRIPLE_DIAGNOSTIC_INDICES)
        support_states.append([transition.state for transition in task.inference.support])
        support_targets.append([transition.next_state for transition in task.inference.support])
        support_actions.append(
            [encode_public_action(transition.action) for transition in task.inference.support]
        )
        query_states.append([probe.state for probe in probes])
        query_actions.append([encode_public_action(probe.action) for probe in probes])
    support_action_tensor, support_action_mask = _pad_public_action_grid(
        torch, support_actions, device
    )
    query_action_tensor, query_action_mask = _pad_public_action_grid(
        torch, query_actions, device
    )
    return RuleGridTensorBatch(
        support_states=torch.tensor(support_states, dtype=torch.long, device=device),
        support_actions=support_action_tensor,
        support_targets=torch.tensor(support_targets, dtype=torch.long, device=device),
        support_mask=torch.ones((len(tasks), support_count), dtype=torch.bool, device=device),
        query_states=torch.tensor(query_states, dtype=torch.long, device=device),
        query_actions=query_action_tensor,
        query_targets=None,
        behavior_targets=None,
        behavior_mass=None,
        support_action_mask=support_action_mask,
        query_action_mask=query_action_mask,
    )


def tensorize_alternative_behavior_panels(
    torch: Any, panels: AlternativeBehaviorPanels, *, device: Any
) -> TensorAlternativeBehaviorPanels:
    """Pad per-task behavior classes without fabricating mass for padding."""

    if not panels.targets:
        raise ValueError("at least one task panel is required")
    batch_size = len(panels.targets)
    if len(panels.masses) != batch_size:
        raise ValueError("behavior panel target/mass task counts differ")
    query_count = len(TRIPLE_DIAGNOSTIC_INDICES)
    max_classes = max(len(task_panels) for task_panels in panels.targets)
    if not 1 <= max_classes <= 4:
        raise ValueError("Coverage@4 requires between one and four behavior classes")
    first_grid = panels.targets[0][0][0]
    height = len(first_grid)
    width = len(first_grid[0])
    targets = torch.zeros(
        (batch_size, max_classes, query_count, height, width),
        dtype=torch.long,
        device=device,
    )
    masses = torch.zeros((batch_size, max_classes), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, max_classes), dtype=torch.bool, device=device)
    for task_index, (task_panels, task_masses) in enumerate(
        zip(panels.targets, panels.masses, strict=True)
    ):
        if len(task_panels) != len(task_masses):
            raise ValueError("per-task class targets/masses differ")
        if not 1 <= len(task_panels) <= max_classes:
            raise ValueError("invalid behavior-class count")
        if abs(sum(task_masses) - 1.0) > 1e-12:
            raise ValueError("per-task behavior masses must sum to one")
        for class_index, (panel, mass) in enumerate(
            zip(task_panels, task_masses, strict=True)
        ):
            if len(panel) != query_count:
                raise ValueError("behavior panel does not have exactly three triple queries")
            targets[task_index, class_index] = torch.tensor(
                panel, dtype=torch.long, device=device
            )
            masses[task_index, class_index] = mass
            mask[task_index, class_index] = True
    if not bool(torch.allclose(masses.sum(dim=1), torch.ones(batch_size, device=device))):
        raise AssertionError("padded behavior masses must sum to one")
    return TensorAlternativeBehaviorPanels(
        targets=targets,
        masses=masses,
        mask=mask,
        compatible_program_counts=panels.compatible_program_counts,
    )


def _mode_cell_log_probabilities(torch: Any, prediction: Any) -> Any:
    """Return factorized per-mode cell log probabilities ``[B,K,C,H,W]``."""

    batch_size, modes, colors, height, width = prediction.new_color_logits.shape
    original = prediction.input_colors[:, None, None, :, :]
    color_ids = torch.arange(colors, device=prediction.new_color_logits.device)[
        None, None, :, None, None
    ]
    masked_logits = prediction.new_color_logits.masked_fill(
        original == color_ids, torch.finfo(prediction.new_color_logits.dtype).min
    )
    changed_log_prob = (
        torch.nn.functional.logsigmoid(prediction.change_logits)[:, :, None]
        + torch.nn.functional.log_softmax(masked_logits, dim=2)
    )
    unchanged_log_prob = torch.nn.functional.logsigmoid(-prediction.change_logits)
    return changed_log_prob.scatter(
        2,
        prediction.input_colors[:, None, None].expand(
            batch_size, modes, 1, height, width
        ),
        unchanged_log_prob[:, :, None],
    )


def predict_all_triple_modes(torch: Any, model: Any, batch: Any, inference: Any) -> Any:
    """Run every triple input under every persistent mode without targets."""

    assert batch.query_states is not None
    assert batch.query_actions is not None
    batch_size, queries, height, width = batch.query_states.shape
    if queries != len(TRIPLE_DIAGNOSTIC_INDICES):
        raise ValueError("coverage audit expects exactly three triple queries")
    flat_states = batch.query_states.reshape(batch_size * queries, height, width)
    if batch.query_actions.ndim == 3:
        flat_actions = batch.query_actions.reshape(batch_size * queries, 4)
    else:
        atoms = batch.query_actions.shape[2]
        flat_actions = batch.query_actions.reshape(batch_size * queries, atoms, 4)
    flat_mask = (
        batch.query_action_mask.reshape(batch_size * queries, -1)
        if batch.query_action_mask is not None
        else None
    )
    flat_modes = (
        inference.modes[:, None]
        .expand(-1, queries, -1, -1)
        .reshape(batch_size * queries, model.config.particles, model.config.rule_dim)
    )
    return model.predict(flat_states, flat_actions, flat_modes, flat_mask)


def score_coverage_at_4(
    torch: Any,
    prediction: Any,
    panels: TensorAlternativeBehaviorPanels,
    *,
    batch_size: int,
    nll_threshold_per_cell: float,
) -> CoverageBatchScore:
    """Score exact class coverage by one coherent mode across all triple frames.

    NLL is the arithmetic mean of negative log probabilities over the entire
    3-frame, 192-cell panel for a class/mode pair.  The same mode must pass the
    NLL threshold and exactly MAP-decode every cell of every one of the three
    frames.  This implements Coverage@4 because all four particles are tested
    but each class may be credited to any one of them.
    """

    if nll_threshold_per_cell <= 0.0:
        raise ValueError("nll_threshold_per_cell must be positive")
    flat_batch, modes, colors, height, width = prediction.new_color_logits.shape
    queries = len(TRIPLE_DIAGNOSTIC_INDICES)
    if flat_batch != batch_size * queries or modes != 4:
        raise ValueError("prediction is incompatible with a K=4 triple audit")
    if panels.targets.shape[:3] != (batch_size, panels.targets.shape[1], queries):
        raise ValueError("alternative panel tensor has incompatible batch/query shape")
    if panels.targets.shape[-2:] != (height, width):
        raise ValueError("alternative panel grid size differs from prediction")
    if panels.mask.shape != panels.masses.shape or panels.mask.shape[:1] != (batch_size,):
        raise ValueError("alternative class mask/mass shapes are incompatible")

    mode_map = _mode_cell_log_probabilities(torch, prediction).argmax(dim=2).reshape(
        batch_size, queries, modes, height, width
    )
    # [B,M,Q,K,H,W] -> every query and cell must match under the same K mode.
    map_exact = (
        mode_map[:, None] == panels.targets[:, :, :, None]
    ).all(dim=(2, 4, 5))
    classes = panels.targets.shape[1]
    nll_by_mode = torch.empty(
        (batch_size, classes, modes),
        dtype=prediction.change_logits.dtype,
        device=prediction.change_logits.device,
    )
    for class_index in range(classes):
        class_targets = panels.targets[:, class_index].reshape(
            batch_size * queries, height, width
        )
        log_prob = prediction.log_prob(class_targets).reshape(batch_size, queries, modes)
        nll_by_mode[:, class_index] = -log_prob.sum(dim=1) / float(
            queries * height * width
        )
    class_mask_by_mode = panels.mask[:, :, None]
    map_exact_by_mode = map_exact & class_mask_by_mode
    nll_threshold_by_mode = (
        nll_by_mode <= nll_threshold_per_cell
    ) & class_mask_by_mode
    qualifying_modes = map_exact_by_mode & nll_threshold_by_mode
    class_covered = qualifying_modes.any(dim=-1) & panels.mask
    return CoverageBatchScore(
        class_covered=class_covered,
        class_mask=panels.mask,
        class_mass=panels.masses,
        map_exact_by_mode=map_exact_by_mode,
        nll_threshold_by_mode=nll_threshold_by_mode,
        qualifying_modes=qualifying_modes,
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.data_master_seed < 0:
        raise SystemExit("--seed and --data-master-seed must be non-negative")
    if args.tasks <= 0 or args.batch_size <= 0:
        raise SystemExit("--tasks and --batch-size must be positive")
    if args.tasks % CANONICAL_TASK_BLOCK:
        raise SystemExit(
            f"canonical Coverage@4 requires --tasks to be a multiple of "
            f"{CANONICAL_TASK_BLOCK}"
        )
    if args.data_master_seed != MASTER_SEED:
        raise SystemExit(
            "canonical Coverage@4 is frozen to "
            f"--data-master-seed {MASTER_SEED}"
        )
    if args.split != "pilot-composition":
        raise SystemExit("coverage audit is frozen to the canonical --split pilot-composition")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from prp_wm.neural import NeuralPRPConfig, PersistentK4
    except ImportError as error:
        raise SystemExit(f"coverage audit requires the optional PyTorch dependency: {error}") from error

    device = _resolve_device(torch, args.device)
    _configure_deterministic_inference(torch, args.seed)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = _load_checkpoint(torch, checkpoint_path, device)
    training = _validate_checkpoint(
        checkpoint, data_master_seed=args.data_master_seed, split=args.split
    )
    raw_config = checkpoint.get("model_config")
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(raw_state, dict):
        raise SystemExit("checkpoint model configuration or state dict is malformed")
    try:
        config = NeuralPRPConfig(**raw_config)
        model = PersistentK4(config).to(device)
        model.load_state_dict(raw_state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"could not restore PersistentK4 checkpoint: {error}") from error
    if config.particles != 4:
        raise SystemExit("Coverage@4 audit requires a four-particle checkpoint")
    model.eval()

    weighted_coverage_sum = 0.0
    all_classes_covered_tasks = 0
    unweighted_covered_classes = 0
    unweighted_map_exact_classes = 0
    unweighted_nll_threshold_classes = 0
    total_classes = 0
    compatible_program_sum = 0
    processed_tasks = 0
    with torch.no_grad():
        for start in range(0, args.tasks, args.batch_size):
            count = min(args.batch_size, args.tasks - start)
            # This is evaluation-only construction.  The model input builder
            # below intentionally excludes target tensors.
            tasks = make_pilot_tasks(
                split=args.split,
                master_seed=args.data_master_seed,
                start=start,
                count=count,
                diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
            )
            panels = construct_alternative_behavior_panels(tasks)
            tensor_panels = tensorize_alternative_behavior_panels(
                torch, panels, device=device
            )
            batch = make_public_triple_input_batch(torch, tasks, device=device)
            inference = model.infer_support(batch)
            prediction = predict_all_triple_modes(torch, model, batch, inference)
            score = score_coverage_at_4(
                torch,
                prediction,
                tensor_panels,
                batch_size=count,
                nll_threshold_per_cell=DEFAULT_NLL_THRESHOLD_PER_CELL,
            )
            weighted_coverage_sum += float(
                (score.class_mass * score.class_covered.to(score.class_mass.dtype))
                .sum()
                .detach()
                .cpu()
            )
            all_classes_covered_tasks += int(
                (score.class_covered | ~score.class_mask).all(dim=1).sum().detach().cpu()
            )
            unweighted_covered_classes += int(score.class_covered.sum().detach().cpu())
            unweighted_map_exact_classes += int(
                score.map_exact_by_mode.any(dim=-1).sum().detach().cpu()
            )
            unweighted_nll_threshold_classes += int(
                score.nll_threshold_by_mode.any(dim=-1).sum().detach().cpu()
            )
            total_classes += int(score.class_mask.sum().detach().cpu())
            compatible_program_sum += sum(tensor_panels.compatible_program_counts)
            processed_tasks += count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary: dict[str, object] = {
        "all_classes_covered_task_rate": all_classes_covered_tasks / processed_tasks,
        "audit_kind": "evaluation_only_alternative_behavior_class_coverage_at_4",
        "benchmark_version": BENCHMARK_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_source_sha256": checkpoint.get("source_sha256"),
        "checkpoint_source_hashes_verified": True,
        "checkpoint_runtime_identity": checkpoint.get("runtime_identity"),
        "checkpoint_training_data_master_seed": training["data_master_seed"],
        "canonical_task_block": CANONICAL_TASK_BLOCK,
        "checkpoint_train_diagnostic_indices": list(NONTRIPLE_DIAGNOSTIC_INDICES),
        "composition_targets_are_evaluation_only": True,
        "coverage_at_4_weighted": weighted_coverage_sum / processed_tasks,
        "data_master_seed": args.data_master_seed,
        "diagnostic_indices": list(TRIPLE_DIAGNOSTIC_INDICES),
        "evaluation_rng_seed": args.seed,
        "mean_compatible_programs_per_task": compatible_program_sum / processed_tasks,
        "model_config": asdict(config),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mode_count": config.particles,
        "nll_threshold_per_cell": DEFAULT_NLL_THRESHOLD_PER_CELL,
        "nll_threshold_scope": "mean_over_all_3_triple_frames_and_64_cells_per_frame",
        "panel_construction": (
            "evaluation_only: version_space(observed_support) -> "
            "behavior_classes(compatible_programs, public_triple_probes) -> simulator labels"
        ),
        "pilot_protocol_version": PILOT_PROTOCOL_VERSION,
        "scored_tasks": processed_tasks,
        "source_sha256": _source_sha256(),
        "split": args.split,
        "true_program_used_for_inference_selection": False,
        "true_target_sidecar_used_for_inference_selection": False,
        "unweighted_behavior_class_any_mode_map_exact_rate_ancillary": (
            unweighted_map_exact_classes / total_classes
        ),
        "unweighted_behavior_class_any_mode_nll_threshold_rate_ancillary": (
            unweighted_nll_threshold_classes / total_classes
        ),
        "unweighted_behavior_class_coverage": unweighted_covered_classes / total_classes,
        "runtime_identity": _runtime_identity(torch, device),
    }
    _atomic_json(args.output.resolve(), summary)
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
