"""Privileged probes for latent-rule execution capacity.

The main PRP-WM model must infer rule hypotheses from public transitions.  A
failed end-to-end particle model, however, does not reveal whether the failure
comes from rule inference or from the shared state-transition executor.  This
module isolates the latter with an explicitly privileged upper baseline:

* the model receives the three true RuleGrid factor values;
* each factor is embedded independently and composed into one continuous rule
  latent (there is no 64-way program lookup table);
* the existing grid encoder, public action encoder, FiLM decoder, and proper
  outcome distribution are reused without modifying the frozen pilot source;
* training may use a balanced auxiliary loss, but evaluation always reports
  the proper full outcome likelihood.

This is a diagnostic ceiling, not an admissible ARC agent.  In particular,
``rulegrid_tasks_to_oracle_factor_batch`` deliberately reads
``task.privileged.true_program`` and its name makes that boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import itertools
import math
from typing import Any, Sequence

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - Stage 0 intentionally has no torch.
    raise ImportError(
        "prp_wm.latent_rules requires PyTorch. Install the optional neural extra."
    ) from error

from .neural import (
    ACTION_FIELDS,
    ActionEncoder,
    FiLMDecoder,
    GridEncoder,
    InferenceState,
    NeuralPRPConfig,
    OutcomePrediction,
    PersistentK4,
    RuleGridTensorBatch,
    encode_public_action,
    rulegrid_tasks_to_tensor_batch,
)


RULE_FACTOR_COUNT = 3
RULE_FACTOR_CARDINALITY = 4


@dataclass(frozen=True)
class OracleFactorBatch:
    """A panel of transitions paired with privileged compositional rule codes.

    ``states`` and ``targets`` have shape ``[B,Q,H,W]``.  ``actions`` has
    either ``[B,Q,4]`` or ``[B,Q,L,4]`` shape.  ``factor_ids`` has shape
    ``[B,3]`` in collision/trigger/relation order.
    """

    states: Tensor
    actions: Tensor
    targets: Tensor
    factor_ids: Tensor
    action_mask: Tensor | None = None
    palette_canonicalized: bool = False

    def to(self, *args: Any, **kwargs: Any) -> "OracleFactorBatch":
        return OracleFactorBatch(
            **{
                field.name: (
                    getattr(self, field.name).to(*args, **kwargs)
                    if isinstance(getattr(self, field.name), Tensor)
                    else getattr(self, field.name)
                )
                for field in fields(self)
            }
        )

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.states.shape[1])

    def validate(self, config: NeuralPRPConfig) -> None:
        if self.states.ndim != 4:
            raise ValueError("states must have [B,Q,H,W] shape")
        if self.states.dtype != torch.long or self.targets.dtype != torch.long:
            raise TypeError("states and targets must use torch.long dtype")
        if self.targets.shape != self.states.shape:
            raise ValueError("targets must have the same shape as states")
        if self.states.shape[-2:] != (config.grid_size, config.grid_size):
            raise ValueError("states use a grid size incompatible with the model")
        if torch.any(self.states < 0) or torch.any(self.states >= config.num_colors):
            raise ValueError("states contain a color outside the model palette")
        if torch.any(self.targets < 0) or torch.any(self.targets >= config.num_colors):
            raise ValueError("targets contain a color outside the model palette")

        batch_size, queries = self.states.shape[:2]
        if self.actions.shape[:2] != (batch_size, queries):
            raise ValueError("actions must share the [B,Q] prefix")
        if self.actions.dtype != torch.long or self.actions.shape[-1] != ACTION_FIELDS:
            raise TypeError("actions must be torch.long tensors ending in four fields")
        if self.actions.ndim == 3:
            if self.action_mask is not None:
                raise ValueError("atomic actions must not have an action mask")
        elif self.actions.ndim == 4:
            if self.action_mask is None or self.action_mask.shape != self.actions.shape[:-1]:
                raise ValueError("composite actions require a matching [B,Q,L] mask")
            if self.action_mask.dtype != torch.bool:
                raise TypeError("action_mask must have torch.bool dtype")
            if not torch.all(self.action_mask.any(dim=-1)):
                raise ValueError("each composite action needs at least one atom")
        else:
            raise ValueError("actions must have [B,Q,4] or [B,Q,L,4] shape")

        if self.factor_ids.shape != (batch_size, RULE_FACTOR_COUNT):
            raise ValueError("factor_ids must have [B,3] shape")
        if self.factor_ids.dtype != torch.long:
            raise TypeError("factor_ids must use torch.long dtype")
        if torch.any(self.factor_ids < 0) or torch.any(
            self.factor_ids >= RULE_FACTOR_CARDINALITY
        ):
            raise ValueError("factor_ids must lie in [0,4)")


@dataclass(frozen=True)
class OracleFactorLoss:
    """Training losses for the privileged executor ceiling."""

    total: Tensor
    proper_nll: Tensor
    balanced_nll: Tensor
    changed_nll: Tensor
    unchanged_nll: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_proper_nll": float(self.proper_nll.detach().cpu()),
            "loss_balanced_nll": float(self.balanced_nll.detach().cpu()),
            "loss_changed_nll": float(self.changed_nll.detach().cpu()),
            "loss_unchanged_nll": float(self.unchanged_nll.detach().cpu()),
        }


def rule_program_factor_ids(program: Any) -> tuple[int, int, int]:
    """Return collision/trigger/relation IDs for a RuleGrid program.

    The local import keeps this privileged dependency visible and preserves
    dependency-free Stage 0 imports when the optional neural module is absent.
    """

    from .rulegrid import ALL_COLLISIONS, ALL_RELATIONS, ALL_TRIGGERS, RuleProgram

    if not isinstance(program, RuleProgram):
        raise TypeError("program must be a RuleProgram")
    return (
        ALL_COLLISIONS.index(program.collision),
        ALL_TRIGGERS.index(program.trigger),
        ALL_RELATIONS.index(program.relation),
    )


def rulegrid_tasks_to_oracle_factor_batch(
    tasks: Sequence[Any],
    *,
    diagnostic_indices: Sequence[int],
    device: torch.device | str | None = None,
    canonicalize_palette: bool = False,
) -> OracleFactorBatch:
    """Build an explicitly privileged factor-conditioned transition panel.

    Only the selected diagnostic targets are materialized.  This lets a
    training call use indices 0..20 while a separate held-out call uses the
    triple-composition indices 21..23.
    """

    materialized = tuple(tasks)
    if not materialized:
        raise ValueError("at least one RuleGrid task is required")
    tensor_batch = rulegrid_tasks_to_tensor_batch(
        materialized,
        include_behavior_targets=False,
        diagnostic_indices=tuple(diagnostic_indices),
        device=device,
    )
    if (
        tensor_batch.query_states is None
        or tensor_batch.query_actions is None
        or tensor_batch.query_targets is None
    ):
        raise AssertionError("RuleGrid adapter did not materialize the requested panel")
    target_device = tensor_batch.query_states.device
    states = tensor_batch.query_states
    targets = tensor_batch.query_targets
    if canonicalize_palette:
        from .rulegrid import NUM_COLORS

        lookup_rows: list[list[int]] = []
        for task in materialized:
            lookup = list(range(NUM_COLORS))
            for canonical_id, field in enumerate(
                fields(task.privileged.palette), start=1
            ):
                lookup[getattr(task.privileged.palette, field.name)] = canonical_id
            lookup_rows.append(lookup)
        lookup_tensor = torch.tensor(
            lookup_rows, dtype=torch.long, device=target_device
        )

        def canonicalize(grids: Tensor) -> Tensor:
            flattened = grids.reshape(grids.shape[0], -1)
            return lookup_tensor.gather(1, flattened).reshape_as(grids)

        states = canonicalize(states)
        targets = canonicalize(targets)
    factors = torch.tensor(
        [rule_program_factor_ids(task.privileged.true_program) for task in materialized],
        dtype=torch.long,
        device=target_device,
    )
    return OracleFactorBatch(
        states=states,
        actions=tensor_batch.query_actions,
        targets=targets,
        factor_ids=factors,
        action_mask=tensor_batch.query_action_mask,
        palette_canonicalized=canonicalize_palette,
    )


def canonicalize_rulegrid_tensor_batch(
    batch: RuleGridTensorBatch,
    tasks: Sequence[Any],
) -> RuleGridTensorBatch:
    """Map a RuleGrid tensor batch from random colors to palette-role IDs.

    This is an explicitly privileged diagnostic transform: it reads each
    task's nuisance palette but never its program.  It lets an experiment ask
    whether a model can infer *rule uncertainty* once the separate color-role
    binding problem has been removed.
    """

    materialized = tuple(tasks)
    if len(materialized) != batch.batch_size:
        raise ValueError("tasks must match the tensor batch size")
    from .rulegrid import NUM_COLORS

    lookup_rows: list[list[int]] = []
    for task in materialized:
        lookup = list(range(NUM_COLORS))
        for canonical_id, field in enumerate(
            fields(task.privileged.palette), start=1
        ):
            lookup[getattr(task.privileged.palette, field.name)] = canonical_id
        lookup_rows.append(lookup)
    lookup_tensor = torch.tensor(
        lookup_rows,
        dtype=torch.long,
        device=batch.support_states.device,
    )

    def transform(grids: Tensor | None) -> Tensor | None:
        if grids is None:
            return None
        if grids.shape[0] != len(materialized):
            raise ValueError("every grid tensor must start with the task batch axis")
        flattened = grids.reshape(grids.shape[0], -1)
        return lookup_tensor.gather(1, flattened).reshape_as(grids)

    return replace(
        batch,
        support_states=transform(batch.support_states),
        support_targets=transform(batch.support_targets),
        query_states=transform(batch.query_states),
        query_targets=transform(batch.query_targets),
        behavior_targets=transform(batch.behavior_targets),
    )


def rulegrid_tasks_to_canonical_tensor_batch(
    tasks: Sequence[Any],
    **kwargs: Any,
) -> RuleGridTensorBatch:
    """Build the standard neural batch, then apply oracle role canonicalization."""

    materialized = tuple(tasks)
    batch = rulegrid_tasks_to_tensor_batch(materialized, **kwargs)
    return canonicalize_rulegrid_tensor_batch(batch, materialized)


def _pad_public_action_panel(
    action_grid: list[list[Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    if not action_grid or not action_grid[0]:
        raise ValueError("action grid cannot be empty")
    count = len(action_grid[0])
    if any(len(row) != count for row in action_grid):
        raise ValueError("all tasks must have equal action counts")
    max_atoms = max(action.shape[0] for row in action_grid for action in row)
    padded = torch.zeros(
        len(action_grid),
        count,
        max_atoms,
        ACTION_FIELDS,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        len(action_grid),
        count,
        max_atoms,
        dtype=torch.bool,
        device=device,
    )
    for task_index, actions in enumerate(action_grid):
        for action_index, action in enumerate(actions):
            atoms = action.shape[0]
            padded[task_index, action_index, :atoms] = action.to(device)
            mask[task_index, action_index, :atoms] = True
    if max_atoms == 1:
        return padded[:, :, 0], None
    return padded, mask


def rulegrid_tasks_to_canonical_behavior_batch(
    tasks: Sequence[Any],
    *,
    diagnostic_indices: Sequence[int],
    prefix_length: int = 6,
    device: torch.device | str | None = None,
) -> RuleGridTensorBatch:
    """Build support-derived unordered behavior supervision without true targets.

    The model-facing inputs contain only public grids/actions and observed
    support outcomes.  The training-only behavior set is derived by applying
    ``version_space`` to that support, then simulating every compatible class.
    No true program, task ID, probe ID, or true diagnostic target is read.
    Palette role canonicalization remains explicitly privileged.
    """

    from .rulegrid import behavior_classes, simulate, version_space

    materialized = tuple(tasks)
    indices = tuple(diagnostic_indices)
    if not materialized:
        raise ValueError("at least one RuleGrid task is required")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("diagnostic_indices must be non-empty and unique")
    if not 1 <= prefix_length <= 6:
        raise ValueError("prefix_length must lie in 1..6")
    target_device = (
        torch.device(device) if device is not None else torch.device("cpu")
    )

    support_states: list[list[Any]] = []
    support_targets: list[list[Any]] = []
    support_actions: list[list[Tensor]] = []
    query_states: list[list[Any]] = []
    query_actions: list[list[Tensor]] = []
    behavior_targets: list[list[list[Any]]] = []
    behavior_masses: list[list[float]] = []
    for task in materialized:
        support = task.inference.support[:prefix_length]
        if len(support) != prefix_length:
            raise ValueError("task has fewer support transitions than requested")
        diagnostics = tuple(task.inference.diagnostics[index] for index in indices)
        support_states.append([transition.state for transition in support])
        support_targets.append([transition.next_state for transition in support])
        support_actions.append(
            [encode_public_action(transition.action) for transition in support]
        )
        query_states.append([probe.state for probe in diagnostics])
        query_actions.append(
            [encode_public_action(probe.action) for probe in diagnostics]
        )
        compatible = version_space(support, task.privileged.palette)
        classes = behavior_classes(
            compatible,
            diagnostics,
            task.privileged.palette,
        )
        ordered = tuple(sorted(classes.items(), key=lambda item: item[0]))
        if not 1 <= len(ordered) <= 4:
            raise ValueError("behavior supervision requires one to four classes")
        task_panels: list[list[Any]] = []
        task_masses: list[float] = []
        for _, programs in ordered:
            task_panels.append(
                [
                    simulate(
                        probe.state,
                        probe.action,
                        programs[0],
                        task.privileged.palette,
                    )
                    for probe in diagnostics
                ]
            )
            task_masses.append(len(programs) / len(compatible))
        behavior_targets.append(task_panels)
        behavior_masses.append(task_masses)

    maximum_classes = max(len(panels) for panels in behavior_targets)
    padded_targets: list[list[list[Any]]] = []
    padded_masses: list[list[float]] = []
    for panels, masses in zip(behavior_targets, behavior_masses, strict=True):
        padded_targets.append(
            panels + [panels[0]] * (maximum_classes - len(panels))
        )
        padded_masses.append(masses + [0.0] * (maximum_classes - len(masses)))
    support_action_tensor, support_action_mask = _pad_public_action_panel(
        support_actions, target_device
    )
    query_action_tensor, query_action_mask = _pad_public_action_panel(
        query_actions, target_device
    )
    raw = RuleGridTensorBatch(
        support_states=torch.tensor(
            support_states, dtype=torch.long, device=target_device
        ),
        support_actions=support_action_tensor,
        support_targets=torch.tensor(
            support_targets, dtype=torch.long, device=target_device
        ),
        support_mask=torch.ones(
            len(materialized), prefix_length, dtype=torch.bool, device=target_device
        ),
        query_states=torch.tensor(
            query_states, dtype=torch.long, device=target_device
        ),
        query_actions=query_action_tensor,
        query_targets=None,
        behavior_targets=torch.tensor(
            padded_targets, dtype=torch.long, device=target_device
        ),
        behavior_mass=torch.tensor(
            padded_masses, dtype=torch.float32, device=target_device
        ),
        support_action_mask=support_action_mask,
        query_action_mask=query_action_mask,
    )
    return canonicalize_rulegrid_tensor_batch(raw, materialized)


class TiedSingleBelief(PersistentK4):
    """Capacity-matched effective-K=1 control using the K=4 tensor interface.

    All four interface slots start from the same averaged parameter and remain
    identical under the shared deterministic updater.  This retains virtually
    the same parameterization and training path while forbidding multiple
    latent hypotheses.
    """

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> InferenceState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = device or self.initial_modes.device
        shared = self.initial_modes.mean(dim=0).to(device)
        modes = shared[None, None].expand(
            batch_size, self.config.particles, -1
        )
        log_weights = torch.full(
            (batch_size, self.config.particles),
            -math.log(self.config.particles),
            device=device,
            dtype=modes.dtype,
        )
        empty = torch.empty((batch_size, 0), dtype=modes.dtype, device=device)
        return InferenceState(
            modes=modes,
            log_weights=log_weights,
            prequential_mixture_log_prob=empty,
        )


def predict_persistent_panel(
    model: PersistentK4,
    batch: RuleGridTensorBatch,
    inference: InferenceState,
) -> OutcomePrediction:
    """Predict every public query with every persistent rule hypothesis."""

    if batch.query_states is None or batch.query_actions is None:
        raise ValueError("query inputs are required")
    batch_size, queries, height, width = batch.query_states.shape
    flat_states = batch.query_states.reshape(batch_size * queries, height, width)
    if batch.query_actions.ndim == 3:
        flat_actions = batch.query_actions.reshape(batch_size * queries, ACTION_FIELDS)
    else:
        atoms = batch.query_actions.shape[2]
        flat_actions = batch.query_actions.reshape(
            batch_size * queries, atoms, ACTION_FIELDS
        )
    flat_mask = (
        batch.query_action_mask.reshape(batch_size * queries, -1)
        if batch.query_action_mask is not None
        else None
    )
    flat_modes = (
        inference.modes[:, None]
        .expand(-1, queries, -1, -1)
        .reshape(
            batch_size * queries,
            model.config.particles,
            model.config.rule_dim,
        )
    )
    return model.predict(flat_states, flat_actions, flat_modes, flat_mask)


def row_hard_assignment_loss(
    cost: Tensor,
    weights: Tensor,
    target_mass: Tensor,
) -> Tensor:
    """Permutation-free set assignment with optional duplicate predictions."""

    if cost.ndim != 3:
        raise ValueError("cost must have [B,K,M] shape")
    batch_size, modes, classes = cost.shape
    if not 1 <= classes <= modes:
        raise ValueError("assignment needs 1 <= M <= K")
    if weights.shape != (batch_size, modes):
        raise ValueError("weights must have [B,K] shape")
    if target_mass.shape != (batch_size, classes):
        raise ValueError("target_mass must have [B,M] shape")
    assignments = torch.tensor(
        list(itertools.product(range(classes), repeat=modes)),
        device=cost.device,
        dtype=torch.long,
    )
    mode_index = torch.arange(modes, device=cost.device)
    chosen_cost = cost[:, mode_index[None, :], assignments]
    expected_cost = (chosen_cost * weights[:, None, :]).sum(dim=-1)
    assignment_one_hot = F.one_hot(assignments, classes).to(dtype=weights.dtype)
    predicted_mass = torch.einsum(
        "bk,zkm->bzm", weights, assignment_one_hot
    )
    delta = 1e-4
    smoothed = (predicted_mass + delta) / (1.0 + classes * delta)
    target = target_mass[:, None, :].expand(-1, assignments.shape[0], -1)
    kl = (
        target
        * (target.clamp_min(delta).log() - smoothed.log())
    ).sum(dim=-1)
    return (expected_cost + 0.25 * kl).amin(dim=1).mean()


def injective_assignment_loss(cost: Tensor) -> Tensor:
    """Match K modes to K behavior panels with a strict one-to-one assignment."""

    if cost.ndim != 3:
        raise ValueError("cost must have [B,K,M] shape")
    _, modes, classes = cost.shape
    if modes != classes:
        raise ValueError("injective assignment currently requires K == M")
    assignments = torch.tensor(
        list(itertools.permutations(range(classes))),
        device=cost.device,
        dtype=torch.long,
    )
    mode_index = torch.arange(modes, device=cost.device)
    chosen = cost[:, mode_index[None, :], assignments]
    return chosen.mean(dim=-1).amin(dim=1).mean()


def balanced_behavior_assignment_loss(
    model: PersistentK4,
    batch: RuleGridTensorBatch,
    inference: InferenceState,
) -> Tensor:
    """Training-only changed/unchanged-balanced full-panel set objective."""

    batch.validate(model.config)
    if (
        batch.query_states is None
        or batch.behavior_targets is None
        or batch.behavior_mass is None
    ):
        raise ValueError("behavior panels and public query inputs are required")
    prediction = predict_persistent_panel(model, batch, inference)
    batch_size, classes, queries, height, width = batch.behavior_targets.shape
    costs: list[Tensor] = []
    flat_states = batch.query_states.reshape(batch_size * queries, height, width)
    for class_index in range(classes):
        target = batch.behavior_targets[:, class_index].reshape(
            batch_size * queries, height, width
        )
        nll = -prediction.log_prob_cells(target).reshape(
            batch_size,
            queries,
            model.config.particles,
            height,
            width,
        )
        changed = target.ne(flat_states).reshape(
            batch_size, queries, 1, height, width
        )
        changed_count = changed.sum(dim=(1, 3, 4)).clamp_min(1)
        unchanged_count = (~changed).sum(dim=(1, 3, 4)).clamp_min(1)
        changed_nll = (nll * changed).sum(dim=(1, 3, 4)) / changed_count
        unchanged_nll = (nll * ~changed).sum(dim=(1, 3, 4)) / unchanged_count
        costs.append(0.5 * (changed_nll + unchanged_nll))
    cost = torch.stack(costs, dim=-1)
    if not torch.all(batch.behavior_mass > 0):
        raise ValueError("balanced static coverage requires exactly K valid classes")
    return injective_assignment_loss(cost)


def outcome_map(prediction: OutcomePrediction) -> Tensor:
    """Return the coherent categorical MAP grid for every prediction mode.

    The returned shape is ``[N,K,H,W]``.  Unlike independently thresholding
    the change head, this compares the no-change outcome with every possible
    changed color in one normalized categorical outcome space.
    """

    _, _, colors, _, _ = prediction.new_color_logits.shape
    original = prediction.input_colors[:, None, None]
    color_ids = torch.arange(
        colors, device=prediction.input_colors.device
    )[None, None, :, None, None]
    masked_logits = prediction.new_color_logits.masked_fill(
        original == color_ids,
        torch.finfo(prediction.new_color_logits.dtype).min,
    )
    log_outcomes = F.logsigmoid(prediction.change_logits)[:, :, None] + F.log_softmax(
        masked_logits, dim=2
    )
    no_change = F.logsigmoid(-prediction.change_logits).unsqueeze(2)
    log_outcomes = log_outcomes.scatter(
        2,
        original.expand(-1, log_outcomes.shape[1], -1, -1, -1),
        no_change,
    )
    return log_outcomes.argmax(dim=2)


class OracleFactorExecutor(nn.Module):
    """Shared transition executor conditioned on three privileged rule factors."""

    def __init__(self, config: NeuralPRPConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralPRPConfig()
        self.grid_encoder = GridEncoder(self.config)
        self.action_encoder = ActionEncoder(self.config)
        self.decoder = FiLMDecoder(self.config)
        self.factor_embeddings = nn.ModuleList(
            nn.Embedding(RULE_FACTOR_CARDINALITY, self.config.rule_dim)
            for _ in range(RULE_FACTOR_COUNT)
        )
        self.factor_mixer = nn.Sequential(
            nn.LayerNorm(self.config.rule_dim),
            nn.Linear(self.config.rule_dim, self.config.rule_dim),
            nn.SiLU(),
            nn.Linear(self.config.rule_dim, self.config.rule_dim),
        )

    def rule_latent(self, factor_ids: Tensor) -> Tensor:
        if factor_ids.ndim != 2 or factor_ids.shape[1] != RULE_FACTOR_COUNT:
            raise ValueError("factor_ids must have [N,3] shape")
        if factor_ids.dtype != torch.long:
            raise TypeError("factor_ids must use torch.long dtype")
        if torch.any(factor_ids < 0) or torch.any(
            factor_ids >= RULE_FACTOR_CARDINALITY
        ):
            raise ValueError("factor_ids must lie in [0,4)")
        composed = sum(
            embedding(factor_ids[:, axis])
            for axis, embedding in enumerate(self.factor_embeddings)
        )
        return self.factor_mixer(composed / (RULE_FACTOR_COUNT**0.5))

    def predict(
        self,
        states: Tensor,
        actions: Tensor,
        factor_ids: Tensor,
        action_mask: Tensor | None = None,
    ) -> OutcomePrediction:
        """Predict one next-grid distribution per transition."""

        if states.ndim != 3 or states.shape[-2:] != (
            self.config.grid_size,
            self.config.grid_size,
        ):
            raise ValueError("states must have [N,H,W] with the configured grid size")
        if states.dtype != torch.long:
            raise TypeError("states must use torch.long dtype")
        if actions.shape[0] != states.shape[0] or actions.shape[-1] != ACTION_FIELDS:
            raise ValueError("actions must share the state batch and end in four fields")
        features = self.grid_encoder(states)
        action_latent = self.action_encoder(actions, action_mask)
        rule_latent = self.rule_latent(factor_ids)
        change, colors = self.decoder(features, torch.cat((rule_latent, action_latent), dim=-1))
        return OutcomePrediction(
            input_colors=states,
            change_logits=change[:, 0].unsqueeze(1),
            new_color_logits=colors.unsqueeze(1),
        )

    @staticmethod
    def _flatten_panel(batch: OracleFactorBatch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        batch_size, queries, height, width = batch.states.shape
        states = batch.states.reshape(batch_size * queries, height, width)
        targets = batch.targets.reshape(batch_size * queries, height, width)
        if batch.actions.ndim == 3:
            actions = batch.actions.reshape(batch_size * queries, ACTION_FIELDS)
        else:
            atoms = batch.actions.shape[2]
            actions = batch.actions.reshape(batch_size * queries, atoms, ACTION_FIELDS)
        action_mask = (
            batch.action_mask.reshape(batch_size * queries, -1)
            if batch.action_mask is not None
            else None
        )
        factor_ids = (
            batch.factor_ids[:, None]
            .expand(-1, queries, -1)
            .reshape(batch_size * queries, RULE_FACTOR_COUNT)
        )
        return states, actions, targets, factor_ids, action_mask

    def predict_panel(self, batch: OracleFactorBatch) -> OutcomePrediction:
        batch.validate(self.config)
        states, actions, _, factor_ids, action_mask = self._flatten_panel(batch)
        return self.predict(states, actions, factor_ids, action_mask)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        count = mask.sum()
        if int(count.detach().cpu()) == 0:
            return values.new_zeros(())
        return values.masked_select(mask).mean()

    def losses(
        self,
        batch: OracleFactorBatch,
        *,
        balanced_weight: float = 1.0,
    ) -> OracleFactorLoss:
        """Return proper NLL plus a training-only sparse-change auxiliary.

        ``proper_nll`` remains the only likelihood used at evaluation.  The
        balanced term prevents the diagnostic ceiling from taking the trivial
        copy shortcut on sparse grids; it must not be interpreted as a
        posterior score.
        """

        if balanced_weight < 0:
            raise ValueError("balanced_weight must be non-negative")
        batch.validate(self.config)
        states, actions, targets, factor_ids, action_mask = self._flatten_panel(batch)
        prediction = self.predict(states, actions, factor_ids, action_mask)
        cell_nll = -prediction.log_prob_cells(targets).squeeze(1)
        changed = targets.ne(states)
        changed_nll = self._masked_mean(cell_nll, changed)
        unchanged_nll = self._masked_mean(cell_nll, ~changed)
        if bool(changed.any().item()) and bool((~changed).any().item()):
            balanced_nll = 0.5 * (changed_nll + unchanged_nll)
        else:
            balanced_nll = changed_nll + unchanged_nll
        proper_nll = cell_nll.mean()
        total = proper_nll + balanced_weight * balanced_nll
        return OracleFactorLoss(
            total=total,
            proper_nll=proper_nll,
            balanced_nll=balanced_nll,
            changed_nll=changed_nll,
            unchanged_nll=unchanged_nll,
        )


class SpatialActionEncoder(nn.Module):
    """Scatter public action atoms onto their grid coordinates.

    The original pilot action encoder mean-pools composite atoms before the
    decoder sees them.  That is a useful generic baseline, but it erases the
    binding between an action atom and the local object it operates on.  This
    encoder retains the same public fields while representing a composite
    action as a sparse, permutation-invariant feature map.
    """

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        atom_dim = max(8, config.action_embedding // 2)
        self.config = config
        self.kind_embedding = nn.Embedding(config.num_action_kinds, atom_dim)
        self.direction_embedding = nn.Embedding(config.num_directions, atom_dim)
        self.project = nn.Sequential(
            nn.Linear(2 * atom_dim, config.action_embedding),
            nn.SiLU(),
            nn.Linear(config.action_embedding, config.action_embedding),
        )

    def forward(
        self, actions: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if actions.ndim == 2:
            if action_mask is not None:
                raise ValueError("atomic actions must not have an action mask")
            atoms = actions[:, None]
            mask = torch.ones(
                actions.shape[0], 1, dtype=torch.bool, device=actions.device
            )
        elif actions.ndim == 3:
            if action_mask is None or action_mask.shape != actions.shape[:-1]:
                raise ValueError("composite actions require a matching mask")
            atoms = actions
            mask = action_mask
        else:
            raise ValueError("actions must have [N,4] or [N,L,4] shape")
        if atoms.dtype != torch.long or atoms.shape[-1] != ACTION_FIELDS:
            raise TypeError("actions must be torch.long tensors ending in four fields")
        kind, row, column, direction = (atoms[..., index] for index in range(ACTION_FIELDS))
        if torch.any(kind < 0) or torch.any(kind >= self.config.num_action_kinds):
            raise ValueError("action kind is outside the model vocabulary")
        if torch.any(direction < 0) or torch.any(direction >= self.config.num_directions):
            raise ValueError("action direction is outside the model vocabulary")
        if torch.any(row < 0) or torch.any(row >= self.config.grid_size):
            raise ValueError("action row is outside the grid")
        if torch.any(column < 0) or torch.any(column >= self.config.grid_size):
            raise ValueError("action column is outside the grid")

        values = self.project(
            torch.cat(
                (self.kind_embedding(kind), self.direction_embedding(direction)), dim=-1
            )
        )
        values = values * mask.to(dtype=values.dtype).unsqueeze(-1)
        flat_positions = row * self.config.grid_size + column
        spatial = values.new_zeros(
            actions.shape[0],
            self.config.action_embedding,
            self.config.grid_size * self.config.grid_size,
        )
        spatial.scatter_add_(
            2,
            flat_positions[:, None, :].expand(-1, self.config.action_embedding, -1),
            values.transpose(1, 2),
        )
        spatial = spatial.reshape(
            actions.shape[0],
            self.config.action_embedding,
            self.config.grid_size,
            self.config.grid_size,
        )
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=values.dtype)
        pooled = values.sum(dim=1) / count.sqrt()
        return spatial, pooled


class SpatialOracleFactorExecutor(OracleFactorExecutor):
    """Oracle executor that preserves atom-to-grid binding for composition."""

    def __init__(self, config: NeuralPRPConfig | None = None) -> None:
        super().__init__(config)
        # The pooled ActionEncoder is intentionally removed in this controlled
        # A/B; every other major component remains the same.
        del self.action_encoder
        self.spatial_action_encoder = SpatialActionEncoder(self.config)
        self.state_action_fuse = nn.Sequential(
            nn.Conv2d(
                self.config.encoder_channels + self.config.action_embedding,
                self.config.encoder_channels,
                kernel_size=1,
            ),
            nn.GroupNorm(
                self.config.normalization_groups, self.config.encoder_channels
            ),
            nn.SiLU(),
        )

    def predict(
        self,
        states: Tensor,
        actions: Tensor,
        factor_ids: Tensor,
        action_mask: Tensor | None = None,
    ) -> OutcomePrediction:
        if states.ndim != 3 or states.shape[-2:] != (
            self.config.grid_size,
            self.config.grid_size,
        ):
            raise ValueError("states must have [N,H,W] with the configured grid size")
        if states.dtype != torch.long:
            raise TypeError("states must use torch.long dtype")
        if actions.shape[0] != states.shape[0] or actions.shape[-1] != ACTION_FIELDS:
            raise ValueError("actions must share the state batch and end in four fields")
        state_features = self.grid_encoder(states)
        action_map, action_latent = self.spatial_action_encoder(actions, action_mask)
        features = self.state_action_fuse(torch.cat((state_features, action_map), dim=1))
        rule_latent = self.rule_latent(factor_ids)
        change, colors = self.decoder(
            features, torch.cat((rule_latent, action_latent), dim=-1)
        )
        return OutcomePrediction(
            input_colors=states,
            change_logits=change[:, 0].unsqueeze(1),
            new_color_logits=colors.unsqueeze(1),
        )


__all__ = [
    "OracleFactorBatch",
    "OracleFactorExecutor",
    "OracleFactorLoss",
    "RULE_FACTOR_CARDINALITY",
    "RULE_FACTOR_COUNT",
    "SpatialActionEncoder",
    "SpatialOracleFactorExecutor",
    "TiedSingleBelief",
    "balanced_behavior_assignment_loss",
    "canonicalize_rulegrid_tensor_batch",
    "injective_assignment_loss",
    "outcome_map",
    "predict_persistent_panel",
    "row_hard_assignment_loss",
    "rule_program_factor_ids",
    "rulegrid_tasks_to_canonical_behavior_batch",
    "rulegrid_tasks_to_canonical_tensor_batch",
    "rulegrid_tasks_to_oracle_factor_batch",
]
