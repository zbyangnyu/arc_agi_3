"""Capacity-matched four-branch RuleGrid executor ablations.

Both executors in this module have the same parameters, public canonical-role
router, spatial branch assignment, and decoder compute.  Their only difference
is the factor-conditioning graph:

* ``MatchedWiderGlobalOracleFactorExecutor`` gives every branch the full
  collision/trigger/relation tuple;
* ``MatchedFactorLocalOracleFactorExecutor`` gives each mechanism branch only
  its own factor value and makes the base branch factor-independent.

The classes remain privileged oracle-code diagnostics.  The router reads only
the canonicalized public state and action, never targets, axis labels, split,
seed, or probe identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .latent_rules import (
    OracleFactorExecutor,
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
)
from .neural import ACTION_FIELDS, FiLMDecoder, OutcomePrediction


BASE_BRANCH = 0
COLLISION_BRANCH = 1
TRIGGER_BRANCH = 2
RELATION_BRANCH = 3
BRANCH_COUNT = 4

_MOVE_ACTION_ID = 0
_ACTIVATE_ACTION_ID = 1
_CANONICAL_ACTOR_COLOR = 1
_CANONICAL_OBJECT_A_COLOR = 3
_CANONICAL_TRIGGER_COLOR = 5
_DIRECTION_VECTORS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def _normalized_atoms(
    states: Tensor,
    actions: Tensor,
    action_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if states.ndim != 3:
        raise ValueError("states must have [N,H,W] shape")
    if states.dtype != torch.long:
        raise TypeError("states must use torch.long dtype")
    if actions.ndim == 2:
        if action_mask is not None:
            raise ValueError("atomic actions cannot have an action mask")
        atoms = actions[:, None]
        mask = torch.ones(
            actions.shape[0],
            1,
            dtype=torch.bool,
            device=actions.device,
        )
    elif actions.ndim == 3:
        if (
            action_mask is None
            or action_mask.shape != actions.shape[:-1]
            or action_mask.dtype != torch.bool
        ):
            raise ValueError(
                "composite actions require a matching boolean action mask"
            )
        atoms = actions
        mask = action_mask
    else:
        raise ValueError("actions must have [N,4] or [N,L,4] shape")
    if (
        atoms.shape[0] != states.shape[0]
        or atoms.shape[-1] != ACTION_FIELDS
        or atoms.dtype != torch.long
    ):
        raise ValueError(
            "actions must share the state batch and end in four long fields"
        )
    height, width = states.shape[-2:]
    if torch.any(atoms[..., 1] < 0) or torch.any(atoms[..., 1] >= height):
        raise ValueError("action row is outside the state grid")
    if torch.any(atoms[..., 2] < 0) or torch.any(atoms[..., 2] >= width):
        raise ValueError("action column is outside the state grid")
    return atoms, mask


def _shift_mask(mask: Tensor, row_shift: int, column_shift: int) -> Tensor:
    """Translate a ``[N,H,W]`` mask without wraparound."""

    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with [N,H,W] shape")
    _, height, width = mask.shape
    output = torch.zeros_like(mask)
    source_row_start = max(0, -row_shift)
    source_row_end = min(height, height - row_shift)
    source_column_start = max(0, -column_shift)
    source_column_end = min(width, width - column_shift)
    if (
        source_row_start >= source_row_end
        or source_column_start >= source_column_end
    ):
        return output
    target_row_start = source_row_start + row_shift
    target_row_end = source_row_end + row_shift
    target_column_start = source_column_start + column_shift
    target_column_end = source_column_end + column_shift
    output[
        :,
        target_row_start:target_row_end,
        target_column_start:target_column_end,
    ] = mask[
        :,
        source_row_start:source_row_end,
        source_column_start:source_column_end,
    ]
    return output


def canonical_spatial_branch_assignment(
    states: Tensor,
    actions: Tensor,
    action_mask: Tensor | None = None,
) -> Tensor:
    """Return the one-hot-equivalent branch index for each output cell.

    The result has shape ``[N,H,W]`` and values in ``0..3``.  Branch zero owns
    all cells outside public mechanism envelopes, including pulse/background
    dynamics.  Collision/relation envelopes cover every possible write under
    their four rule values; trigger owns its public payload and socket cells.
    """

    atoms, atom_mask = _normalized_atoms(states, actions, action_mask)
    batch_size, height, width = states.shape
    assignment = torch.zeros(
        batch_size,
        height,
        width,
        dtype=torch.long,
        device=states.device,
    )
    batch_indices = torch.arange(batch_size, device=states.device)

    def assign_support(support: Tensor, branch: int) -> None:
        conflict = support & assignment.ne(BASE_BRANCH) & assignment.ne(branch)
        if bool(conflict.any().item()):
            raise ValueError("public mechanism branch envelopes overlap")
        assignment.masked_fill_(support, branch)

    for atom_index in range(atoms.shape[1]):
        valid = atom_mask[:, atom_index]
        atom = atoms[:, atom_index]
        kinds = atom[:, 0]
        rows = atom[:, 1]
        columns = atom[:, 2]
        directions = atom[:, 3]
        source_colors = states[batch_indices, rows, columns]

        trigger_examples = (
            valid
            & kinds.eq(_ACTIVATE_ACTION_ID)
            & source_colors.eq(_CANONICAL_TRIGGER_COLOR)
        )
        trigger_support = torch.zeros_like(assignment, dtype=torch.bool)
        for offset in (1, 2):
            target_columns = columns + offset
            inside = trigger_examples & target_columns.lt(width)
            trigger_support[
                batch_indices[inside],
                rows[inside],
                target_columns[inside],
            ] = True
        assign_support(trigger_support, TRIGGER_BRANCH)

        for role_color, branch, shifts in (
            (
                _CANONICAL_ACTOR_COLOR,
                COLLISION_BRANCH,
                (-1, 0, 1, 2),
            ),
            (
                _CANONICAL_OBJECT_A_COLOR,
                RELATION_BRANCH,
                (0, 1, 2),
            ),
        ):
            role_examples = (
                valid
                & kinds.eq(_MOVE_ACTION_ID)
                & source_colors.eq(role_color)
            )
            role_cells = states.eq(role_color)
            support = torch.zeros_like(role_cells)
            for direction_id, (row_vector, column_vector) in enumerate(
                _DIRECTION_VECTORS
            ):
                selected = role_examples & directions.eq(direction_id)
                if not bool(selected.any().item()):
                    continue
                selected_cells = role_cells & selected[:, None, None]
                for shift in shifts:
                    support |= _shift_mask(
                        selected_cells,
                        row_vector * shift,
                        column_vector * shift,
                    )
            assign_support(support, branch)

    return assignment


class _MatchedFourBranchOracleFactorExecutor(OracleFactorExecutor):
    """Shared implementation for the parameter-identical P1 conditions."""

    factor_local_conditioning: bool

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.axis_decoders = nn.ModuleList(
            FiLMDecoder(self.config) for _ in range(RULE_FACTOR_COUNT)
        )

    def initialize_from_oracle_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
    ) -> None:
        """Clone one audited decoder into all four matched branches."""

        expected_source_names = {
            name
            for name in self.state_dict()
            if not name.startswith("axis_decoders.")
        }
        if set(state_dict) != expected_source_names:
            raise ValueError("source state dict is not an OracleFactorExecutor")
        expanded = self.state_dict()
        for name in expanded:
            if name.startswith("axis_decoders."):
                _, _, suffix = name.split(".", 2)
                source_name = f"decoder.{suffix}"
            else:
                source_name = name
            expanded[name] = state_dict[source_name].detach().clone()
        self.load_state_dict(expanded, strict=True)

    def _masked_rule_latent(
        self,
        factor_ids: Tensor,
        active_axis: int | None,
    ) -> Tensor:
        if factor_ids.ndim != 2 or factor_ids.shape[1] != RULE_FACTOR_COUNT:
            raise ValueError("factor_ids must have [N,3] shape")
        if factor_ids.dtype != torch.long:
            raise TypeError("factor_ids must use torch.long dtype")
        if torch.any(factor_ids < 0) or torch.any(
            factor_ids >= RULE_FACTOR_CARDINALITY
        ):
            raise ValueError("factor_ids must lie in [0,4)")
        components = []
        for axis, embedding in enumerate(self.factor_embeddings):
            active = embedding(factor_ids[:, axis])
            reference = embedding.weight.mean(dim=0)[None].expand_as(active)
            components.append(active if axis == active_axis else reference)
        return self.factor_mixer(
            sum(components) / (RULE_FACTOR_COUNT**0.5)
        )

    def _branch_rule_latents(self, factor_ids: Tensor) -> tuple[Tensor, ...]:
        if not self.factor_local_conditioning:
            latent = self.rule_latent(factor_ids)
            return (latent, latent, latent, latent)
        return (
            self._masked_rule_latent(factor_ids, None),
            self._masked_rule_latent(factor_ids, 0),
            self._masked_rule_latent(factor_ids, 1),
            self._masked_rule_latent(factor_ids, 2),
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
        atoms, normalized_mask = _normalized_atoms(states, actions, action_mask)
        normalized_actions: Tensor = atoms[:, 0] if actions.ndim == 2 else atoms
        decoder_mask = None if actions.ndim == 2 else normalized_mask
        features = self.grid_encoder(states)
        action_latent = self.action_encoder(normalized_actions, decoder_mask)
        latents = self._branch_rule_latents(factor_ids)
        decoders = (self.decoder, *tuple(self.axis_decoders))
        changes = []
        colors = []
        for decoder, rule_latent in zip(decoders, latents, strict=True):
            branch_change, branch_colors = decoder(
                features,
                torch.cat((rule_latent, action_latent), dim=-1),
            )
            changes.append(branch_change[:, 0])
            colors.append(branch_colors)
        branch_assignment = canonical_spatial_branch_assignment(
            states,
            actions,
            action_mask,
        )
        stacked_changes = torch.stack(changes, dim=1)
        selected_change = torch.gather(
            stacked_changes,
            1,
            branch_assignment[:, None],
        )
        stacked_colors = torch.stack(colors, dim=1)
        color_indices = branch_assignment[:, None, None].expand(
            -1,
            1,
            self.config.num_colors,
            -1,
            -1,
        )
        selected_colors = torch.gather(
            stacked_colors,
            1,
            color_indices,
        )
        return OutcomePrediction(
            input_colors=states,
            change_logits=selected_change,
            new_color_logits=selected_colors,
        )


class MatchedWiderGlobalOracleFactorExecutor(
    _MatchedFourBranchOracleFactorExecutor
):
    """Four matched decoders, each conditioned on the complete factor tuple."""

    factor_local_conditioning = False


class MatchedFactorLocalOracleFactorExecutor(
    _MatchedFourBranchOracleFactorExecutor
):
    """Four matched decoders with structurally local factor conditioning."""

    factor_local_conditioning = True


__all__ = [
    "BASE_BRANCH",
    "BRANCH_COUNT",
    "COLLISION_BRANCH",
    "MatchedFactorLocalOracleFactorExecutor",
    "MatchedWiderGlobalOracleFactorExecutor",
    "RELATION_BRANCH",
    "TRIGGER_BRANCH",
    "canonical_spatial_branch_assignment",
]
