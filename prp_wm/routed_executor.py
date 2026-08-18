"""Experimental factor-local routing for the privileged RuleGrid executor.

This module intentionally lives outside :mod:`prp_wm.latent_rules`: historical
capacity experiments pin that module by SHA256, while the routed executor is a
new architectural ablation with its own provenance boundary.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .latent_rules import (
    OracleFactorExecutor,
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
)
from .neural import ACTION_FIELDS, OutcomePrediction


class CanonicalRoleRoutedOracleFactorExecutor(OracleFactorExecutor):
    """Parameter-matched hard factor-routing diagnostic.

    The input must use the oracle-canonical RuleGrid palette. Public action
    atoms and the color at each action coordinate identify whether an atom is
    a collision, trigger, relation, or neutral event. Only factor values for
    the routed axes enter the rule latent; every inactive axis is replaced by
    the mean of its four learned embeddings.

    This is an oracle-role architectural ablation, not a deployable raw-input
    router. Its purpose is to test factor-local conditioning before training a
    public learned router.
    """

    _MOVE_ACTION_ID = 0
    _ACTIVATE_ACTION_ID = 1
    _CANONICAL_ACTOR_COLOR = 1
    _CANONICAL_OBJECT_A_COLOR = 3
    _CANONICAL_TRIGGER_COLOR = 5

    @classmethod
    def active_factor_mask(
        cls,
        states: Tensor,
        actions: Tensor,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Return routed collision/trigger/relation axes as ``[N,3]`` bool."""

        if states.ndim != 3:
            raise ValueError("states must have [N,H,W] shape")
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

        rows = atoms[..., 1]
        columns = atoms[..., 2]
        batch = torch.arange(
            states.shape[0],
            device=states.device,
        )[:, None].expand_as(rows)
        source_colors = states[batch, rows, columns]
        kinds = atoms[..., 0]
        collision = (
            mask
            & kinds.eq(cls._MOVE_ACTION_ID)
            & source_colors.eq(cls._CANONICAL_ACTOR_COLOR)
        ).any(dim=1)
        trigger = (
            mask
            & kinds.eq(cls._ACTIVATE_ACTION_ID)
            & source_colors.eq(cls._CANONICAL_TRIGGER_COLOR)
        ).any(dim=1)
        relation = (
            mask
            & kinds.eq(cls._MOVE_ACTION_ID)
            & source_colors.eq(cls._CANONICAL_OBJECT_A_COLOR)
        ).any(dim=1)
        return torch.stack((collision, trigger, relation), dim=-1)

    def routed_rule_latent(
        self,
        factor_ids: Tensor,
        active_factor_mask: Tensor,
    ) -> Tensor:
        """Compose active factor values and mean-reference inactive axes."""

        if factor_ids.ndim != 2 or factor_ids.shape[1] != RULE_FACTOR_COUNT:
            raise ValueError("factor_ids must have [N,3] shape")
        if factor_ids.dtype != torch.long:
            raise TypeError("factor_ids must use torch.long dtype")
        if torch.any(factor_ids < 0) or torch.any(
            factor_ids >= RULE_FACTOR_CARDINALITY
        ):
            raise ValueError("factor_ids must lie in [0,4)")
        if (
            active_factor_mask.shape != factor_ids.shape
            or active_factor_mask.dtype != torch.bool
        ):
            raise ValueError(
                "active_factor_mask must be boolean with factor_ids shape"
            )
        components = []
        for axis, embedding in enumerate(self.factor_embeddings):
            active = embedding(factor_ids[:, axis])
            reference = embedding.weight.mean(dim=0)[None].expand_as(active)
            components.append(
                torch.where(
                    active_factor_mask[:, axis, None],
                    active,
                    reference,
                )
            )
        composed = sum(components)
        return self.factor_mixer(composed / (RULE_FACTOR_COUNT**0.5))

    def predict_from_rule_latent(
        self,
        states: Tensor,
        actions: Tensor,
        rule_latent: Tensor,
        action_mask: Tensor | None = None,
    ) -> OutcomePrediction:
        """Run the inherited trunk from an explicitly composed rule latent."""

        if states.ndim != 3 or states.shape[-2:] != (
            self.config.grid_size,
            self.config.grid_size,
        ):
            raise ValueError("states must have [N,H,W] with the configured grid size")
        if states.dtype != torch.long:
            raise TypeError("states must use torch.long dtype")
        if actions.shape[0] != states.shape[0] or actions.shape[-1] != ACTION_FIELDS:
            raise ValueError("actions must share the state batch and end in four fields")
        if rule_latent.shape != (states.shape[0], self.config.rule_dim):
            raise ValueError("rule_latent must have [N,rule_dim] shape")
        features = self.grid_encoder(states)
        action_latent = self.action_encoder(actions, action_mask)
        change, colors = self.decoder(
            features, torch.cat((rule_latent, action_latent), dim=-1)
        )
        return OutcomePrediction(
            input_colors=states,
            change_logits=change[:, 0].unsqueeze(1),
            new_color_logits=colors.unsqueeze(1),
        )

    def predict(
        self,
        states: Tensor,
        actions: Tensor,
        factor_ids: Tensor,
        action_mask: Tensor | None = None,
    ) -> OutcomePrediction:
        routed = self.active_factor_mask(states, actions, action_mask)
        return self.predict_from_rule_latent(
            states,
            actions,
            self.routed_rule_latent(factor_ids, routed),
            action_mask,
        )


__all__ = ["CanonicalRoleRoutedOracleFactorExecutor"]
