"""Axis-structured causal-mechanism hypothesis ceiling for RuleGrid.

This module tests one deliberately strong inductive bias: an episode rule is a
composition of three persistent mechanisms, while the value of each mechanism
must be inferred from public support transitions.  The mechanism axes and a
previously verified factor-conditioned executor are privileged structure; no
program ID, factor-value label, task/probe string, or true query target enters
inference or training.

The result is therefore a diagnostic ceiling for modular causal abstraction,
not an ARC agent and not autonomous causal discovery from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.causal_rules requires PyTorch") from error

from .latent_rules import (
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
    OracleFactorExecutor,
)
from .neural import (
    ACTION_FIELDS,
    NeuralPRPConfig,
    OutcomePrediction,
    RuleGridTensorBatch,
    SetInteraction,
)


@dataclass(frozen=True)
class CausalMechanismInference:
    """A persistent set of hard compositional mechanism hypotheses."""

    factor_logits: Tensor  # [B,K,3,4]
    factor_probabilities: Tensor  # [B,K,3,4]
    factor_codes: Tensor  # straight-through hard one-hot [B,K,3,4]
    factor_ids: Tensor  # [B,K,3]
    rule_latents: Tensor  # [B,K,D]

    @property
    def batch_size(self) -> int:
        return int(self.factor_ids.shape[0])

    @property
    def particles(self) -> int:
        return int(self.factor_ids.shape[1])


@dataclass(frozen=True)
class CausalMechanismLoss:
    """Strict behavior-set objective plus passive causal consistency."""

    total: Tensor
    set_nll: Tensor
    support_nll: Tensor
    duplicate_probability: Tensor
    factor_entropy: Tensor
    inference: CausalMechanismInference

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_set_nll": float(self.set_nll.detach().cpu()),
            "loss_support_nll": float(self.support_nll.detach().cpu()),
            "loss_duplicate_probability": float(
                self.duplicate_probability.detach().cpu()
            ),
            "factor_entropy_nats": float(self.factor_entropy.detach().cpu()),
        }


class MechanismCrossLayer(nn.Module):
    """Permutation-invariant cross-attention from hypothesis slots to evidence."""

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(config.rule_dim)
        self.token_norm = nn.LayerNorm(config.rule_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.rule_dim,
            config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.slot_interaction = SetInteraction(config)

    def forward(
        self,
        slots: Tensor,
        tokens: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        attended, _ = self.cross_attention(
            self.slot_norm(slots),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=~support_mask,
            need_weights=False,
        )
        return self.slot_interaction(slots + attended)


class AxisStructuredCausalK4(nn.Module):
    """Infer four hard, axis-composed mechanism codes from public support."""

    def __init__(
        self,
        executor: OracleFactorExecutor,
        *,
        attention_layers: int = 2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if attention_layers <= 0:
            raise ValueError("attention_layers must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.executor = executor
        self.config = executor.config
        self.temperature = float(temperature)
        for parameter in self.executor.parameters():
            parameter.requires_grad_(False)

        channels = self.config.encoder_channels
        transition_input = 5 * channels + self.config.action_embedding + 1
        self.transition_project = nn.Sequential(
            nn.LayerNorm(transition_input),
            nn.Linear(transition_input, self.config.attention_ffn),
            nn.SiLU(),
            nn.Linear(self.config.attention_ffn, self.config.rule_dim),
        )
        self.initial_slots = nn.Parameter(
            torch.empty(self.config.particles, self.config.rule_dim)
        )
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)
        self.cross_layers = nn.ModuleList(
            MechanismCrossLayer(self.config) for _ in range(attention_layers)
        )
        self.factor_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.config.rule_dim),
                nn.Linear(self.config.rule_dim, RULE_FACTOR_CARDINALITY),
            )
            for _ in range(RULE_FACTOR_COUNT)
        )

    @staticmethod
    def _changed_pool(features: Tensor, changed: Tensor) -> Tensor:
        denominator = changed.sum(dim=(-2, -1)).clamp_min(1.0)
        return (features * changed).sum(dim=(-2, -1)) / denominator

    def _transition_tokens(self, batch: RuleGridTensorBatch) -> Tensor:
        states = batch.support_states
        targets = batch.support_targets
        batch_size, steps, height, width = states.shape
        flat_states = states.reshape(batch_size * steps, height, width)
        flat_targets = targets.reshape(batch_size * steps, height, width)
        state_features = self.executor.grid_encoder(flat_states)
        target_features = self.executor.grid_encoder(flat_targets)
        if batch.support_actions.ndim == 3:
            flat_actions = batch.support_actions.reshape(
                batch_size * steps, ACTION_FIELDS
            )
        else:
            atoms = batch.support_actions.shape[2]
            flat_actions = batch.support_actions.reshape(
                batch_size * steps, atoms, ACTION_FIELDS
            )
        flat_action_mask = (
            batch.support_action_mask.reshape(batch_size * steps, -1)
            if batch.support_action_mask is not None
            else None
        )
        action_features = self.executor.action_encoder(
            flat_actions, flat_action_mask
        )
        changed = flat_states.ne(flat_targets).to(
            dtype=state_features.dtype
        )[:, None]
        change_fraction = changed.mean(dim=(-2, -1))
        evidence = torch.cat(
            (
                state_features.mean(dim=(-2, -1)),
                target_features.mean(dim=(-2, -1)),
                (target_features - state_features).mean(dim=(-2, -1)),
                self._changed_pool(state_features, changed),
                self._changed_pool(target_features, changed),
                action_features,
                change_fraction,
            ),
            dim=-1,
        )
        return self.transition_project(evidence).reshape(
            batch_size, steps, self.config.rule_dim
        )

    def rule_latents_from_codes(self, factor_codes: Tensor) -> Tensor:
        """Compose differentiable one-hot codes through the frozen executor."""

        if factor_codes.ndim != 4 or factor_codes.shape[-2:] != (
            RULE_FACTOR_COUNT,
            RULE_FACTOR_CARDINALITY,
        ):
            raise ValueError("factor_codes must have [B,K,3,4] shape")
        composed = sum(
            factor_codes[:, :, axis] @ embedding.weight
            for axis, embedding in enumerate(self.executor.factor_embeddings)
        ) / math.sqrt(RULE_FACTOR_COUNT)
        shape = composed.shape
        return self.executor.factor_mixer(
            composed.reshape(-1, self.config.rule_dim)
        ).reshape(shape)

    def infer_support(
        self,
        batch: RuleGridTensorBatch,
        *,
        temperature: float | None = None,
    ) -> CausalMechanismInference:
        """Infer without accessing any query label or privileged rule value."""

        batch.validate(self.config)
        current_temperature = self.temperature if temperature is None else temperature
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")
        tokens = self._transition_tokens(batch)
        slots = self.initial_slots[None].expand(batch.batch_size, -1, -1)
        for layer in self.cross_layers:
            slots = layer(slots, tokens, batch.support_mask)
        logits = torch.stack(
            [head(slots) for head in self.factor_heads], dim=2
        )
        return self.inference_from_factor_logits(
            logits, temperature=current_temperature
        )

    def inference_from_factor_logits(
        self,
        logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> CausalMechanismInference:
        """Convert arbitrary factor logits into straight-through hard rules."""

        current_temperature = self.temperature if temperature is None else temperature
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")
        if logits.ndim != 4 or logits.shape[1:] != (
            self.config.particles,
            RULE_FACTOR_COUNT,
            RULE_FACTOR_CARDINALITY,
        ):
            raise ValueError("logits must have [B,K,3,4] shape")
        probabilities = F.softmax(logits / current_temperature, dim=-1)
        factor_ids = probabilities.argmax(dim=-1)
        hard = F.one_hot(
            factor_ids, RULE_FACTOR_CARDINALITY
        ).to(dtype=probabilities.dtype)
        # Forward is exactly a discrete code; backward follows the simplex.
        codes = hard + probabilities - probabilities.detach()
        return CausalMechanismInference(
            factor_logits=logits,
            factor_probabilities=probabilities,
            factor_codes=codes,
            factor_ids=factor_ids,
            rule_latents=self.rule_latents_from_codes(codes),
        )

    def _predict_with_rule_latents(
        self,
        states: Tensor,
        actions: Tensor,
        rule_latents: Tensor,
        action_mask: Tensor | None,
    ) -> OutcomePrediction:
        batch_size, height, width = states.shape
        particles = rule_latents.shape[1]
        if rule_latents.shape != (
            batch_size,
            particles,
            self.config.rule_dim,
        ):
            raise ValueError("rule_latents must have [N,K,D] shape")
        features = self.executor.grid_encoder(states)
        action_latent = self.executor.action_encoder(actions, action_mask)
        repeated_features = (
            features[:, None]
            .expand(-1, particles, -1, -1, -1)
            .reshape(
                batch_size * particles,
                self.config.encoder_channels,
                height,
                width,
            )
        )
        condition = torch.cat(
            (
                rule_latents,
                action_latent[:, None].expand(-1, particles, -1),
            ),
            dim=-1,
        ).reshape(batch_size * particles, -1)
        change, colors = self.executor.decoder(repeated_features, condition)
        return OutcomePrediction(
            input_colors=states,
            change_logits=change.reshape(batch_size, particles, height, width),
            new_color_logits=colors.reshape(
                batch_size,
                particles,
                self.config.num_colors,
                height,
                width,
            ),
        )

    @staticmethod
    def _flatten_panel_inputs(
        states: Tensor,
        actions: Tensor,
        action_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        batch_size, count, height, width = states.shape
        flat_states = states.reshape(batch_size * count, height, width)
        if actions.ndim == 3:
            flat_actions = actions.reshape(batch_size * count, ACTION_FIELDS)
        else:
            atoms = actions.shape[2]
            flat_actions = actions.reshape(
                batch_size * count, atoms, ACTION_FIELDS
            )
        flat_mask = (
            action_mask.reshape(batch_size * count, -1)
            if action_mask is not None
            else None
        )
        return flat_states, flat_actions, flat_mask

    def predict_panel(
        self,
        batch: RuleGridTensorBatch,
        inference: CausalMechanismInference,
    ) -> OutcomePrediction:
        if batch.query_states is None or batch.query_actions is None:
            raise ValueError("query inputs are required")
        batch_size, queries = batch.query_states.shape[:2]
        states, actions, action_mask = self._flatten_panel_inputs(
            batch.query_states,
            batch.query_actions,
            batch.query_action_mask,
        )
        rules = (
            inference.rule_latents[:, None]
            .expand(-1, queries, -1, -1)
            .reshape(
                batch_size * queries,
                self.config.particles,
                self.config.rule_dim,
            )
        )
        return self._predict_with_rule_latents(
            states, actions, rules, action_mask
        )

    def predict_support(
        self,
        batch: RuleGridTensorBatch,
        inference: CausalMechanismInference,
    ) -> OutcomePrediction:
        batch_size, steps = batch.support_states.shape[:2]
        states, actions, action_mask = self._flatten_panel_inputs(
            batch.support_states,
            batch.support_actions,
            batch.support_action_mask,
        )
        rules = (
            inference.rule_latents[:, None]
            .expand(-1, steps, -1, -1)
            .reshape(
                batch_size * steps,
                self.config.particles,
                self.config.rule_dim,
            )
        )
        return self._predict_with_rule_latents(
            states, actions, rules, action_mask
        )

    @staticmethod
    def _balanced_panel_cost(
        prediction: OutcomePrediction,
        input_states: Tensor,
        targets: Tensor,
        *,
        batch_size: int,
        panel_count: int,
        proper_weight: float,
        balanced_weight: float,
    ) -> Tensor:
        height, width = targets.shape[-2:]
        particles = prediction.change_logits.shape[1]
        flat_targets = targets.reshape(batch_size * panel_count, height, width)
        nll = -prediction.log_prob_cells(flat_targets).reshape(
            batch_size, panel_count, particles, height, width
        )
        changed = flat_targets.ne(input_states).reshape(
            batch_size, panel_count, 1, height, width
        )
        proper = nll.mean(dim=(1, 3, 4))
        changed_count = changed.sum(dim=(1, 3, 4)).clamp_min(1)
        unchanged_count = (~changed).sum(dim=(1, 3, 4)).clamp_min(1)
        changed_nll = (nll * changed).sum(dim=(1, 3, 4)) / changed_count
        unchanged_nll = (nll * ~changed).sum(dim=(1, 3, 4)) / unchanged_count
        return proper_weight * proper + balanced_weight * 0.5 * (
            changed_nll + unchanged_nll
        )

    @staticmethod
    def _soft_permutation_loss(cost: Tensor, temperature: float) -> Tensor:
        if cost.ndim != 3 or cost.shape[1] != cost.shape[2]:
            raise ValueError("cost must have square [B,K,K] shape")
        if temperature < 0:
            raise ValueError("assignment temperature must be non-negative")
        particles = cost.shape[1]
        permutations = torch.tensor(
            list(itertools.permutations(range(particles))),
            dtype=torch.long,
            device=cost.device,
        )
        mode_index = torch.arange(particles, device=cost.device)
        totals = cost[:, mode_index[None], permutations].mean(dim=-1)
        if temperature == 0:
            return totals.amin(dim=1).mean()
        # Normalization makes the zero-cost optimum exactly zero.
        return (
            -temperature * torch.logsumexp(-totals / temperature, dim=1)
            + temperature * math.log(permutations.shape[0])
        ).mean()

    @staticmethod
    def _duplicate_probability(probabilities: Tensor) -> Tensor:
        particles = probabilities.shape[1]
        pair_scores: list[Tensor] = []
        for left in range(particles):
            for right in range(left + 1, particles):
                per_axis_same = (
                    probabilities[:, left] * probabilities[:, right]
                ).sum(dim=-1)
                pair_scores.append(per_axis_same.prod(dim=-1))
        return torch.stack(pair_scores, dim=1).mean()

    def losses(
        self,
        batch: RuleGridTensorBatch,
        *,
        support_weight: float = 0.1,
        proper_weight: float = 1.0,
        balanced_weight: float = 1.0,
        duplicate_weight: float = 0.05,
        assignment_temperature: float = 0.05,
        temperature: float | None = None,
    ) -> CausalMechanismLoss:
        if min(
            support_weight,
            proper_weight,
            balanced_weight,
            duplicate_weight,
            assignment_temperature,
        ) < 0:
            raise ValueError("loss weights and assignment temperature cannot be negative")
        batch.validate(self.config)
        if (
            batch.query_states is None
            or batch.behavior_targets is None
            or batch.behavior_mass is None
        ):
            raise ValueError("public queries and unordered behavior panels are required")
        if batch.behavior_targets.shape[1] != self.config.particles:
            raise ValueError("causal K4 ceiling requires exactly four behavior classes")
        if not torch.all(batch.behavior_mass > 0):
            raise ValueError("all four behavior classes must be valid")

        inference = self.infer_support(batch, temperature=temperature)
        query_prediction = self.predict_panel(batch, inference)
        batch_size, classes, queries, height, width = batch.behavior_targets.shape
        flat_query_states = batch.query_states.reshape(
            batch_size * queries, height, width
        )
        class_costs: list[Tensor] = []
        for class_index in range(classes):
            class_costs.append(
                self._balanced_panel_cost(
                    query_prediction,
                    flat_query_states,
                    batch.behavior_targets[:, class_index],
                    batch_size=batch_size,
                    panel_count=queries,
                    proper_weight=proper_weight,
                    balanced_weight=balanced_weight,
                )
            )
        cost = torch.stack(class_costs, dim=-1)
        set_nll = self._soft_permutation_loss(
            cost, assignment_temperature
        )

        support_prediction = self.predict_support(batch, inference)
        support_states = batch.support_states.reshape(
            batch_size * batch.support_steps, height, width
        )
        support_cost = self._balanced_panel_cost(
            support_prediction,
            support_states,
            batch.support_targets,
            batch_size=batch_size,
            panel_count=batch.support_steps,
            proper_weight=proper_weight,
            balanced_weight=balanced_weight,
        )
        # Every particle, not a mixture or best mode, must fit observed support.
        support_nll = support_cost.mean()
        duplicate_probability = self._duplicate_probability(
            inference.factor_probabilities
        )
        factor_entropy = -(
            inference.factor_probabilities
            * inference.factor_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        total = (
            set_nll
            + support_weight * support_nll
            + duplicate_weight * duplicate_probability
        )
        return CausalMechanismLoss(
            total=total,
            set_nll=set_nll,
            support_nll=support_nll,
            duplicate_probability=duplicate_probability,
            factor_entropy=factor_entropy,
            inference=inference,
        )


__all__ = [
    "AxisStructuredCausalK4",
    "CausalMechanismInference",
    "CausalMechanismLoss",
    "MechanismCrossLayer",
]
