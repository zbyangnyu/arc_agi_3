"""Amortized K=4 inference trained against exact discrete rule costs.

The support encoder retains the three-axis categorical inductive bias from
``AxisStructuredCausalK4``.  Unlike its straight-through objective, training
never differentiates through a frozen decoder at interpolated rule latents.
Instead, the executor evaluates all 64 integer mechanism tuples under
``no_grad`` and supplies a detached cost table.  Gradients reach the encoder
only through the expected cost of its factorized categorical posterior.

The behavior panels and mechanism axes remain privileged training structure;
inference itself reads public support transitions only.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.discrete_causal_rules requires PyTorch") from error

from .causal_filter import enumerate_factor_codes
from .causal_rules import (
    AxisStructuredCausalK4,
    CausalMechanismInference,
)
from .latent_rules import OracleFactorExecutor
from .neural import RuleGridTensorBatch


@dataclass(frozen=True)
class ExpectedDiscreteLoss:
    """Strict behavior-set matching over a factorized posterior on 64 rules."""

    total: Tensor
    set_cost: Tensor
    validity_cost: Tensor
    diversity_barrier: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    inference: CausalMechanismInference
    joint_probabilities: Tensor  # [B,K,64]
    behavior_costs: Tensor  # detached [B,64,4]

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_set_cost": float(self.set_cost.detach().cpu()),
            "loss_validity_cost": float(self.validity_cost.detach().cpu()),
            "loss_diversity_barrier": float(
                self.diversity_barrier.detach().cpu()
            ),
            "joint_entropy_nats": float(self.joint_entropy.detach().cpu()),
            "mean_top_rule_probability": float(
                self.mean_top_probability.detach().cpu()
            ),
        }


class ExpectedDiscreteCausalK4(AxisStructuredCausalK4):
    """Infer K=4 rules with exact integer-code counterfactual credit assignment."""

    def __init__(
        self,
        executor: OracleFactorExecutor,
        *,
        attention_layers: int = 2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            executor,
            attention_layers=attention_layers,
            temperature=temperature,
        )
        factor_bank = enumerate_factor_codes()
        self.register_buffer("factor_bank", factor_bank, persistent=False)
        with torch.no_grad():
            bank_latents = self.executor.rule_latent(
                factor_bank.to(next(self.executor.parameters()).device)
            )
        self.register_buffer(
            "factor_bank_rule_latents",
            bank_latents.detach(),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "ExpectedDiscreteCausalK4":
        """Train the amortizer while keeping the frozen executor in eval mode."""

        super().train(mode)
        self.executor.eval()
        return self

    def joint_rule_log_probabilities(
        self,
        factor_logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        """Return the factorized joint log posterior with shape ``[B,K,64]``."""

        current_temperature = self.temperature if temperature is None else temperature
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")
        if factor_logits.ndim != 4 or factor_logits.shape[1:] != (
            self.config.particles,
            3,
            4,
        ):
            raise ValueError("factor_logits must have [B,K,3,4] shape")
        axis_log_probabilities = F.log_softmax(
            factor_logits / current_temperature,
            dim=-1,
        )
        selected = [
            axis_log_probabilities[:, :, axis, self.factor_bank[:, axis]]
            for axis in range(3)
        ]
        return torch.stack(selected, dim=0).sum(dim=0)

    def joint_rule_probabilities(
        self,
        factor_logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        return self.joint_rule_log_probabilities(
            factor_logits,
            temperature=temperature,
        ).exp()

    def _predict_all_query_codes(
        self,
        batch: RuleGridTensorBatch,
    ):
        if batch.query_states is None or batch.query_actions is None:
            raise ValueError("query inputs are required")
        batch_size, queries = batch.query_states.shape[:2]
        states, actions, action_mask = self._flatten_panel_inputs(
            batch.query_states,
            batch.query_actions,
            batch.query_action_mask,
        )
        rules = (
            self.factor_bank_rule_latents[None, None]
            .expand(batch_size, queries, -1, -1)
            .reshape(
                batch_size * queries,
                self.factor_bank.shape[0],
                self.config.rule_dim,
            )
        )
        return self._predict_with_rule_latents(
            states,
            actions,
            rules,
            action_mask,
        )

    def _predict_all_support_codes(
        self,
        batch: RuleGridTensorBatch,
    ):
        batch_size, steps = batch.support_states.shape[:2]
        states, actions, action_mask = self._flatten_panel_inputs(
            batch.support_states,
            batch.support_actions,
            batch.support_action_mask,
        )
        rules = (
            self.factor_bank_rule_latents[None, None]
            .expand(batch_size, steps, -1, -1)
            .reshape(
                batch_size * steps,
                self.factor_bank.shape[0],
                self.config.rule_dim,
            )
        )
        return self._predict_with_rule_latents(
            states,
            actions,
            rules,
            action_mask,
        )

    def discrete_behavior_costs(
        self,
        batch: RuleGridTensorBatch,
        *,
        proper_weight: float = 1.0,
        balanced_weight: float = 1.0,
    ) -> Tensor:
        """Return detached exact-code-to-behavior costs ``[B,64,4]``."""

        if proper_weight < 0 or balanced_weight < 0:
            raise ValueError("behavior cost weights cannot be negative")
        batch.validate(self.config)
        if (
            batch.query_states is None
            or batch.behavior_targets is None
            or batch.behavior_mass is None
        ):
            raise ValueError("query inputs and unordered behavior panels are required")
        if batch.behavior_targets.shape[1] != self.config.particles:
            raise ValueError("expected exactly four behavior classes")
        if not torch.all(batch.behavior_mass > 0):
            raise ValueError("all behavior classes must be valid")
        batch_size, classes, queries, height, width = batch.behavior_targets.shape
        with torch.no_grad():
            prediction = self._predict_all_query_codes(batch)
            flat_query_states = batch.query_states.reshape(
                batch_size * queries,
                height,
                width,
            )
            class_costs = [
                self._balanced_panel_cost(
                    prediction,
                    flat_query_states,
                    batch.behavior_targets[:, class_index],
                    batch_size=batch_size,
                    panel_count=queries,
                    proper_weight=proper_weight,
                    balanced_weight=balanced_weight,
                )
                for class_index in range(classes)
            ]
        return torch.stack(class_costs, dim=-1).detach()

    def discrete_support_costs(
        self,
        batch: RuleGridTensorBatch,
        *,
        proper_weight: float = 1.0,
        balanced_weight: float = 1.0,
    ) -> Tensor:
        """Return detached integer-code support costs with shape ``[B,64]``."""

        if proper_weight < 0 or balanced_weight < 0:
            raise ValueError("support cost weights cannot be negative")
        batch.validate(self.config)
        batch_size, steps, height, width = batch.support_states.shape
        if not torch.all(batch.support_mask):
            raise ValueError("expected-discrete ceiling requires a full support panel")
        with torch.no_grad():
            prediction = self._predict_all_support_codes(batch)
            flat_support_states = batch.support_states.reshape(
                batch_size * steps,
                height,
                width,
            )
            costs = self._balanced_panel_cost(
                prediction,
                flat_support_states,
                batch.support_targets,
                batch_size=batch_size,
                panel_count=steps,
                proper_weight=proper_weight,
                balanced_weight=balanced_weight,
            )
        return costs.detach()

    @staticmethod
    def _diversity_barrier(joint_probabilities: Tensor) -> Tensor:
        particles = joint_probabilities.shape[1]
        penalties: list[Tensor] = []
        epsilon = torch.finfo(joint_probabilities.dtype).eps
        for left in range(particles):
            for right in range(left + 1, particles):
                overlap = (
                    joint_probabilities[:, left]
                    * joint_probabilities[:, right]
                ).sum(dim=-1)
                penalties.append(
                    -torch.log1p(-overlap.clamp(max=1.0 - epsilon))
                )
        return torch.stack(penalties, dim=1).mean()

    def losses(
        self,
        batch: RuleGridTensorBatch,
        *,
        validity_weight: float = 0.10,
        diversity_weight: float = 0.10,
        sharpening_weight: float = 0.0,
        proper_weight: float = 1.0,
        balanced_weight: float = 1.0,
        assignment_temperature: float = 0.0,
        temperature: float | None = None,
    ) -> ExpectedDiscreteLoss:
        if min(
            validity_weight,
            diversity_weight,
            sharpening_weight,
            proper_weight,
            balanced_weight,
            assignment_temperature,
        ) < 0:
            raise ValueError("loss weights and assignment temperature cannot be negative")
        inference = self.infer_support(batch, temperature=temperature)
        joint_log_probabilities = self.joint_rule_log_probabilities(
            inference.factor_logits,
            temperature=temperature,
        )
        joint_probabilities = joint_log_probabilities.exp()
        behavior_costs = self.discrete_behavior_costs(
            batch,
            proper_weight=proper_weight,
            balanced_weight=balanced_weight,
        )
        support_costs = self.discrete_support_costs(
            batch,
            proper_weight=proper_weight,
            balanced_weight=balanced_weight,
        )
        expected_class_cost = torch.einsum(
            "bkr,brm->bkm",
            joint_probabilities,
            behavior_costs,
        )
        set_cost = self._soft_permutation_loss(
            expected_class_cost,
            assignment_temperature,
        )
        validity_cost = torch.einsum(
            "bkr,br->bk",
            joint_probabilities,
            support_costs,
        ).mean()
        diversity_barrier = self._diversity_barrier(joint_probabilities)
        joint_entropy = -(
            joint_probabilities
            * joint_log_probabilities
        ).sum(dim=-1).mean()
        mean_top_probability = joint_probabilities.amax(dim=-1).mean()
        total = (
            set_cost
            + validity_weight * validity_cost
            + diversity_weight * diversity_barrier
            + sharpening_weight * joint_entropy
        )
        return ExpectedDiscreteLoss(
            total=total,
            set_cost=set_cost,
            validity_cost=validity_cost,
            diversity_barrier=diversity_barrier,
            joint_entropy=joint_entropy,
            mean_top_probability=mean_top_probability,
            inference=inference,
            joint_probabilities=joint_probabilities,
            behavior_costs=behavior_costs,
        )


__all__ = [
    "ExpectedDiscreteCausalK4",
    "ExpectedDiscreteLoss",
]
