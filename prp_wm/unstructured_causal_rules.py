"""Parameter-matched unstructured 64-way causal-rule amortization.

This is the capacity control for :mod:`prp_wm.discrete_causal_rules`.  It
reuses exactly the same public-support encoder, frozen executor, and detached
integer-code cost tables, but replaces the three independent four-way heads
with one 64-way categorical head.  The parameter-matched primary control uses
a low-rank head; a direct ``LayerNorm -> Linear(64)`` head is available as a
capacity sensitivity.  Neither head factorizes its learned posterior by axis.

The factor bank remains necessary only to translate an opaque 64-way class
back into the frozen executor's three integer inputs.  It is not used to
factorize the learned posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError(
        "prp_wm.unstructured_causal_rules requires PyTorch"
    ) from error

from .discrete_causal_rules import ExpectedDiscreteCausalK4
from .latent_rules import (
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
    OracleFactorExecutor,
)
from .neural import RuleGridTensorBatch


RULE_CLASS_COUNT = RULE_FACTOR_CARDINALITY**RULE_FACTOR_COUNT


@dataclass(frozen=True)
class UnstructuredCausalInference:
    """Four opaque categorical hypotheses over all 64 rule classes."""

    rule_logits: Tensor  # [B,K,64]
    rule_log_probabilities: Tensor  # [B,K,64]
    rule_probabilities: Tensor  # [B,K,64]
    rule_codes: Tensor  # hard one-hot [B,K,64]
    rule_ids: Tensor  # [B,K]
    factor_ids: Tensor  # decoded executor inputs [B,K,3]
    rule_latents: Tensor  # [B,K,D]

    @property
    def factor_logits(self) -> Tensor:
        """Compatibility view consumed by the shared coverage evaluator.

        The singleton dimension is emphatically not a mechanism axis: the
        final dimension remains one indivisible 64-way categorical variable.
        """

        return self.rule_logits.unsqueeze(2)

    @property
    def factor_probabilities(self) -> Tensor:
        return self.rule_probabilities.unsqueeze(2)

    @property
    def factor_codes(self) -> Tensor:
        return self.rule_codes.unsqueeze(2)

    @property
    def batch_size(self) -> int:
        return int(self.rule_ids.shape[0])

    @property
    def particles(self) -> int:
        return int(self.rule_ids.shape[1])


@dataclass(frozen=True)
class UnstructuredExpectedDiscreteLoss:
    """Hard set matching and support validity for a direct 64-way posterior."""

    total: Tensor
    set_cost: Tensor
    validity_cost: Tensor
    diversity_barrier: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    inference: UnstructuredCausalInference
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


class LowRankRuleHead(nn.Module):
    """A low-rank map from one hypothesis slot to 64 opaque rule logits."""

    def __init__(self, rule_dim: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self.layers = nn.Sequential(
            nn.LayerNorm(rule_dim),
            nn.Linear(rule_dim, self.rank),
            nn.SiLU(),
            nn.Linear(self.rank, RULE_CLASS_COUNT),
        )

    def forward(self, slots: Tensor) -> Tensor:
        return self.layers(slots)


class DirectRuleHead(nn.Module):
    """A direct affine map from one hypothesis slot to 64 opaque logits."""

    def __init__(self, rule_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(rule_dim),
            nn.Linear(rule_dim, RULE_CLASS_COUNT),
        )

    def forward(self, slots: Tensor) -> Tensor:
        return self.layers(slots)


class UnstructuredDiscreteCausalK4(ExpectedDiscreteCausalK4):
    """Infer K=4 opaque rules from a direct, non-factorized 64-way posterior."""

    def __init__(
        self,
        executor: OracleFactorExecutor,
        *,
        attention_layers: int = 2,
        temperature: float = 1.0,
        head_kind: str = "low-rank",
        head_rank: int | None = None,
    ) -> None:
        if head_kind not in {"low-rank", "direct-linear"}:
            raise ValueError(
                "head_kind must be 'low-rank' or 'direct-linear'"
            )
        if head_kind == "direct-linear" and head_rank is not None:
            raise ValueError(
                "head_rank is only valid when head_kind='low-rank'"
            )
        super().__init__(
            executor,
            attention_layers=attention_layers,
            temperature=temperature,
        )

        # Remove the only axis-factorized trainable component.  The frozen
        # factor bank below is merely the executor's class-to-input adapter.
        del self.factor_heads
        self._head_kind = head_kind
        if head_kind == "low-rank":
            if head_rank is None:
                head_rank = self._closest_parameter_rank(self.config.rule_dim)
            self.rule_head = LowRankRuleHead(self.config.rule_dim, head_rank)
        else:
            self.rule_head = DirectRuleHead(self.config.rule_dim)

    @staticmethod
    def _factorized_head_parameter_count(rule_dim: int) -> int:
        # Per axis: affine LayerNorm + D-by-4 Linear (with four biases).
        return RULE_FACTOR_COUNT * (
            2 * rule_dim
            + rule_dim * RULE_FACTOR_CARDINALITY
            + RULE_FACTOR_CARDINALITY
        )

    @staticmethod
    def _low_rank_head_parameter_count(rule_dim: int, rank: int) -> int:
        # affine LayerNorm + D-by-R Linear + R-by-64 Linear, both with biases.
        return (
            2 * rule_dim
            + rule_dim * rank
            + rank
            + rank * RULE_CLASS_COUNT
            + RULE_CLASS_COUNT
        )

    @classmethod
    def _closest_parameter_rank(cls, rule_dim: int) -> int:
        target = cls._factorized_head_parameter_count(rule_dim)
        return min(
            range(1, rule_dim + 1),
            key=lambda rank: abs(
                cls._low_rank_head_parameter_count(rule_dim, rank) - target
            ),
        )

    @property
    def head_kind(self) -> str:
        return self._head_kind

    @property
    def head_rank(self) -> int | None:
        return getattr(self.rule_head, "rank", None)

    def infer_support(
        self,
        batch: RuleGridTensorBatch,
        *,
        temperature: float | None = None,
    ) -> UnstructuredCausalInference:
        """Infer opaque rule classes using public support transitions only."""

        batch.validate(self.config)
        tokens = self._transition_tokens(batch)
        slots = self.initial_slots[None].expand(batch.batch_size, -1, -1)
        for layer in self.cross_layers:
            slots = layer(slots, tokens, batch.support_mask)
        return self.inference_from_rule_logits(
            self.rule_head(slots),
            temperature=temperature,
        )

    def inference_from_rule_logits(
        self,
        logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> UnstructuredCausalInference:
        """Turn arbitrary direct 64-way logits into four executable rules."""

        current_temperature = self.temperature if temperature is None else temperature
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")
        if logits.ndim != 3 or logits.shape[1:] != (
            self.config.particles,
            RULE_CLASS_COUNT,
        ):
            raise ValueError("logits must have [B,K,64] shape")
        log_probabilities = F.log_softmax(
            logits / current_temperature,
            dim=-1,
        )
        probabilities = log_probabilities.exp()
        rule_ids = probabilities.argmax(dim=-1)
        hard_codes = F.one_hot(
            rule_ids,
            RULE_CLASS_COUNT,
        ).to(dtype=probabilities.dtype)
        factor_ids = self.factor_bank[rule_ids]
        rule_latents = hard_codes @ self.factor_bank_rule_latents
        return UnstructuredCausalInference(
            rule_logits=logits,
            rule_log_probabilities=log_probabilities,
            rule_probabilities=probabilities,
            rule_codes=hard_codes,
            rule_ids=rule_ids,
            factor_ids=factor_ids,
            rule_latents=rule_latents,
        )

    def joint_rule_log_probabilities(
        self,
        rule_logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        """Return direct 64-way log probabilities with shape ``[B,K,64]``."""

        current_temperature = self.temperature if temperature is None else temperature
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")
        if rule_logits.ndim != 3 or rule_logits.shape[1:] != (
            self.config.particles,
            RULE_CLASS_COUNT,
        ):
            raise ValueError("rule_logits must have [B,K,64] shape")
        return F.log_softmax(rule_logits / current_temperature, dim=-1)

    def joint_rule_probabilities(
        self,
        rule_logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        return self.joint_rule_log_probabilities(
            rule_logits,
            temperature=temperature,
        ).exp()

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
    ) -> UnstructuredExpectedDiscreteLoss:
        """Train direct q(rule) with detached exact-code costs and hard 4! matching."""

        if min(
            validity_weight,
            diversity_weight,
            sharpening_weight,
            proper_weight,
            balanced_weight,
            assignment_temperature,
        ) < 0:
            raise ValueError("loss weights and assignment temperature cannot be negative")
        if assignment_temperature != 0:
            raise ValueError(
                "unstructured control uses exact hard 4! assignment; "
                "assignment_temperature must be zero"
            )

        inference = self.infer_support(batch, temperature=temperature)
        probabilities = inference.rule_probabilities
        log_probabilities = inference.rule_log_probabilities
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
            probabilities,
            behavior_costs,
        )
        set_cost = self._soft_permutation_loss(expected_class_cost, 0.0)
        validity_cost = torch.einsum(
            "bkr,br->bk",
            probabilities,
            support_costs,
        ).mean()
        diversity_barrier = self._diversity_barrier(probabilities)
        joint_entropy = -(
            probabilities * log_probabilities
        ).sum(dim=-1).mean()
        mean_top_probability = probabilities.amax(dim=-1).mean()
        total = (
            set_cost
            + validity_weight * validity_cost
            + diversity_weight * diversity_barrier
            + sharpening_weight * joint_entropy
        )
        return UnstructuredExpectedDiscreteLoss(
            total=total,
            set_cost=set_cost,
            validity_cost=validity_cost,
            diversity_barrier=diversity_barrier,
            joint_entropy=joint_entropy,
            mean_top_probability=mean_top_probability,
            inference=inference,
            joint_probabilities=probabilities,
            behavior_costs=behavior_costs,
        )


# Backwards-friendly descriptive alias; the canonical experiment/model type is
# ``UnstructuredDiscreteCausalK4``.
UnstructuredExpectedDiscreteK4 = UnstructuredDiscreteCausalK4


__all__ = [
    "DirectRuleHead",
    "LowRankRuleHead",
    "RULE_CLASS_COUNT",
    "UnstructuredCausalInference",
    "UnstructuredDiscreteCausalK4",
    "UnstructuredExpectedDiscreteK4",
    "UnstructuredExpectedDiscreteLoss",
]
