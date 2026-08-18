"""Persistent stratified proposal adapter for a frozen GRAM rule model.

The legacy GRAM proposer is exchangeable across inference width: every
trajectory starts from the same state and differs only through iid Gaussian
noise.  This module keeps that checkpoint frozen and adds a small public-only
adapter with fixed slot identities.  The same identity is used at every
recursive step and evidence stage.

The adapter is deliberately specific to the privileged 3x4 RuleGrid factor
space.  It is a structural ceiling for systematic latent hypotheses, not a
claim that arbitrary ARC rules have already been discovered from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.stratified_gram requires PyTorch") from error

from .causal_rules import CausalMechanismInference
from .gram_causal_rules import GRAMFactorizedCausalK4, GRAMRuleTrajectories
from .latent_rules import RULE_FACTOR_CARDINALITY, RULE_FACTOR_COUNT
from .neural import RuleGridTensorBatch


ANCHOR_WIDTH = 32
CANONICAL_SLOTS = 4


def _gf4_multiply_by_two(value: int) -> int:
    """Multiply a two-bit GF(4) element by x modulo x^2+x+1."""

    table = (0, 2, 3, 1)
    if value not in range(4):
        raise ValueError("GF(4) values must lie in [0,3]")
    return table[value]


def nested_stratified_anchor_codes() -> tuple[tuple[int, int, int], ...]:
    """Return a nested 32-run strength-two orthogonal-array anchor bank.

    For each of two cosets, all 16 pairs ``(x,y)`` are enumerated and
    ``z = x + 2y + c`` is formed in GF(4).  Ordering by coset, offset, then x
    makes every 4/8/16/32 prefix axis-balanced.  The first 16 anchors contain
    every ordered value pair once on every pair of axes; all 32 contain each
    pair twice.
    """

    codes: list[tuple[int, int, int]] = []
    for coset in (0, 1):
        for offset in range(4):
            for x in range(4):
                y = x ^ offset
                z = x ^ _gf4_multiply_by_two(y) ^ coset
                codes.append((x, y, z))
    result = tuple(codes)
    if len(result) != ANCHOR_WIDTH or len(set(result)) != ANCHOR_WIDTH:
        raise AssertionError("stratified anchor bank must contain 32 unique codes")
    for width in (4, 8, 16, 32):
        prefix = result[:width]
        for axis in range(RULE_FACTOR_COUNT):
            counts = [sum(code[axis] == value for code in prefix) for value in range(4)]
            if counts != [width // 4] * 4:
                raise AssertionError("anchor prefixes must be axis-balanced")
    for width, expected in ((16, 1), (32, 2)):
        prefix = result[:width]
        for left in range(RULE_FACTOR_COUNT):
            for right in range(left + 1, RULE_FACTOR_COUNT):
                counts = {
                    (x, y): sum(
                        code[left] == x and code[right] == y for code in prefix
                    )
                    for x in range(4)
                    for y in range(4)
                }
                if set(counts.values()) != {expected}:
                    raise AssertionError("anchor bank must have strength two")
    return result


@dataclass(frozen=True)
class StratifiedPublicCoverageLoss:
    """Canonical public-version-space loss for persistent slot identities."""

    total: Tensor
    joint_nll: Tensor
    joint_margin: Tensor
    ambiguity_bce: Tensor
    invalid_mass: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    hard_version_space_recall: Tensor
    hard_all_four_rate: Tensor
    hard_valid_particle_rate: Tensor
    hard_mean_unique_codes: Tensor
    step_objectives: Tensor
    step_joint_nll: Tensor
    step_joint_margin: Tensor
    step_invalid_mass: Tensor
    step_hard_version_space_recall: Tensor
    step_hard_all_four_rate: Tensor
    step_hard_valid_particle_rate: Tensor
    step_hard_mean_unique_codes: Tensor
    deep_supervision_weights: Tensor
    trajectories: GRAMRuleTrajectories
    compatible_mask: Tensor
    compatible_indices: Tensor
    canonical_target_indices: Tensor
    varying_axes: Tensor
    ambiguity_probabilities: Tensor

    @property
    def inference(self) -> CausalMechanismInference:
        return self.trajectories.final_inference

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_joint_nll": float(self.joint_nll.detach().cpu()),
            "loss_joint_margin": float(self.joint_margin.detach().cpu()),
            "loss_ambiguity_bce": float(self.ambiguity_bce.detach().cpu()),
            "invalid_probability_mass": float(self.invalid_mass.detach().cpu()),
            "joint_entropy_nats": float(self.joint_entropy.detach().cpu()),
            "mean_top_rule_probability": float(
                self.mean_top_probability.detach().cpu()
            ),
            "hard_version_space_recall": float(
                self.hard_version_space_recall.detach().cpu()
            ),
            "hard_all_four_rate": float(self.hard_all_four_rate.detach().cpu()),
            "hard_valid_particle_rate": float(
                self.hard_valid_particle_rate.detach().cpu()
            ),
            "hard_mean_unique_codes": float(
                self.hard_mean_unique_codes.detach().cpu()
            ),
        }


class PersistentStratifiedGRAMProposal(nn.Module):
    """Small structural proposal adapter around a fully frozen legacy GRAM."""

    def __init__(
        self,
        legacy: GRAMFactorizedCausalK4,
        *,
        initial_anchor_gain: float = 4.0,
        legacy_logit_mode: str = "residual",
    ) -> None:
        super().__init__()
        if initial_anchor_gain <= 0:
            raise ValueError("initial_anchor_gain must be positive")
        if legacy_logit_mode not in {"residual", "replace"}:
            raise ValueError("legacy_logit_mode must be 'residual' or 'replace'")
        self.legacy_logit_mode = legacy_logit_mode
        self.legacy = legacy
        for parameter in self.legacy.parameters():
            parameter.requires_grad_(False)
        self.legacy.eval()

        anchors = torch.tensor(
            nested_stratified_anchor_codes(),
            dtype=torch.long,
        )
        self.register_buffer("anchor_bank", anchors, persistent=True)
        dimension = self.legacy.config.rule_dim
        # A single affine public-support adapter keeps the new capacity small:
        # LayerNorm(2D) + Linear(2D, 12 correction + 3 ambiguity logits).
        self.support_adapter = nn.Sequential(
            nn.LayerNorm(2 * dimension),
            nn.Linear(
                2 * dimension,
                RULE_FACTOR_COUNT * RULE_FACTOR_CARDINALITY + RULE_FACTOR_COUNT,
            ),
        )
        nn.init.zeros_(self.support_adapter[-1].weight)
        nn.init.zeros_(self.support_adapter[-1].bias)
        inverse_softplus = math.log(math.expm1(initial_anchor_gain))
        self.anchor_log_gain = nn.Parameter(
            torch.full((RULE_FACTOR_COUNT,), inverse_softplus)
        )

    @property
    def config(self):
        return self.legacy.config

    @property
    def recursive_steps(self) -> int:
        return self.legacy.recursive_steps

    @property
    def factor_bank(self) -> Tensor:
        return self.legacy.factor_bank

    @property
    def executor(self):
        return self.legacy.executor

    @property
    def anchor_gain(self) -> Tensor:
        return F.softplus(self.anchor_log_gain)

    def train(self, mode: bool = True) -> "PersistentStratifiedGRAMProposal":
        super().train(mode)
        self.legacy.eval()
        return self

    def adapter_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if not name.startswith("legacy.") and parameter.requires_grad
        ]

    def discrete_support_costs(self, *args, **kwargs):
        return self.legacy.discrete_support_costs(*args, **kwargs)

    def public_support_exact_mask(self, batch: RuleGridTensorBatch) -> Tensor:
        return self.legacy.public_support_exact_mask(batch)

    def _public_context(self, batch: RuleGridTensorBatch) -> Tensor:
        support = self.legacy._support_only_batch(batch)
        support.validate(self.config)
        tokens = self.legacy._transition_tokens(support)
        pooled = self.legacy._masked_mean(tokens, support.support_mask)
        encoded = self.legacy.support_context(pooled)
        return torch.cat((pooled, encoded), dim=-1)

    def adapter_outputs(
        self,
        batch: RuleGridTensorBatch,
    ) -> tuple[Tensor, Tensor]:
        output = self.support_adapter(self._public_context(batch))
        correction_size = RULE_FACTOR_COUNT * RULE_FACTOR_CARDINALITY
        correction = output[:, :correction_size].reshape(
            batch.batch_size,
            RULE_FACTOR_COUNT,
            RULE_FACTOR_CARDINALITY,
        )
        ambiguity = output[:, correction_size:].sigmoid()
        return correction, ambiguity

    def slot_anchors(self, width: int) -> Tensor:
        if width <= 0 or width > int(self.anchor_bank.shape[0]):
            raise ValueError(f"width must lie in [1,{self.anchor_bank.shape[0]}]")
        return self.anchor_bank[:width]

    def _adapt_trajectories(
        self,
        batch: RuleGridTensorBatch,
        trajectories: GRAMRuleTrajectories,
    ) -> tuple[GRAMRuleTrajectories, Tensor]:
        correction, ambiguity = self.adapter_outputs(batch)
        anchors = self.slot_anchors(trajectories.width).to(
            device=trajectories.factor_logits.device
        )
        anchor_one_hot = F.one_hot(
            anchors,
            RULE_FACTOR_CARDINALITY,
        ).to(dtype=trajectories.factor_logits.dtype)
        centered = anchor_one_hot - 1.0 / RULE_FACTOR_CARDINALITY
        anchor_delta = (
            ambiguity[:, None, :, None]
            * self.anchor_gain[None, None, :, None]
            * centered[None]
        )
        legacy_logits = (
            trajectories.factor_logits
            if self.legacy_logit_mode == "residual"
            else torch.zeros_like(trajectories.factor_logits)
        )
        logits = legacy_logits + correction[None, :, None] + anchor_delta[None]
        probabilities = F.softmax(logits / self.legacy.temperature, dim=-1)
        factor_ids = probabilities.argmax(dim=-1)
        hard = F.one_hot(
            factor_ids,
            RULE_FACTOR_CARDINALITY,
        ).to(dtype=probabilities.dtype)
        factor_codes = hard + probabilities - probabilities.detach()
        rule_latents = torch.stack(
            [self.legacy.rule_latents_from_codes(step) for step in factor_codes]
        )
        return (
            replace(
                trajectories,
                factor_logits=logits,
                factor_probabilities=probabilities,
                factor_codes=factor_codes,
                factor_ids=factor_ids,
                rule_latents=rule_latents,
            ),
            ambiguity,
        )

    def sample_trajectories(
        self,
        batch: RuleGridTensorBatch,
        *,
        width: int | None = None,
        recursive_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        sample_noise: bool = False,
    ) -> GRAMRuleTrajectories:
        if temperature is not None and temperature != self.legacy.temperature:
            # The adapter transforms logits before the same categorical
            # temperature is applied.  Keep one explicit temperature source.
            original = self.legacy.temperature
            self.legacy.temperature = float(temperature)
            try:
                trajectories = self.legacy.sample_trajectories(
                    batch,
                    width=width,
                    recursive_steps=recursive_steps,
                    generator=generator,
                    seed=seed,
                    temperature=temperature,
                    sample_noise=sample_noise,
                )
                adapted, _ = self._adapt_trajectories(batch, trajectories)
            finally:
                self.legacy.temperature = original
            return adapted
        trajectories = self.legacy.sample_trajectories(
            batch,
            width=width,
            recursive_steps=recursive_steps,
            generator=generator,
            seed=seed,
            temperature=temperature,
            sample_noise=sample_noise,
        )
        adapted, _ = self._adapt_trajectories(batch, trajectories)
        return adapted

    def sample_width_candidates(self, batch: RuleGridTensorBatch, **kwargs):
        return self.sample_trajectories(batch, **kwargs).final_inference

    def infer_support(self, batch: RuleGridTensorBatch, **kwargs):
        kwargs.setdefault("width", self.config.particles)
        return self.sample_width_candidates(batch, **kwargs)

    def hard_public_version_space_loss(
        self,
        batch: RuleGridTensorBatch,
        **kwargs,
    ) -> StratifiedPublicCoverageLoss:
        return hard_public_version_space_loss(self, batch, **kwargs)


def _canonical_public_targets(
    model: PersistentStratifiedGRAMProposal,
    compatible_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map each persistent W4 anchor to one code in the unordered public set."""

    if compatible_mask.ndim != 2 or compatible_mask.shape[1] != 64:
        raise ValueError("compatible_mask must have shape [B,64]")
    if not torch.all(compatible_mask.sum(dim=-1) == CANONICAL_SLOTS):
        raise ValueError("every public version space must contain exactly four codes")
    bank_indices = torch.arange(64, device=compatible_mask.device)[None].expand(
        compatible_mask.shape[0], -1
    )
    compatible_indices = bank_indices[compatible_mask].reshape(-1, CANONICAL_SLOTS)
    compatible_codes = model.factor_bank[compatible_indices]
    targets: list[Tensor] = []
    varying_axes: list[int] = []
    anchors = model.slot_anchors(CANONICAL_SLOTS).to(compatible_codes.device)
    for task_codes in compatible_codes:
        varying = [
            axis
            for axis in range(RULE_FACTOR_COUNT)
            if torch.unique(task_codes[:, axis]).numel() == RULE_FACTOR_CARDINALITY
        ]
        if len(varying) != 1:
            raise ValueError(
                "canonical K4 supervision requires exactly one four-valued axis"
            )
        axis = varying[0]
        if any(
            torch.unique(task_codes[:, fixed]).numel() != 1
            for fixed in range(RULE_FACTOR_COUNT)
            if fixed != axis
        ):
            raise ValueError("non-varying public factors must be constant")
        selected: list[Tensor] = []
        for value in anchors[:, axis]:
            matches = torch.nonzero(
                task_codes[:, axis] == value,
                as_tuple=False,
            ).flatten()
            if matches.numel() != 1:
                raise AssertionError("each anchor value must select one public code")
            selected.append(compatible_indices[len(targets), matches[0]])
        targets.append(torch.stack(selected))
        varying_axes.append(axis)
    return (
        torch.stack(targets).detach(),
        torch.tensor(varying_axes, device=compatible_mask.device).detach(),
        compatible_indices.detach(),
    )


def _joint_log_probabilities(logits: Tensor, factor_bank: Tensor) -> Tensor:
    axis_log_probabilities = F.log_softmax(logits, dim=-1)
    selected = [
        axis_log_probabilities[:, :, axis, factor_bank[:, axis]]
        for axis in range(RULE_FACTOR_COUNT)
    ]
    return torch.stack(selected).sum(dim=0)


def _joint_scores(logits: Tensor, factor_bank: Tensor) -> Tensor:
    selected = [
        logits[:, :, axis, factor_bank[:, axis]]
        for axis in range(RULE_FACTOR_COUNT)
    ]
    return torch.stack(selected).sum(dim=0)


def _hard_metrics(
    predictions: Tensor,
    compatible_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    recalls: list[float] = []
    all_four: list[float] = []
    unique_counts: list[float] = []
    for task_index, row in enumerate(predictions.detach()):
        predicted = {int(value) for value in row.cpu().tolist()}
        compatible = {
            int(value)
            for value in torch.nonzero(
                compatible_mask[task_index], as_tuple=False
            ).flatten().cpu().tolist()
        }
        covered = len(predicted.intersection(compatible))
        recalls.append(covered / CANONICAL_SLOTS)
        all_four.append(float(covered == CANONICAL_SLOTS))
        unique_counts.append(float(len(predicted)))
    reference = predictions.new_tensor(recalls, dtype=torch.float32)
    valid = compatible_mask.gather(1, predictions).to(dtype=torch.float32).mean()
    return (
        reference.mean(),
        predictions.new_tensor(all_four, dtype=torch.float32).mean(),
        valid,
        predictions.new_tensor(unique_counts, dtype=torch.float32).mean(),
    )


def hard_public_version_space_loss(
    model: PersistentStratifiedGRAMProposal,
    batch: RuleGridTensorBatch,
    *,
    margin_weight: float = 0.10,
    ambiguity_weight: float = 0.10,
    validity_weight: float = 0.0,
    joint_margin: float = 1.0,
    deep_supervision_decay: float = 1.0,
    temperature: float | None = None,
    sample_noise: bool = False,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> StratifiedPublicCoverageLoss:
    """Train W4 persistent slots against the whole public-compatible set.

    No selected program, query target, or behavior panel is read.  The target
    set is the detached all-64 frozen-executor MAP equality mask.  Canonical
    assignment uses only the unique varying axis of that complete set.
    """

    if min(
        margin_weight,
        ambiguity_weight,
        validity_weight,
        joint_margin,
        deep_supervision_decay,
    ) < 0:
        raise ValueError("loss weights, margin, and decay must be non-negative")
    if deep_supervision_decay == 0:
        raise ValueError("deep_supervision_decay must be positive")
    support = model.legacy._support_only_batch(batch)
    compatible_mask = model.public_support_exact_mask(support)
    canonical_targets, varying_axes, compatible_indices = _canonical_public_targets(
        model,
        compatible_mask,
    )
    trajectories = model.sample_trajectories(
        support,
        width=CANONICAL_SLOTS,
        recursive_steps=model.recursive_steps,
        generator=generator,
        seed=seed,
        temperature=temperature,
        sample_noise=sample_noise,
    )
    _, ambiguity_probabilities = model.adapter_outputs(support)
    ambiguity_targets = F.one_hot(
        varying_axes,
        RULE_FACTOR_COUNT,
    ).to(dtype=ambiguity_probabilities.dtype)
    ambiguity_bce = F.binary_cross_entropy(
        ambiguity_probabilities,
        ambiguity_targets,
    )

    step_nll: list[Tensor] = []
    step_margin: list[Tensor] = []
    step_invalid: list[Tensor] = []
    step_entropy: list[Tensor] = []
    step_top: list[Tensor] = []
    step_recall: list[Tensor] = []
    step_all_four: list[Tensor] = []
    step_valid: list[Tensor] = []
    step_unique: list[Tensor] = []
    factor_bank = model.factor_bank
    for logits in trajectories.factor_logits:
        joint_log_probabilities = _joint_log_probabilities(logits, factor_bank)
        joint_probabilities = joint_log_probabilities.exp()
        target_log_probability = joint_log_probabilities.gather(
            dim=-1,
            index=canonical_targets[:, :, None],
        ).squeeze(-1)
        step_nll.append(-target_log_probability.mean())

        scores = _joint_scores(logits, factor_bank)
        target_scores = scores.gather(
            dim=-1,
            index=canonical_targets[:, :, None],
        ).squeeze(-1)
        target_mask = F.one_hot(canonical_targets, 64).to(dtype=torch.bool)
        competing_scores = scores.masked_fill(target_mask, -torch.inf).amax(dim=-1)
        step_margin.append(
            F.relu(joint_margin + competing_scores - target_scores).mean()
        )
        compatible_mass = torch.einsum(
            "bkr,br->bk",
            joint_probabilities,
            compatible_mask.to(dtype=joint_probabilities.dtype),
        )
        step_invalid.append((1.0 - compatible_mass).mean())
        step_entropy.append(
            -(joint_probabilities * joint_log_probabilities).sum(dim=-1).mean()
        )
        step_top.append(joint_probabilities.amax(dim=-1).mean())
        predictions = scores.argmax(dim=-1)
        recall, all_four, valid, unique = _hard_metrics(
            predictions,
            compatible_mask,
        )
        step_recall.append(recall)
        step_all_four.append(all_four)
        step_valid.append(valid)
        step_unique.append(unique)

    nll = torch.stack(step_nll)
    margin = torch.stack(step_margin)
    invalid = torch.stack(step_invalid)
    entropy = torch.stack(step_entropy)
    top = torch.stack(step_top)
    hard_recall = torch.stack(step_recall)
    hard_all_four = torch.stack(step_all_four)
    hard_valid = torch.stack(step_valid)
    hard_unique = torch.stack(step_unique)
    objectives = (
        nll
        + margin_weight * margin
        + validity_weight * invalid
        + ambiguity_weight * ambiguity_bce
    )
    weights = model.legacy._deep_supervision_weights(
        model.recursive_steps,
        deep_supervision_decay,
        objectives,
    )
    return StratifiedPublicCoverageLoss(
        total=(weights * objectives).sum(),
        joint_nll=(weights * nll).sum(),
        joint_margin=(weights * margin).sum(),
        ambiguity_bce=ambiguity_bce,
        invalid_mass=(weights * invalid).sum(),
        joint_entropy=(weights * entropy).sum(),
        mean_top_probability=(weights * top).sum(),
        hard_version_space_recall=hard_recall[-1],
        hard_all_four_rate=hard_all_four[-1],
        hard_valid_particle_rate=hard_valid[-1],
        hard_mean_unique_codes=hard_unique[-1],
        step_objectives=objectives,
        step_joint_nll=nll,
        step_joint_margin=margin,
        step_invalid_mass=invalid,
        step_hard_version_space_recall=hard_recall,
        step_hard_all_four_rate=hard_all_four,
        step_hard_valid_particle_rate=hard_valid,
        step_hard_mean_unique_codes=hard_unique,
        deep_supervision_weights=weights,
        trajectories=trajectories,
        compatible_mask=compatible_mask.detach(),
        compatible_indices=compatible_indices,
        canonical_target_indices=canonical_targets,
        varying_axes=varying_axes,
        ambiguity_probabilities=ambiguity_probabilities,
    )


__all__ = [
    "ANCHOR_WIDTH",
    "CANONICAL_SLOTS",
    "PersistentStratifiedGRAMProposal",
    "StratifiedPublicCoverageLoss",
    "hard_public_version_space_loss",
    "nested_stratified_anchor_codes",
]
