"""Public-only persistent K4 mechanism-set abstraction.

This model keeps the privileged frozen RuleGrid executor and known 3x4 factor
space, but learns its support encoder and four persistent slots from scratch.
The complete public-support version space is the only training teacher.  No
selected program, diagnostic query target, or behavior panel is consumed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.public_version_k4 requires PyTorch") from error

from .causal_rules import CausalMechanismInference
from .discrete_causal_rules import ExpectedDiscreteCausalK4
from .latent_rules import RULE_FACTOR_CARDINALITY, RULE_FACTOR_COUNT, outcome_map
from .neural import ACTION_FIELDS, RuleGridTensorBatch


@dataclass(frozen=True)
class PublicVersionSpaceK4Loss:
    """Canonical hard-joint objective for the four public-compatible rules."""

    total: Tensor
    joint_nll: Tensor
    joint_margin: Tensor
    invalid_mass: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    hard_version_space_recall: Tensor
    hard_all_four_rate: Tensor
    hard_valid_particle_rate: Tensor
    hard_mean_unique_codes: Tensor
    inference: CausalMechanismInference
    joint_probabilities: Tensor
    compatible_mask: Tensor
    compatible_indices: Tensor
    canonical_target_indices: Tensor
    varying_axes: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_joint_nll": float(self.joint_nll.detach().cpu()),
            "loss_joint_margin": float(self.joint_margin.detach().cpu()),
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


@dataclass(frozen=True)
class FactorizedPublicVersionSpaceK4Loss:
    """Direct unknown-axis plus fixed-factor objective for a K4 version set."""

    total: Tensor
    fixed_factor_nll: Tensor
    varying_axis_nll: Tensor
    invalid_mass: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    hard_version_space_recall: Tensor
    hard_all_four_rate: Tensor
    hard_valid_particle_rate: Tensor
    hard_mean_unique_codes: Tensor
    inference: CausalMechanismInference
    compatible_mask: Tensor
    varying_axes: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_fixed_factor_nll": float(self.fixed_factor_nll.detach().cpu()),
            "loss_varying_axis_nll": float(self.varying_axis_nll.detach().cpu()),
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


@dataclass(frozen=True)
class TransitionEvidencePublicVersionSpaceK4Loss:
    """Transition routing and per-axis value supervision for causal evidence."""

    total: Tensor
    evidence_axis_nll: Tensor
    evidence_value_nll: Tensor
    task_varying_axis_nll: Tensor
    task_fixed_factor_nll: Tensor
    evidence_axis_accuracy: Tensor
    evidence_value_accuracy: Tensor
    invalid_mass: Tensor
    hard_version_space_recall: Tensor
    hard_all_four_rate: Tensor
    hard_valid_particle_rate: Tensor
    hard_mean_unique_codes: Tensor
    inference: CausalMechanismInference
    compatible_mask: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_evidence_axis_nll": float(self.evidence_axis_nll.detach().cpu()),
            "loss_evidence_value_nll": float(self.evidence_value_nll.detach().cpu()),
            "loss_task_varying_axis_nll": float(
                self.task_varying_axis_nll.detach().cpu()
            ),
            "loss_task_fixed_factor_nll": float(
                self.task_fixed_factor_nll.detach().cpu()
            ),
            "evidence_axis_accuracy": float(
                self.evidence_axis_accuracy.detach().cpu()
            ),
            "evidence_value_accuracy": float(
                self.evidence_value_accuracy.detach().cpu()
            ),
            "invalid_probability_mass": float(self.invalid_mass.detach().cpu()),
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


@dataclass(frozen=True)
class SymmetryAwareFactorBelief:
    """Compact per-axis belief before any Cartesian particle expansion."""

    factor_probabilities: Tensor  # [B,3,4]
    evidence_strength: Tensor  # [B,3]
    evidence_axis_logits: Tensor  # [B,T,4]
    evidence_value_logits: Tensor  # [B,T,3,4]


@dataclass(frozen=True)
class SymmetryAwareFactorBeliefLoss:
    total: Tensor
    evidence_axis_nll: Tensor
    evidence_value_set_nll: Tensor
    task_factor_set_nll: Tensor
    evidence_axis_accuracy: Tensor
    evidence_value_set_accuracy: Tensor
    target_probability_mass: Tensor
    factor_set_recall: Tensor
    factor_set_precision: Tensor
    exact_task_factor_set_rate: Tensor
    belief: SymmetryAwareFactorBelief

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_evidence_axis_nll": float(self.evidence_axis_nll.detach().cpu()),
            "loss_evidence_value_set_nll": float(
                self.evidence_value_set_nll.detach().cpu()
            ),
            "loss_task_factor_set_nll": float(
                self.task_factor_set_nll.detach().cpu()
            ),
            "evidence_axis_accuracy": float(
                self.evidence_axis_accuracy.detach().cpu()
            ),
            "evidence_value_set_accuracy": float(
                self.evidence_value_set_accuracy.detach().cpu()
            ),
            "target_probability_mass": float(
                self.target_probability_mass.detach().cpu()
            ),
            "factor_set_recall": float(self.factor_set_recall.detach().cpu()),
            "factor_set_precision": float(
                self.factor_set_precision.detach().cpu()
            ),
            "exact_task_factor_set_rate": float(
                self.exact_task_factor_set_rate.detach().cpu()
            ),
        }


class PublicVersionSpaceCausalK4(ExpectedDiscreteCausalK4):
    """Four learned support-attending slots trained on a public rule set."""

    recursive_steps = 1

    def __init__(
        self,
        executor,
        *,
        attention_layers: int = 2,
        temperature: float = 1.0,
        independent_support_encoders: bool = False,
    ) -> None:
        super().__init__(
            executor,
            attention_layers=attention_layers,
            temperature=temperature,
        )
        self.independent_support_encoders = bool(independent_support_encoders)
        if self.independent_support_encoders:
            self.support_grid_encoder = copy.deepcopy(self.executor.grid_encoder)
            self.support_action_encoder = copy.deepcopy(self.executor.action_encoder)
            for parameter in self.support_grid_encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in self.support_action_encoder.parameters():
                parameter.requires_grad_(True)
        else:
            self.support_grid_encoder = None
            self.support_action_encoder = None

    @staticmethod
    def _support_only_batch(batch: RuleGridTensorBatch) -> RuleGridTensorBatch:
        return RuleGridTensorBatch(
            support_states=batch.support_states,
            support_actions=batch.support_actions,
            support_targets=batch.support_targets,
            support_mask=batch.support_mask,
            support_action_mask=batch.support_action_mask,
        )

    def _transition_tokens(self, batch: RuleGridTensorBatch) -> Tensor:
        if not self.independent_support_encoders:
            return super()._transition_tokens(batch)
        assert self.support_grid_encoder is not None
        assert self.support_action_encoder is not None
        states = batch.support_states
        targets = batch.support_targets
        batch_size, steps, height, width = states.shape
        flat_states = states.reshape(batch_size * steps, height, width)
        flat_targets = targets.reshape(batch_size * steps, height, width)
        state_features = self.support_grid_encoder(flat_states)
        target_features = self.support_grid_encoder(flat_targets)
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
        action_features = self.support_action_encoder(
            flat_actions,
            flat_action_mask,
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
            batch_size,
            steps,
            self.config.rule_dim,
        )

    def public_support_exact_mask(self, batch: RuleGridTensorBatch) -> Tensor:
        """Return detached all-64 exact compatibility from public support."""

        support = self._support_only_batch(batch)
        support.validate(self.config)
        batch_size, steps, height, width = support.support_states.shape
        with torch.no_grad():
            prediction = self._predict_all_support_codes(support)
            maps = outcome_map(prediction).reshape(
                batch_size,
                steps,
                self.factor_bank.shape[0],
                height,
                width,
            )
            valid = support.support_mask[:, :, None, None, None]
            wrong = maps.ne(support.support_targets[:, :, None]) & valid
            exact = ~wrong.any(dim=(1, 3, 4))
        return exact.detach()

    def sample_width_candidates(
        self,
        batch: RuleGridTensorBatch,
        *,
        width: int = 4,
        recursive_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        sample_noise: bool = False,
    ) -> CausalMechanismInference:
        """Duck-typed deterministic W4 interface used by proposal audits."""

        if width != self.config.particles or width != 4:
            raise ValueError("public version-space K4 has exactly four slots")
        if recursive_steps not in (None, 1):
            raise ValueError("public version-space K4 has no recursive depth")
        if generator is not None or seed is not None or sample_noise:
            raise ValueError("public version-space K4 inference is deterministic")
        support = self._support_only_batch(batch)
        return self.infer_support(support, temperature=temperature)

    def hard_public_version_space_loss(
        self,
        batch: RuleGridTensorBatch,
        *,
        compatible_mask: Tensor | None = None,
        margin_weight: float = 0.10,
        validity_weight: float = 0.0,
        joint_margin: float = 1.0,
        temperature: float | None = None,
    ) -> PublicVersionSpaceK4Loss:
        if min(margin_weight, validity_weight, joint_margin) < 0:
            raise ValueError("loss weights and joint margin must be non-negative")
        support = self._support_only_batch(batch)
        support.validate(self.config)
        if compatible_mask is None:
            compatible_mask = self.public_support_exact_mask(support)
        else:
            if compatible_mask.shape != (support.batch_size, 64):
                raise ValueError("compatible_mask must have shape [B,64]")
            if compatible_mask.dtype != torch.bool:
                raise TypeError("compatible_mask must have torch.bool dtype")
            compatible_mask = compatible_mask.to(
                device=support.support_states.device
            ).detach()
        canonical_targets, varying_axes, compatible_indices = (
            self._canonical_public_targets(compatible_mask)
        )
        inference = self.infer_support(support, temperature=temperature)
        current_temperature = self.temperature if temperature is None else temperature
        joint_log_probabilities = self.joint_rule_log_probabilities(
            inference.factor_logits,
            temperature=current_temperature,
        )
        joint_probabilities = joint_log_probabilities.exp()
        target_log_probabilities = joint_log_probabilities.gather(
            dim=-1,
            index=canonical_targets[:, :, None],
        ).squeeze(-1)
        joint_nll = -target_log_probabilities.mean()

        scores = self._joint_scores(inference.factor_logits)
        target_scores = scores.gather(
            dim=-1,
            index=canonical_targets[:, :, None],
        ).squeeze(-1)
        target_mask = F.one_hot(canonical_targets, 64).to(dtype=torch.bool)
        competing = scores.masked_fill(target_mask, -torch.inf).amax(dim=-1)
        margin = F.relu(joint_margin + competing - target_scores).mean()
        compatible_mass = torch.einsum(
            "bkr,br->bk",
            joint_probabilities,
            compatible_mask.to(dtype=joint_probabilities.dtype),
        )
        invalid_mass = (1.0 - compatible_mass).mean()
        entropy = -(
            joint_probabilities * joint_log_probabilities
        ).sum(dim=-1).mean()
        top_probability = joint_probabilities.amax(dim=-1).mean()
        predictions = scores.argmax(dim=-1)
        recall, all_four, valid, unique = self._hard_metrics(
            predictions,
            compatible_mask,
        )
        return PublicVersionSpaceK4Loss(
            total=(
                joint_nll
                + margin_weight * margin
                + validity_weight * invalid_mass
            ),
            joint_nll=joint_nll,
            joint_margin=margin,
            invalid_mass=invalid_mass,
            joint_entropy=entropy,
            mean_top_probability=top_probability,
            hard_version_space_recall=recall,
            hard_all_four_rate=all_four,
            hard_valid_particle_rate=valid,
            hard_mean_unique_codes=unique,
            inference=inference,
            joint_probabilities=joint_probabilities,
            compatible_mask=compatible_mask,
            compatible_indices=compatible_indices,
            canonical_target_indices=canonical_targets,
            varying_axes=varying_axes,
        )

    def _canonical_public_targets(
        self,
        compatible_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if compatible_mask.ndim != 2 or compatible_mask.shape[1] != 64:
            raise ValueError("compatible_mask must have shape [B,64]")
        if not torch.all(compatible_mask.sum(dim=-1) == 4):
            raise ValueError("every public version space must contain exactly four codes")
        bank_indices = torch.arange(
            64,
            device=compatible_mask.device,
        )[None].expand(compatible_mask.shape[0], -1)
        compatible_indices = bank_indices[compatible_mask].reshape(-1, 4)
        compatible_codes = self.factor_bank[compatible_indices]
        targets: list[Tensor] = []
        varying_axes: list[int] = []
        for task_index, task_codes in enumerate(compatible_codes):
            varying = [
                axis
                for axis in range(RULE_FACTOR_COUNT)
                if torch.unique(task_codes[:, axis]).numel()
                == RULE_FACTOR_CARDINALITY
            ]
            if len(varying) != 1:
                raise ValueError(
                    "canonical K4 supervision requires one four-valued axis"
                )
            axis = varying[0]
            if any(
                torch.unique(task_codes[:, fixed]).numel() != 1
                for fixed in range(RULE_FACTOR_COUNT)
                if fixed != axis
            ):
                raise ValueError("non-varying public factors must be constant")
            selected = []
            for slot_value in range(4):
                matches = torch.nonzero(
                    task_codes[:, axis] == slot_value,
                    as_tuple=False,
                ).flatten()
                if matches.numel() != 1:
                    raise AssertionError("each slot value must select one code")
                selected.append(compatible_indices[task_index, matches[0]])
            targets.append(torch.stack(selected))
            varying_axes.append(axis)
        return (
            torch.stack(targets).detach(),
            torch.tensor(varying_axes, device=compatible_mask.device).detach(),
            compatible_indices.detach(),
        )

    def _joint_scores(self, logits: Tensor) -> Tensor:
        selected = [
            logits[:, :, axis, self.factor_bank[:, axis]]
            for axis in range(RULE_FACTOR_COUNT)
        ]
        return torch.stack(selected).sum(dim=0)

    @staticmethod
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
            recalls.append(covered / 4)
            all_four.append(float(covered == 4))
            unique_counts.append(float(len(predicted)))
        valid = compatible_mask.gather(1, predictions).to(dtype=torch.float32).mean()
        return (
            predictions.new_tensor(recalls, dtype=torch.float32).mean(),
            predictions.new_tensor(all_four, dtype=torch.float32).mean(),
            valid,
            predictions.new_tensor(unique_counts, dtype=torch.float32).mean(),
        )


class FactorizedPublicVersionSpaceCausalK4(PublicVersionSpaceCausalK4):
    """Predict one unknown axis and compose its four compatible latent rules."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for module in (self.cross_layers, self.factor_heads):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.initial_slots.requires_grad_(False)
        self.set_summary_norm = torch.nn.LayerNorm(self.config.rule_dim)
        self.varying_axis_head = torch.nn.Linear(
            self.config.rule_dim,
            RULE_FACTOR_COUNT,
        )
        self.fixed_factor_heads = torch.nn.ModuleList(
            torch.nn.Linear(self.config.rule_dim, RULE_FACTOR_CARDINALITY)
            for _ in range(RULE_FACTOR_COUNT)
        )

    def _factorized_support_outputs(
        self,
        batch: RuleGridTensorBatch,
    ) -> tuple[Tensor, Tensor]:
        support = self._support_only_batch(batch)
        support.validate(self.config)
        tokens = self._transition_tokens(support)
        weights = support.support_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        summary = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        summary = self.set_summary_norm(summary)
        varying_axis_logits = self.varying_axis_head(summary)
        fixed_factor_logits = torch.stack(
            [head(summary) for head in self.fixed_factor_heads],
            dim=1,
        )
        return fixed_factor_logits, varying_axis_logits

    def _compose_factorized_inference(
        self,
        fixed_factor_logits: Tensor,
        varying_axis_logits: Tensor,
        *,
        temperature: float | None = None,
    ) -> CausalMechanismInference:
        batch_size = fixed_factor_logits.shape[0]
        varying_axes = varying_axis_logits.argmax(dim=-1)
        base = fixed_factor_logits[:, None].expand(
            -1,
            self.config.particles,
            -1,
            -1,
        )
        slot_values = torch.arange(
            RULE_FACTOR_CARDINALITY,
            device=fixed_factor_logits.device,
        )
        enumeration = F.one_hot(
            slot_values,
            RULE_FACTOR_CARDINALITY,
        ).to(dtype=fixed_factor_logits.dtype)
        enumeration = (40.0 * enumeration - 20.0)[None, :, None, :].expand(
            batch_size,
            -1,
            RULE_FACTOR_COUNT,
            -1,
        )
        varying_mask = F.one_hot(
            varying_axes,
            RULE_FACTOR_COUNT,
        ).to(dtype=torch.bool)[:, None, :, None]
        logits = torch.where(varying_mask, enumeration, base)
        return self.inference_from_factor_logits(logits, temperature=temperature)

    def infer_support(
        self,
        batch: RuleGridTensorBatch,
        *,
        temperature: float | None = None,
    ) -> CausalMechanismInference:
        fixed_logits, varying_logits = self._factorized_support_outputs(batch)
        return self._compose_factorized_inference(
            fixed_logits,
            varying_logits,
            temperature=temperature,
        )

    def hard_public_version_space_loss(
        self,
        batch: RuleGridTensorBatch,
        *,
        compatible_mask: Tensor | None = None,
        varying_axis_weight: float = 1.0,
        temperature: float | None = None,
        **unused: float,
    ) -> FactorizedPublicVersionSpaceK4Loss:
        if varying_axis_weight < 0:
            raise ValueError("varying_axis_weight must be non-negative")
        support = self._support_only_batch(batch)
        support.validate(self.config)
        if compatible_mask is None:
            compatible_mask = self.public_support_exact_mask(support)
        else:
            if compatible_mask.shape != (support.batch_size, 64):
                raise ValueError("compatible_mask must have shape [B,64]")
            if compatible_mask.dtype != torch.bool:
                raise TypeError("compatible_mask must have torch.bool dtype")
            compatible_mask = compatible_mask.to(
                support.support_states.device
            ).detach()
        _, varying_axes, compatible_indices = self._canonical_public_targets(
            compatible_mask
        )
        fixed_logits, varying_logits = self._factorized_support_outputs(support)
        inference = self._compose_factorized_inference(
            fixed_logits,
            varying_logits,
            temperature=temperature,
        )
        representative = self.factor_bank[compatible_indices[:, 0]]
        fixed_log_probabilities = F.log_softmax(fixed_logits, dim=-1)
        selected = fixed_log_probabilities.gather(
            -1,
            representative[:, :, None],
        ).squeeze(-1)
        fixed_axis_mask = ~F.one_hot(
            varying_axes,
            RULE_FACTOR_COUNT,
        ).to(dtype=torch.bool)
        fixed_factor_nll = -selected[fixed_axis_mask].mean()
        varying_axis_nll = F.cross_entropy(varying_logits, varying_axes)
        joint_log_probabilities = self.joint_rule_log_probabilities(
            inference.factor_logits,
            temperature=self.temperature if temperature is None else temperature,
        )
        joint_probabilities = joint_log_probabilities.exp()
        compatible_mass = torch.einsum(
            "bkr,br->bk",
            joint_probabilities,
            compatible_mask.to(dtype=joint_probabilities.dtype),
        )
        invalid_mass = (1.0 - compatible_mass).mean()
        joint_entropy = -(
            joint_probabilities * joint_log_probabilities
        ).sum(dim=-1).mean()
        top_probability = joint_probabilities.amax(dim=-1).mean()
        predictions = self._joint_scores(inference.factor_logits).argmax(dim=-1)
        recall, all_four, valid, unique = self._hard_metrics(
            predictions,
            compatible_mask,
        )
        return FactorizedPublicVersionSpaceK4Loss(
            total=fixed_factor_nll + varying_axis_weight * varying_axis_nll,
            fixed_factor_nll=fixed_factor_nll,
            varying_axis_nll=varying_axis_nll,
            invalid_mass=invalid_mass,
            joint_entropy=joint_entropy,
            mean_top_probability=top_probability,
            hard_version_space_recall=recall,
            hard_all_four_rate=all_four,
            hard_valid_particle_rate=valid,
            hard_mean_unique_codes=unique,
            inference=inference,
            compatible_mask=compatible_mask,
            varying_axes=varying_axes,
        )


class TransitionEvidencePublicVersionSpaceCausalK4(
    FactorizedPublicVersionSpaceCausalK4
):
    """Route each public transition to a causal axis before belief composition."""

    neutral_evidence_class = RULE_FACTOR_COUNT

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for module in (
            self.set_summary_norm,
            self.varying_axis_head,
            self.fixed_factor_heads,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.evidence_axis_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.rule_dim),
            torch.nn.Linear(self.config.rule_dim, RULE_FACTOR_COUNT + 1),
        )
        self.evidence_value_heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.LayerNorm(self.config.rule_dim),
                torch.nn.Linear(self.config.rule_dim, RULE_FACTOR_CARDINALITY),
            )
            for _ in range(RULE_FACTOR_COUNT)
        )

    def _transition_evidence_outputs(
        self,
        batch: RuleGridTensorBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        support = self._support_only_batch(batch)
        support.validate(self.config)
        tokens = self._transition_tokens(support)
        axis_logits = self.evidence_axis_head(tokens)
        value_logits = torch.stack(
            [head(tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        axis_probabilities = F.softmax(axis_logits, dim=-1)[..., :RULE_FACTOR_COUNT]
        axis_probabilities = axis_probabilities * support.support_mask[..., None]
        evidence_strength = axis_probabilities.sum(dim=1)
        varying_axis_logits = -evidence_strength
        denominator = evidence_strength[:, :, None].clamp_min(1e-6)
        fixed_factor_logits = (
            axis_probabilities[:, :, :, None] * value_logits
        ).sum(dim=1) / denominator
        return (
            fixed_factor_logits,
            varying_axis_logits,
            axis_logits,
            value_logits,
        )

    def infer_support(
        self,
        batch: RuleGridTensorBatch,
        *,
        temperature: float | None = None,
    ) -> CausalMechanismInference:
        fixed_logits, varying_logits, _, _ = self._transition_evidence_outputs(
            batch
        )
        return self._compose_factorized_inference(
            fixed_logits,
            varying_logits,
            temperature=temperature,
        )

    def hard_public_version_space_loss(
        self,
        batch: RuleGridTensorBatch,
        *,
        compatible_mask: Tensor | None = None,
        evidence_axis_targets: Tensor | None = None,
        evidence_value_targets: Tensor | None = None,
        task_consistency_weight: float = 0.5,
        temperature: float | None = None,
        **unused: float,
    ) -> TransitionEvidencePublicVersionSpaceK4Loss:
        if task_consistency_weight < 0:
            raise ValueError("task_consistency_weight must be non-negative")
        support = self._support_only_batch(batch)
        support.validate(self.config)
        if evidence_axis_targets is None or evidence_value_targets is None:
            raise ValueError("transition-evidence targets are required during training")
        expected_shape = support.support_mask.shape
        if evidence_axis_targets.shape != expected_shape:
            raise ValueError("evidence_axis_targets must have shape [B,T]")
        if evidence_value_targets.shape != expected_shape:
            raise ValueError("evidence_value_targets must have shape [B,T]")
        evidence_axis_targets = evidence_axis_targets.to(
            support.support_states.device
        ).detach()
        evidence_value_targets = evidence_value_targets.to(
            support.support_states.device
        ).detach()
        if compatible_mask is None:
            compatible_mask = self.public_support_exact_mask(support)
        else:
            compatible_mask = compatible_mask.to(
                support.support_states.device
            ).detach()
        _, varying_axes, compatible_indices = self._canonical_public_targets(
            compatible_mask
        )
        fixed_logits, varying_logits, axis_logits, value_logits = (
            self._transition_evidence_outputs(support)
        )
        inference = self._compose_factorized_inference(
            fixed_logits,
            varying_logits,
            temperature=temperature,
        )
        valid_steps = support.support_mask
        evidence_axis_nll = F.cross_entropy(
            axis_logits[valid_steps],
            evidence_axis_targets[valid_steps],
        )
        informative = valid_steps & (
            evidence_axis_targets < self.neutral_evidence_class
        )
        informative_indices = torch.nonzero(informative, as_tuple=False)
        selected_value_logits = value_logits[
            informative_indices[:, 0],
            informative_indices[:, 1],
            evidence_axis_targets[informative],
        ]
        evidence_value_nll = F.cross_entropy(
            selected_value_logits,
            evidence_value_targets[informative],
        )
        axis_accuracy = (
            axis_logits[valid_steps].argmax(dim=-1)
            == evidence_axis_targets[valid_steps]
        ).to(dtype=torch.float32).mean()
        value_accuracy = (
            selected_value_logits.argmax(dim=-1)
            == evidence_value_targets[informative]
        ).to(dtype=torch.float32).mean()
        task_varying_axis_nll = F.cross_entropy(varying_logits, varying_axes)
        representative = self.factor_bank[compatible_indices[:, 0]]
        task_log_probabilities = F.log_softmax(fixed_logits, dim=-1)
        selected = task_log_probabilities.gather(
            -1,
            representative[:, :, None],
        ).squeeze(-1)
        fixed_axis_mask = ~F.one_hot(
            varying_axes,
            RULE_FACTOR_COUNT,
        ).to(dtype=torch.bool)
        task_fixed_factor_nll = -selected[fixed_axis_mask].mean()
        joint_probabilities = self.joint_rule_probabilities(
            inference.factor_logits,
            temperature=self.temperature if temperature is None else temperature,
        )
        compatible_mass = torch.einsum(
            "bkr,br->bk",
            joint_probabilities,
            compatible_mask.to(dtype=joint_probabilities.dtype),
        )
        invalid_mass = (1.0 - compatible_mass).mean()
        predictions = self._joint_scores(inference.factor_logits).argmax(dim=-1)
        recall, all_four, valid, unique = self._hard_metrics(
            predictions,
            compatible_mask,
        )
        return TransitionEvidencePublicVersionSpaceK4Loss(
            total=(
                evidence_axis_nll
                + evidence_value_nll
                + task_consistency_weight
                * (task_varying_axis_nll + task_fixed_factor_nll)
            ),
            evidence_axis_nll=evidence_axis_nll,
            evidence_value_nll=evidence_value_nll,
            task_varying_axis_nll=task_varying_axis_nll,
            task_fixed_factor_nll=task_fixed_factor_nll,
            evidence_axis_accuracy=axis_accuracy,
            evidence_value_accuracy=value_accuracy,
            invalid_mass=invalid_mass,
            hard_version_space_recall=recall,
            hard_all_four_rate=all_four,
            hard_valid_particle_rate=valid,
            hard_mean_unique_codes=unique,
            inference=inference,
            compatible_mask=compatible_mask,
        )


class SymmetryAwareFactorBeliefCausalK4(
    TransitionEvidencePublicVersionSpaceCausalK4
):
    """Retain public observational symmetries as compact factor value sets."""

    factor_set_threshold = 0.20
    supports_agent_probe_result_context = False

    def infer_factor_belief(
        self,
        batch: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None = None,
    ) -> SymmetryAwareFactorBelief:
        support = self._support_only_batch(batch)
        support.validate(self.config)
        axis_logits, value_logits = self._factor_belief_evidence_outputs(
            support,
            is_agent_probe_result=is_agent_probe_result,
        )
        axis_probabilities = F.softmax(axis_logits, dim=-1)[..., :RULE_FACTOR_COUNT]
        axis_probabilities = axis_probabilities * support.support_mask[..., None]
        raw_evidence_strength = axis_probabilities.sum(dim=1)
        evidence_strength = raw_evidence_strength.clamp(0.0, 1.0)
        # Per-transition value logits act as log evidence.  Summation makes a
        # later active result refine the earlier set instead of overwriting it.
        accumulated_value_logits = (
            axis_probabilities[:, :, :, None] * value_logits
        ).sum(dim=1)
        evidence_belief = F.softmax(accumulated_value_logits, dim=-1)
        uniform = torch.full_like(
            evidence_belief,
            1.0 / RULE_FACTOR_CARDINALITY,
        )
        factor_probabilities = (
            evidence_strength[..., None] * evidence_belief
            + (1.0 - evidence_strength[..., None]) * uniform
        )
        return SymmetryAwareFactorBelief(
            factor_probabilities=factor_probabilities,
            evidence_strength=evidence_strength,
            evidence_axis_logits=axis_logits,
            evidence_value_logits=value_logits,
        )

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if is_agent_probe_result is not None:
            if is_agent_probe_result.shape != support.support_mask.shape:
                raise ValueError("is_agent_probe_result must have shape [B,T]")
            if is_agent_probe_result.dtype != torch.bool:
                raise ValueError("is_agent_probe_result must be boolean")
            if bool(is_agent_probe_result.any().item()):
                raise ValueError(
                    "this factor-belief head does not encode controller probe phase"
                )
        _, _, axis_logits, value_logits = self._transition_evidence_outputs(
            support
        )
        return axis_logits, value_logits

    @staticmethod
    def _factor_value_masks(
        factor_bank: Tensor,
        compatible_mask: Tensor,
    ) -> Tensor:
        selected = compatible_mask[:, :, None, None] & F.one_hot(
            factor_bank,
            RULE_FACTOR_CARDINALITY,
        ).to(dtype=torch.bool)[None]
        return selected.any(dim=1)

    @classmethod
    def _threshold_factor_sets(cls, probabilities: Tensor) -> Tensor:
        selected = probabilities >= cls.factor_set_threshold
        empty = ~selected.any(dim=-1)
        if empty.any():
            selected = selected | F.one_hot(
                probabilities.argmax(dim=-1),
                RULE_FACTOR_CARDINALITY,
            ).to(dtype=torch.bool)
        return selected

    def symmetry_aware_factor_belief_loss(
        self,
        batch: RuleGridTensorBatch,
        *,
        compatible_mask: Tensor,
        evidence_axis_targets: Tensor,
        evidence_value_target_mask: Tensor,
        task_factor_weight: float = 0.5,
        is_agent_probe_result: Tensor | None = None,
    ) -> SymmetryAwareFactorBeliefLoss:
        if task_factor_weight < 0:
            raise ValueError("task_factor_weight must be non-negative")
        support = self._support_only_batch(batch)
        support.validate(self.config)
        compatible_mask = compatible_mask.to(
            support.support_states.device
        ).detach()
        evidence_axis_targets = evidence_axis_targets.to(
            support.support_states.device
        ).detach()
        evidence_value_target_mask = evidence_value_target_mask.to(
            support.support_states.device
        ).detach()
        if compatible_mask.shape != (support.batch_size, 64):
            raise ValueError("compatible_mask must have shape [B,64]")
        if evidence_axis_targets.shape != support.support_mask.shape:
            raise ValueError("evidence_axis_targets must have shape [B,T]")
        if evidence_value_target_mask.shape != (
            support.batch_size,
            support.support_steps,
            RULE_FACTOR_COUNT,
            RULE_FACTOR_CARDINALITY,
        ):
            raise ValueError("evidence_value_target_mask has the wrong shape")
        belief = self.infer_factor_belief(
            support,
            is_agent_probe_result=is_agent_probe_result,
        )
        valid_steps = support.support_mask
        axis_logits = belief.evidence_axis_logits
        evidence_axis_nll = F.cross_entropy(
            axis_logits[valid_steps],
            evidence_axis_targets[valid_steps],
        )
        informative = valid_steps & (
            evidence_axis_targets < self.neutral_evidence_class
        )
        indices = torch.nonzero(informative, as_tuple=False)
        selected_logits = belief.evidence_value_logits[
            indices[:, 0],
            indices[:, 1],
            evidence_axis_targets[informative],
        ]
        selected_masks = evidence_value_target_mask[
            indices[:, 0],
            indices[:, 1],
            evidence_axis_targets[informative],
        ]
        soft_evidence_targets = selected_masks.to(dtype=selected_logits.dtype)
        soft_evidence_targets = soft_evidence_targets / soft_evidence_targets.sum(
            dim=-1,
            keepdim=True,
        )
        evidence_value_set_nll = -(
            soft_evidence_targets * F.log_softmax(selected_logits, dim=-1)
        ).sum(dim=-1).mean()
        target_factor_masks = self._factor_value_masks(
            self.factor_bank,
            compatible_mask,
        )
        soft_factor_targets = target_factor_masks.to(
            dtype=belief.factor_probabilities.dtype
        )
        soft_factor_targets = soft_factor_targets / soft_factor_targets.sum(
            dim=-1,
            keepdim=True,
        )
        task_factor_set_nll = -(
            soft_factor_targets
            * belief.factor_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        predicted_sets = self._threshold_factor_sets(
            belief.factor_probabilities
        )
        intersection = (predicted_sets & target_factor_masks).sum(dim=-1)
        target_count = target_factor_masks.sum(dim=-1)
        predicted_count = predicted_sets.sum(dim=-1)
        recall_by_factor = intersection / target_count
        precision_by_factor = intersection / predicted_count
        axis_accuracy = (
            axis_logits[valid_steps].argmax(dim=-1)
            == evidence_axis_targets[valid_steps]
        ).to(dtype=torch.float32).mean()
        value_set_accuracy = selected_masks.gather(
            1,
            selected_logits.argmax(dim=-1, keepdim=True),
        ).to(dtype=torch.float32).mean()
        target_mass = (
            belief.factor_probabilities
            * target_factor_masks.to(dtype=belief.factor_probabilities.dtype)
        ).sum(dim=-1).mean()
        exact_task = (predicted_sets == target_factor_masks).all(dim=(1, 2))
        return SymmetryAwareFactorBeliefLoss(
            total=(
                evidence_axis_nll
                + evidence_value_set_nll
                + task_factor_weight * task_factor_set_nll
            ),
            evidence_axis_nll=evidence_axis_nll,
            evidence_value_set_nll=evidence_value_set_nll,
            task_factor_set_nll=task_factor_set_nll,
            evidence_axis_accuracy=axis_accuracy,
            evidence_value_set_accuracy=value_set_accuracy,
            target_probability_mass=target_mass,
            factor_set_recall=recall_by_factor.mean(),
            factor_set_precision=precision_by_factor.mean(),
            exact_task_factor_set_rate=exact_task.to(dtype=torch.float32).mean(),
            belief=belief,
        )


class ProbeAwareSymmetryFactorBeliefCausalK4(
    SymmetryAwareFactorBeliefCausalK4
):
    """Condition evidence on whether feedback follows an agent-chosen probe."""

    supports_agent_probe_result_context = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.agent_probe_result_embedding = torch.nn.Parameter(
            torch.empty(self.config.rule_dim)
        )
        torch.nn.init.normal_(self.agent_probe_result_embedding, std=0.02)

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        mask = self._validated_agent_probe_mask(
            support,
            is_agent_probe_result,
        )
        tokens = self._transition_tokens(support)
        tokens = tokens + (
            mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
        )
        axis_logits = self.evidence_axis_head(tokens)
        value_logits = torch.stack(
            [head(tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        return axis_logits, value_logits

    @staticmethod
    def _validated_agent_probe_mask(
        support: RuleGridTensorBatch,
        is_agent_probe_result: Tensor | None,
    ) -> Tensor:
        if is_agent_probe_result is None:
            mask = torch.zeros_like(support.support_mask)
        else:
            if is_agent_probe_result.shape != support.support_mask.shape:
                raise ValueError("is_agent_probe_result must have shape [B,T]")
            if is_agent_probe_result.dtype != torch.bool:
                raise ValueError("is_agent_probe_result must be boolean")
            mask = is_agent_probe_result.to(
                device=support.support_mask.device,
            )
        if bool((mask & ~support.support_mask).any().item()):
            raise ValueError("is_agent_probe_result selected a padded support step")
        return mask


class HistoryConditionedProbeFactorBeliefCausalK4(
    ProbeAwareSymmetryFactorBeliefCausalK4
):
    """Decode probe feedback after attending to its preceding public history."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probe_history_attention = torch.nn.MultiheadAttention(
            self.config.rule_dim,
            self.config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.probe_history_norm = torch.nn.LayerNorm(self.config.rule_dim)
        self.repeated_transition_embedding = torch.nn.Parameter(
            torch.empty(self.config.rule_dim)
        )
        torch.nn.init.normal_(self.repeated_transition_embedding, std=0.02)
        self.probe_history_ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.rule_dim),
            torch.nn.Linear(self.config.rule_dim, self.config.attention_ffn),
            torch.nn.GELU(),
            torch.nn.Linear(self.config.attention_ffn, self.config.rule_dim),
        )

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        mask = self._validated_agent_probe_mask(
            support,
            is_agent_probe_result,
        )
        tokens = self._transition_tokens(support)
        repeated = self._has_identical_prior_transition(support) & mask
        queries = tokens + (
            mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
        ) + (
            repeated.to(dtype=tokens.dtype)[..., None]
            * self.repeated_transition_embedding[None, None]
        )
        steps = support.support_steps
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.probe_history_attention(
            queries,
            queries,
            queries,
            attn_mask=causal_mask,
            key_padding_mask=~support.support_mask,
            need_weights=False,
        )
        contextual = self.probe_history_norm(queries + attended)
        contextual = contextual + self.probe_history_ffn(contextual)
        evidence_tokens = torch.where(mask[..., None], contextual, tokens)
        axis_logits = self.evidence_axis_head(evidence_tokens)
        value_logits = torch.stack(
            [head(evidence_tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        return axis_logits, value_logits

    @staticmethod
    def _has_identical_prior_transition(
        support: RuleGridTensorBatch,
    ) -> Tensor:
        def pairwise_equal(values: Tensor) -> Tensor:
            comparison = values[:, :, None] == values[:, None, :]
            return comparison.flatten(start_dim=3).all(dim=-1)

        same = (
            pairwise_equal(support.support_states)
            & pairwise_equal(support.support_actions)
            & pairwise_equal(support.support_targets)
        )
        if support.support_action_mask is not None:
            same = same & pairwise_equal(support.support_action_mask)
        steps = support.support_steps
        prior = torch.tril(
            torch.ones(steps, steps, dtype=torch.bool, device=same.device),
            diagonal=-1,
        )
        valid_pairs = (
            support.support_mask[:, :, None]
            & support.support_mask[:, None, :]
        )
        return (same & prior[None] & valid_pairs).any(dim=-1)


class RelativeEventSetEncoder(torch.nn.Module):
    """Encode public transitions as a translation-invariant set of events.

    Non-background or changed cells are assigned to their nearest public
    action atom.  Cells are described only by action-relative offsets, colors,
    and before/after change; absolute coordinates are used for assignment but
    never embedded.  Mean pooling makes the transition representation
    invariant to the ordering of composite action atoms.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.grid_size = int(config.grid_size)
        event_dim = int(config.rule_dim)
        embedding_dim = max(8, event_dim // 4)
        self.row_offset_embedding = torch.nn.Embedding(
            2 * self.grid_size - 1,
            embedding_dim,
        )
        self.column_offset_embedding = torch.nn.Embedding(
            2 * self.grid_size - 1,
            embedding_dim,
        )
        self.pre_color_embedding = torch.nn.Embedding(
            config.num_colors,
            embedding_dim,
        )
        self.post_color_embedding = torch.nn.Embedding(
            config.num_colors,
            embedding_dim,
        )
        self.kind_embedding = torch.nn.Embedding(
            config.num_action_kinds,
            embedding_dim,
        )
        self.direction_embedding = torch.nn.Embedding(
            config.num_directions,
            embedding_dim,
        )
        self.cell_mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim + 1, event_dim),
            torch.nn.GELU(),
            torch.nn.Linear(event_dim, event_dim),
        )
        self.atom_meta_mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim, event_dim),
            torch.nn.GELU(),
            torch.nn.Linear(event_dim, event_dim),
        )
        self.atom_norm = torch.nn.LayerNorm(event_dim)
        self.transition_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(event_dim),
            torch.nn.Linear(event_dim, event_dim),
            torch.nn.GELU(),
            torch.nn.Linear(event_dim, event_dim),
        )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(self.grid_size),
                torch.arange(self.grid_size),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        self.register_buffer("cell_coordinates", coordinates, persistent=False)

    @staticmethod
    def _action_atoms(support: RuleGridTensorBatch) -> tuple[Tensor, Tensor]:
        if support.support_actions.ndim == 3:
            actions = support.support_actions[:, :, None]
            mask = support.support_mask[:, :, None]
        else:
            actions = support.support_actions
            assert support.support_action_mask is not None
            mask = support.support_action_mask & support.support_mask[:, :, None]
        return actions, mask

    def forward(self, support: RuleGridTensorBatch) -> tuple[Tensor, Tensor]:
        actions, atom_mask = self._action_atoms(support)
        states = support.support_states
        targets = support.support_targets
        batch_size, steps, height, width = states.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError("relative event grid size differs from configuration")
        atoms = actions.shape[2]
        cells = height * width
        flat_states = states.reshape(batch_size, steps, cells)
        flat_targets = targets.reshape(batch_size, steps, cells)
        centers = actions[..., 1:3]
        relative = (
            self.cell_coordinates[None, None, None]
            - centers[..., None, :]
        )
        distance = relative.abs().sum(dim=-1)
        invalid_distance = 4 * self.grid_size
        valid_distance = distance.masked_fill(
            ~atom_mask[..., None],
            invalid_distance,
        )
        nearest = valid_distance.min(dim=2, keepdim=True).values
        interesting = flat_states.ne(0) | flat_targets.ne(0) | flat_states.ne(
            flat_targets
        )
        owner = (
            valid_distance.eq(nearest)
            & atom_mask[..., None]
            & interesting[:, :, None]
        )
        owner_count = owner.sum(dim=2, keepdim=True).clamp_min(1)
        weights = owner.to(dtype=torch.float32) / owner_count.to(
            dtype=torch.float32
        )

        relative_row = relative[..., 0] + self.grid_size - 1
        relative_column = relative[..., 1] + self.grid_size - 1
        pre = flat_states[:, :, None].expand(-1, -1, atoms, -1)
        post = flat_targets[:, :, None].expand(-1, -1, atoms, -1)
        changed = pre.ne(post).to(dtype=self.pre_color_embedding.weight.dtype)
        cell_features = torch.cat(
            (
                self.row_offset_embedding(relative_row),
                self.column_offset_embedding(relative_column),
                self.pre_color_embedding(pre),
                self.post_color_embedding(post),
                changed[..., None],
            ),
            dim=-1,
        )
        encoded_cells = self.cell_mlp(cell_features)
        normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_cells = (
            encoded_cells * weights[..., None]
        ).sum(dim=-2) / normalizer

        anchor_index = actions[..., 1] * width + actions[..., 2]
        anchor_pre = flat_states.gather(2, anchor_index)
        anchor_post = flat_targets.gather(2, anchor_index)
        meta = torch.cat(
            (
                self.kind_embedding(actions[..., 0]),
                self.direction_embedding(actions[..., 3]),
                self.pre_color_embedding(anchor_pre),
                self.post_color_embedding(anchor_post),
            ),
            dim=-1,
        )
        atom_events = self.atom_norm(
            pooled_cells + self.atom_meta_mlp(meta)
        )
        atom_events = atom_events * atom_mask[..., None].to(
            dtype=atom_events.dtype
        )
        atom_count = atom_mask.sum(dim=2, keepdim=True).clamp_min(1).to(
            dtype=atom_events.dtype
        )
        transition_events = atom_events.sum(dim=2) / atom_count
        transition_events = self.transition_mlp(transition_events)
        transition_events = transition_events * support.support_mask[..., None].to(
            dtype=transition_events.dtype
        )
        return atom_events, transition_events

    def public_atom_fields(
        self,
        support: RuleGridTensorBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Place public cell colors in action-relative atom fields.

        The returned color IDs are used only through equality comparisons by
        relational heads.  Their coordinates are relative to each public
        action anchor, so the fields are translation invariant.
        """

        actions, atom_mask = self._action_atoms(support)
        states = support.support_states
        targets = support.support_targets
        batch_size, steps, height, width = states.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError("relative event grid size differs from configuration")
        atoms = actions.shape[2]
        cells = height * width
        flat_states = states.reshape(batch_size, steps, cells)
        flat_targets = targets.reshape(batch_size, steps, cells)
        centers = actions[..., 1:3]
        relative = (
            self.cell_coordinates[None, None, None]
            - centers[..., None, :]
        )
        distance = relative.abs().sum(dim=-1)
        valid_distance = distance.masked_fill(
            ~atom_mask[..., None],
            4 * self.grid_size,
        )
        nearest = valid_distance.min(dim=2, keepdim=True).values
        interesting = flat_states.ne(0) | flat_targets.ne(0) | flat_states.ne(
            flat_targets
        )
        owner = (
            valid_distance.eq(nearest)
            & atom_mask[..., None]
            & interesting[:, :, None]
        )
        side = 2 * self.grid_size - 1
        relative_index = (
            (relative[..., 0] + self.grid_size - 1) * side
            + relative[..., 1]
            + self.grid_size
            - 1
        )
        field_shape = (batch_size, steps, atoms, side * side)
        pre_field = torch.zeros(
            field_shape,
            dtype=flat_states.dtype,
            device=flat_states.device,
        )
        post_field = torch.zeros_like(pre_field)
        present = torch.zeros(
            field_shape,
            dtype=torch.bool,
            device=flat_states.device,
        )
        pre = flat_states[:, :, None].expand(-1, -1, atoms, -1)
        post = flat_targets[:, :, None].expand(-1, -1, atoms, -1)
        pre_field.scatter_(
            -1,
            relative_index,
            torch.where(owner, pre, torch.zeros_like(pre)),
        )
        post_field.scatter_(
            -1,
            relative_index,
            torch.where(owner, post, torch.zeros_like(post)),
        )
        present.scatter_(-1, relative_index, owner)
        return pre_field, post_field, present, actions, atom_mask


class RelativeEventHistoryProbeFactorBeliefCausalK4(
    HistoryConditionedProbeFactorBeliefCausalK4
):
    """Use action-centered event sets for controller-marked probe results."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relative_event_encoder = RelativeEventSetEncoder(self.config)

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        mask = self._validated_agent_probe_mask(
            support,
            is_agent_probe_result,
        )
        tokens = self._transition_tokens(support)
        _, relative_events = self.relative_event_encoder(support)
        repeated = self._has_identical_prior_transition(support) & mask
        history_tokens = tokens + relative_events
        # The marked query deliberately excludes the absolute-position token.
        # Consequently, translating one public event while preserving its
        # action-relative cell set cannot change the probe representation.
        queries = torch.where(
            mask[..., None],
            relative_events,
            history_tokens,
        ) + (
            mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
        ) + (
            repeated.to(dtype=tokens.dtype)[..., None]
            * self.repeated_transition_embedding[None, None]
        )
        steps = support.support_steps
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.probe_history_attention(
            queries,
            queries,
            queries,
            attn_mask=causal_mask,
            key_padding_mask=~support.support_mask,
            need_weights=False,
        )
        contextual = self.probe_history_norm(queries + attended)
        contextual = contextual + self.probe_history_ffn(contextual)
        evidence_tokens = torch.where(mask[..., None], contextual, tokens)
        axis_logits = self.evidence_axis_head(evidence_tokens)
        value_logits = torch.stack(
            [head(evidence_tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        return axis_logits, value_logits


class CompositeRelativeEventHistoryProbeFactorBeliefCausalK4(
    RelativeEventHistoryProbeFactorBeliefCausalK4
):
    """Use relative event sets only where a composite needs decomposition.

    Single-action probes retain the established raw transition path.  A
    composite result replaces its raw absolute-position token with the
    permutation- and translation-invariant event-set token.
    """

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        mask = self._validated_agent_probe_mask(
            support,
            is_agent_probe_result,
        )
        tokens = self._transition_tokens(support)
        _, relative_events = self.relative_event_encoder(support)
        if support.support_action_mask is None:
            action_count = torch.ones_like(support.support_mask, dtype=torch.long)
        else:
            action_count = support.support_action_mask.sum(dim=-1)
        relative_probe = mask & action_count.gt(1)
        repeated = self._has_identical_prior_transition(support) & mask
        queries = torch.where(
            relative_probe[..., None],
            relative_events,
            tokens,
        ) + (
            mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
        ) + (
            repeated.to(dtype=tokens.dtype)[..., None]
            * self.repeated_transition_embedding[None, None]
        )
        steps = support.support_steps
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.probe_history_attention(
            queries,
            queries,
            queries,
            attn_mask=causal_mask,
            key_padding_mask=~support.support_mask,
            need_weights=False,
        )
        contextual = self.probe_history_norm(queries + attended)
        contextual = contextual + self.probe_history_ffn(contextual)
        evidence_tokens = torch.where(mask[..., None], contextual, tokens)
        axis_logits = self.evidence_axis_head(evidence_tokens)
        value_logits = torch.stack(
            [head(evidence_tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        return axis_logits, value_logits


class RelationalCompositeEventHistoryProbeFactorBeliefCausalK4(
    RelativeEventHistoryProbeFactorBeliefCausalK4
):
    """Compare composite queries with history in a shared relative-event space.

    Ordinary evidence decoding still uses the established raw tokens.  For
    attention keys/values, prior public observations receive a relative-event
    residual; a composite probe uses only its invariant event-set query, while
    a single-action probe keeps its raw query.
    """

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        mask = self._validated_agent_probe_mask(
            support,
            is_agent_probe_result,
        )
        tokens = self._transition_tokens(support)
        _, relative_events = self.relative_event_encoder(support)
        if support.support_action_mask is None:
            action_count = torch.ones_like(support.support_mask, dtype=torch.long)
        else:
            action_count = support.support_action_mask.sum(dim=-1)
        relative_probe = mask & action_count.gt(1)
        repeated = self._has_identical_prior_transition(support) & mask
        history_tokens = tokens + relative_events
        base_queries = torch.where(mask[..., None], tokens, history_tokens)
        queries = torch.where(
            relative_probe[..., None],
            relative_events,
            base_queries,
        ) + (
            mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
        ) + (
            repeated.to(dtype=tokens.dtype)[..., None]
            * self.repeated_transition_embedding[None, None]
        )
        steps = support.support_steps
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.probe_history_attention(
            queries,
            queries,
            queries,
            attn_mask=causal_mask,
            key_padding_mask=~support.support_mask,
            need_weights=False,
        )
        contextual = self.probe_history_norm(queries + attended)
        contextual = contextual + self.probe_history_ffn(contextual)
        evidence_tokens = torch.where(mask[..., None], contextual, tokens)
        axis_logits = self.evidence_axis_head(evidence_tokens)
        value_logits = torch.stack(
            [head(evidence_tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        return axis_logits, value_logits


class AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4(
    RelativeEventHistoryProbeFactorBeliefCausalK4
):
    """Match each composite event atom against earlier public event atoms.

    A transition-level mean cannot express that different atoms in one
    composite may be explained by different observations.  This head keeps
    the established raw path for ordinary evidence and single-action probes,
    but gives every atom in a marked composite its own causal retrieval over
    all earlier public atoms.  The per-atom relations are pooled as a set, so
    translating or permuting a composite does not change its representation.
    """

    atom_pair_feature_dim = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        event_dim = int(self.config.rule_dim)
        hidden_dim = int(self.config.attention_ffn)
        relation_dim = 4 * event_dim + 1 + int(self.atom_pair_feature_dim)
        self.atom_relation_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(relation_dim),
            torch.nn.Linear(relation_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, event_dim),
        )
        self.atom_relation_pool = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * event_dim + 2),
            torch.nn.Linear(2 * event_dim + 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, event_dim),
        )
        self.composite_probe_norm = torch.nn.LayerNorm(event_dim)
        self.composite_probe_ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(event_dim),
            torch.nn.Linear(event_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, event_dim),
        )

    def _matched_atom_pair_features(
        self,
        support: RuleGridTensorBatch,
        nearest_weight: Tensor,
    ) -> Tensor | None:
        del support, nearest_weight
        return None

    def _causal_atom_match_context(
        self,
        support: RuleGridTensorBatch,
        atom_events: Tensor,
    ) -> Tensor:
        """Return a permutation-invariant relation summary for every step."""

        _, atom_mask = self.relative_event_encoder._action_atoms(support)
        atom_mask = atom_mask & support.support_mask[..., None]
        normalized = F.normalize(atom_events, dim=-1, eps=1e-6)
        # [B, query_step, query_atom, history_step, history_atom]
        similarity = torch.einsum(
            "btld,bsmd->btlsm",
            normalized,
            normalized,
        )
        steps = support.support_steps
        atoms = atom_events.shape[2]
        causal = torch.tril(
            torch.ones(
                steps,
                steps,
                dtype=torch.bool,
                device=atom_events.device,
            ),
            diagonal=-1,
        )
        candidate_mask = (
            causal[None, :, None, :, None]
            & atom_mask[:, None, None, :, :]
        ).expand(-1, -1, atoms, -1, -1)
        has_prior = candidate_mask.any(dim=(-1, -2))
        masked_similarity = similarity.masked_fill(~candidate_mask, -2.0)
        max_similarity = masked_similarity.amax(dim=(-1, -2))
        max_similarity = torch.where(
            has_prior,
            max_similarity,
            torch.zeros_like(max_similarity),
        )

        # Average exact maximum ties rather than selecting an index.  This is
        # invariant to history-atom order and preserves an exact replay match.
        nearest = candidate_mask & (
            similarity >= max_similarity[..., None, None] - 1e-6
        )
        nearest_weight = nearest.to(dtype=atom_events.dtype)
        nearest_weight = nearest_weight / nearest_weight.sum(
            dim=(-1, -2),
            keepdim=True,
        ).clamp_min(1.0)
        retrieved = torch.einsum(
            "btlsm,bsmd->btld",
            nearest_weight,
            atom_events,
        )
        relation_parts = (
            atom_events,
            retrieved,
            (atom_events - retrieved).abs(),
            atom_events * retrieved,
            max_similarity[..., None],
        )
        pair_features = self._matched_atom_pair_features(
            support,
            nearest_weight,
        )
        if pair_features is not None:
            relation_parts = relation_parts + (pair_features,)
        relation = self.atom_relation_mlp(torch.cat(relation_parts, dim=-1))
        query_mask = atom_mask & has_prior
        relation = relation * query_mask[..., None].to(dtype=relation.dtype)
        query_count = query_mask.sum(dim=2, keepdim=True).clamp_min(1).to(
            dtype=relation.dtype
        )
        mean_relation = relation.sum(dim=2) / query_count
        lowest = torch.finfo(relation.dtype).min
        max_relation = relation.masked_fill(
            ~query_mask[..., None],
            lowest,
        ).amax(dim=2)
        has_query = query_mask.any(dim=2)
        max_relation = torch.where(
            has_query[..., None],
            max_relation,
            torch.zeros_like(max_relation),
        )
        similarity_sum = (
            max_similarity
            * query_mask.to(dtype=max_similarity.dtype)
        ).sum(dim=2)
        mean_similarity = similarity_sum / query_count.squeeze(-1)
        min_similarity = max_similarity.masked_fill(~query_mask, 2.0).amin(dim=2)
        min_similarity = torch.where(
            has_query,
            min_similarity,
            torch.zeros_like(min_similarity),
        )
        pooled = self.atom_relation_pool(
            torch.cat(
                (
                    mean_relation,
                    max_relation,
                    mean_similarity[..., None],
                    min_similarity[..., None],
                ),
                dim=-1,
            )
        )
        return pooled * support.support_mask[..., None].to(dtype=pooled.dtype)

    def _factor_belief_evidence_outputs(
        self,
        support: RuleGridTensorBatch,
        *,
        is_agent_probe_result: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        mask = self._validated_agent_probe_mask(
            support,
            is_agent_probe_result,
        )
        tokens = self._transition_tokens(support)
        atom_events, relative_events = self.relative_event_encoder(support)
        if support.support_action_mask is None:
            action_count = torch.ones_like(support.support_mask, dtype=torch.long)
        else:
            action_count = support.support_action_mask.sum(dim=-1)
        composite_probe = mask & action_count.gt(1)
        repeated = self._has_identical_prior_transition(support) & mask

        raw_queries = tokens + (
            mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
        ) + (
            repeated.to(dtype=tokens.dtype)[..., None]
            * self.repeated_transition_embedding[None, None]
        )
        steps = support.support_steps
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.probe_history_attention(
            raw_queries,
            raw_queries,
            raw_queries,
            attn_mask=causal_mask,
            key_padding_mask=~support.support_mask,
            need_weights=False,
        )
        raw_contextual = self.probe_history_norm(raw_queries + attended)
        raw_contextual = raw_contextual + self.probe_history_ffn(raw_contextual)

        atom_context = self._causal_atom_match_context(support, atom_events)
        composite_query = (
            relative_events
            + atom_context
            + mask.to(dtype=tokens.dtype)[..., None]
            * self.agent_probe_result_embedding[None, None]
            + repeated.to(dtype=tokens.dtype)[..., None]
            * self.repeated_transition_embedding[None, None]
        )
        composite_contextual = self.composite_probe_norm(composite_query)
        composite_contextual = (
            composite_contextual
            + self.composite_probe_ffn(composite_contextual)
        )
        probe_contextual = torch.where(
            composite_probe[..., None],
            composite_contextual,
            raw_contextual,
        )
        evidence_tokens = torch.where(mask[..., None], probe_contextual, tokens)
        axis_logits = self.evidence_axis_head(evidence_tokens)
        value_logits = torch.stack(
            [head(evidence_tokens) for head in self.evidence_value_heads],
            dim=2,
        )
        return axis_logits, value_logits


class PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4(
    AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4
):
    """Add color-renaming-invariant relations to causal atom matching.

    Raw colors never enter these pair features as numeric identities.  The
    model sees only whether colors at corresponding action-relative cells are
    equal across the query and its retrieved public precedent.  This exposes
    relations such as ``query.post == history.pre`` without naming a palette
    role or a simulator mechanism.
    """

    atom_pair_feature_dim = 16

    def _palette_invariant_atom_pair_features(
        self,
        support: RuleGridTensorBatch,
    ) -> Tensor:
        pre, post, present, actions, _ = (
            self.relative_event_encoder.public_atom_fields(support)
        )
        query_pre = pre[:, :, :, None, None, :]
        query_post = post[:, :, :, None, None, :]
        query_present = present[:, :, :, None, None, :]
        history_pre = pre[:, None, None, :, :, :]
        history_post = post[:, None, None, :, :, :]
        history_present = present[:, None, None, :, :, :]
        overlap = query_present & history_present
        union = query_present | history_present
        overlap_count = overlap.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        union_count = union.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        query_count = query_present.sum(dim=-1).clamp_min(1).to(
            dtype=torch.float32
        )
        history_count = history_present.sum(dim=-1).clamp_min(1).to(
            dtype=torch.float32
        )
        query_changed = query_pre.ne(query_post)
        history_changed = history_pre.ne(history_post)
        changed_region = overlap & (query_changed | history_changed)
        changed_count = changed_region.sum(dim=-1).clamp_min(1).to(
            dtype=torch.float32
        )

        def fraction(condition: Tensor, region: Tensor, denominator: Tensor) -> Tensor:
            return (condition & region).sum(dim=-1).to(dtype=torch.float32) / denominator

        equality = (
            query_pre.eq(history_pre),
            query_pre.eq(history_post),
            query_post.eq(history_pre),
            query_post.eq(history_post),
        )
        query_kind = actions[:, :, :, None, None, 0]
        history_kind = actions[:, None, None, :, :, 0]
        query_direction = actions[:, :, :, None, None, 3]
        history_direction = actions[:, None, None, :, :, 3]
        features = (
            overlap.sum(dim=-1).to(dtype=torch.float32) / query_count,
            overlap.sum(dim=-1).to(dtype=torch.float32) / history_count,
            overlap.sum(dim=-1).to(dtype=torch.float32) / union_count,
            fraction(query_changed, overlap, overlap_count),
            fraction(history_changed, overlap, overlap_count),
            fraction(query_changed.eq(history_changed), overlap, overlap_count),
            *(fraction(item, overlap, overlap_count) for item in equality),
            *(fraction(item, changed_region, changed_count) for item in equality),
            query_kind.eq(history_kind).to(dtype=torch.float32),
            query_direction.eq(history_direction).to(dtype=torch.float32),
        )
        return torch.stack(features, dim=-1).to(
            device=support.support_states.device,
        )

    def _matched_atom_pair_features(
        self,
        support: RuleGridTensorBatch,
        nearest_weight: Tensor,
    ) -> Tensor:
        pair_features = self._palette_invariant_atom_pair_features(support)
        return torch.einsum(
            "btlsm,btlsmf->btlf",
            nearest_weight,
            pair_features.to(dtype=nearest_weight.dtype),
        )


class TranslationInvariantHistoryProbeFactorBeliefCausalK4(
    HistoryConditionedProbeFactorBeliefCausalK4
):
    """Ablate absolute coordinates from the learned public support encoder.

    Convolutions and global/changed-cell pooling retain local layout, while
    zero frozen row/column embeddings prevent a mechanism from being tied to
    its fixture coordinate.  This is intentionally a small causal ablation;
    it does not assume any named RuleGrid mechanism or privileged role map.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.independent_support_encoders:
            raise ValueError(
                "translation-invariant history head requires independent "
                "support encoders"
            )
        assert self.support_grid_encoder is not None
        assert self.support_action_encoder is not None
        embeddings = (
            self.support_grid_encoder.row_embedding,
            self.support_grid_encoder.column_embedding,
            self.support_action_encoder.row_embedding,
            self.support_action_encoder.column_embedding,
        )
        with torch.no_grad():
            for embedding in embeddings:
                embedding.weight.zero_()
        for embedding in embeddings:
            embedding.weight.requires_grad_(False)


__all__ = [
    "AtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4",
    "CompositeRelativeEventHistoryProbeFactorBeliefCausalK4",
    "FactorizedPublicVersionSpaceCausalK4",
    "FactorizedPublicVersionSpaceK4Loss",
    "HistoryConditionedProbeFactorBeliefCausalK4",
    "PaletteInvariantAtomMatchedCompositeEventHistoryProbeFactorBeliefCausalK4",
    "PublicVersionSpaceCausalK4",
    "PublicVersionSpaceK4Loss",
    "ProbeAwareSymmetryFactorBeliefCausalK4",
    "RelativeEventHistoryProbeFactorBeliefCausalK4",
    "RelativeEventSetEncoder",
    "RelationalCompositeEventHistoryProbeFactorBeliefCausalK4",
    "SymmetryAwareFactorBelief",
    "SymmetryAwareFactorBeliefCausalK4",
    "SymmetryAwareFactorBeliefLoss",
    "TransitionEvidencePublicVersionSpaceCausalK4",
    "TransitionEvidencePublicVersionSpaceK4Loss",
    "TranslationInvariantHistoryProbeFactorBeliefCausalK4",
]
