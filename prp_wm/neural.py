"""Small, grid-native Persistent Rule Particle World Model.

This module is deliberately independent of the symbolic RuleGrid simulator.  The
simulator owns hidden programs and privileged diagnostic targets; this module
only consumes integer grids, public action tensors, observed transitions, and
optionally privileged targets for *training losses*.  Keeping that boundary
explicit makes it much harder to accidentally feed a rule ID or a probe kind to
the controller.

The implementation matches the intended model-architecture boundary and serves
as a runnable training scaffold:

* 8x8 categorical grids are encoded with color/row/column embeddings and a
  no-downsampling CNN;
* four persistent rule modes are updated recurrently after each observed
  transition, followed by one shared set-attention layer;
* a single FiLM-conditioned decoder is evaluated once per mode (there are not
  four independent decoders);
* outcome scores are proper per-cell change/new-color probabilities; and
* support losses are prequential, while post-update mode weights are computed
  by replaying the observed history under the moved modes.

It is intentionally a scaffold rather than a claim that the full Stage-1
protocol has been run: materialized manifests, baselines, and the complete
evaluation protocol remain separate pieces of work.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import itertools
import math
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - exercised on dependency-free Stage 0 installs.
    raise ImportError(
        "prp_wm.neural requires PyTorch. Install the optional extra with "
        "`pip install -e '.[neural]'` or install a matching torch build."
    ) from error


ACTION_FIELDS = 4
"""Public action tensor columns: ``(kind, row, column, direction)``.

``direction=4`` is the public no-direction sentinel used by ``ACTIVATE``.
Composite actions may be represented by an additional atom axis: ``[..., L, 4]``
plus a boolean mask.  The action encoder is permutation-invariant across atoms.
"""


@dataclass(frozen=True)
class NeuralPRPConfig:
    """Architecture defaults only; they do not imply a completed experiment."""

    grid_size: int = 8
    num_colors: int = 16
    color_embedding: int = 64
    position_embedding: int = 64
    encoder_channels: int = 64
    encoder_resblocks: int = 4
    normalization_groups: int = 8
    action_embedding: int = 32
    num_action_kinds: int = 4
    num_directions: int = 5
    particles: int = 4
    rule_dim: int = 128
    attention_heads: int = 4
    attention_ffn: int = 256
    decoder_resblocks: int = 4
    beta: float = 1.0
    lambda_joint: float = 1.0
    lambda_assign: float = 0.5

    def __post_init__(self) -> None:
        if self.grid_size <= 1:
            raise ValueError("grid_size must be at least 2")
        if self.num_colors <= 1:
            raise ValueError("num_colors must be at least 2")
        if self.color_embedding <= 0 or self.position_embedding <= 0:
            raise ValueError("grid embedding dimensions must be positive")
        if self.color_embedding != self.position_embedding:
            raise ValueError(
                "color_embedding and position_embedding must match for additive grid embeddings"
            )
        if self.encoder_channels <= 0:
            raise ValueError("encoder_channels must be positive")
        if self.encoder_channels % self.normalization_groups:
            raise ValueError("encoder_channels must be divisible by normalization_groups")
        if self.particles != 4:
            raise ValueError("the Stage-1 scaffold deliberately fixes K=4")
        if self.rule_dim % self.attention_heads:
            raise ValueError("rule_dim must be divisible by attention_heads")
        if self.num_action_kinds <= 0 or self.num_directions <= 0:
            raise ValueError("action vocabularies must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")


@dataclass(frozen=True)
class RuleGridTensorBatch:
    """Tensor-only boundary between RuleGrid records and the neural model.

    Grids have ``long`` dtype and use the palette IDs ``[0, num_colors)``.
    Support tensors have shape ``[B, T, H, W]`` and query tensors have shape
    ``[B, Q, H, W]``.  Actions can be either one public atom per probe
    (``[B, T/Q, 4]``) or a public composite action
    (``[B, T/Q, L, 4]``).  The optional action masks then have shape
    ``[B, T/Q, L]``.

    ``behavior_targets`` is privileged training-only data.  It contains a
    complete diagnostic panel for each currently compatible behavior class,
    with shape ``[B, M, Q, H, W]``.  No inference method reads it.
    """

    support_states: Tensor
    support_actions: Tensor
    support_targets: Tensor
    support_mask: Tensor
    query_states: Tensor | None = None
    query_actions: Tensor | None = None
    query_targets: Tensor | None = None
    behavior_targets: Tensor | None = None
    behavior_mass: Tensor | None = None
    support_action_mask: Tensor | None = None
    query_action_mask: Tensor | None = None

    def to(self, *args: Any, **kwargs: Any) -> "RuleGridTensorBatch":
        """Return a copy whose tensors were moved with ``Tensor.to``."""

        return RuleGridTensorBatch(
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
        return int(self.support_states.shape[0])

    @property
    def support_steps(self) -> int:
        return int(self.support_states.shape[1])

    def validate(self, config: NeuralPRPConfig) -> None:
        _validate_grid_tensor("support_states", self.support_states, config, 4)
        _validate_grid_tensor("support_targets", self.support_targets, config, 4)
        if self.support_states.shape != self.support_targets.shape:
            raise ValueError("support_states and support_targets must have the same shape")
        batch_size, support_steps, _, _ = self.support_states.shape
        if self.support_mask.shape != (batch_size, support_steps):
            raise ValueError("support_mask must have shape [B, T]")
        if self.support_mask.dtype != torch.bool:
            raise TypeError("support_mask must have torch.bool dtype")
        _validate_actions(
            "support_actions",
            self.support_actions,
            (batch_size, support_steps),
            self.support_action_mask,
            config,
        )

        query_fields = (self.query_states, self.query_actions)
        if any(value is None for value in query_fields):
            if any(
                value is not None
                for value in (
                    self.query_states,
                    self.query_actions,
                    self.query_targets,
                    self.behavior_targets,
                    self.behavior_mass,
                    self.query_action_mask,
                )
            ):
                raise ValueError("query fields must be supplied together")
            return

        assert self.query_states is not None and self.query_actions is not None
        _validate_grid_tensor("query_states", self.query_states, config, 4)
        if self.query_states.shape[0] != batch_size:
            raise ValueError("support and query batch sizes must match")
        query_count = self.query_states.shape[1]
        _validate_actions(
            "query_actions",
            self.query_actions,
            (batch_size, query_count),
            self.query_action_mask,
            config,
        )
        if self.query_targets is not None:
            _validate_grid_tensor("query_targets", self.query_targets, config, 4)
            if self.query_targets.shape != self.query_states.shape:
                raise ValueError("query_targets must have query_states shape")
        if self.behavior_targets is not None:
            _validate_grid_tensor("behavior_targets", self.behavior_targets, config, 5)
            if self.behavior_targets.shape[0] != batch_size:
                raise ValueError("behavior_targets batch size must match support")
            if self.behavior_targets.shape[2:] != self.query_states.shape[1:]:
                raise ValueError("behavior_targets must have shape [B, M, Q, H, W]")
            if self.behavior_targets.shape[1] > config.particles:
                raise ValueError("behavior class count M must not exceed K=4")
            if self.behavior_mass is None:
                raise ValueError("behavior_mass is required with behavior_targets")
            if self.behavior_mass.shape != self.behavior_targets.shape[:2]:
                raise ValueError("behavior_mass must have shape [B, M]")
            if not torch.is_floating_point(self.behavior_mass):
                raise TypeError("behavior_mass must be floating point")
            if torch.any(self.behavior_mass < 0):
                raise ValueError("behavior_mass cannot be negative")
            if not torch.allclose(
                self.behavior_mass.sum(dim=1),
                torch.ones(batch_size, device=self.behavior_mass.device, dtype=self.behavior_mass.dtype),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("behavior_mass must sum to one per task")
        elif self.behavior_mass is not None:
            raise ValueError("behavior_mass requires behavior_targets")


@dataclass(frozen=True)
class OutcomePrediction:
    """A proper factorized next-grid outcome distribution for all K modes."""

    input_colors: Tensor  # [B, H, W]
    change_logits: Tensor  # [B, K, H, W]
    new_color_logits: Tensor  # [B, K, C, H, W], original color excluded at scoring time

    def log_prob_cells(self, target: Tensor) -> Tensor:
        """Return ``log p(target_cell | input_cell, mode)`` as ``[B,K,H,W]``."""

        if target.ndim != 3 or target.shape != self.input_colors.shape:
            raise ValueError("target must have the same [B,H,W] shape as input_colors")
        if target.dtype != torch.long:
            raise TypeError("target grid must have torch.long dtype")
        batch_size, modes, colors, height, width = self.new_color_logits.shape
        if torch.any(target < 0) or torch.any(target >= colors):
            raise ValueError("target contains a color outside the model palette")
        original = self.input_colors[:, None, None, :, :]
        masked_color_logits = self.new_color_logits.masked_fill(
            original == torch.arange(colors, device=target.device)[None, None, :, None, None],
            torch.finfo(self.new_color_logits.dtype).min,
        )
        color_log_probs = F.log_softmax(masked_color_logits, dim=2)
        target_index = target[:, None, None, :, :].expand(
            batch_size, modes, 1, height, width
        )
        changed_color_log_prob = color_log_probs.gather(2, target_index).squeeze(2)
        unchanged_log_prob = F.logsigmoid(-self.change_logits)
        changed_log_prob = F.logsigmoid(self.change_logits) + changed_color_log_prob
        return torch.where(
            target[:, None, :, :] == self.input_colors[:, None, :, :],
            unchanged_log_prob,
            changed_log_prob,
        )

    def log_prob(self, target: Tensor) -> Tensor:
        """Return full-grid log probability with shape ``[B,K]``."""

        return self.log_prob_cells(target).sum(dim=(-2, -1))


@dataclass(frozen=True)
class InferenceState:
    """Public inference result after a support prefix."""

    modes: Tensor  # [B,K,D]
    log_weights: Tensor  # [B,K]
    prequential_mixture_log_prob: Tensor  # [B,T]

    @property
    def weights(self) -> Tensor:
        return self.log_weights.exp()


@dataclass(frozen=True)
class NeuralPRPOutput:
    """Forward output, without exposing privileged targets to inference."""

    inference: InferenceState
    query_log_prob_by_mode: Tensor | None = None  # [B,Q,K] if query_targets were given


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    support: Tensor
    joint: Tensor
    assignment: Tensor
    inference: InferenceState

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_support": float(self.support.detach().cpu()),
            "loss_joint": float(self.joint.detach().cpu()),
            "loss_assignment": float(self.assignment.detach().cpu()),
        }


def _validate_grid_tensor(
    name: str, tensor: Tensor, config: NeuralPRPConfig, expected_ndim: int
) -> None:
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions")
    if tensor.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long dtype")
    if tensor.shape[-2:] != (config.grid_size, config.grid_size):
        raise ValueError(f"{name} must use {config.grid_size}x{config.grid_size} grids")
    if torch.any(tensor < 0) or torch.any(tensor >= config.num_colors):
        raise ValueError(f"{name} contains a color outside [0, {config.num_colors})")


def _validate_actions(
    name: str,
    actions: Tensor,
    prefix_shape: tuple[int, int],
    action_mask: Tensor | None,
    config: NeuralPRPConfig,
) -> None:
    if actions.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long dtype")
    if actions.shape[:2] != prefix_shape or actions.shape[-1] != ACTION_FIELDS:
        raise ValueError(f"{name} must have [B,T/Q,4] or [B,T/Q,L,4] shape")
    if actions.ndim == 3:
        if action_mask is not None:
            raise ValueError(f"{name} has no atom axis, so its mask must be None")
    elif actions.ndim == 4:
        if action_mask is None or action_mask.shape != actions.shape[:-1]:
            raise ValueError(f"{name} composite actions require [B,T/Q,L] action_mask")
        if action_mask.dtype != torch.bool:
            raise TypeError(f"{name} action_mask must use torch.bool dtype")
        if not torch.all(action_mask.any(dim=-1)):
            raise ValueError(f"{name} has an empty composite action")
    else:
        raise ValueError(f"{name} must have [B,T/Q,4] or [B,T/Q,L,4] shape")
    kind, row, column, direction = (actions[..., index] for index in range(ACTION_FIELDS))
    if torch.any(kind < 0) or torch.any(kind >= config.num_action_kinds):
        raise ValueError(f"{name} kind ID is outside vocabulary")
    if torch.any(row < 0) or torch.any(row >= config.grid_size):
        raise ValueError(f"{name} row is outside grid")
    if torch.any(column < 0) or torch.any(column >= config.grid_size):
        raise ValueError(f"{name} column is outside grid")
    if torch.any(direction < 0) or torch.any(direction >= config.num_directions):
        raise ValueError(f"{name} direction ID is outside vocabulary")


class ResidualBlock(nn.Module):
    """No-downsampling GroupNorm/SiLU residual block used by the grid encoder."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.conv1(F.silu(self.norm1(value)))
        value = self.conv2(F.silu(self.norm2(value)))
        return value + residual


class GridEncoder(nn.Module):
    """Additive categorical grid encoder with no resizing or downsampling."""

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        self.config = config
        self.color_embedding = nn.Embedding(config.num_colors, config.color_embedding)
        self.row_embedding = nn.Embedding(config.grid_size, config.position_embedding)
        self.column_embedding = nn.Embedding(config.grid_size, config.position_embedding)
        input_channels = config.color_embedding
        self.stem = nn.Conv2d(input_channels, config.encoder_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *(
                ResidualBlock(config.encoder_channels, config.normalization_groups)
                for _ in range(config.encoder_resblocks)
            )
        )

    def forward(self, grids: Tensor) -> Tensor:
        if grids.ndim != 3:
            raise ValueError("GridEncoder expects [B,H,W] grids")
        batch_size, height, width = grids.shape
        if (height, width) != (self.config.grid_size, self.config.grid_size):
            raise ValueError("grid size differs from model configuration")
        color = self.color_embedding(grids)
        rows = self.row_embedding(torch.arange(height, device=grids.device))
        columns = self.column_embedding(torch.arange(width, device=grids.device))
        row_map = rows[None, :, None, :].expand(batch_size, height, width, -1)
        column_map = columns[None, None, :, :].expand(batch_size, height, width, -1)
        value = (color + row_map + column_map).permute(0, 3, 1, 2)
        return self.blocks(self.stem(value))


class ActionEncoder(nn.Module):
    """Public-action encoder with mean pooling over optional composite atoms."""

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        atom_dim = max(8, config.action_embedding // 2)
        self.kind_embedding = nn.Embedding(config.num_action_kinds, atom_dim)
        self.row_embedding = nn.Embedding(config.grid_size, atom_dim)
        self.column_embedding = nn.Embedding(config.grid_size, atom_dim)
        self.direction_embedding = nn.Embedding(config.num_directions, atom_dim)
        self.project = nn.Sequential(
            nn.Linear(4 * atom_dim, config.action_embedding),
            nn.SiLU(),
            nn.Linear(config.action_embedding, config.action_embedding),
        )

    def forward(self, actions: Tensor, action_mask: Tensor | None = None) -> Tensor:
        if actions.ndim not in (2, 3) or actions.shape[-1] != ACTION_FIELDS:
            raise ValueError("ActionEncoder expects [B,4] or [B,L,4]")
        atoms = torch.cat(
            (
                self.kind_embedding(actions[..., 0]),
                self.row_embedding(actions[..., 1]),
                self.column_embedding(actions[..., 2]),
                self.direction_embedding(actions[..., 3]),
            ),
            dim=-1,
        )
        atoms = self.project(atoms)
        if actions.ndim == 2:
            if action_mask is not None:
                raise ValueError("single actions cannot have an action mask")
            return atoms
        if action_mask is None or action_mask.shape != actions.shape[:-1]:
            raise ValueError("composite actions require a matching boolean mask")
        weights = action_mask.to(dtype=atoms.dtype).unsqueeze(-1)
        return (atoms * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FiLMResidualBlock(nn.Module):
    """A decoder residual block modulated by a rule-mode/action condition."""

    def __init__(self, channels: int, groups: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film1 = nn.Linear(condition_dim, 2 * channels)
        self.film2 = nn.Linear(condition_dim, 2 * channels)

    @staticmethod
    def _modulate(value: Tensor, parameters: Tensor) -> Tensor:
        scale, shift = parameters.chunk(2, dim=-1)
        return value * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        residual = value
        value = self._modulate(self.norm1(value), self.film1(condition))
        value = self.conv1(F.silu(value))
        value = self._modulate(self.norm2(value), self.film2(condition))
        value = self.conv2(F.silu(value))
        return value + residual


class FiLMDecoder(nn.Module):
    """One shared decoder, called K times through a flattened batch axis."""

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        condition_dim = config.rule_dim + config.action_embedding
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(
                config.encoder_channels,
                config.normalization_groups,
                condition_dim,
            )
            for _ in range(config.decoder_resblocks)
        )
        self.change_head = nn.Conv2d(config.encoder_channels, 1, kernel_size=1)
        self.color_head = nn.Conv2d(config.encoder_channels, config.num_colors, kernel_size=1)

    def forward(self, features: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            features = block(features, condition)
        return self.change_head(features), self.color_head(features)


class EvidenceEncoder(nn.Module):
    """Compress state/observed-target/spatial-residual evidence for a mode."""

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        channels = config.encoder_channels
        self.map = nn.Sequential(
            nn.Conv2d(2 * channels + 1, channels, kernel_size=1),
            nn.GroupNorm(config.normalization_groups, channels),
            nn.SiLU(),
            ResidualBlock(channels, config.normalization_groups),
        )
        self.project = nn.Sequential(
            nn.Linear(channels, config.rule_dim),
            nn.SiLU(),
            nn.Linear(config.rule_dim, config.rule_dim),
        )

    def forward(self, state_features: Tensor, target_features: Tensor, residual: Tensor) -> Tensor:
        # state/target features: [B*K,C,H,W], residual: [B*K,1,H,W]
        value = torch.cat((state_features, target_features - state_features, residual), dim=1)
        value = self.map(value).mean(dim=(-2, -1))
        return self.project(value)


class SetInteraction(nn.Module):
    """One particle-set attention layer after the shared recurrent proposal."""

    def __init__(self, config: NeuralPRPConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.rule_dim)
        self.attention = nn.MultiheadAttention(
            config.rule_dim,
            config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.rule_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.rule_dim, config.attention_ffn),
            nn.SiLU(),
            nn.Linear(config.attention_ffn, config.rule_dim),
        )

    def forward(self, modes: Tensor) -> Tensor:
        normalized = self.norm1(modes)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        modes = modes + attended
        return modes + self.ffn(self.norm2(modes))


class PersistentK4(nn.Module):
    """The minimal persistent K=4 neural PRP world model.

    ``infer_support`` never accesses a hidden program, a candidate probe kind,
    or diagnostic outcomes.  ``losses`` accepts diagnostic outcomes only to
    compute an explicitly training-only objective.
    """

    def __init__(self, config: NeuralPRPConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralPRPConfig()
        self.grid_encoder = GridEncoder(self.config)
        self.action_encoder = ActionEncoder(self.config)
        self.decoder = FiLMDecoder(self.config)
        self.evidence_encoder = EvidenceEncoder(self.config)
        self.action_to_rule = nn.Linear(self.config.action_embedding, self.config.rule_dim)
        self.updater = nn.GRUCell(self.config.rule_dim, self.config.rule_dim)
        self.interaction = SetInteraction(self.config)
        self.initial_modes = nn.Parameter(torch.empty(self.config.particles, self.config.rule_dim))
        nn.init.normal_(self.initial_modes, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> InferenceState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = device or self.initial_modes.device
        modes = self.initial_modes.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        log_weights = torch.full(
            (batch_size, self.config.particles),
            -math.log(self.config.particles),
            device=device,
            dtype=modes.dtype,
        )
        empty = torch.empty((batch_size, 0), dtype=modes.dtype, device=device)
        return InferenceState(modes=modes, log_weights=log_weights, prequential_mixture_log_prob=empty)

    def _predict_from_features(
        self,
        states: Tensor,
        state_features: Tensor,
        actions: Tensor,
        action_mask: Tensor | None,
        modes: Tensor,
    ) -> OutcomePrediction:
        batch_size, channels, height, width = state_features.shape
        if modes.shape != (batch_size, self.config.particles, self.config.rule_dim):
            raise ValueError("modes have incompatible [B,K,D] shape")
        action_features = self.action_encoder(actions, action_mask)
        mode_count = self.config.particles
        features = (
            state_features[:, None, :, :, :]
            .expand(-1, mode_count, -1, -1, -1)
            .reshape(batch_size * mode_count, channels, height, width)
        )
        condition = torch.cat(
            (
                modes,
                action_features[:, None, :].expand(-1, mode_count, -1),
            ),
            dim=-1,
        ).reshape(batch_size * mode_count, -1)
        change, colors = self.decoder(features, condition)
        return OutcomePrediction(
            input_colors=states,
            change_logits=change.reshape(batch_size, mode_count, height, width),
            new_color_logits=colors.reshape(
                batch_size, mode_count, self.config.num_colors, height, width
            ),
        )

    def predict(
        self,
        states: Tensor,
        actions: Tensor,
        modes: Tensor,
        action_mask: Tensor | None = None,
    ) -> OutcomePrediction:
        """Predict next-grid distributions under all persistent modes."""

        _validate_grid_tensor("states", states, self.config, 3)
        _validate_single_actions("actions", actions, action_mask, states.shape[0], self.config)
        return self._predict_from_features(
            states,
            self.grid_encoder(states),
            actions,
            action_mask,
            modes,
        )

    def _update_modes(
        self,
        state_features: Tensor,
        target_features: Tensor,
        prediction: OutcomePrediction,
        target: Tensor,
        actions: Tensor,
        action_mask: Tensor | None,
        modes: Tensor,
    ) -> Tensor:
        batch_size, channels, height, width = state_features.shape
        mode_count = self.config.particles
        residual = (-prediction.log_prob_cells(target)).clamp(min=0.0, max=30.0)
        repeated_state = (
            state_features[:, None]
            .expand(-1, mode_count, -1, -1, -1)
            .reshape(batch_size * mode_count, channels, height, width)
        )
        repeated_target = (
            target_features[:, None]
            .expand(-1, mode_count, -1, -1, -1)
            .reshape(batch_size * mode_count, channels, height, width)
        )
        evidence = self.evidence_encoder(
            repeated_state,
            repeated_target,
            residual.reshape(batch_size * mode_count, 1, height, width),
        )
        action_evidence = self.action_to_rule(self.action_encoder(actions, action_mask))
        proposal = self.updater(
            evidence + action_evidence[:, None, :].expand(-1, mode_count, -1).reshape(batch_size * mode_count, -1),
            modes.reshape(batch_size * mode_count, -1),
        ).reshape(batch_size, mode_count, -1)
        return self.interaction(proposal)

    def _history_evidence(
        self,
        modes: Tensor,
        states: Tensor,
        actions: Tensor,
        targets: Tensor,
        mask: Tensor,
        action_mask: Tensor | None,
    ) -> Tensor:
        """Score all observed support transitions under *current moved* modes."""

        batch_size, steps, height, width = states.shape
        total = torch.zeros(
            batch_size,
            self.config.particles,
            device=states.device,
            dtype=modes.dtype,
        )
        for index in range(steps):
            current_action_mask = action_mask[:, index] if action_mask is not None else None
            prediction = self.predict(
                states[:, index],
                actions[:, index],
                modes,
                current_action_mask,
            )
            total = total + prediction.log_prob(targets[:, index]) * mask[:, index, None]
        return total / float(height * width)

    def _weights_after_history(
        self,
        modes: Tensor,
        states: Tensor,
        actions: Tensor,
        targets: Tensor,
        mask: Tensor,
        action_mask: Tensor | None,
    ) -> Tensor:
        evidence = self._history_evidence(modes, states, actions, targets, mask, action_mask)
        return F.log_softmax(self.config.beta * evidence, dim=-1)

    def infer_support(self, batch: RuleGridTensorBatch) -> InferenceState:
        """Run prequential prediction, recurrent updates, and history-replay weights."""

        batch.validate(self.config)
        states = batch.support_states
        targets = batch.support_targets
        batch_size, steps, height, width = states.shape
        current = self.initial_state(batch_size, device=states.device)
        modes = current.modes
        log_weights = current.log_weights
        prequential: list[Tensor] = []

        for index in range(steps):
            action_mask = (
                batch.support_action_mask[:, index]
                if batch.support_action_mask is not None
                else None
            )
            state = states[:, index]
            target = targets[:, index]
            state_features = self.grid_encoder(state)
            prediction = self._predict_from_features(
                state,
                state_features,
                batch.support_actions[:, index],
                action_mask,
                modes,
            )
            mode_log_prob = prediction.log_prob(target)
            prequential.append(torch.logsumexp(log_weights + mode_log_prob, dim=-1))

            proposed = self._update_modes(
                state_features,
                self.grid_encoder(target),
                prediction,
                target,
                batch.support_actions[:, index],
                action_mask,
                modes,
            )
            valid = batch.support_mask[:, index]
            modes = torch.where(valid[:, None, None], proposed, modes)
            replay_weights = self._weights_after_history(
                modes,
                states[:, : index + 1],
                batch.support_actions[:, : index + 1],
                targets[:, : index + 1],
                batch.support_mask[:, : index + 1],
                (
                    batch.support_action_mask[:, : index + 1]
                    if batch.support_action_mask is not None
                    else None
                ),
            )
            log_weights = torch.where(valid[:, None], replay_weights, log_weights)

        return InferenceState(
            modes=modes,
            log_weights=log_weights,
            prequential_mixture_log_prob=torch.stack(prequential, dim=1),
        )

    def _query_log_probs(
        self,
        inference: InferenceState,
        query_states: Tensor,
        query_actions: Tensor,
        targets: Tensor,
        query_action_mask: Tensor | None,
    ) -> Tensor:
        """Score query targets as ``[B,Q,K]`` without per-query mode matching."""

        batch_size, queries, height, width = query_states.shape
        flat_states = query_states.reshape(batch_size * queries, height, width)
        flat_targets = targets.reshape(batch_size * queries, height, width)
        if query_actions.ndim == 3:
            flat_actions = query_actions.reshape(batch_size * queries, ACTION_FIELDS)
        else:
            atoms = query_actions.shape[2]
            flat_actions = query_actions.reshape(batch_size * queries, atoms, ACTION_FIELDS)
        flat_mask = (
            query_action_mask.reshape(batch_size * queries, -1)
            if query_action_mask is not None
            else None
        )
        flat_modes = (
            inference.modes[:, None]
            .expand(-1, queries, -1, -1)
            .reshape(batch_size * queries, self.config.particles, self.config.rule_dim)
        )
        prediction = self.predict(flat_states, flat_actions, flat_modes, flat_mask)
        return prediction.log_prob(flat_targets).reshape(batch_size, queries, self.config.particles)

    def forward(self, batch: RuleGridTensorBatch) -> NeuralPRPOutput:
        """Infer from support and, if supplied, score public query transitions."""

        inference = self.infer_support(batch)
        if batch.query_targets is None:
            return NeuralPRPOutput(inference=inference)
        assert batch.query_states is not None and batch.query_actions is not None
        return NeuralPRPOutput(
            inference=inference,
            query_log_prob_by_mode=self._query_log_probs(
                inference,
                batch.query_states,
                batch.query_actions,
                batch.query_targets,
                batch.query_action_mask,
            ),
        )

    def _assignment_loss(
        self,
        cost: Tensor,
        weights: Tensor,
        target_mass: Tensor,
    ) -> Tensor:
        """Quality-aware row-hard assignment by exact enumeration (M^K <= 256)."""

        batch_size, modes, classes = cost.shape
        if modes != self.config.particles or classes > modes:
            raise ValueError("assignment cost must be [B,K,M] with M <= K")
        assignments = torch.tensor(
            list(itertools.product(range(classes), repeat=modes)),
            device=cost.device,
            dtype=torch.long,
        )
        assignment_count = assignments.shape[0]
        mode_index = torch.arange(modes, device=cost.device)
        chosen_cost = cost[:, mode_index[None, :], assignments]
        expected_cost = (chosen_cost * weights[:, None, :]).sum(dim=-1)
        assignment_one_hot = F.one_hot(assignments, classes).to(dtype=weights.dtype)
        predicted_mass = torch.einsum("bk,zkm->bzm", weights, assignment_one_hot)
        delta = 1e-4
        smoothed = (predicted_mass + delta) / (1.0 + classes * delta)
        target = target_mass[:, None, :].expand(batch_size, assignment_count, -1)
        kl = (target * (target.clamp_min(delta).log() - smoothed.log())).sum(dim=-1)
        return (expected_cost + 0.25 * kl).amin(dim=1).mean()

    def losses(self, batch: RuleGridTensorBatch) -> LossOutput:
        """Compute the frozen Stage-1-style losses from a tensor batch.

        The returned support loss is strictly prequential: it uses the weights
        before an observed transition was fed into the recurrent updater.
        """

        batch.validate(self.config)
        inference = self.infer_support(batch)
        height = width = self.config.grid_size
        support_denominator = batch.support_mask.sum().clamp_min(1).to(
            dtype=inference.prequential_mixture_log_prob.dtype
        ) * (height * width)
        support_loss = -(
            inference.prequential_mixture_log_prob * batch.support_mask.to(
                dtype=inference.prequential_mixture_log_prob.dtype
            )
        ).sum() / support_denominator

        zero = support_loss.new_zeros(())
        joint_loss = zero
        assignment_loss = zero
        if batch.query_targets is not None:
            assert batch.query_states is not None and batch.query_actions is not None
            query_score = self._query_log_probs(
                inference,
                batch.query_states,
                batch.query_actions,
                batch.query_targets,
                batch.query_action_mask,
            )
            queries = query_score.shape[1]
            joint_log_prob = torch.logsumexp(
                inference.log_weights + query_score.sum(dim=1), dim=-1
            )
            joint_loss = -(joint_log_prob / float(queries * height * width)).mean()

        if batch.behavior_targets is not None:
            assert batch.query_states is not None and batch.query_actions is not None
            assert batch.behavior_mass is not None
            class_scores = []
            for class_index in range(batch.behavior_targets.shape[1]):
                class_score = self._query_log_probs(
                    inference,
                    batch.query_states,
                    batch.query_actions,
                    batch.behavior_targets[:, class_index],
                    batch.query_action_mask,
                )
                class_scores.append(class_score.sum(dim=1))
            # [B,K,M]: each row must explain one complete diagnostic signature.
            cost = -torch.stack(class_scores, dim=-1) / float(
                batch.query_states.shape[1] * height * width
            )
            assignment_loss = self._assignment_loss(
                cost,
                inference.weights,
                batch.behavior_mass.to(dtype=cost.dtype),
            )

        total = support_loss + self.config.lambda_joint * joint_loss + self.config.lambda_assign * assignment_loss
        return LossOutput(
            total=total,
            support=support_loss,
            joint=joint_loss,
            assignment=assignment_loss,
            inference=inference,
        )


def _validate_single_actions(
    name: str,
    actions: Tensor,
    action_mask: Tensor | None,
    batch_size: int,
    config: NeuralPRPConfig,
) -> None:
    if actions.dtype != torch.long or actions.shape[0] != batch_size or actions.shape[-1] != ACTION_FIELDS:
        raise ValueError(f"{name} must have [B,4] or [B,L,4] long shape")
    if actions.ndim == 2:
        if action_mask is not None:
            raise ValueError(f"{name} has no atom axis, so action_mask must be None")
    elif actions.ndim == 3:
        if action_mask is None or action_mask.shape != actions.shape[:-1] or action_mask.dtype != torch.bool:
            raise ValueError(f"{name} composite actions require [B,L] bool action_mask")
    else:
        raise ValueError(f"{name} must have [B,4] or [B,L,4] shape")
    kind, row, column, direction = (actions[..., index] for index in range(ACTION_FIELDS))
    if torch.any(kind < 0) or torch.any(kind >= config.num_action_kinds):
        raise ValueError(f"{name} kind ID is outside vocabulary")
    if torch.any(row < 0) or torch.any(row >= config.grid_size):
        raise ValueError(f"{name} row is outside grid")
    if torch.any(column < 0) or torch.any(column >= config.grid_size):
        raise ValueError(f"{name} column is outside grid")
    if torch.any(direction < 0) or torch.any(direction >= config.num_directions):
        raise ValueError(f"{name} direction ID is outside vocabulary")


def _toy_apply_rule(states: Tensor, actions: Tensor, rule_id: Tensor, num_colors: int) -> Tensor:
    """A deterministic public-action toy transition for smoke tests only."""

    target = states.clone()
    batch_index = torch.arange(states.shape[0], device=states.device)
    row = actions[:, 1]
    column = actions[:, 2]
    old_color = target[batch_index, row, column]
    target[batch_index, row, column] = (old_color + rule_id + 1) % num_colors
    direction = actions[:, 3]
    row_offset = torch.where(direction == 0, -1, torch.where(direction == 1, 1, 0))
    column_offset = torch.where(direction == 2, -1, torch.where(direction == 3, 1, 0))
    neighbor_row = (row + row_offset).clamp(0, states.shape[-2] - 1)
    neighbor_column = (column + column_offset).clamp(0, states.shape[-1] - 1)
    target[batch_index, neighbor_row, neighbor_column] = (
        old_color + 2 * rule_id + 1
    ) % num_colors
    return target


def make_toy_rulegrid_batch(
    *,
    batch_size: int = 8,
    support_steps: int = 3,
    query_count: int = 3,
    config: NeuralPRPConfig | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> RuleGridTensorBatch:
    """Create a tiny hidden-rule batch that exercises the full train path.

    It is not the RuleGrid benchmark and must never be reported as a research
    result.  Its only purpose is CI/GPU verification before the materialized
    RuleGrid data pipeline is available.
    """

    config = config or NeuralPRPConfig()
    if batch_size <= 0 or support_steps <= 0 or query_count <= 0:
        raise ValueError("batch_size, support_steps, and query_count must be positive")
    device = torch.device(device) if device is not None else torch.device("cpu")
    shape_support = (batch_size, support_steps, config.grid_size, config.grid_size)
    shape_query = (batch_size, query_count, config.grid_size, config.grid_size)
    support_states = torch.randint(
        config.num_colors, shape_support, device=device, generator=generator, dtype=torch.long
    )
    query_states = torch.randint(
        config.num_colors, shape_query, device=device, generator=generator, dtype=torch.long
    )
    support_actions = torch.stack(
        (
            torch.randint(config.num_action_kinds, (batch_size, support_steps), device=device, generator=generator),
            torch.randint(config.grid_size, (batch_size, support_steps), device=device, generator=generator),
            torch.randint(config.grid_size, (batch_size, support_steps), device=device, generator=generator),
            torch.randint(config.num_directions, (batch_size, support_steps), device=device, generator=generator),
        ),
        dim=-1,
    ).to(torch.long)
    query_actions = torch.stack(
        (
            torch.randint(config.num_action_kinds, (batch_size, query_count), device=device, generator=generator),
            torch.randint(config.grid_size, (batch_size, query_count), device=device, generator=generator),
            torch.randint(config.grid_size, (batch_size, query_count), device=device, generator=generator),
            torch.randint(config.num_directions, (batch_size, query_count), device=device, generator=generator),
        ),
        dim=-1,
    ).to(torch.long)
    hidden_rule = torch.randint(config.particles, (batch_size,), device=device, generator=generator)
    support_targets = torch.stack(
        [
            _toy_apply_rule(support_states[:, index], support_actions[:, index], hidden_rule, config.num_colors)
            for index in range(support_steps)
        ],
        dim=1,
    )
    query_targets = torch.stack(
        [
            _toy_apply_rule(query_states[:, index], query_actions[:, index], hidden_rule, config.num_colors)
            for index in range(query_count)
        ],
        dim=1,
    )
    behavior_targets = torch.stack(
        [
            torch.stack(
                [
                    _toy_apply_rule(
                        query_states[:, query_index],
                        query_actions[:, query_index],
                        torch.full_like(hidden_rule, rule),
                        config.num_colors,
                    )
                    for query_index in range(query_count)
                ],
                dim=1,
            )
            for rule in range(config.particles)
        ],
        dim=1,
    )
    return RuleGridTensorBatch(
        support_states=support_states,
        support_actions=support_actions,
        support_targets=support_targets,
        support_mask=torch.ones((batch_size, support_steps), dtype=torch.bool, device=device),
        query_states=query_states,
        query_actions=query_actions,
        query_targets=query_targets,
        behavior_targets=behavior_targets,
        behavior_mass=torch.full(
            (batch_size, config.particles),
            1.0 / config.particles,
            dtype=torch.float32,
            device=device,
        ),
    )


def _normalize_diagnostic_indices(
    diagnostic_indices: tuple[int, ...] | list[int] | None,
    diagnostic_count: int,
) -> tuple[int, ...]:
    """Validate an ordered public diagnostic-panel subset.

    This is deliberately local to the RuleGrid adapter rather than inferred
    from a query tensor.  Callers must make a visible, auditable choice of
    which privileged diagnostic targets may be materialized.
    """

    if diagnostic_count <= 0:
        raise ValueError("diagnostic_count must be positive")
    if diagnostic_indices is None:
        return tuple(range(diagnostic_count))
    try:
        normalized = tuple(diagnostic_indices)
    except TypeError as error:
        raise TypeError("diagnostic_indices must be an iterable of integers") from error
    if not normalized:
        raise ValueError("diagnostic_indices cannot be empty")
    if any(type(index) is not int for index in normalized):
        raise TypeError("diagnostic_indices must contain plain integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("diagnostic_indices cannot contain duplicates")
    if any(index < 0 or index >= diagnostic_count for index in normalized):
        raise ValueError(
            f"diagnostic_indices must lie in [0, {diagnostic_count}), got {normalized!r}"
        )
    return normalized


def rulegrid_tasks_to_tensor_batch(
    tasks: tuple[Any, ...] | list[Any],
    *,
    prefix_length: int = 6,
    include_behavior_targets: bool = True,
    diagnostic_indices: tuple[int, ...] | list[int] | None = None,
    device: torch.device | str | None = None,
) -> RuleGridTensorBatch:
    """Adapt materialized ``RuleGridTask`` objects to a neural tensor batch.

    The adapter intentionally has a narrow ownership boundary:

    * support inputs and support targets come from observed public transitions;
    * query inputs come from the public diagnostic panel;
    * true diagnostic targets and all alternative behavior signatures are read
      only when constructing a supervised training batch; and
    * no true program ID, probe kind, candidate kind, or oracle score enters
      the returned model input tensors.

    ``diagnostic_indices`` restricts both public query inputs and privileged
    query targets to an explicitly selected, ordered subset of the diagnostic
    panel.  It is intentionally applied *before* targets or behavior panels
    are materialized.  This is the boundary used by the composition pilot:
    training selects indices ``0..20`` and therefore never reads the triple
    diagnostic targets at ``21..23``.  ``None`` retains the historical
    all-diagnostics behavior.

    ``prefix_length`` is useful for prequential training prefixes from the six
    initial support observations.  The later active-probe prefixes should be
    materialized as additional ``RuleGridTransition`` records by the data
    pipeline and passed through the lower-level :class:`RuleGridTensorBatch`
    constructor; this adapter purposefully does not choose active actions.
    """

    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("at least one RuleGridTask is required")
    if type(prefix_length) is not int or not 1 <= prefix_length <= 6:
        raise ValueError("prefix_length must be an integer in 1..6")
    device = torch.device(device) if device is not None else torch.device("cpu")
    # Local import keeps Stage 0 dependency-free and documents that the neural
    # module is a consumer of RuleGrid, not its owner.
    from .rulegrid import behavior_classes, simulate, version_space

    first = tasks[0]
    diagnostic_count = len(first.inference.diagnostics)
    if not diagnostic_count:
        raise ValueError("RuleGrid tasks need at least one diagnostic probe")
    selected_diagnostic_indices = _normalize_diagnostic_indices(
        diagnostic_indices, diagnostic_count
    )
    support_grids: list[list[Any]] = []
    support_targets: list[list[Any]] = []
    support_actions: list[list[Tensor]] = []
    query_grids: list[list[Any]] = []
    query_actions: list[list[Tensor]] = []
    query_targets: list[list[Any]] = []
    class_panels: list[list[list[Any]]] = []
    class_masses: list[list[float]] = []

    for task in tasks:
        if len(task.inference.support) < prefix_length:
            raise ValueError("task has fewer support transitions than prefix_length")
        if len(task.inference.diagnostics) != diagnostic_count:
            raise ValueError("all tasks in a tensor batch need the same diagnostic count")
        available_target_indices = task.privileged.diagnostic_target_indices
        if available_target_indices is None:  # Defensive; RuleGrid normalizes this in __post_init__.
            raise ValueError("task has no diagnostic target index mapping")
        unavailable = tuple(
            index
            for index in selected_diagnostic_indices
            if index not in available_target_indices
        )
        if unavailable:
            raise ValueError(
                "task did not materialize requested diagnostic target indices "
                f"{unavailable!r}"
            )
        support = task.inference.support[:prefix_length]
        support_grids.append([transition.state for transition in support])
        support_targets.append([transition.next_state for transition in support])
        support_actions.append([encode_public_action(transition.action) for transition in support])
        selected_diagnostics = tuple(
            task.inference.diagnostics[index] for index in selected_diagnostic_indices
        )
        query_grids.append([probe.state for probe in selected_diagnostics])
        query_actions.append([encode_public_action(probe.action) for probe in selected_diagnostics])
        query_targets.append(
            [
                task.privileged.diagnostic_target_for(index)
                for index in selected_diagnostic_indices
            ]
        )

        if include_behavior_targets:
            compatible = version_space(support, task.privileged.palette)
            classes = behavior_classes(
                compatible, selected_diagnostics, task.privileged.palette
            )
            if not classes:
                raise ValueError("support has an empty RuleGrid version space")
            sorted_classes = sorted(classes.items(), key=lambda item: item[0])
            panels: list[list[Any]] = []
            masses: list[float] = []
            for _, programs in sorted_classes:
                representative = programs[0]
                panels.append(
                    [
                        simulate(probe.state, probe.action, representative, task.privileged.palette)
                        for probe in selected_diagnostics
                    ]
                )
                masses.append(len(programs) / len(compatible))
            if len(panels) > 4:
                raise ValueError("the Stage-1 K=4 adapter received more than four behavior classes")
            class_panels.append(panels)
            class_masses.append(masses)

    support_state_tensor = torch.tensor(support_grids, dtype=torch.long, device=device)
    support_target_tensor = torch.tensor(support_targets, dtype=torch.long, device=device)
    query_state_tensor = torch.tensor(query_grids, dtype=torch.long, device=device)
    query_target_tensor = torch.tensor(query_targets, dtype=torch.long, device=device)
    support_action_tensor, support_action_mask = _pad_public_action_grid(support_actions, device)
    query_action_tensor, query_action_mask = _pad_public_action_grid(query_actions, device)

    behavior_target_tensor: Tensor | None = None
    behavior_mass_tensor: Tensor | None = None
    if include_behavior_targets:
        maximum_classes = max(len(panels) for panels in class_panels)
        padded_panels: list[list[list[Any]]] = []
        padded_masses: list[list[float]] = []
        for panels, masses in zip(class_panels, class_masses, strict=True):
            # A zero-mass padding row is legal for the assignment objective;
            # duplicate values are irrelevant because its target mass is zero.
            padded_panels.append(panels + [panels[0]] * (maximum_classes - len(panels)))
            padded_masses.append(masses + [0.0] * (maximum_classes - len(masses)))
        behavior_target_tensor = torch.tensor(padded_panels, dtype=torch.long, device=device)
        behavior_mass_tensor = torch.tensor(padded_masses, dtype=torch.float32, device=device)

    return RuleGridTensorBatch(
        support_states=support_state_tensor,
        support_actions=support_action_tensor,
        support_targets=support_target_tensor,
        support_mask=torch.ones(
            (len(tasks), prefix_length), dtype=torch.bool, device=device
        ),
        query_states=query_state_tensor,
        query_actions=query_action_tensor,
        query_targets=query_target_tensor,
        behavior_targets=behavior_target_tensor,
        behavior_mass=behavior_mass_tensor,
        support_action_mask=support_action_mask,
        query_action_mask=query_action_mask,
    )


def _pad_public_action_grid(
    action_grid: list[list[Tensor]], device: torch.device
) -> tuple[Tensor, Tensor | None]:
    """Pad ``[B][T/Q][L,4]`` public actions to a model tensor and mask."""

    batch_size = len(action_grid)
    if not batch_size or not action_grid[0]:
        raise ValueError("action grid cannot be empty")
    per_task = len(action_grid[0])
    if any(len(actions) != per_task for actions in action_grid):
        raise ValueError("all tasks must contain equal action counts")
    max_atoms = max(action.shape[0] for actions in action_grid for action in actions)
    if max_atoms <= 0:
        raise ValueError("public actions must contain at least one atom")
    padded = torch.zeros(
        (batch_size, per_task, max_atoms, ACTION_FIELDS), dtype=torch.long, device=device
    )
    mask = torch.zeros((batch_size, per_task, max_atoms), dtype=torch.bool, device=device)
    for batch_index, actions in enumerate(action_grid):
        for action_index, action in enumerate(actions):
            atom_count = action.shape[0]
            padded[batch_index, action_index, :atom_count] = action.to(device)
            mask[batch_index, action_index, :atom_count] = True
    if max_atoms == 1:
        return padded[:, :, 0], None
    return padded, mask


_DEFAULT_ACTION_KIND_IDS = {"MOVE": 0, "ACTIVATE": 1, "NOOP": 2}
_DEFAULT_DIRECTION_IDS = {
    "UP": 0,
    "NORTH": 0,
    "N": 0,
    "DOWN": 1,
    "SOUTH": 1,
    "S": 1,
    "LEFT": 2,
    "WEST": 2,
    "W": 2,
    "RIGHT": 3,
    "EAST": 3,
    "E": 3,
    "NONE": 4,
    "NO_DIRECTION": 4,
}


def encode_public_action(action: Any, *, grid_size: int = 8) -> Tensor:
    """Convert a RuleGrid-like public action object into ``[L,4]`` IDs.

    This deliberately relies only on public ``kind``, ``coord``, ``direction``
    and optional ``actions`` attributes.  It accepts the planned
    ``GridAction``/``CompositeAction`` API without importing ``rulegrid`` so the
    neural module remains independently smoke-testable.  Enum integer values
    are accepted directly; string/enum names use the frozen basic mapping.
    """

    atoms = tuple(getattr(action, "actions", (action,)))
    if not atoms:
        raise ValueError("a composite public action cannot be empty")
    encoded: list[list[int]] = []
    for atom in atoms:
        raw_kind = getattr(atom, "kind")
        kind = _public_enum_id(raw_kind, _DEFAULT_ACTION_KIND_IDS, "action kind")
        coord = getattr(atom, "coord")
        if not isinstance(coord, tuple) or len(coord) != 2:
            raise TypeError("public action coord must be a (row, column) tuple")
        row, column = (int(coord[0]), int(coord[1]))
        if not (0 <= row < grid_size and 0 <= column < grid_size):
            raise ValueError("public action coord is outside grid")
        raw_direction = getattr(atom, "direction", None)
        direction = 4 if raw_direction is None else _public_enum_id(
            raw_direction, _DEFAULT_DIRECTION_IDS, "direction"
        )
        encoded.append([kind, row, column, direction])
    return torch.tensor(encoded, dtype=torch.long)


def _public_enum_id(value: Any, mapping: dict[str, int], label: str) -> int:
    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        return raw_value
    name = str(getattr(value, "name", raw_value)).upper()
    if name not in mapping:
        raise ValueError(f"unsupported public {label}: {name!r}")
    return mapping[name]


__all__ = [
    "ACTION_FIELDS",
    "InferenceState",
    "LossOutput",
    "NeuralPRPConfig",
    "NeuralPRPOutput",
    "OutcomePrediction",
    "PersistentK4",
    "RuleGridTensorBatch",
    "encode_public_action",
    "make_toy_rulegrid_batch",
    "rulegrid_tasks_to_tensor_batch",
]
