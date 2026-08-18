"""Explicit latent causal-hypothesis filtering for RuleGrid.

The hypothesis space is the Cartesian product of the three privileged
RuleGrid mechanism axes.  Inference itself uses only public support
transitions: every latent tuple is run through a frozen executor, scored by
how well it explains the observed outcomes, and ranked.  This is a diagnostic
ceiling for hypothesize-and-test reasoning, not autonomous discovery of the
axes from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.causal_filter requires PyTorch") from error

from .latent_rules import (
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
    OracleFactorExecutor,
    outcome_map,
)
from .neural import ACTION_FIELDS, OutcomePrediction


@dataclass(frozen=True)
class HypothesisBankScores:
    """Support evidence for every member of a shared latent hypothesis bank."""

    factor_ids: Tensor  # [H,3]
    proper_nll_per_cell: Tensor  # [B,H]
    balanced_nll_per_cell: Tensor  # [B,H]
    map_error_cells: Tensor  # [B,H]
    map_exact: Tensor  # [B,H]

    @property
    def batch_size(self) -> int:
        return int(self.proper_nll_per_cell.shape[0])

    @property
    def hypotheses(self) -> int:
        return int(self.factor_ids.shape[0])


def enumerate_factor_codes(*, device: torch.device | str | None = None) -> Tensor:
    """Return all 64 collision/trigger/relation tuples in stable order."""

    values = torch.arange(
        RULE_FACTOR_CARDINALITY,
        dtype=torch.long,
        device=device,
    )
    return torch.cartesian_prod(*(values for _ in range(RULE_FACTOR_COUNT)))


def _expanded_actions(actions: Tensor, hypotheses: int) -> Tensor:
    if actions.ndim == 3:
        batch_size, panel_count, fields = actions.shape
        if fields != ACTION_FIELDS:
            raise ValueError("actions must end in the public action fields")
        return (
            actions[:, :, None]
            .expand(-1, -1, hypotheses, -1)
            .reshape(batch_size * panel_count * hypotheses, fields)
        )
    if actions.ndim == 4:
        batch_size, panel_count, atoms, fields = actions.shape
        if fields != ACTION_FIELDS:
            raise ValueError("actions must end in the public action fields")
        return (
            actions[:, :, None]
            .expand(-1, -1, hypotheses, -1, -1)
            .reshape(batch_size * panel_count * hypotheses, atoms, fields)
        )
    raise ValueError("actions must have [B,P,4] or [B,P,L,4] shape")


def _expanded_action_mask(
    action_mask: Tensor | None,
    hypotheses: int,
) -> Tensor | None:
    if action_mask is None:
        return None
    batch_size, panel_count, atoms = action_mask.shape
    return (
        action_mask[:, :, None]
        .expand(-1, -1, hypotheses, -1)
        .reshape(batch_size * panel_count * hypotheses, atoms)
    )


def predict_factor_panel(
    executor: OracleFactorExecutor,
    states: Tensor,
    actions: Tensor,
    factor_ids: Tensor,
    action_mask: Tensor | None = None,
) -> OutcomePrediction:
    """Predict one panel under a different persistent code for every mode.

    ``states`` is ``[B,P,H,W]`` and ``factor_ids`` is ``[B,K,3]``.  The
    returned prediction uses the conventional flattened panel batch
    ``[B*P,K,...]`` so that one code remains persistent across all P probes.
    """

    if states.ndim != 4:
        raise ValueError("states must have [B,P,H,W] shape")
    if factor_ids.ndim != 3 or factor_ids.shape[0] != states.shape[0] or factor_ids.shape[2] != RULE_FACTOR_COUNT:
        raise ValueError("factor_ids must have [B,K,3] shape")
    batch_size, panel_count, height, width = states.shape
    hypotheses = factor_ids.shape[1]
    if actions.shape[:2] != (batch_size, panel_count):
        raise ValueError("states and actions must share [B,P]")
    repeated_states = (
        states[:, :, None]
        .expand(-1, -1, hypotheses, -1, -1)
        .reshape(batch_size * panel_count * hypotheses, height, width)
    )
    repeated_codes = (
        factor_ids[:, None]
        .expand(-1, panel_count, -1, -1)
        .reshape(batch_size * panel_count * hypotheses, RULE_FACTOR_COUNT)
    )
    raw = executor.predict(
        repeated_states,
        _expanded_actions(actions, hypotheses),
        repeated_codes,
        _expanded_action_mask(action_mask, hypotheses),
    )
    return OutcomePrediction(
        input_colors=states.reshape(batch_size * panel_count, height, width),
        change_logits=raw.change_logits.reshape(
            batch_size * panel_count, hypotheses, height, width
        ),
        new_color_logits=raw.new_color_logits.reshape(
            batch_size * panel_count,
            hypotheses,
            executor.config.num_colors,
            height,
            width,
        ),
    )


def score_hypothesis_bank(
    executor: OracleFactorExecutor,
    states: Tensor,
    actions: Tensor,
    targets: Tensor,
    support_mask: Tensor,
    action_mask: Tensor | None = None,
    *,
    factor_ids: Tensor | None = None,
) -> HypothesisBankScores:
    """Score all latent rules using only observed support transitions."""

    if states.shape != targets.shape or states.ndim != 4:
        raise ValueError("states and targets must share [B,T,H,W]")
    if support_mask.shape != states.shape[:2] or support_mask.dtype != torch.bool:
        raise ValueError("support_mask must be boolean [B,T]")
    bank = (
        enumerate_factor_codes(device=states.device)
        if factor_ids is None
        else factor_ids
    )
    if bank.ndim != 2 or bank.shape[1] != RULE_FACTOR_COUNT:
        raise ValueError("factor_ids bank must have [H,3] shape")
    batch_size, steps, height, width = states.shape
    hypotheses = bank.shape[0]
    task_codes = bank[None].expand(batch_size, -1, -1)
    prediction = predict_factor_panel(
        executor,
        states,
        actions,
        task_codes,
        action_mask,
    )
    flat_targets = targets.reshape(batch_size * steps, height, width)
    cell_nll = -prediction.log_prob_cells(flat_targets).reshape(
        batch_size, steps, hypotheses, height, width
    )
    valid = support_mask[:, :, None, None, None]
    valid_cells = support_mask.sum(dim=1, keepdim=True).clamp_min(1) * height * width
    proper = (cell_nll * valid).sum(dim=(1, 3, 4)) / valid_cells

    changed = targets.ne(states)[:, :, None] & valid
    unchanged = ~targets.ne(states)[:, :, None] & valid
    changed_count = changed.sum(dim=(1, 3, 4)).clamp_min(1)
    unchanged_count = unchanged.sum(dim=(1, 3, 4)).clamp_min(1)
    changed_nll = (cell_nll * changed).sum(dim=(1, 3, 4)) / changed_count
    unchanged_nll = (cell_nll * unchanged).sum(dim=(1, 3, 4)) / unchanged_count
    balanced = 0.5 * (changed_nll + unchanged_nll)

    maps = outcome_map(prediction).reshape(
        batch_size, steps, hypotheses, height, width
    )
    wrong = maps.ne(targets[:, :, None]) & valid
    map_error_cells = wrong.sum(dim=(1, 3, 4))
    map_exact = ~wrong.any(dim=(1, 3, 4))
    return HypothesisBankScores(
        factor_ids=bank,
        proper_nll_per_cell=proper,
        balanced_nll_per_cell=balanced,
        map_error_cells=map_error_cells,
        map_exact=map_exact,
    )


def select_hypotheses(
    scores: HypothesisBankScores,
    *,
    particles: int = 4,
    method: str = "map_then_balanced_nll",
) -> Tensor:
    """Return distinct hypothesis-bank indices ordered best first."""

    if not 1 <= particles <= scores.hypotheses:
        raise ValueError("particles must lie within the hypothesis bank")
    if method == "proper_nll":
        order = torch.argsort(
            scores.proper_nll_per_cell, dim=1, stable=True
        )
    elif method == "balanced_nll":
        order = torch.argsort(
            scores.balanced_nll_per_cell, dim=1, stable=True
        )
    elif method == "map_then_balanced_nll":
        # Stable two-pass sorting implements an exact lexicographic order:
        # hard causal consistency first, calibrated evidence as the tie-break.
        nll_order = torch.argsort(
            scores.balanced_nll_per_cell, dim=1, stable=True
        )
        errors_in_nll_order = scores.map_error_cells.gather(1, nll_order)
        error_order = torch.argsort(
            errors_in_nll_order, dim=1, stable=True
        )
        order = nll_order.gather(1, error_order)
    else:
        raise ValueError(f"unknown hypothesis selection method: {method}")
    return order[:, :particles]


def selected_factor_ids(
    scores: HypothesisBankScores,
    selected_indices: Tensor,
) -> Tensor:
    """Materialize ``[B,K,3]`` persistent codes from bank indices."""

    if selected_indices.ndim != 2 or selected_indices.shape[0] != scores.batch_size:
        raise ValueError("selected_indices must have [B,K] shape")
    return scores.factor_ids[selected_indices]


__all__ = [
    "HypothesisBankScores",
    "enumerate_factor_codes",
    "predict_factor_panel",
    "score_hypothesis_bank",
    "select_hypotheses",
    "selected_factor_ids",
]
