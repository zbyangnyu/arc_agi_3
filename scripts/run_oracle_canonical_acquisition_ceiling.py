#!/usr/bin/env python3
"""Privileged oracle-canonical ceiling for model-directed active acquisition.

This runner deliberately isolates one question: if a public rule-belief model
proposes an initial distribution over the known 64 RuleGrid factor codes, can
an audited oracle-palette executor use that distribution to choose useful
experiments?

The controller receives an explicit door axis, joint code weights, canonical
candidate state/action tensors, and executor-predicted candidate outcomes.  It
does not receive task/probe IDs, candidate kinds, the hidden program, or any
candidate target.  After a candidate index has been selected, the environment
reveals only that candidate's canonical next state.  The posterior is then
updated with the frozen executor's full-grid log likelihood.

Palette canonicalization, the fixed 3x4 codebook, the door axis, and executor
pretraining are privileged.  Results are therefore an acquisition ceiling,
not a public-only ARC-AGI-3 result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


RESULT_SCHEMA_VERSION = "prp-wm.oracle-canonical-acquisition-ceiling.v1"
ACTIVE_EXECUTOR_SCHEMA_VERSION = (
    "prp-wm.active-support-calibrated-factor-executor.v1"
)
COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION = (
    "prp-wm.counterfactual-locality-finetune.v3"
)
SUPPORTED_ACTIVE_EXECUTOR_SCHEMA_VERSIONS = frozenset(
    {
        ACTIVE_EXECUTOR_SCHEMA_VERSION,
        COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION,
    }
)
DEFAULT_BELIEF_CHECKPOINT = REPOSITORY_ROOT / (
    "runs/public_belief_mixed_sequence_ft1000_fold0_seed20260872/"
    "checkpoint_last.pt"
)
DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT = REPOSITORY_ROOT / (
    "runs/active_support_calibrated_executor_cont300_seed20260731/"
    "checkpoint_last.pt"
)
DEFAULT_BUDGETS = (0, 1, 2, 4, 8)
POLICIES = ("expected-door-gain", "uniform")
_AUDITED_SOURCE_FILES = (
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/matched_executor.py",
    "prp_wm/neural.py",
    "prp_wm/public_version_k4.py",
    "prp_wm/routed_executor.py",
    "prp_wm/rulegrid.py",
    "scripts/run_active_support_calibrated_executor.py",
    "scripts/run_gram_public_coverage_finetune.py",
    "scripts/run_oracle_canonical_acquisition_ceiling.py",
    "scripts/run_public_version_space_k4.py",
)


@dataclass(frozen=True)
class AcquisitionScore:
    """One candidate's deterministic predicted-partition utility."""

    candidate_index: int
    expected_door_gain: float
    outcome_information_gain_nats: float
    predicted_outcome_classes: int


@dataclass(frozen=True)
class PublicDoorQuery:
    """Explicit controller query; never infer the axis from a task identifier."""

    axis_index: int

    def __post_init__(self) -> None:
        if type(self.axis_index) is not int or self.axis_index not in range(3):
            raise ValueError("door axis must be an integer in [0,3)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--belief-checkpoint",
        type=Path,
        default=DEFAULT_BELIEF_CHECKPOINT,
    )
    parser.add_argument(
        "--active-executor-checkpoint",
        type=Path,
        default=DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-axis", type=int, default=64)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trace-tasks-per-axis", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260872)
    parser.add_argument("--data-master-seed", type=int, default=2026071601)
    parser.add_argument(
        "--split",
        default="oracle-canonical-acquisition-ceiling",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.groups_per_axis <= 0:
        raise SystemExit("--groups-per-axis must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.trace_tasks_per_axis < 0:
        raise SystemExit("--trace-tasks-per-axis must be non-negative")
    if args.seed < 0 or args.data_master_seed < 0:
        raise SystemExit("seeds must be non-negative")
    if not args.split or "/" in args.split:
        raise SystemExit("--split must be a non-empty slash-free name")
    budgets = tuple(int(value) for value in args.budgets)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(value < 0 or value > 8 for value in budgets)
    ):
        raise SystemExit("--budgets must be unique integers in [0,8]")
    return tuple(sorted(budgets))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    return {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in _AUDITED_SOURCE_FILES
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalise_log_weights(torch: Any, log_weights: Any) -> Any:
    if log_weights.ndim < 1 or log_weights.shape[-1] <= 0:
        raise ValueError("log weights need a non-empty hypothesis axis")
    if not bool(torch.isfinite(log_weights).any(dim=-1).all().item()):
        raise ValueError("every posterior row needs finite probability mass")
    return log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)


def factor_marginals_to_joint_log_weights(
    torch: Any,
    factor_probabilities: Any,
    factor_bank: Any,
) -> Any:
    """Expand public ``[B,3,4]`` marginals into a normalized 64-code joint."""

    if factor_probabilities.ndim != 3 or factor_probabilities.shape[1:] != (3, 4):
        raise ValueError("factor probabilities must have [B,3,4] shape")
    if factor_bank.ndim != 2 or factor_bank.shape[1] != 3:
        raise ValueError("factor bank must have [H,3] shape")
    bank = factor_bank.to(device=factor_probabilities.device)
    if bank.dtype != torch.long:
        raise TypeError("factor bank must use torch.long dtype")
    if bool(((bank < 0) | (bank >= 4)).any().item()):
        raise ValueError("factor bank values must lie in [0,4)")
    probabilities = factor_probabilities.clamp_min(
        torch.finfo(factor_probabilities.dtype).tiny
    )
    selected = tuple(
        probabilities[:, axis].index_select(1, bank[:, axis])
        for axis in range(3)
    )
    return _normalise_log_weights(
        torch,
        torch.stack(selected, dim=0).log().sum(dim=0),
    )


def bayesian_log_likelihood_update(
    torch: Any,
    log_weights: Any,
    log_likelihood: Any,
) -> tuple[Any, Any]:
    """Apply one explicit Bayes update and return posterior plus log evidence."""

    if log_weights.shape != log_likelihood.shape:
        raise ValueError("prior and likelihood must have the same shape")
    prior = _normalise_log_weights(torch, log_weights)
    unnormalised = prior + log_likelihood
    log_evidence = torch.logsumexp(unnormalised, dim=-1)
    if not bool(torch.isfinite(log_evidence).all().item()):
        raise ValueError("executor likelihood assigned zero mass to an observation")
    return (
        unnormalised - log_evidence[..., None],
        log_evidence,
    )


def _door_marginals(
    torch: Any,
    weights: Any,
    door_values: Any,
) -> Any:
    if weights.ndim != 1 or door_values.shape != weights.shape:
        raise ValueError("weights and door values must share one hypothesis axis")
    result = torch.zeros(4, dtype=weights.dtype, device=weights.device)
    result = result.scatter_add(0, door_values, weights)
    return result / result.sum()


def _acquisition_score(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    predicted_outcomes: Any,
    *,
    candidate_index: int,
) -> AcquisitionScore:
    """Score a candidate by expected terminal door value, then outcome EIG."""

    if predicted_outcomes.ndim != 3:
        raise ValueError("candidate outcomes must have [H,H,W] shape")
    if predicted_outcomes.shape[0] != log_weights.shape[0]:
        raise ValueError("candidate outcomes must cover every hypothesis")
    posterior = _normalise_log_weights(torch, log_weights)
    weights = posterior.exp()
    weights = weights / weights.sum()
    flattened = predicted_outcomes.reshape(predicted_outcomes.shape[0], -1)
    _, inverse = torch.unique(
        flattened,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    classes = int(inverse.max().item()) + 1
    class_door_mass = torch.zeros(
        (classes, 4),
        dtype=weights.dtype,
        device=weights.device,
    )
    flat_index = inverse * 4 + door_values
    class_door_mass.view(-1).scatter_add_(0, flat_index, weights)
    class_mass = class_door_mass.sum(dim=-1)
    current_best_door_mass = _door_marginals(
        torch,
        weights,
        door_values,
    ).max()
    expected_posterior_best_mass = class_door_mass.max(dim=-1).values.sum()
    expected_door_gain = (
        expected_posterior_best_mass - current_best_door_mass
    ).clamp_min(0.0)
    nonzero = class_mass > 0
    outcome_information_gain = (
        -(
            class_mass[nonzero] * class_mass[nonzero].log()
        ).sum()
    ).clamp_min(0.0)
    return AcquisitionScore(
        candidate_index=candidate_index,
        expected_door_gain=float(expected_door_gain),
        outcome_information_gain_nats=float(outcome_information_gain),
        predicted_outcome_classes=classes,
    )


def _select_acquisition_candidate(
    torch: Any,
    log_weights: Any,
    door_values: Any,
    candidate_outcomes: Any,
    available: Any,
) -> AcquisitionScore:
    """Pure tensor selector with no identifier, kind, or feedback argument."""

    if candidate_outcomes.ndim != 4:
        raise ValueError("candidate outcomes must have [P,H,H,W] shape")
    if available.shape != candidate_outcomes.shape[:1] or available.dtype != torch.bool:
        raise ValueError("available must be a boolean candidate mask")
    candidates = []
    for candidate_index in range(candidate_outcomes.shape[0]):
        if bool(available[candidate_index].item()):
            candidates.append(
                _acquisition_score(
                    torch,
                    log_weights,
                    door_values,
                    candidate_outcomes[candidate_index],
                    candidate_index=candidate_index,
                )
            )
    if not candidates:
        raise ValueError("at least one candidate must remain available")
    # Public bank position is the final deterministic tie breaker.  The bank
    # order is nuisance-randomized independently of the hidden factor value.
    return min(
        candidates,
        key=lambda item: (
            -item.expected_door_gain,
            -item.outcome_information_gain_nats,
            item.candidate_index,
        ),
    )


def _selector_boundary_audit() -> dict[str, object]:
    """Mechanically record the arguments and forbidden-name source audit."""

    signature = inspect.signature(_select_acquisition_candidate)
    parameters = tuple(signature.parameters)
    expected = (
        "torch",
        "log_weights",
        "door_values",
        "candidate_outcomes",
        "available",
    )
    forbidden = (
        "task_id",
        "probe_id",
        "candidate_kind",
        "active_target",
        "true_program",
    )
    source = inspect.getsource(_select_acquisition_candidate)
    checks = {
        "tensor_only_signature": parameters == expected,
        "forbidden_names_absent": all(name not in source for name in forbidden),
    }
    return {
        "parameters": list(parameters),
        "forbidden_names": list(forbidden),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_active_executor_artifact(
    *,
    checkpoint: dict[str, object],
    result: dict[str, object],
    checkpoint_sha256: str,
) -> None:
    """Accept only audited base executors or their explicit v3 continuation.

    The locality finetune does not change the model architecture, but it does
    use a distinct checkpoint schema.  Supporting it by a bare schema
    allow-list would make stale or unrelated artifacts easy to pass into the
    downstream audits, so the continuation additionally has to identify an
    audited active-support parent.  Both schemas must retain the original
    active-prefix gate and the result/checkpoint checksum binding.
    """

    schema = checkpoint.get("checkpoint_schema_version")
    if schema not in SUPPORTED_ACTIVE_EXECUTOR_SCHEMA_VERSIONS:
        raise SystemExit("active executor has an unexpected checkpoint schema")
    model_type = checkpoint.get("model_type")
    supported_model_types = {
        "OracleFactorExecutor",
        "CanonicalRoleRoutedOracleFactorExecutor",
        "MatchedWiderGlobalOracleFactorExecutor",
        "MatchedFactorLocalOracleFactorExecutor",
    }
    if model_type not in supported_model_types:
        raise SystemExit("active executor has an unexpected model type")
    if result.get("model_type") != model_type:
        raise SystemExit("active executor result/checkpoint model type mismatch")
    if result.get("checkpoint_schema_version") != schema:
        raise SystemExit("active executor result/checkpoint schema mismatch")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise SystemExit("active executor result/checkpoint checksum mismatch")
    if result.get("active_prefix_executor_gate", {}).get("passed") is not True:
        raise SystemExit("active executor did not pass its active-prefix gate")

    if schema == COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION:
        checkpoint_parent = checkpoint.get("initial_checkpoint_provenance")
        result_parent = result.get("initial_checkpoint_provenance")
        if (
            not isinstance(checkpoint_parent, dict)
            or checkpoint_parent != result_parent
            or checkpoint_parent.get("schema_version")
            != ACTIVE_EXECUTOR_SCHEMA_VERSION
            or not isinstance(checkpoint_parent.get("sha256"), str)
            or len(checkpoint_parent["sha256"]) != 64
        ):
            raise SystemExit(
                "counterfactual locality executor lacks audited parent lineage"
            )
    elif model_type != "OracleFactorExecutor":
        raise SystemExit(
            "experimental executors are accepted only as audited locality "
            "continuations"
        )


def _load_active_executor(
    torch: Any,
    checkpoint_path: Path,
    device: Any,
) -> tuple[Any, dict[str, object], dict[str, object]]:
    from scripts.run_causal_mechanism_coverage import _load_executor

    resolved = checkpoint_path.resolve()
    executor, checkpoint = _load_executor(torch, resolved, device)
    result_path = resolved.parent / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate_active_executor_artifact(
        checkpoint=checkpoint,
        result=result,
        checkpoint_sha256=_sha256_file(resolved),
    )
    return executor, checkpoint, result


def _build_axis_tasks(
    axis: Any,
    *,
    groups: int,
    split: str,
    master_seed: int,
) -> tuple[Any, ...]:
    """Balanced evaluator tasks; the query axis is supplied separately."""

    from scripts.run_public_belief_door_game import _build_axis_tasks as build

    return build(
        axis,
        groups=groups,
        split=split,
        master_seed=master_seed,
    )


def _uniform_orders(
    *,
    torch: Any,
    axis_index: int,
    groups: int,
    candidates: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """One nuisance-only order per group, shared across its four hidden values."""

    if groups <= 0 or candidates <= 0:
        raise ValueError("groups and candidates must be positive")
    result = []
    for group_index in range(groups):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            seed + 1_000_003 * axis_index + 10_007 * group_index
        )
        order = tuple(
            int(value)
            for value in torch.randperm(candidates, generator=generator).tolist()
        )
        result.extend((order,) * 4)
    return tuple(result)


def _initial_joint_belief(
    *,
    torch: Any,
    model: Any,
    tasks: Sequence[Any],
    device: Any,
) -> Any:
    from scripts.run_gram_public_coverage_finetune import (
        _raw_public_history_batch,
    )

    histories = tuple(task.inference.support for task in tasks)
    public = _raw_public_history_batch(torch, histories, device=device)
    with torch.no_grad():
        belief = model.infer_factor_belief(public)
    return factor_marginals_to_joint_log_weights(
        torch,
        belief.factor_probabilities,
        model.factor_bank,
    ).detach().cpu()


def _candidate_panel(
    *,
    torch: Any,
    tasks: Sequence[Any],
    device: Any,
) -> tuple[Any, Any, Any | None, Any]:
    """Materialize canonical candidate inputs and environment feedback sidecar."""

    from prp_wm.neural import encode_public_action
    from scripts.run_active_support_calibrated_executor import (
        _canonicalize_grid_tensor,
    )
    from scripts.run_gram_public_coverage_finetune import _pad_public_actions

    materialized = tuple(tasks)
    if not materialized:
        raise ValueError("tasks cannot be empty")
    candidate_count = len(materialized[0].inference.active_candidates)
    if candidate_count <= 0 or any(
        len(task.inference.active_candidates) != candidate_count
        for task in materialized
    ):
        raise ValueError("all tasks must share a positive candidate count")
    raw_states = torch.tensor(
        [
            [candidate.state for candidate in task.inference.active_candidates]
            for task in materialized
        ],
        dtype=torch.long,
        device=device,
    )
    action_rows = tuple(
        tuple(
            encode_public_action(candidate.action)
            for candidate in task.inference.active_candidates
        )
        for task in materialized
    )
    actions, action_mask = _pad_public_actions(
        torch,
        action_rows,
        device=device,
    )
    # This sidecar belongs to the environment.  It is indexed only after the
    # pure selector returns a candidate index and is never passed to selection.
    raw_feedback = torch.tensor(
        [task.privileged.active_targets for task in materialized],
        dtype=torch.long,
        device=device,
    )
    return (
        _canonicalize_grid_tensor(torch, raw_states, materialized),
        actions,
        action_mask,
        _canonicalize_grid_tensor(torch, raw_feedback, materialized),
    )


def _selected_prediction(
    *,
    torch: Any,
    prediction: Any,
    candidate_indices: Sequence[int],
    candidate_count: int,
) -> Any:
    from prp_wm.neural import OutcomePrediction

    device = prediction.change_logits.device
    rows = (
        torch.arange(len(candidate_indices), device=device) * candidate_count
        + torch.tensor(candidate_indices, dtype=torch.long, device=device)
    )
    return OutcomePrediction(
        input_colors=prediction.input_colors.index_select(0, rows),
        change_logits=prediction.change_logits.index_select(0, rows),
        new_color_logits=prediction.new_color_logits.index_select(0, rows),
    )


def _symbolic_door_identified(
    task: Any,
    history: Sequence[Any],
    query: PublicDoorQuery,
) -> bool:
    """Audit only; this value never controls action selection or stopping."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space

    values = {
        rule_program_factor_ids(program)[query.axis_index]
        for program in version_space(history, task.privileged.palette)
    }
    return len(values) == 1


def _task_metrics(
    *,
    torch: Any,
    log_weights: Any,
    factor_bank: Any,
    target_doors: Any,
    first_identified_steps: Sequence[int | None],
    budget: int,
    selected_scores: Sequence[Sequence[AcquisitionScore]],
    predictive_log_probabilities: Sequence[Sequence[float]],
) -> list[dict[str, object]]:
    weights = _normalise_log_weights(torch, log_weights).exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    records = []
    for task_index in range(weights.shape[0]):
        records.append(
            {
                "_weights": weights[task_index],
                "_target_door": int(target_doors[task_index]),
                "first_symbolic_identification_step": first_identified_steps[
                    task_index
                ],
                "selected_scores": tuple(selected_scores[task_index][:budget]),
                "predictive_log_probabilities": tuple(
                    predictive_log_probabilities[task_index][:budget]
                ),
            }
        )
    return records


def _finalise_task_records(
    *,
    torch: Any,
    partial: Sequence[dict[str, object]],
    factor_bank: Any,
    query: PublicDoorQuery,
) -> list[dict[str, object]]:
    door_values = factor_bank[:, query.axis_index]
    result = []
    for item in partial:
        weights = item["_weights"]
        weights = weights / weights.sum()
        target = int(item["_target_door"])
        door_probabilities = _door_marginals(torch, weights, door_values)
        door_probabilities = door_probabilities.clamp(0.0, 1.0)
        door_probabilities = door_probabilities / door_probabilities.sum()
        entropy = (
            -(
                door_probabilities.clamp_min(1e-12)
                * door_probabilities.clamp_min(1e-12).log()
            ).sum()
        ).clamp_min(0.0)
        joint_entropy = (
            -(
                weights.clamp_min(1e-12)
                * weights.clamp_min(1e-12).log()
            ).sum()
        ).clamp_min(0.0)
        scores = item["selected_scores"]
        log_probabilities = item["predictive_log_probabilities"]
        result.append(
            {
                "won": int(door_probabilities.argmax().item()) == target,
                "true_door_probability": float(door_probabilities[target]),
                "confidence": float(door_probabilities.max()),
                "door_entropy_nats": float(entropy),
                "joint_effective_hypotheses": float(joint_entropy.exp()),
                "first_symbolic_identification_step": item[
                    "first_symbolic_identification_step"
                ],
                "cumulative_expected_door_gain": sum(
                    score.expected_door_gain for score in scores
                ),
                "cumulative_outcome_information_gain_nats": sum(
                    score.outcome_information_gain_nats for score in scores
                ),
                "mean_observed_log_predictive_probability": (
                    sum(log_probabilities) / len(log_probabilities)
                    if log_probabilities
                    else None
                ),
            }
        )
    return result


def _rollout_policy(
    *,
    torch: Any,
    policy: str,
    tasks: Sequence[Any],
    query: PublicDoorQuery,
    initial_log_weights: Any,
    factor_bank: Any,
    outcome_maps: Any,
    prediction: Any,
    canonical_feedback: Any,
    uniform_orders: Sequence[Sequence[int]],
    budgets: Sequence[int],
) -> tuple[dict[int, list[dict[str, object]]], list[dict[str, object]]]:
    """Play fixed-budget trajectories; feedback is read only after selection."""

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import RuleGridTransition

    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    materialized = tuple(tasks)
    batch_size, candidate_count, hypotheses = outcome_maps.shape[:3]
    if batch_size != len(materialized):
        raise ValueError("prediction batch must match tasks")
    if initial_log_weights.shape != (batch_size, hypotheses):
        raise ValueError("initial posterior has the wrong shape")
    max_budget = max(budgets)
    log_weights = initial_log_weights.clone()
    available = torch.ones(
        (batch_size, candidate_count),
        dtype=torch.bool,
    )
    histories = [list(task.inference.support) for task in materialized]
    first_identified: list[int | None] = [None] * batch_size
    selected_scores: list[list[AcquisitionScore]] = [
        [] for _ in materialized
    ]
    selected_indices: list[list[int]] = [[] for _ in materialized]
    predictive_log_probabilities: list[list[float]] = [
        [] for _ in materialized
    ]
    target_doors = torch.tensor(
        [
            rule_program_factor_ids(task.privileged.true_program)[
                query.axis_index
            ]
            for task in materialized
        ],
        dtype=torch.long,
    )
    door_values = factor_bank[:, query.axis_index]
    snapshots: dict[int, list[dict[str, object]]] = {}

    def snapshot(budget: int) -> None:
        partial = _task_metrics(
            torch=torch,
            log_weights=log_weights,
            factor_bank=factor_bank,
            target_doors=target_doors,
            first_identified_steps=first_identified,
            budget=budget,
            selected_scores=selected_scores,
            predictive_log_probabilities=predictive_log_probabilities,
        )
        snapshots[budget] = _finalise_task_records(
            torch=torch,
            partial=partial,
            factor_bank=factor_bank,
            query=query,
        )

    if 0 in budgets:
        snapshot(0)
    for step in range(1, max_budget + 1):
        choices: list[int] = []
        step_scores: list[AcquisitionScore] = []
        for task_index in range(batch_size):
            if policy == "expected-door-gain":
                score = _select_acquisition_candidate(
                    torch,
                    log_weights[task_index],
                    door_values,
                    outcome_maps[task_index],
                    available[task_index],
                )
            else:
                candidate_index = int(uniform_orders[task_index][step - 1])
                if not bool(available[task_index, candidate_index].item()):
                    raise AssertionError("uniform order repeated a candidate")
                score = _acquisition_score(
                    torch,
                    log_weights[task_index],
                    door_values,
                    outcome_maps[task_index, candidate_index],
                    candidate_index=candidate_index,
                )
            choices.append(score.candidate_index)
            step_scores.append(score)
        for task_index, candidate_index in enumerate(choices):
            available[task_index, candidate_index] = False
            selected_scores[task_index].append(step_scores[task_index])
            selected_indices[task_index].append(candidate_index)

        # Environment reveal starts here, after every policy choice is frozen.
        feedback_rows = canonical_feedback[
            torch.arange(batch_size, device=canonical_feedback.device),
            torch.tensor(
                choices,
                dtype=torch.long,
                device=canonical_feedback.device,
            ),
        ]
        selected_prediction = _selected_prediction(
            torch=torch,
            prediction=prediction,
            candidate_indices=choices,
            candidate_count=candidate_count,
        )
        with torch.no_grad():
            likelihood = selected_prediction.log_prob(feedback_rows).cpu()
        updated, log_evidence = bayesian_log_likelihood_update(
            torch,
            log_weights,
            likelihood,
        )
        log_weights = updated
        for task_index, candidate_index in enumerate(choices):
            predictive_log_probabilities[task_index].append(
                float(log_evidence[task_index])
            )
            probe = materialized[
                task_index
            ].inference.active_candidates[candidate_index]
            observed = materialized[
                task_index
            ].privileged.active_targets[candidate_index]
            histories[task_index].append(
                RuleGridTransition(probe.state, probe.action, observed)
            )
            if (
                first_identified[task_index] is None
                and _symbolic_door_identified(
                    materialized[task_index],
                    histories[task_index],
                    query,
                )
            ):
                first_identified[task_index] = step
        if step in budgets:
            snapshot(step)

    traces = [
        {
            "task_batch_index": task_index,
            "selected_candidate_indices": selected_indices[task_index],
            "first_symbolic_identification_step": first_identified[task_index],
            "step_scores": [
                {
                    "candidate_index": score.candidate_index,
                    "expected_door_gain": score.expected_door_gain,
                    "outcome_information_gain_nats": (
                        score.outcome_information_gain_nats
                    ),
                    "predicted_outcome_classes": score.predicted_outcome_classes,
                }
                for score in selected_scores[task_index]
            ],
        }
        for task_index in range(batch_size)
    ]
    return snapshots, traces


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _summarise_policy(
    records: Sequence[dict[str, object]],
    *,
    budget: int,
) -> dict[str, object]:
    identified_steps = tuple(
        int(item["first_symbolic_identification_step"])
        for item in records
        if item["first_symbolic_identification_step"] is not None
        and int(item["first_symbolic_identification_step"]) <= budget
    )
    observed_log_probabilities = tuple(
        float(item["mean_observed_log_predictive_probability"])
        for item in records
        if item["mean_observed_log_predictive_probability"] is not None
    )
    return {
        "tasks": len(records),
        "terminal_win_rate": _mean(int(bool(item["won"])) for item in records),
        "mean_true_door_probability": _mean(
            float(item["true_door_probability"]) for item in records
        ),
        "mean_confidence": _mean(
            float(item["confidence"]) for item in records
        ),
        "mean_door_entropy_nats": _mean(
            float(item["door_entropy_nats"]) for item in records
        ),
        "mean_joint_effective_hypotheses": _mean(
            float(item["joint_effective_hypotheses"]) for item in records
        ),
        "symbolic_door_identified_rate": len(identified_steps) / len(records),
        "mean_probes_to_symbolic_identification_among_identified": (
            _mean(identified_steps) if identified_steps else None
        ),
        "failure_penalized_mean_probes_to_identification": _mean(
            (
                int(item["first_symbolic_identification_step"])
                if item["first_symbolic_identification_step"] is not None
                and int(item["first_symbolic_identification_step"]) <= budget
                else budget + 1
            )
            for item in records
        ),
        "mean_cumulative_expected_door_gain": _mean(
            float(item["cumulative_expected_door_gain"]) for item in records
        ),
        "mean_cumulative_outcome_information_gain_nats": _mean(
            float(item["cumulative_outcome_information_gain_nats"])
            for item in records
        ),
        "mean_observed_log_predictive_probability": (
            _mean(observed_log_probabilities)
            if observed_log_probabilities
            else None
        ),
    }


def _paired_summary(
    active: Sequence[dict[str, object]],
    uniform: Sequence[dict[str, object]],
    *,
    budget: int,
) -> dict[str, object]:
    if len(active) != len(uniform):
        raise ValueError("paired policies must contain the same tasks")
    active_wins = tuple(bool(item["won"]) for item in active)
    uniform_wins = tuple(bool(item["won"]) for item in uniform)

    def penalized(item: dict[str, object]) -> int:
        step = item["first_symbolic_identification_step"]
        return (
            int(step)
            if step is not None and int(step) <= budget
            else budget + 1
        )

    return {
        "tasks": len(active),
        "paired_terminal_win_rate_gain": _mean(active_wins)
        - _mean(uniform_wins),
        "active_only_wins": sum(
            left and not right
            for left, right in zip(active_wins, uniform_wins, strict=True)
        ),
        "uniform_only_wins": sum(
            right and not left
            for left, right in zip(active_wins, uniform_wins, strict=True)
        ),
        "both_win": sum(
            left and right
            for left, right in zip(active_wins, uniform_wins, strict=True)
        ),
        "neither_wins": sum(
            not left and not right
            for left, right in zip(active_wins, uniform_wins, strict=True)
        ),
        "paired_failure_penalized_probe_delta_active_minus_uniform": _mean(
            penalized(left) - penalized(right)
            for left, right in zip(active, uniform, strict=True)
        ),
    }


def _summarise_scope(
    records: dict[str, dict[int, list[dict[str, object]]]],
    budgets: Sequence[int],
) -> dict[str, object]:
    return {
        str(budget): {
            "expected-door-gain": _summarise_policy(
                records["expected-door-gain"][budget],
                budget=budget,
            ),
            "uniform": _summarise_policy(
                records["uniform"][budget],
                budget=budget,
            ),
            "paired": _paired_summary(
                records["expected-door-gain"][budget],
                records["uniform"][budget],
                budget=budget,
            ),
        }
        for budget in budgets
    }


def main() -> None:
    args = parse_args()
    budgets = _validate_args(args)

    import torch

    from prp_wm.causal_filter import predict_factor_panel
    from prp_wm.latent_rules import outcome_map
    from prp_wm.rulegrid import ALL_AXES
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _resolve_device,
    )
    from scripts.run_public_version_space_k4 import (
        load_public_version_k4_checkpoint,
    )

    _configure_determinism(torch, args.seed)
    device = _resolve_device(torch, args.device)
    belief_path = args.belief_checkpoint.resolve()
    active_executor_path = args.active_executor_checkpoint.resolve()
    belief_model, belief_checkpoint, _, _ = load_public_version_k4_checkpoint(
        torch,
        belief_path,
        device=device,
    )
    if belief_checkpoint.get("support_input") != "raw":
        raise SystemExit("belief checkpoint must infer from raw public support")
    if not hasattr(belief_model, "infer_factor_belief"):
        raise SystemExit("belief checkpoint has no factor-belief interface")
    active_executor, active_checkpoint, active_result = _load_active_executor(
        torch,
        active_executor_path,
        device,
    )
    if belief_model.config != active_executor.config:
        raise SystemExit("belief and active executor model configs disagree")
    factor_bank = belief_model.factor_bank.detach().cpu()
    if factor_bank.shape != (64, 3):
        raise SystemExit("acquisition ceiling requires the complete 64-code bank")

    by_axis_records: dict[
        str,
        dict[str, dict[int, list[dict[str, object]]]],
    ] = {}
    aggregate_records = {
        policy: {budget: [] for budget in budgets}
        for policy in POLICIES
    }
    traces: dict[str, dict[str, list[dict[str, object]]]] = {}

    for query in (
        PublicDoorQuery(axis_index) for axis_index in range(len(ALL_AXES))
    ):
        axis = ALL_AXES[query.axis_index]
        tasks = _build_axis_tasks(
            axis,
            groups=args.groups_per_axis,
            split=args.split,
            master_seed=args.data_master_seed,
        )
        candidate_count = len(tasks[0].inference.active_candidates)
        uniform_orders = _uniform_orders(
            torch=torch,
            axis_index=query.axis_index,
            groups=args.groups_per_axis,
            candidates=candidate_count,
            seed=args.seed,
        )
        axis_records = {
            policy: {budget: [] for budget in budgets}
            for policy in POLICIES
        }
        axis_traces = {policy: [] for policy in POLICIES}
        for start in range(0, len(tasks), args.batch_size):
            selected_tasks = tasks[start : start + args.batch_size]
            selected_orders = uniform_orders[start : start + args.batch_size]
            initial = _initial_joint_belief(
                torch=torch,
                model=belief_model,
                tasks=selected_tasks,
                device=device,
            )
            states, actions, action_mask, feedback = _candidate_panel(
                torch=torch,
                tasks=selected_tasks,
                device=device,
            )
            codes = factor_bank.to(device)[None].expand(
                len(selected_tasks),
                -1,
                -1,
            )
            with torch.no_grad():
                prediction = predict_factor_panel(
                    active_executor,
                    states,
                    actions,
                    codes,
                    action_mask,
                )
                maps = outcome_map(prediction).reshape(
                    len(selected_tasks),
                    candidate_count,
                    factor_bank.shape[0],
                    active_executor.config.grid_size,
                    active_executor.config.grid_size,
                ).cpu()
            for policy in POLICIES:
                snapshots, batch_traces = _rollout_policy(
                    torch=torch,
                    policy=policy,
                    tasks=selected_tasks,
                    query=query,
                    initial_log_weights=initial,
                    factor_bank=factor_bank,
                    outcome_maps=maps,
                    prediction=prediction,
                    canonical_feedback=feedback,
                    uniform_orders=selected_orders,
                    budgets=budgets,
                )
                for budget, records in snapshots.items():
                    axis_records[policy][budget].extend(records)
                    aggregate_records[policy][budget].extend(records)
                remaining_traces = max(
                    0,
                    args.trace_tasks_per_axis - len(axis_traces[policy]),
                )
                for trace in batch_traces[:remaining_traces]:
                    axis_traces[policy].append(
                        {
                            "global_task_index": start
                            + int(trace["task_batch_index"]),
                            "door_axis": axis.value,
                            **{
                                key: value
                                for key, value in trace.items()
                                if key != "task_batch_index"
                            },
                        }
                    )
        by_axis_records[axis.value] = axis_records
        traces[axis.value] = axis_traces

    selector_audit = _selector_boundary_audit()
    if not selector_audit["passed"]:
        raise AssertionError("selector boundary audit failed")
    evaluations = {
        "by_axis": {
            axis: _summarise_scope(records, budgets)
            for axis, records in by_axis_records.items()
        },
        "aggregate": _summarise_scope(aggregate_records, budgets),
    }
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "experiment": "oracle_canonical_model_directed_acquisition_ceiling",
        "status": "complete",
        "interpretation": (
            "Privileged ceiling only: public t0 belief proposal, known 3x4 "
            "codebook, explicit door axis, oracle palette binding, and an "
            "active-prefix-calibrated frozen factor executor."
        ),
        "controller_reads_door_axis": True,
        "controller_infers_axis_from_task_id": False,
        "candidate_selection_reads_task_or_probe_id": False,
        "candidate_selection_reads_candidate_kind": False,
        "candidate_selection_reads_true_program_or_feedback": False,
        "feedback_read_only_after_candidate_selection": True,
        "selection_objective": (
            "expected terminal door-success gain; deterministic outcome "
            "information gain in nats as secondary key; public bank position "
            "as final tie breaker"
        ),
        "posterior_initialization": (
            "product of public belief factor_probabilities over the recorded "
            "64-code factor_bank; no executor support-likelihood rescore"
        ),
        "posterior_update": (
            "explicit Bayes update with selected candidate's frozen-executor "
            "full-grid OutcomePrediction.log_prob"
        ),
        "candidate_outcome_model": (
            "MAP partitions from active-support-calibrated oracle-canonical "
            "OracleFactorExecutor"
        ),
        "paired_uniform_protocol": (
            "same tasks, t0 posterior, executor, candidates, and feedback; one "
            "nuisance-seeded order per four-task group"
        ),
        "budgets": list(budgets),
        "groups_per_axis": args.groups_per_axis,
        "tasks_per_axis": 4 * args.groups_per_axis,
        "total_tasks": 12 * args.groups_per_axis,
        "seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "split": args.split,
        "device": str(device),
        "belief_checkpoint": str(belief_path),
        "belief_checkpoint_sha256": _sha256_file(belief_path),
        "belief_checkpoint_model_type": belief_checkpoint.get("model_type"),
        "active_executor_checkpoint": str(active_executor_path),
        "active_executor_checkpoint_sha256": _sha256_file(
            active_executor_path
        ),
        "active_executor_checkpoint_schema": active_checkpoint.get(
            "checkpoint_schema_version"
        ),
        "active_executor_gate_passed": active_result.get(
            "active_prefix_executor_gate",
            {},
        ).get("passed"),
        "selector_boundary_audit": selector_audit,
        "evaluations": evaluations,
        "episode_traces": traces,
        "source_sha256": _source_sha256(),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
