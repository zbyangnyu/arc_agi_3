#!/usr/bin/env python3
"""Train and audit the GRAM-style factorized causal-rule proposer.

Training uses four target-conditioned posterior trajectories because the
privileged unordered behavior set contains four classes.  Public inference is
different: ``--inference-widths W ...`` requests exactly W iid prior
trajectories in one call, not W repetitions of a K=4 proposer.  The width audit
reports both raw samples and a verifier-assisted top-four selection made only
from sampled unique candidates and public-support executor costs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


CHECKPOINT_SCHEMA_VERSION = "prp-wm.gram-factorized-causal-k4.v1"
_AUDITED_SOURCE_FILES = (
    "prp_wm/gram_causal_rules.py",
    "prp_wm/discrete_causal_rules.py",
    "prp_wm/causal_rules.py",
    "prp_wm/causal_filter.py",
    "prp_wm/latent_rules.py",
    "prp_wm/neural.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_gram_causal_screen.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_causal_mechanism_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, required=True)
    parser.add_argument("--context-fold", type=int, choices=range(4))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-pool-tasks", type=int, default=144)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--seen-eval-tasks", type=int, default=144)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tail-steps", type=int, default=0)
    parser.add_argument("--tail-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--recursive-steps", type=int, default=4)
    parser.add_argument("--guidance-dim", type=int, default=32)
    parser.add_argument(
        "--guidance-mode",
        choices=("stochastic", "mean"),
        default="stochastic",
        help=(
            "sample Gaussian residual guidance or use its deterministic mean; "
            "mean is the recursive-compute control"
        ),
    )
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument(
        "--kl-balance",
        type=float,
        default=0.8,
        help="fraction of balanced KL gradient assigned to fitting prior p to q",
    )
    parser.add_argument(
        "--kl-warmup-steps",
        type=int,
        default=100,
        help="linearly reach --kl-weight after this many completed steps; 0 disables warmup",
    )
    parser.add_argument("--deep-supervision-decay", type=float, default=1.0)
    parser.add_argument("--validity-weight", type=float, default=0.10)
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.0,
        help=(
            "explicit particle repulsion; the GRAM screen defaults to zero so "
            "sample diversity cannot be attributed to this auxiliary barrier"
        ),
    )
    parser.add_argument("--sharpening-weight", type=float, default=0.0)
    parser.add_argument("--proper-weight", type=float, default=1.0)
    parser.add_argument("--balanced-weight", type=float, default=1.0)
    parser.add_argument("--factor-temperature-start", type=float, default=1.0)
    parser.add_argument("--factor-temperature-end", type=float, default=1.0)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--nll-threshold", type=float, default=0.05)
    parser.add_argument(
        "--inference-widths",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
        metavar="W",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default="gram-causal-train")
    parser.add_argument("--eval-split", default="gram-causal-composition")
    return parser.parse_args()


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


def _cli_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Return every parsed argument in a JSON-safe representation."""

    encoded: dict[str, object] = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            encoded[name] = str(value.resolve())
        elif isinstance(value, (tuple, list)):
            encoded[name] = list(value)
        else:
            encoded[name] = value
    return encoded


def _kl_warmup(end: float, completed_step: int, warmup_steps: int) -> float:
    if warmup_steps == 0:
        return end
    return end * min(completed_step / warmup_steps, 1.0)


def _ordered_unique(
    candidates: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    unique: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _support_ranked_unique(
    candidates: list[tuple[int, int, int]],
    support_cost_by_code: dict[tuple[int, int, int], float],
    *,
    limit: int = 4,
) -> list[tuple[int, int, int]]:
    """Rank sampled unique codes by public support cost with stable tie breaks."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    unique = _ordered_unique(candidates)
    missing = [code for code in unique if code not in support_cost_by_code]
    if missing:
        raise ValueError(f"missing support costs for sampled codes: {missing}")
    return sorted(
        unique,
        key=lambda code: (support_cost_by_code[code], code),
    )[:limit]


def _uniform_four_mode_coupon_collector(width: int) -> tuple[float, float]:
    """Return expected recall and full coverage for iid uniform draws on 4 modes."""

    if width <= 0:
        raise ValueError("width must be positive")
    recall = 1.0 - (3.0 / 4.0) ** width
    all_covered = (
        1.0
        - 4.0 * (3.0 / 4.0) ** width
        + 6.0 * (1.0 / 2.0) ** width
        - 4.0 * (1.0 / 4.0) ** width
    )
    return recall, max(0.0, min(1.0, all_covered))


class _GuidanceEvaluationModel:
    """Pin the guidance control while reusing the established evaluator."""

    def __init__(self, model: Any, *, sample_noise: bool) -> None:
        self.model = model
        self.sample_noise = sample_noise
        self.config = model.config

    def eval(self) -> "_GuidanceEvaluationModel":
        self.model.eval()
        return self

    def infer_support(self, batch: Any, *, temperature: float | None = None) -> Any:
        return self.model.infer_support(
            batch,
            temperature=temperature,
            sample_noise=self.sample_noise,
        )

    def predict_panel(self, batch: Any, inference: Any) -> Any:
        return self.model.predict_panel(batch, inference)

    def predict_support(self, batch: Any, inference: Any) -> Any:
        return self.model.predict_support(batch, inference)


def _four_item_gate(evaluation: dict[str, object]) -> dict[str, object]:
    thresholds = {
        "coverage_at_4_mass_weighted": 0.90,
        "all_classes_covered_task_rate": 0.90,
        "factor_tuple_coverage_at_4": 0.90,
        "all_particles_support_exact_task_rate": 0.90,
    }
    checks = {
        name: {
            "value": float(evaluation[name]),
            "threshold_gte": threshold,
            "passed": float(evaluation[name]) >= threshold,
        }
        for name, threshold in thresholds.items()
    }
    return {
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks.values()),
    }


def _reset_evaluation_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_inference_width_curve(
    *,
    torch: Any,
    model: Any,
    device: Any,
    tasks: tuple[Any, ...],
    batch_size: int,
    widths: tuple[int, ...],
    recursive_steps: int,
    factor_temperature: float,
    sample_noise: bool,
    inference_seed: int,
    proper_weight: float,
    balanced_weight: float,
    make_behavior_batch: Any,
    triple_indices: tuple[int, ...],
    rule_program_factor_ids: Any,
    version_space: Any,
) -> dict[str, object]:
    """Measure prior proposal recall before and after public-cost top-four."""

    bank_codes = [
        tuple(int(value) for value in row)
        for row in model.factor_bank.detach().cpu().tolist()
    ]
    bank_index = {code: index for index, code in enumerate(bank_codes)}
    totals = {
        width: {
            "raw_unique": 0,
            "raw_compatible_unique": 0,
            "raw_compatible_trajectories": 0,
            "all_version_space": 0,
            "top4_compatible": 0,
            "top4_all_version_space": 0,
            "top4_size": 0,
        }
        for width in widths
    }

    model.eval()
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[start : start + batch_size]
            batch = make_behavior_batch(
                batch_tasks,
                diagnostic_indices=triple_indices,
                device=device,
            )
            support_costs = model.discrete_support_costs(
                batch,
                proper_weight=proper_weight,
                balanced_weight=balanced_weight,
            ).detach().cpu()
            compatible_sets = [
                {
                    rule_program_factor_ids(program)
                    for program in version_space(
                        task.inference.support,
                        task.privileged.palette,
                    )
                }
                for task in batch_tasks
            ]
            if any(len(compatible) != 4 for compatible in compatible_sets):
                raise AssertionError("every public version space must contain four rules")

            for width in widths:
                inference = model.sample_width_candidates(
                    batch,
                    width=width,
                    recursive_steps=recursive_steps,
                    seed=inference_seed + start,
                    temperature=factor_temperature,
                    sample_noise=sample_noise,
                )
                if inference.particles != width:
                    raise AssertionError(
                        "inference width must equal total sampled trajectories"
                    )
                for task_index, compatible in enumerate(compatible_sets):
                    candidates = [
                        tuple(int(value) for value in row)
                        for row in inference.factor_ids[task_index].cpu().tolist()
                    ]
                    unique = _ordered_unique(candidates)
                    unique_set = set(unique)
                    compatible_unique = unique_set.intersection(compatible)
                    cost_row = support_costs[task_index]
                    cost_by_code = {
                        code: float(cost_row[bank_index[code]]) for code in unique
                    }
                    top4 = _support_ranked_unique(
                        candidates,
                        cost_by_code,
                        limit=4,
                    )
                    top4_compatible = set(top4).intersection(compatible)
                    width_totals = totals[width]
                    width_totals["raw_unique"] += len(unique)
                    width_totals["raw_compatible_unique"] += len(compatible_unique)
                    width_totals["raw_compatible_trajectories"] += sum(
                        candidate in compatible for candidate in candidates
                    )
                    width_totals["all_version_space"] += int(
                        compatible.issubset(unique_set)
                    )
                    width_totals["top4_compatible"] += len(top4_compatible)
                    width_totals["top4_all_version_space"] += int(
                        compatible.issubset(set(top4))
                    )
                    width_totals["top4_size"] += len(top4)

    task_count = len(tasks)
    points = []
    for width in widths:
        values = totals[width]
        oracle_recall, oracle_all_covered = _uniform_four_mode_coupon_collector(
            width
        )
        points.append(
            {
                "width_total_trajectories": width,
                "tasks": task_count,
                "mean_raw_unique_candidates": values["raw_unique"] / task_count,
                "mean_raw_compatible_unique_candidates": (
                    values["raw_compatible_unique"] / task_count
                ),
                "raw_trajectory_compatible_rate": (
                    values["raw_compatible_trajectories"] / (task_count * width)
                ),
                "compatible_rule_recall": (
                    values["raw_compatible_unique"] / (task_count * 4)
                ),
                "all_version_space_coverage_task_rate": (
                    values["all_version_space"] / task_count
                ),
                "mean_support_ranked_top4_size": values["top4_size"] / task_count,
                "support_ranked_top4_compatible_rule_recall": (
                    values["top4_compatible"] / (task_count * 4)
                ),
                "support_ranked_top4_all_version_space_coverage_task_rate": (
                    values["top4_all_version_space"] / task_count
                ),
                "uniform_four_mode_oracle_compatible_rule_recall": oracle_recall,
                "uniform_four_mode_oracle_all_version_space_coverage_task_rate": (
                    oracle_all_covered
                ),
            }
        )
    return {
        "width_semantics": (
            "W is the total number of trajectories sampled iid in one prior call; "
            "it is not W repeats of K=4"
        ),
        "guidance_mode": "stochastic" if sample_noise else "mean",
        "iid_trajectory_sampling": sample_noise,
        "candidate_deduplication": "exact integer factor tuple",
        "ranking": (
            "sampled unique candidates only, ascending detached executor cost on "
            "the six public support transitions; deterministic tuple tie break"
        ),
        "top_k": 4,
        "compatible_rules_per_task": 4,
        "coupon_collector_reference": (
            "ideal iid draws uniformly distributed over exactly the four valid "
            "rules; e.g. width=4 has 68.36% expected recall but only 9.375% "
            "probability of collecting all four"
        ),
        "points": points,
    }


def main() -> None:
    args = parse_args()
    for name in (
        "steps",
        "batch_size",
        "train_pool_tasks",
        "eval_tasks",
        "seen_eval_tasks",
        "eval_batch_size",
        "recursive_steps",
        "guidance_dim",
        "attention_layers",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.kl_warmup_steps < 0:
        raise SystemExit("--kl-warmup-steps must be non-negative")
    if args.tail_steps < 0 or args.tail_steps > args.steps:
        raise SystemExit("--tail-steps must lie in [0,steps]")
    if not 0.0 <= args.kl_balance <= 1.0:
        raise SystemExit("--kl-balance must lie in [0,1]")
    for name in (
        "learning_rate",
        "tail_learning_rate",
        "max_grad_norm",
        "deep_supervision_decay",
        "factor_temperature_start",
        "factor_temperature_end",
        "nll_threshold",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "kl_weight",
        "validity_weight",
        "diversity_weight",
        "sharpening_weight",
        "proper_weight",
        "balanced_weight",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size > args.train_pool_tasks:
        raise SystemExit("--batch-size cannot exceed --train-pool-tasks")
    if args.train_split == args.eval_split:
        raise SystemExit("training and evaluation splits must differ")
    if any(width <= 0 for width in args.inference_widths):
        raise SystemExit("--inference-widths values must be positive")
    widths = tuple(sorted(set(args.inference_widths)))

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from prp_wm.gram_causal_rules import GRAMFactorizedCausalK4
    from prp_wm.latent_rules import (
        outcome_map,
        rule_program_factor_ids,
        rulegrid_tasks_to_canonical_behavior_batch,
    )
    from prp_wm.pilot import (
        NONTRIPLE_DIAGNOSTIC_INDICES,
        TRIPLE_DIAGNOSTIC_INDICES,
        make_pilot_tasks,
    )
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import (
        _configure_determinism,
        _evaluate,
        _resolve_device,
    )
    from scripts.run_expected_discrete_causal_coverage import (
        _build_context_pool,
        _cached_task_factory,
        _load_audited_executor,
        _support_context_key,
        _linear_schedule,
    )

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.seed)
    executor_path = args.executor_checkpoint.resolve()
    executor, executor_checkpoint = _load_audited_executor(
        torch,
        executor_path,
        device,
    )
    model = GRAMFactorizedCausalK4(
        executor,
        recursive_steps=args.recursive_steps,
        guidance_dim=args.guidance_dim,
        attention_layers=args.attention_layers,
        temperature=args.factor_temperature_start,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    pool_arguments = {
        "make_pilot_tasks": make_pilot_tasks,
        "master_seed": args.data_master_seed,
        "factor_ids_for_program": rule_program_factor_ids,
        "version_space": version_space,
        "context_fold": args.context_fold,
    }
    train_pool = _build_context_pool(
        **pool_arguments,
        split=args.train_split,
        diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
        count=args.train_pool_tasks,
        heldout=False,
    )
    eval_pool = _build_context_pool(
        **pool_arguments,
        split=args.eval_split,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.eval_tasks,
        heldout=True,
    )
    seen_eval_pool = _build_context_pool(
        **pool_arguments,
        split=args.eval_split,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.seen_eval_tasks,
        heldout=False,
    )
    train_contexts = {
        _support_context_key(
            task,
            factor_ids_for_program=rule_program_factor_ids,
            version_space=version_space,
        )
        for task in train_pool
    }
    eval_contexts = {
        _support_context_key(
            task,
            factor_ids_for_program=rule_program_factor_ids,
            version_space=version_space,
        )
        for task in eval_pool
    }
    if train_contexts.intersection(eval_contexts):
        raise AssertionError("train and held-out support contexts must be disjoint")
    if len(train_contexts) != 36 or len(eval_contexts) != 12:
        raise AssertionError("context split must contain 36 train and 12 eval contexts")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    result_path = output / "result.json"
    sample_noise = args.guidance_mode == "stochastic"
    run_config: dict[str, object] = {
        "experiment": "gram_factorized_causal_rule_screen",
        "result_kind": "privileged_recursive_stochastic_rule_proposer_screen",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": type(model).__name__,
        "cli_arguments": _cli_arguments(args),
        "guidance_mode": args.guidance_mode,
        "sample_noise": sample_noise,
        "training_trajectory_width": model.config.particles,
        "training_width_reason": "one posterior trajectory per unordered behavior class",
        "inference_widths_total_trajectories": list(widths),
        "recursive_steps": args.recursive_steps,
        "guidance_dim": args.guidance_dim,
        "truncate_between_recursive_steps": model.truncate_between_steps,
        "guidance_log_variance_bounds": [
            model.minimum_log_variance,
            model.maximum_log_variance,
        ],
        "initial_guidance_log_variance": model.initial_log_variance,
        "mechanism_axes_given": ["collision", "trigger", "relation"],
        "mechanism_value_labels_used_for_training": False,
        "program_labels_used_for_training": False,
        "individual_true_program_query_target_used_for_training": False,
        "privileged_unordered_behavior_set_supervision": True,
        "behavior_set_source": (
            "simulator outputs for all four rule behaviors compatible with public support"
        ),
        "all_64_integer_codes_evaluated_for_training": True,
        "discrete_cost_table_detached": True,
        "privileged_palette_canonicalization": True,
        "executor_frozen_and_eval": True,
        "executor_support_calibrated": True,
        "executor_checkpoint": str(executor_path),
        "executor_checkpoint_sha256": _sha256_file(executor_path),
        "executor_checkpoint_schema_version": executor_checkpoint[
            "checkpoint_schema_version"
        ],
        "context_fold": args.context_fold,
        "context_split_kind": (
            "legacy_diagonal" if args.context_fold is None else "latin_modulo_4"
        ),
        "unique_train_support_contexts": len(train_contexts),
        "unique_eval_support_contexts": len(eval_contexts),
        "train_support_contexts": [list(context) for context in sorted(train_contexts)],
        "eval_support_contexts": [list(context) for context in sorted(eval_contexts)],
        "train_eval_contexts_disjoint": True,
        "model_seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "model_config": asdict(model.config),
        "device": str(device),
        "torch_version": torch.__version__,
        "source_sha256": _source_sha256(),
    }

    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(args.seed + 1)
    sample_order = torch.randperm(len(train_pool), generator=sampler).tolist()
    sample_cursor = 0
    latest: dict[str, object] = {}
    started = time.perf_counter()
    model.train()
    with progress_path.open("w", encoding="utf-8") as progress_file:
        for step in range(args.steps):
            learning_rate = (
                args.tail_learning_rate
                if args.tail_steps and step >= args.steps - args.tail_steps
                else args.learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            if sample_cursor + args.batch_size > len(sample_order):
                sample_order = torch.randperm(
                    len(train_pool), generator=sampler
                ).tolist()
                sample_cursor = 0
            indices = sample_order[sample_cursor : sample_cursor + args.batch_size]
            sample_cursor += args.batch_size
            tasks = tuple(train_pool[index] for index in indices)
            batch = rulegrid_tasks_to_canonical_behavior_batch(
                tasks,
                diagnostic_indices=NONTRIPLE_DIAGNOSTIC_INDICES,
                device=device,
            )
            temperature = _linear_schedule(
                args.factor_temperature_start,
                args.factor_temperature_end,
                step,
                args.steps,
            )
            kl_weight = _kl_warmup(
                args.kl_weight,
                step + 1,
                args.kl_warmup_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.losses(
                batch,
                kl_weight=kl_weight,
                kl_balance=args.kl_balance,
                validity_weight=args.validity_weight,
                diversity_weight=args.diversity_weight,
                sharpening_weight=args.sharpening_weight,
                proper_weight=args.proper_weight,
                balanced_weight=args.balanced_weight,
                deep_supervision_decay=args.deep_supervision_decay,
                temperature=temperature,
                sample_noise=sample_noise,
            )
            if not bool(torch.isfinite(loss.total).item()):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            final_ids = loss.trajectories.factor_ids[-1]
            mean_unique = sum(
                int(torch.unique(ids, dim=0).shape[0]) for ids in final_ids
            ) / batch.batch_size
            latest = loss.detached_metrics() | {
                "gradient_norm": gradient_norm,
                "factor_temperature": temperature,
                "kl_weight": kl_weight,
                "learning_rate": learning_rate,
                "guidance_mode": args.guidance_mode,
                "mean_unique_factor_tuples": mean_unique,
                "recursive_step_objectives": [
                    float(value) for value in loss.step_objectives.detach().cpu()
                ],
                "recursive_step_kls": [
                    float(value) for value in loss.step_kls.detach().cpu()
                ],
                "deep_supervision_weights": [
                    float(value)
                    for value in loss.deep_supervision_weights.detach().cpu()
                ],
            }
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                record = {
                    "step": completed,
                    "tasks_seen": completed * args.batch_size,
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress_file.write(encoded + "\n")
                progress_file.flush()
                print(encoded, flush=True)

    training_seconds = time.perf_counter() - started
    checkpoint: dict[str, object] = {
        **run_config,
        "steps": args.steps,
        "tasks_seen": args.steps * args.batch_size,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "latest_training_metrics": latest,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    evaluation_model = _GuidanceEvaluationModel(
        model,
        sample_noise=sample_noise,
    )
    common_evaluation = {
        "torch": torch,
        "model": evaluation_model,
        "device": device,
        "split": args.eval_split,
        "data_master_seed": args.data_master_seed,
        "task_count": args.eval_tasks,
        "batch_size": args.eval_batch_size,
        "nll_threshold": args.nll_threshold,
        "factor_temperature": args.factor_temperature_end,
        "make_pilot_tasks": _cached_task_factory(eval_pool),
        "make_behavior_batch": rulegrid_tasks_to_canonical_behavior_batch,
        "outcome_map": outcome_map,
        "triple_indices": TRIPLE_DIAGNOSTIC_INDICES,
        "rule_program_factor_ids": rule_program_factor_ids,
        "version_space": version_space,
    }
    evaluation_seed = args.seed + 10_000
    _reset_evaluation_seed(torch, evaluation_seed)
    heldout = _evaluate(**common_evaluation, support_ablation="none")
    _reset_evaluation_seed(torch, evaluation_seed)
    shuffled = _evaluate(
        **common_evaluation,
        support_ablation="shuffle-targets",
    )
    _reset_evaluation_seed(torch, evaluation_seed)
    seen = _evaluate(
        **(
            common_evaluation
            | {
                "task_count": len(seen_eval_pool),
                "make_pilot_tasks": _cached_task_factory(seen_eval_pool),
            }
        ),
        support_ablation="none",
    )
    width_curve = _evaluate_inference_width_curve(
        torch=torch,
        model=model,
        device=device,
        tasks=eval_pool,
        batch_size=args.eval_batch_size,
        widths=widths,
        recursive_steps=args.recursive_steps,
        factor_temperature=args.factor_temperature_end,
        sample_noise=sample_noise,
        inference_seed=args.seed + 20_000,
        proper_weight=args.proper_weight,
        balanced_weight=args.balanced_weight,
        make_behavior_batch=rulegrid_tasks_to_canonical_behavior_batch,
        triple_indices=TRIPLE_DIAGNOSTIC_INDICES,
        rule_program_factor_ids=rule_program_factor_ids,
        version_space=version_space,
    )
    evaluation_gates = {
        "heldout": _four_item_gate(heldout),
        "seen": _four_item_gate(seen),
        "shuffled_support_control": _four_item_gate(shuffled),
    }
    result: dict[str, object] = {
        **run_config,
        "steps": args.steps,
        "tasks_seen": args.steps * args.batch_size,
        "training_seconds": round(training_seconds, 6),
        "latest_training_metrics": latest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "heldout_triple_coverage": heldout,
        "seen_context_triple_coverage": seen,
        "shuffled_support_target_control": shuffled,
        "evaluation_gates": evaluation_gates,
        "static_gate": evaluation_gates["heldout"],
        "inference_width_curve": width_curve,
        "interpretation": (
            "This screen tests whether stochastic recursive prior trajectories can "
            "recover the four public-support-compatible factor tuples and whether "
            "public replay cost can select them. A pass does not establish discovery "
            "of the privileged mechanism axes, factor values, or palette roles."
        ),
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
