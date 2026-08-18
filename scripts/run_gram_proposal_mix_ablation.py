#!/usr/bin/env python3
"""Matched-W32 proposal-source ablation for sequential RuleGrid evidence.

This runner deliberately isolates *proposal coverage* under the privileged
symbolic verifier.  It reuses the fixed t0..t3 factual/counterfactual protocol
from ``run_gram_smc_active_screen.py`` and compares:

* 32 GRAM proposals;
* nested GRAM/uniform mixtures 24+8, 16+16, and 8+24;
* 32 iid-uniform proposals; and
* a randomized 32-code, pairwise-balanced Latin covering bank.

Every method receives exactly 32 fresh proposals per task and stage and the
same carry limit.  The stochastic mixture family is built from prefixes of
one shared 32-wide GRAM stream and one shared 32-wide uniform stream.  The
factual and counterfactual branches use the same stage seed (common random
numbers); their GRAM codes can nevertheless differ because their observed
targets differ.  Uniform and covering codes are identical across branches.

The true code and symbolic version space are used only after proposal
generation, for exact-consistency filtering and metrics.  There is no exact
bank fallback and no injection of a missing compatible code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED
from scripts.run_gram_smc_active_screen import (
    BeliefSnapshot,
    PairedEvidence,
    _build_paired_evidence,
    _factor_code,
    _method_report,
    _snapshot_from_particles,
    _snapshots_from_inference,
    _support_batch,
    _symbolic_population_stages,
)


FRESH_PROPOSALS = 32
STAGES = 4
MIXTURE_GRAM_COUNTS = (32, 24, 16, 8, 0)
RESULT_SCHEMA_VERSION = "prp-wm.gram-proposal-mix-ablation.v1"
DEFAULT_EXECUTOR = REPOSITORY_ROOT / (
    "runs/support_calibrated_executor_seed20260724/checkpoint_last.pt"
)
DEFAULT_GRAM = REPOSITORY_ROOT / (
    "runs/gram_causal_screen600_fold0_seed20260728/checkpoint_last.pt"
)
_AUDITED_SOURCE_FILES = (
    "prp_wm/gram_causal_rules.py",
    "prp_wm/latent_rules.py",
    "prp_wm/pilot.py",
    "prp_wm/rulegrid.py",
    "scripts/run_gram_proposal_mix_ablation.py",
    "scripts/run_gram_smc_active_screen.py",
    "scripts/run_expected_discrete_causal_coverage.py",
    "scripts/run_support_calibrated_executor.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", type=Path, default=DEFAULT_EXECUTOR)
    parser.add_argument("--gram-checkpoint", type=Path, default=DEFAULT_GRAM)
    parser.add_argument(
        "--inference-seeds",
        type=int,
        nargs="+",
        default=(20260729, 20260730, 20260731),
        help="paired Monte-Carlo seeds; task pool is held fixed across seeds",
    )
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--eval-split",
        default="gram-vps-sequential-assimilation",
        help="kept equal to the main sequential screen for matched held-out tasks",
    )
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--recursive-steps", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--carry-limit", type=int, default=FRESH_PROPOSALS)
    parser.add_argument("--device", default="auto")
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


def _json_cli(args: argparse.Namespace) -> dict[str, object]:
    encoded: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            encoded[key] = str(value.resolve())
        elif isinstance(value, (tuple, list)):
            encoded[key] = list(value)
        else:
            encoded[key] = value
    return encoded


def _stream_seed(base_seed: int, namespace: str, stage: int, task: int = -1) -> int:
    """Derive independent, reproducible 63-bit stream seeds without overlap."""

    if type(base_seed) is not int or type(stage) is not int or type(task) is not int:
        raise ValueError("base_seed, stage, and task must be integers")
    if stage < 0 or task < -1 or not namespace:
        raise ValueError("stage/task range and namespace must be valid")
    payload = f"{base_seed}|{namespace}|{stage}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _latin_pairwise_cover_codes(seed: int) -> tuple[tuple[int, int, int], ...]:
    """Return 32 unique codes with exactly balanced 1- and 2-axis marginals.

    Before random relabeling, the construction is
    ``z = x + y + offset (mod 4)`` for two randomly chosen offsets.  Thus each
    axis value occurs eight times and each ordered value pair on every pair of
    axes occurs twice.  Axis order and all value labels are then independently
    permuted, preserving those properties while avoiding a privileged code.
    """

    rng = random.Random(seed)
    offsets = rng.sample(range(4), 2)
    value_permutations = [rng.sample(range(4), 4) for _ in range(3)]
    axis_permutation = rng.sample(range(3), 3)
    codes: list[tuple[int, int, int]] = []
    for x in range(4):
        for y in range(4):
            for offset in offsets:
                raw = (x, y, (x + y + offset) % 4)
                relabeled = tuple(
                    value_permutations[axis][raw[axis]] for axis in range(3)
                )
                codes.append(tuple(relabeled[axis] for axis in axis_permutation))
    rng.shuffle(codes)
    result = tuple(codes)
    _assert_pairwise_cover(result)
    return result


def _assert_pairwise_cover(codes: Sequence[Sequence[int]]) -> None:
    materialized = tuple(tuple(int(value) for value in code) for code in codes)
    if len(materialized) != FRESH_PROPOSALS or len(set(materialized)) != FRESH_PROPOSALS:
        raise AssertionError("Latin covering must contain exactly 32 unique codes")
    for axis in range(3):
        if Counter(code[axis] for code in materialized) != Counter({value: 8 for value in range(4)}):
            raise AssertionError("Latin covering has an unbalanced one-axis marginal")
    for left in range(3):
        for right in range(left + 1, 3):
            expected = Counter({(x, y): 2 for x in range(4) for y in range(4)})
            observed = Counter((code[left], code[right]) for code in materialized)
            if observed != expected:
                raise AssertionError("Latin covering has an unbalanced pair marginal")


def _iid_uniform_proposal_stages(
    *, task_count: int, seed: int
) -> tuple[tuple[BeliefSnapshot, ...], ...]:
    """Draw one shared 32-wide uniform bank for each task/stage."""

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return tuple(
        tuple(
            _snapshot_from_particles(
                tuple(
                    (rng.randrange(4), rng.randrange(4), rng.randrange(4))
                    for _ in range(FRESH_PROPOSALS)
                )
            )
            for task_index in range(task_count)
            for rng in (
                random.Random(_stream_seed(seed, "uniform", stage, task_index)),
            )
        )
        for stage in range(STAGES)
    )


def _latin_cover_proposal_stages(
    *, task_count: int, seed: int
) -> tuple[tuple[BeliefSnapshot, ...], ...]:
    """Build branch-shared, independently randomized Latin banks by task/stage."""

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return tuple(
        tuple(
            _snapshot_from_particles(
                _latin_pairwise_cover_codes(
                    _stream_seed(seed, "latin_pairwise_cover", stage, task_index)
                )
            )
            for task_index in range(task_count)
        )
        for stage in range(STAGES)
    )


def _compose_mixture_stages(
    gram: Sequence[Sequence[BeliefSnapshot]],
    uniform: Sequence[Sequence[BeliefSnapshot]],
    *,
    gram_count: int,
) -> tuple[tuple[BeliefSnapshot, ...], ...]:
    """Compose a matched-width mixture from nested shared-stream prefixes."""

    if gram_count not in MIXTURE_GRAM_COUNTS:
        raise ValueError(f"gram_count must be one of {MIXTURE_GRAM_COUNTS}")
    uniform_count = FRESH_PROPOSALS - gram_count
    if len(gram) != STAGES or len(uniform) != STAGES:
        raise ValueError("proposal banks must contain four stages")
    result: list[tuple[BeliefSnapshot, ...]] = []
    for gram_stage, uniform_stage in zip(gram, uniform, strict=True):
        if len(gram_stage) != len(uniform_stage) or not gram_stage:
            raise ValueError("GRAM and uniform stage task counts must match")
        composed: list[BeliefSnapshot] = []
        for gram_task, uniform_task in zip(gram_stage, uniform_stage, strict=True):
            if (
                len(gram_task.particle_codes) != FRESH_PROPOSALS
                or len(uniform_task.particle_codes) != FRESH_PROPOSALS
            ):
                raise ValueError("source banks must each contain exactly 32 proposals")
            codes = (
                gram_task.particle_codes[:gram_count]
                + uniform_task.particle_codes[:uniform_count]
            )
            if len(codes) != FRESH_PROPOSALS:
                raise AssertionError("every mixture must contain exactly 32 fresh codes")
            composed.append(_snapshot_from_particles(codes))
        result.append(tuple(composed))
    return tuple(result)


def _proposal_family(
    gram: Sequence[Sequence[BeliefSnapshot]],
    uniform: Sequence[Sequence[BeliefSnapshot]],
    latin: Sequence[Sequence[BeliefSnapshot]],
) -> dict[str, tuple[tuple[BeliefSnapshot, ...], ...]]:
    family = {
        f"gram{gram_count}_uniform{FRESH_PROPOSALS - gram_count}": (
            _compose_mixture_stages(gram, uniform, gram_count=gram_count)
        )
        for gram_count in MIXTURE_GRAM_COUNTS
    }
    family["latin_pairwise_cover32"] = tuple(tuple(stage) for stage in latin)
    for stages in family.values():
        if any(
            len(snapshot.particle_codes) != FRESH_PROPOSALS
            for stage in stages
            for snapshot in stage
        ):
            raise AssertionError("proposal budget mismatch")
    return family


def _mixture_gram_count(method_name: str) -> int | None:
    if method_name == "latin_pairwise_cover32":
        return None
    prefix, separator, suffix = method_name.partition("_uniform")
    if separator != "_uniform" or not prefix.startswith("gram") or not suffix.isdigit():
        raise ValueError(f"unrecognized proposal method {method_name!r}")
    gram_count = int(prefix.removeprefix("gram"))
    if gram_count not in MIXTURE_GRAM_COUNTS or int(suffix) != FRESH_PROPOSALS - gram_count:
        raise ValueError(f"invalid matched mixture name {method_name!r}")
    return gram_count


def _load_gram(torch: Any, args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    from prp_wm.gram_causal_rules import GRAMFactorizedCausalK4
    from scripts.run_expected_discrete_causal_coverage import _load_audited_executor

    executor, executor_metadata = _load_audited_executor(
        torch, args.executor_checkpoint.resolve(), device
    )
    checkpoint = torch.load(
        args.gram_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("checkpoint_schema_version") != "prp-wm.gram-factorized-causal-k4.v1":
        raise SystemExit("unexpected GRAM checkpoint schema")
    cli = checkpoint.get("cli_arguments", {})
    bounds = checkpoint.get("guidance_log_variance_bounds", (-8.0, 4.0))
    model = GRAMFactorizedCausalK4(
        executor,
        recursive_steps=int(checkpoint["recursive_steps"]),
        guidance_dim=int(checkpoint["guidance_dim"]),
        attention_layers=int(cli.get("attention_layers", 2)),
        temperature=float(cli.get("factor_temperature_end", args.temperature)),
        minimum_log_variance=float(bounds[0]),
        maximum_log_variance=float(bounds[1]),
        initial_log_variance=float(checkpoint.get("initial_guidance_log_variance", -2.0)),
        truncate_between_steps=bool(checkpoint.get("truncate_between_recursive_steps", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, {"executor": executor_metadata, "gram": checkpoint}


def _gram_proposal_stages(
    *,
    torch: Any,
    gram: Any,
    batches: Sequence[Any],
    seed: int,
    recursive_steps: int,
    temperature: float,
) -> tuple[tuple[BeliefSnapshot, ...], ...]:
    if len(batches) != STAGES:
        raise ValueError("the sequential protocol must contain four batches")
    stages: list[tuple[BeliefSnapshot, ...]] = []
    with torch.no_grad():
        for stage, batch in enumerate(batches):
            inference = gram.sample_width_candidates(
                batch,
                width=FRESH_PROPOSALS,
                recursive_steps=recursive_steps,
                seed=_stream_seed(seed, "gram_common_random_numbers", stage),
                temperature=temperature,
                sample_noise=True,
            )
            snapshots = _snapshots_from_inference(inference, unique_uniform=False)
            if any(len(snapshot.particle_codes) != FRESH_PROPOSALS for snapshot in snapshots):
                raise AssertionError("GRAM did not return exactly 32 proposals")
            stages.append(snapshots)
    return tuple(stages)


def _fresh_proposal_diagnostics(
    *,
    proposals: Sequence[Sequence[BeliefSnapshot]],
    tasks: Sequence[Any],
    histories: Sequence[Sequence[Sequence[Any]]],
    target_programs: Sequence[Any],
) -> tuple[dict[str, object], ...]:
    """Measure generated banks; target/version information is metrics-only."""

    from prp_wm.rulegrid import version_space

    records: list[dict[str, object]] = []
    for stage_index, (stage, stage_histories) in enumerate(
        zip(proposals, histories, strict=True)
    ):
        target_hits = 0
        compatible_recall = 0.0
        compatible_particle_rate = 0.0
        unique = 0
        marginals = [[0 for _ in range(4)] for _ in range(3)]
        for snapshot, task, history, target_program in zip(
            stage, tasks, stage_histories, target_programs, strict=True
        ):
            if len(snapshot.particle_codes) != FRESH_PROPOSALS:
                raise AssertionError("fresh proposal diagnostics require W32")
            codes = set(snapshot.particle_codes)
            compatible = {
                _factor_code(program)
                for program in version_space(history, task.privileged.palette)
            }
            target = _factor_code(target_program)
            target_hits += int(target in codes)
            compatible_recall += len(codes.intersection(compatible)) / len(compatible)
            compatible_particle_rate += sum(
                code in compatible for code in snapshot.particle_codes
            ) / FRESH_PROPOSALS
            unique += len(codes)
            for code in snapshot.particle_codes:
                for axis, value in enumerate(code):
                    marginals[axis][value] += 1
        count = len(stage)
        records.append(
            {
                "stage_index": stage_index,
                "fresh_proposals_per_task": FRESH_PROPOSALS,
                "target_code_in_fresh_proposals_rate": target_hits / count,
                "mean_fresh_unique_codes": unique / count,
                "mean_fresh_exact_version_space_recall": compatible_recall / count,
                "mean_compatible_fresh_particle_rate": compatible_particle_rate / count,
                "aggregate_factor_value_counts": marginals,
            }
        )
    return tuple(records)


def _conditional_probability(
    event: Sequence[bool], condition: Sequence[bool]
) -> dict[str, object]:
    if len(event) != len(condition):
        raise ValueError("event and condition must have equal lengths")
    denominator = sum(condition)
    numerator = sum(
        bool(event_value and condition_value)
        for event_value, condition_value in zip(event, condition, strict=True)
    )
    return {
        "numerator_tasks": numerator,
        "denominator_tasks": denominator,
        "probability": numerator / denominator if denominator else None,
    }


def _proposal_retention_gate_audit(
    *,
    fresh: Sequence[Sequence[BeliefSnapshot]],
    retained: Sequence[Sequence[BeliefSnapshot]],
    target_programs: Sequence[Any],
) -> dict[str, object]:
    """Audit fresh F_t, cumulative-union U_t, and retained belief B_t.

    ``U_t`` is the union of the actual fresh proposal codes from stages
    ``0..t`` for the same task.  It is not a version space and never receives
    the true code by construction.  ``B_t`` is the post-symbolic-filter,
    post-carry population returned by ``_symbolic_population_stages``.
    """

    if len(fresh) != STAGES or len(retained) != STAGES:
        raise ValueError("gate audit requires four fresh and retained stages")
    count = len(target_programs)
    if count <= 0 or any(len(stage) != count for stage in (*fresh, *retained)):
        raise ValueError("every gate stage must match the target count")
    targets = tuple(_factor_code(program) for program in target_programs)
    cumulative = [set() for _ in range(count)]
    fresh_presence: list[list[bool]] = []
    union_presence: list[list[bool]] = []
    belief_presence: list[list[bool]] = []
    stages: list[dict[str, object]] = []
    for stage_index, (fresh_stage, retained_stage) in enumerate(
        zip(fresh, retained, strict=True)
    ):
        current_fresh: list[bool] = []
        current_union: list[bool] = []
        current_belief: list[bool] = []
        for task_index, (fresh_snapshot, retained_snapshot, target) in enumerate(
            zip(fresh_stage, retained_stage, targets, strict=True)
        ):
            proposed = set(fresh_snapshot.particle_codes)
            cumulative[task_index].update(proposed)
            current_fresh.append(target in proposed)
            current_union.append(target in cumulative[task_index])
            current_belief.append(target in retained_snapshot.unique_codes)
        fresh_presence.append(current_fresh)
        union_presence.append(current_union)
        belief_presence.append(current_belief)
        retention = _conditional_probability(current_belief, current_union)
        no_injection = _conditional_probability(
            current_union, current_belief
        )
        stages.append(
            {
                "stage_index": stage_index,
                "p_F_t_fresh_target": sum(current_fresh) / count,
                "p_U_t_cumulative_union_target": sum(current_union) / count,
                "p_B_t_retained_target": sum(current_belief) / count,
                "retention_p_B_t_given_U_t": retention,
                "no_injection_p_U_t_given_B_t": no_injection,
                "target_presence_set_equality_U_t_equals_B_t_rate": sum(
                    union == belief
                    for union, belief in zip(
                        current_union, current_belief, strict=True
                    )
                )
                / count,
            }
        )
    return {
        "definitions": {
            "F_t": "target occurs in exactly the W32 fresh bank at stage t",
            "U_t": "target occurs in the union of fresh banks from stages 0..t",
            "B_t": "target occurs in the symbolic-filtered retained population at stage t",
        },
        "stages": stages,
        "fresh_recovery_p_F3_given_not_U2": _conditional_probability(
            fresh_presence[3], [not value for value in union_presence[2]]
        ),
        "belief_recovery_p_B3_given_not_B2": _conditional_probability(
            belief_presence[3], [not value for value in belief_presence[2]]
        ),
        "neutral_retention_p_B2_given_B1": _conditional_probability(
            belief_presence[2], belief_presence[1]
        ),
    }


def _mixture_target_source_attribution(
    *,
    gram: Sequence[Sequence[BeliefSnapshot]],
    uniform: Sequence[Sequence[BeliefSnapshot]],
    gram_count: int,
    target_programs: Sequence[Any],
) -> dict[str, object]:
    """Attribute fresh target hits to the two mixture components, metrics-only."""

    if gram_count not in MIXTURE_GRAM_COUNTS:
        raise ValueError(f"gram_count must be one of {MIXTURE_GRAM_COUNTS}")
    if len(gram) != STAGES or len(uniform) != STAGES:
        raise ValueError("source attribution requires four stages")
    uniform_count = FRESH_PROPOSALS - gram_count
    targets = tuple(_factor_code(program) for program in target_programs)
    records: list[dict[str, object]] = []
    for stage_index, (gram_stage, uniform_stage) in enumerate(
        zip(gram, uniform, strict=True)
    ):
        counts = Counter({
            "gram_only": 0,
            "uniform_only": 0,
            "both": 0,
            "neither": 0,
        })
        for gram_snapshot, uniform_snapshot, target in zip(
            gram_stage, uniform_stage, targets, strict=True
        ):
            in_gram = target in gram_snapshot.particle_codes[:gram_count]
            in_uniform = target in uniform_snapshot.particle_codes[:uniform_count]
            if in_gram and in_uniform:
                category = "both"
            elif in_gram:
                category = "gram_only"
            elif in_uniform:
                category = "uniform_only"
            else:
                category = "neither"
            counts[category] += 1
        total = len(targets)
        records.append(
            {
                "stage_index": stage_index,
                "gram_prefix_proposals": gram_count,
                "uniform_prefix_proposals": uniform_count,
                "counts": dict(counts),
                "rates": {key: value / total for key, value in counts.items()},
            }
        )
    return {
        "scope": "fresh component prefixes before symbolic filtering and carry",
        "true_code_used_for_metrics_only": True,
        "stages": records,
    }


def _uniform_analytic_baseline() -> dict[str, object]:
    """Exact target-hit baseline for iid W32 draws from 64 rule codes."""

    miss_one_draw = 63.0 / 64.0
    fresh_hit = 1.0 - miss_one_draw**FRESH_PROPOSALS
    stages = []
    for stage in range(STAGES):
        cumulative_hit = 1.0 - miss_one_draw ** (FRESH_PROPOSALS * (stage + 1))
        stages.append(
            {
                "stage_index": stage,
                "p_F_t_fresh_target": fresh_hit,
                "p_U_t_cumulative_union_target": cumulative_hit,
                "p_B_t_retained_target_under_exact_symbolic_carry": cumulative_hit,
            }
        )
    return {
        "assumptions": (
            "iid draws with replacement from 64 codes; independent stages; target "
            "is always symbolically compatible; exact carry never drops it"
        ),
        "expected_unique_codes_per_fresh_bank": 64.0 * (1.0 - miss_one_draw**FRESH_PROPOSALS),
        "stages": stages,
        "fresh_recovery_p_F3_given_not_U2": fresh_hit,
        "belief_recovery_p_B3_given_not_B2": fresh_hit,
        "retention_p_B_t_given_U_t": 1.0,
        "neutral_retention_p_B2_given_B1": 1.0,
    }


def _pooled_inputs(
    tasks: Sequence[Any], paired: PairedEvidence, repeats: int
) -> tuple[tuple[Any, ...], PairedEvidence]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    pooled_tasks = tuple(task for _ in range(repeats) for task in tasks)
    pooled = PairedEvidence(
        steps=tuple(step for _ in range(repeats) for step in paired.steps),
        factual_histories=tuple(
            tuple(history for _ in range(repeats) for history in stage)
            for stage in paired.factual_histories
        ),
        counterfactual_histories=tuple(
            tuple(history for _ in range(repeats) for history in stage)
            for stage in paired.counterfactual_histories
        ),
        factual_programs=tuple(
            program for _ in range(repeats) for program in paired.factual_programs
        ),
        counterfactual_programs=tuple(
            program for _ in range(repeats) for program in paired.counterfactual_programs
        ),
    )
    return pooled_tasks, pooled


def _pool_stages(
    seed_stages: Sequence[Sequence[Sequence[BeliefSnapshot]]],
) -> tuple[tuple[BeliefSnapshot, ...], ...]:
    if not seed_stages or any(len(stages) != STAGES for stages in seed_stages):
        raise ValueError("each seed must contain four stages")
    return tuple(
        tuple(snapshot for stages in seed_stages for snapshot in stages[stage])
        for stage in range(STAGES)
    )


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def _gate_comparison_row(name: str, gates: dict[str, Any]) -> dict[str, object]:
    branches = (gates["factual"], gates["counterfactual"])
    row: dict[str, object] = {
        "method": name,
        "mean_p_F3_fresh_target": _mean_defined(
            [branch["stages"][3]["p_F_t_fresh_target"] for branch in branches]
        ),
        "mean_p_U3_cumulative_target": _mean_defined(
            [
                branch["stages"][3]["p_U_t_cumulative_union_target"]
                for branch in branches
            ]
        ),
        "mean_p_B3_retained_target": _mean_defined(
            [branch["stages"][3]["p_B_t_retained_target"] for branch in branches]
        ),
        "mean_fresh_recovery_p_F3_given_not_U2": _mean_defined(
            [
                branch["fresh_recovery_p_F3_given_not_U2"]["probability"]
                for branch in branches
            ]
        ),
        "mean_belief_recovery_p_B3_given_not_B2": _mean_defined(
            [
                branch["belief_recovery_p_B3_given_not_B2"]["probability"]
                for branch in branches
            ]
        ),
        "mean_retention_p_B3_given_U3": _mean_defined(
            [
                branch["stages"][3]["retention_p_B_t_given_U_t"]["probability"]
                for branch in branches
            ]
        ),
        "mean_neutral_retention_p_B2_given_B1": _mean_defined(
            [
                branch["neutral_retention_p_B2_given_B1"]["probability"]
                for branch in branches
            ]
        ),
    }
    score_fields = (
        "mean_p_B3_retained_target",
        "mean_fresh_recovery_p_F3_given_not_U2",
        "mean_belief_recovery_p_B3_given_not_B2",
        "mean_retention_p_B3_given_U3",
        "mean_neutral_retention_p_B2_given_B1",
    )
    row["multi_gate_score"] = _mean_defined(
        [row[field] for field in score_fields]  # type: ignore[list-item]
    )
    return row


def _validate_args(args: argparse.Namespace) -> None:
    if args.eval_tasks <= 0:
        raise SystemExit("--eval-tasks must be positive")
    if not args.inference_seeds or len(set(args.inference_seeds)) != len(args.inference_seeds):
        raise SystemExit("--inference-seeds must be non-empty and unique")
    if args.recursive_steps is not None and args.recursive_steps <= 0:
        raise SystemExit("--recursive-steps must be positive")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if args.carry_limit <= 0:
        raise SystemExit("--carry-limit must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from prp_wm.pilot import make_pilot_tasks
    from prp_wm.rulegrid import version_space
    from scripts.run_causal_mechanism_coverage import _configure_determinism, _resolve_device
    from scripts.run_expected_discrete_causal_coverage import _build_context_pool

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, args.inference_seeds[0])
    gram, checkpoints = _load_gram(torch, args, device)
    context_fold = checkpoints["gram"].get("context_fold")
    if context_fold is None:
        raise SystemExit("the GRAM checkpoint must declare a Latin context fold")
    tasks = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split=args.eval_split,
        master_seed=args.data_master_seed,
        diagnostic_indices=(0,),
        count=args.eval_tasks,
        heldout=True,
        factor_ids_for_program=_factor_code,
        version_space=version_space,
        context_fold=int(context_fold),
    )
    paired = _build_paired_evidence(tasks)
    factual_batches = tuple(
        _support_batch(torch, tasks, histories, device=device)
        for histories in paired.factual_histories
    )
    counterfactual_batches = tuple(
        _support_batch(torch, tasks, histories, device=device)
        for histories in paired.counterfactual_histories
    )
    recursive_steps = gram.recursive_steps if args.recursive_steps is None else args.recursive_steps

    started = time.perf_counter()
    per_seed: dict[str, dict[str, Any]] = {}
    pooled_factual: dict[str, list[tuple[BeliefSnapshot, ...]]] = {}
    pooled_counterfactual: dict[str, list[tuple[BeliefSnapshot, ...]]] = {}
    pooled_fresh_factual: dict[str, list[tuple[BeliefSnapshot, ...]]] = {}
    pooled_fresh_counterfactual: dict[str, list[tuple[BeliefSnapshot, ...]]] = {}
    pooled_source_gram_factual: list[tuple[tuple[BeliefSnapshot, ...], ...]] = []
    pooled_source_gram_counterfactual: list[tuple[tuple[BeliefSnapshot, ...], ...]] = []
    pooled_source_uniform: list[tuple[tuple[BeliefSnapshot, ...], ...]] = []

    for inference_seed in args.inference_seeds:
        uniform = _iid_uniform_proposal_stages(task_count=len(tasks), seed=inference_seed)
        latin = _latin_cover_proposal_stages(task_count=len(tasks), seed=inference_seed)
        gram_factual = _gram_proposal_stages(
            torch=torch,
            gram=gram,
            batches=factual_batches,
            seed=inference_seed,
            recursive_steps=recursive_steps,
            temperature=args.temperature,
        )
        gram_counterfactual = _gram_proposal_stages(
            torch=torch,
            gram=gram,
            batches=counterfactual_batches,
            seed=inference_seed,
            recursive_steps=recursive_steps,
            temperature=args.temperature,
        )
        # At t0, both branches have identical evidence.  Equal outputs verify
        # that their same stage seed really drives common random numbers.
        if gram_factual[0] != gram_counterfactual[0]:
            raise AssertionError("paired t0 GRAM proposals differ despite identical evidence/seed")
        pooled_source_gram_factual.append(gram_factual)
        pooled_source_gram_counterfactual.append(gram_counterfactual)
        pooled_source_uniform.append(uniform)

        factual_family = _proposal_family(gram_factual, uniform, latin)
        counterfactual_family = _proposal_family(gram_counterfactual, uniform, latin)
        seed_methods: dict[str, Any] = {}
        for method_name in factual_family:
            factual_fresh = factual_family[method_name]
            counterfactual_fresh = counterfactual_family[method_name]
            factual_retained = _symbolic_population_stages(
                fresh_proposals_by_stage=factual_fresh,
                tasks=tasks,
                histories_by_stage=paired.factual_histories,
                carry_limit=args.carry_limit,
            )
            counterfactual_retained = _symbolic_population_stages(
                fresh_proposals_by_stage=counterfactual_fresh,
                tasks=tasks,
                histories_by_stage=paired.counterfactual_histories,
                carry_limit=args.carry_limit,
            )
            gate_audit = {
                "factual": _proposal_retention_gate_audit(
                    fresh=factual_fresh,
                    retained=factual_retained,
                    target_programs=paired.factual_programs,
                ),
                "counterfactual": _proposal_retention_gate_audit(
                    fresh=counterfactual_fresh,
                    retained=counterfactual_retained,
                    target_programs=paired.counterfactual_programs,
                ),
            }
            gram_count = _mixture_gram_count(method_name)
            if gram_count is None:
                source_attribution: dict[str, object] = {
                    "kind": "not_applicable_single_latin_covering_source"
                }
            else:
                source_attribution = {
                    "kind": "fresh_target_hit_attribution_between_mixture_prefixes",
                    "factual": _mixture_target_source_attribution(
                        gram=gram_factual,
                        uniform=uniform,
                        gram_count=gram_count,
                        target_programs=paired.factual_programs,
                    ),
                    "counterfactual": _mixture_target_source_attribution(
                        gram=gram_counterfactual,
                        uniform=uniform,
                        gram_count=gram_count,
                        target_programs=paired.counterfactual_programs,
                    ),
                }
            seed_methods[method_name] = {
                "report": _method_report(
                    tasks=tasks,
                    paired=paired,
                    factual=factual_retained,
                    counterfactual=counterfactual_retained,
                ),
                "fresh_proposals": {
                    "factual": _fresh_proposal_diagnostics(
                        proposals=factual_fresh,
                        tasks=tasks,
                        histories=paired.factual_histories,
                        target_programs=paired.factual_programs,
                    ),
                    "counterfactual": _fresh_proposal_diagnostics(
                        proposals=counterfactual_fresh,
                        tasks=tasks,
                        histories=paired.counterfactual_histories,
                        target_programs=paired.counterfactual_programs,
                    ),
                },
                "proposal_retention_gate_audit": gate_audit,
                "mixture_target_source_attribution": source_attribution,
            }
            pooled_factual.setdefault(method_name, []).append(factual_retained)
            pooled_counterfactual.setdefault(method_name, []).append(counterfactual_retained)
            pooled_fresh_factual.setdefault(method_name, []).append(factual_fresh)
            pooled_fresh_counterfactual.setdefault(method_name, []).append(counterfactual_fresh)
        per_seed[str(inference_seed)] = seed_methods

    pooled_tasks, pooled_paired = _pooled_inputs(tasks, paired, len(args.inference_seeds))
    source_gram_factual = _pool_stages(pooled_source_gram_factual)
    source_gram_counterfactual = _pool_stages(pooled_source_gram_counterfactual)
    source_uniform = _pool_stages(pooled_source_uniform)
    methods: dict[str, Any] = {}
    for method_name in pooled_factual:
        factual = _pool_stages(pooled_factual[method_name])
        counterfactual = _pool_stages(pooled_counterfactual[method_name])
        factual_fresh = _pool_stages(pooled_fresh_factual[method_name])
        counterfactual_fresh = _pool_stages(pooled_fresh_counterfactual[method_name])
        gate_audit = {
            "factual": _proposal_retention_gate_audit(
                fresh=factual_fresh,
                retained=factual,
                target_programs=pooled_paired.factual_programs,
            ),
            "counterfactual": _proposal_retention_gate_audit(
                fresh=counterfactual_fresh,
                retained=counterfactual,
                target_programs=pooled_paired.counterfactual_programs,
            ),
        }
        gram_count = _mixture_gram_count(method_name)
        if gram_count is None:
            source_attribution = {
                "kind": "not_applicable_single_latin_covering_source"
            }
        else:
            source_attribution = {
                "kind": "fresh_target_hit_attribution_between_mixture_prefixes",
                "factual": _mixture_target_source_attribution(
                    gram=source_gram_factual,
                    uniform=source_uniform,
                    gram_count=gram_count,
                    target_programs=pooled_paired.factual_programs,
                ),
                "counterfactual": _mixture_target_source_attribution(
                    gram=source_gram_counterfactual,
                    uniform=source_uniform,
                    gram_count=gram_count,
                    target_programs=pooled_paired.counterfactual_programs,
                ),
            }
        methods[method_name] = {
            "proposal_mode": method_name,
            "fresh_proposals_per_task_stage": FRESH_PROPOSALS,
            "carry_limit": args.carry_limit,
            "symbolic_verifier": True,
            "no_exact_bank_fallback": True,
            "true_code_or_version_space_in_proposal_generation": False,
            "report": _method_report(
                tasks=pooled_tasks,
                paired=pooled_paired,
                factual=factual,
                counterfactual=counterfactual,
            ),
            "fresh_proposals": {
                "factual": _fresh_proposal_diagnostics(
                    proposals=factual_fresh,
                    tasks=pooled_tasks,
                    histories=pooled_paired.factual_histories,
                    target_programs=pooled_paired.factual_programs,
                ),
                "counterfactual": _fresh_proposal_diagnostics(
                    proposals=counterfactual_fresh,
                    tasks=pooled_tasks,
                    histories=pooled_paired.counterfactual_histories,
                    target_programs=pooled_paired.counterfactual_programs,
                ),
            },
            "proposal_retention_gate_audit": gate_audit,
            "mixture_target_source_attribution": source_attribution,
        }

    leaderboard = sorted(
        (
            _gate_comparison_row(
                name, payload["proposal_retention_gate_audit"]
            )
            for name, payload in methods.items()
        ),
        key=lambda row: (-float(row["multi_gate_score"]), str(row["method"])),
    )
    elapsed = time.perf_counter() - started
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "experiment": "matched_w32_gram_proposal_source_ablation",
        "status": "complete",
        "cli_arguments": _json_cli(args),
        "tasks_per_seed": len(tasks),
        "pooled_seed_task_pairs": len(pooled_tasks),
        "context_fold": context_fold,
        "checkpoint_model_seed": checkpoints["gram"].get("model_seed"),
        "device": str(device),
        "elapsed_seconds": round(elapsed, 6),
        "protocol": {
            "stage_names": ["t0_support", "t1_partial", "t2_neutral", "t3_strong"],
            "factual_counterfactual_public_states_actions_matched": True,
            "model_selects_actions": False,
            "active_policy_claim_eligible": False,
        },
        "proposal_budget_contract": {
            "fresh_proposals_every_task_stage": FRESH_PROPOSALS,
            "mixture_gram_counts": list(MIXTURE_GRAM_COUNTS),
            "mixture_uniform_counts": [FRESH_PROPOSALS - value for value in MIXTURE_GRAM_COUNTS],
            "mixtures_use_nested_prefixes_of_shared_w32_streams": True,
            "fresh_stream_changes_between_stages": True,
            "factual_counterfactual_same_stage_gram_seed": True,
            "factual_counterfactual_uniform_codes_identical": True,
            "factual_counterfactual_latin_codes_identical": True,
            "carry_limit_identical_all_methods": args.carry_limit,
            "true_code_or_version_space_passed_to_proposer": False,
            "missing_compatible_code_injected": False,
        },
        "symbolic_verifier_scope": {
            "privileged": True,
            "used_after_proposal_generation_only": True,
            "retains_compatible_fresh_or_carried_codes": True,
            "when_none_compatible_retains_unfiltered_failure_candidates": True,
        },
        "latin_pairwise_cover": {
            "codes": FRESH_PROPOSALS,
            "unique_codes": FRESH_PROPOSALS,
            "each_axis_value_count": 8,
            "each_ordered_pair_count_on_every_axis_pair": 2,
            "randomized_per_task_stage": True,
        },
        "uniform_iid_analytic_baseline": _uniform_analytic_baseline(),
        "multi_gate_leaderboard": {
            "sort_key": (
                "unweighted mean of pooled branch-mean B3 coverage, F3|not U2, "
                "B3|not B2, B3|U3 retention, and B2|B1 neutral retention"
            ),
            "not_ranked_on_B3_alone": True,
            "rows": leaderboard,
        },
        "methods": methods,
        "per_seed": per_seed,
        "checkpoints": {
            "executor": {
                "path": str(args.executor_checkpoint.resolve()),
                "sha256": _sha256_file(args.executor_checkpoint.resolve()),
                "schema": checkpoints["executor"].get("checkpoint_schema_version"),
            },
            "gram": {
                "path": str(args.gram_checkpoint.resolve()),
                "sha256": _sha256_file(args.gram_checkpoint.resolve()),
                "schema": checkpoints["gram"].get("checkpoint_schema_version"),
            },
        },
        "source_sha256": _source_sha256(),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
