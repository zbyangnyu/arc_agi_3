#!/usr/bin/env python3
"""Let a public rule-belief checkpoint play a probe-then-door game.

The controller receives only RuleGrid public support, public candidate probes,
and the next-state returned after each selected probe.  Candidate selection is
uniform without replacement in this first audit so action selection and belief
assimilation are measured separately.  After a fixed exploration budget, the
model chooses one of four doors for the held-out causal factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULT_SCHEMA_VERSION = "prp-wm.public-belief-rulegrid-door-game.v1"
DEFAULT_CHECKPOINT = ROOT / (
    "runs/raw_palette_invariant_atom_matched_hard600_fold0_seed20260863/"
    "checkpoint_last.pt"
)
AUDITED_SOURCE_FILES = (
    "prp_wm/public_version_k4.py",
    "prp_wm/rulegrid.py",
    "scripts/run_public_belief_door_game.py",
    "scripts/run_public_version_space_k4.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-axis", type=int, default=64)
    parser.add_argument("--budgets", type=int, nargs="+", default=(0, 1, 2, 4, 8))
    parser.add_argument(
        "--probe-context",
        choices=("marked", "unmarked", "both"),
        default="both",
        help="whether online results carry the controller-owned probe-result bit",
    )
    parser.add_argument("--seed", type=int, default=20260870)
    parser.add_argument("--data-master-seed", type=int, default=2026071601)
    parser.add_argument("--split", default="public-belief-door-validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trace-groups", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_index(axis: Any) -> int:
    from prp_wm.rulegrid import Axis

    return {
        Axis.COLLISION: 0,
        Axis.TRIGGER: 1,
        Axis.RELATION: 2,
    }[axis]


def _build_axis_tasks(
    axis: Any,
    *,
    groups: int,
    split: str,
    master_seed: int,
) -> tuple[Any, ...]:
    from prp_wm.rulegrid import (
        ALL_COLLISIONS,
        ALL_RELATIONS,
        ALL_TRIGGERS,
        RuleProgram,
        make_rulegrid_task,
    )

    modes = (ALL_COLLISIONS, ALL_TRIGGERS, ALL_RELATIONS)
    axis_index = _axis_index(axis)
    result = []
    for group in range(groups):
        factor_ids = [
            group % 4,
            (group // 4) % 4,
            (group // 16) % 4,
        ]
        for mode_index in range(4):
            selected = list(factor_ids)
            selected[axis_index] = mode_index
            program = RuleProgram(
                modes[0][selected[0]],
                modes[1][selected[1]],
                modes[2][selected[2]],
            )
            result.append(
                make_rulegrid_task(
                    program,
                    axis,
                    group,
                    split=split,
                    master_seed=master_seed,
                    diagnostic_indices=(0,),
                )
            )
    return tuple(result)


def _probe_orders(
    axis: Any,
    *,
    groups: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    result = []
    for group in range(groups):
        order = list(range(8))
        random.Random(f"{seed}|{axis.value}|{group}").shuffle(order)
        result.extend((tuple(order),) * 4)
    return tuple(result)


def _public_histories(
    tasks: tuple[Any, ...],
    orders: tuple[tuple[int, ...], ...],
    budget: int,
) -> tuple[tuple[Any, ...], ...]:
    from prp_wm.rulegrid import RuleGridTransition

    histories = []
    for task, order in zip(tasks, orders, strict=True):
        history = list(task.inference.support)
        for candidate_index in order[:budget]:
            probe = task.inference.active_candidates[candidate_index]
            history.append(
                RuleGridTransition(
                    probe.state,
                    probe.action,
                    task.privileged.active_targets[candidate_index],
                )
            )
        histories.append(tuple(history))
    return tuple(histories)


def _infer_factor_probabilities(
    *,
    torch: Any,
    model: Any,
    histories: tuple[tuple[Any, ...], ...],
    online_steps: int,
    marked: bool,
    batch_size: int,
    device: Any,
):
    from scripts.run_gram_public_coverage_finetune import (
        _raw_public_history_batch,
    )

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            selected = histories[start : start + batch_size]
            batch = _raw_public_history_batch(
                torch,
                selected,
                device=device,
            )
            probe_mask = None
            if marked:
                probe_mask = torch.zeros_like(batch.support_mask)
                if online_steps:
                    probe_mask[:, -online_steps:] = True
            belief = model.infer_factor_belief(
                batch,
                is_agent_probe_result=probe_mask,
            )
            outputs.append(belief.factor_probabilities.cpu())
    return torch.cat(outputs, dim=0)


def _evaluate_budget(
    *,
    torch: Any,
    model: Any,
    tasks: tuple[Any, ...],
    orders: tuple[tuple[int, ...], ...],
    axis: Any,
    budget: int,
    marked: bool,
    batch_size: int,
    device: Any,
) -> dict[str, object]:
    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.rulegrid import version_space
    from scripts.run_public_version_space_k4 import (
        _symmetry_expanded_version_space_mask,
    )

    histories = _public_histories(tasks, orders, budget)
    factor_probabilities = _infer_factor_probabilities(
        torch=torch,
        model=model,
        histories=histories,
        online_steps=budget,
        marked=marked,
        batch_size=batch_size,
        device=device,
    )
    axis_index = _axis_index(axis)
    door_probabilities = factor_probabilities[:, axis_index]
    targets = torch.tensor(
        [
            rule_program_factor_ids(task.privileged.true_program)[axis_index]
            for task in tasks
        ],
        dtype=torch.long,
    )
    guesses = door_probabilities.argmax(dim=-1)
    target_mass = door_probabilities[
        torch.arange(len(tasks)),
        targets,
    ]
    entropy = -(
        door_probabilities.clamp_min(1e-8)
        * door_probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1)

    # A raw public history does not name the payload-p1/payload-p2 roles.  Keep
    # the strict hidden-name door reward, but also score the observable rule
    # equivalence class used by this checkpoint's teacher.  Materialize this
    # target only after inference so it cannot leak into the controller.
    symmetry_compatible = _symmetry_expanded_version_space_mask(
        torch,
        model,
        tasks,
        device=device,
        histories=histories,
    )
    symmetry_factor_sets = model._factor_value_masks(
        model.factor_bank,
        symmetry_compatible,
    ).cpu()
    symmetry_door_targets = symmetry_factor_sets[:, axis_index]
    symmetry_target_counts = symmetry_door_targets.sum(dim=-1)
    if bool((symmetry_target_counts == 0).any().item()):
        raise AssertionError("public symmetry target set cannot be empty")
    if not bool(
        symmetry_door_targets[
            torch.arange(len(tasks)),
            targets,
        ].all().item()
    ):
        raise AssertionError("hidden door must belong to its public target set")
    symmetry_compatible_guesses = symmetry_door_targets.gather(
        1,
        guesses[:, None],
    ).squeeze(1)
    symmetry_target_mass = (
        door_probabilities * symmetry_door_targets.to(door_probabilities.dtype)
    ).sum(dim=-1)
    publicly_identified = symmetry_target_counts.eq(1)
    identified_and_correct = publicly_identified & guesses.eq(targets)
    singleton_exact_win = (
        float(guesses[publicly_identified].eq(targets[publicly_identified]).float().mean())
        if bool(publicly_identified.any().item())
        else None
    )

    # Simulator-side success audit, deliberately computed after model inference.
    symbolically_identified = []
    for task, history in zip(tasks, histories, strict=True):
        values = {
            rule_program_factor_ids(program)[axis_index]
            for program in version_space(history, task.privileged.palette)
        }
        symbolically_identified.append(len(values) == 1)
    kinds_seen: dict[str, int] = {}
    for task, order in zip(tasks, orders, strict=True):
        for candidate_index in order[:budget]:
            kind = task.privileged.candidate_kinds[candidate_index]
            kinds_seen[kind] = kinds_seen.get(kind, 0) + 1
    return {
        "axis": axis.value,
        "budget": budget,
        "probe_context": "marked" if marked else "unmarked",
        "tasks": len(tasks),
        "terminal_win_rate": float(guesses.eq(targets).float().mean()),
        "public_equivalence_class_win_rate": float(
            symmetry_compatible_guesses.float().mean()
        ),
        "mean_true_door_probability": float(target_mass.mean()),
        "mean_public_equivalence_class_probability": float(
            symmetry_target_mass.mean()
        ),
        "mean_public_equivalence_class_nll": float(
            -symmetry_target_mass.clamp_min(1e-8).log().mean()
        ),
        "mean_confidence": float(door_probabilities.max(dim=-1).values.mean()),
        "mean_door_entropy": float(entropy.mean()),
        "simulator_axis_identified_rate": sum(symbolically_identified)
        / len(symbolically_identified),
        "public_symmetry_axis_identified_rate": float(
            publicly_identified.float().mean()
        ),
        "public_identified_and_correct_rate": float(
            identified_and_correct.float().mean()
        ),
        "singleton_only_exact_win_rate": singleton_exact_win,
        "mean_public_equivalence_class_size": float(
            symmetry_target_counts.float().mean()
        ),
        "uniform_equivalence_class_exact_door_ceiling": float(
            symmetry_target_counts.float().reciprocal().mean()
        ),
        "selected_probe_kinds": kinds_seen,
        "controller_reads_candidate_kind": False,
        "controller_reads_true_program_or_target": False,
    }


def _episode_traces(
    *,
    torch: Any,
    model: Any,
    tasks: tuple[Any, ...],
    orders: tuple[tuple[int, ...], ...],
    axis: Any,
    max_budget: int,
    marked: bool,
    trace_groups: int,
    device: Any,
) -> list[dict[str, object]]:
    from prp_wm.latent_rules import rule_program_factor_ids

    axis_index = _axis_index(axis)
    traces = []
    for task_index in range(min(len(tasks), 4 * trace_groups)):
        task = tasks[task_index]
        target = rule_program_factor_ids(task.privileged.true_program)[axis_index]
        order = orders[task_index]
        steps = []
        for budget in range(max_budget + 1):
            history = _public_histories((task,), (order,), budget)
            probabilities = _infer_factor_probabilities(
                torch=torch,
                model=model,
                histories=history,
                online_steps=budget,
                marked=marked,
                batch_size=1,
                device=device,
            )[0, axis_index]
            guess = int(probabilities.argmax().item())
            step: dict[str, object] = {
                "probes_used": budget,
                "door_probabilities": [
                    round(float(value), 8) for value in probabilities
                ],
                "door_guess": guess,
                "would_win": guess == target,
            }
            if budget < max_budget:
                candidate_index = order[budget]
                probe = task.inference.active_candidates[candidate_index]
                step["next_public_probe"] = {
                    "candidate_index": candidate_index,
                    "probe_id": probe.probe_id,
                }
                step["audit_only_candidate_kind"] = (
                    task.privileged.candidate_kinds[candidate_index]
                )
            steps.append(step)
        traces.append(
            {
                "task_id": task.inference.task_id,
                "axis": axis.value,
                "hidden_door_materialized_after_play": target,
                "probe_context": "marked" if marked else "unmarked",
                "steps": steps,
            }
        )
    return traces


def main() -> None:
    args = parse_args()
    if args.groups_per_axis <= 0 or args.batch_size <= 0:
        raise SystemExit("groups-per-axis and batch-size must be positive")
    if args.trace_groups < 0:
        raise SystemExit("trace-groups must be non-negative")
    budgets = tuple(args.budgets)
    if not budgets or len(set(budgets)) != len(budgets):
        raise SystemExit("budgets must be non-empty and unique")
    if any(type(value) is not int or not 0 <= value <= 8 for value in budgets):
        raise SystemExit("each budget must be an integer in [0,8]")

    import torch
    from prp_wm.rulegrid import Axis
    from scripts.run_public_version_space_k4 import (
        load_public_version_k4_checkpoint,
    )

    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint, _, _ = load_public_version_k4_checkpoint(
        torch,
        checkpoint_path,
        device=device,
    )
    contexts = (
        (False, True)
        if args.probe_context == "both"
        else (args.probe_context == "marked",)
    )
    evaluations = []
    traces = []
    for axis in (Axis.COLLISION, Axis.TRIGGER, Axis.RELATION):
        tasks = _build_axis_tasks(
            axis,
            groups=args.groups_per_axis,
            split=args.split,
            master_seed=args.data_master_seed,
        )
        orders = _probe_orders(
            axis,
            groups=args.groups_per_axis,
            seed=args.seed,
        )
        for marked in contexts:
            for budget in budgets:
                evaluations.append(
                    _evaluate_budget(
                        torch=torch,
                        model=model,
                        tasks=tasks,
                        orders=orders,
                        axis=axis,
                        budget=budget,
                        marked=marked,
                        batch_size=args.batch_size,
                        device=device,
                    )
                )
            if args.trace_groups:
                traces.extend(
                    _episode_traces(
                        torch=torch,
                        model=model,
                        tasks=tasks,
                        orders=orders,
                        axis=axis,
                        max_budget=max(budgets),
                        marked=marked,
                        trace_groups=args.trace_groups,
                        device=device,
                    )
                )

    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "experiment": "public_rule_belief_probe_then_door_game",
        "status": "complete",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_model_type": checkpoint.get("model_type"),
        "checkpoint_version_head": checkpoint.get("version_head"),
        "split": args.split,
        "seed": args.seed,
        "data_master_seed": args.data_master_seed,
        "groups_per_axis": args.groups_per_axis,
        "tasks_per_axis": 4 * args.groups_per_axis,
        "budgets": list(budgets),
        "candidate_selection": "uniform_without_replacement",
        "terminal_action": "argmax held-out factor door",
        "terminal_reward": "one iff selected door equals simulator hidden factor",
        "evaluations": evaluations,
        "episode_traces": traces,
        "source_sha256": {
            name: _sha256(ROOT / name) for name in AUDITED_SOURCE_FILES
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(result_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
