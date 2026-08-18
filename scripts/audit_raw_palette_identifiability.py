#!/usr/bin/env python3
"""Audit latent-rule identifiability from raw public RuleGrid histories."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prp_wm.rulegrid import MASTER_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-fold", type=int, choices=range(4), default=0)
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--data-master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--split", default="gram-causal-composition")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.tasks <= 0:
        raise SystemExit("--tasks must be positive")

    from prp_wm.latent_rules import rule_program_factor_ids
    from prp_wm.pilot import TRIPLE_DIAGNOSTIC_INDICES, make_pilot_tasks
    from prp_wm.rulegrid import (
        ActionKind,
        ALL_TRIGGERS,
        GridAction,
        RuleGridProbe,
        Trigger,
        grid_with_cells,
        simulate,
        version_space,
    )
    from scripts.run_expected_discrete_causal_coverage import _build_context_pool

    tasks = _build_context_pool(
        make_pilot_tasks=make_pilot_tasks,
        split=args.split,
        master_seed=args.data_master_seed,
        diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        count=args.tasks,
        heldout=True,
        factor_ids_for_program=rule_program_factor_ids,
        version_space=version_space,
        context_fold=args.context_fold,
    )
    cardinalities: Counter[tuple[int, int]] = Counter()
    ambiguous_tasks = 0
    active_breaks = 0
    records: list[dict[str, object]] = []
    trigger_swap = {
        Trigger.TOGGLE: Trigger.RECOLOR,
        Trigger.RECOLOR: Trigger.TOGGLE,
    }

    for task in tasks:
        history = task.inference.support[:6]
        palette = task.privileged.palette
        swapped_palette = replace(
            palette,
            payload_p1=palette.payload_p2,
            payload_p2=palette.payload_p1,
        )
        canonical = version_space(history, palette)
        swapped = version_space(history, swapped_palette)
        expanded = tuple(sorted(set(canonical).union(swapped)))
        cardinalities[(len(canonical), len(expanded))] += 1
        codes = tuple(rule_program_factor_ids(program) for program in expanded)
        factor_sets = [sorted({code[axis] for code in codes}) for axis in range(3)]
        record: dict[str, object] = {
            "task_id": task.inference.task_id,
            "oracle_palette_version_space_size": len(canonical),
            "payload_name_swap_version_space_size": len(expanded),
            "factor_value_sets": factor_sets,
            "requires_more_than_k4_joint_hypotheses": len(expanded) > 4,
        }
        if len(expanded) > len(canonical):
            ambiguous_tasks += 1
            original = canonical[0]
            if original.trigger not in trigger_swap:
                raise AssertionError("only TOGGLE/RECOLOR may expand under p1/p2 swap")
            alternative = replace(
                original,
                trigger=trigger_swap[original.trigger],
            )
            if alternative not in swapped:
                raise AssertionError("swapped-palette partner is not history-consistent")
            activate = next(
                transition
                for transition in history
                if transition.action.kind is ActionKind.ACTIVATE
            )
            row, column = activate.action.coord
            payload_coord = (row, column + 1)
            socket_coord = (row, column + 2)
            observed_output_color = activate.next_state[payload_coord[0]][
                payload_coord[1]
            ]
            public_probe = RuleGridProbe(
                "raw-symmetry-break",
                grid_with_cells(
                    {
                        (row, column): activate.state[row][column],
                        payload_coord: observed_output_color,
                        socket_coord: activate.state[socket_coord[0]][socket_coord[1]],
                    }
                ),
                GridAction(ActionKind.ACTIVATE, (row, column)),
            )
            original_outcome = simulate(
                public_probe.state,
                public_probe.action,
                original,
                palette,
            )
            alternative_outcome = simulate(
                public_probe.state,
                public_probe.action,
                alternative,
                swapped_palette,
            )
            breaks = original_outcome != alternative_outcome
            active_breaks += int(breaks)
            record.update(
                {
                    "paired_trigger_codes": [
                        ALL_TRIGGERS.index(original.trigger),
                        ALL_TRIGGERS.index(alternative.trigger),
                    ],
                    "public_history_identical_under_pair": all(
                        simulate(
                            transition.state,
                            transition.action,
                            alternative,
                            swapped_palette,
                        )
                        == transition.next_state
                        for transition in history
                    ),
                    "observed-output-reuse_probe_breaks_symmetry": breaks,
                }
            )
        records.append(record)

    if active_breaks != ambiguous_tasks:
        raise AssertionError("every ambiguous history should admit the public break probe")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_files = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "prp_wm/rulegrid.py",
        REPOSITORY_ROOT / "prp_wm/pilot.py",
    )
    result = {
        "schema_version": "prp-wm.raw-palette-identifiability-audit.v1",
        "status": "complete",
        "context_fold": args.context_fold,
        "tasks": len(tasks),
        "controller_observation": "raw support state/action/observed next_state",
        "tested_symmetry": "swap simulator payload_p1 and payload_p2 role names",
        "oracle_to_symmetry_expanded_cardinality_histogram": {
            f"{before}->{after}": count
            for (before, after), count in sorted(cardinalities.items())
        },
        "non_identifiable_t0_tasks": ambiguous_tasks,
        "non_identifiable_t0_task_rate": ambiguous_tasks / len(tasks),
        "maximum_joint_hypotheses_after_symmetry_expansion": max(
            record["payload_name_swap_version_space_size"] for record in records
        ),
        "k4_joint_enumeration_sufficient_for_all_tasks": all(
            not record["requires_more_than_k4_joint_hypotheses"]
            for record in records
        ),
        "ambiguous_tasks_with_public_active_break_probe": active_breaks,
        "conclusion": (
            "named latent programs are not identifiable at t0 from raw observations; "
            "a factorized hypothesis set can retain the symmetry and an active probe "
            "that reuses the observed trigger output color separates the pair"
        ),
        "records": records,
        "source_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in source_files
        },
    }
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
