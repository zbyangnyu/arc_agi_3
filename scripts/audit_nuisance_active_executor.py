#!/usr/bin/env python3
"""Audit the learned canonical executor on the atomic nuisance protocol.

The primary nuisance experiment intentionally uses exact simulator
partitions.  This companion audit measures whether the existing learned
active-support executor can replace those partitions without changing the
candidate panel or posterior.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.run_oracle_canonical_acquisition_ceiling import (  # noqa: E402
    DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    PublicDoorQuery,
    _candidate_panel,
    _load_active_executor,
)
from scripts.run_oracle_canonical_nuisance_acquisition_ceiling import (  # noqa: E402
    PROGRAMS_PER_GROUP,
    _build_nuisance_tasks,
    _exact_candidate_outcome_maps,
    _select_global_information_candidate,
    _select_query_conditioned_candidate,
    _symbolic_initial_joint_log_weights,
)


AUDIT_SCHEMA_VERSION = "prp-wm.nuisance-active-executor-audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-executor-checkpoint",
        type=Path,
        default=DEFAULT_ACTIVE_EXECUTOR_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-query", type=int, default=64)
    parser.add_argument("--candidate-seed", type=int, default=20260873)
    parser.add_argument("--data-master-seed", type=int, default=2026071601)
    parser.add_argument(
        "--split",
        default="oracle-canonical-nuisance-acquisition-ceiling",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_counter() -> dict[str, Any]:
    return {
        "exact_grids": 0,
        "predicted_grids": 0,
        "partition_exact_panels": 0,
        "candidate_panels": 0,
        "exact_to_predicted_class_count": Counter(),
    }


def _finalise(counter: dict[str, Any]) -> dict[str, object]:
    grids = int(counter["predicted_grids"])
    panels = int(counter["candidate_panels"])
    return {
        "predicted_grids": grids,
        "exact_grid_rate": int(counter["exact_grids"]) / grids,
        "candidate_panels": panels,
        "exact_partition_rate": (
            int(counter["partition_exact_panels"]) / panels
        ),
        "exact_to_predicted_class_count": {
            key: value
            for key, value in sorted(
                counter["exact_to_predicted_class_count"].items()
            )
        },
    }


def _new_selector_counter() -> dict[str, Any]:
    return {
        "environments": 0,
        "index_matches": 0,
        "category_matches": 0,
        "learned_selected_categories": Counter(),
    }


def _finalise_selector(counter: dict[str, Any]) -> dict[str, object]:
    environments = int(counter["environments"])
    return {
        "environments": environments,
        "top1_index_match_rate": int(counter["index_matches"]) / environments,
        "top1_category_match_rate": (
            int(counter["category_matches"]) / environments
        ),
        "learned_selected_category_counts": {
            key: value
            for key, value in sorted(
                counter["learned_selected_categories"].items()
            )
        },
    }


def main() -> None:
    args = parse_args()
    if args.groups_per_query <= 0 or args.batch_size <= 0:
        raise SystemExit("groups and batch size must be positive")
    if args.candidate_seed < 0 or args.data_master_seed < 0:
        raise SystemExit("seeds must be non-negative")

    import torch

    from prp_wm.causal_filter import predict_factor_panel
    from prp_wm.latent_rules import outcome_map
    from prp_wm.rulegrid import ALL_AXES
    from scripts.run_active_support_calibrated_executor import (
        _canonicalize_grid_tensor,
    )
    from scripts.run_causal_mechanism_coverage import _resolve_device

    device = _resolve_device(torch, args.device)
    checkpoint_path = args.active_executor_checkpoint.resolve()
    executor, checkpoint, checkpoint_result = _load_active_executor(
        torch,
        checkpoint_path,
        device,
    )
    factor_bank = torch.tensor(
        tuple(product(range(4), repeat=3)),
        dtype=torch.long,
    )
    grid_size = executor.config.grid_size
    aggregate = _new_counter()
    by_axis = defaultdict(_new_counter)
    by_axis_and_category = defaultdict(_new_counter)
    selector_aggregate = {
        "query-conditioned": _new_selector_counter(),
        "global-information-gain": _new_selector_counter(),
    }
    selector_by_axis = defaultdict(
        lambda: {
            "query-conditioned": _new_selector_counter(),
            "global-information-gain": _new_selector_counter(),
        }
    )

    for query_index, query_axis in enumerate(ALL_AXES):
        query = PublicDoorQuery(query_index)
        all_tasks = _build_nuisance_tasks(
            query_axis,
            groups=args.groups_per_query,
            split=args.split,
            master_seed=args.data_master_seed,
            candidate_seed=args.candidate_seed,
        )
        # The three hidden query modes share the same public panel.  One task
        # per group is sufficient for an executor partition audit.
        tasks = all_tasks[::PROGRAMS_PER_GROUP]
        for start in range(0, len(tasks), args.batch_size):
            selected = tasks[start : start + args.batch_size]
            states, actions, action_mask, _ = _candidate_panel(
                torch=torch,
                tasks=selected,
                device=device,
            )
            codes = factor_bank.to(device)[None].expand(
                len(selected),
                -1,
                -1,
            )
            with torch.no_grad():
                learned = outcome_map(
                    predict_factor_panel(
                        executor,
                        states,
                        actions,
                        codes,
                        action_mask,
                    )
                ).reshape(
                    len(selected),
                    -1,
                    factor_bank.shape[0],
                    grid_size,
                    grid_size,
                ).cpu()
            exact, _ = _exact_candidate_outcome_maps(
                torch=torch,
                tasks=selected,
                factor_bank=factor_bank,
            )
            exact = _canonicalize_grid_tensor(torch, exact, selected)
            grid_exact = learned.eq(exact).flatten(start_dim=-2).all(dim=-1)
            initial = _symbolic_initial_joint_log_weights(
                torch=torch,
                tasks=selected,
                factor_bank=factor_bank,
            )
            query_values = factor_bank[:, query.axis_index]
            available = torch.ones(
                learned.shape[1],
                dtype=torch.bool,
            )

            for task_index, task in enumerate(selected):
                exact_query = _select_query_conditioned_candidate(
                    torch,
                    initial[task_index],
                    query_values,
                    exact[task_index],
                    available,
                )
                learned_query = _select_query_conditioned_candidate(
                    torch,
                    initial[task_index],
                    query_values,
                    learned[task_index],
                    available,
                )
                exact_global = _select_global_information_candidate(
                    torch,
                    initial[task_index],
                    exact[task_index],
                    available,
                )
                learned_global = _select_global_information_candidate(
                    torch,
                    initial[task_index],
                    learned[task_index],
                    available,
                )
                for policy, exact_score, learned_score in (
                    (
                        "query-conditioned",
                        exact_query,
                        learned_query,
                    ),
                    (
                        "global-information-gain",
                        exact_global,
                        learned_global,
                    ),
                ):
                    exact_index = int(exact_score.candidate_index)
                    learned_index = int(learned_score.candidate_index)
                    exact_category = str(
                        task.privileged.candidate_kinds[exact_index]
                    )
                    learned_category = str(
                        task.privileged.candidate_kinds[learned_index]
                    )
                    for counter in (
                        selector_aggregate[policy],
                        selector_by_axis[query_axis.value][policy],
                    ):
                        counter["environments"] += 1
                        counter["index_matches"] += int(
                            exact_index == learned_index
                        )
                        counter["category_matches"] += int(
                            exact_category == learned_category
                        )
                        counter["learned_selected_categories"][
                            learned_category
                        ] += 1

            for task_index, task in enumerate(selected):
                for candidate_index, category in enumerate(
                    task.privileged.candidate_kinds
                ):
                    exact_flat = exact[
                        task_index, candidate_index
                    ].reshape(factor_bank.shape[0], -1)
                    learned_flat = learned[
                        task_index, candidate_index
                    ].reshape(factor_bank.shape[0], -1)
                    exact_inverse = torch.unique(
                        exact_flat,
                        dim=0,
                        sorted=True,
                        return_inverse=True,
                    )[1]
                    learned_inverse = torch.unique(
                        learned_flat,
                        dim=0,
                        sorted=True,
                        return_inverse=True,
                    )[1]
                    partition_exact = bool(
                        (
                            (
                                exact_inverse[:, None]
                                == exact_inverse[None, :]
                            )
                            == (
                                learned_inverse[:, None]
                                == learned_inverse[None, :]
                            )
                        ).all()
                    )
                    class_pair = (
                        f"{int(exact_inverse.max()) + 1}"
                        f"->{int(learned_inverse.max()) + 1}"
                    )
                    exact_grids = int(
                        grid_exact[task_index, candidate_index].sum()
                    )
                    for counter in (
                        aggregate,
                        by_axis[query_axis.value],
                        by_axis_and_category[
                            (query_axis.value, str(category))
                        ],
                    ):
                        counter["exact_grids"] += exact_grids
                        counter["predicted_grids"] += factor_bank.shape[0]
                        counter["partition_exact_panels"] += int(
                            partition_exact
                        )
                        counter["candidate_panels"] += 1
                        counter["exact_to_predicted_class_count"][
                            class_pair
                        ] += 1

    source = Path(__file__).resolve()
    payload: dict[str, object] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "interpretation": (
            "Oracle-canonical learned-executor bridge audit; the primary "
            "nuisance result remains an exact symbolic ceiling."
        ),
        "groups_per_query": args.groups_per_query,
        "query_axes": len(ALL_AXES),
        "factor_codes": int(factor_bank.shape[0]),
        "candidate_seed": args.candidate_seed,
        "data_master_seed": args.data_master_seed,
        "split": args.split,
        "device": str(device),
        "active_executor_checkpoint": str(checkpoint_path),
        "active_executor_checkpoint_sha256": _sha256(checkpoint_path),
        "active_executor_checkpoint_schema": checkpoint.get(
            "checkpoint_schema_version"
        ),
        "active_executor_original_gate_passed": checkpoint_result.get(
            "active_prefix_executor_gate",
            {},
        ).get("passed"),
        "aggregate": _finalise(aggregate),
        "by_query_axis": {
            axis: _finalise(counter)
            for axis, counter in sorted(by_axis.items())
        },
        "by_query_axis_and_candidate_category": {
            f"{axis}/{category}": _finalise(counter)
            for (axis, category), counter in sorted(
                by_axis_and_category.items()
            )
        },
        "selector_bridge_audit": {
            "comparison": (
                "learned MAP partitions versus exact simulator partitions "
                "under the same exact 48-code posterior"
            ),
            "aggregate": {
                policy: _finalise_selector(counter)
                for policy, counter in selector_aggregate.items()
            },
            "by_query_axis": {
                axis: {
                    policy: _finalise_selector(counter)
                    for policy, counter in policies.items()
                }
                for axis, policies in sorted(selector_by_axis.items())
            },
        },
        "audit_source": str(source),
        "audit_source_sha256": _sha256(source),
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
