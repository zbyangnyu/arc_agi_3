#!/usr/bin/env python3
"""Run a pinned, deterministic, Docker-free Symbolic Alchemy smoke test.

This script deliberately exercises only the official pure-Python symbolic
environment.  It does not start the Unity environment or a Docker daemon.

Example:

    /path/to/python scripts/run_symbolic_alchemy_smoke.py \
      --source-root /path/to/dm_alchemy \
      --output runs/symbolic_alchemy_smoke_seed123/result.json
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import enum
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


OFFICIAL_REPOSITORY = "https://github.com/google-deepmind/dm_alchemy.git"
OFFICIAL_ARCHIVED_HEAD = "68a26254b5c0f15e84fa0c15d66bf0c626ede8e0"
LEVEL_NAME = (
    "alchemy/perceptual_mapping_randomized_"
    "with_rotation_and_random_bottleneck"
)
DEPENDENCY_DISTRIBUTIONS = (
    "absl-py",
    "dataclasses",
    "dm-alchemy",
    "dm-env",
    "dm-env-rpc",
    "dm-tree",
    "docker",
    "frozendict",
    "grpcio",
    "grpcio-tools",
    "numpy",
    "portpicker",
    "protobuf",
    "scipy",
    "setuptools",
    "wheel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--max-steps-per-trial", type=int, default=20)
    parser.add_argument(
        "--expected-source-commit",
        default=OFFICIAL_ARCHIVED_HEAD,
        help="Refuse to run if source-root is not at this exact commit.",
    )
    parser.add_argument(
        "--expected-sequence-sha256",
        default=None,
        help="Optionally require the deterministic no-op sequence hash.",
    )
    return parser.parse_args()


def _git(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(source_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite float cannot be serialized: {value}")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported JSON value {type(value).__name__}: {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _update_sequence_hash(digest: Any, value: Any) -> bytes:
    encoded = _canonical_bytes(value)
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)
    return encoded


def _spec_to_json(spec: Any) -> Any:
    if isinstance(spec, Mapping):
        return {
            str(key): _spec_to_json(value)
            for key, value in sorted(spec.items(), key=lambda pair: str(pair[0]))
        }
    result: dict[str, Any] = {
        "class": type(spec).__name__,
        "dtype": np.dtype(spec.dtype).name,
        "name": spec.name,
        "shape": list(spec.shape),
    }
    if hasattr(spec, "minimum"):
        result["minimum"] = _jsonable(spec.minimum)
    if hasattr(spec, "maximum"):
        result["maximum"] = _jsonable(spec.maximum)
    return result


def _timestep_to_json(timestep: Any, env: Any) -> dict[str, Any]:
    return {
        "discount": _jsonable(timestep.discount),
        "is_new_trial": bool(env.is_new_trial()),
        "observation": _jsonable(timestep.observation),
        "reward": _jsonable(timestep.reward),
        "step_type": timestep.step_type.name,
        "trial_number": int(env.trial_number),
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "MISSING"
    return versions


def _generated_proto_hashes(source_root: Path) -> dict[str, str]:
    generated = sorted(source_root.glob("dm_alchemy/**/*_pb2*.py"))
    return {
        str(path.relative_to(source_root)): _sha256_file(path)
        for path in generated
    }


def _make_env(
    symbolic_alchemy: Any,
    *,
    seed: int,
    num_trials: int,
    max_steps_per_trial: int,
) -> Any:
    return symbolic_alchemy.get_symbolic_alchemy_level(
        LEVEL_NAME,
        seed=seed,
        num_trials=num_trials,
        num_stones_per_trial=3,
        num_potions_per_trial=12,
        max_steps_per_trial=max_steps_per_trial,
        observe_used=True,
        end_trial_action=False,
    )


def _run_dual_noop_rollout(
    symbolic_alchemy: Any,
    *,
    seed: int,
    num_trials: int,
    max_steps_per_trial: int,
) -> tuple[dict[str, Any], Any, Any]:
    build_started = time.perf_counter()
    env_a = _make_env(
        symbolic_alchemy,
        seed=seed,
        num_trials=num_trials,
        max_steps_per_trial=max_steps_per_trial,
    )
    env_b = _make_env(
        symbolic_alchemy,
        seed=seed,
        num_trials=num_trials,
        max_steps_per_trial=max_steps_per_trial,
    )
    build_seconds = time.perf_counter() - build_started

    rollout_started = time.perf_counter()
    timestep_a = env_a.reset()
    timestep_b = env_b.reset()
    digest_a = hashlib.sha256()
    digest_b = hashlib.sha256()
    initial_a = {"action": None, "timestep": _timestep_to_json(timestep_a, env_a)}
    initial_b = {"action": None, "timestep": _timestep_to_json(timestep_b, env_b)}
    encoded_a = _update_sequence_hash(digest_a, initial_a)
    encoded_b = _update_sequence_hash(digest_b, initial_b)
    if encoded_a != encoded_b:
        raise AssertionError("same-seed environments differ at reset")

    action = 0
    total_reward_a = 0.0
    total_reward_b = 0.0
    steps = 0
    trial_boundaries = 0
    expected_steps = num_trials * max_steps_per_trial
    while not timestep_a.last():
        if timestep_b.last():
            raise AssertionError(f"environment B ended early at step {steps}")
        timestep_a = env_a.step(action)
        timestep_b = env_b.step(action)
        steps += 1
        total_reward_a += float(timestep_a.reward)
        total_reward_b += float(timestep_b.reward)
        trial_boundaries += int(env_a.is_new_trial() or timestep_a.last())
        entry_a = {
            "action": action,
            "timestep": _timestep_to_json(timestep_a, env_a),
        }
        entry_b = {
            "action": action,
            "timestep": _timestep_to_json(timestep_b, env_b),
        }
        encoded_a = _update_sequence_hash(digest_a, entry_a)
        encoded_b = _update_sequence_hash(digest_b, entry_b)
        if encoded_a != encoded_b:
            raise AssertionError(
                f"same-seed environments first differ after action step {steps}"
            )
        if steps > expected_steps:
            raise AssertionError(
                f"no-op rollout exceeded expected {expected_steps} steps"
            )

    if not timestep_b.last():
        raise AssertionError("environment B did not end with environment A")
    rollout_seconds = time.perf_counter() - rollout_started
    sequence_sha256_a = digest_a.hexdigest()
    sequence_sha256_b = digest_b.hexdigest()
    return (
        {
            "action": action,
            "entries_hashed_including_reset": steps + 1,
            "expected_steps": expected_steps,
            "final_step_type": timestep_a.step_type.name,
            "sequence_sha256": sequence_sha256_a,
            "sequence_sha256_peer": sequence_sha256_b,
            "step_count": steps,
            "total_reward": total_reward_a,
            "total_reward_peer": total_reward_b,
            "trial_boundaries": trial_boundaries,
            "timing_seconds": {
                "dual_environment_build": build_seconds,
                "dual_rollout": rollout_seconds,
                "dual_rollout_per_environment_step": (
                    rollout_seconds / (2 * steps) if steps else 0.0
                ),
            },
        },
        env_a,
        env_b,
    )


def _run_non_noop_probe(
    symbolic_alchemy: Any,
    *,
    seed: int,
    num_trials: int,
    max_steps_per_trial: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    env_a = _make_env(
        symbolic_alchemy,
        seed=seed,
        num_trials=num_trials,
        max_steps_per_trial=max_steps_per_trial,
    )
    env_b = _make_env(
        symbolic_alchemy,
        seed=seed,
        num_trials=num_trials,
        max_steps_per_trial=max_steps_per_trial,
    )
    before_a = env_a.reset()
    before_b = env_b.reset()
    non_noop_action = 2
    action_tuple = symbolic_alchemy.int_action_to_tuple(
        non_noop_action,
        slot_based=True,
        end_trial_action=False,
    )
    if tuple(action_tuple) != (0, 0):
        raise AssertionError(
            f"expected action 2 to mean stone slot 0 + potion slot 0, got {action_tuple}"
        )
    after_a = env_a.step(non_noop_action)
    after_b = env_b.step(non_noop_action)

    before_vector = np.asarray(before_a.observation["symbolic_obs"])
    after_vector = np.asarray(after_a.observation["symbolic_obs"])
    changed_indices = np.flatnonzero(before_vector != after_vector).tolist()
    if not changed_indices:
        raise AssertionError("valid non-noop potion action did not change observation")

    transition_a = {
        "action": non_noop_action,
        "action_tuple": action_tuple,
        "after": _timestep_to_json(after_a, env_a),
        "before": _timestep_to_json(before_a, env_a),
    }
    transition_b = {
        "action": non_noop_action,
        "action_tuple": action_tuple,
        "after": _timestep_to_json(after_b, env_b),
        "before": _timestep_to_json(before_b, env_b),
    }
    transition_bytes_a = _canonical_bytes(transition_a)
    transition_bytes_b = _canonical_bytes(transition_b)
    if transition_bytes_a != transition_bytes_b:
        raise AssertionError("same-seed non-noop transitions are not deterministic")

    return {
        "action": non_noop_action,
        "action_semantics": {
            "potion_slot": int(action_tuple[1]),
            "stone_slot": int(action_tuple[0]),
        },
        "after_observation_sha256": hashlib.sha256(
            _canonical_bytes(after_a.observation)
        ).hexdigest(),
        "before_observation_sha256": hashlib.sha256(
            _canonical_bytes(before_a.observation)
        ).hexdigest(),
        "changed_feature_count": len(changed_indices),
        "changed_feature_indices": changed_indices,
        "deterministic_peer_match": True,
        "reward": float(after_a.reward),
        "step_type": after_a.step_type.name,
        "timing_seconds": time.perf_counter() - started,
        "transition_sha256": hashlib.sha256(transition_bytes_a).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.num_trials <= 0 or args.max_steps_per_trial <= 0:
        raise SystemExit("--num-trials and --max-steps-per-trial must be positive")

    total_started = time.perf_counter()
    source_root = args.source_root.resolve(strict=True)
    source_commit = _git(source_root, "rev-parse", "HEAD")
    if source_commit != args.expected_source_commit:
        raise SystemExit(
            "source commit mismatch: "
            f"expected {args.expected_source_commit}, found {source_commit}"
        )
    tracked_status = _git(
        source_root,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if tracked_status:
        raise SystemExit(
            f"refusing modified tracked dm_alchemy source:\n{tracked_status}"
        )

    from dm_alchemy import symbolic_alchemy
    import dm_alchemy

    installed_package = Path(dm_alchemy.__file__).resolve()
    try:
        installed_package.relative_to(source_root)
    except ValueError as error:
        raise SystemExit(
            "installed dm_alchemy does not resolve inside --source-root: "
            f"{installed_package}"
        ) from error

    noop, env_a, _ = _run_dual_noop_rollout(
        symbolic_alchemy,
        seed=args.seed,
        num_trials=args.num_trials,
        max_steps_per_trial=args.max_steps_per_trial,
    )
    expected_steps = args.num_trials * args.max_steps_per_trial
    assertions = {
        "action_range_is_0_through_39": (
            int(env_a.action_spec().minimum) == 0
            and int(env_a.action_spec().maximum) == 39
        ),
        "final_timestep_is_last": noop["final_step_type"] == "LAST",
        "noop_reward_is_zero": noop["total_reward"] == 0.0,
        "noop_steps_match_protocol": noop["step_count"] == expected_steps,
        "observation_shape_is_39": (
            tuple(env_a.observation_spec()["symbolic_obs"].shape) == (39,)
        ),
        "same_seed_full_sequence_matches": (
            noop["sequence_sha256"] == noop["sequence_sha256_peer"]
        ),
        "same_seed_total_reward_matches": (
            noop["total_reward"] == noop["total_reward_peer"]
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"failed no-op smoke assertions: {failed}")
    if (
        args.expected_sequence_sha256 is not None
        and noop["sequence_sha256"] != args.expected_sequence_sha256
    ):
        raise AssertionError(
            "sequence hash mismatch: "
            f"expected {args.expected_sequence_sha256}, "
            f"found {noop['sequence_sha256']}"
        )

    non_noop = _run_non_noop_probe(
        symbolic_alchemy,
        seed=args.seed,
        num_trials=args.num_trials,
        max_steps_per_trial=args.max_steps_per_trial,
    )
    assertions["non_noop_changed_observation"] = (
        non_noop["changed_feature_count"] > 0
    )
    assertions["non_noop_same_seed_transition_matches"] = non_noop[
        "deterministic_peer_match"
    ]

    result = {
        "assertions": assertions,
        "dependencies": _dependency_versions(),
        "dependency_constraints": {
            "description": (
                "Minimal compatibility envelope for the unmodified archived "
                "setup.py; exact resolved versions above are the run record."
            ),
            "numpy": ">=1.26,<2",
            "python": ">=3.11,<3.12",
            "setuptools": ">=65,<70",
        },
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "experiment": "official_symbolic_alchemy_docker_free_smoke",
        "non_noop_probe": non_noop,
        "noop_episode": noop,
        "protocol": {
            "end_trial_action": False,
            "level_name": LEVEL_NAME,
            "max_steps_per_trial": args.max_steps_per_trial,
            "num_potions_per_trial": 12,
            "num_stones_per_trial": 3,
            "num_trials": args.num_trials,
            "observe_used": True,
            "seed": args.seed,
        },
        "source": {
            "commit": source_commit,
            "expected_commit": args.expected_source_commit,
            "generated_proto_sha256": _generated_proto_hashes(source_root),
            "installed_package_path": str(installed_package),
            "official_repository": OFFICIAL_REPOSITORY,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "source_root": str(source_root),
            "tracked_worktree_clean": True,
        },
        "specs": {
            "action": _spec_to_json(env_a.action_spec()),
            "observation": _spec_to_json(env_a.observation_spec()),
        },
        "status": "PASS",
        "timing_seconds": {
            "total": time.perf_counter() - total_started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
