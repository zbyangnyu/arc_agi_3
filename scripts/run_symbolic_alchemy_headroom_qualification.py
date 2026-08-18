#!/usr/bin/env python3
"""Qualify Symbolic Alchemy benchmark headroom on the official 1,000 episodes.

The expensive ideal-observer and search-oracle planning runs are deliberately
not repeated.  Their bundled official action-event traces are replayed against
the matching bundled chemistries and items with dm_alchemy's official replay
API.  The official baseline trace is treated as a published learned-agent
baseline, not as the random-action policy.

Example:

    /path/to/python scripts/run_symbolic_alchemy_headroom_qualification.py \
      --source-root /path/to/dm_alchemy \
      --output runs/symbolic_alchemy_headroom/result.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


OFFICIAL_REPOSITORY = "https://github.com/google-deepmind/dm_alchemy.git"
OFFICIAL_ARCHIVED_HEAD = "68a26254b5c0f15e84fa0c15d66bf0c626ede8e0"
EXPECTED_EPISODES = 1_000
EXPECTED_TRIALS_PER_EPISODE = 10
EXPECTED_STONES_PER_TRIAL = 3
EXPECTED_POTIONS_PER_TRIAL = 12
DEFAULT_RANDOM_SEED = 20_260_724
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
REPLAY_CROSSCHECK_EPISODES = (0, 499, 999)

CHEMISTRY_RESOURCE = (
    "chemistries/perceptual_mapping_randomized_with_random_bottleneck/"
    "chemistries"
)
OFFICIAL_EVENT_RESOURCES = {
    "baseline": "agent_events/baseline",
    "ideal_observer": "agent_events/ideal_observer",
    "search_oracle": "agent_events/search_oracle",
}
EXPECTED_RESOURCE_SHA256 = {
    CHEMISTRY_RESOURCE:
        "7a9e4fbfb2328810e557a7f00bac757664b0e8ae0d9cb7e0a44d2c42547500c9",
    "agent_events/baseline":
        "6423c5871fac5f216fb948802b86500ff1df2a468ca37c65289ded368a06ffcb",
    "agent_events/ideal_observer":
        "1d0c07208c0798a78bd98950e0f87fc1478670257ee672b6b58bb88aa022ec0c",
    "agent_events/search_oracle":
        "30d15dceaf835a6e9427dbc1df13a48b983315f35dd5095382a5f550161423f8",
}
OFFICIAL_NOTEBOOK = "examples/AlchemyGettingStarted.ipynb"
EXPECTED_NOTEBOOK_SHA256 = (
    "f14a8269330966db79d03607d8afc657c63ce057f2ebbc746600720806a7120b"
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
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--expected-source-commit",
        default=OFFICIAL_ARCHIVED_HEAD,
        help="Refuse to run if source-root is not at this exact commit.",
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "MISSING"
    return versions


def _stable_uint64(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(material).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    exact_histogram = Counter(int(value) for value in values)
    return {
        "count": int(array.size),
        "sum": float(np.sum(array)),
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "q05": _quantile(array, 0.05),
        "q25": _quantile(array, 0.25),
        "median": _quantile(array, 0.50),
        "q75": _quantile(array, 0.75),
        "q95": _quantile(array, 0.95),
        "max": float(np.max(array)),
        "exact_histogram": {
            str(key): int(exact_histogram[key])
            for key in sorted(exact_histogram)
        },
    }


def _flatten(nested: Sequence[Sequence[int]]) -> list[int]:
    return [int(value) for row in nested for value in row]


def _event_lists(
    nested_trackers: Sequence[Sequence[Any]],
) -> tuple[list[list[list[list[int]]]], list[list[int]]]:
    """Return timestamped events and effective event counts by trial."""
    timestamped: list[list[list[list[int]]]] = []
    counts: list[list[int]] = []
    for episode_trackers in nested_trackers:
        episode_events: list[list[list[int]]] = []
        episode_counts: list[int] = []
        for tracker in episode_trackers:
            trial_events = [
                [int(stone), int(potion), int(frame)]
                for stone, potion, frame in tracker.events_list()
            ]
            episode_events.append(trial_events)
            episode_counts.append(len(trial_events))
        timestamped.append(episode_events)
        counts.append(episode_counts)
    return timestamped, counts


def _actions_without_timestamps(
    timestamped_events: Sequence[Sequence[Sequence[Sequence[int]]]],
) -> list[list[list[list[int]]]]:
    return [
        [
            [[int(event[0]), int(event[1])] for event in trial]
            for trial in episode
        ]
        for episode in timestamped_events
    ]


def _validate_fixed_episode_set(chemistries_and_items: Sequence[Any]) -> None:
    if len(chemistries_and_items) != EXPECTED_EPISODES:
        raise AssertionError(
            f"expected {EXPECTED_EPISODES} evaluation episodes, got "
            f"{len(chemistries_and_items)}"
        )
    for episode_index, (_, episode_items) in enumerate(chemistries_and_items):
        if episode_items.num_trials != EXPECTED_TRIALS_PER_EPISODE:
            raise AssertionError(
                f"episode {episode_index}: expected "
                f"{EXPECTED_TRIALS_PER_EPISODE} trials, got "
                f"{episode_items.num_trials}"
            )
        for trial_index, trial_items in enumerate(episode_items.trials):
            if trial_items.num_stones != EXPECTED_STONES_PER_TRIAL:
                raise AssertionError(
                    f"episode {episode_index} trial {trial_index}: expected "
                    f"{EXPECTED_STONES_PER_TRIAL} stones, got "
                    f"{trial_items.num_stones}"
                )
            if trial_items.num_potions != EXPECTED_POTIONS_PER_TRIAL:
                raise AssertionError(
                    f"episode {episode_index} trial {trial_index}: expected "
                    f"{EXPECTED_POTIONS_PER_TRIAL} potions, got "
                    f"{trial_items.num_potions}"
                )


def _validate_official_events(name: str, events: Sequence[Sequence[Any]]) -> None:
    if len(events) != EXPECTED_EPISODES:
        raise AssertionError(
            f"{name}: expected {EXPECTED_EPISODES} episodes, got {len(events)}"
        )
    expected_shape = (
        EXPECTED_STONES_PER_TRIAL,
        EXPECTED_POTIONS_PER_TRIAL + 1,
    )
    for episode_index, episode_events in enumerate(events):
        if len(episode_events) != EXPECTED_TRIALS_PER_EPISODE:
            raise AssertionError(
                f"{name} episode {episode_index}: expected "
                f"{EXPECTED_TRIALS_PER_EPISODE} trials, got "
                f"{len(episode_events)}"
            )
        for trial_index, tracker in enumerate(episode_events):
            if tuple(tracker.events.shape) != expected_shape:
                raise AssertionError(
                    f"{name} episode {episode_index} trial {trial_index}: "
                    f"expected event matrix {expected_shape}, got "
                    f"{tuple(tracker.events.shape)}"
                )


def _score_replayed_events(
    *,
    name: str,
    chemistries_and_items: Sequence[Any],
    nested_trackers: Sequence[Sequence[Any]],
    reward_weights: Any,
    event_tracker_module: Any,
) -> tuple[list[list[int]], float]:
    """Score official events with the official low-level replay API."""
    started = time.perf_counter()
    trial_returns_by_episode: list[list[int]] = []
    for episode_index, ((chemistry, episode_items), episode_events) in enumerate(
        zip(chemistries_and_items, nested_trackers)
    ):
        trial_returns: list[int] = []
        for trial_index, (trial_items, trial_events) in enumerate(
            zip(episode_items.trials, episode_events)
        ):
            reward_tracker = event_tracker_module.RewardTracker(reward_weights)
            game_state = event_tracker_module.GameState(
                graph=chemistry.graph,
                trial_items=trial_items,
                event_trackers=[reward_tracker],
            )
            try:
                event_tracker_module.replay_events(game_state, trial_events)
            except Exception as error:
                raise RuntimeError(
                    f"{name} replay failed at episode {episode_index}, "
                    f"trial {trial_index}"
                ) from error
            trial_returns.append(int(reward_tracker.reward))
        trial_returns_by_episode.append(trial_returns)
    return trial_returns_by_episode, time.perf_counter() - started


def _full_replay_crosscheck(
    *,
    name: str,
    chemistries_and_items: Sequence[Any],
    nested_trackers: Sequence[Sequence[Any]],
    direct_trial_returns: Sequence[Sequence[int]],
    reward_weights: Any,
    symbolic_alchemy: Any,
    symbolic_alchemy_bots: Any,
    symbolic_alchemy_trackers: Any,
) -> dict[str, Any]:
    """Cross-check selected episodes through ReplayBot + ScoreTracker."""
    checks: list[dict[str, Any]] = []
    for episode_index in REPLAY_CROSSCHECK_EPISODES:
        chemistry, episode_items = chemistries_and_items[episode_index]
        source_episode_events = nested_trackers[episode_index]
        env = symbolic_alchemy.get_symbolic_alchemy_fixed(
            chemistry=chemistry,
            episode_items=episode_items,
            reward_weights=reward_weights,
        )
        score_tracker = symbolic_alchemy_trackers.ScoreTracker(reward_weights)
        matrix_tracker = symbolic_alchemy_trackers.AddMatrixEventTracker()
        env.add_trackers({
            score_tracker.name: score_tracker,
            matrix_tracker.name: matrix_tracker,
        })
        results = symbolic_alchemy_bots.ReplayBot(
            source_episode_events,
            env,
        ).run_episode()
        replay_returns = [
            int(value) for value in results["score"]["per_trial"]
        ]
        replay_trackers = results["matrix_event"]["event_tracker"]
        source_actions = [
            [(int(stone), int(potion)) for stone, potion, _ in tracker.events_list()]
            for tracker in source_episode_events
        ]
        replay_actions = [
            [(int(stone), int(potion)) for stone, potion, _ in tracker.events_list()]
            for tracker in replay_trackers
        ]
        expected_returns = [
            int(value) for value in direct_trial_returns[episode_index]
        ]
        if replay_returns != expected_returns:
            raise AssertionError(
                f"{name} episode {episode_index}: ReplayBot scores do not "
                "match direct official replay"
            )
        if replay_actions != source_actions:
            raise AssertionError(
                f"{name} episode {episode_index}: ReplayBot action sequence "
                "does not match the bundled action sequence"
            )
        checks.append({
            "episode_index": episode_index,
            "action_sequence_match": True,
            "trial_return_match": True,
            "trial_returns": replay_returns,
            "effective_event_count": sum(len(trial) for trial in replay_actions),
        })
    return {
        "api_path": (
            "dm_alchemy.symbolic_alchemy_bots.ReplayBot + "
            "dm_alchemy.symbolic_alchemy_trackers.ScoreTracker"
        ),
        "episodes": checks,
    }


def _run_generated_bot(
    *,
    name: str,
    chemistries_and_items: Sequence[Any],
    reward_weights: Any,
    symbolic_alchemy: Any,
    symbolic_alchemy_bots: Any,
    symbolic_alchemy_trackers: Any,
    random_seed: int | None,
) -> tuple[list[list[int]], list[list[Any]], float]:
    started = time.perf_counter()
    returns_by_episode: list[list[int]] = []
    events_by_episode: list[list[Any]] = []
    for episode_index, (chemistry, episode_items) in enumerate(
        chemistries_and_items
    ):
        if random_seed is not None:
            episode_seed = _stable_uint64(
                "symbolic-alchemy-random-action-v1",
                random_seed,
                episode_index,
            )
            random.seed(episode_seed)
        env = symbolic_alchemy.get_symbolic_alchemy_fixed(
            chemistry=chemistry,
            episode_items=episode_items,
            reward_weights=reward_weights,
        )
        score_tracker = symbolic_alchemy_trackers.ScoreTracker(reward_weights)
        matrix_tracker = symbolic_alchemy_trackers.AddMatrixEventTracker()
        env.add_trackers({
            score_tracker.name: score_tracker,
            matrix_tracker.name: matrix_tracker,
        })
        if name == "random_action":
            bot = symbolic_alchemy_bots.RandomActionBot(reward_weights, env)
        elif name == "noop":
            bot = symbolic_alchemy_bots.NoOpBot(env)
        else:
            raise ValueError(f"unsupported generated bot: {name}")
        results = bot.run_episode()
        trial_returns = [
            int(value) for value in results["score"]["per_trial"]
        ]
        trial_events = list(results["matrix_event"]["event_tracker"])
        if len(trial_returns) != EXPECTED_TRIALS_PER_EPISODE:
            raise AssertionError(
                f"{name} episode {episode_index}: incomplete score trace"
            )
        if len(trial_events) != EXPECTED_TRIALS_PER_EPISODE:
            raise AssertionError(
                f"{name} episode {episode_index}: incomplete event trace"
            )
        returns_by_episode.append(trial_returns)
        events_by_episode.append(trial_events)
        if (episode_index + 1) % 100 == 0:
            print(
                f"{name}: completed {episode_index + 1}/{EXPECTED_EPISODES}",
                file=sys.stderr,
                flush=True,
            )
    return (
        returns_by_episode,
        events_by_episode,
        time.perf_counter() - started,
    )


def _policy_result(
    *,
    source: str,
    trial_returns_by_episode: Sequence[Sequence[int]],
    nested_trackers: Sequence[Sequence[Any]],
    replay_seconds: float,
    notes: Sequence[str],
    full_replay_crosscheck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(trial_returns_by_episode) != EXPECTED_EPISODES:
        raise AssertionError("policy result has wrong episode count")
    timestamped_events, event_counts_by_trial = _event_lists(nested_trackers)
    action_events = _actions_without_timestamps(timestamped_events)
    episode_returns = [
        int(sum(trial_returns))
        for trial_returns in trial_returns_by_episode
    ]
    episode_event_counts = [
        int(sum(trial_counts))
        for trial_counts in event_counts_by_trial
    ]
    flat_trial_returns = _flatten(trial_returns_by_episode)
    flat_trial_event_counts = _flatten(event_counts_by_trial)
    mean_return_by_trial_index = [
        float(np.mean([
            episode_returns_for_trials[trial_index]
            for episode_returns_for_trials in trial_returns_by_episode
        ]))
        for trial_index in range(EXPECTED_TRIALS_PER_EPISODE)
    ]
    total_events = int(sum(episode_event_counts))
    total_return = int(sum(episode_returns))
    result: dict[str, Any] = {
        "source": source,
        "notes": list(notes),
        "episodes": EXPECTED_EPISODES,
        "trials": EXPECTED_EPISODES * EXPECTED_TRIALS_PER_EPISODE,
        "episode_returns": episode_returns,
        "trial_returns_by_episode": [
            [int(value) for value in trial_returns]
            for trial_returns in trial_returns_by_episode
        ],
        "episode_return_distribution": _distribution(episode_returns),
        "trial_return_distribution": _distribution(flat_trial_returns),
        "mean_return_by_trial_index": mean_return_by_trial_index,
        "effective_events": {
            "definition": (
                "stone-potion and stone-cauldron events recorded by "
                "MatrixEventTracker; no-op/end-trial padding is excluded"
            ),
            "total": total_events,
            "episode_counts": episode_event_counts,
            "trial_counts_by_episode": event_counts_by_trial,
            "episode_count_distribution": _distribution(episode_event_counts),
            "trial_count_distribution": _distribution(flat_trial_event_counts),
            "return_per_effective_event": (
                None if total_events == 0 else float(total_return / total_events)
            ),
        },
        "hashes": {
            "episode_returns_sha256": _canonical_sha256(episode_returns),
            "trial_returns_sha256": _canonical_sha256(
                trial_returns_by_episode
            ),
            "action_events_sha256": _canonical_sha256(action_events),
            "timestamped_events_sha256": _canonical_sha256(
                timestamped_events
            ),
        },
        "runtime_seconds": float(replay_seconds),
    }
    if full_replay_crosscheck is not None:
        result["full_replay_crosscheck"] = full_replay_crosscheck
    return result


def _bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[list[float], str]:
    if resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    block_size = 100
    for start in range(0, resamples, block_size):
        stop = min(start + block_size, resamples)
        indices = rng.integers(
            0,
            differences.size,
            size=(stop - start, differences.size),
        )
        bootstrap_means[start:stop] = np.mean(
            differences[indices],
            axis=1,
        )
    ci = [
        _quantile(bootstrap_means, 0.025),
        _quantile(bootstrap_means, 0.975),
    ]
    return ci, hashlib.sha256(
        bootstrap_means.astype("<f8", copy=False).tobytes()
    ).hexdigest()


def _paired_comparison(
    *,
    better_name: str,
    reference_name: str,
    policies: dict[str, dict[str, Any]],
    bootstrap_resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    better_episode = np.asarray(
        policies[better_name]["episode_returns"],
        dtype=np.int64,
    )
    reference_episode = np.asarray(
        policies[reference_name]["episode_returns"],
        dtype=np.int64,
    )
    better_trial = np.asarray(
        policies[better_name]["trial_returns_by_episode"],
        dtype=np.int64,
    )
    reference_trial = np.asarray(
        policies[reference_name]["trial_returns_by_episode"],
        dtype=np.int64,
    )
    episode_difference = better_episode - reference_episode
    trial_difference = better_trial - reference_trial
    bootstrap_seed = _stable_uint64(
        "symbolic-alchemy-paired-bootstrap-v1",
        random_seed,
        better_name,
        reference_name,
    )
    ci, bootstrap_hash = _bootstrap_mean_ci(
        episode_difference.astype(np.float64),
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "definition": f"{better_name} minus {reference_name}",
        "pairing": (
            "same official evaluation episode, chemistry, trial items, and "
            "episode index"
        ),
        "episode_difference_distribution": _distribution(
            episode_difference.tolist()
        ),
        "trial_difference_distribution": _distribution(
            trial_difference.reshape(-1).tolist()
        ),
        "episodes_positive": int(np.sum(episode_difference > 0)),
        "episodes_tied": int(np.sum(episode_difference == 0)),
        "episodes_negative": int(np.sum(episode_difference < 0)),
        "fraction_episodes_positive": float(np.mean(episode_difference > 0)),
        "paired_bootstrap_mean_episode_difference_95_ci": ci,
        "bootstrap": {
            "method": (
                "percentile paired bootstrap over the 1,000 episode-index "
                "differences"
            ),
            "resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed),
            "bootstrap_means_little_endian_f64_sha256": bootstrap_hash,
        },
        "episode_difference_sha256": _canonical_sha256(
            episode_difference.tolist()
        ),
    }


def _source_provenance(
    source_root: Path,
    expected_commit: str,
    dm_alchemy_module: Any,
) -> dict[str, Any]:
    actual_commit = _git(source_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"source commit mismatch: expected {expected_commit}, "
            f"got {actual_commit}"
        )
    tracked_status = _git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(
            "refusing modified tracked dm_alchemy source:\n" + tracked_status
        )
    imported_package = Path(dm_alchemy_module.__file__).resolve().parent
    expected_package = (source_root / "dm_alchemy").resolve()
    if imported_package != expected_package:
        raise RuntimeError(
            f"dm_alchemy import came from {imported_package}, expected "
            f"{expected_package}"
        )
    untracked = _git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    generated_proto_hashes = {
        str(path.relative_to(source_root)): _sha256_file(path)
        for path in sorted(source_root.glob("dm_alchemy/**/*_pb2*.py"))
    }
    try:
        remote = _git(source_root, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        remote = "MISSING"
    return {
        "repository": OFFICIAL_REPOSITORY,
        "configured_origin": remote,
        "commit": actual_commit,
        "expected_commit": expected_commit,
        "tracked_worktree_clean": True,
        "imported_package": str(imported_package),
        "untracked_entry_count": len(untracked),
        "untracked_entries": untracked,
        "generated_proto_hashes": generated_proto_hashes,
    }


def _verify_resources(source_root: Path) -> dict[str, Any]:
    package_root = source_root / "dm_alchemy"
    resources: dict[str, Any] = {}
    for relative_path, expected_sha256 in EXPECTED_RESOURCE_SHA256.items():
        path = package_root / relative_path
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"resource hash mismatch for {path}: expected "
                f"{expected_sha256}, got {actual_sha256}"
            )
        resources[relative_path] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
            "match": True,
        }
    notebook_path = source_root / OFFICIAL_NOTEBOOK
    notebook_sha256 = _sha256_file(notebook_path)
    if notebook_sha256 != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError(
            f"official notebook hash mismatch: expected "
            f"{EXPECTED_NOTEBOOK_SHA256}, got {notebook_sha256}"
        )
    return {
        "bundled_resources": resources,
        "official_protocol_notebook": {
            "path": str(notebook_path.resolve()),
            "bytes": notebook_path.stat().st_size,
            "sha256": notebook_sha256,
            "expected_sha256": EXPECTED_NOTEBOOK_SHA256,
            "match": True,
        },
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    source_root = args.source_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")

    # Imports happen only after the overwrite guard so a rerun fails quickly.
    import dm_alchemy
    from dm_alchemy import event_tracker
    from dm_alchemy import io
    from dm_alchemy import symbolic_alchemy
    from dm_alchemy import symbolic_alchemy_bots
    from dm_alchemy import symbolic_alchemy_trackers
    from dm_alchemy.encode import chemistries_proto_conversion
    from dm_alchemy.encode import symbolic_actions_pb2
    from dm_alchemy.encode import symbolic_actions_proto_conversion
    from dm_alchemy.types import stones_and_potions

    started = time.perf_counter()
    source = _source_provenance(
        source_root,
        args.expected_source_commit,
        dm_alchemy,
    )
    resource_provenance = _verify_resources(source_root)

    reward_weights = stones_and_potions.RewardWeights(
        coefficients=[1, 1, 1],
        offset=0,
        bonus=12,
    )
    chemistries_and_items = (
        chemistries_proto_conversion.load_chemistries_and_items(
            CHEMISTRY_RESOURCE
        )
    )
    _validate_fixed_episode_set(chemistries_and_items)

    official_events: dict[str, list[list[Any]]] = {}
    for name, relative_path in OFFICIAL_EVENT_RESOURCES.items():
        serialized = io.read_proto(relative_path)
        proto = symbolic_actions_pb2.EvaluationSetEvents.FromString(serialized)
        events = (
            symbolic_actions_proto_conversion.proto_to_evaluation_set_events(
                proto
            )
        )
        _validate_official_events(name, events)
        official_events[name] = events

    policies: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "ideal_observer", "search_oracle"):
        trial_returns, replay_seconds = _score_replayed_events(
            name=name,
            chemistries_and_items=chemistries_and_items,
            nested_trackers=official_events[name],
            reward_weights=reward_weights,
            event_tracker_module=event_tracker,
        )
        crosscheck = _full_replay_crosscheck(
            name=name,
            chemistries_and_items=chemistries_and_items,
            nested_trackers=official_events[name],
            direct_trial_returns=trial_returns,
            reward_weights=reward_weights,
            symbolic_alchemy=symbolic_alchemy,
            symbolic_alchemy_bots=symbolic_alchemy_bots,
            symbolic_alchemy_trackers=symbolic_alchemy_trackers,
        )
        notes = [
            "Bundled official action-event trace; no planner was rerun.",
            "Scored by dm_alchemy.event_tracker.replay_events on the matching "
            "bundled chemistry and trial items.",
        ]
        if name == "baseline":
            notes.append(
                "This is the released learned-agent baseline trace, not the "
                "random-action policy."
            )
        policies[name] = _policy_result(
            source=f"bundled_official_trace:{OFFICIAL_EVENT_RESOURCES[name]}",
            trial_returns_by_episode=trial_returns,
            nested_trackers=official_events[name],
            replay_seconds=replay_seconds,
            notes=notes,
            full_replay_crosscheck=crosscheck,
        )
        print(f"{name}: official replay complete", file=sys.stderr, flush=True)

    random_returns, random_events, random_seconds = _run_generated_bot(
        name="random_action",
        chemistries_and_items=chemistries_and_items,
        reward_weights=reward_weights,
        symbolic_alchemy=symbolic_alchemy,
        symbolic_alchemy_bots=symbolic_alchemy_bots,
        symbolic_alchemy_trackers=symbolic_alchemy_trackers,
        random_seed=args.random_seed,
    )
    policies["random_action"] = _policy_result(
        source="official_bot_generated:RandomActionBot",
        trial_returns_by_episode=random_returns,
        nested_trackers=random_events,
        replay_seconds=random_seconds,
        notes=[
            "Generated on every official fixed evaluation episode with "
            "dm_alchemy.symbolic_alchemy_bots.RandomActionBot.",
            "Python's random module is reseeded independently for each episode "
            "using the recorded SHA-256-derived uint64 seed.",
        ],
    )

    noop_returns, noop_events, noop_seconds = _run_generated_bot(
        name="noop",
        chemistries_and_items=chemistries_and_items,
        reward_weights=reward_weights,
        symbolic_alchemy=symbolic_alchemy,
        symbolic_alchemy_bots=symbolic_alchemy_bots,
        symbolic_alchemy_trackers=symbolic_alchemy_trackers,
        random_seed=None,
    )
    policies["noop"] = _policy_result(
        source="official_bot_generated:NoOpBot",
        trial_returns_by_episode=noop_returns,
        nested_trackers=noop_events,
        replay_seconds=noop_seconds,
        notes=[
            "Generated on every official fixed evaluation episode with "
            "dm_alchemy.symbolic_alchemy_bots.NoOpBot.",
            "No-op/end-trial padding is intentionally absent from "
            "MatrixEventTracker effective-event counts.",
        ],
    )

    comparison_specs = (
        ("ideal_observer", "random_action"),
        ("search_oracle", "random_action"),
        ("ideal_observer", "baseline"),
        ("search_oracle", "baseline"),
        ("ideal_observer", "noop"),
        ("search_oracle", "noop"),
        ("random_action", "noop"),
    )
    paired_comparisons: dict[str, dict[str, Any]] = {}
    for better_name, reference_name in comparison_specs:
        key = f"{better_name}_minus_{reference_name}"
        paired_comparisons[key] = _paired_comparison(
            better_name=better_name,
            reference_name=reference_name,
            policies=policies,
            bootstrap_resamples=args.bootstrap_resamples,
            random_seed=args.random_seed,
        )

    required_headroom_pairs = (
        "ideal_observer_minus_random_action",
        "search_oracle_minus_random_action",
        "ideal_observer_minus_baseline",
        "search_oracle_minus_baseline",
    )
    headroom_gates: dict[str, bool] = {}
    for key in required_headroom_pairs:
        comparison = paired_comparisons[key]
        mean_difference = comparison[
            "episode_difference_distribution"
        ]["mean"]
        ci_lower = comparison[
            "paired_bootstrap_mean_episode_difference_95_ci"
        ][0]
        headroom_gates[f"{key}:mean_positive"] = mean_difference > 0
        headroom_gates[f"{key}:bootstrap_95_ci_lower_positive"] = ci_lower > 0

    integrity_gates = {
        "source_commit_exact": source["commit"] == args.expected_source_commit,
        "tracked_source_clean": source["tracked_worktree_clean"],
        "resource_hashes_exact": all(
            resource["match"]
            for resource in resource_provenance["bundled_resources"].values()
        ),
        "official_notebook_hash_exact": (
            resource_provenance["official_protocol_notebook"]["match"]
        ),
        "episode_count_exact": len(chemistries_and_items) == EXPECTED_EPISODES,
        "trial_count_exact": all(
            len(policy["trial_returns_by_episode"]) == EXPECTED_EPISODES
            and all(
                len(trials) == EXPECTED_TRIALS_PER_EPISODE
                for trials in policy["trial_returns_by_episode"]
            )
            for policy in policies.values()
        ),
        "official_full_replay_crosschecks_pass": all(
            all(
                check["action_sequence_match"] and check["trial_return_match"]
                for check in policies[name]["full_replay_crosscheck"]["episodes"]
            )
            for name in ("baseline", "ideal_observer", "search_oracle")
        ),
    }
    status = (
        "PASS"
        if all(integrity_gates.values()) and all(headroom_gates.values())
        else "FAIL"
    )

    result = {
        "schema_version": 1,
        "experiment": "symbolic_alchemy_headroom_qualification",
        "status": status,
        "qualification_question": (
            "Do the bundled ideal-observer and search-oracle traces retain "
            "statistically positive paired return headroom over both an "
            "official RandomActionBot and the released baseline trace on the "
            "same 1,000 fixed evaluation episodes?"
        ),
        "qualification_rule": {
            "integrity": (
                "All source/resource/count/replay-integrity gates must pass."
            ),
            "headroom": (
                "For ideal_observer and search_oracle versus random_action and "
                "baseline, both the paired mean episode-return gap and the "
                "lower endpoint of its paired-bootstrap 95% interval must be "
                "strictly positive."
            ),
            "integrity_gates": integrity_gates,
            "headroom_gates": headroom_gates,
        },
        "protocol": {
            "episode_set": (
                "official bundled 1,000 evaluation chemistries/items"
            ),
            "episodes": EXPECTED_EPISODES,
            "trials_per_episode": EXPECTED_TRIALS_PER_EPISODE,
            "stones_per_trial": EXPECTED_STONES_PER_TRIAL,
            "potions_per_trial": EXPECTED_POTIONS_PER_TRIAL,
            "reward_weights": {
                "coefficients": [1, 1, 1],
                "offset": 0,
                "bonus": 12,
            },
            "official_trace_scoring_api": (
                "dm_alchemy.event_tracker.replay_events + RewardTracker"
            ),
            "expensive_planners_rerun": False,
            "full_environment_replay_crosscheck_episode_indices": list(
                REPLAY_CROSSCHECK_EPISODES
            ),
            "random_action": {
                "root_seed": int(args.random_seed),
                "episode_seed_derivation": (
                    "uint64_big_endian("
                    "sha256('symbolic-alchemy-random-action-v1|"
                    "{root_seed}|{episode_index}')[:8])"
                ),
                "python_random_implementation": (
                    "standard-library random module from recorded Python "
                    "runtime"
                ),
            },
            "paired_bootstrap": {
                "resamples": int(args.bootstrap_resamples),
                "root_seed": int(args.random_seed),
                "unit": "episode-index paired difference",
            },
        },
        "provenance": {
            "source": source,
            "resources": resource_provenance,
            "runtime": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "dependency_versions": _dependency_versions(),
            },
        },
        "policies": policies,
        "paired_headroom": paired_comparisons,
        "runtime_seconds": float(time.perf_counter() - started),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": status,
                "runtime_seconds": result["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
