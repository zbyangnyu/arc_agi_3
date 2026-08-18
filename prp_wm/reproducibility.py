"""Small, dependency-free helpers for byte-level Stage 0-A replay.

The benchmark result is deliberately JSON-only.  Keeping the configuration,
canonical serializer, reference artifact, and source manifest in one module
makes it possible to verify a rerun without relying on an experiment tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


CONFIG_SCHEMA_VERSION = "prp-wm.stage0a.config.v1"
MANIFEST_SCHEMA_VERSION = "prp-wm.stage0a.manifest.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVALUATION_KEYS = frozenset(
    {
        "bootstrap_resamples",
        "budget",
        "gate_threshold",
        "repeats",
        "seed",
        "trials",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "evaluation",
        "expected_result_sha256",
        "experiment_id",
        "manifest_path",
        "python_version",
        "result_path",
        "runtime_files",
        "schema_version",
    }
)


class ReproducibilityError(RuntimeError):
    """Raised when a frozen artifact or reproducibility contract is invalid."""


@dataclass(frozen=True)
class Stage0AConfig:
    """Validated, immutable parameters for the official Stage 0-A replay."""

    experiment_id: str
    python_version: str
    evaluation: dict[str, int | float]
    expected_result_sha256: str
    result_path: str
    manifest_path: str
    runtime_files: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON in the exact form used by the reference result artifact."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReproducibilityError(f"cannot read JSON object {path}: {error}") from error
    if type(value) is not dict:
        raise ReproducibilityError(f"{path} must contain a JSON object")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ReproducibilityError(
            f"{label} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ReproducibilityError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _require_relative_file_path(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ReproducibilityError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ReproducibilityError(f"{label} must remain inside the repository: {value!r}")
    return path.as_posix()


def parse_stage0a_config(payload: Mapping[str, Any]) -> Stage0AConfig:
    """Validate a decoded Stage 0-A config without accepting extra fields."""

    _require_exact_keys(payload, _CONFIG_KEYS, "Stage 0-A config")
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ReproducibilityError(
            f"unsupported config schema: {payload['schema_version']!r}"
        )
    if type(payload["experiment_id"]) is not str or not payload["experiment_id"]:
        raise ReproducibilityError("experiment_id must be a non-empty string")
    if type(payload["python_version"]) is not str or not re.fullmatch(
        r"\d+\.\d+\.\d+", payload["python_version"]
    ):
        raise ReproducibilityError("python_version must use major.minor.patch form")

    evaluation = payload["evaluation"]
    if type(evaluation) is not dict:
        raise ReproducibilityError("evaluation must be a JSON object")
    _require_exact_keys(evaluation, _EVALUATION_KEYS, "Stage 0-A evaluation")
    integer_fields = ("bootstrap_resamples", "budget", "repeats", "seed", "trials")
    for field in integer_fields:
        if type(evaluation[field]) is not int:
            raise ReproducibilityError(f"evaluation.{field} must be an integer")
    if evaluation["bootstrap_resamples"] <= 0:
        raise ReproducibilityError("evaluation.bootstrap_resamples must be positive")
    if evaluation["budget"] <= 0:
        raise ReproducibilityError("evaluation.budget must be positive")
    if evaluation["repeats"] <= 0:
        raise ReproducibilityError("evaluation.repeats must be positive")
    if evaluation["seed"] < 0:
        raise ReproducibilityError("evaluation.seed must be non-negative")
    if evaluation["trials"] <= 0:
        raise ReproducibilityError("evaluation.trials must be positive")
    if type(evaluation["gate_threshold"]) not in (int, float) or type(
        evaluation["gate_threshold"]
    ) is bool:
        raise ReproducibilityError("evaluation.gate_threshold must be numeric")
    if not 0.0 < float(evaluation["gate_threshold"]) < 1.0:
        raise ReproducibilityError(
            "evaluation.gate_threshold must lie strictly between zero and one"
        )
    if (
        evaluation["trials"] < 500
        or evaluation["trials"] % 8 != 0
        or evaluation["repeats"] < 4
        or evaluation["bootstrap_resamples"] < 1_000
        or evaluation["budget"] < 2
    ):
        raise ReproducibilityError(
            "official Stage 0-A config must satisfy the preregistered gate-eligibility "
            "minimums"
        )

    raw_runtime_files = payload["runtime_files"]
    if type(raw_runtime_files) is not list or not raw_runtime_files:
        raise ReproducibilityError("runtime_files must be a non-empty JSON array")
    runtime_files = tuple(
        _require_relative_file_path(value, f"runtime_files[{index}]")
        for index, value in enumerate(raw_runtime_files)
    )
    if len(set(runtime_files)) != len(runtime_files):
        raise ReproducibilityError("runtime_files must not contain duplicates")

    return Stage0AConfig(
        experiment_id=payload["experiment_id"],
        python_version=payload["python_version"],
        evaluation=dict(evaluation),
        expected_result_sha256=_require_sha256(
            payload["expected_result_sha256"], "expected_result_sha256"
        ),
        result_path=_require_relative_file_path(payload["result_path"], "result_path"),
        manifest_path=_require_relative_file_path(
            payload["manifest_path"], "manifest_path"
        ),
        runtime_files=runtime_files,
    )


def load_stage0a_config(path: Path) -> Stage0AConfig:
    """Load the frozen Stage 0-A configuration and reject silent drift."""

    return parse_stage0a_config(_read_json_object(path))


def assert_exact_python_version(expected: str) -> str:
    """Require an exact interpreter patch release for a byte-level claim."""

    actual = ".".join(str(part) for part in sys.version_info[:3])
    if actual != expected:
        raise ReproducibilityError(
            f"Stage 0-A R1 replay requires Python {expected}; found {actual}"
        )
    return actual


def reference_result_bytes(config: Stage0AConfig) -> bytes:
    """Run the frozen evaluation and return its canonical result bytes."""

    from .evaluation import evaluate_gate0

    report = evaluate_gate0(**config.evaluation)
    return canonical_json_bytes(report.to_dict())


def build_runtime_file_hashes(root: Path, runtime_files: tuple[str, ...]) -> dict[str, str]:
    """Hash exactly the files declared in the frozen configuration."""

    hashes: dict[str, str] = {}
    for relative_path in runtime_files:
        path = root / relative_path
        if not path.is_file():
            raise ReproducibilityError(f"declared runtime file is missing: {relative_path}")
        hashes[relative_path] = sha256_file(path)
    return hashes


def build_stage0a_manifest(root: Path, config: Stage0AConfig) -> dict[str, object]:
    """Create the deterministic content for the source manifest (without writing it)."""

    config_path = root / "configs/stage0a.json"
    result_path = root / config.result_path
    if not config_path.is_file():
        raise ReproducibilityError("configs/stage0a.json is missing")
    if not result_path.is_file():
        raise ReproducibilityError(f"reference result is missing: {config.result_path}")
    return {
        "artifact_hashes": {config.result_path: sha256_file(result_path)},
        "config_sha256": sha256_file(config_path),
        "experiment_id": config.experiment_id,
        "file_hashes": build_runtime_file_hashes(root, config.runtime_files),
        "manifest_version": MANIFEST_SCHEMA_VERSION,
    }


def verify_stage0a_manifest(root: Path, config: Stage0AConfig) -> dict[str, object]:
    """Verify the checked-in source/result manifest against the current tree."""

    manifest_path = root / config.manifest_path
    payload = _read_json_object(manifest_path)
    expected = build_stage0a_manifest(root, config)
    if payload != expected:
        raise ReproducibilityError(
            f"Stage 0-A manifest mismatch: regenerate and review {config.manifest_path}"
        )
    return payload


def verify_stage0a_reference(root: Path, config: Stage0AConfig) -> dict[str, object]:
    """Perform the complete R1 verification, including source and result hashes."""

    actual_python = assert_exact_python_version(config.python_version)
    result_bytes = reference_result_bytes(config)
    result_hash = sha256_bytes(result_bytes)
    if result_hash != config.expected_result_sha256:
        raise ReproducibilityError(
            "fresh Stage 0-A replay does not match expected result SHA256: "
            f"expected {config.expected_result_sha256}, got {result_hash}"
        )
    result_path = root / config.result_path
    if not result_path.is_file():
        raise ReproducibilityError(f"reference result is missing: {config.result_path}")
    if result_path.read_bytes() != result_bytes:
        raise ReproducibilityError(
            f"checked-in result differs byte-for-byte from fresh replay: {config.result_path}"
        )
    manifest = verify_stage0a_manifest(root, config)
    return {
        "config_sha256": manifest["config_sha256"],
        "experiment_id": config.experiment_id,
        "python_version": actual_python,
        "result_sha256": result_hash,
        "source_manifest_verified": True,
    }
