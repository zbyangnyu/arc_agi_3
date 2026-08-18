from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from prp_wm.reproducibility import (
    ReproducibilityError,
    canonical_json_bytes,
    load_stage0a_config,
    parse_stage0a_config,
    repository_root,
    sha256_bytes,
    verify_stage0a_manifest,
)


ROOT = repository_root()
CONFIG_PATH = ROOT / "configs/stage0a.json"


class Stage0AReproducibilityTests(unittest.TestCase):
    def test_reference_artifact_hash_is_frozen(self) -> None:
        config = load_stage0a_config(CONFIG_PATH)
        artifact = ROOT / config.result_path
        self.assertEqual(sha256_bytes(artifact.read_bytes()), config.expected_result_sha256)

    def test_manifest_matches_declared_runtime_files(self) -> None:
        config = load_stage0a_config(CONFIG_PATH)
        manifest = verify_stage0a_manifest(ROOT, config)
        self.assertEqual(manifest["experiment_id"], config.experiment_id)
        self.assertEqual(
            set(manifest["file_hashes"]),  # type: ignore[arg-type]
            set(config.runtime_files),
        )

    def test_fresh_reference_result_matches_checked_in_bytes(self) -> None:
        config = load_stage0a_config(CONFIG_PATH)
        from prp_wm.reproducibility import reference_result_bytes

        fresh_result = reference_result_bytes(config)
        artifact = (ROOT / config.result_path).read_bytes()
        self.assertEqual(fresh_result, artifact)
        self.assertEqual(sha256_bytes(fresh_result), config.expected_result_sha256)

    def test_config_rejects_unknown_fields(self) -> None:
        payload = copy.deepcopy(json.loads(CONFIG_PATH.read_text()))
        payload["unreviewed_change"] = True
        with self.assertRaises(ReproducibilityError):
            parse_stage0a_config(payload)

    def test_canonical_json_terminates_with_one_newline(self) -> None:
        self.assertEqual(canonical_json_bytes({"b": 1, "a": 2}), b'{\n  "a": 2,\n  "b": 1\n}\n')


if __name__ == "__main__":
    unittest.main()
