"""Optional-PyTorch smoke tests for the standalone Coverage@4 audit helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


try:
    import torch
except ImportError:
    torch = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "scripts" / "eval_rulegrid_coverage_audit.py"


def _load_audit_module() -> object:
    module_name = "prp_wm_test_coverage_audit"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load coverage audit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CoverageAuditPureTests(unittest.TestCase):
    def test_canonical_support_yields_four_weighted_triple_classes(self) -> None:
        """The coverage denominator cannot silently shrink below four classes."""

        from prp_wm.pilot import TRIPLE_DIAGNOSTIC_INDICES, make_pilot_tasks

        audit = _load_audit_module()
        tasks = make_pilot_tasks(
            split="pilot-composition",
            master_seed=2026071601,
            start=0,
            count=3,
            diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        )
        panels = audit.construct_alternative_behavior_panels(tasks)
        self.assertEqual(panels.compatible_program_counts, (4, 4, 4))
        self.assertTrue(all(len(task_panels) == 4 for task_panels in panels.targets))
        for masses in panels.masses:
            self.assertAlmostEqual(sum(masses), 1.0)
            self.assertEqual(masses, (0.25, 0.25, 0.25, 0.25))


@unittest.skipIf(torch is None, "PyTorch is an optional Stage-1 dependency")
class CoverageAuditTests(unittest.TestCase):
    def test_support_derived_four_class_triple_coverage_path(self) -> None:
        """Exercise lazy tasks, input-only inference, panels, and Coverage@4 shapes."""

        assert torch is not None
        from prp_wm.neural import NeuralPRPConfig, PersistentK4
        from prp_wm.pilot import TRIPLE_DIAGNOSTIC_INDICES, make_pilot_tasks

        audit = _load_audit_module()
        tasks = make_pilot_tasks(
            split="pilot-composition",
            master_seed=2026071601,
            start=0,
            count=2,
            diagnostic_indices=TRIPLE_DIAGNOSTIC_INDICES,
        )
        panels = audit.construct_alternative_behavior_panels(tasks)
        self.assertEqual(panels.compatible_program_counts, (4, 4))
        self.assertTrue(all(len(task_panels) == 4 for task_panels in panels.targets))
        for masses in panels.masses:
            self.assertAlmostEqual(sum(masses), 1.0)

        config = NeuralPRPConfig(
            color_embedding=16,
            position_embedding=16,
            encoder_channels=16,
            encoder_resblocks=1,
            normalization_groups=4,
            action_embedding=16,
            rule_dim=32,
            attention_ffn=64,
            decoder_resblocks=1,
        )
        model = PersistentK4(config).eval()
        batch = audit.make_public_triple_input_batch(torch, tasks, device=torch.device("cpu"))
        batch.validate(config)
        tensor_panels = audit.tensorize_alternative_behavior_panels(
            torch, panels, device=torch.device("cpu")
        )
        with torch.no_grad():
            inference = model.infer_support(batch)
            prediction = audit.predict_all_triple_modes(torch, model, batch, inference)
            score = audit.score_coverage_at_4(
                torch,
                prediction,
                tensor_panels,
                batch_size=2,
                nll_threshold_per_cell=0.05,
            )
        self.assertEqual(score.class_covered.shape, (2, 4))
        self.assertEqual(score.map_exact_by_mode.shape, (2, 4, 4))
        self.assertEqual(score.nll_threshold_by_mode.shape, (2, 4, 4))
        self.assertEqual(score.qualifying_modes.shape, (2, 4, 4))
        self.assertTrue(torch.allclose(score.class_mass.sum(dim=1), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
