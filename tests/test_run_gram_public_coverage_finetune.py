"""Pure runner checks for the public-only GRAM coverage continuation."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from prp_wm.causal_filter import enumerate_factor_codes
    from prp_wm.gram_causal_rules import GRAMFactorizedCausalK4
    from prp_wm.latent_rules import OracleFactorExecutor
    from prp_wm.neural import NeuralPRPConfig
    from prp_wm.pilot import make_pilot_tasks
    from scripts.run_gram_public_coverage_finetune import (
        ARCHITECTURE_CHECKPOINT_SCHEMA_VERSION,
        _configure_trainable_scope,
        _context_from_codes,
        _load_warm_start,
        _mask_balance_audit,
        _public_support_batch,
        _sha256_file,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class GRAMPublicCoverageRunnerTests(unittest.TestCase):
    def _config(self) -> NeuralPRPConfig:
        return NeuralPRPConfig(
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

    def test_training_adapter_never_reads_privileged_targets_or_program(self) -> None:
        assert torch is not None
        tasks = make_pilot_tasks(
            split="gram-public-runner-test",
            master_seed=2026072203,
            start=0,
            count=2,
            diagnostic_indices=(0,),
        )

        class PaletteOnlyPrivilege:
            def __init__(self, palette):
                self.palette = palette

            def __getattr__(self, name):
                raise AssertionError(f"training adapter read privileged field {name!r}")

        guarded = tuple(
            SimpleNamespace(
                inference=task.inference,
                privileged=PaletteOnlyPrivilege(task.privileged.palette),
            )
            for task in tasks
        )
        batch = _public_support_batch(torch, guarded, device=torch.device("cpu"))
        self.assertEqual(batch.support_states.shape, (2, 6, 8, 8))
        self.assertEqual(batch.support_targets.shape, (2, 6, 8, 8))
        self.assertIsNone(batch.query_states)
        self.assertIsNone(batch.query_actions)
        self.assertIsNone(batch.query_targets)
        self.assertIsNone(batch.behavior_targets)
        self.assertIsNone(batch.behavior_mass)

    def test_public_context_signature_is_set_order_invariant(self) -> None:
        codes = ((1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3))
        self.assertEqual(_context_from_codes(codes), (2, 1, 2))
        self.assertEqual(
            _context_from_codes((codes[2], codes[0], codes[3], codes[1])),
            (2, 1, 2),
        )
        with self.assertRaisesRegex(ValueError, "four distinct"):
            _context_from_codes((codes[0], codes[0], codes[2], codes[3]))

    def test_fold_zero_mask_audit_proves_axis_value_balance(self) -> None:
        assert torch is not None
        bank = enumerate_factor_codes()
        bank_index = {
            tuple(int(value) for value in row): index
            for index, row in enumerate(bank.tolist())
        }
        masks = []
        # The Latin fold-0 train side has 36 contexts.  Repeat each four times,
        # matching the 144-task deterministic pool without selecting a true code.
        for free_axis in range(3):
            fixed_axes = [axis for axis in range(3) if axis != free_axis]
            for left in range(4):
                for right in range(4):
                    if (left + right) % 4 == 0:
                        continue
                    mask = torch.zeros(64, dtype=torch.bool)
                    for free_value in range(4):
                        code = [0, 0, 0]
                        code[free_axis] = free_value
                        code[fixed_axes[0]] = left
                        code[fixed_axes[1]] = right
                        mask[bank_index[tuple(code)]] = True
                    masks.extend([mask.clone() for _ in range(4)])
        audit = _mask_balance_audit(torch, bank, torch.stack(masks))
        self.assertEqual(audit["tasks"], 144)
        self.assertEqual(audit["unique_public_contexts"], 36)
        self.assertEqual(audit["context_multiplicity_histogram"], {"4": 36})
        self.assertTrue(audit["axis_value_exactly_balanced"])
        self.assertEqual(
            audit["axis_value_counts"],
            [[144, 144, 144, 144]] * 3,
        )
        self.assertEqual(audit["joint_code_union_size"], 62)
        self.assertEqual(audit["missing_joint_codes"], [[0, 0, 0], [2, 2, 2]])

    def test_warm_start_enforces_schema_sha_and_freezes_posterior(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor_path = root / "executor.pt"
            executor_path.write_bytes(b"audited executor identity")
            executor = OracleFactorExecutor(self._config())
            for parameter in executor.parameters():
                parameter.requires_grad_(False)
            original = GRAMFactorizedCausalK4(
                executor,
                recursive_steps=2,
                guidance_dim=8,
                attention_layers=2,
            )
            checkpoint = {
                "checkpoint_schema_version": ARCHITECTURE_CHECKPOINT_SCHEMA_VERSION,
                "model_type": "GRAMFactorizedCausalK4",
                "context_fold": 0,
                "executor_checkpoint": str(executor_path),
                "executor_checkpoint_sha256": _sha256_file(executor_path),
                "recursive_steps": 2,
                "guidance_dim": 8,
                "guidance_log_variance_bounds": [-8.0, 4.0],
                "initial_guidance_log_variance": -2.0,
                "truncate_between_recursive_steps": True,
                "cli_arguments": {
                    "attention_layers": 2,
                    "factor_temperature_end": 1.0,
                },
                "model_state_dict": original.state_dict(),
            }
            checkpoint_path = root / "gram.pt"
            torch.save(checkpoint, checkpoint_path)
            args = SimpleNamespace(
                initial_gram_checkpoint=checkpoint_path,
                executor_checkpoint=None,
                context_fold=None,
            )
            loader_result = {
                "checkpoint_schema_version": "audited-executor-test.v1"
            }
            with patch(
                "scripts.run_expected_discrete_causal_coverage._load_audited_executor",
                return_value=(executor, loader_result),
            ):
                loaded, metadata, initial_path, loaded_executor_path, executor_metadata = (
                    _load_warm_start(torch, args, torch.device("cpu"))
                )
            self.assertEqual(metadata["context_fold"], 0)
            self.assertEqual(initial_path, checkpoint_path.resolve())
            self.assertEqual(loaded_executor_path, executor_path.resolve())
            self.assertIs(executor_metadata, loader_result)
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in loaded.posterior_head.parameters()
                )
            )
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in loaded.posterior_behavior_to_guidance.parameters()
                )
            )

            bad_schema = dict(checkpoint)
            bad_schema["checkpoint_schema_version"] = "wrong.v1"
            bad_schema_path = root / "bad_schema.pt"
            torch.save(bad_schema, bad_schema_path)
            with self.assertRaisesRegex(SystemExit, "schema"):
                _load_warm_start(
                    torch,
                    SimpleNamespace(
                        initial_gram_checkpoint=bad_schema_path,
                        executor_checkpoint=None,
                        context_fold=None,
                    ),
                    torch.device("cpu"),
                )

            bad_sha = dict(checkpoint)
            bad_sha["executor_checkpoint_sha256"] = "0" * 64
            bad_sha_path = root / "bad_sha.pt"
            torch.save(bad_sha, bad_sha_path)
            with self.assertRaisesRegex(SystemExit, "SHA256"):
                _load_warm_start(
                    torch,
                    SimpleNamespace(
                        initial_gram_checkpoint=bad_sha_path,
                        executor_checkpoint=None,
                        context_fold=None,
                    ),
                    torch.device("cpu"),
                )

    def test_prior_head_only_is_the_only_gradient_and_update_scope(self) -> None:
        assert torch is not None
        torch.manual_seed(701)
        model = GRAMFactorizedCausalK4(
            OracleFactorExecutor(self._config()),
            recursive_steps=2,
            guidance_dim=8,
        ).train()
        trainable, names = _configure_trainable_scope(model, "prior-head-only")
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("prior_head.") for name in names))
        self.assertEqual(
            names,
            [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
        )
        tasks = make_pilot_tasks(
            split="gram-prior-only-update-test",
            master_seed=2026072204,
            start=0,
            count=1,
            diagnostic_indices=(0,),
        )
        batch = _public_support_batch(
            torch,
            tasks,
            device=torch.device("cpu"),
        )
        compatible = torch.zeros(1, 64, dtype=torch.bool)
        compatible[:, :4] = True
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        with patch.object(
            model,
            "public_support_exact_mask",
            return_value=compatible,
        ):
            loss = model.coverage_losses(batch, seed=47)
        loss.total.backward()
        gradient_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        self.assertTrue(gradient_names)
        self.assertTrue(
            all(name.startswith("prior_head.") for name in gradient_names)
        )
        optimizer.step()
        updated_names = {
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(before[name], parameter.detach())
        }
        self.assertTrue(updated_names)
        self.assertTrue(all(name.startswith("prior_head.") for name in updated_names))
        self.assertTrue(updated_names.issubset(set(names)))
        for name, parameter in model.named_parameters():
            if not name.startswith("prior_head."):
                self.assertTrue(torch.equal(before[name], parameter.detach()), name)


if __name__ == "__main__":
    unittest.main()
