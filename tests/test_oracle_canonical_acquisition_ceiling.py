"""Tests for the privileged oracle-canonical acquisition ceiling."""

from __future__ import annotations

from argparse import Namespace
import inspect
import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency.
    torch = None

from scripts.run_oracle_canonical_acquisition_ceiling import (
    ACTIVE_EXECUTOR_SCHEMA_VERSION,
    COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION,
    _select_acquisition_candidate,
    _selector_boundary_audit,
    _uniform_orders,
    _validate_active_executor_artifact,
    _validate_args,
    bayesian_log_likelihood_update,
    factor_marginals_to_joint_log_weights,
)


class AcquisitionCeilingArgumentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> Namespace:
        values: dict[str, object] = {
            "groups_per_axis": 2,
            "batch_size": 2,
            "trace_tasks_per_axis": 0,
            "seed": 1,
            "data_master_seed": 2,
            "split": "test",
            "budgets": (0, 1, 2, 4, 8),
        }
        values.update(overrides)
        return Namespace(**values)

    def test_budget_validation_sorts_without_duplicates(self) -> None:
        self.assertEqual(
            _validate_args(self._args(budgets=(8, 0, 4, 1, 2))),
            (0, 1, 2, 4, 8),
        )
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 1, 1)))
        with self.assertRaises(SystemExit):
            _validate_args(self._args(budgets=(0, 9)))


@unittest.skipIf(torch is None, "PyTorch is an optional neural dependency")
class AcquisitionCeilingTensorTests(unittest.TestCase):
    def test_factor_marginals_expand_to_normalized_joint(self) -> None:
        bank = torch.cartesian_prod(*(torch.arange(4) for _ in range(3)))
        marginals = torch.tensor(
            [
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.1, 0.1, 0.7],
                ]
            ],
            dtype=torch.float64,
        )
        joint = factor_marginals_to_joint_log_weights(
            torch,
            marginals,
            bank,
        )
        self.assertEqual(tuple(joint.shape), (1, 64))
        self.assertTrue(
            torch.allclose(
                joint.exp().sum(dim=-1),
                torch.ones(1, dtype=joint.dtype),
            )
        )
        self.assertEqual(bank[int(joint.argmax())].tolist(), [0, 2, 3])

    def test_expected_door_gain_is_primary_selection_key(self) -> None:
        # Candidate 0 distinguishes within each door but never resolves the
        # door. Candidate 1 directly separates the two possible doors.
        outcomes = torch.tensor(
            [
                [[[0]], [[1]], [[0]], [[1]]],
                [[[0]], [[0]], [[1]], [[1]]],
            ],
            dtype=torch.long,
        )
        selected = _select_acquisition_candidate(
            torch,
            torch.full((4,), -torch.log(torch.tensor(4.0))),
            torch.tensor([0, 0, 1, 1]),
            outcomes,
            torch.ones(2, dtype=torch.bool),
        )
        self.assertEqual(selected.candidate_index, 1)
        self.assertAlmostEqual(selected.expected_door_gain, 0.5, places=6)

    def test_outcome_eig_is_secondary_then_index_breaks_tie(self) -> None:
        log_weights = torch.full((4,), -torch.log(torch.tensor(4.0)))
        door_values = torch.zeros(4, dtype=torch.long)
        outcomes = torch.tensor(
            [
                [[[0]], [[0]], [[0]], [[0]]],
                [[[0]], [[0]], [[1]], [[1]]],
                [[[0]], [[0]], [[1]], [[1]]],
            ],
            dtype=torch.long,
        )
        selected = _select_acquisition_candidate(
            torch,
            log_weights,
            door_values,
            outcomes,
            torch.ones(3, dtype=torch.bool),
        )
        self.assertEqual(selected.candidate_index, 1)
        self.assertAlmostEqual(selected.expected_door_gain, 0.0, places=6)
        self.assertAlmostEqual(
            selected.outcome_information_gain_nats,
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )

    def test_likelihood_update_is_normalized_bayes_rule(self) -> None:
        prior = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
        likelihood = torch.log(
            torch.tensor([[0.9, 0.1]], dtype=torch.float64)
        )
        posterior, log_evidence = bayesian_log_likelihood_update(
            torch,
            prior,
            likelihood,
        )
        self.assertTrue(
            torch.allclose(
                posterior.exp(),
                torch.tensor([[0.9, 0.1]], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                log_evidence,
                torch.log(torch.tensor([0.5], dtype=torch.float64)),
            )
        )

    def test_uniform_orders_are_shared_within_hidden_value_group(self) -> None:
        orders = _uniform_orders(
            torch=torch,
            axis_index=1,
            groups=3,
            candidates=8,
            seed=17,
        )
        self.assertEqual(len(orders), 12)
        for start in range(0, len(orders), 4):
            self.assertEqual(len(set(orders[start : start + 4])), 1)
            self.assertEqual(sorted(orders[start]), list(range(8)))


class AcquisitionCeilingBoundaryTests(unittest.TestCase):
    def test_selector_has_tensor_only_non_leaking_boundary(self) -> None:
        audit = _selector_boundary_audit()
        self.assertTrue(audit["passed"])
        self.assertEqual(
            tuple(inspect.signature(_select_acquisition_candidate).parameters),
            (
                "torch",
                "log_weights",
                "door_values",
                "candidate_outcomes",
                "available",
            ),
        )


class ActiveExecutorArtifactValidationTests(unittest.TestCase):
    @staticmethod
    def _artifact(
        schema: str = ACTIVE_EXECUTOR_SCHEMA_VERSION,
    ) -> tuple[dict[str, object], dict[str, object], str]:
        checksum = "a" * 64
        checkpoint: dict[str, object] = {
            "checkpoint_schema_version": schema,
            "model_type": "OracleFactorExecutor",
        }
        result: dict[str, object] = {
            "checkpoint_schema_version": schema,
            "checkpoint_sha256": checksum,
            "model_type": "OracleFactorExecutor",
            "active_prefix_executor_gate": {"passed": True},
        }
        if schema == COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION:
            lineage = {
                "schema_version": ACTIVE_EXECUTOR_SCHEMA_VERSION,
                "sha256": "b" * 64,
                "path": "/frozen/parent/checkpoint_last.pt",
            }
            checkpoint["initial_checkpoint_provenance"] = lineage
            result["initial_checkpoint_provenance"] = dict(lineage)
        return checkpoint, result, checksum

    def test_accepts_audited_base_and_v3_continuation(self) -> None:
        for schema in (
            ACTIVE_EXECUTOR_SCHEMA_VERSION,
            COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION,
        ):
            checkpoint, result, checksum = self._artifact(schema)
            _validate_active_executor_artifact(
                checkpoint=checkpoint,
                result=result,
                checkpoint_sha256=checksum,
            )

    def test_accepts_routed_executor_only_as_v3_continuation(self) -> None:
        experimental_types = (
            "CanonicalRoleRoutedOracleFactorExecutor",
            "MatchedWiderGlobalOracleFactorExecutor",
            "MatchedFactorLocalOracleFactorExecutor",
        )
        for model_type in experimental_types:
            checkpoint, result, checksum = self._artifact(
                COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION
            )
            checkpoint["model_type"] = model_type
            result["model_type"] = model_type
            _validate_active_executor_artifact(
                checkpoint=checkpoint,
                result=result,
                checkpoint_sha256=checksum,
            )

            checkpoint, result, checksum = self._artifact()
            checkpoint["model_type"] = model_type
            result["model_type"] = model_type
            with self.assertRaises(SystemExit):
                _validate_active_executor_artifact(
                    checkpoint=checkpoint,
                    result=result,
                    checkpoint_sha256=checksum,
                )

    def test_rejects_v3_without_matching_parent_lineage(self) -> None:
        checkpoint, result, checksum = self._artifact(
            COUNTERFACTUAL_LOCALITY_EXECUTOR_SCHEMA_VERSION
        )
        checkpoint["initial_checkpoint_provenance"] = {
            "schema_version": "wrong",
            "sha256": "b" * 64,
        }
        with self.assertRaises(SystemExit):
            _validate_active_executor_artifact(
                checkpoint=checkpoint,
                result=result,
                checkpoint_sha256=checksum,
            )

    def test_rejects_stale_result_or_failed_gate(self) -> None:
        checkpoint, result, checksum = self._artifact()
        result["checkpoint_sha256"] = "c" * 64
        with self.assertRaises(SystemExit):
            _validate_active_executor_artifact(
                checkpoint=checkpoint,
                result=result,
                checkpoint_sha256=checksum,
            )
        result["checkpoint_sha256"] = checksum
        result["active_prefix_executor_gate"] = {"passed": False}
        with self.assertRaises(SystemExit):
            _validate_active_executor_artifact(
                checkpoint=checkpoint,
                result=result,
                checkpoint_sha256=checksum,
            )


if __name__ == "__main__":
    unittest.main()
