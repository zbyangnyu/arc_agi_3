from __future__ import annotations

import statistics
import unittest
from unittest.mock import patch

import prp_wm.rulegrid as rulegrid_module

from prp_wm.rulegrid import (
    ALL_AXES,
    ALL_PROGRAMS,
    ActionKind,
    Axis,
    Collision,
    CompositeAction,
    DEFAULT_PALETTE,
    Direction,
    GridAction,
    Relation,
    RuleGridProbe,
    RuleProgram,
    Trigger,
    behavior_classes,
    count_changed_cells,
    derive_seed64,
    expected_heldout_version_space,
    grid_with_cells,
    make_rulegrid_task,
    outcome_partition,
    partition_sizes,
    select_exact_oracle_probe,
    simulate,
    version_space,
)
from prp_wm.rulegrid_evaluation import evaluate_gate0b


P = DEFAULT_PALETTE


def program_for(
    collision: Collision = Collision.STOP,
    trigger: Trigger = Trigger.TOGGLE,
    relation: Relation = Relation.SWAP,
) -> RuleProgram:
    return RuleProgram(collision, trigger, relation)


class RuleGridDSLSemanticsTests(unittest.TestCase):
    def test_program_id_bijection_is_protocol_order(self) -> None:
        self.assertEqual(tuple(program.program_id for program in ALL_PROGRAMS), tuple(range(64)))
        self.assertEqual(
            RuleProgram.from_program_id(0),
            RuleProgram(Collision.STOP, Trigger.TOGGLE, Relation.SWAP),
        )
        self.assertEqual(
            RuleProgram.from_program_id(63),
            RuleProgram(Collision.PUSH, Trigger.RECOLOR, Relation.NONE),
        )
        for program in ALL_PROGRAMS:
            self.assertEqual(RuleProgram.from_program_id(program.program_id), program)

    def test_collision_truth_table(self) -> None:
        state = grid_with_cells({(3, 2): P.actor, (3, 3): P.blocker})
        action = GridAction(ActionKind.MOVE, (3, 2), Direction.EAST)
        expected = {
            Collision.STOP: state,
            Collision.BOUNCE: grid_with_cells({(3, 1): P.actor, (3, 3): P.blocker}),
            Collision.PASS: grid_with_cells({(3, 4): P.actor, (3, 3): P.blocker}),
            Collision.PUSH: grid_with_cells({(3, 3): P.actor, (3, 4): P.blocker}),
        }
        for mode, target in expected.items():
            self.assertEqual(simulate(state, action, program_for(collision=mode)), target)

    def test_collision_invalid_target_is_a_full_noop(self) -> None:
        state = grid_with_cells(
            {(3, 1): P.actor, (3, 2): P.blocker, (3, 3): P.distractor}
        )
        action = GridAction(ActionKind.MOVE, (3, 1), Direction.EAST)
        self.assertEqual(
            simulate(state, action, program_for(collision=Collision.PASS)), state
        )
        self.assertEqual(
            simulate(state, action, program_for(collision=Collision.PUSH)), state
        )
        self.assertEqual(
            simulate(state, action, program_for(collision=Collision.STOP)), state
        )
        self.assertEqual(
            simulate(state, action, program_for(collision=Collision.BOUNCE)),
            grid_with_cells(
                {(3, 0): P.actor, (3, 2): P.blocker, (3, 3): P.distractor}
            ),
        )

    def test_trigger_truth_table_and_trigger_is_unchanged(self) -> None:
        state = grid_with_cells(
            {(3, 2): P.trigger, (3, 3): P.payload_p0, (3, 4): P.socket}
        )
        action = GridAction(ActionKind.ACTIVATE, (3, 2))
        expected = {
            Trigger.TOGGLE: grid_with_cells(
                {(3, 2): P.trigger, (3, 3): P.payload_p1, (3, 4): P.socket}
            ),
            Trigger.DELETE: grid_with_cells({(3, 2): P.trigger, (3, 4): P.socket}),
            Trigger.SPAWN: grid_with_cells(
                {(3, 2): P.trigger, (3, 3): P.payload_p0, (3, 4): P.payload_p0}
            ),
            Trigger.RECOLOR: grid_with_cells(
                {(3, 2): P.trigger, (3, 3): P.payload_p2, (3, 4): P.socket}
            ),
        }
        for mode, target in expected.items():
            self.assertEqual(simulate(state, action, program_for(trigger=mode)), target)

    def test_relation_truth_table(self) -> None:
        state = grid_with_cells({(3, 2): P.object_a, (3, 3): P.object_b})
        action = GridAction(ActionKind.MOVE, (3, 2), Direction.EAST)
        expected = {
            Relation.SWAP: grid_with_cells({(3, 2): P.object_b, (3, 3): P.object_a}),
            Relation.FOLLOW: grid_with_cells({(3, 3): P.object_a, (3, 4): P.object_b}),
            Relation.REPEL: grid_with_cells({(3, 2): P.object_a, (3, 4): P.object_b}),
            Relation.NONE: state,
        }
        for mode, target in expected.items():
            self.assertEqual(simulate(state, action, program_for(relation=mode)), target)

    def test_pulse_is_rule_independent_and_changes_four_cells(self) -> None:
        state = grid_with_cells(
            {
                (2, 2): P.pulse_d0,
                (2, 3): P.pulse_d1,
                (3, 2): P.pulse_d1,
                (3, 3): P.pulse_d0,
            }
        )
        action = GridAction(ActionKind.MOVE, (0, 0), Direction.EAST)
        targets = {simulate(state, action, program) for program in ALL_PROGRAMS}
        self.assertEqual(len(targets), 1)
        target = targets.pop()
        self.assertEqual(count_changed_cells(state, target), 4)
        self.assertEqual(target[2][2], P.pulse_d1)
        self.assertEqual(target[2][3], P.pulse_d0)

    def test_composite_events_are_simultaneous_and_disjoint(self) -> None:
        state = grid_with_cells(
            {
                (1, 1): P.actor,
                (1, 2): P.blocker,
                (4, 2): P.trigger,
                (4, 3): P.payload_p0,
                (4, 4): P.socket,
            }
        )
        action = CompositeAction(
            (
                GridAction(ActionKind.MOVE, (1, 1), Direction.EAST),
                GridAction(ActionKind.ACTIVATE, (4, 2)),
            )
        )
        target = simulate(
            state,
            action,
            program_for(collision=Collision.PUSH, trigger=Trigger.SPAWN),
        )
        self.assertEqual(
            target,
            grid_with_cells(
                {
                    (1, 2): P.actor,
                    (1, 3): P.blocker,
                    (4, 2): P.trigger,
                    (4, 3): P.payload_p0,
                    (4, 4): P.payload_p0,
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            CompositeAction(
                (
                    GridAction(ActionKind.MOVE, (1, 1), Direction.EAST),
                    GridAction(ActionKind.MOVE, (1, 1), Direction.EAST),
                )
            )


class RuleGridTaskAndOracleTests(unittest.TestCase):
    def test_seed_stream_is_stable_and_has_no_program_input(self) -> None:
        first = derive_seed64("gate0b", Axis.COLLISION, 7, "palette")
        self.assertEqual(first, derive_seed64("gate0b", Axis.COLLISION, 7, "palette"))
        self.assertNotEqual(first, derive_seed64("gate0b", Axis.COLLISION, 7, "geometry"))

    def test_calibration_keeps_exactly_heldout_axis_and_neutral_support_is_neutral(self) -> None:
        program = RuleProgram.from_program_id(37)
        for axis in ALL_AXES:
            task = make_rulegrid_task(program, axis, replicate=3)
            expected = expected_heldout_version_space(program, axis)
            self.assertEqual(version_space(task.inference.support[:2], task.privileged.palette), expected)
            self.assertEqual(version_space(task.inference.support, task.privileged.palette), expected)
            self.assertEqual(len(expected), 4)

    def test_active_bank_has_exact_partitions_and_large_neutral_change(self) -> None:
        program = RuleProgram.from_program_id(37)
        for axis in ALL_AXES:
            task = make_rulegrid_task(program, axis, replicate=4)
            versions = expected_heldout_version_space(program, axis)
            strong_changes: list[int] = []
            neutral_changes: list[int] = []
            kinds = dict.fromkeys(("strong", "partial", "neutral-large-change"), 0)
            for kind, probe in zip(
                task.privileged.candidate_kinds,
                task.inference.active_candidates,
                strict=True,
            ):
                kinds[kind] += 1
                sizes = partition_sizes(versions, probe, task.privileged.palette)
                if kind == "strong":
                    self.assertEqual(sizes, (1, 1, 1, 1))
                    strong_changes.extend(
                        count_changed_cells(
                            probe.state,
                            simulate(probe.state, probe.action, rule, task.privileged.palette),
                        )
                        for rule in versions
                    )
                elif kind == "partial":
                    self.assertEqual(sizes, (1, 3))
                else:
                    self.assertEqual(sizes, (4,))
                    neutral_changes.extend(
                        count_changed_cells(
                            probe.state,
                            simulate(probe.state, probe.action, rule, task.privileged.palette),
                        )
                        for rule in versions
                    )
            self.assertEqual(kinds, {"strong": 2, "partial": 2, "neutral-large-change": 4})
            self.assertGreaterEqual(min(neutral_changes), statistics.median(strong_changes))

    def test_panel_behavior_signature_identifies_all_programs(self) -> None:
        task = make_rulegrid_task(RuleProgram.from_program_id(0), Axis.COLLISION, 0)
        classes = behavior_classes(
            ALL_PROGRAMS, task.inference.diagnostics, task.privileged.palette
        )
        self.assertEqual(len(classes), 64)
        self.assertTrue(all(len(members) == 1 for members in classes.values()))

    def test_selected_diagnostic_targets_are_constructed_without_triples(self) -> None:
        """The composition holdout must not be simulated during train construction."""

        real_simulate = rulegrid_module.simulate
        nontriple = tuple(range(21))
        with patch.object(rulegrid_module, "simulate", wraps=real_simulate) as mocked:
            task = rulegrid_module.make_rulegrid_task(
                RuleProgram.from_program_id(37),
                Axis.COLLISION,
                replicate=12,
                diagnostic_indices=nontriple,
            )
        # Six observed support transitions, eight active sidecar targets, and
        # exactly the 21 selected diagnostic targets.  Public triple probes do
        # not invoke the simulator merely by being present in the controller view.
        self.assertEqual(mocked.call_count, 6 + 8 + len(nontriple))
        self.assertEqual(task.privileged.diagnostic_target_indices, nontriple)
        self.assertEqual(len(task.privileged.diagnostic_targets), len(nontriple))
        with self.assertRaisesRegex(ValueError, "was not materialized"):
            task.privileged.diagnostic_target_for(21)

    def test_triple_only_diagnostic_construction_is_indexed(self) -> None:
        triples = (21, 22, 23)
        real_simulate = rulegrid_module.simulate
        with patch.object(rulegrid_module, "simulate", wraps=real_simulate) as mocked:
            task = make_rulegrid_task(
                RuleProgram.from_program_id(37),
                Axis.COLLISION,
                replicate=12,
                diagnostic_indices=triples,
            )
        self.assertEqual(mocked.call_count, 6 + 8 + len(triples))
        self.assertEqual(task.privileged.diagnostic_target_indices, triples)
        self.assertEqual(len(task.privileged.diagnostic_targets), len(triples))
        with self.assertRaisesRegex(ValueError, "was not materialized"):
            task.privileged.diagnostic_target_for(20)

    def test_controller_view_has_no_privileged_fields_or_heldout_mode_leakage(self) -> None:
        base = RuleProgram(Collision.STOP, Trigger.DELETE, Relation.REPEL)
        tasks = [
            make_rulegrid_task(program, Axis.COLLISION, 1)
            for program in expected_heldout_version_space(base, Axis.COLLISION)
        ]
        for task in tasks[1:]:
            self.assertEqual(task.inference.support, tasks[0].inference.support)
            self.assertEqual(task.inference.active_candidates, tasks[0].inference.active_candidates)
            self.assertEqual(task.inference.diagnostics, tasks[0].inference.diagnostics)
        with self.assertRaises(PermissionError):
            _ = tasks[0].inference.true_rule
        with self.assertRaises(PermissionError):
            _ = tasks[0].inference.version_space

    def test_exact_oracle_selects_a_public_strong_probe(self) -> None:
        program = RuleProgram.from_program_id(37)
        task = make_rulegrid_task(program, Axis.RELATION, 9)
        versions = version_space(task.inference.support, task.privileged.palette)
        selected = select_exact_oracle_probe(
            versions,
            task.inference.active_candidates,
            task.inference.diagnostics,
            task.privileged.palette,
        )
        probe_index = next(
            index
            for index, probe in enumerate(task.inference.active_candidates)
            if probe.probe_id == selected.probe_id
        )
        self.assertEqual(task.privileged.candidate_kinds[probe_index], "strong")
        self.assertAlmostEqual(selected.eig_bits, 2.0)

    def test_small_exact_gate0b_reports_headroom_but_is_not_eligible(self) -> None:
        report = evaluate_gate0b(repeats=1, bootstrap_resamples=20, seed=17)
        self.assertEqual(report.tasks, 64 * 3)
        self.assertAlmostEqual(report.oracle_rmst4, 1.0)
        self.assertAlmostEqual(report.uniform_exact_rmst4, 17.0 / 7.0)
        self.assertGreater(report.relative_rmst_reduction, 0.25)
        self.assertGreater(report.uniform_minus_oracle_rmst4_ci95[0], 0.0)
        self.assertFalse(report.gate_eligible)
        self.assertFalse(report.passes)


if __name__ == "__main__":
    unittest.main()
