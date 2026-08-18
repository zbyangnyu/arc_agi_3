from __future__ import annotations

import math
import unittest

from prp_wm.rulegame import (
    CONTINUE,
    FIRST_DOOR,
    PROBE,
    RuleGame,
    make_rulegame_specs,
)


class RuleGameEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = make_rulegame_specs(
            split="rulegame-unit", master_seed=17, start=0, count=4
        )

    def test_four_modes_share_initial_and_terminal_frames(self) -> None:
        games = [RuleGame(spec) for spec in self.specs]
        self.assertEqual(len({game.observation.grid for game in games}), 1)
        evidences = []
        decisions = []
        for game in games:
            first = game.step(PROBE)
            self.assertEqual(first.reward, 0.0)
            evidences.append(first.observation.grid)
            second = game.step(CONTINUE)
            self.assertEqual(second.reward, 0.0)
            decisions.append(second.observation.grid)
        self.assertEqual(len(set(evidences)), 4)
        self.assertEqual(len(set(decisions)), 1)

    def test_only_correct_terminal_door_receives_reward(self) -> None:
        for mode_index, spec in enumerate(self.specs):
            game = RuleGame(spec)
            game.step(PROBE)
            game.step(CONTINUE)
            result = game.step(FIRST_DOOR + mode_index)
            self.assertTrue(result.done)
            self.assertTrue(result.won)
            self.assertEqual(result.reward, 1.0)

            skipped = RuleGame(spec)
            skipped.step(CONTINUE)
            skipped.step(CONTINUE)
            result = skipped.step(FIRST_DOOR + mode_index)
            self.assertFalse(result.won)
            self.assertEqual(result.reward, 0.0)


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class RuleGameRLTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = make_rulegame_specs(
            split="rulegame-rl-unit", master_seed=19, start=0, count=8
        )

    def test_ppo_and_grpo_updates_are_finite(self) -> None:
        from prp_wm.rulegame_rl import (
            RuleGamePolicy,
            train_grpo_update,
            train_ppo_update,
        )

        for algorithm in ("ppo", "grpo"):
            torch.manual_seed(7)
            policy = RuleGamePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
            before = next(policy.parameters()).detach().clone()
            metrics = (
                train_ppo_update(policy, optimizer, self.specs)
                if algorithm == "ppo"
                else train_grpo_update(
                    policy, optimizer, self.specs[:4], group_size=4
                )
            )
            self.assertTrue(all(math.isfinite(value) for value in metrics.__dict__.values()))
            self.assertFalse(torch.equal(before, next(policy.parameters()).detach()))

    def test_rollout_has_terminal_reward_only(self) -> None:
        from prp_wm.rulegame_rl import RuleGamePolicy, collect_rulegame_rollout

        torch.manual_seed(23)
        rollout = collect_rulegame_rollout(RuleGamePolicy(), self.specs)
        self.assertTrue(torch.equal(rollout.rewards[:, :2], torch.zeros_like(rollout.rewards[:, :2])))
        self.assertTrue(torch.all((rollout.rewards[:, 2] == 0) | (rollout.rewards[:, 2] == 1)))


if __name__ == "__main__":
    unittest.main()
