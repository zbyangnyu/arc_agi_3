from __future__ import annotations

from dataclasses import fields
import math
import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class RuleGridRLBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from prp_wm.pilot import make_pilot_tasks

        cls.tasks = make_pilot_tasks(
            split="rl-unit",
            master_seed=2026071601,
            start=0,
            count=8,
            diagnostic_indices=(0,),
        )

    def test_controller_observation_has_only_public_fields(self) -> None:
        from prp_wm.rl import RuleGridRLEpisode, RuleGridRLObservation

        observation = RuleGridRLEpisode(self.tasks[0]).observation
        self.assertEqual(
            {item.name for item in fields(RuleGridRLObservation)},
            {"history", "candidates", "available"},
        )
        self.assertEqual(observation.history, self.tasks[0].inference.support)
        self.assertEqual(observation.candidates, self.tasks[0].inference.active_candidates)
        for forbidden in ("true_program", "palette", "candidate_kinds", "active_targets"):
            self.assertFalse(hasattr(observation, forbidden))

    def test_sparse_reward_only_arrives_on_identification(self) -> None:
        from prp_wm.rl import RuleGridRLEpisode

        task = self.tasks[0]
        neutral = task.privileged.candidate_kinds.index("neutral-large-change")
        neutral_result = RuleGridRLEpisode(task).step(neutral)
        self.assertEqual(neutral_result.reward, 0.0)
        self.assertFalse(neutral_result.identified)

        strong = task.privileged.candidate_kinds.index("strong")
        strong_result = RuleGridRLEpisode(task).step(strong)
        self.assertEqual(strong_result.reward, 1.0)
        self.assertTrue(strong_result.identified)
        self.assertTrue(strong_result.done)

    def test_tensorization_and_policy_mask_used_candidates(self) -> None:
        from prp_wm.rl import (
            RuleGridActorCritic,
            RuleGridRLEpisode,
            tensorize_rl_observations,
        )

        episode = RuleGridRLEpisode(self.tasks[0])
        result = episode.step(self.tasks[0].privileged.candidate_kinds.index("neutral-large-change"))
        batch = tensorize_rl_observations((result.observation,))
        policy = RuleGridActorCritic()
        logits, value = policy(batch)
        self.assertEqual(tuple(logits.shape), (1, 8))
        self.assertEqual(tuple(value.shape), (1,))
        used = result.observation.available.index(False)
        self.assertLess(float(logits[0, used].detach()), -1e20)

    def test_actor_critic_update_is_finite(self) -> None:
        from prp_wm.rl import RuleGridActorCritic, train_actor_critic_batch

        torch.manual_seed(11)
        policy = RuleGridActorCritic()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        before = next(policy.parameters()).detach().clone()
        metrics = train_actor_critic_batch(policy, self.tasks, optimizer)
        self.assertTrue(all(math.isfinite(value) for value in metrics.__dict__.values()))
        self.assertGreaterEqual(metrics.success_rate, 0.0)
        self.assertLessEqual(metrics.success_rate, 1.0)
        self.assertFalse(torch.equal(before, next(policy.parameters()).detach()))

    def test_calibration_ablation_removes_only_first_two_transitions(self) -> None:
        from prp_wm.rl import RuleGridRLEpisode, remove_calibration_history

        observation = RuleGridRLEpisode(self.tasks[0]).observation
        ablated = remove_calibration_history(observation)
        self.assertEqual(ablated.history, observation.history[2:])
        self.assertEqual(ablated.candidates, observation.candidates)
        self.assertEqual(ablated.available, observation.available)


if __name__ == "__main__":
    unittest.main()
