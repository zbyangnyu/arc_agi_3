"""Recurrent PPO and GRPO baselines for the terminal-reward RuleGame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .rulegame import HORIZON, NUM_ACTIONS, PROBE, RuleGame, RuleGameSpec

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover
    raise ImportError("prp_wm.rulegame_rl requires PyTorch") from error


PREVIOUS_ACTION_SENTINEL = NUM_ACTIONS


@dataclass(frozen=True)
class RuleGamePolicyConfig:
    color_dim: int = 16
    conv_channels: int = 32
    grid_dim: int = 96
    action_dim: int = 16
    hidden_dim: int = 128


class RuleGamePolicy(nn.Module):
    def __init__(self, config: RuleGamePolicyConfig = RuleGamePolicyConfig()) -> None:
        super().__init__()
        self.config = config
        self.colors = nn.Embedding(16, config.color_dim)
        self.rows = nn.Embedding(8, config.color_dim)
        self.columns = nn.Embedding(8, config.color_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(config.color_dim, config.conv_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.conv_channels, config.conv_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.grid_output = nn.Linear(config.conv_channels * 64, config.grid_dim)
        self.previous_action = nn.Embedding(NUM_ACTIONS + 1, config.action_dim)
        self.recurrent = nn.GRUCell(config.grid_dim + config.action_dim, config.hidden_dim)
        self.actor = nn.Linear(config.hidden_dim, NUM_ACTIONS)
        self.critic = nn.Linear(config.hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros((batch_size, self.config.hidden_dim), device=device)

    def encode_grid(self, grids: Tensor) -> Tensor:
        flat = grids.reshape(-1, 8, 8)
        rows = torch.arange(8, device=grids.device)[None, :, None]
        columns = torch.arange(8, device=grids.device)[None, None, :]
        embedded = (
            self.colors(flat)
            + self.rows(rows).expand(flat.shape[0], 8, 8, -1)
            + self.columns(columns).expand(flat.shape[0], 8, 8, -1)
        )
        features = self.conv(embedded.permute(0, 3, 1, 2)).flatten(1)
        return torch.tanh(self.grid_output(features)).reshape(*grids.shape[:-2], -1)

    def step(
        self,
        grids: Tensor,
        previous_actions: Tensor,
        legal_actions: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        token = torch.cat(
            (self.encode_grid(grids), self.previous_action(previous_actions)), dim=-1
        )
        hidden = self.recurrent(token, hidden)
        logits = self.actor(hidden).masked_fill(
            ~legal_actions, torch.finfo(hidden.dtype).min
        )
        return logits, self.critic(hidden).squeeze(-1), hidden


@dataclass(frozen=True)
class RuleGameRollout:
    grids: Tensor
    legal_actions: Tensor
    previous_actions: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    rewards: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.grids.shape[0])

    @property
    def terminal_rewards(self) -> Tensor:
        return self.rewards[:, -1]


def _observations_to_tensors(
    games: Sequence[RuleGame], device: torch.device | str
) -> tuple[Tensor, Tensor]:
    grids = torch.tensor(
        [[list(row) for row in game.observation.grid] for game in games],
        dtype=torch.long,
        device=device,
    )
    legal = torch.tensor(
        [game.observation.legal_actions for game in games],
        dtype=torch.bool,
        device=device,
    )
    return grids, legal


@torch.no_grad()
def collect_rulegame_rollout(
    policy: RuleGamePolicy,
    specs: Sequence[RuleGameSpec],
    *,
    device: torch.device | str = "cpu",
    stochastic: bool = True,
    reset_memory_at_decision: bool = False,
) -> RuleGameRollout:
    if not specs:
        raise ValueError("specs cannot be empty")
    games = [RuleGame(spec) for spec in specs]
    hidden = policy.initial_hidden(len(games), device)
    previous = torch.full(
        (len(games),), PREVIOUS_ACTION_SENTINEL, dtype=torch.long, device=device
    )
    grids_list: list[Tensor] = []
    legal_list: list[Tensor] = []
    previous_list: list[Tensor] = []
    actions_list: list[Tensor] = []
    log_probs_list: list[Tensor] = []
    values_list: list[Tensor] = []
    rewards_list: list[Tensor] = []
    for time_index in range(HORIZON):
        if reset_memory_at_decision and time_index == HORIZON - 1:
            hidden = policy.initial_hidden(len(games), device)
            previous = torch.full_like(previous, PREVIOUS_ACTION_SENTINEL)
        grids, legal = _observations_to_tensors(games, device)
        logits, values, hidden = policy.step(grids, previous, legal, hidden)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample() if stochastic else logits.argmax(dim=-1)
        rewards = []
        for game, action in zip(games, actions.detach().cpu().tolist(), strict=True):
            rewards.append(game.step(action).reward)
        grids_list.append(grids)
        legal_list.append(legal)
        previous_list.append(previous)
        actions_list.append(actions)
        log_probs_list.append(distribution.log_prob(actions))
        values_list.append(values)
        rewards_list.append(torch.tensor(rewards, dtype=values.dtype, device=device))
        previous = actions
    return RuleGameRollout(
        grids=torch.stack(grids_list, dim=1),
        legal_actions=torch.stack(legal_list, dim=1),
        previous_actions=torch.stack(previous_list, dim=1),
        actions=torch.stack(actions_list, dim=1),
        old_log_probs=torch.stack(log_probs_list, dim=1),
        old_values=torch.stack(values_list, dim=1),
        rewards=torch.stack(rewards_list, dim=1),
    )


def recompute_rollout(
    policy: RuleGamePolicy, rollout: RuleGameRollout
) -> tuple[Tensor, Tensor, Tensor]:
    hidden = policy.initial_hidden(rollout.batch_size, rollout.grids.device)
    log_probs: list[Tensor] = []
    values: list[Tensor] = []
    entropies: list[Tensor] = []
    for time_index in range(HORIZON):
        logits, value, hidden = policy.step(
            rollout.grids[:, time_index],
            rollout.previous_actions[:, time_index],
            rollout.legal_actions[:, time_index],
            hidden,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs.append(distribution.log_prob(rollout.actions[:, time_index]))
        values.append(value)
        entropies.append(distribution.entropy())
    return (
        torch.stack(log_probs, dim=1),
        torch.stack(values, dim=1),
        torch.stack(entropies, dim=1),
    )


def discounted_terminal_returns(rollout: RuleGameRollout, gamma: float) -> Tensor:
    factors = torch.tensor(
        [gamma ** (HORIZON - 1 - index) for index in range(HORIZON)],
        dtype=rollout.rewards.dtype,
        device=rollout.rewards.device,
    )
    return rollout.terminal_rewards[:, None] * factors[None, :]


@dataclass(frozen=True)
class RuleGameTrainMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    win_rate: float
    probe_rate: float


def train_ppo_update(
    policy: RuleGamePolicy,
    optimizer: torch.optim.Optimizer,
    specs: Sequence[RuleGameSpec],
    *,
    device: torch.device | str = "cpu",
    gamma: float = 0.99,
    clip: float = 0.2,
    epochs: int = 4,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> RuleGameTrainMetrics:
    policy.eval()
    rollout = collect_rulegame_rollout(policy, specs, device=device, stochastic=True)
    returns = discounted_terminal_returns(rollout, gamma)
    advantages = returns - rollout.old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    latest = (torch.tensor(0.0),) * 4
    policy.train()
    for _ in range(epochs):
        log_probs, values, entropies = recompute_rollout(policy, rollout)
        ratio = torch.exp(log_probs - rollout.old_log_probs)
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = entropies.mean()
        loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        latest = loss, policy_loss, value_loss, entropy
    return RuleGameTrainMetrics(
        loss=float(latest[0].detach().cpu()),
        policy_loss=float(latest[1].detach().cpu()),
        value_loss=float(latest[2].detach().cpu()),
        entropy=float(latest[3].detach().cpu()),
        win_rate=float(rollout.terminal_rewards.mean().cpu()),
        probe_rate=float((rollout.actions[:, 0] == PROBE).float().mean().cpu()),
    )


def train_grpo_update(
    policy: RuleGamePolicy,
    optimizer: torch.optim.Optimizer,
    base_specs: Sequence[RuleGameSpec],
    *,
    group_size: int = 8,
    device: torch.device | str = "cpu",
    clip: float = 0.2,
    epochs: int = 4,
    entropy_coefficient: float = 0.01,
) -> RuleGameTrainMetrics:
    if group_size < 2:
        raise ValueError("GRPO group_size must be at least two")
    expanded = tuple(spec for spec in base_specs for _ in range(group_size))
    policy.eval()
    rollout = collect_rulegame_rollout(policy, expanded, device=device, stochastic=True)
    scores = rollout.terminal_rewards.reshape(len(base_specs), group_size)
    group_mean = scores.mean(dim=1, keepdim=True)
    group_std = scores.std(dim=1, unbiased=False, keepdim=True)
    group_advantage = torch.where(
        group_std > 1e-8,
        (scores - group_mean) / (group_std + 1e-8),
        torch.zeros_like(scores),
    ).reshape(-1)
    advantages = group_advantage[:, None].expand(-1, HORIZON)
    latest = (torch.tensor(0.0),) * 3
    policy.train()
    for _ in range(epochs):
        log_probs, _, entropies = recompute_rollout(policy, rollout)
        ratio = torch.exp(log_probs - rollout.old_log_probs)
        policy_loss = -torch.minimum(
            ratio * advantages,
            ratio.clamp(1.0 - clip, 1.0 + clip) * advantages,
        ).mean()
        entropy = entropies.mean()
        loss = policy_loss - entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        latest = loss, policy_loss, entropy
    return RuleGameTrainMetrics(
        loss=float(latest[0].detach().cpu()),
        policy_loss=float(latest[1].detach().cpu()),
        value_loss=0.0,
        entropy=float(latest[2].detach().cpu()),
        win_rate=float(rollout.terminal_rewards.mean().cpu()),
        probe_rate=float((rollout.actions[:, 0] == PROBE).float().mean().cpu()),
    )


@dataclass(frozen=True)
class RuleGameEvaluation:
    tasks: int
    win_rate: float
    probe_rate: float


@torch.no_grad()
def evaluate_rulegame_policy(
    policy: RuleGamePolicy,
    specs: Sequence[RuleGameSpec],
    *,
    device: torch.device | str = "cpu",
    reset_memory_at_decision: bool = False,
) -> RuleGameEvaluation:
    policy.eval()
    rollout = collect_rulegame_rollout(
        policy,
        specs,
        device=device,
        stochastic=False,
        reset_memory_at_decision=reset_memory_at_decision,
    )
    return RuleGameEvaluation(
        tasks=len(specs),
        win_rate=float(rollout.terminal_rewards.mean().cpu()),
        probe_rate=float((rollout.actions[:, 0] == PROBE).float().mean().cpu()),
    )


__all__ = [
    "RuleGameEvaluation",
    "RuleGamePolicy",
    "RuleGamePolicyConfig",
    "RuleGameRollout",
    "RuleGameTrainMetrics",
    "collect_rulegame_rollout",
    "discounted_terminal_returns",
    "evaluate_rulegame_policy",
    "recompute_rollout",
    "train_grpo_update",
    "train_ppo_update",
]
