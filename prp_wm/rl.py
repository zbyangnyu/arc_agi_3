"""From-scratch actor-critic baseline for active RuleGrid identification.

The controller receives only public grids, actions, and observed transitions.
The environment keeps the true program, palette, and version-space computation
behind the reward boundary.  No reconstruction, diagnostic-target, rule-ID, or
oracle-EIG loss is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence

from .rulegrid import (
    ActionKind,
    CompositeAction,
    Direction,
    Grid,
    GridAction,
    RuleGridInferenceView,
    RuleGridProbe,
    RuleGridTask,
    RuleGridTransition,
    behavior_identified,
    version_space,
)

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - Stage 0 remains dependency-free.
    raise ImportError(
        "prp_wm.rl requires PyTorch; install the optional neural dependency"
    ) from error


@dataclass(frozen=True)
class RuleGridRLObservation:
    """The complete controller input; every field is public."""

    history: tuple[RuleGridTransition, ...]
    candidates: tuple[RuleGridProbe, ...]
    available: tuple[bool, ...]

    @classmethod
    def initial(cls, view: RuleGridInferenceView) -> "RuleGridRLObservation":
        return cls(
            history=view.support,
            candidates=view.active_candidates,
            available=(True,) * len(view.active_candidates),
        )


def remove_calibration_history(
    observation: RuleGridRLObservation,
) -> RuleGridRLObservation:
    """Ablation retaining neutral/online transitions but removing calibration."""

    if len(observation.history) < 2:
        raise ValueError("RuleGrid observations require two calibration transitions")
    return RuleGridRLObservation(
        history=observation.history[2:],
        candidates=observation.candidates,
        available=observation.available,
    )


@dataclass(frozen=True)
class RLEpisodeStep:
    observation: RuleGridRLObservation
    reward: float
    done: bool
    identified: bool


class RuleGridRLEpisode:
    """Resettable-probe episode with a sparse simulator-side success reward."""

    def __init__(self, task: RuleGridTask, *, budget: int = 4) -> None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("budget must be a positive integer")
        if len(task.inference.active_candidates) != len(task.privileged.active_targets):
            raise ValueError("each public candidate requires one private target")
        self._task = task
        self._budget = budget
        self._history = list(task.inference.support)
        self._available = [True] * len(task.inference.active_candidates)
        self._actions = 0
        self._done = False
        self._identified = False

    @property
    def observation(self) -> RuleGridRLObservation:
        return RuleGridRLObservation(
            history=tuple(self._history),
            candidates=self._task.inference.active_candidates,
            available=tuple(self._available),
        )

    def step(self, candidate_index: int) -> RLEpisodeStep:
        if self._done:
            raise RuntimeError("cannot step a finished episode")
        if type(candidate_index) is not int:
            raise TypeError("candidate_index must be an integer")
        if not 0 <= candidate_index < len(self._available):
            raise IndexError("candidate_index is outside the candidate bank")
        if not self._available[candidate_index]:
            raise ValueError("candidate has already been used")

        probe = self._task.inference.active_candidates[candidate_index]
        target = self._task.privileged.active_targets[candidate_index]
        self._history.append(RuleGridTransition(probe.state, probe.action, target))
        self._available[candidate_index] = False
        self._actions += 1

        compatible = version_space(self._history, self._task.privileged.palette)
        self._identified = behavior_identified(
            compatible,
            self._task.inference.diagnostics,
            self._task.privileged.palette,
        )
        self._done = (
            self._identified
            or self._actions >= self._budget
            or not any(self._available)
        )
        # Deliberately sparse: no EIG, rule, diagnostic, or shaping reward.
        reward = 1.0 if self._identified else 0.0
        return RLEpisodeStep(self.observation, reward, self._done, self._identified)


@dataclass(frozen=True)
class RLPolicyConfig:
    color_dim: int = 16
    conv_channels: int = 32
    action_dim: int = 32
    token_dim: int = 96
    hidden_dim: int = 128
    candidates: int = 8


@dataclass(frozen=True)
class RLTensorBatch:
    history_states: Tensor
    history_actions: Tensor
    history_targets: Tensor
    history_mask: Tensor
    candidate_states: Tensor
    candidate_actions: Tensor
    available: Tensor

    def to(self, device: torch.device | str) -> "RLTensorBatch":
        return RLTensorBatch(**{name: value.to(device) for name, value in self.__dict__.items()})


def _encode_action(action: GridAction | CompositeAction) -> tuple[int, int, int, int]:
    if isinstance(action, CompositeAction):
        if len(action.actions) != 1:
            raise ValueError("RL baseline currently supports one public action atom")
        action = action.actions[0]
    kind = 0 if action.kind is ActionKind.MOVE else 1
    direction = 4 if action.direction is None else tuple(Direction).index(action.direction)
    return kind, action.coord[0], action.coord[1], direction


def _grid_tensor(grid: Grid) -> Tensor:
    return torch.tensor(grid, dtype=torch.long)


def tensorize_rl_observations(
    observations: Sequence[RuleGridRLObservation],
) -> RLTensorBatch:
    if not observations:
        raise ValueError("at least one observation is required")
    batch_size = len(observations)
    history_steps = max(len(item.history) for item in observations)
    candidates = len(observations[0].candidates)
    if candidates <= 0:
        raise ValueError("candidate bank cannot be empty")
    if any(len(item.candidates) != candidates for item in observations):
        raise ValueError("all observations must have equal candidate counts")

    states = torch.zeros((batch_size, history_steps, 8, 8), dtype=torch.long)
    targets = torch.zeros_like(states)
    actions = torch.zeros((batch_size, history_steps, 4), dtype=torch.long)
    history_mask = torch.zeros((batch_size, history_steps), dtype=torch.bool)
    candidate_states = torch.zeros((batch_size, candidates, 8, 8), dtype=torch.long)
    candidate_actions = torch.zeros((batch_size, candidates, 4), dtype=torch.long)
    available = torch.zeros((batch_size, candidates), dtype=torch.bool)
    for batch_index, observation in enumerate(observations):
        for time_index, transition in enumerate(observation.history):
            states[batch_index, time_index] = _grid_tensor(transition.state)
            targets[batch_index, time_index] = _grid_tensor(transition.next_state)
            actions[batch_index, time_index] = torch.tensor(
                _encode_action(transition.action), dtype=torch.long
            )
            history_mask[batch_index, time_index] = True
        for candidate_index, probe in enumerate(observation.candidates):
            candidate_states[batch_index, candidate_index] = _grid_tensor(probe.state)
            candidate_actions[batch_index, candidate_index] = torch.tensor(
                _encode_action(probe.action), dtype=torch.long
            )
        available[batch_index] = torch.tensor(observation.available, dtype=torch.bool)
    return RLTensorBatch(
        states,
        actions,
        targets,
        history_mask,
        candidate_states,
        candidate_actions,
        available,
    )


class _GridEncoder(nn.Module):
    def __init__(self, config: RLPolicyConfig) -> None:
        super().__init__()
        self.colors = nn.Embedding(16, config.color_dim)
        self.rows = nn.Embedding(8, config.color_dim)
        self.columns = nn.Embedding(8, config.color_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(config.color_dim, config.conv_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.conv_channels, config.conv_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.output = nn.Linear(config.conv_channels * 8 * 8, config.token_dim)

    def forward(self, grids: Tensor) -> Tensor:
        shape = grids.shape[:-2]
        flat = grids.reshape(-1, 8, 8)
        rows = torch.arange(8, device=grids.device)[None, :, None]
        columns = torch.arange(8, device=grids.device)[None, None, :]
        embedded = (
            self.colors(flat)
            + self.rows(rows).expand(flat.shape[0], 8, 8, -1)
            + self.columns(columns).expand(flat.shape[0], 8, 8, -1)
        )
        features = self.conv(embedded.permute(0, 3, 1, 2)).flatten(1)
        return self.output(features).reshape(*shape, -1)


class _ActionEncoder(nn.Module):
    def __init__(self, config: RLPolicyConfig) -> None:
        super().__init__()
        width = 8
        self.kind = nn.Embedding(2, width)
        self.row = nn.Embedding(8, width)
        self.column = nn.Embedding(8, width)
        self.direction = nn.Embedding(5, width)
        self.output = nn.Linear(4 * width, config.action_dim)

    def forward(self, actions: Tensor) -> Tensor:
        return self.output(
            torch.cat(
                (
                    self.kind(actions[..., 0]),
                    self.row(actions[..., 1]),
                    self.column(actions[..., 2]),
                    self.direction(actions[..., 3]),
                ),
                dim=-1,
            )
        )


class RuleGridActorCritic(nn.Module):
    """Recurrent raw-grid policy with no predictive or reconstruction head."""

    def __init__(self, config: RLPolicyConfig = RLPolicyConfig()) -> None:
        super().__init__()
        self.config = config
        self.grid = _GridEncoder(config)
        self.action = _ActionEncoder(config)
        transition_width = 2 * config.token_dim + config.action_dim
        self.transition = nn.Linear(transition_width, config.token_dim)
        self.history = nn.GRUCell(config.token_dim, config.hidden_dim)
        candidate_width = config.token_dim + config.action_dim
        self.candidate = nn.Linear(candidate_width, config.token_dim)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_dim + config.token_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: RLTensorBatch) -> tuple[Tensor, Tensor]:
        state = self.grid(batch.history_states)
        target = self.grid(batch.history_targets)
        action = self.action(batch.history_actions)
        tokens = torch.tanh(self.transition(torch.cat((state, target, action), dim=-1)))
        hidden = torch.zeros(
            (tokens.shape[0], self.config.hidden_dim),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        for index in range(tokens.shape[1]):
            updated = self.history(tokens[:, index], hidden)
            hidden = torch.where(batch.history_mask[:, index, None], updated, hidden)

        candidate = torch.tanh(
            self.candidate(
                torch.cat(
                    (self.grid(batch.candidate_states), self.action(batch.candidate_actions)),
                    dim=-1,
                )
            )
        )
        context = hidden[:, None, :].expand(-1, candidate.shape[1], -1)
        logits = self.score(torch.cat((context, candidate), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch.available, torch.finfo(logits.dtype).min)
        return logits, self.value(hidden).squeeze(-1)


@dataclass(frozen=True)
class RLBatchMetrics:
    loss: float
    actor_loss: float
    value_loss: float
    entropy: float
    success_rate: float
    mean_actions: float


def train_actor_critic_batch(
    policy: RuleGridActorCritic,
    tasks: Sequence[RuleGridTask],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    budget: int = 4,
    gamma: float = 0.97,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 1.0,
) -> RLBatchMetrics:
    """One Monte-Carlo actor-critic update using only sparse success rewards."""

    if not tasks:
        raise ValueError("tasks cannot be empty")
    episodes = [RuleGridRLEpisode(task, budget=budget) for task in tasks]
    trajectories: list[list[tuple[Tensor, Tensor, Tensor, float]]] = [
        [] for _ in episodes
    ]
    active = list(range(len(episodes)))
    successes = [False] * len(episodes)
    actions_used = [0] * len(episodes)
    while active:
        observations = [episodes[index].observation for index in active]
        batch = tensorize_rl_observations(observations).to(device)
        logits, values = policy(batch)
        distribution = torch.distributions.Categorical(logits=logits)
        selected = distribution.sample()
        log_probs = distribution.log_prob(selected)
        entropies = distribution.entropy()
        next_active: list[int] = []
        for local_index, episode_index in enumerate(active):
            step = episodes[episode_index].step(int(selected[local_index].detach().cpu()))
            trajectories[episode_index].append(
                (log_probs[local_index], values[local_index], entropies[local_index], step.reward)
            )
            actions_used[episode_index] += 1
            successes[episode_index] = step.identified
            if not step.done:
                next_active.append(episode_index)
        active = next_active

    all_log_probs: list[Tensor] = []
    all_values: list[Tensor] = []
    all_entropies: list[Tensor] = []
    all_returns: list[float] = []
    for trajectory in trajectories:
        running = 0.0
        returns = [0.0] * len(trajectory)
        for index in range(len(trajectory) - 1, -1, -1):
            running = trajectory[index][3] + gamma * running
            returns[index] = running
        for item, return_value in zip(trajectory, returns, strict=True):
            all_log_probs.append(item[0])
            all_values.append(item[1])
            all_entropies.append(item[2])
            all_returns.append(return_value)

    log_probs_tensor = torch.stack(all_log_probs)
    values_tensor = torch.stack(all_values)
    entropy = torch.stack(all_entropies).mean()
    returns_tensor = torch.tensor(all_returns, dtype=values_tensor.dtype, device=values_tensor.device)
    advantages = returns_tensor - values_tensor
    normalized = advantages.detach()
    if normalized.numel() > 1 and float(normalized.std(unbiased=False)) > 1e-8:
        normalized = (normalized - normalized.mean()) / normalized.std(unbiased=False)
    actor_loss = -(log_probs_tensor * normalized).mean()
    value_loss = F.mse_loss(values_tensor, returns_tensor)
    loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return RLBatchMetrics(
        loss=float(loss.detach().cpu()),
        actor_loss=float(actor_loss.detach().cpu()),
        value_loss=float(value_loss.detach().cpu()),
        entropy=float(entropy.detach().cpu()),
        success_rate=sum(successes) / len(successes),
        mean_actions=sum(actions_used) / len(actions_used),
    )


@dataclass(frozen=True)
class RLEvaluation:
    tasks: int
    success_rate: float
    mean_actions: float


@torch.no_grad()
def evaluate_rl_policy(
    policy: RuleGridActorCritic | None,
    tasks: Sequence[RuleGridTask],
    *,
    device: torch.device | str = "cpu",
    budget: int = 4,
    random_seed: int = 0,
    observation_transform: Callable[[RuleGridRLObservation], RuleGridRLObservation] | None = None,
) -> RLEvaluation:
    """Evaluate greedy policy, or uniform-without-replacement when policy is None."""

    rng = random.Random(random_seed)
    successes = 0
    action_counts = 0
    if policy is not None:
        policy.eval()
    for task in tasks:
        episode = RuleGridRLEpisode(task, budget=budget)
        while True:
            observation = episode.observation
            policy_observation = (
                observation_transform(observation)
                if observation_transform is not None
                else observation
            )
            if policy is None:
                available = [
                    index for index, value in enumerate(policy_observation.available) if value
                ]
                selected = rng.choice(available)
            else:
                logits, _ = policy(
                    tensorize_rl_observations((policy_observation,)).to(device)
                )
                selected = int(logits.argmax(dim=-1).item())
            result = episode.step(selected)
            action_counts += 1
            if result.done:
                successes += int(result.identified)
                break
    return RLEvaluation(
        tasks=len(tasks),
        success_rate=successes / len(tasks),
        mean_actions=action_counts / len(tasks),
    )


__all__ = [
    "RLBatchMetrics",
    "RLEpisodeStep",
    "RLEvaluation",
    "RLPolicyConfig",
    "RLTensorBatch",
    "RuleGridActorCritic",
    "RuleGridRLEpisode",
    "RuleGridRLObservation",
    "evaluate_rl_policy",
    "remove_calibration_history",
    "tensorize_rl_observations",
    "train_actor_critic_batch",
]
