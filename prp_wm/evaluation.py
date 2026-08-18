"""Paired Stage 0 evaluation for exact and heuristic exploration policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import statistics
from typing import Literal

from .gf2 import (
    ALL_ACTIONS,
    ALL_RULES,
    Action,
    Belief,
    Rule,
    select_change_seeking_action,
    transition,
    update_belief,
)
from .schema import InferenceView, Transition, choose_information_action


Strategy = Literal["oracle_eig", "uniform", "coverage", "change_seeking"]
STRATEGIES: tuple[Strategy, ...] = (
    "oracle_eig",
    "uniform",
    "coverage",
    "change_seeking",
)


@dataclass(frozen=True)
class EpisodeResult:
    strategy: Strategy
    steps: int
    identified: bool
    history: tuple[Transition, ...]
    final_belief: Belief


def _choose_action(
    strategy: Strategy,
    belief: Belief,
    view: InferenceView,
    rng: random.Random,
) -> Action:
    if strategy == "oracle_eig":
        return choose_information_action(view, belief)
    if strategy == "uniform":
        return rng.choice(view.candidate_actions)
    if strategy == "coverage":
        tried = {transition_item.action for transition_item in view.support}
        untried = tuple(
            action for action in view.candidate_actions if action not in tried
        )
        return rng.choice(untried or view.candidate_actions)
    if strategy == "change_seeking":
        return select_change_seeking_action(
            belief,
            view.current_state,
            view.candidate_actions,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def run_episode(
    rule: Rule,
    strategy: Strategy,
    *,
    initial_state: int = 0,
    budget: int = 12,
    rng: random.Random | None = None,
) -> EpisodeResult:
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be non-negative")
    if type(initial_state) is not int or initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 or 1")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    rng = rng or random.Random(0)
    state = initial_state
    belief = Belief.uniform()
    history: list[Transition] = []

    while len(history) < budget and not belief.is_identified:
        view = InferenceView(
            support=tuple(history),
            current_state=state,
            candidate_actions=ALL_ACTIONS,
        )
        action = _choose_action(strategy, belief, view, rng)
        next_state = transition(state, action, rule)
        next_belief = update_belief(belief, state, action, next_state)
        history.append(Transition(state, action, next_state))
        state = next_state
        belief = next_belief

    return EpisodeResult(
        strategy=strategy,
        steps=len(history),
        identified=belief.is_identified,
        history=tuple(history),
        final_belief=belief,
    )


@dataclass(frozen=True)
class StrategySummary:
    restricted_mean_steps: float
    median_trial_mean_steps: float
    identification_rate: float
    rollouts: int


@dataclass(frozen=True)
class Gate0Report:
    trials: int
    repeats: int
    budget: int
    bootstrap_resamples: int
    seed: int
    oracle_eig: StrategySummary
    uniform: StrategySummary
    coverage: StrategySummary
    change_seeking: StrategySummary
    exact_uniform_restricted_mean_steps: float
    exact_uniform_identification_rate: float
    exact_uniform_relative_step_reduction: float
    uniform_relative_step_reduction: float
    uniform_minus_oracle_mean_ci95: tuple[float, float]
    gate_threshold: float
    gate_eligible: bool
    passes: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stratified_bootstrap_mean_ci95(
    paired_differences_by_stratum: dict[tuple[Rule, int], list[float]],
    *,
    resamples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not paired_differences_by_stratum:
        raise ValueError("at least one bootstrap stratum is required")
    count = sum(len(values) for values in paired_differences_by_stratum.values())
    estimates = []
    for _ in range(resamples):
        total = 0.0
        for values in paired_differences_by_stratum.values():
            if not values:
                raise ValueError("bootstrap strata cannot be empty")
            total += sum(values[rng.randrange(len(values))] for _ in range(len(values)))
        estimates.append(total / count)
    estimates.sort()
    lower_index = max(0, int(0.025 * resamples) - 1)
    upper_index = min(resamples - 1, int(0.975 * resamples))
    return estimates[lower_index], estimates[upper_index]


def _summarize(
    task_scores: list[float],
    raw_identified: list[bool],
) -> StrategySummary:
    return StrategySummary(
        restricted_mean_steps=statistics.fmean(task_scores),
        median_trial_mean_steps=statistics.median(task_scores),
        identification_rate=statistics.fmean(float(value) for value in raw_identified),
        rollouts=len(raw_identified),
    )


def exact_uniform_statistics(budget: int) -> tuple[float, float]:
    """Return exact RMST and identification rate for uniform random probes."""

    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    survival = lambda steps: 3.0 / (2**steps) - 2.0 / (4**steps)
    restricted_mean = sum(survival(step) for step in range(budget))
    identification_rate = 1.0 - survival(budget)
    return restricted_mean, identification_rate


def evaluate_gate0(
    *,
    trials: int = 1_000,
    repeats: int = 8,
    budget: int = 12,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
    gate_threshold: float = 0.25,
) -> Gate0Report:
    """Evaluate exact EIG against paired baselines on balanced hidden rules.

    Unidentified episodes are right-censored at ``budget`` for the restricted
    mean step score, and their failure is retained in ``identification_rate``.
    """

    if type(trials) is not int or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be positive")
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be positive")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if not 0.0 < gate_threshold < 1.0:
        raise ValueError("gate_threshold must lie strictly between 0 and 1")

    task_specs = [
        (ALL_RULES[index % len(ALL_RULES)], (index // len(ALL_RULES)) % 2)
        for index in range(trials)
    ]

    task_scores: dict[Strategy, list[float]] = {name: [] for name in STRATEGIES}
    raw_identified: dict[Strategy, list[bool]] = {name: [] for name in STRATEGIES}

    for trial_index, (rule, initial_state) in enumerate(task_specs):
        for strategy in STRATEGIES:
            strategy_repeats = 1 if strategy in ("oracle_eig", "change_seeking") else repeats
            episode_scores = []
            for repeat_index in range(strategy_repeats):
                episode_rng = random.Random(
                    f"prp-wm:{seed}:{trial_index}:{strategy}:{repeat_index}"
                )
                result = run_episode(
                    rule,
                    strategy,
                    initial_state=initial_state,
                    budget=budget,
                    rng=episode_rng,
                )
                episode_scores.append(float(result.steps))
                raw_identified[strategy].append(result.identified)
            task_scores[strategy].append(statistics.fmean(episode_scores))

    summaries = {
        strategy: _summarize(task_scores[strategy], raw_identified[strategy])
        for strategy in STRATEGIES
    }
    oracle_mean = summaries["oracle_eig"].restricted_mean_steps
    uniform_mean = summaries["uniform"].restricted_mean_steps
    relative_reduction = (uniform_mean - oracle_mean) / uniform_mean
    paired_differences = [
        uniform_score - oracle_score
        for uniform_score, oracle_score in zip(
            task_scores["uniform"], task_scores["oracle_eig"], strict=True
        )
    ]
    paired_by_stratum: dict[tuple[Rule, int], list[float]] = {}
    for task_spec, difference in zip(task_specs, paired_differences, strict=True):
        paired_by_stratum.setdefault(task_spec, []).append(difference)
    ci95 = _stratified_bootstrap_mean_ci95(
        paired_by_stratum,
        resamples=bootstrap_resamples,
        rng=random.Random(seed ^ 0xB0057),
    )
    exact_uniform_mean, exact_uniform_rate = exact_uniform_statistics(budget)
    exact_relative_reduction = (exact_uniform_mean - oracle_mean) / exact_uniform_mean
    gate_eligible = (
        trials >= 500
        and trials % (len(ALL_RULES) * 2) == 0
        and repeats >= 4
        and bootstrap_resamples >= 1_000
        and budget >= 2
    )
    passes = (
        gate_eligible
        and exact_relative_reduction >= gate_threshold
        and relative_reduction >= gate_threshold
        and ci95[0] > 0.0
    )

    return Gate0Report(
        trials=trials,
        repeats=repeats,
        budget=budget,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        oracle_eig=summaries["oracle_eig"],
        uniform=summaries["uniform"],
        coverage=summaries["coverage"],
        change_seeking=summaries["change_seeking"],
        exact_uniform_restricted_mean_steps=exact_uniform_mean,
        exact_uniform_identification_rate=exact_uniform_rate,
        exact_uniform_relative_step_reduction=exact_relative_reduction,
        uniform_relative_step_reduction=relative_reduction,
        uniform_minus_oracle_mean_ci95=ci95,
        gate_threshold=gate_threshold,
        gate_eligible=gate_eligible,
        passes=passes,
    )
