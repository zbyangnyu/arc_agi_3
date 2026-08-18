"""Exact four-rule environment used to validate the Stage 0 protocol.

The hidden, persistent rule is a two-bit vector r. An action is another
two-bit vector a, and the observed state evolves according to

    s' = s XOR <r, a> over GF(2).

The environment is deliberately tiny. It validates belief updates,
information-gain action selection, and evaluation plumbing; it is not meant
to demonstrate that a learned particle world model works.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def _require_bit(value: int, name: str) -> None:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")


@dataclass(frozen=True, order=True)
class Rule:
    """A persistent two-bit rule vector."""

    r0: int
    r1: int

    def __post_init__(self) -> None:
        _require_bit(self.r0, "r0")
        _require_bit(self.r1, "r1")


@dataclass(frozen=True, order=True)
class Action:
    """A two-bit probe vector."""

    a0: int
    a1: int

    def __post_init__(self) -> None:
        _require_bit(self.a0, "a0")
        _require_bit(self.a1, "a1")


ALL_RULES: tuple[Rule, ...] = tuple(
    Rule(r0, r1) for r0 in (0, 1) for r1 in (0, 1)
)
ALL_ACTIONS: tuple[Action, ...] = tuple(
    Action(a0, a1) for a0 in (0, 1) for a1 in (0, 1)
)


def transition(state: int, action: Action, rule: Rule) -> int:
    """Apply one exact environment transition."""

    _require_bit(state, "state")
    dot = (rule.r0 * action.a0 + rule.r1 * action.a1) % 2
    return state ^ dot


def _logsumexp(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("cannot normalize an empty collection")
    maximum = max(values)
    if maximum == -math.inf:
        raise ValueError("observation is impossible under every rule")
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True)
class Belief:
    """A normalized categorical belief over ``ALL_RULES`` in log-space."""

    log_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.log_weights) != len(ALL_RULES):
            raise ValueError(
                f"expected {len(ALL_RULES)} log weights, got {len(self.log_weights)}"
            )
        normalizer = _logsumexp(self.log_weights)
        if not math.isclose(normalizer, 0.0, abs_tol=1e-10):
            raise ValueError("log_weights must already be normalized")

    @classmethod
    def uniform(cls) -> Belief:
        log_weight = -math.log(len(ALL_RULES))
        return cls((log_weight,) * len(ALL_RULES))

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(
            0.0 if log_weight == -math.inf else math.exp(log_weight)
            for log_weight in self.log_weights
        )

    @property
    def support(self) -> tuple[Rule, ...]:
        return tuple(
            rule
            for rule, log_weight in zip(ALL_RULES, self.log_weights, strict=True)
            if log_weight > -math.inf
        )

    @property
    def is_identified(self) -> bool:
        return len(self.support) == 1

    def entropy_bits(self) -> float:
        return -sum(weight * math.log2(weight) for weight in self.weights if weight > 0)


def update_belief(
    belief: Belief,
    state: int,
    action: Action,
    observed_next_state: int,
) -> Belief:
    """Perform an exact deterministic Bayes update in log-space."""

    _require_bit(observed_next_state, "observed_next_state")
    unnormalized = tuple(
        log_weight
        if transition(state, action, rule) == observed_next_state
        else -math.inf
        for rule, log_weight in zip(ALL_RULES, belief.log_weights, strict=True)
    )
    normalizer = _logsumexp(unnormalized)
    return Belief(
        tuple(
            value - normalizer if value > -math.inf else -math.inf
            for value in unnormalized
        )
    )


def predictive_distribution(
    belief: Belief,
    state: int,
    action: Action,
) -> dict[int, float]:
    """Return the exact posterior predictive distribution over the next bit."""

    probabilities = {0: 0.0, 1: 0.0}
    for rule, weight in zip(ALL_RULES, belief.weights, strict=True):
        probabilities[transition(state, action, rule)] += weight
    return probabilities


def expected_information_gain(
    belief: Belief,
    state: int,
    action: Action,
) -> float:
    """Exact expected reduction in rule entropy, measured in bits."""

    prior_entropy = belief.entropy_bits()
    expected_posterior_entropy = 0.0
    for outcome, probability in predictive_distribution(belief, state, action).items():
        if probability == 0.0:
            continue
        posterior = update_belief(belief, state, action, outcome)
        expected_posterior_entropy += probability * posterior.entropy_bits()
    information_gain = prior_entropy - expected_posterior_entropy
    return max(0.0, information_gain)


def select_information_action(
    belief: Belief,
    state: int,
    actions: tuple[Action, ...] = ALL_ACTIONS,
) -> Action:
    """Choose the first maximum-EIG action for deterministic replayability."""

    if not actions:
        raise ValueError("at least one candidate action is required")
    best_action = actions[0]
    best_score = expected_information_gain(belief, state, best_action)
    for action in actions[1:]:
        score = expected_information_gain(belief, state, action)
        if score > best_score:
            best_action = action
            best_score = score
    return best_action


def expected_state_change(
    belief: Belief,
    state: int,
    action: Action,
) -> float:
    """Expected probability that the visible state bit changes."""

    return sum(
        weight
        for rule, weight in zip(ALL_RULES, belief.weights, strict=True)
        if transition(state, action, rule) != state
    )


def select_change_seeking_action(
    belief: Belief,
    state: int,
    actions: tuple[Action, ...] = ALL_ACTIONS,
) -> Action:
    """A deliberately non-epistemic baseline that seeks visible change."""

    if not actions:
        raise ValueError("at least one candidate action is required")
    best_action = actions[0]
    best_score = expected_state_change(belief, state, best_action)
    for action in actions[1:]:
        score = expected_state_change(belief, state, action)
        if score > best_score:
            best_action = action
            best_score = score
    return best_action
