"""Data contracts that keep inference inputs separate from privileged labels."""

from __future__ import annotations

from dataclasses import dataclass

from .gf2 import (
    ALL_ACTIONS,
    ALL_RULES,
    Action,
    Belief,
    Rule,
    select_information_action,
    transition,
)


@dataclass(frozen=True)
class Transition:
    state: int
    action: Action
    next_state: int


@dataclass(frozen=True)
class InferenceView:
    """The only task data an inference model may consume."""

    support: tuple[Transition, ...]
    current_state: int
    candidate_actions: tuple[Action, ...] = ALL_ACTIONS


@dataclass(frozen=True)
class PrivilegedTargets:
    """Simulator-only labels for training loss and evaluation."""

    true_rule: Rule
    counterfactual_next_by_action_and_rule: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TaskBundle:
    inference: InferenceView
    privileged: PrivilegedTargets


def choose_information_action(view: InferenceView, belief: Belief) -> Action:
    """Controller boundary: selection can consume only the inference view."""

    return select_information_action(
        belief,
        state=view.current_state,
        actions=view.candidate_actions,
    )


def make_task_bundle(
    true_rule: Rule,
    current_state: int = 0,
    support: tuple[Transition, ...] = (),
) -> TaskBundle:
    if type(current_state) is not int or current_state not in (0, 1):
        raise ValueError("current_state must be 0 or 1")
    previous_next_state: int | None = None
    for index, item in enumerate(support):
        if type(item.state) is not int or item.state not in (0, 1):
            raise ValueError(f"support[{index}].state must be 0 or 1")
        if type(item.next_state) is not int or item.next_state not in (0, 1):
            raise ValueError(f"support[{index}].next_state must be 0 or 1")
        if previous_next_state is not None and item.state != previous_next_state:
            raise ValueError(f"support is discontinuous at index {index}")
        if transition(item.state, item.action, true_rule) != item.next_state:
            raise ValueError(f"support[{index}] is inconsistent with true_rule")
        previous_next_state = item.next_state
    if support and support[-1].next_state != current_state:
        raise ValueError("current_state must equal the final support next_state")

    counterfactuals = tuple(
        tuple(transition(current_state, action, rule) for rule in ALL_RULES)
        for action in ALL_ACTIONS
    )
    return TaskBundle(
        inference=InferenceView(support=support, current_state=current_state),
        privileged=PrivilegedTargets(
            true_rule=true_rule,
            counterfactual_next_by_action_and_rule=counterfactuals,
        ),
    )
