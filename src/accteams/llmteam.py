"""Replayable role-based agent team for testing failure attribution (the WP6 testbed core).

A team of three roles solves a checkable task: a planner turns the task into steps, a solver executes them, and a
verifier checks the solver and may send the work back once. Every random draw of agent i at turn t comes from its
own generator seeded by (episode seed, agent, turn), so an episode replays exactly and any agent can be replaced by
its declared reference (a certified, error-free version of the role) while the others keep their draws: common
random numbers by construction.

The agents here are scripted with error rates (planted faults, or a uniformly weaker "model" for natural failures).
`LLMRole` is the slot for a real model served at temperature 0 with a fixed seed; it is not exercised in tests.

What the module measures for a failed episode:
  * responsibility: Shapley values of the interventional game v(S) = success when only the agents in S act as
    recorded and the rest take their reference, computed exactly (3 roles = 8 coalitions);
  * the earliest decisive error: the first step whose single correction turns failure into success (the convention
    of Who&When, AgenTracer and Aegis);
  * repair: rerun with one chosen agent replaced by its reference and check success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .branching import all_masks, shapley_from_mask_values

ROLES = ("planner", "solver", "verifier")
OPS = ("add", "sub", "mul")


@dataclass(frozen=True)
class Task:
    start: int
    steps: tuple                     # ((op, value), ...)

    def answer(self) -> int:
        return run_plan(self.start, self.steps)


def run_plan(x, steps):
    for op, v in steps:
        x = x + v if op == "add" else x - v if op == "sub" else x * v
    return x


def make_task(rng, n_steps=5) -> Task:
    steps = tuple((OPS[rng.integers(3)], int(rng.integers(2, 10))) for _ in range(n_steps))
    return Task(int(rng.integers(1, 20)), steps)


@dataclass
class Rates:
    """Per-role error probabilities. The reference of every role has all rates 0."""
    plan_drop: float = 0.0           # planner drops one step
    plan_swap: float = 0.0           # planner swaps two adjacent steps
    solve_slip: float = 0.0          # solver makes an arithmetic slip on a step (off by +-1..3)
    verify_miss: float = 0.0         # verifier accepts a wrong result
    verify_false_alarm: float = 0.0  # verifier rejects a right result and the redo slips


def _rng(seed, agent, turn):
    return np.random.default_rng([int(seed), ROLES.index(agent), int(turn)])


def planner(task, rates, seed):
    r = _rng(seed, "planner", 0)
    steps = list(task.steps)
    if len(steps) > 1 and r.random() < rates.plan_drop:
        steps.pop(int(r.integers(len(steps))))
    if len(steps) > 1 and r.random() < rates.plan_swap:
        i = int(r.integers(len(steps) - 1))
        steps[i], steps[i + 1] = steps[i + 1], steps[i]
    return tuple(steps)


def solver(task, plan, rates, seed, turn):
    r = _rng(seed, "solver", turn)
    x, trace = task.start, []
    for op, v in plan:
        x = run_plan(x, [(op, v)])
        if r.random() < rates.solve_slip:
            x += int(r.choice([-3, -2, -1, 1, 2, 3]))
        trace.append(x)
    return x, tuple(trace)


def verifier(task, plan, result, rates, seed):
    """Recomputes the plan (it trusts the plan, as a role-separated verifier would). Returns accept?"""
    r = _rng(seed, "verifier", 0)
    correct = result == run_plan(task.start, plan)
    if correct:
        return not (r.random() < rates.verify_false_alarm)
    return r.random() < rates.verify_miss


@dataclass
class Episode:
    task: Task
    seed: int
    plan: tuple
    result: int
    accepted_first: bool
    success: bool
    steps: list = field(default_factory=list)   # chronological (agent, turn) decisions


def run_episode(task, rates_by_role, seed, reference=frozenset(), correct=frozenset()):
    """Run the team. Roles in `reference` use the error-free reference policy for the whole episode; decisions
    (agent, turn) in `correct` are replaced by the reference output for that single decision only."""
    ref = Rates()
    rate = lambda role, turn=0: ref if (role in reference or (role, turn) in correct) else rates_by_role[role]
    plan = planner(task, rate("planner"), seed)
    result, _ = solver(task, plan, rate("solver", 0), seed, 0)
    accepted = verifier(task, plan, result, rate("verifier"), seed)
    steps = [("planner", 0), ("solver", 0), ("verifier", 0)]
    final = result
    if not accepted:                                  # one redo
        final, _ = solver(task, plan, rate("solver", 1), seed, 1)
        steps.append(("solver", 1))
    return Episode(task, seed, plan, result, accepted, final == task.answer(), steps)


def responsibility(task, rates_by_role, seed):
    """Shapley values of v(S) = success with agents outside S on their reference. Negative = blame for failure."""
    n = len(ROLES)
    vals = []
    for mask in all_masks(n):
        ref = frozenset(ROLES[i] for i in range(n) if mask[i] == 0)
        vals.append(float(run_episode(task, rates_by_role, seed, reference=ref).success))
    return shapley_from_mask_values(np.array(vals), n)


def earliest_decisive_error(task, rates_by_role, seed):
    """First decision (in time order) whose single correction turns the failure into success; None if none does."""
    ep = run_episode(task, rates_by_role, seed)
    if ep.success:
        return None
    for step in ep.steps:
        if run_episode(task, rates_by_role, seed, correct=frozenset([step])).success:
            return step
    return None


def repair_success(task, rates_by_role, seed, role):
    """Rerun the same episode with `role` replaced by its reference."""
    return run_episode(task, rates_by_role, seed, reference=frozenset([role])).success


class LLMRole:
    """Slot for a real model. `call(prompt) -> str` must be deterministic (temperature 0, fixed seed); replay
    fidelity is then measured by running the same episode twice and comparing transcripts. Not used in tests."""

    def __init__(self, role: str, call):
        assert role in ROLES
        self.role, self.call = role, call


def replay_fidelity(task, rates_by_role, seeds):
    """Fraction of episodes whose outcome and trace are identical when rerun (1.0 for the scripted team)."""
    same = 0
    for s in seeds:
        a, b = run_episode(task, rates_by_role, s), run_episode(task, rates_by_role, s)
        same += (a.plan, a.result, a.success, a.steps) == (b.plan, b.result, b.success, b.steps)
    return same / len(seeds)


__all__ = ["ROLES", "Task", "Rates", "make_task", "run_episode", "responsibility", "earliest_decisive_error",
           "repair_success", "replay_fidelity", "LLMRole"]
