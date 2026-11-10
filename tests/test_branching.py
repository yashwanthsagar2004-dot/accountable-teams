"""Checks of the branching wrapper on Jumanji Level-Based Foraging (skipped if jumanji is not installed)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from accteams.branching import (all_masks, audits_needed, make_branch_fn, measured_responsibility,
                                permutation_audit, run_branches, shapley_from_mask_values)
from accteams.shapley import shapley_exact

NOOP = 0  # Jumanji LBF actions: 0 no-op, 1 up, 2 down, 3 left, 4 right, 5 load


@pytest.fixture(scope="module")
def lbf():
    env = jumanji.make("LevelBasedForaging-v0")
    n = env.num_agents
    policy = lambda obs, k: jax.random.randint(k, (n,), 0, 6)       # uniform random team
    branch = jax.jit(make_branch_fn(env.step, policy, jnp.full((n,), NOOP), horizon=12, gamma=0.99))
    state, _ = env.reset(jax.random.PRNGKey(3))
    return env, n, branch, state


def test_noop_reference_keeps_agents_still(lbf):
    env, n, _, state = lbf
    new_state, _ = env.step(state, jnp.full((n,), NOOP))
    assert np.array_equal(np.asarray(new_state.agents.position), np.asarray(state.agents.position))


def test_branches_are_reproducible_and_vectorised(lbf):
    _, n, branch, state = lbf
    actions = jnp.array([4] * n)
    masks = jnp.asarray(all_masks(n))
    g1 = run_branches(branch, state, actions, masks, jax.random.PRNGKey(0))
    g2 = run_branches(branch, state, actions, masks, jax.random.PRNGKey(0))
    assert g1.shape == (2 ** n,) and np.allclose(g1, g2)


def test_shapley_from_masks_matches_enumeration():
    n = 3
    vals = np.random.default_rng(0).normal(size=2 ** n)
    v = lambda s: vals[sum(1 << i for i in s)]
    assert np.allclose(shapley_from_mask_values(vals, n), shapley_exact(v, n))


def adjacent_state(state, food_level):
    """Both agents next to food item 0. Agent levels are 2 and 1: a level-2 item needs only agent 0,
    a level-3 item needs both."""
    food = state.food_items.replace(level=state.food_items.level.at[0].set(food_level))
    agents = state.agents.replace(position=jnp.array([[2, 3], [1, 2]]))
    return state.replace(agents=agents, food_items=food)


@pytest.mark.parametrize("food_level", [2, 3])
def test_measured_responsibility_and_audits_on_lbf(lbf, food_level):
    _, n, branch, state = lbf
    if n != 2:
        pytest.skip("constructed state assumes the default two-agent LBF")
    s = adjacent_state(state, food_level)
    actions = jnp.array([5] * n)                                      # both agents load
    phi, se = measured_responsibility(branch, s, actions, n, reps=400, key=jax.random.PRNGKey(1))
    assert np.all(se < 0.02)
    if food_level == 2:      # agent 0 can collect alone: it carries the responsibility
        assert phi[0] > 0.15 and abs(phi[1]) < 0.05
    else:                    # both are needed: responsibility is shared
        assert phi.min() > 0.15 and abs(phi[0] - phi[1]) < 0.05
    for i in range(n):
        est, hw = permutation_audit(branch, s, actions, n, i, m=400, key=jax.random.PRNGKey(10 + i),
                                    outcome_range=1.0)
        assert abs(est - phi[i]) <= hw + 3 * se[i]


def test_audit_budget_formula():
    assert audits_needed(1.0, 0.1, 4) == int(np.ceil(2 * np.log(2 * 4 / 0.05) / 0.01))
