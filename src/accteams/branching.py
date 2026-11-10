"""Executed coalition branches for JAX environments with a functional step (Jumanji, JaxMARL-style).

A branch starts from a saved decision state. Agents in the coalition S keep their realised actions, the others take
the declared reference action, and then every agent follows the policy for `horizon - 1` more steps. The discounted
team return (plus an optional bootstrap from a value function) is one noisy draw of v(S).

Because JAX environment states are immutable pytrees, "copying the simulator" is free: the same state is passed to
every branch, and jax.vmap runs the branches in parallel.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


def make_branch_fn(env_step: Callable, policy: Callable, reference_action, horizon: int = 5, gamma: float = 0.99,
                   bootstrap: Optional[Callable] = None, reseed: Optional[Callable] = None, window: int = 1):
    """Build branch(state, actions, mask, key) -> discounted team return of one executed branch.

    env_step(state, actions) -> (state, timestep), with timestep.reward of shape (n,), timestep.observation and
        timestep.last() as in Jumanji.
    policy(observation, key) -> actions of shape (n,).
    reference_action: array of shape (n,), e.g. the no-op action.
    bootstrap(observation) -> scalar value estimate used after the last step (truncated branches), or None.
    reseed(state, key) -> state with fresh environment randomness, so branches are independent draws, or None.
    window: number of steps for which absent agents keep taking the reference (1 = a single-decision
        intervention). A longer window defines responsibility for a stretch of behaviour; the game v(S) changes,
        the theory does not. Use it when one decision alone moves the return too little to measure.
    """
    reference_action = jnp.asarray(reference_action)

    def branch(state, actions, mask, key):
        if reseed is not None:
            key, sub = jax.random.split(key)
            state = reseed(state, sub)
        first = jnp.where(mask.astype(bool), actions, reference_action)
        state, ts = env_step(state, first)
        ret = ts.reward.sum()
        weight = gamma * (1.0 - ts.last().astype(jnp.float32))

        def body(carry, inp):
            state, ts, ret, weight = carry
            k, t = inp
            a = policy(ts.observation, k)
            a = jnp.where((t < window) & ~mask.astype(bool), reference_action, a)
            state, ts = env_step(state, a)
            ret = ret + weight * ts.reward.sum()
            weight = weight * gamma * (1.0 - ts.last().astype(jnp.float32))
            return (state, ts, ret, weight), None

        (state, ts, ret, weight), _ = jax.lax.scan(body, (state, ts, ret, weight),
                                                   (jax.random.split(key, horizon - 1), jnp.arange(1, horizon)))
        if bootstrap is not None:
            ret = ret + weight * bootstrap(ts.observation)
        return ret

    return branch


def run_branches(branch, state, actions, masks, key):
    """Run len(masks) independent branches from one decision state (vectorised)."""
    keys = jax.random.split(key, masks.shape[0])
    return jax.vmap(branch, in_axes=(None, None, 0, 0))(state, actions, masks, keys)


def all_masks(n: int) -> np.ndarray:
    """All 2^n coalition masks; row b has agent i present iff bit i of b is set."""
    return np.array([[(b >> i) & 1 for i in range(n)] for b in range(2 ** n)], dtype=np.float32)


def shapley_from_mask_values(values: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley values from v(S) indexed as in all_masks (values has length 2^n)."""
    phi = np.zeros(n)
    for i in range(n):
        for b in range(2 ** n):
            if (b >> i) & 1:
                continue
            s = bin(b).count("1")
            w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            phi[i] += w * (values[b | (1 << i)] - values[b])
    return phi


def measured_responsibility(branch, state, actions, n: int, reps: int, key):
    """Tier-2 ground truth: every coalition, `reps` branches each. Returns (phi, standard error of phi)."""
    masks = jnp.asarray(np.repeat(all_masks(n), reps, axis=0))
    g = np.asarray(run_branches(branch, state, actions, masks, key)).reshape(2 ** n, reps)
    phi = shapley_from_mask_values(g.mean(1), n)
    # standard error by the delta method: phi is linear in the coalition means
    var_mean = g.var(1, ddof=1) / reps
    se = np.zeros(n)
    for i in range(n):
        coef = np.zeros(2 ** n)
        for b in range(2 ** n):
            if not (b >> i) & 1:
                s = bin(b).count("1")
                w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                coef[b | (1 << i)] += w
                coef[b] -= w
        se[i] = math.sqrt(float(coef ** 2 @ var_mean))
    return phi, se


def permutation_audit(branch, state, actions, n: int, agent: int, m: int, key, outcome_range: float,
                      eta: float = 0.05):
    """Unbiased audit of one agent's responsibility from m random permutations (2m branches).

    Returns (estimate, half_width): with probability at least 1 - eta the true value lies within half_width,
    by Hoeffding's inequality with marginal contributions in [-R, R] and a union bound over n agents.
    """
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))
    before = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        perm = rng.permutation(n)
        pos = int(np.where(perm == agent)[0][0])
        before[row, perm[:pos]] = 1.0
    with_agent = before.copy()
    with_agent[:, agent] = 1.0
    masks = jnp.asarray(np.concatenate([with_agent, before]))
    g = np.asarray(run_branches(branch, state, actions, masks, key))
    est = float(np.mean(g[:m] - g[m:]))
    half_width = outcome_range * math.sqrt(2 * math.log(2 * n / eta) / m)
    return est, half_width


def audits_needed(outcome_range: float, eps: float, n: int, eta: float = 0.05) -> int:
    """Permutations per agent so that all n audits are within eps with probability >= 1 - eta."""
    return math.ceil(2 * outcome_range ** 2 * math.log(2 * n / eta) / eps ** 2)
