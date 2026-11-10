"""Tier-2 environments with scripted teams, behind one interface.

get_env(name) returns an EnvSpec with
  step(state, actions) -> (state, timestep)   timestep.observation is what `policy` reads
  reset(key) -> (state, timestep)
  policy(observation, key) -> actions (n,)    the scripted team, with EPS random actions
  features(state) -> (d,) float32             decision features for the critic
  live(states) -> bool mask                   decision points worth measuring
  n, num_actions, reference_action, horizon, collect_steps

lbf   : Jumanji Level-Based Foraging, 4 agents, 8x8, 3 food items that need cooperation. The team walks to the
        first uneaten item and loads it.
rware : Jumanji RobotWarehouse, 4 agents, one row of 3 shelf columns of height 8 (48 shelves), 4 requests.
        The policy reads the full state (a scripted team, not a learned one): fetch the nearest requested
        shelf, carry it to the nearest goal, put it back on the nearest free storage cell. A collision ends the
        episode, which the branch return then reflects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

EPS = 0.25


@dataclass
class EnvSpec:
    name: str
    step: Callable
    reset: Callable
    policy: Callable
    features: Callable
    live: Callable
    n: int
    num_actions: int
    reference_action: jnp.ndarray
    horizon: int
    collect_steps: int


# ---------------------------------------------------------------- Level-Based Foraging
def _lbf():
    from jumanji.environments.routing.lbf import env as lbfenv, generator
    n, food_n, grid, A = 4, 3, 8, 6
    env = lbfenv.LevelBasedForaging(
        generator=generator.RandomGenerator(grid_size=grid, fov=grid, num_agents=n, num_food=food_n, force_coop=True),
        time_limit=50)

    def policy(obs, key):
        view = obs.agents_view
        food = view[:, : 3 * food_n].reshape(n, food_n, 3)
        alive = food[:, :, 2] > 0
        first = jnp.argmax(alive, axis=1)
        tgt = jnp.take_along_axis(food, first[:, None, None], axis=1)[:, 0, :2]
        me = view[:, 3 * food_n: 3 * food_n + 2]
        d = tgt - me
        adjacent = jnp.abs(d).sum(-1) == 1
        vertical = jnp.abs(d[:, 0]) >= jnp.abs(d[:, 1])
        move = jnp.where(vertical, jnp.where(d[:, 0] < 0, 1, 2), jnp.where(d[:, 1] < 0, 3, 4))
        act = jnp.where(adjacent, 5, move)
        act = jnp.where(alive.any(axis=1), act, 0)
        k1, k2 = jax.random.split(key)
        return jnp.where(jax.random.uniform(k1, (n,)) < EPS, jax.random.randint(k2, (n,), 0, A), act)

    def features(state):
        a, f = state.agents, state.food_items
        return jnp.concatenate([(a.position / grid).reshape(-1), a.level / 5.0, (f.position / grid).reshape(-1),
                                f.level / 5.0, f.eaten.astype(jnp.float32), state.step_count[None] / 50.0])

    live = lambda states: ~np.asarray(states.food_items.eaten).all(1)
    return EnvSpec("lbf", env.step, env.reset, policy, features, live, n, A, jnp.zeros(n, jnp.int32), 15, 35)


# ---------------------------------------------------------------- RobotWarehouse
def _rware():
    from jumanji.environments.routing.robot_warehouse import env as rw, generator
    n, A = 4, 5
    env = rw.RobotWarehouse(generator=generator.RandomGenerator(
        shelf_rows=1, shelf_columns=3, column_height=8, num_agents=n, sensor_range=1, request_queue_size=4),
        time_limit=100)
    SH, AG = 0, 1                                             # grid layers: shelves, agents (checked 10 Nov 2026)
    highways = jnp.asarray(env.highways)
    goals_x, goals_y = jnp.asarray(env.goals)[:, 1], jnp.asarray(env.goals)[:, 0]   # goals are stored as (y, x)
    X, Y = highways.shape
    gx, gy = jnp.meshgrid(jnp.arange(X), jnp.arange(Y), indexing="ij")

    def step(state, actions):
        state, ts = env.step(state, actions)
        return state, ts.replace(observation=state)

    def reset(key):
        state, ts = env.reset(key)
        return state, ts.replace(observation=state)

    def nearest(px, py, cand_x, cand_y, ok):
        d = jnp.abs(cand_x[None, :] - px[:, None]) + jnp.abs(cand_y[None, :] - py[:, None])
        d = jnp.where(ok[None, :], d, 10 ** 6)
        k = jnp.argmin(d, axis=1)
        return cand_x[k], cand_y[k]

    def policy(state, key):
        ag, sh, grid = state.agents, state.shelves, state.grid
        px, py, dr, carry = ag.position.x, ag.position.y, ag.direction, ag.is_carrying.astype(bool)
        shelf_here = grid[SH, px, py]                                     # shelf id + 1 under each agent
        carried_req = carry & (sh.is_requested[jnp.maximum(shelf_here - 1, 0)] == 1) & (shelf_here > 0)
        # targets
        req = sh.is_requested == 1
        fx, fy = nearest(px, py, sh.position.x, sh.position.y, req)       # a requested shelf to fetch
        tx_goal, ty_goal = nearest(px, py, goals_x, goals_y, jnp.ones_like(goals_x, bool))
        free = (highways == 0) & (grid[SH] == 0)
        ex, ey = nearest(px, py, gx.reshape(-1), gy.reshape(-1), free.reshape(-1))   # free storage cell
        tx = jnp.where(~carry, fx, jnp.where(carried_req, tx_goal, ex))
        ty = jnp.where(~carry, fy, jnp.where(carried_req, ty_goal, ey))
        at = (tx == px) & (ty == py)
        # direction 0 up (x-1), 1 right (y+1), 2 down (x+1), 3 left (y-1).
        # Steer by a shortest-path distance field to the target. A carrying agent can only pass highway cells
        # and free storage cells; an empty-handed agent can pass under shelves.
        blocked_carry = (highways == 0) & (grid[SH] > 0)

        def want_dir(px1, py1, tx1, ty1, c1):
            passable = jnp.where(c1, ~blocked_carry, jnp.ones_like(blocked_carry))
            passable = passable.at[px1, py1].set(True).at[tx1, ty1].set(True)
            big = 10 ** 4
            d = jnp.full((X, Y), big).at[tx1, ty1].set(0)

            def relax(d, _):
                up = jnp.pad(d, ((1, 0), (0, 0)), constant_values=big)[:-1]
                down = jnp.pad(d, ((0, 1), (0, 0)), constant_values=big)[1:]
                left = jnp.pad(d, ((0, 0), (1, 0)), constant_values=big)[:, :-1]
                right = jnp.pad(d, ((0, 0), (0, 1)), constant_values=big)[:, 1:]
                nd = jnp.minimum(d, jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right)) + 1)
                return jnp.where(passable, nd, big), None

            d, _ = jax.lax.scan(relax, d, None, length=X + Y + 12)
            nb = jnp.stack([d[jnp.maximum(px1 - 1, 0), py1], d[px1, jnp.minimum(py1 + 1, Y - 1)],
                            d[jnp.minimum(px1 + 1, X - 1), py1], d[px1, jnp.maximum(py1 - 1, 0)]])
            edge = jnp.array([px1 == 0, py1 == Y - 1, px1 == X - 1, py1 == 0])
            return jnp.argmin(jnp.where(edge, big, nb))

        want = jax.vmap(want_dir)(px, py, tx, ty, carry)
        diff = (want - dr) % 4
        turn = jnp.where(diff == 1, 3, 2)                                  # 3 = turn right, 2 = turn left
        act = jnp.where(dr == want, 1, turn)
        act = jnp.where(at & (~carry | ~carried_req), 4, act)               # load, or put back
        act = jnp.where(at & carried_req, 0, act)
        # respect the action mask; random valid actions with probability EPS
        mask = state.action_mask
        k1, k2 = jax.random.split(key)
        rand = jax.random.categorical(k2, jnp.where(mask, 0.0, -1e9), axis=-1)
        ok = jnp.take_along_axis(mask, act[:, None], axis=1)[:, 0]
        act = jnp.where(ok, act, rand)
        act = jnp.where(jax.random.uniform(k1, (n,)) < EPS, rand, act).astype(jnp.int32)
        # collision guard: no forward move into a cell that is occupied now or claimed by a lower-numbered agent
        nx = jnp.clip(px + jnp.where(dr == 0, -1, jnp.where(dr == 2, 1, 0)), 0, X - 1)
        ny = jnp.clip(py + jnp.where(dr == 1, 1, jnp.where(dr == 3, -1, 0)), 0, Y - 1)
        fwd = act == 1
        occupied = grid[AG, nx, ny] > 0
        same = (nx[:, None] == nx[None, :]) & (ny[:, None] == ny[None, :]) & fwd[None, :]
        claimed = jnp.tril(same, k=-1).any(axis=1)
        return jnp.where(fwd & (occupied | claimed), 0, act).astype(jnp.int32)

    def features(state):
        ag, sh = state.agents, state.shelves
        req = sh.is_requested == 1
        order = jnp.argsort(~req)[:4]                                      # the 4 requested shelves
        return jnp.concatenate([ag.position.x / X, ag.position.y / Y, jax.nn.one_hot(ag.direction, 4).reshape(-1),
                                ag.is_carrying.astype(jnp.float32), sh.position.x[order] / X, sh.position.y[order] / Y,
                                state.step_count[None] / 100.0])

    live = lambda states: np.ones(np.asarray(states.step_count).shape[0], bool)
    return EnvSpec("rware", step, reset, policy, features, live, n, A, jnp.zeros(n, jnp.int32), 15, 90)


def get_env(name: str) -> EnvSpec:
    return {"lbf": _lbf, "rware": _rware}[name]()


def collect(spec: EnvSpec, key, num_envs):
    """Decision points (state, observation) visited by the scripted team, flattened over episodes and time."""
    def one(k):
        k0, k1 = jax.random.split(k)
        state, ts = spec.reset(k0)

        def body(carry, kk):
            state, ts = carry
            out = (state, ts.observation, ts.last())
            state, ts = spec.step(state, spec.policy(ts.observation, kk))
            return (state, ts), out

        _, traj = jax.lax.scan(body, (state, ts), jax.random.split(k1, spec.collect_steps))
        return traj

    states, obs, last = jax.jit(jax.vmap(one))(jax.random.split(key, num_envs))
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    return jax.tree_util.tree_map(flat, states), jax.tree_util.tree_map(flat, obs), np.asarray(flat(last))
