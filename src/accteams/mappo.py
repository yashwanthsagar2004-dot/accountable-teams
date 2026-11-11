"""MAPPO for Jumanji tasks behind accteams' EnvSpec, with responsibility-based credit.

One jitted update = roll out E parallel environments for T steps, compute advantages, run PPO epochs.
The actor is shared across agents (parameter sharing) and reads each agent's own token
spec.agent_features(state)[i] plus a one-hot agent index; the centralised value function reads the whole state.

Credit modes (the advantage of agent i at step t):
  team   A_i = GAE of the team reward                              (standard MAPPO, the baseline)
  resp   A_i = GAE + beta * c_i,  c_i = phi_i(s, a) - E_{a'~pi_i} phi_i(s, a', a_-i)
         where phi is the SetKernelCritic responsibility. The critic is trained online: at every update a
         random sample of the rollout's decisions is branched once each (coalition from the Shapley kernel,
         absent agents take the reference action for `window` steps, everyone then follows the CURRENT policy
         for `horizon` steps), so the critic tracks the responsibility of the learning team.
  dr     A_i = GAE + beta * c_i with c_i from a centralised Q(s, a) critic trained on the team's own returns
         (COMA-style counterfactual baseline: Q(s, a) - E_{a'~pi_i} Q(s, a', a_-i)); the on-policy
         counterpart of `resp`, so the comparison isolates executed branches against on-policy fitting.
c_i is standardised over the batch before beta is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .branching import make_branch_fn
from .critic import mlp_apply, mlp_init
from .setcritic import SetKernelCritic, init_set_params
from .shapley import kernel_size_probs


@dataclass(frozen=True)
class Config:
    num_envs: int = 64
    rollout: int = 32
    updates: int = 500
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 0.01
    lr: float = 3e-4
    hidden: int = 128
    credit: str = "team"            # team | resp | dr
    beta: float = 0.5
    branch_frac: float = 0.25       # fraction of rollout decisions branched once each (resp)
    window: int = 10
    horizon: int = 10
    critic_steps: int = 50          # responsibility / Q critic SGD steps per update
    critic_width: int = 64
    seed: int = 0


def _state_step(spec):
    def step(state, actions):
        state, ts = spec.step(state, actions)
        return state, ts.replace(observation=state)
    return step


def _kernel_masks(key, n, m):
    """m coalition masks from the Shapley kernel (sizes 1..n-1), as floats."""
    ks, kp = jax.random.split(key)
    size = jax.random.choice(ks, jnp.arange(1, n), (m,), p=jnp.asarray(kernel_size_probs(n)))
    ranks = jnp.argsort(jax.random.uniform(kp, (m, n)), axis=1).argsort(axis=1)
    return (ranks < size[:, None]).astype(jnp.float32)


def make_trainer(spec, cfg: Config):
    n, A = spec.n, spec.num_actions
    step_fn = _state_step(spec)
    k0 = jax.random.PRNGKey(cfg.seed)
    s0, _ = spec.reset(k0)
    d_agent = spec.agent_features(s0).shape[1]
    d_global = spec.global_features(s0).shape[0]
    d_flat = spec.features(s0).shape[0]
    eye = jnp.eye(n)

    def actor_in(state):
        return jnp.concatenate([spec.agent_features(state), eye], 1)            # (n, d_agent + n)

    def logits_fn(ap, state):
        return mlp_apply(ap, actor_in(state))                                    # (n, A)

    def value_fn(vp, state):
        return mlp_apply(vp, spec.features(state)[None])[0, 0]

    def policy_from(ap):
        def pol(state, key):
            return jax.random.categorical(key, logits_fn(ap, state)).astype(jnp.int32)
        return pol

    def init(key):
        ka, kv, kc, kq, ke = jax.random.split(key, 5)
        params = {"actor": mlp_init(ka, [d_agent + n, cfg.hidden, cfg.hidden, A]),
                  "value": mlp_init(kv, [d_flat, cfg.hidden * 2, cfg.hidden * 2, 1])}
        if cfg.credit == "resp":
            params["resp"] = init_set_params(kc, d_agent, d_global, A, cfg.critic_width, 2, 4, out_extra=2)
        if cfg.credit == "dr":
            params["q"] = mlp_init(kq, [d_flat + n * A, cfg.hidden * 2, cfg.hidden * 2, 1])
        states, _ = jax.vmap(spec.reset)(jax.random.split(ke, cfg.num_envs))
        return params, states

    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    copt = optax.adam(1e-3)

    def strip(p):                                     # integer-free pytree for the optimiser
        return {k: v for k, v in p.items() if k != "nheads"} if isinstance(p, dict) else p

    def env_step(state, action, key):
        nstate, ts = step_fn(state, action)
        done = ts.last()
        rstate, _ = spec.reset(key)
        nstate = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), rstate, nstate)
        return nstate, ts.reward.sum(), done

    def rollout(ap, vp, states, key):
        def body(carry, k):
            states = carry
            k1, k2 = jax.random.split(k)
            logits = jax.vmap(logits_fn, (None, 0))(ap, states)
            act = jax.random.categorical(k1, logits).astype(jnp.int32)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), act[..., None], -1)[..., 0]
            val = jax.vmap(value_fn, (None, 0))(vp, states)
            nstates, rew, done = jax.vmap(env_step)(states, act, jax.random.split(k2, cfg.num_envs))
            return nstates, (states, act, logp, val, rew, done)
        states, traj = jax.lax.scan(body, states, jax.random.split(key, cfg.rollout))
        return states, traj

    def gae(val, rew, done, last_val):
        def body(carry, x):
            adv, nv = carry
            v, r, d = x
            delta = r + cfg.gamma * nv * (1 - d) - v
            adv = delta + cfg.gamma * cfg.lam * (1 - d) * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_val), last_val), (val, rew, done.astype(jnp.float32)),
                              reverse=True)
        return adv, adv + val

    def credit_scores(params, flat_states, act):
        """c_i = f_i(s, a) - E_{a'_i ~ pi_i} f_i(s, a'_i, a_-i) for every decision; shape (B, n)."""
        logits = jax.vmap(logits_fn, (None, 0))(params["actor"], flat_states)
        pi = jax.nn.softmax(logits)                                               # (B, n, A)
        oh = jax.nn.one_hot(act, A)                                               # (B, n, A)
        if cfg.credit == "resp":
            x = jax.vmap(spec.agent_features)(flat_states)
            g = jax.vmap(spec.global_features)(flat_states)
            rp = params["resp"]
            f = lambda oh_: SetKernelCritic.heads(rp, x, oh_, g)[2]                # (B, n)
            base = f(oh)
            alt = jnp.stack([f(oh.at[:, i].set(jax.nn.one_hot(jnp.full(act.shape[0], b), A)))[:, i]
                             for i in range(n) for b in range(A)], 1).reshape(-1, n, A)
        else:
            h = jax.vmap(spec.features)(flat_states)
            q = lambda oh_: mlp_apply(params["q"], jnp.concatenate([h, oh_.reshape(oh_.shape[0], -1)], 1))[:, 0]
            base = jnp.broadcast_to(q(oh)[:, None], act.shape)
            alt = jnp.stack([q(oh.at[:, i].set(jax.nn.one_hot(jnp.full(act.shape[0], b), A)))
                             for i in range(n) for b in range(A)], 1).reshape(-1, n, A)
        c = base - (pi * alt).sum(-1)
        return (c - c.mean()) / (c.std() + 1e-8)

    def train_resp(params, cstate, flat_states, act, key):
        """One executed branch for a random sample of decisions, then SGD on the kernel-critic loss."""
        B = act.shape[0]
        m = max(8, int(cfg.branch_frac * B))
        ki, km, kk, kb = jax.random.split(key, 4)
        idx = jax.random.choice(ki, B, (m,), replace=False)
        s, a = jax.tree_util.tree_map(lambda z: z[idx], flat_states), act[idx]
        kind = jax.random.choice(kk, 3, (m,), p=jnp.array([0.1, 0.1, 0.8]))
        mask = _kernel_masks(km, n, m)
        mask = jnp.where((kind == 0)[:, None], 0.0, jnp.where((kind == 1)[:, None], 1.0, mask))
        branch = make_branch_fn(step_fn, policy_from(params["actor"]), spec.reference_action,
                                horizon=cfg.horizon, gamma=cfg.gamma, window=cfg.window)
        ret = jax.vmap(branch)(s, a, mask, jax.random.split(kb, m))
        x, g = jax.vmap(spec.agent_features)(s), jax.vmap(spec.global_features)(s)
        oh = jax.nn.one_hot(a, A)
        heads = params["resp"]["nheads"]

        def body(carry, _):
            rp, cs = carry
            l, gr = jax.value_and_grad(lambda q: SetKernelCritic.loss({**q, "nheads": heads},
                                                                      (x, oh, g, mask, ret, kind)))(rp)
            up, cs = copt.update(gr, cs, rp)
            return (optax.apply_updates(rp, up), cs), l
        (rp, cstate), losses = jax.lax.scan(body, (strip(params["resp"]), cstate), None, length=cfg.critic_steps)
        return {**params, "resp": {**rp, "nheads": heads}}, cstate, losses[-1], ret.std()

    def train_q(params, cstate, flat_states, act, targets):
        h = jax.vmap(spec.features)(flat_states)
        inp = jnp.concatenate([h, jax.nn.one_hot(act, A).reshape(act.shape[0], -1)], 1)

        def body(carry, _):
            qp, cs = carry
            l, gr = jax.value_and_grad(lambda q: jnp.mean((mlp_apply(q, inp)[:, 0] - targets) ** 2))(qp)
            up, cs = copt.update(gr, cs, qp)
            return (optax.apply_updates(qp, up), cs), l
        (qp, cstate), losses = jax.lax.scan(body, (params["q"], cstate), None, length=cfg.critic_steps)
        return {**params, "q": qp}, cstate, losses[-1], targets.std()

    def update(carry, key):
        params, ostate, cstate, states = carry
        kr, kc, kp = jax.random.split(key, 3)
        states, (S, act, logp, val, rew, done) = rollout(params["actor"], params["value"], states, kr)
        last_val = jax.vmap(value_fn, (None, 0))(params["value"], states)
        adv, ret = gae(val, rew, done, last_val)                                   # (T, E)
        T, E = cfg.rollout, cfg.num_envs
        flat = lambda z: z.reshape((T * E,) + z.shape[2:])
        fS = jax.tree_util.tree_map(flat, S)
        fact, flogp, fadv, fret = flat(act), flat(logp), flat(adv), flat(ret)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        aux_loss, aux_sd = jnp.float32(0), jnp.float32(0)
        if cfg.credit == "resp":
            params, cstate, aux_loss, aux_sd = train_resp(params, cstate, fS, fact, kc)
        if cfg.credit == "dr":
            params, cstate, aux_loss, aux_sd = train_q(params, cstate, fS, fact, fret)
        if cfg.credit in ("resp", "dr"):
            agent_adv = fadv[:, None] + cfg.beta * jax.lax.stop_gradient(credit_scores(params, fS, fact))
        else:
            agent_adv = jnp.broadcast_to(fadv[:, None], fact.shape)

        def loss_fn(p, mb):
            s, a, lp_old, adv_i, r = mb
            logits = jax.vmap(logits_fn, (None, 0))(p["actor"], s)
            lsm = jax.nn.log_softmax(logits)
            lp = jnp.take_along_axis(lsm, a[..., None], -1)[..., 0]
            ratio = jnp.exp(lp - lp_old)
            pg = -jnp.minimum(ratio * adv_i, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_i).mean()
            ent = -(jnp.exp(lsm) * lsm).sum(-1).mean()
            v = jax.vmap(value_fn, (None, 0))(p["value"], s)
            return pg + 0.5 * ((v - r) ** 2).mean() - cfg.ent * ent, (pg, ent)

        def epoch(carry, k):
            p, o = carry
            perm = jax.random.permutation(k, T * E).reshape(cfg.minibatches, -1)

            def mb_step(carry, idx):
                p, o = carry
                mb = (jax.tree_util.tree_map(lambda z: z[idx], fS), fact[idx], flogp[idx], agent_adv[idx], fret[idx])
                ap = {"actor": p["actor"], "value": p["value"]}
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(ap, mb)
                up, o = opt.update(g, o, ap)
                ap = optax.apply_updates(ap, up)
                return ({**p, **ap}, o), l
            (p, o), ls = jax.lax.scan(mb_step, (p, o), perm)
            return (p, o), ls.mean()
        (params, ostate), losses = jax.lax.scan(epoch, (params, ostate), jax.random.split(kp, cfg.epochs))
        ep_ret = rew.sum() / jnp.maximum(done.sum(), 1)                            # mean return of finished episodes
        metrics = dict(reward_per_step=rew.mean(), episodes=done.sum(), return_per_episode=ep_ret,
                       loss=losses.mean(), aux_loss=aux_loss, aux_target_sd=aux_sd)
        return (params, ostate, cstate, states), metrics

    def train(key, updates=None):
        params, states = init(key)
        ostate = opt.init({"actor": params["actor"], "value": params["value"]})
        if cfg.credit == "resp":
            cstate = copt.init(strip(params["resp"]))
        elif cfg.credit == "dr":
            cstate = copt.init(params["q"])
        else:
            cstate = jnp.zeros(())
        (params, _, _, _), metrics = jax.lax.scan(update, (params, ostate, cstate, states),
                                                  jax.random.split(jax.random.fold_in(key, 1), updates or cfg.updates))
        return params, metrics

    return train, dict(d_agent=d_agent, d_global=d_global, d_flat=d_flat)
