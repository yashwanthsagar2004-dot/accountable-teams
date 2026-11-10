"""Coalition critics in plain JAX (no Flax needed).

DeltaKernelCritic - kernel heads trained on paired branch differences with common random numbers; use this
                 when one decision moves the return only a little (most environment steps).

KernelCritic   - the default. Per-agent responsibility heads r_i(h, a) fitted to executed branch returns with
                 coalitions drawn from the Shapley kernel, plus learned v(empty) and v(N) with efficient
                 normalisation. At the optimum r_i is the exact Shapley value (Lundberg and Lee 2017, Thm 2).
PairCritic     - anchored order-2 interaction terms fitted on coalitions of size <= 2 (exact dividends up to
                 order 2; the readout misses the order >= 3 tail, see T2).
AdditiveOnPolicy - VDN-style additive model of the team value fitted on the team's own data (the baseline
                 whose bias T1b describes).
"""
from __future__ import annotations
import itertools

import jax
import jax.numpy as jnp
import numpy as np


def mlp_init(key, sizes):
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (din, dout)) / jnp.sqrt(din), jnp.zeros(dout)))
    return params


def mlp_apply(params, x):
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


class KernelCritic:
    """v_hat(S) = b + sum_{i in S} r_i, with r efficiently normalised so that b + sum_i r_i = v_hat(N)."""

    def __init__(self, n, d, hidden=128, key=jax.random.PRNGKey(0), a_dim=None):
        """d: width of the decision features h; a_dim: width of the action encoding (default n, one number per
        agent; use n * num_actions for one-hot actions)."""
        self.n = n
        self.params = mlp_init(key, [d + (n if a_dim is None else a_dim), hidden, hidden, n + 2])

    @staticmethod
    def heads(params, h, a, n):
        out = mlp_apply(params, jnp.concatenate([h, a], -1))
        b, total, raw = out[:, 0], out[:, 1], out[:, 2:]
        r = raw + ((total - b - raw.sum(-1)) / n)[:, None]      # efficient normalisation
        return b, total, r

    @staticmethod
    def loss(params, batch, n):
        h, a, mask, g, kind = batch                               # kind: 0 = empty, 1 = full team, 2 = kernel
        b, total, r = KernelCritic.heads(params, h, a, n)
        pred = jnp.where(kind == 0, b, jnp.where(kind == 1, total, b + (mask * r).sum(-1)))
        return jnp.mean((pred - g) ** 2)

    def readout(self, h, a):
        return np.asarray(KernelCritic.heads(self.params, h, a, self.n)[2])


class DeltaKernelCritic:
    """Kernel heads trained on paired differences d = G_S - G_empty, where the two branches share their random
    numbers. Predicts v(S) - v(empty) = sum_{i in S} r_i, with r efficiently normalised so that
    sum_i r_i = total(h, a), the predicted v(N) - v(empty). The minimiser is the same Shapley value as for
    KernelCritic (T1a'), but the regression noise is the noise of a paired difference, which is far smaller
    when a single decision changes the return only a little."""

    def __init__(self, n, d, hidden=128, key=jax.random.PRNGKey(0), a_dim=None):
        self.n = n
        self.params = mlp_init(key, [d + (n if a_dim is None else a_dim), hidden, hidden, n + 1])

    @staticmethod
    def heads(params, h, a, n):
        out = mlp_apply(params, jnp.concatenate([h, a], -1))
        total, raw = out[:, 0], out[:, 1:]
        return total, raw + ((total - raw.sum(-1)) / n)[:, None]

    @staticmethod
    def loss(params, batch, n):
        h, a, mask, d, full = batch                               # full = 1 when S is the whole team
        total, r = DeltaKernelCritic.heads(params, h, a, n)
        pred = jnp.where(full == 1, total, (mask * r).sum(-1))
        return jnp.mean((pred - d) ** 2)

    def readout(self, h, a):
        return np.asarray(DeltaKernelCritic.heads(self.params, h, a, self.n)[1])


class PairCritic:
    """v_hat(S) = g0 + sum_{i in S} g_i + sum_{i<j in S} g_ij (anchored: absent agents contribute nothing)."""

    def __init__(self, n, d, hidden=128, key=jax.random.PRNGKey(0)):
        self.n = n
        self.pairs = np.array(list(itertools.combinations(range(n), 2)))
        self.params = mlp_init(key, [d + n, hidden, hidden, 1 + n + len(self.pairs)])

    @staticmethod
    def parts(params, h, a, n, npairs):
        out = mlp_apply(params, jnp.concatenate([h, a], -1))
        return out[:, 0], out[:, 1:1 + n], out[:, 1 + n:1 + n + npairs]

    @staticmethod
    def loss(params, batch, n, pairs):
        h, a, mask, g = batch
        g0, g1, g2 = PairCritic.parts(params, h, a, n, len(pairs))
        both = mask[:, pairs[:, 0]] * mask[:, pairs[:, 1]]
        pred = g0 + (mask * g1).sum(-1) + (both * g2).sum(-1)
        return jnp.mean((pred - g) ** 2)

    def readout(self, h, a):
        g0, g1, g2 = PairCritic.parts(self.params, h, a, self.n, len(self.pairs))
        phi = np.array(g1)
        g2 = np.asarray(g2)
        for k, (i, j) in enumerate(self.pairs):
            phi[:, i] += g2[:, k] / 2
            phi[:, j] += g2[:, k] / 2
        return phi


class AdditiveOnPolicy:
    """Q_hat(h, x) = b(h) + sum_i u_i(h) x_i, fitted on (h, a, return) from the team's own behaviour."""

    def __init__(self, n, d, hidden=128, key=jax.random.PRNGKey(0)):
        self.n = n
        self.params = mlp_init(key, [d, hidden, hidden, n + 1])

    @staticmethod
    def loss(params, batch, n):
        h, a, g = batch
        out = mlp_apply(params, h)
        return jnp.mean((out[:, 0] + (a * out[:, 1:]).sum(-1) - g) ** 2)

    def readout(self, h, a):
        out = np.asarray(mlp_apply(self.params, h))
        return out[:, 1:] * a                                      # credit of a_i against the reference 0


def fit(model, loss_fn, data, steps=6000, batch=512, lr=1e-3, seed=0, val_frac=0.1, eval_every=100, patience=5, **kw):
    """Adam with early stopping on a held-out split, so small budgets do not overfit."""
    import optax
    opt = optax.adam(lr)
    state = opt.init(model.params)
    rng = np.random.default_rng(seed)
    size = data[0].shape[0]
    perm = rng.permutation(size)
    nval = max(1, int(val_frac * size))
    val = tuple(jnp.asarray(x[perm[:nval]]) for x in data)
    train = tuple(jnp.asarray(x[perm[nval:]]) for x in data)
    ntrain = size - nval

    @jax.jit
    def step(params, state, idx):
        b = tuple(x[idx] for x in train)
        l, grads = jax.value_and_grad(loss_fn)(params, b, **kw)
        upd, state = opt.update(grads, state)
        return optax.apply_updates(params, upd), state, l

    val_loss = jax.jit(lambda p: loss_fn(p, val, **kw))
    params, best, best_params, bad = model.params, np.inf, model.params, 0
    for t in range(1, steps + 1):
        params, state, _ = step(params, state, jnp.asarray(rng.integers(0, ntrain, batch)))
        if t % eval_every == 0:
            v = float(val_loss(params))
            if v < best - 1e-6:
                best, best_params, bad = v, params, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    model.params = best_params
    return best
