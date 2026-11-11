"""Permutation-equivariant responsibility critics, ensembles and audit-hybrid estimators.

SetKernelCritic is KernelCritic with the right inductive bias for a team: every agent is a token (its own
features and its one-hot action), the tokens share one encoder, self-attention lets agents read each other, and a
shared head reads out each agent's responsibility. Permuting the agents permutes the outputs (equivariance), the
empty-team and full-team values are permutation-invariant, and the same weights work for any team size.

As with KernelCritic, the heads r_i are fitted to executed branch returns of coalitions drawn from the Shapley
kernel, with efficient normalisation b + sum_i r_i = v_hat(N); at the optimum r_i is the exact Shapley value of
the interventional game (T1a'). SetDeltaKernelCritic is the paired-difference version.

Ensembles give a per-decision, per-agent uncertainty. Two estimators then spend a small audit budget:
  gated_hybrid       - executed permutation audits only on the decisions the ensemble is least sure about;
  precision_weighted - every decision gets k audit branches, combined with the critic by inverse variance, with
                       the critic's variance calibrated on a held-out calibration set.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .critic import fit


def _dense(key, din, dout):
    return (jax.random.normal(key, (din, dout)) / jnp.sqrt(din), jnp.zeros(dout))


def _ln(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)


def init_set_params(key, d_agent, d_global, a_dim, width=128, layers=2, heads=4, out_extra=2):
    """Parameters of the set critic. out_extra: number of team-level outputs (2: v(empty), v(N); 1: v(N))."""
    keys = iter(jax.random.split(key, 6 + 6 * layers))
    p = {"embed": _dense(next(keys), d_agent + a_dim + d_global, width),
         "embed2": _dense(next(keys), width, width),
         "blocks": [],
         "head": _dense(next(keys), width, 1),
         "team": _dense(next(keys), 2 * width, width),
         "team_out": _dense(next(keys), width, out_extra),
         "nheads": jnp.zeros((heads,))}          # the head count, carried in a shape so jit sees it as static
    for _ in range(layers):
        p["blocks"].append({"qkv": _dense(next(keys), width, 3 * width), "proj": _dense(next(keys), width, width),
                            "ff1": _dense(next(keys), width, 2 * width), "ff2": _dense(next(keys), 2 * width, width)})
    return p


def set_forward(p, agent_x, a, g):
    """agent_x (B, n, d_agent), a (B, n, a_dim), g (B, d_global) -> per-agent raw (B, n), team outputs (B, k)."""
    n = agent_x.shape[1]
    x = jnp.concatenate([agent_x, a, jnp.broadcast_to(g[:, None, :], g.shape[:1] + (n, g.shape[-1]))], -1)
    w, b = p["embed"]
    h = jax.nn.gelu(x @ w + b)
    w, b = p["embed2"]
    h = h + jax.nn.gelu(h @ w + b)
    H = p["nheads"].shape[0]
    for blk in p["blocks"]:
        z = _ln(h)
        w, b = blk["qkv"]
        q, k, v = jnp.split(z @ w + b, 3, -1)
        split = lambda t: t.reshape(t.shape[:2] + (H, -1)).transpose(0, 2, 1, 3)       # (B, H, n, dh)
        q, k, v = split(q), split(k), split(v)
        att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(q.shape[-1]), -1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(h.shape)
        w, b = blk["proj"]
        h = h + o @ w + b
        z = _ln(h)
        w1, b1 = blk["ff1"]
        w2, b2 = blk["ff2"]
        h = h + jax.nn.gelu(z @ w1 + b1) @ w2 + b2
    h = _ln(h)
    w, b = p["head"]
    raw = (h @ w + b)[..., 0]
    pooled = jnp.concatenate([h.mean(1), h.max(1)], -1)                            # permutation-invariant
    w, b = p["team"]
    w2, b2 = p["team_out"]
    team = jax.nn.gelu(pooled @ w + b) @ w2 + b2
    return raw, team


class SetKernelCritic:
    """v_hat(S) = b + sum_{i in S} r_i with b = v_hat(empty), r efficiently normalised to v_hat(N)."""

    def __init__(self, d_agent, d_global, a_dim, width=128, layers=2, heads=4, key=jax.random.PRNGKey(0)):
        self.params = init_set_params(key, d_agent, d_global, a_dim, width, layers, heads, out_extra=2)

    @staticmethod
    def heads(params, x, a, g):
        raw, team = set_forward(params, x, a, g)
        b, total = team[:, 0], team[:, 1]
        r = raw + ((total - b - raw.sum(-1)) / raw.shape[1])[:, None]
        return b, total, r

    @staticmethod
    def loss(params, batch):
        x, a, g, mask, ret, kind = batch                          # kind: 0 = empty, 1 = full team, 2 = kernel
        b, total, r = SetKernelCritic.heads(params, x, a, g)
        pred = jnp.where(kind == 0, b, jnp.where(kind == 1, total, b + (mask * r).sum(-1)))
        return jnp.mean((pred - ret) ** 2)

    def readout(self, x, a, g, chunk=4096):
        f = jax.jit(lambda p, x, a, g: SetKernelCritic.heads(p, x, a, g)[2])
        return np.concatenate([np.asarray(f(self.params, x[i:i + chunk], a[i:i + chunk], g[i:i + chunk]))
                               for i in range(0, len(x), chunk)])


class SetDeltaKernelCritic:
    """Paired-difference version: predicts v(S) - v(empty) = sum_{i in S} r_i, with sum_i r_i = v_hat(N) - v_hat(empty)."""

    def __init__(self, d_agent, d_global, a_dim, width=128, layers=2, heads=4, key=jax.random.PRNGKey(0)):
        self.params = init_set_params(key, d_agent, d_global, a_dim, width, layers, heads, out_extra=1)

    @staticmethod
    def heads(params, x, a, g):
        raw, team = set_forward(params, x, a, g)
        total = team[:, 0]
        return total, raw + ((total - raw.sum(-1)) / raw.shape[1])[:, None]

    @staticmethod
    def loss(params, batch):
        x, a, g, mask, d, full = batch
        total, r = SetDeltaKernelCritic.heads(params, x, a, g)
        return jnp.mean((jnp.where(full == 1, total, (mask * r).sum(-1)) - d) ** 2)

    def readout(self, x, a, g, chunk=4096):
        f = jax.jit(lambda p, x, a, g: SetDeltaKernelCritic.heads(p, x, a, g)[1])
        return np.concatenate([np.asarray(f(self.params, x[i:i + chunk], a[i:i + chunk], g[i:i + chunk]))
                               for i in range(0, len(x), chunk)])


def fit_ensemble(cls, data, members=4, d_agent=None, d_global=None, a_dim=None, width=128, layers=2, heads=4,
                 seed=0, bootstrap=True, **fit_kw):
    """Train `members` critics on bootstrap resamples of the branch data. Returns the list of fitted critics."""
    rng = np.random.default_rng(seed)
    out = []
    for m in range(members):
        idx = rng.integers(0, len(data[0]), len(data[0])) if bootstrap else np.arange(len(data[0]))
        model = cls(d_agent, d_global, a_dim, width, layers, heads, key=jax.random.PRNGKey(1000 * seed + m))
        fit(model, cls.loss, tuple(x[idx] for x in data), seed=seed * 100 + m, **fit_kw)
        out.append(model)
    return out


def ensemble_readout(models, x, a, g):
    """Mean and standard deviation over members, each of shape (B, n)."""
    r = np.stack([m.readout(x, a, g) for m in models])
    return r.mean(0), r.std(0, ddof=1) if len(models) > 1 else np.zeros(r.shape[1:])


# ------------------------------------------------------------------ audit hybrids
def permutation_estimate(vals, n, perms_per_agent, rng):
    """Permutation-sampling Shapley estimate from a table of branch returns vals[mask, rep] for ONE decision.
    Uses 2 * perms_per_agent * n branches; returns the estimate and the per-agent sample variance of the mean."""
    reps = vals.shape[1]
    est, var = np.zeros(n), np.zeros(n)
    for i in range(n):
        d = []
        for _ in range(perms_per_agent):
            perm = rng.permutation(n)
            pos = int(np.where(perm == i)[0][0])
            s = sum(1 << int(q) for q in perm[:pos])
            r1, r2 = rng.integers(reps), rng.integers(reps)        # two independent branches
            d.append(vals[s | (1 << i), r1] - vals[s, r2])
        d = np.asarray(d)
        est[i] = d.mean()
        var[i] = d.var(ddof=1) / len(d) if len(d) > 1 else np.nan
    return est, var


def gated_hybrid(critic_mean, critic_std, audit_est, frac):
    """Replace the critic by the audit estimate on the `frac` of decisions with the largest ensemble spread."""
    if frac <= 0:
        return critic_mean.copy()
    score = critic_std.sum(1)
    cut = np.quantile(score, 1 - frac)
    out = critic_mean.copy()
    pick = score >= cut
    out[pick] = audit_est[pick]
    return out


def calibrate_scale(critic_mean, critic_std, truth):
    """One multiplier c so that c * ensemble std matches the critic's actual error (least squares on squares)."""
    err2, s2 = (critic_mean - truth) ** 2, critic_std ** 2 + 1e-12
    return float(np.sqrt(np.sum(err2 * s2) / np.sum(s2 * s2)))


def precision_weighted(critic_mean, critic_std, scale, audit_est, audit_var):
    """Inverse-variance combination of the critic (variance (scale * std)^2) and the audit (its sample variance)."""
    vc = (scale * critic_std) ** 2 + 1e-12
    va = np.where(np.isfinite(audit_var), audit_var, np.inf) + 1e-12
    w = (1 / vc) / (1 / vc + 1 / va)
    return w * critic_mean + (1 - w) * audit_est
