import jax
import jax.numpy as jnp
import numpy as np

from accteams.games import PlantedGame
from accteams.setcritic import (SetDeltaKernelCritic, SetKernelCritic, calibrate_scale, fit_ensemble, gated_hybrid,
                                permutation_estimate, precision_weighted)
from accteams.shapley import sample_kernel_coalitions, shapley_exact


def _inputs(B=5, n=4, da=3, dg=2, A=2, seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(B, n, da)), jnp.float32), jnp.asarray(rng.normal(size=(B, n, A)), jnp.float32),
            jnp.asarray(rng.normal(size=(B, dg)), jnp.float32))


def test_equivariance_and_efficiency():
    x, a, g = _inputs()
    m = SetKernelCritic(3, 2, 2, width=32, layers=2, heads=4)
    b, total, r = SetKernelCritic.heads(m.params, x, a, g)
    perm = np.array([2, 0, 3, 1])
    b2, total2, r2 = SetKernelCritic.heads(m.params, x[:, perm], a[:, perm], g)
    assert np.allclose(r2, r[:, perm], atol=1e-5)                       # equivariant responsibilities
    assert np.allclose(b2, b, atol=1e-5) and np.allclose(total2, total, atol=1e-5)   # invariant team values
    assert np.allclose(b + r.sum(1), total, atol=1e-5)                  # efficiency
    d = SetDeltaKernelCritic(3, 2, 2, width=32, layers=1, heads=2)
    t, rd = SetDeltaKernelCritic.heads(d.params, x, a, g)
    assert np.allclose(rd.sum(1), t, atol=1e-5)


def test_any_team_size():
    m = SetKernelCritic(3, 2, 2, width=32)
    for n in (2, 5, 9):
        x, a, g = _inputs(n=n)
        assert m.readout(np.asarray(x), np.asarray(a), np.asarray(g)).shape == (5, n)


def test_set_critic_recovers_shapley_on_a_planted_game():
    n, budget = 4, 30000
    game = PlantedGame.make(n=n, max_order=3, seed=0, noise=0.3)
    rng = np.random.default_rng(1)
    h, act, _ = game.sample_decisions(budget, rng)
    kind = rng.choice(3, size=budget, p=[0.1, 0.1, 0.8])
    mask = sample_kernel_coalitions(n, budget, rng)
    mask[kind == 0], mask[kind == 1] = 0.0, 1.0
    ret = game.branch_return(h, act, mask, rng).astype(np.float32)
    ident = np.broadcast_to(np.eye(n, dtype=np.float32), (budget, n, n))   # agents are told apart by their identity
    data = (ident, act[:, :, None].astype(np.float32), h, mask, ret, kind.astype(np.int32))
    models = fit_ensemble(SetKernelCritic, data, members=2, d_agent=n, d_global=game.d, a_dim=1, width=64,
                          layers=1, heads=2, steps=3000, batch=512, lr=2e-3, eval_every=200, patience=6)
    he, ae, _ = game.sample_decisions(2000, rng)
    phi = game.responsibility(he, ae)
    ie = np.broadcast_to(np.eye(n, dtype=np.float32), (2000, n, n))
    est = np.mean([m.readout(ie, ae[:, :, None].astype(np.float32), he) for m in models], 0)
    rel = np.mean(np.abs(est - phi)) / np.mean(np.abs(phi))
    assert rel < 0.45, rel                    # the flat critic reaches about 0.37 at 20k returns in the pilot


def test_audit_estimators():
    rng = np.random.default_rng(0)
    n = 3
    true_v = {s: rng.normal() for s in range(1 << n)}
    phi = shapley_exact(lambda S: true_v[sum(1 << j for j in S)], n)
    vals = np.array([[true_v[s] + rng.normal(0, 0.1) for _ in range(400)] for s in range(1 << n)])
    est, var = permutation_estimate(vals, n, 200, rng)
    assert np.all(np.abs(est - phi) < 4 * np.sqrt(var) + 0.05)
    cm, cs = np.zeros((10, n)), np.ones((10, n))
    cs[0] = 5.0
    out = gated_hybrid(cm, cs, np.ones((10, n)), frac=0.1)
    assert np.all(out[0] == 1) and np.all(out[1:] == 0)
    scale = calibrate_scale(cm, cs, np.full((10, n), 0.5))
    pw = precision_weighted(cm, cs, scale, np.ones((10, n)), np.full((10, n), 1e-6))
    assert np.allclose(pw, 1, atol=1e-3)                                # a near-exact audit dominates
