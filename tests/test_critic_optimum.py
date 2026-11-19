"""Regression test for adversarial-review finding C1: the trained critic's optimum must be the Shapley value
even when the game has a third-order dividend (v = 1 iff agents 0, 1 and 2 are all present, agent 3 a dummy)."""
import jax
import numpy as np

from accteams.critic import KernelCritic, fit
from accteams.shapley import sample_kernel_coalitions


def test_kernel_critic_recovers_shapley_on_three_way_and():
    n, m = 4, 6000
    rng = np.random.default_rng(0)
    kind = rng.choice(3, size=m, p=[0.1, 0.1, 0.8]).astype(np.int32)
    mask = sample_kernel_coalitions(n, m, rng, paired=True).astype(np.float32)
    mask[kind == 0], mask[kind == 1] = 0.0, 1.0
    g = (mask[:, :3].sum(1) == 3).astype(np.float32)          # noise-free 3-way AND
    h = np.ones((m, 1), np.float32)
    a = np.zeros((m, n), np.float32)
    model = KernelCritic(n, 1, hidden=32, key=jax.random.PRNGKey(0))
    fit(model, KernelCritic.loss, (h, a, mask, g, kind), steps=4000, batch=1024, lr=3e-3, seed=0, n=n,
        eval_every=500, patience=10 ** 6)
    phi = model.readout(h[:1], a[:1])[0]
    assert np.allclose(phi, [1 / 3, 1 / 3, 1 / 3, 0.0], atol=0.03), phi


def test_delta_critic_recovers_shapley_on_pairwise_and():
    from accteams.critic import DeltaKernelCritic
    n, m = 4, 6000
    rng = np.random.default_rng(1)
    full = (rng.random(m) < 0.15).astype(np.int32)
    mask = sample_kernel_coalitions(n, m, rng, paired=True).astype(np.float32)
    mask[full == 1] = 1.0
    d = (mask[:, 0] * mask[:, 1]).astype(np.float32)          # v = AND(0, 1), v(empty) = 0
    h, a = np.ones((m, 1), np.float32), np.zeros((m, n), np.float32)
    model = DeltaKernelCritic(n, 1, hidden=32, key=jax.random.PRNGKey(0))
    fit(model, DeltaKernelCritic.loss, (h, a, mask, d, full), steps=4000, batch=1024, lr=3e-3, seed=0, n=n,
        eval_every=500, patience=10 ** 6)
    phi = model.readout(h[:1], a[:1])[0]
    assert np.allclose(phi, [0.5, 0.5, 0.0, 0.0], atol=0.03), phi
