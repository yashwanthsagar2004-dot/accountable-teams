"""Fast checks of the facts the method relies on (run: pytest -q)."""
import itertools
import math

import numpy as np

from accteams.games import PlantedGame
from accteams.shapley import (all_subsets, dividends, kernel_least_squares, kernel_size_probs,
                              readout_from_dividends, sample_kernel_coalitions, shapley_exact)

RNG = np.random.default_rng(0)


def random_game(n):
    vals = {s: RNG.normal() for s in all_subsets(n)}
    return lambda s: vals[frozenset(s)]


def test_kernel_least_squares_is_shapley():
    for n in range(2, 7):
        v = random_game(n)
        assert np.allclose(kernel_least_squares(v, n), shapley_exact(v, n), atol=1e-9)


def test_low_order_dividends_from_small_groups_are_exact_and_error_is_tail():   # T1a, T2
    n, k = 5, 2
    v = random_game(n)
    full = dividends(v, n)
    low = dividends(v, n, max_order=k)                 # uses only v(R) with |R| <= k
    for t, c in low.items():
        assert abs(c - full[t]) < 1e-12
    err = readout_from_dividends(low, n, k) - shapley_exact(v, n)
    tail = np.array([sum(c / len(t) for t, c in full.items() if i in t and len(t) > k) for i in range(n)])
    assert np.allclose(err, -tail)


def test_additive_onpolicy_bias_matches_closed_form():                          # T1b, k = 1 (P1)
    n, p = 3, np.array([0.9, 0.5, 0.2])
    m = {frozenset([0]): .3, frozenset([1]): .2, frozenset([2]): .1, frozenset([0, 1, 2]): 1.5}
    xs = list(itertools.product((0, 1), repeat=n))
    w = np.array([np.prod([p[j] if x[j] else 1 - p[j] for j in range(n)]) for x in xs])
    a = np.array([[1.0, *x] for x in xs])
    y = np.array([sum(c for t, c in m.items() if all(x[j] for j in t)) for x in xs])
    beta = np.linalg.solve(a.T @ (w[:, None] * a), a.T @ (w * y))[1:]
    v = lambda s: sum(c for t, c in m.items() if t <= s)
    phi = shapley_exact(v, n)
    pred = [1.5 * (np.prod([p[j] for j in range(n) if j != i]) - 1 / 3) for i in range(n)]
    assert np.allclose(beta - phi, pred)


def test_unanchored_terms_are_not_unique():                                      # T1c
    c = 0.6
    q_a = lambda x1, x2: 0.5 * x1 + 0.3 * x2 + 0.4 * x1 * x2
    q_b = lambda x1, x2: (0.5 - c) * x1 + 0.3 * x2 + (0.4 * x1 * x2 + c * x1)
    assert all(abs(q_a(a, b) - q_b(a, b)) < 1e-12 for a in (0, 1) for b in (0, 1))
    ro_a = np.array([0.5 + 0.2, 0.3 + 0.2])
    ro_b = np.array([0.5 - c + (0.4 + c) / 2, 0.3 + (0.4 + c) / 2])
    assert np.allclose(ro_b - ro_a, [-c / 2, c / 2])


def test_detectable_audit_shifts_half_the_blame():                               # T6
    for delta in (0.0, 0.1, 0.5, 1.0):
        vals = {frozenset(): 0., frozenset([0]): 0., frozenset([1]): 1 - delta, frozenset([0, 1]): 1.}
        phi = shapley_exact(lambda s: vals[frozenset(s)], 2)
        assert abs(phi[0] - delta / 2) < 1e-12 and phi[0] <= 2 * delta + 1e-12


def test_planted_game_responsibility_matches_enumeration():
    game = PlantedGame.make(n=4, max_order=3, seed=1)
    h, a, _ = game.sample_decisions(5, np.random.default_rng(2))
    phi = game.responsibility(h, a)
    for b in range(5):
        v = lambda s: float(game.coalition_value(h[b:b + 1], a[b:b + 1],
                                                 np.array([[1.0 if i in s else 0.0 for i in range(4)]]))[0])
        assert np.allclose(phi[b], shapley_exact(v, 4), atol=1e-9)


def test_kernel_size_probabilities():
    for n in range(2, 9):
        p = kernel_size_probs(n)
        assert abs(p.sum() - 1) < 1e-12 and np.allclose(p, p[::-1])


def test_kernel_sampler_sizes_and_pairing():
    n, m = 6, 40000
    rng = np.random.default_rng(0)
    masks = sample_kernel_coalitions(n, m, rng)
    sizes = masks.sum(1).astype(int)
    assert sizes.min() >= 1 and sizes.max() <= n - 1
    freq = np.bincount(sizes, minlength=n)[1:n] / m
    assert np.allclose(freq, kernel_size_probs(n), atol=0.01)
    assert np.allclose(masks.mean(0), 0.5, atol=0.01)               # every agent is present half the time
    paired = sample_kernel_coalitions(n, m, rng, paired=True)
    assert np.array_equal(paired[0::2] + paired[1::2], np.ones((m // 2, n), dtype=np.float32))
