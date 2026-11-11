import numpy as np
from accteams.stats import holm, iqm, permutation_test, prob_improvement, stratified_bootstrap_ci


def test_iqm_is_the_mean_of_the_middle_half():
    assert iqm(np.arange(8)) == 3.5
    assert abs(iqm([0, 0, 1, 2, 3, 100]) - 1.5) < 1e-12          # outlier ignored; fractional trimming
    assert abs(iqm(np.arange(5)) - 2.0) < 1e-12


def test_bootstrap_interval_covers_the_mean():
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(200):
        s = rng.normal(1.0, 1.0, size=(10, 3))
        _, lo, hi = stratified_bootstrap_ci(s, stat=np.mean, reps=300, seed=trial)
        hits += lo <= 1.0 <= hi
    assert 0.85 <= hits / 200 <= 0.99                             # nominal 0.95; percentile intervals run a little narrow


def test_prob_improvement():
    assert prob_improvement([3, 4], [1, 2]) == 1.0
    assert prob_improvement([1, 2], [1, 2]) == 0.5
    assert prob_improvement([0.2, 0.3], [0.5, 0.6], lower_is_better=True) == 1.0


def test_permutation_test_and_holm():
    rng = np.random.default_rng(1)
    assert permutation_test(rng.normal(0, 1, 10), rng.normal(3, 1, 10), reps=2000) < 0.01
    assert permutation_test(rng.normal(0, 1, 10), rng.normal(0, 1, 10), reps=2000) > 0.05
    adj, rej = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and rej.tolist() == [True, False, False]
