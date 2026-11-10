"""Third round of checks for the Proofs tab (round 6).

1. T3(4): noise factor of the shared Shapley-kernel design, sum_S c_S^2 / p(S) = 2 (n-1) H_{n-1} / n,
   checked against enumeration and against a simulated kernel-weighted least-squares fit (stratified design),
   plus the random-design version for an additive and a non-additive game.
2. T3(5): variance of permutation audits, (2 Var[Delta_i] + 4 sigma^2) / B.
3. T4(2): weak-minus-strong error gap of the additive on-policy fit for one interaction on a group T.
4. T5 remark: Le Cam's two-point bound for logged data with exploration rate eps.
Run from the repository root:  python theory/check_theory_A3.py
"""
import itertools
import math

import numpy as np

rng = np.random.default_rng(3)


def subsets(n, k=None):
    top = n if k is None else k
    return [frozenset(c) for r in range(top + 1) for c in itertools.combinations(range(n), r)]


def shapley_coef(n, i, S):
    s = len(S)
    if i in S:
        return math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
    return -math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def kernel_p(n):
    """Probability of each proper non-empty coalition under the normalised Shapley kernel."""
    props = [S for S in subsets(n) if 0 < len(S) < n]
    w = np.array([(n - 1) / (math.comb(n, len(S)) * len(S) * (n - len(S))) for S in props])
    return props, w / w.sum()


def wls_shapley(n, props, vals, weights, v0, vn):
    """Constrained weighted least squares of v(S) - v0 on membership indicators (Lemma 3)."""
    A = np.array([[1.0 if i in S else 0.0 for i in range(n)] for S in props])
    y = np.asarray(vals) - v0
    kkt = np.block([[2 * A.T @ (weights[:, None] * A), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    rhs = np.concatenate([2 * A.T @ (weights * y), [vn - v0]])
    return np.linalg.solve(kkt, rhs)[:n]


# ---------- 1. kernel noise factor ----------
print("1. Shared kernel design: per-agent noise factor sum_S c_S^2/p(S) (proper coalitions)")
for n in range(2, 9):
    props, p = kernel_p(n)
    enum = sum(shapley_coef(n, 0, S) ** 2 / pS for S, pS in zip(props, p))
    formula = 2 * (n - 1) * sum(1 / t for t in range(1, n)) / n
    best = sum(1 / math.sqrt(s * (n - s)) for s in range(1, n)) ** 2
    print(f"   n={n}: enumeration {enum:.4f}  formula {formula:.4f}  best symmetric design {best:.4f}")

n, sigma, B, reps = 4, 1.0, 20000, 4000
props, p = kernel_p(n)
games = {
    'non-additive': {S: rng.normal() for S in subsets(n)},
    'additive': None,
}
w_add = rng.normal(size=n)
games['additive'] = {S: float(sum(w_add[j] for j in S)) for S in subsets(n)}
F = 2 * (n - 1) * sum(1 / t for t in range(1, n)) / n
for name, v in games.items():
    v0, vn = v[frozenset()], v[frozenset(range(n))]
    truth = wls_shapley(n, props, [v[S] for S in props], p, v0, vn)
    est = []
    for _ in range(reps):                       # stratified: m_S = B p(S) branches, mean has variance sigma^2/m_S
        noisy = [v[S] + rng.normal(0, sigma / math.sqrt(B * pS)) for S, pS in zip(props, p)]
        est.append(wls_shapley(n, props, noisy, p, v0, vn)[0])
    strat = np.var(est) * B / sigma ** 2
    est = []
    for _ in range(reps // 4):                  # random design: B coalitions drawn from the kernel, one return each
        idx = rng.choice(len(props), size=B, p=p)
        counts = np.bincount(idx, minlength=len(props)).astype(float)
        sums = np.array([v[props[j]] for j in range(len(props))]) * counts + rng.normal(0, sigma, len(props)) * np.sqrt(counts)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        est.append(wls_shapley(n, props, means, counts / B, v0, vn)[0])
    rand = np.var(est) * B / sigma ** 2
    print(f"   n=4 {name:12s}: B*Var/sigma^2 stratified {strat:.3f} (formula {F:.3f}); random design {rand:.3f}")

# ---------- 2. permutation audits ----------
n, sigma, B, reps = 5, 0.7, 2000, 4000
v = {S: rng.normal() for S in subsets(n)}
deltas, probs = [], []
for S in subsets(n):
    if 0 not in S:
        deltas.append(v[S | {0}] - v[S])
        probs.append(math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n))
deltas, probs = np.array(deltas), np.array(probs)
var_delta = probs @ deltas ** 2 - (probs @ deltas) ** 2
others = [j for j in range(1, n)]
est = []
for _ in range(reps):
    tot = 0.0
    for _ in range(B // 2):
        perm = rng.permutation(n)
        pos = int(np.where(perm == 0)[0][0])
        S = frozenset(perm[:pos].tolist())
        tot += v[S | {0}] - v[S] + rng.normal(0, sigma) - rng.normal(0, sigma)
    est.append(tot / (B // 2))
print(f"2. Permutation audits (n=5, B={B}): empirical variance {np.var(est):.5f}, formula (2Var+4sigma^2)/B = {(2 * var_delta + 4 * sigma ** 2) / B:.5f}")


# ---------- 3. T4(2) ----------
def fit_additive(m, n, p):
    X = list(itertools.product((0, 1), repeat=n))
    w = np.array([np.prod([p[j] if x[j] else 1 - p[j] for j in range(n)]) for x in X])
    A = np.array([[1.0, *x] for x in X])
    y = np.array([sum(c for T, c in m.items() if all(x[j] for j in T)) for x in X])
    return np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))[1:]


worst = 0.0
for trial in range(300):
    n = int(rng.integers(3, 7)); p = rng.uniform(0.05, 0.95, n); w, s = 0, n - 1
    extra = [j for j in range(1, n - 1) if rng.random() < 0.5]
    T = frozenset([w, s, *extra]); kappa = rng.normal()
    m = {frozenset([j]): rng.normal() for j in range(n)}
    m[T] = kappa
    ghat = fit_additive(m, n, p)
    phi = np.array([m[frozenset([j])] + (kappa / len(T) if j in T else 0.0) for j in range(n)])
    err = ghat - phi
    pred = kappa * np.prod([p[j] for j in T - {w, s}]) * (p[s] - p[w])
    worst = max(worst, abs((err[w] - err[s]) - pred))
print(f"3. T4(2) gap formula vs least-squares fit, 300 random cases (n=3..6): max abs diff {worst:.1e}")

# ---------- 4. T5 remark ----------
print("4. T5 with exploration: lower bound (1-eps)^N / 2 on the larger error probability")
for eps in (0.01, 0.05):
    for N in (10, 100, 1000):
        print(f"   eps={eps}, N={N}: {(1 - eps) ** N / 2:.3f}")
