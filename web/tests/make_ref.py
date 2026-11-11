"""Reference values from the repo's own Python, for the JavaScript core's cross-check."""
import itertools, json, math, sys
import numpy as np
sys.path.insert(0, "src")
from accteams.shapley import shapley_exact, dividends, kernel_least_squares
rng = np.random.default_rng(7)
cases = []
for n in (2, 3, 4, 5):
    for _ in range(3):
        m = {frozenset(c): float(rng.normal()) for r in range(1, n + 1) for c in itertools.combinations(range(n), r)}
        v = lambda S: sum(c for T, c in m.items() if T <= S)
        p = rng.uniform(0.05, 0.95, n)
        X = list(itertools.product((0, 1), repeat=n))
        w = np.array([np.prod([p[j] if x[j] else 1 - p[j] for j in range(n)]) for x in X])
        A = np.array([[1.0, *x] for x in X]); y = np.array([v(frozenset(j for j in range(n) if x[j])) for x in X])
        ols = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))[1:]
        mask = lambda T: sum(1 << j for j in T)
        cases.append(dict(n=n, m={str(mask(T)): c for T, c in m.items()}, p=p.tolist(),
                          values=[v(frozenset(j for j in range(n) if s >> j & 1)) for s in range(1 << n)],
                          shapley=shapley_exact(v, n).tolist(), kernel_ls=kernel_least_squares(v, n).tolist(),
                          onpolicy=ols.tolist(),
                          dividends={str(mask(T)): c for T, c in dividends(v, n).items() if T}))
json.dump(cases, open("web/tests/ref.json", "w"))
print(len(cases), "cases")
