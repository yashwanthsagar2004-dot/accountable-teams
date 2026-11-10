"""Second round of checks for the Proofs tab (candidate A).
1. T1b general-k closed form for on-policy least squares (binary actions, product Bernoulli policy, anchor 0).
2. Lemma 2 + T2(iii): dividends of the mu-interventional game are mu-ANOVA components; mean-square truncation error.
3. T3: Neyman allocation and budget-normalised variance factors for k = 1..n.
4. T5: two teams with identical on-path data and different responsibility.
5. T7: strategies that differ only on a set E of adversary observations shift Shapley values by <= 2R max_S P_S(E)."""
import itertools, math, numpy as np
rng = np.random.default_rng(1)
subsets = lambda n, k=None: [frozenset(c) for r in range(0, (n if k is None else k) + 1) for c in itertools.combinations(range(n), r)]
def shapley(v, n):
    phi = np.zeros(n)
    for i in range(n):
        o = [j for j in range(n) if j != i]
        for r in range(n):
            for S in itertools.combinations(o, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                phi[i] += w * (v(frozenset(S) | {i}) - v(frozenset(S)))
    return phi

# ---------- 1. T1b closed form ----------
def fit_onpolicy(m, n, k, p):
    X = list(itertools.product((0, 1), repeat=n))
    w = np.array([np.prod([p[j] if x[j] else 1 - p[j] for j in range(n)]) for x in X])
    basis = subsets(n, k)
    A = np.array([[float(all(x[j] for j in T)) for T in basis] for x in X])
    y = np.array([sum(c for T, c in m.items() if all(x[j] for j in T)) for x in X])
    coef = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    return dict(zip(basis, coef))
def closed_form(m, n, k, p):
    out = {}
    for U in subsets(n, k):
        val = m.get(U, 0.0)
        for T, c in m.items():
            if len(T) > k and U < T:
                val += c * np.prod([p[j] for j in T - U]) * (-1) ** (k - len(U)) * math.comb(len(T) - len(U) - 1, k - len(U))
        out[U] = val
    return out
worst = 0
for trial in range(300):
    n = int(rng.integers(3, 7)); k = int(rng.integers(1, n)); p = rng.uniform(0.05, 0.95, n)
    m = {T: rng.normal() for T in subsets(n) if T}
    f, cf = fit_onpolicy(m, n, k, p), closed_form(m, n, k, p)
    worst = max(worst, max(abs(f[U] - cf[U]) for U in f if U))
print(f"1. T1b closed form vs least squares, 300 random games (n=3..6, all k<n): max abs diff {worst:.2e}")

# ---------- 2. Lemma 2 and T2(iii) ----------
def anova(Qtab, n, q):  # Qtab: dict x->value on {0,1}^n; q: P(x_j=1) under product reference mu
    X = list(itertools.product((0, 1), repeat=n))
    def marg(x, R):  # E over x'_{-R} ~ mu of Q(x_R, x'_{-R})
        tot = 0.0
        for xp in itertools.product((0, 1), repeat=n):
            z = tuple(x[j] if j in R else xp[j] for j in range(n))
            wt = np.prod([(q[j] if xp[j] else 1 - q[j]) if j not in R else 0.5 for j in range(n)])
            tot += wt * Qtab[z]
        return tot
    comps = {}
    for T in subsets(n):
        comps[T] = {x: sum((-1) ** (len(T) - len(R)) * marg(x, R) for r in range(len(T) + 1) for R in map(frozenset, itertools.combinations(sorted(T), r))) for x in X}
    return comps, marg
max_lemma, rows = 0, []
for trial in range(20):
    n = 4; q = rng.uniform(0.1, 0.9, n); X = list(itertools.product((0, 1), repeat=n))
    Qtab = {x: rng.normal() for x in X}
    comps, marg = anova(Qtab, n, q)
    # zero-mean property
    for T in subsets(n):
        for j in T:
            for x in X:
                x1 = tuple(1 if t == j else x[t] for t in range(n)); x0 = tuple(0 if t == j else x[t] for t in range(n))
                assert abs(q[j] * comps[T][x1] + (1 - q[j]) * comps[T][x0]) < 1e-9
    for k in (1, 2, 3):
        mse, pred = 0.0, 0.0
        for x in X:
            wx = np.prod([(q[j] if x[j] else 1 - q[j]) for j in range(n)])
            v = lambda S, x=x: marg(x, S)
            phi = shapley(v, n)
            for T in subsets(n):
                max_lemma = max(max_lemma, abs(sum((-1) ** (len(T) - len(R)) * v(R) for r in range(len(T) + 1) for R in map(frozenset, itertools.combinations(sorted(T), r))) - comps[T][x]))
            ro = np.array([sum(comps[T][x] / len(T) for T in subsets(n, k) if i in T and T) for i in range(n)])
            mse += wx * (ro[0] - phi[0]) ** 2
            pred += wx * sum(comps[T][x] ** 2 / len(T) ** 2 for T in subsets(n) if 0 in T and len(T) > k)
        rows.append((k, mse, pred))
print(f"2. Lemma 2: max |dividend - mu-ANOVA component| = {max_lemma:.1e}; T2(iii) mean-square error vs formula: max rel diff {max(abs(a-b)/max(b,1e-12) for _,a,b in rows):.1e}")

# ---------- 3. T3 allocation and budget ----------
def coefs(n, k, i=0):
    return {R: sum((-1) ** (len(T) - len(R)) / len(T) for T in subsets(n, k) if T and i in T and R <= T) for R in subsets(n, k)}
print("3. Budget-normalised variance factor (sigma^2 / total branches B):  uniform = groups x C(n,k);  Neyman = (sum |c|)^2")
for n in (2, 3, 4, 5, 6, 8):
    cells = []
    for k in sorted(set([1, 2, 3, n])):
        if k > n: continue
        c = coefs(n, k); C = sum(v * v for v in c.values()); G = sum(1 for v in c.values() if abs(v) > 1e-12)
        cells.append(f"k={k}: {G*C:7.2f} / {sum(abs(v) for v in c.values())**2:7.2f}")
    print(f"   n={n}: " + " | ".join(cells))

# ---------- 4. T5 construction ----------
# Agent 1 acts at h0 (on-path always a1=1); agent 2 then acts seeing a1. Reference for agent 1: a1=0.
for plan, name in ((1, 'compensate'), (0, 'fail')):
    v_empty = plan            # a1 replaced by 0: success iff agent 2's off-path plan compensates
    v_one = 1                 # a1 = 1: agent 2 plays 1 on path: success
    print(f"4. T5 team with off-path plan '{name}': on-path data identical (a1=1, a2=1, success); responsibility of agent 1 = {v_one - v_empty}")

# ---------- 5. T7 with a set E ----------
worst_ratio = 0.0
for trial in range(2000):
    n = 3; O = 6
    P = {S: rng.dirichlet(np.ones(O)) for S in subsets(n)}           # adversary's observation law in each branch
    base = {S: rng.uniform(0, 1, O) for S in subsets(n)}               # outcome given observation under strategy sigma
    E = rng.random(O) < 0.4
    alt = {S: np.where(E, rng.uniform(0, 1, O), base[S]) for S in subsets(n)}  # sigma' differs only on E
    v1 = lambda S: float(P[S] @ base[S]); v2 = lambda S: float(P[S] @ alt[S])
    shift = np.max(np.abs(shapley(v1, n) - shapley(v2, n)))
    bound = 2 * 1.0 * max(float(P[S][E].sum()) for S in subsets(n))
    if bound > 0: worst_ratio = max(worst_ratio, shift / bound)
print(f"5. T7 set version: max shift / (2R max_S P_S(E)) over 2000 random instances = {worst_ratio:.3f} (must be <= 1)")
