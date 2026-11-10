"""Numerical checks for candidate A theory (T1-T4, T6/T7, audit budget).
Binary actions, anchor (reference) b = 0. A team value Q on {0,1}^n is written in its
anchored (Harsanyi / multilinear) form Q(x) = sum_T m(T) prod_{j in T} x_j.
For realized action a, the interventional game is v_a(S) = Q(a_S, b_-S) and its dividends
are m(T) * prod_{j in T} a_j."""
import itertools, math, numpy as np
rng = np.random.default_rng(0)

def subsets(n, maxk=None):
    out = []
    for r in range(0, n + 1):
        if maxk is not None and r > maxk: break
        out += [frozenset(c) for c in itertools.combinations(range(n), r)]
    return out

def Q_from_div(m, x):
    return sum(c * all(x[j] for j in T) for T, c in m.items())

def v_game(m, a, S):  # Q(a_S, 0_-S)
    x = [a[j] if j in S else 0 for j in range(len(a))]
    return Q_from_div(m, x)

def shapley_exact(vfun, n):
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                phi[i] += w * (vfun(frozenset(S) | {i}) - vfun(frozenset(S)))
    return phi

def mobius(vfun, T):
    return sum((-1) ** (len(T) - len(R)) * vfun(frozenset(R)) for r in range(len(T) + 1) for R in itertools.combinations(sorted(T), r))

def readout(div, n, k):  # order-k Shapley readout from dividends
    phi = np.zeros(n)
    for T, c in div.items():
        if 1 <= len(T) <= k:
            for i in T: phi[i] += c / len(T)
    return phi

def random_game(n, maxorder, scale=1.0):
    return {T: rng.normal(0, scale) for T in subsets(n, maxorder) if len(T) >= 1}

print("== T1(i) interventional small-coalition design gives exact dividends; T2 tail ==")
worst_t2_slack = np.inf; checks = 0
for trial in range(200):
    n = rng.integers(3, 7); k = rng.integers(1, n)
    m = random_game(n, n); a = [1] * n
    vf = lambda S: v_game(m, a, S)
    phi = shapley_exact(vf, n)
    est = {T: mobius(vf, T) for T in subsets(n, k) if len(T) >= 1}   # uses only v(R), |R|<=k
    for T, c in est.items():
        assert abs(c - m[T]) < 1e-9, "dividend mismatch"
    ro = readout(est, n, k)
    tail = np.array([sum(m[T] / len(T) for T in m if i in T and len(T) > k) for i in range(n)])
    bound = np.array([sum(abs(m[T]) for T in m if i in T and len(T) > k) / (k + 1) for i in range(n)])
    assert np.allclose(phi - ro, tail), "error != tail"
    assert np.all(np.abs(phi - ro) <= bound + 1e-12), "T2 bound violated"
    checks += 1
print(f"  {checks} random games (n=3..6, k<n): low-order dividends exact; error == tail; |error| <= tail mass/(k+1): OK")

print("== T1(ii) on-policy least squares mixes high-order terms with policy weights ==")
def onpolicy_fit(m, n, k, p):
    X = list(itertools.product((0, 1), repeat=n)); w = np.array([np.prod([p[j] if x[j] else 1 - p[j] for j in range(n)]) for x in X])
    basis = [T for T in subsets(n, k)]
    A = np.array([[1.0 if all(x[j] for j in T) else 0.0 for T in basis] for x in X]); y = np.array([Q_from_div(m, x) for x in X])
    W = np.diag(w); coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    return {T: c for T, c in zip(basis, coef) if len(T) >= 1}
n = 3; m = {frozenset([0]): .3, frozenset([1]): .2, frozenset([2]): .1, frozenset([0, 1, 2]): 1.5}
a = [1, 1, 1]; phi = shapley_exact(lambda S: v_game(m, a, S), n)
for p in ([.9, .9, .9], [.9, .5, .2], [.2, .2, .9]):
    fit1 = onpolicy_fit(m, n, 1, p); ro1 = readout(fit1, n, 1)
    pred = np.array([m[frozenset([i])] + 1.5 * np.prod([p[j] for j in range(n) if j != i]) for i in range(n)])
    fit2 = onpolicy_fit(m, n, 2, p); ro2 = readout(fit2, n, 2)
    print(f"  p={p}: k=1 readout-Shapley={np.round(ro1 - phi, 4)} (closed form {np.round(pred - phi, 4)}); k=2 readout-Shapley={np.round(ro2 - phi, 4)}")
    assert np.allclose(ro1, pred)
print("  k=1 matches P1's Proposition 2 exactly; k=2 bias still depends on p (not identified as responsibility)")

print("== T1(iii) gauge freedom without anchoring ==")
# Q(a1,a2) = g1(a1)+g2(a2)+g12(a1,a2). Move h(a1)=c*a1 from g1 into g12: same Q, readout shifts.
g1 = {0: 0., 1: .5}; g2 = {0: 0., 1: .3}; g12 = lambda x1, x2: .4 * x1 * x2
c = .6
Qa = lambda x1, x2: g1[x1] + g2[x2] + g12(x1, x2)
Qb = lambda x1, x2: (g1[x1] - c * x1) + g2[x2] + (g12(x1, x2) + c * x1)
assert all(abs(Qa(x1, x2) - Qb(x1, x2)) < 1e-12 for x1 in (0, 1) for x2 in (0, 1))
ro_a = np.array([g1[1] + g12(1, 1) / 2, g2[1] + g12(1, 1) / 2])
ro_b = np.array([(g1[1] - c) + (g12(1, 1) + c) / 2, g2[1] + (g12(1, 1) + c) / 2])
print(f"  identical Q, readouts {ro_a} vs {ro_b}: agent 1 shifts by {ro_b[0]-ro_a[0]:+.2f}, agent 2 by {ro_b[1]-ro_a[1]:+.2f} (= -c/2, +c/2)")

print("== T3 Mobius estimator: unbiased, variance = sigma^2/m * sum_R c_iR^2 ==")
def coef_table(n, k, i):
    cs = {}
    for R in subsets(n, k):
        cs[R] = sum((-1) ** (len(T) - len(R)) / len(T) for T in subsets(n, k) if i in T and R <= T and len(T) >= 1)
    return cs
for (n, k) in [(4, 2)]:
    m = random_game(n, n); a = [1] * n; i = 0
    true = readout({T: mobius(lambda S: v_game(m, a, S), T) for T in subsets(n, k) if T}, n, k)[i]
    cs = coef_table(n, k, i); sigma, mm, reps = 1.0, 20, 20000
    vals = np.array([v_game(m, a, R) for R in cs]); cvec = np.array(list(cs.values()))
    est = (vals[None, :] + rng.normal(0, sigma / math.sqrt(mm), (reps, len(cs)))) @ cvec
    print(f"  n={n},k={k}: mean {est.mean():.4f} vs target {true:.4f}; var {est.var():.5f} vs formula {sigma**2/mm*np.sum(cvec**2):.5f}")
print("  C(n,k)=sum_R c_iR^2 and coalitions N_k=sum_{t<=k} C(n,t):")
for n in (2, 3, 4, 5, 6, 8):
    row = []
    for k in (1, 2, 3):
        if k > n: row.append('   -   '); continue
        cs = coef_table(n, k, 0); row.append(f"{sum(v*v for v in cs.values()):6.2f}/{len(cs)-0:3d}")
    print(f"  n={n}: " + "  ".join(row))

print("== T4 weak/strong disparity: additive on-policy credit vs order-k interventional ==")
for kappa in (1.0, -1.0):
    for pw, ps in ((.2, .9), (.5, .9), (.8, .9)):
        m = {frozenset([0]): .3, frozenset([1]): .3, frozenset([0, 1]): kappa}; a = [1, 1]
        phi = shapley_exact(lambda S: v_game(m, a, S), 2)
        ro = readout(onpolicy_fit(m, 2, 1, [pw, ps]), 2, 1)
        err = ro - phi; disp = err[0] - err[1]   # agent 0 weak, agent 1 strong
        print(f"  kappa={kappa:+.0f} p_weak={pw} p_strong={ps}: error weak {err[0]:+.3f}, strong {err[1]:+.3f}, gap {disp:+.3f} (formula kappa*(p_s-p_w)={kappa*(ps-pw):+.3f}); order-2 interventional gap 0")

print("== T6/T7 conditional sabotage: blame shift vs detectability delta ==")
def blame(delta):   # agents: X honest (0), A adversary (1); outcome = failure indicator
    v = {frozenset(): 0., frozenset([0]): 0., frozenset([1]): 1 - delta, frozenset([0, 1]): 1.}
    return shapley_exact(lambda S: v[frozenset(S)], 2)
for d in (0., .1, .5, 1.):
    b = blame(d); print(f"  delta={d:.1f}: blame X={b[0]:.3f}, A={b[1]:.3f}; shift onto X = {b[0]:.3f} <= 2*R*delta = {2*d:.3f}")

print("== Audit budget: permutation-sampling Shapley, Hoeffding coverage ==")
n = 5; m = random_game(n, n, .5); a = [1] * n; vf = lambda S: v_game(m, a, S); phi = shapley_exact(vf, n)
R = max(abs(vf(frozenset(S) | {0}) - vf(frozenset(S))) for S in subsets(n) if 0 not in S)
eps, eta = .1, .05; mreq = math.ceil(2 * R * R * math.log(2 / eta) / eps ** 2)
miss = 0; reps = 2000
for _ in range(reps):
    tot = 0.
    for _ in range(mreq):
        perm = rng.permutation(n); pos = int(np.where(perm == 0)[0][0]); S = frozenset(perm[:pos].tolist())
        tot += vf(S | {0}) - vf(S)
    miss += abs(tot / mreq - phi[0]) > eps
print(f"  range R={R:.3f}, eps={eps}, eta={eta}: m={mreq} permutations; miss rate {miss/reps:.4f} <= {eta}")
