"""Exact and sampled Shapley values, Moebius (Harsanyi) dividends and the Shapley kernel.

All functions take a coalition-value function v(S) where S is a frozenset of agent indices.
"""
from __future__ import annotations
import itertools
import math
from typing import Callable, Dict, FrozenSet, Iterable

import numpy as np

Coalition = FrozenSet[int]


def all_subsets(n: int, max_size: int | None = None) -> list[Coalition]:
    top = n if max_size is None else max_size
    return [frozenset(c) for r in range(top + 1) for c in itertools.combinations(range(n), r)]


def shapley_exact(v: Callable[[Coalition], float], n: int) -> np.ndarray:
    """Exact Shapley values by enumeration (fine for n <= 10)."""
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for s in itertools.combinations(others, r):
                s = frozenset(s)
                phi[i] += w * (v(s | {i}) - v(s))
    return phi


def dividends(v: Callable[[Coalition], float], n: int, max_order: int | None = None) -> Dict[Coalition, float]:
    """Harsanyi dividends m(T) = sum_{R subset T} (-1)^{|T|-|R|} v(R), for |T| <= max_order."""
    out = {}
    for t in all_subsets(n, max_order):
        out[t] = sum((-1) ** (len(t) - len(r)) * v(frozenset(r))
                     for k in range(len(t) + 1) for r in itertools.combinations(sorted(t), k))
    return out


def readout_from_dividends(div: Dict[Coalition, float], n: int, max_order: int | None = None) -> np.ndarray:
    """phi_i = sum over groups T containing i (|T| <= max_order) of m(T)/|T|."""
    phi = np.zeros(n)
    for t, c in div.items():
        if len(t) == 0 or (max_order is not None and len(t) > max_order):
            continue
        for i in t:
            phi[i] += c / len(t)
    return phi


def kernel_size_probs(n: int) -> np.ndarray:
    """Distribution over coalition sizes s = 1..n-1 induced by the Shapley kernel.

    The kernel weight of one coalition of size s is (n-1) / (C(n,s) s (n-s)); summing over the
    C(n,s) coalitions of that size gives a size weight proportional to 1 / (s (n-s)).
    """
    s = np.arange(1, n)
    w = 1.0 / (s * (n - s))
    return w / w.sum()


def sample_kernel_coalitions(n: int, m: int, rng: np.random.Generator, paired: bool = False) -> np.ndarray:
    """m coalition masks (shape m x n, 0/1) drawn from the Shapley kernel over proper, non-empty coalitions.

    paired=True returns each sampled coalition together with its complement (rows 2i and 2i+1). The kernel is
    symmetric in s and n - s, so the complement has the same distribution; pairing lowers the variance of
    kernel estimates (Covert and Lee 2021) and is the setting FastSHAP uses.
    """
    draws = (m + 1) // 2 if paired else m
    sizes = rng.choice(np.arange(1, n), size=draws, p=kernel_size_probs(n))
    masks = np.zeros((draws, n), dtype=np.float32)
    for row, s in enumerate(sizes):
        masks[row, rng.choice(n, size=s, replace=False)] = 1.0
    if paired:
        masks = np.stack([masks, 1.0 - masks], axis=1).reshape(2 * draws, n)[:m]
    return masks


def shapley_permutation(v: Callable[[Coalition], float], n: int, i: int, m: int, rng: np.random.Generator) -> float:
    """Unbiased permutation-sampling estimate of agent i's Shapley value (2 evaluations per sample)."""
    tot = 0.0
    for _ in range(m):
        perm = rng.permutation(n)
        pos = int(np.where(perm == i)[0][0])
        s = frozenset(perm[:pos].tolist())
        tot += v(s | {i}) - v(s)
    return tot / m


def kernel_least_squares(v: Callable[[Coalition], float], n: int) -> np.ndarray:
    """Shapley values as the constrained, kernel-weighted least-squares fit of an additive coalition model.

    Exact for every game (Lundberg and Lee 2017, Theorem 2); used here as a check and as a teaching tool.
    """
    rows, y, w = [], [], []
    v0, vn = v(frozenset()), v(frozenset(range(n)))
    for s in all_subsets(n):
        if 0 < len(s) < n:
            rows.append([1.0 if i in s else 0.0 for i in range(n)])
            y.append(v(s) - v0)
            w.append((n - 1) / (math.comb(n, len(s)) * len(s) * (n - len(s))))
    a, y, w = np.array(rows), np.array(y), np.array(w)
    kkt = np.block([[2 * a.T @ (w[:, None] * a), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    rhs = np.concatenate([2 * a.T @ (w * y), [vn - v0]])
    return np.linalg.solve(kkt, rhs)[:n]
