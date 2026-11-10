"""BlameBench tier 1: planted team games with known responsibility.

A planted contextual game has n agents with binary actions (1 = act, 0 = reference/no-op) and a context
vector h. Its team value is multilinear with context-dependent dividends:

    Q(h, x) = sum over groups T of m_T(h) * prod_{j in T} x_j,

so with reference action 0 the interventional game at realised actions a is v(S) = Q(h, a_S, 0_{-S}) and the
true responsibility is phi_i = sum_{T containing i} m_T(h) prod_{j in T} a_j / |T|.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .shapley import all_subsets


@dataclass
class PlantedGame:
    n: int
    d: int
    groups: list            # list of frozensets with a planted dividend
    weights: np.ndarray     # (len(groups), d) linear maps h -> dividend
    bias: np.ndarray        # (len(groups),)
    pol_w: np.ndarray       # (n, d) policy logits weights
    pol_b: np.ndarray       # (n,) policy logits bias (sets weak/strong agents)
    noise: float = 0.5      # std of return noise

    @staticmethod
    def make(n=4, d=6, max_order=3, scale=(1.0, 0.8, 0.8), competences=None, noise=0.5, seed=0):
        rng = np.random.default_rng(seed)
        groups = [t for t in all_subsets(n, max_order) if len(t) >= 1]
        sc = np.array([scale[len(t) - 1] for t in groups])
        weights = rng.normal(0, 1, (len(groups), d)) * sc[:, None] / np.sqrt(d)
        bias = rng.normal(0, 1, len(groups)) * sc
        comp = np.linspace(-1.5, 1.5, n) if competences is None else np.asarray(competences)
        return PlantedGame(n, d, groups, weights, bias, rng.normal(0, 0.5, (n, d)), comp, noise)

    def dividends(self, h: np.ndarray) -> np.ndarray:            # (batch, len(groups))
        return np.tanh(h @ self.weights.T + self.bias)

    def policy_probs(self, h: np.ndarray) -> np.ndarray:         # (batch, n)
        return 1.0 / (1.0 + np.exp(-(h @ self.pol_w.T + self.pol_b)))

    def value(self, h: np.ndarray, x: np.ndarray) -> np.ndarray:  # Q(h, x), x in {0,1}^n, batched
        m = self.dividends(h)
        on = np.stack([np.prod(x[:, sorted(t)], axis=1) for t in self.groups], axis=1)
        return (m * on).sum(1)

    def coalition_value(self, h, a, mask):                       # v(S) at realised a with mask = 1 for S
        return self.value(h, a * mask)

    def responsibility(self, h, a) -> np.ndarray:                # exact Shapley, (batch, n)
        m = self.dividends(h)
        phi = np.zeros((h.shape[0], self.n))
        for g, t in enumerate(self.groups):
            on = np.prod(a[:, sorted(t)], axis=1)
            for i in t:
                phi[:, i] += m[:, g] * on / len(t)
        return phi

    def sample_decisions(self, num: int, rng: np.random.Generator):
        h = rng.normal(0, 1, (num, self.d))
        p = self.policy_probs(h)
        a = (rng.random(p.shape) < p).astype(np.float32)
        return h.astype(np.float32), a, p

    def branch_return(self, h, a, mask, rng):                    # one executed branch: noisy return
        return self.coalition_value(h, a, mask) + rng.normal(0, self.noise, h.shape[0])
