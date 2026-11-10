"""Tier-1 pilot: does training on executed branches recover known responsibility better than on-policy fitting?

Run:  python -m accteams.train_tier1 --seeds 3
Each method gets the same number of executed returns. On-policy returns are free in RL, branches are not, so
the kernel critic and the pair critic spend their whole budget on branches, while the additive baseline gets
the same number of on-policy returns.
"""
from __future__ import annotations
import argparse
import json
import time

import jax
import numpy as np

from .critic import AdditiveOnPolicy, KernelCritic, PairCritic, fit
from .games import PlantedGame
from .shapley import sample_kernel_coalitions


def kernel_dataset(game, num, rng):
    h, a, _ = game.sample_decisions(num, rng)
    kind = rng.choice(3, size=num, p=[0.1, 0.1, 0.8])               # empty, full team, kernel coalition
    mask = sample_kernel_coalitions(game.n, num, rng)
    mask[kind == 0] = 0.0
    mask[kind == 1] = 1.0
    g = game.branch_return(h, a, mask, rng).astype(np.float32)
    return h, a, mask, g, kind.astype(np.int32)


def small_dataset(game, num, rng, k=2):
    h, a, _ = game.sample_decisions(num, rng)
    sizes = rng.integers(0, k + 1, num)
    mask = np.zeros((num, game.n), np.float32)
    for row, s in enumerate(sizes):
        mask[row, rng.choice(game.n, size=s, replace=False)] = 1.0
    g = game.branch_return(h, a, mask, rng).astype(np.float32)
    return h, a, mask, g


def onpolicy_dataset(game, num, rng):
    h, a, _ = game.sample_decisions(num, rng)
    g = game.branch_return(h, a, np.ones_like(a), rng).astype(np.float32)
    return h, a, g


def evaluate(game, model, rng, num=4000):
    h, a, p = game.sample_decisions(num, rng)
    phi = game.responsibility(h, a)
    est = model.readout(h, a)
    err = est - phi
    scale = np.mean(np.abs(phi)) + 1e-9
    return dict(mae=float(np.mean(np.abs(err)) / scale),
                gap=float(np.mean(err[:, 0]) - np.mean(err[:, -1])))     # agent 0 weakest, agent n-1 strongest


def run(seed, budget, n, max_order, steps):
    game = PlantedGame.make(n=n, max_order=max_order, seed=seed)
    rng = np.random.default_rng(seed + 100)
    key = jax.random.PRNGKey(seed)
    out = {}
    m = KernelCritic(n, game.d, key=key)
    fit(m, KernelCritic.loss, kernel_dataset(game, budget, rng), steps=steps, seed=seed, n=n)
    out['kernel critic (branches)'] = evaluate(game, m, rng)
    m = PairCritic(n, game.d, key=key)
    fit(m, PairCritic.loss, small_dataset(game, budget, rng), steps=steps, seed=seed, n=n, pairs=m.pairs)
    out['order-2 critic (small groups)'] = evaluate(game, m, rng)
    m = AdditiveOnPolicy(n, game.d, key=key)
    fit(m, AdditiveOnPolicy.loss, onpolicy_dataset(game, budget, rng), steps=steps, seed=seed, n=n)
    out['additive, on-policy (VDN-like)'] = evaluate(game, m, rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--budget', type=int, default=20000)
    ap.add_argument('--agents', type=int, default=4)
    ap.add_argument('--max-order', type=int, default=3)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--out', default='tier1_results.json')
    args = ap.parse_args()
    t0 = time.time()
    res = [run(s, args.budget, args.agents, args.max_order, args.steps) for s in range(args.seeds)]
    summary = {}
    for name in res[0]:
        for metric in ('mae', 'gap'):
            vals = np.array([r[name][metric] for r in res])
            summary.setdefault(name, {})[metric] = (float(vals.mean()), float(vals.std()))
    json.dump(dict(args=vars(args), runs=res, summary=summary), open(args.out, 'w'), indent=1)
    for name, s in summary.items():
        print(f"{name:34s} relative error {s['mae'][0]:.3f} +/- {s['mae'][1]:.3f} | weak-strong gap {s['gap'][0]:+.3f} +/- {s['gap'][1]:.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
