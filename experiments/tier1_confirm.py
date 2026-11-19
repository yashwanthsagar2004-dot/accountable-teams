"""H1 on planted games, confirmatory form: more seeds, per-seed records, IQM with stratified bootstrap intervals,
a paired permutation test of executed-branch critic against the additive on-policy model, Holm across budgets.

    python experiments/tier1_confirm.py --seeds 10 --agents 4 --order 3 --budgets 20000 80000
"""
import argparse, json, sys, time
import numpy as np
sys.path.insert(0, "src")
from accteams.provenance import stamp
from accteams.stats import holm, iqm, stratified_bootstrap_ci
from accteams.train_tier1 import run

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--agents", type=int, default=4)
ap.add_argument("--order", type=int, default=3)
ap.add_argument("--budgets", type=int, nargs="+", default=[20000, 80000])
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--out", default="experiments/tier1_confirm.json")
args = ap.parse_args()


def paired_sign_flip(x, y, reps=20000, seed=0):
    """Two-sided p-value for a mean paired difference: flip the sign of each seed's difference at random."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(reps, d.size))
    null = np.abs((flips * d).mean(1))
    return float((1 + np.sum(null >= abs(d.mean()) - 1e-12)) / (reps + 1))


t0, per_seed, summary, pvals = time.time(), [], {}, {}
for budget in args.budgets:
    for s in range(args.seeds):
        r = run(s, budget, args.agents, args.order, args.steps)
        for name, v in r.items():
            per_seed.append(dict(budget=budget, seed=s, method=name, mae=float(v["mae"]), gap=float(v["gap"])))
        print(f"budget {budget} seed {s} " + " | ".join(f"{k}: {v['mae']:.3f}" for k, v in r.items()), flush=True)
    methods = sorted({p["method"] for p in per_seed if p["budget"] == budget})
    for m in methods:
        e = np.array([p["mae"] for p in per_seed if p["budget"] == budget and p["method"] == m])
        g = np.array([p["gap"] for p in per_seed if p["budget"] == budget and p["method"] == m])
        est, lo, hi = stratified_bootstrap_ci(e, stat=iqm, reps=4000)
        summary[f"{budget}/{m}"] = dict(iqm=est, ci95=[lo, hi], mean=float(e.mean()), gap_mean=float(g.mean()),
                                         gap_sd=float(g.std(ddof=1)), seeds=len(e))
    k = [m for m in methods if m.startswith("kernel")][0]
    a = [m for m in methods if m.startswith("additive")][0]
    ek = np.array([p["mae"] for p in per_seed if p["budget"] == budget and p["method"] == k])
    ea = np.array([p["mae"] for p in per_seed if p["budget"] == budget and p["method"] == a])
    pvals[budget] = paired_sign_flip(ek, ea)
adj = holm(list(pvals.values()))[0]
tests = {str(b): dict(p=pvals[b], p_holm=float(q), kernel_better_on_seeds=int(sum(
    x < y for x, y in zip([p["mae"] for p in per_seed if p["budget"] == b and p["method"].startswith("kernel")],
                          [p["mae"] for p in per_seed if p["budget"] == b and p["method"].startswith("additive")]))))
         for b, q in zip(pvals, adj)}
json.dump(dict(args=vars(args), per_seed=per_seed, summary=summary, tests=tests, seconds=round(time.time() - t0),
               provenance=stamp(vars(args))), open(args.out, "w"), indent=1)
print(json.dumps(tests, indent=1)); print(f"done in {time.time() - t0:.0f}s")
