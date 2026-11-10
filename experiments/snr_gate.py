"""Signal-to-noise gate (build step 2): which intervention window and branch length make responsibility measurable?

For each (window, horizon) setting this measures, on held-out decision points of four-agent Level-Based Foraging:
  - mean absolute responsibility (the signal), from all 16 groups with `reps` shared-random-number repetitions;
  - the standard error of that ground truth, relative to the signal (gate condition 1: below 0.10);
  - the relative error of a 32-branch permutation (Monte Carlo) estimate (gate condition 2: below 1.0, the zero
    predictor), using fresh branches that are independent of the ground truth;
  - the standard deviation of a single return and of a paired difference.
No critic is trained, so a full sweep is cheap. Condition 3 (the paired critic beats Monte Carlo at equal branches)
is tested afterwards with lbf_smoke.py on the settings that pass 1 and 2.

Run from the repository root:  python experiments/snr_gate.py --eval 60 --reps 400
"""
import argparse, itertools, json, sys, time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from accteams.branching import all_masks, make_branch_fn, shapley_from_mask_values
from lbf_smoke import FOOD, GRID, N, NUM_ACTIONS, T_COLLECT, collect, scripted_policy, take
from jumanji.environments.routing.lbf import env as lbfenv, generator


def shapley_weights(n):
    import math
    w = np.zeros((n, 2 ** n))
    for i in range(n):
        for b in range(2 ** n):
            if not (b >> i) & 1:
                s = bin(b).count("1")
                c = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                w[i, b | (1 << i)] += c
                w[i, b] -= c
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="1,3,5,10")
    ap.add_argument("--horizons", default="5,10,15,25")
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--mc-perms", type=int, default=4, help="permutations per agent; 2 branches each")
    ap.add_argument("--diagonal", action="store_true", help="only settings with window == horizon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/snr_gate.json")
    args = ap.parse_args()
    env = lbfenv.LevelBasedForaging(
        generator=generator.RandomGenerator(grid_size=GRID, fov=GRID, num_agents=N, num_food=FOOD, force_coop=True),
        time_limit=50)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    states, obs = collect(env, key, args.eval * 4 // T_COLLECT + 10)
    live = np.where(~np.asarray(states.food_items.eaten).all(1))[0]
    ev = rng.permutation(live)[: args.eval]
    s_ev, o_ev = take(states, ev), take(obs, ev)
    a_ev = jax.vmap(scripted_policy)(o_ev, jax.random.split(jax.random.fold_in(key, 1), len(ev)))
    W = shapley_weights(N)
    masks16 = jnp.asarray(np.repeat(all_masks(N), args.reps, axis=0))
    rows = []
    for window, horizon in itertools.product(map(int, args.windows.split(",")), map(int, args.horizons.split(","))):
        if window > horizon or (args.diagonal and window != horizon):
            continue
        t0 = time.time()
        branch = make_branch_fn(env.step, scripted_policy, jnp.zeros(N, jnp.int32), horizon=horizon, gamma=0.99,
                                window=window)
        crn = jax.jit(lambda s, a, k: jax.vmap(branch, in_axes=(None, None, 0, 0))(
            s, a, masks16, jnp.tile(jax.random.split(k, args.reps), (2 ** N, 1))))
        phi, se, mc, sd_ret, sd_pair = [], [], [], [], []
        for j in range(len(ev)):
            k = jax.random.fold_in(jax.random.PRNGKey(1000 + window * 100 + horizon), j)
            vals = np.asarray(crn(take(s_ev, j), a_ev[j], k)).reshape(2 ** N, args.reps)
            half = args.reps // 2
            # ground truth from the first half; Monte Carlo draws from the second half, so they are independent
            gt, fresh = vals[:, :half], vals[:, half:]
            phi.append(W @ gt.mean(1))
            parts = np.stack([W @ gt[:, b::10].mean(1) for b in range(10)])
            se.append(parts.std(0, ddof=1) / np.sqrt(10))
            sub = np.random.default_rng(j)
            est = np.zeros(N)
            for i in range(N):
                for _ in range(args.mc_perms):
                    perm = sub.permutation(N); pos = int(np.where(perm == i)[0][0])
                    b0 = sum(1 << int(q) for q in perm[:pos]); r = sub.integers(fresh.shape[1])
                    est[i] += (fresh[b0 | (1 << i), r] - fresh[b0, r]) / args.mc_perms
            mc.append(est)
            sd_ret.append(vals[-1].std())
            sd_pair.append((vals[-1] - vals[0]).std())
        phi, se, mc = map(np.array, (phi, se, mc))
        signal = float(np.mean(np.abs(phi)))
        row = dict(window=window, horizon=horizon, signal=signal,
                   ground_truth_se_rel=float(np.mean(se) / signal),
                   mc32_rel_err=float(np.mean(np.abs(mc - phi)) / signal),
                   return_sd=float(np.mean(sd_ret)), paired_difference_sd=float(np.mean(sd_pair)),
                   snr=float(signal / np.mean(sd_pair)),
                   seconds=round(time.time() - t0, 1))
        row["passes_1_and_2"] = bool(row["ground_truth_se_rel"] < 0.10 and row["mc32_rel_err"] < 1.0)
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(dict(args=vars(args), rows=rows), open(args.out, "w"), indent=1)
    print("\nwindow horizon  signal  gt_se  mc32  snr  pass")
    for r in rows:
        print(f"{r['window']:6d} {r['horizon']:7d}  {r['signal']:.4f}  {r['ground_truth_se_rel']:.2f}  "
              f"{r['mc32_rel_err']:.2f}  {r['snr']:.3f}  {'yes' if r['passes_1_and_2'] else 'no'}")


if __name__ == "__main__":
    main()
