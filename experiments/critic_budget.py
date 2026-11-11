"""Accuracy against audit budget: does an amortised responsibility critic save executed branches, and where?

For one task and team size this trains, on ONE executed branch per training decision,
  - the flat KernelCritic (an MLP on concatenated features; the 10 Nov baseline),
  - an ensemble of SetKernelCritic models (shared per-agent encoder, self-attention, per-agent heads),
then measures, on held-out decisions with measured ground truth (all 2^n coalitions, `reps` branches each, common
random numbers), the relative error of
  - each critic with no branches at read-out,
  - per-decision permutation Monte Carlo with K branches (K = 8 ... 128),
  - the gated hybrid: 32-branch audits on the fraction of decisions where the ensemble disagrees most,
  - the precision-weighted hybrid: K audit branches on every decision, combined with the critic by inverse
    variance (critic variance calibrated on a held-out calibration split, reported on the test split only).
Audit branches come from a separate table with independent keys, so audits and ground truth do not share noise.

Run from the repository root, e.g.
  python experiments/critic_budget.py --env rware --agents 4 --window 10 --horizon 10 --seeds 3
"""
import argparse, json, os, pickle, sys, time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from accteams.branching import all_masks, make_branch_fn, shapley_from_mask_values
from accteams.critic import KernelCritic, fit
from accteams.provenance import stamp
from accteams.setcritic import (SetKernelCritic, calibrate_scale, ensemble_readout, fit_ensemble, gated_hybrid,
                                permutation_estimate, precision_weighted)
from accteams.shapley import sample_kernel_coalitions
from envs import collect, get_env


def take(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


class StageCache:
    """Pickled results of the expensive stages of one seed, so a lost runtime resumes inside a seed."""

    def __init__(self, out, seed):
        self.dir = f"{out}.cache/seed{seed}"
        os.makedirs(self.dir, exist_ok=True)

    def get(self, name, make):
        path = os.path.join(self.dir, name + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                print(f"  cache hit: {path}", flush=True)
                return pickle.load(f)
        val = make()
        with open(path + ".tmp", "wb") as f:
            pickle.dump(jax.device_get(val), f)
        os.replace(path + ".tmp", path)
        return val


def sampled_shapley(run_masks, s, a, n, perms, reps, key, rng):
    """Permutation-sampled Shapley values of the interventional game for one decision.

    Each permutation contributes the n marginal contributions along its n + 1 prefix coalitions; every prefix
    value is the mean of `reps` executed branches, with common random numbers shared by the prefixes of a
    permutation. Returns the estimate and its standard error over permutations (perms * (n + 1) * reps branches).
    """
    order = np.stack([rng.permutation(n) for _ in range(perms)])                      # (P, n)
    masks = np.zeros((perms, n + 1, n), np.float32)
    for k in range(1, n + 1):
        masks[:, k] = masks[:, k - 1]
        masks[np.arange(perms), k, order[:, k - 1]] = 1.0
    keys = jax.random.split(key, perms * reps).reshape(perms, 1, reps, -1)
    keys = jnp.broadcast_to(keys, (perms, n + 1, reps, keys.shape[-1])).reshape(-1, keys.shape[-1])
    m = jnp.asarray(np.repeat(masks.reshape(-1, n), reps, axis=0))
    vals = np.asarray(run_masks(s, a, m, keys)).reshape(perms, n + 1, reps).mean(2)
    contrib = np.zeros((perms, n))
    contrib[np.arange(perms)[:, None], order] = np.diff(vals, axis=1)
    return contrib.mean(0), contrib.std(0, ddof=1) / np.sqrt(perms)


def direct_permutation_audit(run_masks, s, a, n, perms_per_agent, key, rng):
    """The audit estimator of setcritic.permutation_estimate, run directly on branches (no 2^n table):
    for each agent, perms_per_agent pairs of independent branches with and without it; 2 * n * ppa branches."""
    ppa = max(1, perms_per_agent)
    pairs = []
    for i in range(n):
        for _ in range(ppa):
            perm = rng.permutation(n)
            before = perm[: int(np.where(perm == i)[0][0])]
            m0 = np.zeros(n, np.float32)
            m0[before] = 1.0
            m1 = m0.copy()
            m1[i] = 1.0
            pairs += [m1, m0]
    vals = np.asarray(run_masks(s, a, jnp.asarray(np.stack(pairs)), jax.random.split(key, len(pairs))))
    d = (vals[0::2] - vals[1::2]).reshape(n, ppa)
    var = d.var(1, ddof=1) / ppa if ppa > 1 else np.full(n, np.nan)
    return d.mean(1), var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="lbf", choices=["lbf", "rware"])
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--decisions", type=int, default=300000, help="training decisions, one branch each")
    ap.add_argument("--eval", type=int, default=100)
    ap.add_argument("--reps", type=int, default=200, help="branches per coalition for the ground truth")
    ap.add_argument("--audit-reps", type=int, default=64, help="branches per coalition in the audit table")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--members", type=int, default=4, help="ensemble size")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--skip-flat", action="store_true")
    ap.add_argument("--size", default=None, choices=[None, "small", "medium", "large"], help="map size variant")
    ap.add_argument("--gt", default="auto", choices=["auto", "exact", "sampled"],
                    help="ground truth: exact over 2^n coalitions, or permutation-sampled (auto: exact for n <= 6)")
    ap.add_argument("--gt-perms", type=int, default=64, help="permutations per decision for sampled ground truth")
    ap.add_argument("--gt-reps", type=int, default=16, help="branches per prefix coalition for sampled ground truth")
    ap.add_argument("--out", default="experiments/critic_budget.json")
    args = ap.parse_args()

    spec = get_env(args.env, args.agents, size=args.size)
    N, A = spec.n, spec.num_actions
    gt_mode = args.gt if args.gt != "auto" else ("exact" if N <= 6 else "sampled")
    branch = make_branch_fn(spec.step, spec.policy, spec.reference_action, horizon=args.horizon, gamma=0.99,
                            window=args.window)
    vbranch = jax.jit(jax.vmap(branch))
    masks_all = all_masks(N) if gt_mode == "exact" else None
    budgets = [k for k in (8, 16, 32, 64, 128) if k >= 2 * N]
    partial = args.out + ".partial"
    rows = json.load(open(partial))["runs"] if os.path.exists(partial) else []  # resume after a lost runtime
    if rows:
        print(f"resuming from {partial}: seeds {[r['seed'] for r in rows]} done", flush=True)
    for seed in range(args.seeds):
        if any(r["seed"] == seed for r in rows):
            continue
        t0 = time.time()
        cache = StageCache(args.out, seed)
        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        k_col, k_act, k_br, k_eval, k_gt, k_au = jax.random.split(key, 6)
        states, obs, last = collect(spec, k_col, int(1.6 * (args.decisions + args.eval) / spec.collect_steps) + 50)
        live = np.where(spec.live(states) & ~last)[0]
        assert len(live) >= args.decisions + args.eval, "collect more environments"
        pick = rng.permutation(live)[: args.decisions + args.eval]
        tr, ev = pick[: args.decisions], pick[args.decisions:]

        # ---- training data: one executed branch per decision
        s_tr, o_tr = take(states, tr), take(obs, tr)
        a_tr = jax.vmap(spec.policy)(o_tr, jax.random.split(k_act, len(tr)))
        kind = rng.choice(3, size=len(tr), p=[0.1, 0.1, 0.8]).astype(np.int32)
        mask = sample_kernel_coalitions(N, len(tr), rng, paired=True)
        mask[kind == 0], mask[kind == 1] = 0.0, 1.0
        g = cache.get("branches", lambda: np.asarray(jnp.concatenate([vbranch(*take((s_tr, a_tr), np.arange(i, min(i + 50000, len(tr)))),
                                                jnp.asarray(mask[i:i + 50000]),
                                                jax.random.split(jax.random.fold_in(k_br, i), min(50000, len(tr) - i)))
                                        for i in range(0, len(tr), 50000)]), np.float32))
        feats = lambda s: (np.asarray(jax.vmap(spec.agent_features)(s), np.float32),
                           np.asarray(jax.vmap(spec.global_features)(s), np.float32))
        x_tr, gl_tr = feats(s_tr)
        oh_tr = np.asarray(jax.nn.one_hot(a_tr, A), np.float32)
        fit_kw = dict(steps=args.train_steps, batch=args.batch, lr=args.lr, eval_every=500, patience=10 ** 6)

        flat = None
        if not args.skip_flat:
            h_tr = np.asarray(jax.vmap(spec.features)(s_tr), np.float32)
            def make_flat():
                m = KernelCritic(N, h_tr.shape[1], hidden=256, key=key, a_dim=N * A)
                fit(m, KernelCritic.loss, (h_tr, oh_tr.reshape(len(tr), -1), mask, g, kind), seed=seed, n=N, **fit_kw)
                return m
            flat = cache.get("flat_critic", make_flat)
        ens = cache.get("set_ensemble", lambda: fit_ensemble(
            SetKernelCritic, (x_tr, oh_tr, gl_tr, mask, g, kind), members=args.members, d_agent=x_tr.shape[2],
            d_global=gl_tr.shape[1], a_dim=A, width=args.width, layers=args.layers, heads=4, seed=seed, **fit_kw))
        t_train = time.time() - t0

        # ---- ground truth and an independent audit table on held-out decisions
        s_ev, o_ev = take(states, ev), take(obs, ev)
        a_ev = jax.vmap(spec.policy)(o_ev, jax.random.split(k_eval, len(ev)))
        def table(s, a, k, reps):
            m = jnp.asarray(np.repeat(masks_all, reps, axis=0))
            keys = jnp.tile(jax.random.split(k, reps), (2 ** N, 1))          # common random numbers across coalitions
            return jax.vmap(branch, in_axes=(None, None, 0, 0))(s, a, m, keys).reshape(2 ** N, reps)
        _table_j = jax.jit(table, static_argnums=3)
        table_j = lambda *z: np.asarray(_table_j(*z))
        run_masks = jax.jit(jax.vmap(branch, in_axes=(None, None, 0, 0)))
        def gt_chunk(lo, hi):
            phi, se, audit_tabs, audit_direct = [], [], [], {K: ([], []) for K in budgets}
            for j in range(lo, hi):
                gt_one(j, phi, se, audit_tabs, audit_direct)
            return phi, se, audit_tabs, audit_direct

        def gt_one(j, phi, se, audit_tabs, audit_direct):
            s_j, a_j = take(s_ev, j), a_ev[j]
            if gt_mode == "exact":
                vals = table_j(s_j, a_j, jax.random.fold_in(k_gt, j), args.reps)
                phi.append(shapley_from_mask_values(vals.mean(1), N))
                parts = np.stack([shapley_from_mask_values(vals[:, b::10].mean(1), N) for b in range(10)])
                se.append(parts.std(0, ddof=1) / np.sqrt(10))
                audit_tabs.append(table_j(s_j, a_j, jax.random.fold_in(k_au, j), args.audit_reps))
            else:
                p_, s_ = sampled_shapley(run_masks, s_j, a_j, N, args.gt_perms, args.gt_reps,
                                         jax.random.fold_in(k_gt, j), np.random.default_rng([seed, j]))
                phi.append(p_), se.append(s_)
                for K in budgets:
                    e_, v_ = direct_permutation_audit(run_masks, s_j, a_j, N, K // (2 * N),
                                                      jax.random.fold_in(jax.random.fold_in(k_au, j), K),
                                                      np.random.default_rng([seed, j, K]))
                    audit_direct[K][0].append(e_), audit_direct[K][1].append(v_)

        phi, se, audit_tabs, audit_direct = [], [], [], {K: ([], []) for K in budgets}
        for lo in range(0, len(ev), 25):
            c = cache.get(f"gt_{lo:05d}", lambda: gt_chunk(lo, min(lo + 25, len(ev))))
            phi += list(c[0]); se += list(c[1]); audit_tabs += list(c[2])
            for K in budgets:
                audit_direct[K][0].extend(c[3][K][0]); audit_direct[K][1].extend(c[3][K][1])
        phi, se = np.array(phi), np.array(se)

        x_ev, gl_ev = feats(s_ev)
        oh_ev = np.asarray(jax.nn.one_hot(a_ev, A), np.float32)
        set_mean, set_std = ensemble_readout(ens, x_ev, oh_ev, gl_ev)
        members = [m.readout(x_ev, oh_ev, gl_ev) for m in ens]
        est = {"set_ensemble": set_mean, "set_single": members[0]}
        if flat is not None:
            est["flat"] = flat.readout(np.asarray(jax.vmap(spec.features)(s_ev), np.float32), oh_ev.reshape(len(ev), -1))
        arng = np.random.default_rng(seed * 7 + 1)
        mc, mc_var = {}, {}
        for K in budgets:
            if gt_mode == "exact":
                out = [permutation_estimate(t, N, K // (2 * N), arng) for t in audit_tabs]
                mc[K], mc_var[K] = np.array([o[0] for o in out]), np.array([o[1] for o in out])
            else:
                mc[K], mc_var[K] = np.array(audit_direct[K][0]), np.array(audit_direct[K][1])
            est[f"mc{K}"] = mc[K]

        ncal = int(args.calib_frac * len(ev))
        cal, test = np.arange(ncal), np.arange(ncal, len(ev))
        scale = calibrate_scale(set_mean[cal], set_std[cal], phi[cal])
        k32 = 32 if 32 in mc else budgets[0]
        for frac in (0.1, 0.25, 0.5):
            est[f"gated{int(frac * 100)}"] = gated_hybrid(set_mean, set_std, mc[k32], frac)
        for K in budgets:
            if K // (2 * N) >= 2:
                est[f"pw{K}"] = precision_weighted(set_mean, set_std, scale, mc[K], mc_var[K])

        ph, sc = phi[test], np.mean(np.abs(phi[test]))
        big = np.abs(ph).sum(1) >= np.quantile(np.abs(ph).sum(1), 0.75)
        rel = {k: float(np.mean(np.abs(v[test] - ph)) / sc) for k, v in est.items()}
        rel_top = {k: float(np.mean(np.abs(v[test][big] - ph[big])) / np.mean(np.abs(ph[big]))) for k, v in est.items()}
        audit_cost = {**{f"mc{K}": K for K in budgets}, **{f"pw{K}": K for K in budgets},
                      **{f"gated{int(f * 100)}": f * k32 for f in (0.1, 0.25, 0.5)}}
        # Monte Carlo branches that would match each critic's error, from the 1/sqrt(K) law fitted on the MC curve
        ks = np.array(budgets, float)
        c = float(np.exp(np.mean(np.log([rel[f"mc{K}"] for K in budgets]) + 0.5 * np.log(ks))))
        equiv = {k: (c / rel[k]) ** 2 for k in ("set_ensemble", "set_single", "flat") if k in rel}
        err = np.abs(set_mean - phi).sum(1)
        row = dict(seed=seed, agents=N, size=args.size, gt_mode=gt_mode, gt_se_mean=float(np.mean(se[test])), train_decisions=len(tr), train_branches=len(tr),
                   mean_abs_phi=float(np.mean(np.abs(phi))), ground_truth_se_rel=float(np.mean(se) / np.mean(np.abs(phi))),
                   rel_err=rel, rel_err_top_quarter=rel_top, audit_branches_per_decision=audit_cost,
                   mc_equivalent_branches=equiv, mc_constant=c, calib_scale=scale,
                   uncertainty_error_spearman=spearman(set_std.sum(1)[test], err[test]),
                   top_agent_match=float(np.mean(set_mean[test].argmax(1) == ph.argmax(1))),
                   seconds_train=round(t_train, 1), seconds=round(time.time() - t0, 1))
        rows.append(row)
        json.dump(dict(args=vars(args), runs=rows), open(partial, "w"))
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "rel_err_top_quarter"}), flush=True)

    keys = rows[0]["rel_err"].keys()
    summary = {k: [float(np.mean([r["rel_err"][k] for r in rows])), float(np.std([r["rel_err"][k] for r in rows]))] for k in keys}
    print("\nrelative error on the test split, mean +/- sd over seeds (audit branches per decision in brackets)")
    for k in sorted(keys, key=lambda k: rows[0]["audit_branches_per_decision"].get(k, 0)):
        print(f"  {k:14s} [{rows[0]['audit_branches_per_decision'].get(k, 0):6.1f}] {summary[k][0]:.3f} +/- {summary[k][1]:.3f}")
    json.dump(dict(args=vars(args), runs=rows, summary=summary, provenance=stamp(vars(args))), open(args.out, "w"), indent=1)
    os.remove(partial)


if __name__ == "__main__":
    main()
