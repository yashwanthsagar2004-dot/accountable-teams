"""Tier-2 smoke test (exploratory): the kernel critic on executed branches in Level-Based Foraging.

Four agents on an 8x8 grid with three food items that need cooperation (Jumanji LBF, force_coop). A scripted
team walks to the first uneaten item and loads it, with 25% random actions. At sampled decision points we run
one executed branch each (coalition from the Shapley kernel, absent agents take the no-op) and fit the kernel
critic. On held-out decisions we compare its responsibility readout with measured ground truth (all 16
coalitions, `--reps` branches each) and with per-decision permutation sampling that spends 32 branches.

Run from the repository root:  python experiments/lbf_smoke.py --decisions 60000
"""
import argparse, json, sys, time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")
from accteams.branching import all_masks, make_branch_fn, shapley_from_mask_values
from accteams.critic import DeltaKernelCritic, KernelCritic, fit
from accteams.shapley import sample_kernel_coalitions

from jumanji.environments.routing.lbf import env as lbfenv, generator

N, FOOD, GRID, T_COLLECT, HORIZON, NUM_ACTIONS, EPS = 4, 3, 8, 35, 15, 6, 0.25


def scripted_policy(obs, key):
    view = obs.agents_view                                   # (n, 3*FOOD + 3*n): food triples, own triple, others
    food = view[:, : 3 * FOOD].reshape(N, FOOD, 3)
    alive = food[:, :, 2] > 0
    first = jnp.argmax(alive, axis=1)                        # every agent sees the same food order: shared target
    tgt = jnp.take_along_axis(food, first[:, None, None], axis=1)[:, 0, :2]
    me = view[:, 3 * FOOD: 3 * FOOD + 2]
    d = tgt - me
    adjacent = jnp.abs(d).sum(-1) == 1
    vertical = jnp.abs(d[:, 0]) >= jnp.abs(d[:, 1])
    move = jnp.where(vertical, jnp.where(d[:, 0] < 0, 1, 2), jnp.where(d[:, 1] < 0, 3, 4))
    act = jnp.where(adjacent, 5, move)
    act = jnp.where(alive.any(axis=1), act, 0)
    k1, k2 = jax.random.split(key)
    return jnp.where(jax.random.uniform(k1, (N,)) < EPS, jax.random.randint(k2, (N,), 0, NUM_ACTIONS), act)


def features(state):
    a, f = state.agents, state.food_items
    return jnp.concatenate([(a.position / GRID).reshape(-1), a.level / 5.0, (f.position / GRID).reshape(-1),
                            f.level / 5.0, f.eaten.astype(jnp.float32), state.step_count[None] / 50.0])


def collect(env, key, num_envs):
    """Decision points (state, observation) visited by the scripted team."""
    def one(k):
        k0, k1 = jax.random.split(k)
        state, ts = env.reset(k0)

        def body(carry, kk):
            state, ts = carry
            out = (state, ts.observation)
            state, ts = env.step(state, scripted_policy(ts.observation, kk))
            return (state, ts), out

        _, traj = jax.lax.scan(body, (state, ts), jax.random.split(k1, T_COLLECT))
        return traj

    states, obs = jax.jit(jax.vmap(one))(jax.random.split(key, num_envs))
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    return jax.tree_util.tree_map(flat, states), jax.tree_util.tree_map(flat, obs)


def take(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=60000)
    ap.add_argument("--eval", type=int, default=150)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--window", type=int, default=1, help="steps for which absent agents take the reference")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="branch length in steps")
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience; large = fixed steps")
    ap.add_argument("--out", default="experiments/lbf_smoke.json")
    args = ap.parse_args()
    env = lbfenv.LevelBasedForaging(
        generator=generator.RandomGenerator(grid_size=GRID, fov=GRID, num_agents=N, num_food=FOOD, force_coop=True),
        time_limit=50)
    branch = make_branch_fn(env.step, scripted_policy, jnp.zeros(N, jnp.int32), horizon=args.horizon, gamma=0.99,
                            window=args.window)
    vbranch = jax.jit(jax.vmap(branch))
    one_hot = lambda a: jax.nn.one_hot(a, NUM_ACTIONS).reshape(a.shape[0], -1)
    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        k_col, k_act, k_br, k_eval, k_gt, k_mc = jax.random.split(key, 6)
        states, obs = collect(env, k_col, int(1.6 * (args.decisions + args.eval) / T_COLLECT) + 50)
        live = np.where(~np.asarray(states.food_items.eaten).all(1))[0]          # skip finished episodes
        assert len(live) >= args.decisions + args.eval, 'collect more environments'
        pick = rng.permutation(live)[: args.decisions + args.eval]
        tr, ev = pick[: args.decisions], pick[args.decisions:]
        # training data: one executed branch per decision
        s_tr, o_tr = take(states, tr), take(obs, tr)
        a_tr = jax.vmap(scripted_policy)(o_tr, jax.random.split(k_act, len(tr)))
        kind = rng.choice(3, size=len(tr), p=[0.1, 0.1, 0.8])
        mask = sample_kernel_coalitions(N, len(tr), rng, paired=True)
        mask[kind == 0], mask[kind == 1] = 0.0, 1.0
        g = vbranch(s_tr, a_tr, jnp.asarray(mask), jax.random.split(k_br, len(tr)))
        h_tr = jax.vmap(features)(s_tr)
        steps_train = len(tr) * args.horizon
        data = (np.asarray(h_tr), np.asarray(one_hot(a_tr)), mask, np.asarray(g, np.float32), kind.astype(np.int32))
        model = KernelCritic(N, h_tr.shape[1], key=key, a_dim=N * NUM_ACTIONS)
        fit(model, KernelCritic.loss, data, steps=args.train_steps, batch=args.batch, eval_every=500,
            patience=args.patience, seed=seed, n=N)
        # paired design at the same number of branches: half the decisions, two branches each (S and the
        # all-reference team) that share their random numbers; the target is the difference of the two returns
        half = len(tr) // 2
        s_p, a_p, h_p = take(s_tr, np.arange(half)), a_tr[:half], h_tr[:half]
        full = (rng.random(half) < 0.15).astype(np.int32)
        mask_p = sample_kernel_coalitions(N, half, rng, paired=True)
        mask_p[full == 1] = 1.0
        keys_p = jax.random.split(jax.random.fold_in(k_br, 1), half)
        d = vbranch(s_p, a_p, jnp.asarray(mask_p), keys_p) - vbranch(s_p, a_p, jnp.zeros((half, N)), keys_p)
        data_p = (np.asarray(h_p), np.asarray(one_hot(a_p)), mask_p, np.asarray(d, np.float32), full)
        paired_model = DeltaKernelCritic(N, h_tr.shape[1], key=key, a_dim=N * NUM_ACTIONS)
        fit(paired_model, DeltaKernelCritic.loss, data_p, steps=args.train_steps, batch=args.batch, eval_every=500,
            patience=args.patience, seed=seed, n=N)
        # ground truth on held-out decisions: all coalitions, reps branches each
        s_ev, o_ev = take(states, ev), take(obs, ev)
        a_ev = jax.vmap(scripted_policy)(o_ev, jax.random.split(k_eval, len(ev)))
        masks16 = jnp.asarray(np.repeat(all_masks(N), args.reps, axis=0))
        # common random numbers: repetition r uses the same key in all 16 coalitions, which keeps every v(S)
        # unbiased and lowers the noise of the differences that the Shapley value is made of
        gt_one = jax.jit(lambda s, a, k: jax.vmap(branch, in_axes=(None, None, 0, 0))(
            s, a, masks16, jnp.tile(jax.random.split(k, args.reps), (2 ** N, 1))))
        phi, se, mc = [], [], []
        for j in range(len(ev)):
            vals = np.asarray(gt_one(take(s_ev, j), a_ev[j], jax.random.fold_in(k_gt, j))).reshape(2 ** N, args.reps)
            phi.append(shapley_from_mask_values(vals.mean(1), N))
            # standard error of phi by splitting the repetitions into 10 batches
            parts = np.stack([shapley_from_mask_values(vals[:, b::10].mean(1), N) for b in range(10)])
            se.append(parts.std(0, ddof=1) / np.sqrt(10))
            # per-decision permutation sampling with 32 branches (4 permutations x 2 branches x 4 agents)
            sub = np.random.default_rng(seed * 100000 + j)
            est = np.zeros(N)
            for i in range(N):
                for _ in range(4):
                    perm = sub.permutation(N); pos = int(np.where(perm == i)[0][0])
                    b0 = sum(1 << int(q) for q in perm[:pos])
                    r = sub.integers(args.reps)
                    est[i] += (vals[b0 | (1 << i), r] - vals[b0, r]) / 4
            mc.append(est)
        phi, se, mc = np.array(phi), np.array(se), np.array(mc)
        h_ev, x_ev = np.asarray(jax.vmap(features)(s_ev)), np.asarray(one_hot(a_ev))
        est, est_p = model.readout(h_ev, x_ev), paired_model.readout(h_ev, x_ev)
        scale = np.mean(np.abs(phi))
        big = np.abs(phi).sum(1) >= np.quantile(np.abs(phi).sum(1), 0.75)       # the most consequential quarter
        rel = lambda e, m=slice(None): float(np.mean(np.abs(e[m] - phi[m])) / np.mean(np.abs(phi[m])))
        row = dict(seed=seed, decisions=len(tr), train_env_steps=steps_train, mean_abs_phi=float(scale),
                   ground_truth_se_rel=float(np.mean(se) / scale),
                   critic_rel_err=float(np.mean(np.abs(est - phi)) / scale),
                   paired_critic_rel_err=rel(est_p), paired_critic_rel_err_top_quarter=rel(est_p, big),
                   paired_top_agent_match=float(np.mean(est_p.argmax(1) == phi.argmax(1))),
                   paired_difference_sd=float(np.asarray(d).std()),
                   mc32_rel_err=float(np.mean(np.abs(mc - phi)) / scale),
                   zero_rel_err=1.0, critic_rel_err_top_quarter=rel(est, big), mc32_rel_err_top_quarter=rel(mc, big),
                   return_sd=float(np.asarray(g).std()),
                   efficiency_gap=float(np.mean(np.abs(est.sum(1) - phi.sum(1))) / scale),
                   top_agent_match=float(np.mean(est.argmax(1) == phi.argmax(1))),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(row, flush=True)
    json.dump(dict(args=vars(args), runs=rows), open(args.out, "w"), indent=1)
    for k in ("critic_rel_err", "paired_critic_rel_err", "mc32_rel_err", "critic_rel_err_top_quarter",
              "paired_critic_rel_err_top_quarter", "mc32_rel_err_top_quarter", "paired_top_agent_match",
              "paired_difference_sd",
              "ground_truth_se_rel", "top_agent_match", "mean_abs_phi", "return_sd"):
        v = np.array([r[k] for r in rows]); print(f"{k:22s} {v.mean():.3f} +/- {v.std():.3f}")


if __name__ == "__main__":
    main()
