"""Learning curves of MAPPO with team credit, responsibility credit (executed branches) and on-policy
counterfactual credit (COMA-style Q), on one task and team size, several seeds.

  python experiments/credit_compare.py --env lbf --agents 4 --updates 2000 --seeds 3
Writes one record with the per-update reward of every run and a summary of the final 10% of training.
"""
import argparse, json, sys, time

import jax
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from accteams.mappo import Config, make_trainer
from accteams.provenance import stamp
from accteams.stats import iqm, stratified_bootstrap_ci
from envs import get_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="lbf", choices=["lbf", "rware"])
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--credits", default="team,resp,dr")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--branch-frac", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="experiments/credit_compare.json")
    args = ap.parse_args()
    spec = get_env(args.env, args.agents)
    runs, curves = [], {}
    for credit in args.credits.split(","):
        cfg = Config(num_envs=args.num_envs, rollout=args.rollout, credit=credit, beta=args.beta, hidden=args.hidden,
                     lr=args.lr, window=args.window, horizon=args.horizon, branch_frac=args.branch_frac)
        train, _ = make_trainer(spec, cfg)
        f = jax.jit(train, static_argnums=1)
        for seed in range(args.seeds):
            t = time.time()
            _, m = f(jax.random.PRNGKey(seed), args.updates)
            r = np.asarray(m["reward_per_step"], float)
            tail = float(r[-max(1, args.updates // 10):].mean())
            branches = int(args.branch_frac * args.num_envs * args.rollout) * args.updates if credit == "resp" else 0
            runs.append(dict(credit=credit, seed=seed, final_reward_per_step=tail, auc=float(r.mean()),
                             branches=branches, branch_env_steps=branches * args.horizon,
                             env_steps=args.num_envs * args.rollout * args.updates, seconds=round(time.time() - t, 1)))
            curves[f"{credit}/{seed}"] = np.round(r, 6).tolist()
            print(json.dumps(runs[-1]), flush=True)
    summary = {}
    for credit in args.credits.split(","):
        fin = np.array([r["final_reward_per_step"] for r in runs if r["credit"] == credit])
        est, lo, hi = stratified_bootstrap_ci(fin, stat=np.mean, reps=2000)
        summary[credit] = dict(mean=est, ci95=[lo, hi], iqm=iqm(fin), seeds=len(fin))
        print(f"{credit:5s} final reward/step {est:.4f} [{lo:.4f}, {hi:.4f}]")
    json.dump(dict(args=vars(args), runs=runs, summary=summary, curves=curves, provenance=stamp(vars(args))),
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
