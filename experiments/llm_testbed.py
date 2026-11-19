"""Failure attribution on the replayable role team (src/accteams/llmteam.py), scripted agents.

Compares measured responsibility (Shapley over executed replays) with the earliest-decisive-error convention used by
Who&When, AgenTracer and Aegis, on planted faults and on natural failures of a uniformly weaker team, and runs the
repair test: replace the agent each rule blames with its reference and rerun the same episode.

  python experiments/llm_testbed.py --episodes 4000 --seed 0 --out experiments/llm_testbed_<date>.json

The agents are scripted, not language models: this checks the measurement and the protocol; the LLM run reuses it
through accteams.llmteam.LLMRole.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from accteams.llmteam import (ROLES, Rates, earliest_decisive_error, make_task, repair_success, replay_fidelity,
                              responsibility, run_episode)
from accteams.provenance import stamp

SETTINGS = {
    # one role is faulty, the others are reliable
    "planted_planner": dict(planner=Rates(plan_drop=0.5), solver=Rates(), verifier=Rates()),
    "planted_solver": dict(planner=Rates(), solver=Rates(solve_slip=0.15), verifier=Rates()),
    # two faults are both needed: a wrong plan only fails if the verifier would also miss solver slips (AND-type),
    # here: a slipping solver and a verifier that misses
    "and_solver_verifier": dict(planner=Rates(), solver=Rates(solve_slip=0.15), verifier=Rates(verify_miss=0.7)),
    # natural failures of a uniformly weaker team
    "weak_team": dict(planner=Rates(plan_drop=0.15, plan_swap=0.10), solver=Rates(solve_slip=0.08),
                      verifier=Rates(verify_miss=0.5, verify_false_alarm=0.05)),
}


def boot(x, rng, reps=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    m = rng.choice(x, (reps, len(x))).mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/llm_testbed.json")
    args = ap.parse_args()
    t0, rng = time.time(), np.random.default_rng(args.seed)
    out = {}
    for name, rates in SETTINGS.items():
        rows = []
        for e in range(args.episodes):
            seed = args.seed * 10 ** 6 + e
            task = make_task(np.random.default_rng([args.seed, e]))
            if run_episode(task, rates, seed).success:
                continue
            phi = responsibility(task, rates, seed)
            blame = -phi
            top = ROLES[int(np.argmax(blame))]
            ede = earliest_decisive_error(task, rates, seed)
            rows.append(dict(
                top=top, ede=None if ede is None else ede[0],
                shared=int((blame > 0.1).sum() >= 2),
                fix_top=int(repair_success(task, rates, seed, top)),
                fix_ede=int(ede is not None and repair_success(task, rates, seed, ede[0])),
                fix_rand=int(repair_success(task, rates, seed, ROLES[int(rng.integers(3))])),
                phi=[round(float(v), 4) for v in phi]))
        ede_found = [r for r in rows if r["ede"] is not None]
        out[name] = dict(
            failures=len(rows), episodes=args.episodes,
            ede_none_rate=boot([r["ede"] is None for r in rows], rng),
            disagree_rate=boot([r["ede"] != r["top"] for r in ede_found], rng),
            shared_blame_rate=boot([r["shared"] for r in rows], rng),
            repair_top=boot([r["fix_top"] for r in rows], rng),
            repair_ede=boot([r["fix_ede"] for r in rows], rng),
            repair_random=boot([r["fix_rand"] for r in rows], rng),
            mean_blame={role: float(np.mean([-r["phi"][i] for r in rows])) if rows else None
                        for i, role in enumerate(ROLES)},
            replay_fidelity=replay_fidelity(make_task(np.random.default_rng(7)), rates, range(200)),
            branches_per_failure={"responsibility": 8, "earliest_error_max": 4})
        o = out[name]
        print(f"{name:20s} fails {o['failures']:5d} | EDE none {o['ede_none_rate'][0]:.3f} | disagree "
              f"{o['disagree_rate'][0]:.3f} | shared {o['shared_blame_rate'][0]:.3f} | repair top/ede/rand "
              f"{o['repair_top'][0]:.3f}/{o['repair_ede'][0]:.3f}/{o['repair_random'][0]:.3f}", flush=True)
    json.dump(dict(args=vars(args), settings={k: {r: vars(v) for r, v in s.items()} for k, s in SETTINGS.items()},
                   results=out, seconds=round(time.time() - t0, 1), provenance=stamp(vars(args))),
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
