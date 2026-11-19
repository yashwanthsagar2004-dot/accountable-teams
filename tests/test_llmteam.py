import numpy as np

from accteams.llmteam import (ROLES, Rates, earliest_decisive_error, make_task, repair_success, replay_fidelity,
                              responsibility, run_episode)

RELIABLE = dict(planner=Rates(), solver=Rates(), verifier=Rates())


def test_reference_team_always_succeeds():
    for e in range(50):
        assert run_episode(make_task(np.random.default_rng(e)), RELIABLE, e).success


def test_replay_is_exact():
    rates = dict(planner=Rates(plan_drop=0.3), solver=Rates(solve_slip=0.2), verifier=Rates(verify_miss=0.5))
    assert replay_fidelity(make_task(np.random.default_rng(1)), rates, range(100)) == 1.0


def test_shapley_efficiency_and_single_culprit():
    rates = dict(planner=Rates(plan_drop=1.0), solver=Rates(), verifier=Rates())
    for e in range(20):
        task = make_task(np.random.default_rng(e))
        ep = run_episode(task, rates, e)
        phi = responsibility(task, rates, e)
        assert abs(phi.sum() - (float(ep.success) - 1.0)) < 1e-9        # v(N) - v(empty), v(empty) = 1
        if not ep.success:
            assert ROLES[int(np.argmax(-phi))] == "planner"
            assert earliest_decisive_error(task, rates, e)[0] == "planner"
            assert repair_success(task, rates, e, "planner")


def test_and_failure_splits_blame():
    # the solver slips on its first attempt only and the verifier misses it: either fix alone would succeed
    # (a reference verifier triggers a clean redo), so the failure needs both. Shapley gives each half; the
    # earliest-decisive-error convention names only the solver.
    from accteams.llmteam import solver
    rates = dict(planner=Rates(), solver=Rates(solve_slip=0.5), verifier=Rates(verify_miss=1.0))
    found = 0
    for e in range(200):
        task = make_task(np.random.default_rng(e))
        plan = task.steps
        first = solver(task, plan, rates["solver"], e, 0)[0] != task.answer()
        redo_ok = solver(task, plan, rates["solver"], e, 1)[0] == task.answer()
        if not (first and redo_ok):
            continue
        phi = responsibility(task, rates, e)
        assert np.allclose(-phi, [0.0, 0.5, 0.5])
        assert earliest_decisive_error(task, rates, e)[0] == "solver"
        found += 1
    assert found >= 5
