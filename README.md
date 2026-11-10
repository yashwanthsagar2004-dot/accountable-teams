# accountable-teams — starter kit

Code for **Accountable-by-Design Agent Teams** (research design, 9 Nov 2026). It starts Phase 1:

- **BlameBench tier 1** — planted team games whose true responsibility is known exactly (`accteams.games`).
- **Shapley tools** — exact enumeration, permutation sampling, Harsanyi dividends, the Shapley kernel and the
  kernel least-squares identity (`accteams.shapley`).
- **Executed branches for JAX environments** (`accteams.branching`): the branch function (single-decision or
  window interventions, optional bootstrap), measured ground truth over all coalitions, permutation audits with
  a Hoeffding band, and the audit budget. Tested on Jumanji Level-Based Foraging.
- **Four critics** in plain JAX (`accteams.critic`):
  - `KernelCritic` (the default): per-agent responsibility heads fitted to *executed* branch returns, with
    coalitions drawn from the Shapley kernel. At the optimum each head is the exact Shapley value.
  - `DeltaKernelCritic`: the same heads trained on paired branch differences that share random numbers; use it
    when one decision moves the return only a little.
  - `PairCritic`: anchored order-2 terms fitted on groups of at most two agents (exact up to the order-3 tail).
  - `AdditiveOnPolicy`: a VDN-style additive model fitted on the team's own data — the baseline whose bias
    theorem T1b gives in closed form.
- **Unit tests** of the facts the method relies on (`tests/`), and the numerical checks of T1–T7 (`theory/`).

Responsibility of agent *i* at a decision point is the Shapley value of the interventional game
`v(S) = E[return | agents in S keep their actions, the others take their declared reference]`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

JAX's CUDA builds are for Linux. On a Windows laptop, run inside WSL2 (Ubuntu) and install the CUDA build of
JAX there (`pip install "jax[cuda12]"`); Colab is already Linux.

## Quick start

```bash
pytest -q                                   # 14 tests, about 15 s (6 need jumanji: pip install -e ".[dev,rl]")
python -m accteams.train_tier1 --seeds 3    # pilot on planted games, under a minute on CPU
python experiments/pilot_sweep.py           # error against branch budget, about 2 minutes on CPU
python theory/check_theory_A3.py            # numerical checks of T3 (kernel design), T4 (n agents), T5
python experiments/lbf_smoke.py --decisions 300000 --eval 100 --reps 200 --train-steps 20000 --batch 2048 \
       --patience 100000 --window 1         # tier-2 smoke test on Level-Based Foraging, about 3 min a seed on CPU
```

## Pilot results (exploratory, not pre-registered; CPU, 3 seeds, 4 agents)

Relative attribution error = mean |estimate − true responsibility| / mean |true responsibility|. Each method
gets the same number of executed returns; early stopping on a 10% validation split.

| Budget | Kernel critic (branches) | Order-2 critic (small groups) | Additive, on-policy |
|---|---|---|---|
| Order-3 game, 5,000 | 0.52 ± 0.03 | 0.71 ± 0.05 | 0.56 ± 0.03 |
| Order-3 game, 20,000 | 0.37 ± 0.05 | 0.55 ± 0.08 | 0.53 ± 0.04 |
| Order-3 game, 80,000 | 0.24 ± 0.03 | 0.40 ± 0.05 | 0.54 ± 0.04 |

The kernel critic keeps improving with budget; the on-policy additive model stops near 0.54 (its bias) and
over-credits the weakest agent (weak-minus-strong error gap +0.13 ± 0.07 at 80,000). Full numbers, including
order-2 games, are in `experiments/pilot_sweep_2026-11-09.json`.

## First Level-Based Foraging measurement (exploratory; 10 Nov 2026, CPU, 3 seeds)

Four agents, 8x8 grid, three food items that need cooperation, a scripted team with 25% random actions,
15-step branches, 300,000 branches per seed. Relative error as above; predicting zero for everyone scores 1.00.

| Intervention | mean abs. responsibility | sd of a return | sd of a paired difference | Kernel critic | Paired-difference critic | Monte Carlo, 32 branches a decision |
|---|---|---|---|---|---|---|
| One decision | 0.006 | 0.190 | 0.087 | 3.72 ± 0.43 | 1.92 ± 0.07 | 1.69 ± 0.12 |
| Five-step window | 0.015 | 0.174 | 0.124 | 1.52 ± 0.11 | 1.20 ± 0.05 | 1.73 ± 0.05 |

**Nothing beats predicting zero yet.** One decision moves the return by about 3% of the return's standard
deviation, so this is a signal-to-noise problem before it is a modelling problem. Pairing branches and widening
the intervention both help, and on the most consequential quarter of decisions the paired critic reaches
0.96 ± 0.03 with a five-step window. The measured ground truth is itself noisy here (its standard error is about
27% of the mean absolute responsibility at 3,200 branches a decision). See section 5 of the Implementation guide
for what to do next. Raw numbers: `experiments/lbf_smoke_2026-11-10_window{1,5}.json`.

## Layout

```
src/accteams/   games.py  shapley.py  critic.py  branching.py  train_tier1.py
tests/          test_theory.py  test_branching.py
theory/         check_theory_A.py  check_theory_A2.py  check_theory_A3.py
experiments/    pilot_sweep.py  lbf_smoke.py  *.json (dated results)
PROJECT_LOG.md  brief, decisions, results, open questions: update it after every session
notebooks/      colab_runner.ipynb
```

## Next steps (see the Implementation tab of the research design)

1. Pass the signal-to-noise gate on Level-Based Foraging and RobotWarehouse (window, horizon, pairing).
2. The kernel critic as MAPPO's centralised critic, in "account" and "credit" modes.
3. Attribution baselines and the tier-2 confirmatory runs.
4. The LLM-team testbed with certified-reference replays.

## Notes

- Licence: Apache-2.0, to match JaxMARL and Jumanji.
- AI use: this starter kit was drafted with Claude (Anthropic) at the author's request. Declare it in any paper or
  thesis that uses it, following the venue's and your institution's rules.
