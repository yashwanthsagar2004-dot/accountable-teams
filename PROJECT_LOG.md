# Project log — Accountable-by-Design Agent Teams

Keep this file current. Claude does not carry project state between sessions; this file, the research design
and the survey are what a new session reads first. Newest entries at the top of each section.

## Brief

- **Question.** When a team of AI agents succeeds or fails, how much did each agent cause the result? The answer
  should hold under real interventions, be fair to weak and strong agents, be cheap while the team learns, and
  resist agents that try to shift blame.
- **Hypothesis.** Credit fitted on the team's own data cannot answer it (it converges to a policy-weighted
  Banzhaf index). A critic trained on executed, declared-reference branches can, with a stated error. Audits that
  agents cannot detect bound blame manipulation by 2R·delta.
- **Target venues.** Phase 1: NeurIPS 2027 (deadline estimated mid-May 2027). Phase 2: ICLR 2028 (estimated late
  September 2027). Confirm both dates when the calls appear.
- **Success.** H1 to H4 hold in the pre-registered tests C1 to C4 (10 seeds, Holm correction) for Phase 1; H5 and
  H6 hold in C5 and C6 for Phase 2.
- **Refutation.** H1 fails (executed-branch training is no better than on-policy training at recovering known
  responsibility), or H6 fails (undetectable audits still allow large blame shifts).
- **Standard.** The unified research design and evaluation standard v2; NUS, NTU, ETH and MIT research practice
  as recorded in the flagship document. The scope is not reduced to fit hardware; see "Keeping the full scope".

## Who supplies what

| Yash | Claude |
|---|---|
| GPU runs on the RTX 4060 (8 GB) and Colab Pro; raw logs and JSON records back into `experiments/` | Code and tests at small scale (CPU), analysis of returned logs |
| The proofs that go into papers, written by hand after each self-test | Reference proofs and numerical checks |
| Final judgement on correctness, and every claim in a paper | Literature searches and novelty audits, with dates |
| The papers, statements of purpose and all authorship | Drafts for reference only; AI use is declared |

Neither side can prove an idea is new. Claims are worded "searches up to <date> found no paper that ...".

## Compute and data plan

- **Data.** No external datasets in Phase 1. Tier 1: planted games (`accteams.games`). Tier 2: Jumanji
  Level-Based Foraging and RobotWarehouse, plus the hidden-gift task. Tier 3: a three-agent LLM-team testbed with
  recorded traces. Every run writes one JSON record (config hash, seed, git commit, metrics).
- **Compute.** Design estimate: 48 ten-seed jobs, 24 to 68 GPU-hours (plus two ablation jobs added 10 Nov).
  Branch generation is cheap: about 4 million environment steps a second on two CPU cores for two-agent LBF with
  a random policy. Training networks and LLM inference are the real costs; measure both in November.
- **Overflow, in order.** Colab Pro; Kaggle notebooks (a free weekly GPU quota, commonly about 30 hours; check
  your account); the IndiaAI Compute Portal (subsidised GPU hours, students are an eligible category; check the
  eligibility page and rate card); cloud credits requested with the workshop paper as evidence.
- **Hand-off.** After each run: commit, push, paste the commit hash and the JSON path under "Results" below.

## Decisions

- 2026-11-10 — Responsibility heads are trained with kernel-sampled branches (FastSHAP's objective on executed
  returns). Pair terms are optional and only display interactions. Reason: T3 noise factors (144 against 5.7 at
  eight agents).
- 2026-11-10 — T1b's closed form is known (Ding et al. 2008; Grabisch, Marichal, Roubens 2000; Hammer and
  Holzman 1992). Claim narrowed to: on-policy credit converges to a policy-weighted Banzhaf index; the
  identification with multi-agent credit, T4 and the fix are ours. P1's Proposition 2 needs the same citations.
- 2026-11-10 — FastSHAP replaces SHAQ in the best-7 base papers; SHAQ stays a baseline. Survey now has 42 papers.
- 2026-11-10 — Paired branches with shared random numbers and window interventions added as options, after the
  first LBF measurement.
- 2026-11-09 — Candidate A chosen. Reference = declared (default action or certified version). Responsibility =
  Shapley value of the interventional game.

## Results (all exploratory until the prereg-v1 tag)

- 2026-11-10 — Colab (T4) build step 1 PASSED: 14/14 tests; tier-1 pilot reproduces the CPU numbers (kernel 0.37,
  pair 0.55, additive 0.53, gap +0.13 at 20k returns). LBF smoke window 1 and 5 reproduce the CPU runs; about
  1 min a seed on the T4 against 3 min on CPU.
- 2026-11-10 — Colab gate sweep (60 decisions, 400 repetitions, 15 settings): conditions 1 and 2 pass only when
  the intervention lasts the whole branch. Window 5 / branch 5: ground-truth se 0.08 of the signal, 32-branch
  Monte Carlo 0.96. Window 10 / branch 10: 0.08 and 0.93. Every setting with steps after the window fails.
  Next: extend the diagonal (15, 20, 25), then condition 3 with lbf_smoke.py --window 10 --horizon 10.

- 2026-11-10 — LBF smoke test, 4 agents, 3 seeds, 300k branches: nothing beats the zero predictor. One decision:
  kernel 3.72, paired 1.92, Monte Carlo (32 branches) 1.69. Five-step window: kernel 1.52, paired 1.20, Monte
  Carlo 1.73; paired critic 0.96 on the most consequential quarter. Mean absolute responsibility 0.006 against a
  return sd of 0.19. Files: `experiments/lbf_smoke_2026-11-10_window{1,5}.json`.
- 2026-11-09 — Tier-1 pilot, 4 agents, 3 seeds, order-3 games, 80k returns: kernel 0.24, order-2 pair critic
  0.40, additive on-policy 0.54 (plateau); weak-minus-strong gap +0.13 against +0.01.
  File: `experiments/pilot_sweep_2026-11-09.json`.
- 2026-11-10 — 14 unit tests pass; three theory-check scripts pass.

## Open questions

1. Signal-to-noise on tier 2: on LBF, window = branch length (5 or 10) passes conditions 1 and 2, narrowly.
   Still open: condition 3 (critic beats Monte Carlo), longer diagonals, RobotWarehouse. Window interventions
   change what responsibility means (a stretch of behaviour, not one decision); the design must say so.
2. Does the amortised critic beat per-decision Kernel SHAP at equal branches? (Reviewer 2's question.)
3. Finite-sample bound for the shared network (T3, amortised).
4. Credit mode: which centring of the responsibility readout to use as MAPPO's advantage.
5. Read in full before wording claims: Ding et al. 2008, Grabisch et al. 2000, Covert and Lee 2021, MCPG.
6. S1.7 (self-test 0) is still open; due 30 Nov 2026.

## Keeping the full scope

The scope stays at two main-track papers with theory, a benchmark, MARL and LLM-team experiments and an
adversarial phase. What makes that workable: gates in a fixed order (signal-to-noise, baselines, tier 1, tier 2,
tier 3, Phase 2), each ending in a pass/fail test; branch generation vectorised in JAX; compute overflow as
listed above; a workshop submission of the tier-1 theory and benchmark for early outside review; and the guide
or one theory-minded reader checking the proofs.

## AI use

This project uses Claude (Anthropic) for code drafts, reference proofs, numerical checks and literature searches,
at the author's request. Declare it in every paper and thesis under the venue's and the institution's rules.
