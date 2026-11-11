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

- 2026-11-11 — After the AgenTracer and Aegis ICLR 2026 reviews: ground-truth validity was the main objection in 8 of
  11 reviews across the three closest papers. The LLM testbed adds natural failures from weaker models, labelled by
  measured responsibility (Aegis declined these as unlabelled). It also reports replay fidelity per task and counts
  where the earliest-decisive-error convention disagrees with measured responsibility. A repair experiment compares
  the top-responsibility agent with the judge's choice and a random agent. Seeds, intervals, sweeps (including β) and
  released records are required from the first submission.
- 2026-11-11 — Positioning after reading the closest work (CAR, C3, Credit Without Ground Truth, Who&When and its ICML
  2025 reviews): executed replay plus Shapley is not new. Lead with T5 (logs cannot identify responsibility), T1b/T4
  (on-policy credit bias and the fairness gap) and T6/T7 (audit-aware agents); present the critic as a tool measured
  against branch budget. The LLM testbed must be replayable and include Who&When's three judges, a process-reward
  earliest-error baseline and Agent-as-Judge. Doc: https://claude.ai/artifact/UMkdHMqE7HgLDquz5i7SkS
- 2026-11-11 — Implementation standard: every claim is executable. `claims/claims.json` lists each theorem and
  experimental claim with its status, check command and the record its numbers come from; `tools/verify_claims.py`
  re-reads every number and runs the theory checks; CI fails if anything drifts. Failures stay in the ledger
  (RobotWarehouse is listed as "fails").
- 2026-11-11 — Interactive supplement: the Intervention Bench (https://claude.ai/artifact/4VZBTozMhc2ZgDDaX5dykb;
  source in `web/`, built page in `docs/index.html`). One coalition lattice with per-agent switches shows the
  counterfactual value, the Shapley readout, on-policy credit against responsibility, and the audit game.
- 2026-11-11 — Confirmatory runs report interquartile mean, stratified bootstrap intervals, probability of
  improvement and Holm-adjusted permutation tests (`accteams.stats`), following Agarwal et al. 2021.
- 2026-11-11 — New records carry a `provenance` block (commit, command, configuration hash, device, versions,
  seconds). The Colab notebook's last cell downloads all records as a zip; no more hand transcription.
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

- 2026-11-11 — Reviewer kit checks (CPU): 14 tests pass, 1 skipped (needs jumanji); `verify_claims.py --checks`:
  14 claims, 9 quoted numbers match their records, 4 check scripts pass. The bench's JavaScript core agrees with
  the repo's Python on 12 random games (max difference 6e-15) and reproduces T3 by simulation at 4 agents:
  kernel design 4.56 measured against 4.66 predicted, order-2 groups 16.5 against 16, permutation audit 4.60
  against 4.50. No new training runs.

- 2026-11-10 — RobotWarehouse (4 agents, scripted team, Colab T4). Gate conditions 1 and 2 pass at every
  window 5-25, far more easily than LBF (ground-truth se 0.03-0.05; 32-branch Monte Carlo 0.23-0.34).
  Condition 3 FAILS with the basic critic inputs at window 10: kernel critic 0.95 +/- 0.06 against Monte Carlo
  0.27 +/- 0.01 (all 3 seeds). With richer inputs (carried-shelf status, target distance, local 5x5 map) and a
  256-wide critic: 0.58 +/- 0.04, still behind Monte Carlo on all 3 seeds. The critic is worth about 7
  per-decision branches here (1 branch per decision in training, none at read-out), against more than 32 on
  LBF. File: `experiments/colab_rware_2026-11-10.json`.

- 2026-11-10 — GATE PASSED on Level-Based Foraging (4 agents), Colab T4. Window = branch length 5 to 25 all pass
  conditions 1 and 2 (ground-truth se 0.07-0.08 of the signal). Condition 3 at window 20: kernel critic 0.65 +/-
  0.05 against 32-branch Monte Carlo 0.93 +/- 0.05, better on all 3 seeds; most consequential quarter 0.56
  against 0.69. At window 10: 0.92 against 0.99 (better on all 3 seeds, narrowly). The paired-difference critic
  does not help when window = branch length. The critic uses about one branch per training decision and none at
  read-out. File: `experiments/colab_gate_2026-11-10.json`.

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

0. Condition 3 on RobotWarehouse fails as written (critic 0.58 against 32-branch Monte Carlo 0.27). Options:
   (a) report accuracy against the per-decision branch budget and claim amortisation only where it holds;
   (b) more training branches and a permutation-invariant (per-agent, shared-weight) critic;
   (c) a hybrid that spends a few audit branches where the critic is uncertain. Decide before prereg-v1.
1. Gate passed on LBF at window = branch length = 20. Still open: the same test with a learned
   (MAPPO) team instead of the scripted one; and the design text, because window interventions change what
   responsibility means (a stretch of an agent's behaviour, not one decision). Decide the window per task and
   freeze it before prereg-v1.
2. Does the amortised critic beat per-decision Kernel SHAP at equal branches? (Reviewer 2's question.)
3. Finite-sample bound for the shared network (T3, amortised).
4. Credit mode: which centring of the responsibility readout to use as MAPPO's advantage.
5. Read in full before wording claims: Ding et al. 2008, Grabisch et al. 2000, Covert and Lee 2021, MCPG.
6. S1.7 (self-test 0) is still open; due 30 Nov 2026.
7. Re-run the LBF and RobotWarehouse gates once so their records are machine-written with provenance; then
   remove the "transcribed" note from REPRODUCE.md.
8. Choose the contribution type for Phase 1 when the NeurIPS 2027 call appears. In 2026 the types were General,
   Theory, Use-Inspired, Concept and Feasibility, Negative Results; a Theory submission is judged on correctness
   and its experiments need only illuminate the theory. Check that 2027 keeps this.
9. Extend the identifying-strings list in `tools/make_supplement.py` before any double-blind submission.
10. If `claims/claims.json` is not on the main branch on GitHub, the 11 Nov reviewer-kit patch (33 files, made
    against commit 16d07aa) has not been pushed yet; a new session should not assume those files exist there.

## Keeping the full scope

The scope stays at two main-track papers with theory, a benchmark, MARL and LLM-team experiments and an
adversarial phase. What makes that workable: gates in a fixed order (signal-to-noise, baselines, tier 1, tier 2,
tier 3, Phase 2), each ending in a pass/fail test; branch generation vectorised in JAX; compute overflow as
listed above; a workshop submission of the tier-1 theory and benchmark for early outside review; and the guide
or one theory-minded reader checking the proofs.

## AI use

This project uses Claude (Anthropic) for code drafts, reference proofs, numerical checks and literature searches,
at the author's request. Declare it in every paper and thesis under the venue's and the institution's rules.
