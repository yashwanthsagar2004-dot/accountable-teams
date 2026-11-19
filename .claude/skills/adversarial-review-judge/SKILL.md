---
name: adversarial-review-judge
description: Attack-judge-author review for research code, claims and results in this repository. Use before a claim enters CLAIMS.md, before a result is reported, or when the user asks for an adversarial review.
---

# Adversarial review with a separate judge

The attack-judge-author pattern that a post Yash shared credits to Felix Rieseberg (Anthropic). No public
repository or primary source from him could be found on 19 Nov 2026, so this is our own write-up of the
pattern, not a vendored copy. No session grades its own work: one session produced the work, several
independent sessions each try to prove it wrong, a further session that wrote neither judges each criticism,
and the author fixes what stands and may decline the rest with a reason. Ask attackers for a method, not a persona.

## Steps

1. **Fix the target.** A commit range, a set of files, or a list of claims with their evidence records.
2. **Launch 3 attackers in parallel**, each with fresh context and one lens:
   - *Statistics and claims*: does every number in CLAIMS.md / PROJECT_LOG follow from a machine-written record?
     Are intervals, seeds and tests adequate? What would a reviewer at NeurIPS reject?
   - *Correctness*: construct concrete inputs that make the code compute the wrong thing (estimator bias, wrong
     masks, off-by-one in coalitions, leakage between calibration and test, resume paths that change results).
   - *Reproducibility and integrity*: can a stranger rerun it and get the same numbers? Hidden state, unseeded
     randomness, records edited by hand, caches that survive a code change.
   Each attacker returns findings as: claim attacked, concrete failure scenario, evidence (file:line or record),
   severity. Attackers must run code to confirm where possible.
3. **Judge.** A separate session receives the target and all findings, reruns what it needs, and labels each
   finding **Stands**, **Open** (needs a decision from Yash) or **Rejected**, with a one-line reason.
4. **Author.** Fix every Stands finding, record each Open one in PROJECT_LOG, and note each Rejected one.
   A fresh session then checks only the fixes.
5. **Record** the review in `reviews/adversarial/<date>.md`: target, findings, verdicts, fixes, commit.
