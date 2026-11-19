# Agent skills used in this project

These skills live in `.claude/skills/`, so anyone who opens the repository in Claude Code gets the same tools.
Each folder's `UPSTREAM.md` gives the source repository, the commit and the date it was copied, plus any trimming;
the licence is in `UPSTREAM_LICENSE`. Star counts come from the GitHub search API on 19 Nov 2026.

| Skill folder | Upstream (licence) | Stars | Used here for |
|---|---|---|---|
| `adversarial-review-judge` | this repository: our own write-up of the attack-judge-author pattern credited to Felix Rieseberg in a post Yash shared (no public repository found) | — | Independent attackers, a separate judge and an author fix step for code and claims (`reviews/adversarial/`) |
| `adversarial-reviewer` | alirezarezvani/claude-skills (MIT) | 30,088 | Review of a diff through three hostile personas before a merge |
| `rederive` | EricJujianZou/rederive (MIT) | 1,075 | Yash re-derives the design decisions before interviews (`docs/REDERIVE.md`) |
| `aris-experiment-audit`, `aris-paper-claim-audit`, `aris-ablation-planner`, `aris-kill-argument`, `aris-novelty-check`, `aris-interview-cheatsheet` | wanshuiyin/Auto-claude-code-research-in-sleep (MIT) | 19,450 | Claim audits against records, ablation plans, the strongest rejection argument, novelty checks |
| `kd-statistical-analysis`, `kd-statistical-power` | K-Dense-AI/claude-scientific-skills (MIT) | not returned by search | Test selection, and power for the seed counts in the version-2 plan |
| `taste-skill` | Leonxlnx/taste-skill (MIT; asset library trimmed) | 118,599 | Design direction of the project page (layout, type, copy without generic AI styling) |
| `od-frontend-design`, `od-emilkowalski-motion` | nexu-io/open-design (Apache-2.0) | 108,393 | Rules for building the page, and a motion pass with restraint |
| `emil-design-eng`, `animate`, `review-animations` | emilkowalski/skills (MIT) | 58,043 | Motion decisions (purpose, curve, duration, reduced motion) and motion review |
| `motionsites-web-builder` | Motionsites-AI/claude-skill (MIT for the skill; examples and media © MotionSites, not vendored) | 2,272 | Spec-first page build and the screenshot-and-fix loop; its advertising instructions are not followed |

Rules
- Skills shape how work is done. They never supply results: every number on the page or in a paper draft comes
  from a machine-written record and is checked by `tools/verify_claims.py`.
- A third-party skill never overrides `CLAUDE.md` or `PROJECT_LOG.md` (truth over comfort, no fabrication,
  AI use declared).
- The ARIS skills expect a second model through Codex MCP. Here the second opinion comes from independent Claude
  sessions under `adversarial-review-judge`.
- To update a skill, copy it again from upstream and record the new commit in its `UPSTREAM.md`.
