# Learning this codebase with rederive

`rederive` (vendored in `.claude/skills/rederive`, from EricJujianZou/rederive, MIT) teaches a codebase by having
you design it yourself. It poses each problem the project had to solve, adds the real constraints one at a time,
and never tells you the answer. Use it before interviews and before writing the paper, so that you can defend
every design choice in your own words.

## How to run a session

1. Open the repository in Claude Code and type `/rederive`.
2. The first question is why the project exists. Answer in your own words; that answer becomes the root problem.
3. Work through one problem per sitting. A session ends with a short summary card.

Your progress is kept in `.rederive/log.md`, which is excluded from git so that it stays private. The decision
map the tutor works from is in `.rederive/map.md`. **Don't open it before you have done the problems**, because it
contains the answers. The map was built on 19 Nov 2026 from commit `bfcca82`. It has nine problems, from defining
blame through to making the reported numbers trustworthy. If you reset it (delete `map.md` and `log.md`), the
skill rebuilds it from the current code.

## Why this matters for the applications

Interviewers at NUS, NTU, ETH and MIT ask why you chose a mechanism, not what the code does. Each rederive
problem follows that pattern: a problem, a constraint, why the obvious design breaks, and what replaces it. When
you can solve all nine without hints, you can explain the whole project without notes.
