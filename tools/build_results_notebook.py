"""Build notebooks/results_v3.ipynb: the v3 GPU records, each followed by a dated plain-words observation.

Run from the repository root:  python tools/build_results_notebook.py
Each section reads a machine-written record in experiments/ and prints its numbers; the observation under it is
written by hand and dated. Add a section when a run finishes; never edit an old observation, add a dated note instead.
"""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells += [md(
    "# Version-3 GPU results, with observations\n\n"
    "Every number below is read from a machine-written record in `experiments/` that carries its commit and config "
    "fingerprint. Under each record is an **Observation after successful output**: what the output says in plain "
    "words, what it means for the project, and what it would mean in the real world. Observations are dated and "
    "never rewritten; later notes are added beneath them.\n\n"
    "How to read relative error: 0 is perfect; 1 is as bad as answering \"nobody is responsible\" for every decision."),
    code(
    "import json\n"
    "def show(path):\n"
    "    d = json.load(open(path))\n"
    "    print('commit', d['fingerprint']['commit'], '| dirty', d['fingerprint']['dirty'], '| config', d['fingerprint']['config_hash'])\n"
    "    for r in d['runs']:\n"
    "        print(f\"\\nseed {r['seed']}: {r['agents']} agents, ground truth {r['gt_mode']}, truth noise {r['ground_truth_se_rel']:.1%} of the signal, \"\n"
    "              f\"{r['seconds'] / 60:.0f} min\")\n"
    "        cost = r['audit_branches_per_decision']\n"
    "        print(f\"  {'estimator':14s} {'reruns/decision':>16s} {'rel. error':>11s} {'top quarter':>12s}\")\n"
    "        for k, v in sorted(r['rel_err'].items(), key=lambda kv: kv[1]):\n"
    "            print(f\"  {k:14s} {cost.get(k, 0):16.1f} {v:11.3f} {r['rel_err_top_quarter'][k]:12.3f}\")\n"
    "        print('  critic worth (Monte Carlo reruns, extrapolated):', {k: round(v, 1) for k, v in r['mc_equivalent_branches'].items()})\n"
    "        print('  ensemble uncertainty vs error (Spearman):', round(r['uncertainty_error_spearman'], 2),\n"
    "              '| most-responsible agent found:', f\"{r['top_agent_match']:.0%} (chance 25%)\")"),
]

nb.cells += [md(
    "## Level-Based Foraging gate (4 agents, window = horizon = 20), seed 0 of 3\n\n"
    "Record: `experiments/critic_budget_lbf_w20_v3.seed0.partial.json` (copied from Drive; seeds 1 and 2 still to run)."),
    code("show('experiments/critic_budget_lbf_w20_v3.seed0.partial.json')"),
    md(
    "### Observation after successful output\n\n"
    "**Recorded:** 25 Nov 2026, 07:35 IST. **Run finished:** 20 Nov 2026, 09:12 IST (03:42 UTC), Colab T4, "
    "commit `f7fca03`, clean tree, critic loss with the 19 Nov fix.\n\n"
    "**What the output says.** The true responsibility of each of the 4 agents was measured exactly on 100 held-out "
    "decisions (16 groups of agents x 200 reruns each; the measurement noise is 7.5% of the signal). On the 70 test "
    "decisions:\n\n"
    "- The **flat critic** answers with no reruns at all and has relative error **0.67**. Plain Monte Carlo needs "
    "**64 reruns per decision** to match it (0.67) and is worse with 32 (0.90). Extrapolated, the critic is worth about "
    "60 reruns per decision. **The gate condition (critic beats 32-rerun Monte Carlo) passes on this seed.**\n"
    "- The new **set-critic ensemble** (attention over agents) has error **1.03**: no better than answering zero. It "
    "fails here. One set critic alone is worse (1.79).\n"
    "- **Critic + reruns combined** (precision-weighted) is the most accurate: 0.75 with 32 reruns (vs 0.90 for reruns "
    "alone) and **0.47 with 128 reruns**, the best number in the run.\n"
    "- The ensemble's disagreement does track its own error (Spearman 0.53), but it names the single most responsible "
    "agent correctly only 37% of the time (chance is 25%).\n\n"
    "**What it means for the project.** This is the first gate result that is machine-written, fingerprinted and uses "
    "the fixed loss. On one seed, a learned critic replaces dozens of expensive reruns, which is the project's central "
    "claim on a real game. Two warnings stay attached: the fancier set critic does not help at 4 agents (it was built "
    "for 8 to 16 agents, which is where it must earn its place), and an error of 0.67 means the critic's mistakes are "
    "two thirds the size of the responsibilities themselves. Still exploratory: 1 seed of 3, the data are split by time "
    "step rather than by episode (review item C5), and this record has no shrunk-Monte-Carlo baseline.\n\n"
    "**What it means in the real world.** For a team of agents that can be replayed (a simulator, a logged LLM "
    "pipeline), a trained critic can give a first estimate of who is responsible for every decision at almost no cost, "
    "and real replays can be spent where it matters; using both together gives the best accuracy. At this accuracy it "
    "is a screening tool that says where to look, not a verdict that says who is to blame."),
]

nb.cells += [md(
    "## Level-Based Foraging gate, seed 1 of 3\n\n"
    "Record: `experiments/critic_budget_lbf_w20_v3.seeds01.partial.json` (same run, same commit and config; it holds "
    "seeds 0 and 1). Seed 1 resumed on 25 Nov from its saved stages (training branches and flat critic of 20 Nov)."),
    code("show('experiments/critic_budget_lbf_w20_v3.seeds01.partial.json')"),
    md(
    "### Observation after successful output\n\n"
    "**Recorded:** 25 Nov 2026, 08:48 IST. **Run finished:** 25 Nov 2026, 08:42 IST (03:12 UTC), Colab T4, "
    "commit `f7fca03`, clean tree.\n\n"
    "**What the output says.** Seed 1 repeats the pattern of seed 0, more weakly:\n\n"
    "- The **flat critic** (no reruns) has error **0.85**, again better than 32-rerun Monte Carlo (0.93) but worse than "
    "64 reruns (0.72). It is worth about 40 reruns per decision. **The gate condition passes again: 2 of 2 seeds.**\n"
    "- The **set-critic ensemble** is again no better than answering zero (1.15). It fails on both seeds.\n"
    "- Combining critic and reruns helps at 32 reruns (0.79 vs 0.93) but not at 128 (0.61 vs 0.56): with many reruns, "
    "the reruns alone are already better than the critic can add.\n"
    "- The ensemble's disagreement tracks its error a little better than on seed 0 (Spearman 0.61); it still finds the "
    "most responsible agent only 37% of the time.\n\n"
    "Mean over the two seeds: flat critic 0.76, 32-rerun Monte Carlo 0.92, 64-rerun Monte Carlo 0.70.\n\n"
    "**What it means for the project.** The cheap critic's advantage over a modest rerun budget holds on a second "
    "seed, so the seed-0 result was not a fluke. It is a narrow advantage (roughly the value of 32 to 64 reruns), and "
    "the attention-based set critic needs work before the 8- and 16-agent experiments, where it is supposed to matter "
    "most. The verdict waits for seed 2.\n\n"
    "**What it means in the real world.** Same as seed 0: a trained critic is a cheap first pass over every decision, "
    "and the replay budget should go to the decisions that matter. Where replays are plentiful, they beat the critic."),
]

NotebookClient(nb, timeout=120, kernel_name="python3").execute()
nbf.write(nb, "notebooks/results_v3.ipynb")
print("wrote notebooks/results_v3.ipynb")
