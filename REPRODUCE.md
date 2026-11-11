# Reproduce

Every number this project quotes is listed in `claims/claims.json`, read from a record in `experiments/`, and
re-checked by `python tools/verify_claims.py` (CI runs it on every push). `CLAIMS.md` is the generated table.

## Environment

Python 3.10 or later. `pip install -e ".[dev]"` for the theory and tier 1; add `jumanji` for tier 2.
CPU is enough for everything in the first table. Tier 2 was run on a Colab Tesla T4.

## Commands

| What | Command | Hardware, time | Record |
|---|---|---|---|
| Unit tests | `pytest -q` | CPU, under 1 min | - |
| Theorems T1-T7, numerical checks | `python tools/verify_claims.py --checks` | CPU, about 1 min | - |
| Tier-1 pilot (H1-tier1, T4 gap) | `python experiments/pilot_sweep.py` | CPU, about 2 min | `experiments/pilot_sweep_2026-11-09.json` |
| LBF, one-step and five-step windows | `python experiments/lbf_smoke.py --decisions 300000 --eval 100 --reps 200 --seeds 3 --train-steps 20000 --batch 2048 --patience 100000 --window W` | CPU about 3 min a seed; T4 60-90 s | `experiments/lbf_smoke_2026-11-10_window{1,5}.json` |
| LBF gate, conditions 1-2 | `python experiments/snr_gate.py --diagonal --windows 5,10,15,20,25 --horizons 5,10,15,20,25 --eval 60 --reps 400` | T4 | `experiments/colab_gate_2026-11-10.json` |
| LBF gate, condition 3 | the LBF command with `--window 20 --horizon 20` | T4 | same file |
| RobotWarehouse gate | the two commands above with `--env rware`; condition 3 adds `--train-steps 40000 --batch 4096 --hidden 256 --window 10 --horizon 10` | T4 | `experiments/colab_rware_2026-11-10.json` |
| Anonymised supplement | `python tools/make_supplement.py` | CPU | `supplement.zip` |

## What is not yet reproducible to this standard

- The two `colab_*.json` records were transcribed by hand from cell output; the session files were lost. New runs
  write a `provenance` block (commit, command, configuration hash, device, versions, seconds), and the last cell of
  `notebooks/colab_runner.ipynb` downloads every record as a zip. Re-run both gates once so the records are
  machine-written, then delete this bullet.
- All results use 3 seeds and are exploratory. The confirmatory runs use 10 seeds and `accteams.stats`
  (interquartile mean, stratified bootstrap intervals, probability of improvement, Holm correction), after the
  `prereg-v1` tag.
- Compute so far is not totalled. From the next run on, sum `provenance.seconds` over the records, and report
  failed and preliminary runs too.
