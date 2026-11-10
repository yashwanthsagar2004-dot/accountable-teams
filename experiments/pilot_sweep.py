"""Pilot sweep for the Implementation tab: error against branch budget, for planted games of order 2 and 3."""
import json, sys, time
import numpy as np
sys.path.insert(0, 'src')
from accteams.train_tier1 import run

rows = []
t0 = time.time()
for order in (2, 3):
    for budget in (5000, 20000, 80000):
        res = [run(s, budget, 4, order, 6000) for s in range(3)]
        for name in res[0]:
            mae = np.array([r[name]['mae'] for r in res]); gap = np.array([r[name]['gap'] for r in res])
            rows.append(dict(order=order, budget=budget, method=name, mae=float(mae.mean()), mae_sd=float(mae.std()), gap=float(gap.mean()), gap_sd=float(gap.std())))
            print(f"order {order} budget {budget:6d} {name:34s} err {mae.mean():.3f}+/-{mae.std():.3f} gap {gap.mean():+.3f}+/-{gap.std():.3f}", flush=True)
json.dump(rows, open('experiments/pilot_sweep.json', 'w'), indent=1)
print(f'done in {time.time()-t0:.0f}s')
