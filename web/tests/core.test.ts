import { readFileSync } from 'node:fs'
import * as C from '../src/core.ts'
let fails = 0
const ok = (name: string, cond: boolean, detail = '') => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${name} ${detail}`); if (!cond) fails++ }
const close = (a: number[], b: number[], tol = 1e-9) => a.every((x, i) => Math.abs(x - b[i]) <= tol)
const ref = JSON.parse(readFileSync('web/tests/ref.json', 'utf8'))
let worst = 0
for (const c of ref) {
  const m = new Float64Array(1 << c.n); for (const [k, x] of Object.entries(c.m)) m[+k] = x as number
  const v = C.valuesFromDividends(c.n, m)
  const d = C.dividends(v, c.n)
  const errs = [
    ...Array.from(v).map((x, i) => x - c.values[i]),
    ...C.shapley(v, c.n).map((x, i) => x - c.shapley[i]),
    ...C.shapley(v, c.n).map((x, i) => x - c.kernel_ls[i]),
    ...C.onPolicyLimit(m, c.n, c.p).map((x, i) => x - c.onpolicy[i]),
    ...C.readout(d, c.n, c.n).map((x, i) => x - c.shapley[i]),
    ...Object.entries(c.dividends).map(([k, x]) => d[+k] - (x as number)),
  ]
  worst = Math.max(worst, ...errs.map(Math.abs))
}
ok('12 random games vs repo Python: values, Shapley, kernel least squares, on-policy limit, dividends', worst < 1e-9, `max abs diff ${worst.toExponential(1)}`)

// T4, repo check: m = .3, .3, kappa = 1, p = (.2, .9) -> gap .7
{ const m = Float64Array.of(0, .3, .3, 1); const v = C.valuesFromDividends(2, m); const phi = C.shapley(v, 2); const b = C.onPolicyLimit(m, 2, [.2, .9])
  ok('T4 gap kappa (p_s - p_w)', Math.abs((b[0] - phi[0]) - (b[1] - phi[1]) - 0.7) < 1e-12, `errors ${(b[0]-phi[0]).toFixed(3)}, ${(b[1]-phi[1]).toFixed(3)}`) }
// T6/T7
for (const d of [0, .1, .5, 1]) { const b = C.shapley(C.auditGame(d), 2); ok(`T6 blame shift delta/2 at delta=${d}`, Math.abs(b[0] - d / 2) < 1e-12 && b[0] <= 2 * d + 1e-12) }
// T5
ok('T5 same log, different responsibility', close(C.shapley(C.backupGame(true), 2), [.5, .5]) && close(C.shapley(C.backupGame(false), 2), [1, 0]))
// T3 factors
ok('T3 factors at n = 8', Math.abs(C.kernelZ(8) - 4.5375) < 1e-4 && Math.abs(C.kernelFactor(8) - 5.665) < 2e-3 && C.order2Factor(8) === 144, `${C.kernelFactor(8).toFixed(3)} vs ${C.order2Factor(8)}`)
{ let s = 0; for (let k = 1; k < 15; k++) s += C.kernelProb(4, k); ok('kernel probabilities sum to 1', Math.abs(s - 1) < 1e-12) }
// T2
{ const c = ref[8]; const m = new Float64Array(16); for (const [k, x] of Object.entries(c.m)) m[+k] = x as number
  const phi = C.shapley(C.valuesFromDividends(4, m), 4); const ro = C.readout(m, 4, 2)
  let good = true; for (let i = 0; i < 4; i++) { let tail = 0, mass = 0; for (let t = 1; t < 16; t++) if (t & (1 << i) && C.popcount(t) > 2) { tail += m[t] / C.popcount(t); mass += Math.abs(m[t]) }
    good &&= Math.abs(phi[i] - ro[i] - tail) < 1e-12 && Math.abs(phi[i] - ro[i]) <= mass / 3 + 1e-12 }
  ok('T2 readout error = dropped tail, within tail mass / (k + 1)', good) }
// simulated estimators against the T3 law
{ const c = ref[7]; const n = c.n; const v = Float64Array.from(c.values); const phi = C.shapley(v, n); const rng = C.mulberry32(1); const B = 4000, sigma = 1, reps = 6000
  const k: number[] = [], o: number[] = [], a: number[] = []
  for (let r = 0; r < reps; r++) { k.push(C.kernelEstimate(v, n, B, sigma, rng)[0]); o.push(C.order2Estimate(v, n, 0, B, sigma, rng)); a.push(C.permutationAudit(v, n, 0, B, sigma, rng)) }
  const fk = C.variance(k) * B, fo = C.variance(o) * B, fa = C.variance(a) * B, pa = 2 * C.varDelta(v, n, 0) + 4
  ok('kernel design: unbiased, variance factor matches (sqrt Z + 2/n)^2', Math.abs(C.mean(k) - phi[0]) < 0.003 && Math.abs(fk / C.kernelFactor(n) - 1) < 0.06, `n=${n} measured ${fk.toFixed(2)} predicted ${C.kernelFactor(n).toFixed(2)}`)
  ok('order-2 design: variance factor matches 4 (n - 2)^2', Math.abs(fo / C.order2Factor(n) - 1) < 0.06, `measured ${fo.toFixed(2)} predicted ${C.order2Factor(n)}`)
  ok('permutation audit: unbiased, variance (2 Var + 4 sigma^2) / B', Math.abs(C.mean(a) - phi[0]) < 0.004 && Math.abs(fa / pa - 1) < 0.06, `measured ${fa.toFixed(2)} predicted ${pa.toFixed(2)}`) }
// sampled on-policy fit converges to the closed form, not to responsibility
{ const c = ref[6]; const m = new Float64Array(1 << c.n); for (const [k, x] of Object.entries(c.m)) m[+k] = x as number
  const fit = C.onPolicyFit(Float64Array.from(c.values), c.n, c.p, 400000, 0.5, C.mulberry32(3))!
  ok('sampled on-policy fit converges to the policy-weighted Banzhaf limit', close(fit, c.onpolicy, 0.03), `max diff ${Math.max(...fit.map((x, i) => Math.abs(x - c.onpolicy[i]))).toFixed(4)}`) }
ok('Hoeffding budget', C.hoeffdingBudget(1, 0.1, 0.05) === Math.ceil(2 * Math.log(40) / 0.01))
console.log(fails ? `${fails} FAILED` : 'all passed'); process.exit(fails ? 1 : 0)
