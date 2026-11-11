// Computation core of the Intervention Bench. Pure functions, no DOM.
// Coalitions are bitmasks: bit i set = agent i keeps its own action; clear = replaced by its declared reference.
// Mirrors src/accteams/shapley.py and theory/check_theory_A*.py; checked against them in core.test.ts.

export type Rng = () => number

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function normal(rng: Rng): number {
  let u = 0
  while (u === 0) u = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng())
}

export const popcount = (x: number): number => {
  let c = 0
  for (; x; x &= x - 1) c++
  return c
}

const FACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
export const choose = (n: number, k: number): number => FACT[n] / (FACT[k] * FACT[n - k])
const harmonic = (k: number): number => {
  let h = 0
  for (let t = 1; t <= k; t++) h += 1 / t
  return h
}

/** v(S) = sum of the dividends m(T) of every group T inside S. `m` is indexed by mask; m[0] is ignored. */
export function valuesFromDividends(n: number, m: ArrayLike<number>): Float64Array {
  const v = new Float64Array(1 << n)
  for (let s = 0; s < 1 << n; s++) {
    let tot = 0
    for (let t = s; t; t = (t - 1) & s) tot += m[t] ?? 0
    v[s] = tot
  }
  return v
}

/** Harsanyi dividends m(T) = sum over R inside T of (-1)^(|T|-|R|) v(R). */
export function dividends(v: ArrayLike<number>, n: number): Float64Array {
  const m = new Float64Array(1 << n)
  for (let t = 0; t < 1 << n; t++) {
    let tot = (popcount(t) % 2 === 0 ? 1 : -1) * v[0]
    for (let r = t; r; r = (r - 1) & t) tot += ((popcount(t) - popcount(r)) % 2 === 0 ? 1 : -1) * v[r]
    m[t] = tot
  }
  m[0] = 0
  return m
}

/** Coefficient of v(S) in agent i's Shapley value. */
export function shapleyCoef(n: number, i: number, s: number): number {
  const k = popcount(s)
  return s & (1 << i) ? (FACT[k - 1] * FACT[n - k]) / FACT[n] : -(FACT[k] * FACT[n - k - 1]) / FACT[n]
}

export function shapley(v: ArrayLike<number>, n: number): number[] {
  const phi = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) for (let s = 0; s < 1 << n; s++) phi[i] += shapleyCoef(n, i, s) * v[s]
  return phi
}

/** The 2^(n-1) marginal contributions of agent i, with their Shapley weights. */
export function marginals(v: ArrayLike<number>, n: number, i: number) {
  const out: { without: number; delta: number; weight: number }[] = []
  for (let s = 0; s < 1 << n; s++) {
    if (s & (1 << i)) continue
    const k = popcount(s)
    out.push({ without: s, delta: v[s | (1 << i)] - v[s], weight: (FACT[k] * FACT[n - k - 1]) / FACT[n] })
  }
  return out
}

/** Order-k readout: phi_i = sum over groups T containing i with |T| <= k of m(T)/|T|. (T2: the error is the dropped tail.) */
export function readout(m: ArrayLike<number>, n: number, k: number): number[] {
  const phi = new Array<number>(n).fill(0)
  for (let t = 1; t < 1 << n; t++) {
    const size = popcount(t)
    if (size > k) continue
    for (let i = 0; i < n; i++) if (t & (1 << i)) phi[i] += m[t] / size
  }
  return phi
}

/** T1b at k = 1: what an additive critic fitted on the team's own data converges to when agent j acts with
 *  probability p[j]: a policy-weighted Banzhaf index, beta_i = sum over T containing i of m(T) * prod_{j in T, j != i} p[j]. */
export function onPolicyLimit(m: ArrayLike<number>, n: number, p: number[]): number[] {
  const beta = new Array<number>(n).fill(0)
  for (let t = 1; t < 1 << n; t++) {
    if (!m[t]) continue
    for (let i = 0; i < n; i++) {
      if (!(t & (1 << i))) continue
      let w = m[t]
      for (let j = 0; j < n; j++) if (j !== i && t & (1 << j)) w *= p[j]
      beta[i] += w
    }
  }
  return beta
}

function solve(a: number[][], b: number[]): number[] | null {
  const k = b.length
  const mtx = a.map((row, r) => [...row, b[r]])
  for (let c = 0; c < k; c++) {
    let piv = c
    for (let r = c + 1; r < k; r++) if (Math.abs(mtx[r][c]) > Math.abs(mtx[piv][c])) piv = r
    if (Math.abs(mtx[piv][c]) < 1e-12) return null
    ;[mtx[c], mtx[piv]] = [mtx[piv], mtx[c]]
    for (let r = 0; r < k; r++) {
      if (r === c) continue
      const f = mtx[r][c] / mtx[c][c]
      for (let cc = c; cc <= k; cc++) mtx[r][cc] -= f * mtx[c][cc]
    }
  }
  return mtx.map((row, r) => row[k] / row[r])
}

/** Additive least squares on N sampled on-policy joint actions with noisy returns. Returns null if an agent never varied. */
export function onPolicyFit(v: ArrayLike<number>, n: number, p: number[], N: number, sigma: number, rng: Rng): number[] | null {
  const k = n + 1
  const ata = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  const aty = new Array<number>(k).fill(0)
  const x = new Array<number>(k).fill(1)
  for (let s = 0; s < N; s++) {
    let mask = 0
    for (let j = 0; j < n; j++) {
      x[j + 1] = rng() < p[j] ? 1 : 0
      if (x[j + 1]) mask |= 1 << j
    }
    const y = v[mask] + sigma * normal(rng)
    for (let r = 0; r < k; r++) {
      aty[r] += x[r] * y
      for (let c = 0; c < k; c++) ata[r][c] += x[r] * x[c]
    }
  }
  const coef = solve(ata, aty)
  return coef ? coef.slice(1) : null
}

/** Probability of one proper, non-empty coalition under the Shapley kernel. */
export function kernelProb(n: number, s: number): number {
  const k = popcount(s)
  if (k === 0 || k === n) return 0
  let norm = 0
  for (let t = 1; t < n; t++) norm += 1 / (t * (n - t))
  return 1 / (k * (n - k)) / norm / choose(n, k)
}

/** T3. Noise factors per unit sigma^2 / B (B = total executed branches for one decision). */
export const kernelZ = (n: number): number => (2 * (n - 1) * harmonic(n - 1)) / n
export const kernelFactor = (n: number): number => (Math.sqrt(kernelZ(n)) + 2 / n) ** 2
export const order2Factor = (n: number): number => (n >= 3 ? 4 * (n - 2) ** 2 : 4)
export const ORDER1_FACTOR = 4

/** Branches per coalition for the kernel design: proper coalitions in proportion to the kernel, the remainder split
 *  between the empty and the full team so that the two parts are balanced as T3 prescribes. */
export function kernelAllocation(n: number, B: number): Float64Array {
  const rootZ = Math.sqrt(kernelZ(n))
  const b1 = (B * rootZ) / (rootZ + 2 / n)
  const alloc = new Float64Array(1 << n)
  const full = (1 << n) - 1
  for (let s = 1; s < full; s++) alloc[s] = Math.max(1, Math.round(b1 * kernelProb(n, s)))
  alloc[0] = alloc[full] = Math.max(1, Math.round((B - b1) / 2))
  return alloc
}

/** Responsibility of every agent from executed branches under the kernel design. A branch return is v(S) plus
 *  Gaussian noise of standard deviation sigma, so the mean of m branches is v(S) + sigma / sqrt(m) * z. */
export function kernelEstimate(v: ArrayLike<number>, n: number, B: number, sigma: number, rng: Rng): number[] {
  const alloc = kernelAllocation(n, B)
  const noisy = new Float64Array(1 << n)
  for (let s = 0; s < 1 << n; s++) noisy[s] = v[s] + (sigma / Math.sqrt(alloc[s])) * normal(rng)
  return shapley(noisy, n)
}

/** Order-2 small-group design for one agent, with its best (Neyman) allocation. Biased by the dropped tail (T2). */
export function order2Estimate(v: ArrayLike<number>, n: number, i: number, B: number, sigma: number, rng: Rng): number {
  const cells: { s: number; c: number }[] = [
    { s: 0, c: -1 + (n - 1) / 2 },
    { s: 1 << i, c: 1 - (n - 1) / 2 },
  ]
  for (let j = 0; j < n; j++) {
    if (j === i) continue
    cells.push({ s: 1 << j, c: -0.5 }, { s: (1 << i) | (1 << j), c: 0.5 })
  }
  const live = cells.filter((x) => Math.abs(x.c) > 1e-12)
  const tot = live.reduce((a, x) => a + Math.abs(x.c), 0)
  let est = 0
  for (const { s, c } of live) {
    const m = Math.max(1, Math.round((B * Math.abs(c)) / tot))
    est += c * (v[s] + (sigma / Math.sqrt(m)) * normal(rng))
  }
  return est
}

/** Permutation audit of one agent: B/2 random orders, two executed branches each. Variance (2 Var[Delta] + 4 sigma^2) / B. */
export function permutationAudit(v: ArrayLike<number>, n: number, i: number, B: number, sigma: number, rng: Rng): number {
  const pairs = Math.max(1, Math.floor(B / 2))
  const order = Array.from({ length: n }, (_, j) => j)
  let tot = 0
  for (let r = 0; r < pairs; r++) {
    for (let j = n - 1; j > 0; j--) {
      const k = Math.floor(rng() * (j + 1))
      ;[order[j], order[k]] = [order[k], order[j]]
    }
    let s = 0
    for (const j of order) {
      if (j === i) break
      s |= 1 << j
    }
    tot += v[s | (1 << i)] - v[s] + sigma * (normal(rng) - normal(rng))
  }
  return tot / pairs
}

export function varDelta(v: ArrayLike<number>, n: number, i: number): number {
  let m1 = 0
  let m2 = 0
  for (const { delta, weight } of marginals(v, n, i)) {
    m1 += weight * delta
    m2 += weight * delta * delta
  }
  return m2 - m1 * m1
}

/** Audit budget from Hoeffding: branches so that |estimate - truth| <= eps with probability 1 - eta, marginals in [-R, R]. */
export const hoeffdingBudget = (R: number, eps: number, eta: number): number => Math.ceil((2 * R * R * Math.log(2 / eta)) / (eps * eps))

/** Relative attribution error, as in the repo: mean |estimate - truth| / mean |truth|. */
export function relErr(est: number[], truth: number[]): number {
  let a = 0
  let b = 0
  for (let i = 0; i < truth.length; i++) {
    a += Math.abs(est[i] - truth[i])
    b += Math.abs(truth[i])
  }
  return b > 0 ? a / b : NaN
}

/** T6/T7 game. Bit 0 = honest agent X, bit 1 = adversary A. A sabotages on the real path; in an audit branch that
 *  replaces X it notices with probability delta and behaves. Value = failure (R = 1). */
export const auditGame = (delta: number): Float64Array => Float64Array.of(0, 0, 1 - delta, 1)

/** T5 game. Bit 0 = agent 1 (does the task), bit 1 = agent 2 (acts after seeing agent 1). On the real path both
 *  teams log the same thing. They differ only in what agent 2 would do if agent 1 had not acted. */
export const backupGame = (compensates: boolean): Float64Array => Float64Array.of(0, 1, compensates ? 1 : 0, 1)

export const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
export function variance(xs: number[]): number {
  const mu = mean(xs)
  return xs.reduce((a, b) => a + (b - mu) ** 2, 0) / xs.length
}
