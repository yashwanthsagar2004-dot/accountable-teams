import { popcount } from './core.ts'
import { label } from './scenarios.ts'

/** Every coalition of the team, one row per size. An edge adds one agent. Click a node to execute that branch. */
export function Lattice({ n, names, v, current, focus, onPick }: { n: number; names: string[]; v: Float64Array; current: number; focus: number; onPick: (s: number) => void }) {
  const W = n === 4 ? 500 : 420, rowH = n === 4 ? 76 : 88, pad = 26, bw = n === 4 ? 66 : 92, bh = 40
  const H = pad * 2 + rowH * n + bh
  const rows: number[][] = Array.from({ length: n + 1 }, () => [])
  for (let s = 0; s < 1 << n; s++) rows[popcount(s)].push(s)
  const pos = new Map<number, [number, number]>()
  rows.forEach((row, k) => row.forEach((s, j) => pos.set(s, [(W * (j + 1)) / (row.length + 1), H - pad - bh / 2 - k * rowH])))
  const edges: { a: number; b: number; i: number }[] = []
  for (let s = 0; s < 1 << n; s++) for (let i = 0; i < n; i++) if (!(s & (1 << i))) edges.push({ a: s, b: s | (1 << i), i })
  const fact = [1, 1, 2, 6, 24]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Lattice of coalitions with the team value of each" style={{ width: '100%', height: 'auto' }}>
      {edges.map(({ a, b, i }) => {
        const [x1, y1] = pos.get(a)!, [x2, y2] = pos.get(b)!
        const hot = i === focus, k = popcount(a), w = (fact[k] * fact[n - k - 1]) / fact[n]
        return <line key={`${a}-${b}`} className={hot ? 'edge hot' : 'edge'} x1={x1} y1={y1 - bh / 2} x2={x2} y2={y2 + bh / 2} strokeWidth={hot ? 1.5 + 7 * w : 1} opacity={hot ? 0.9 : 0.7} />
      })}
      {[...pos.entries()].map(([s, [x, y]]) => (
        <g key={s} className={s === current ? 'node on' : 'node'} tabIndex={0} role="button" aria-pressed={s === current}
          aria-label={`Keep ${label(names, s)}; value ${v[s].toFixed(2)}`} onClick={() => onPick(s)}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onPick(s) } }}>
          <rect x={x - bw / 2} y={y - bh / 2} width={bw} height={bh} rx={6} />
          <text x={x} y={y - 4} textAnchor="middle">{label(names, s)}</text>
          <text className="v" x={x} y={y + 12} textAnchor="middle">{v[s].toFixed(2)}</text>
        </g>
      ))}
      <text x={6} y={14}>all keep their action</text>
      <text x={6} y={H - 6}>all replaced by the reference</text>
    </svg>
  )
}
