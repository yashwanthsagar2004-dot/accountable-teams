import type { ReactNode } from 'react'

export interface Series { name: string; color: string; values: (number | null)[]; err?: (number | null)[]; dash?: boolean }
const fmt = (x: number) => (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2))
const nice = (lo: number, hi: number) => {
  const span = hi - lo || 1
  const step = Math.pow(10, Math.floor(Math.log10(span / 4)))
  const s = [1, 2, 5, 10].map((k) => k * step).find((k) => span / k <= 5) ?? step * 10
  const ticks: number[] = []
  for (let t = Math.ceil(lo / s) * s; t <= hi + 1e-9; t += s) ticks.push(+t.toFixed(10))
  return ticks
}

export function Legend({ series, extra }: { series: Series[]; extra?: ReactNode }) {
  return (
    <div className="legend">
      {series.map((s) => (<span key={s.name}><i style={{ background: s.color }} />{s.name}</span>))}
      {extra}
    </div>
  )
}

/** Grouped bars from a zero baseline, a value on every bar, optional ±1 standard error whiskers. */
export function Bars({ groups, series, refLine, height = 230, yLabel }: { groups: string[]; series: Series[]; refLine?: { y: number; label: string }; height?: number; yLabel?: string }) {
  const W = 520, L = 40, R = 8, T = 18, Bm = 34
  const all = series.flatMap((s) => s.values.map((v, i) => (v == null ? [] : [v + (s.err?.[i] ?? 0), v - (s.err?.[i] ?? 0)])).flat())
  const lo = Math.min(0, ...all), hi = Math.max(0.05, ...all, refLine?.y ?? 0) * 1.12
  const y = (v: number) => T + ((hi - v) / (hi - lo)) * (height - T - Bm)
  const gw = (W - L - R) / groups.length
  const bw = Math.min(34, (gw - 14) / series.length - 2)
  return (
    <div className="grid" style={{ gap: 8 }}>
      <Legend series={series} extra={refLine && <span><i style={{ height: 0, width: 14, borderTop: '2px dashed var(--ink-2)', borderRadius: 0 }} />{refLine.label}</span>} />
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={yLabel ?? 'bar chart'} style={{ width: '100%', height: 'auto' }}>
        {nice(lo, hi).map((t) => (<g key={t}><line className="axis" x1={L} x2={W - R} y1={y(t)} y2={y(t)} /><text x={L - 6} y={y(t) + 4} textAnchor="end">{fmt(t)}</text></g>))}
        <line className="zero" x1={L} x2={W - R} y1={y(0)} y2={y(0)} />
        {refLine && <line x1={L} x2={W - R} y1={y(refLine.y)} y2={y(refLine.y)} stroke="var(--ink-2)" strokeDasharray="4 4" />}
        {groups.map((g, gi) => {
          const x0 = L + gi * gw + (gw - series.length * (bw + 2)) / 2
          return (
            <g key={g}>
              {series.map((s, si) => {
                const v = s.values[gi]
                if (v == null) return null
                const x = x0 + si * (bw + 2), top = y(Math.max(v, 0)), h = Math.max(1, Math.abs(y(v) - y(0))), e = s.err?.[gi]
                return (
                  <g key={s.name}>
                    <rect x={x} y={top} width={bw} height={h} rx={3} fill={s.color}><title>{`${g}, ${s.name}: ${fmt(v)}${e ? ` ± ${fmt(e)}` : ''}`}</title></rect>
                    {e ? <line x1={x + bw / 2} x2={x + bw / 2} y1={y(v + e)} y2={y(v - e)} stroke="var(--ink)" strokeWidth={1.5} /> : null}
                    <text className="val" x={x + bw / 2} y={(v >= 0 ? y(v + (e ?? 0)) - 4 : y(v - (e ?? 0)) + 12)} textAnchor="middle">{fmt(v)}</text>
                  </g>
                )
              })}
              <text x={L + gi * gw + gw / 2} y={height - 14} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{g}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** Lines with 8px markers; log or linear x; optional ±sd whiskers and a dashed reference line. */
export function Lines({ x, series, logX, xLabel, yLabel, refLine, height = 250, yMax }: { x: number[]; series: Series[]; logX?: boolean; xLabel: string; yLabel: string; refLine?: { y: number; label: string }; height?: number; yMax?: number }) {
  const W = 520, L = 42, R = 12, T = 12, Bm = 40
  const all = series.flatMap((s) => s.values.map((v, i) => (v == null ? 0 : v + (s.err?.[i] ?? 0))))
  const hi = yMax ?? Math.max(...all, refLine?.y ?? 0) * 1.1
  const fx = (v: number) => (logX ? Math.log(v) : v)
  const x0 = fx(x[0]), x1 = fx(x[x.length - 1])
  const px = (v: number) => L + ((fx(v) - x0) / (x1 - x0 || 1)) * (W - L - R)
  const py = (v: number) => T + ((hi - v) / hi) * (height - T - Bm)
  return (
    <div className="grid" style={{ gap: 8 }}>
      <Legend series={series} />
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${yLabel} against ${xLabel}`} style={{ width: '100%', height: 'auto' }}>
        {nice(0, hi).map((t) => (<g key={t}><line className="axis" x1={L} x2={W - R} y1={py(t)} y2={py(t)} /><text x={L - 6} y={py(t) + 4} textAnchor="end">{fmt(t)}</text></g>))}
        {x.map((v) => (<text key={v} x={px(v)} y={height - 22} textAnchor="middle">{v >= 1000 ? `${v / 1000}k` : v}</text>))}
        <text x={(L + W - R) / 2} y={height - 5} textAnchor="middle">{xLabel}</text>
        {refLine && (<g><line x1={L} x2={W - R} y1={py(refLine.y)} y2={py(refLine.y)} stroke="var(--ink-2)" strokeDasharray="4 4" /><text x={W - R} y={py(refLine.y) - 5} textAnchor="end">{refLine.label}</text></g>)}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (v == null ? null : [px(x[i]), py(Math.min(v, hi))] as const)).filter((p): p is readonly [number, number] => !!p)
          return (
            <g key={s.name}>
              <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash ? '5 4' : undefined} />
              {s.values.map((v, i) => v == null ? null : (
                <g key={i}>
                  {s.err?.[i] ? <line x1={px(x[i])} x2={px(x[i])} y1={py(Math.min(hi, v + s.err[i]!))} y2={py(Math.max(0, v - s.err[i]!))} stroke={s.color} strokeWidth={1.5} /> : null}
                  {!s.dash && <circle cx={px(x[i])} cy={py(Math.min(v, hi))} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2}><title>{`${s.name} at ${x[i]}: ${fmt(v)}${s.err?.[i] ? ` ± ${fmt(s.err[i]!)}` : ''}`}</title></circle>}
                </g>
              ))}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
