import { useEffect, useMemo, useState } from 'react'
import * as C from './core.ts'
import { Bars, Lines, type Series } from './charts.tsx'
import { Lattice } from './Lattice.tsx'
import { build, DEFAULTS, label, SCENARIOS, type Params, type ScenarioId } from './scenarios.ts'
import ledger from '../../claims/claims.json'
import pilot from '../../experiments/pilot_sweep_2026-11-09.json'
import gate from '../../experiments/colab_gate_2026-11-10.json'
import rware from '../../experiments/colab_rware_2026-11-10.json'
import lbf1 from '../../experiments/lbf_smoke_2026-11-10_window1.json'
import lbf5 from '../../experiments/lbf_smoke_2026-11-10_window5.json'

const COMMIT = '16d07aa'
const SIGMA = 0.5
const f2 = (x: number) => (Object.is(x, -0) ? 0 : x).toFixed(2)
const signed = (x: number) => (Math.abs(x) < 0.005 ? '0.00' : `${x > 0 ? '+' : '−'}${Math.abs(x).toFixed(2)}`)
const col = { truth: 'var(--s-truth)', on: 'var(--s-onpolicy)', branch: 'var(--s-branch)', o2: 'var(--s-order2)', mc: 'var(--s-mc)', grey: 'var(--ink-3)' }
const STATUS: Record<string, string> = { known: 'Known result', checked: 'Checked numerically', exploratory: 'Exploratory', fails: 'Fails as written', untested: 'Not tested yet' }
const Chip = ({ s }: { s: string }) => <span className={`chip ${s}`}><i />{STATUS[s]}</span>

function Slider({ id, text, value, min, max, step, onChange }: { id: string; text: string; value: number; min: number; max: number; step: number; onChange: (x: number) => void }) {
  return (
    <label className="slider" htmlFor={id}>
      <span>{text}</span><span className="num">{value.toFixed(2)}</span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  )
}

export default function App() {
  const [sid, setSid] = useState<ScenarioId>('team')
  const [prm, setPrm] = useState<Params>(DEFAULTS)
  const [cur, setCur] = useState(15)
  const [focus, setFocus] = useState(0)
  const [B, setB] = useState(512)
  const [seed, setSeed] = useState(1)
  const [order, setOrder] = useState(2)
  const sc = SCENARIOS.find((s) => s.id === sid)!
  const g = useMemo(() => build(sid, prm), [sid, prm])
  const full = (1 << g.n) - 1
  const s = cur & full
  const phi = useMemo(() => C.shapley(g.v, g.n), [g])
  const beta = useMemo(() => (g.p ? C.onPolicyLimit(g.m, g.n, g.p) : null), [g])
  const est = useMemo(() => C.kernelEstimate(g.v, g.n, B, SIGMA, C.mulberry32(seed * 7919 + B)), [g, B, seed])
  const se = SIGMA * Math.sqrt(C.kernelFactor(g.n) / B)
  const pick = (id: ScenarioId) => { setSid(id); setCur(id === 'team' ? 15 : 3); setFocus(0) }
  const tryIt = (id: string | null) => { if (id) pick(id as ScenarioId); document.getElementById('bench')?.scrollIntoView() }

  const removed = [...g.m.keys()].filter((t) => t && Math.abs(g.m[t]) > 1e-12 && (t & s) !== t)
  const weak = g.p ? g.p.indexOf(Math.min(...g.p)) : 0
  const strong = g.p ? g.p.indexOf(Math.max(...g.p)) : 0
  const gap = beta ? beta[weak] - phi[weak] - (beta[strong] - phi[strong]) : 0
  const ro = C.readout(g.m, g.n, order)

  let series: Series[]
  if (g.p && beta) {
    series = [
      { name: 'Responsibility (exact, from all branches)', color: col.truth, values: phi },
      { name: 'On-policy credit (limit of a fit on team data)', color: col.on, values: beta },
      { name: `Estimate from ${B} executed branches, ±1 s.e.`, color: col.branch, values: est, err: est.map(() => se) },
    ]
  } else if (sid === 'backup') {
    series = [
      { name: `This team (agent 2 would ${prm.compensates ? 'step in' : 'let it fail'})`, color: col.truth, values: phi },
      { name: `The other team, same log (agent 2 would ${prm.compensates ? 'let it fail' : 'step in'})`, color: col.grey, values: C.shapley(C.backupGame(!prm.compensates), 2) },
    ]
  } else {
    series = [
      { name: 'Blame if audits cannot be noticed (δ = 0)', color: col.truth, values: C.shapley(C.auditGame(0), 2) },
      { name: `Blame measured when A notices audits (δ = ${prm.delta.toFixed(2)})`, color: col.on, values: phi },
    ]
  }

  // ---- Mechanism numbers for the current scenario
  const sizeNorm = Array.from({ length: g.n - 1 }, (_, k) => 1 / ((k + 1) * (g.n - k - 1))).reduce((a, b) => a + b, 0)
  const alloc = C.kernelAllocation(g.n, B)
  const bySize = Array.from({ length: g.n + 1 }, (_, k) => alloc[(1 << k) - 1])
  const R = Math.max(...Array.from({ length: g.n }, (_, i) => Math.max(...C.marginals(g.v, g.n, i).map((x) => Math.abs(x.delta)))))

  // ---- Experiments (seeded; run once in the browser)
  const [nE1, setNE1] = useState(8)
  const e1 = useMemo(() => {
    const rng = C.mulberry32(11 + nE1), n = nE1, m = new Float64Array(1 << n)
    for (let t = 1; t < 1 << n; t++) if (C.popcount(t) <= 3) m[t] = C.normal(rng) * [0, 1, 0.5, 0.3][C.popcount(t)]
    const v = C.valuesFromDividends(n, m), truth = C.shapley(v, n)[0], Bx = 8192, reps = 1200
    const k: number[] = [], o: number[] = [], a: number[] = []
    for (let r = 0; r < reps; r++) { k.push(C.kernelEstimate(v, n, Bx, 1, rng)[0]); o.push(C.order2Estimate(v, n, 0, Bx, 1, rng)); a.push(C.permutationAudit(v, n, 0, Bx, 1, rng)) }
    return {
      predicted: [C.kernelFactor(n), C.order2Factor(n), 2 * C.varDelta(v, n, 0) + 4],
      measured: [C.variance(k) * Bx, C.variance(o) * Bx, C.variance(a) * Bx],
      bias: [C.mean(k) - truth, C.mean(o) - truth, C.mean(a) - truth], reps, Bx,
    }
  }, [nE1])
  const [e2, setE2] = useState<{ x: number[]; kern: number[]; onp: (number | null)[]; bias: number } | null>(null)
  const [e3, setE3] = useState<{ x: number[]; meas: number[]; pred: number[] } | null>(null)
  useEffect(() => {
    const t = setTimeout(() => {
      const team = build('team', DEFAULTS), truth = C.shapley(team.v, 4), rng = C.mulberry32(5)
      const x = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
      const kern: number[] = [], onp: (number | null)[] = []
      for (const N of x) {
        const reps = N > 3000 ? 60 : 150
        const a: number[] = [], b: number[] = []
        for (let r = 0; r < reps; r++) {
          a.push(C.relErr(C.kernelEstimate(team.v, 4, N, SIGMA, rng), truth))
          const fit = C.onPolicyFit(team.v, 4, team.p!, N, SIGMA, rng)
          if (fit) b.push(C.relErr(fit, truth))
        }
        kern.push(C.mean(a)); onp.push(b.length ? C.mean(b) : null)
      }
      setE2({ x, kern, onp, bias: C.relErr(C.onPolicyLimit(team.m, 4, team.p!), truth) })
      const pw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], m = Float64Array.of(0, 0.3, 0.3, 1), v = C.valuesFromDividends(2, m), ph = C.shapley(v, 2)
      setE3({ x: pw, pred: pw.map((p) => 0.9 - p), meas: pw.map((p) => { const fit = C.onPolicyFit(v, 2, [p, 0.9], 40000, SIGMA, rng)!; return fit[0] - ph[0] - (fit[1] - ph[1]) }) })
    }, 0)
    return () => clearTimeout(t)
  }, [])

  // ---- Evidence, read from the repo's records
  const pick3 = (method: string, field: 'mae' | 'mae_sd') => [5000, 20000, 80000].map((b) => pilot.find((r) => r.order === 3 && r.budget === b && r.method.startsWith(method))![field])
  const runStat = (runs: { critic_rel_err: number; mc32_rel_err: number }[], key: 'critic_rel_err' | 'mc32_rel_err') => { const xs = runs.map((r) => r[key]); return [C.mean(xs), Math.sqrt(C.variance(xs))] }
  const c3 = gate.condition_3, rb = rware.condition_3_window_10_basic_features.mean_sd, rr = rware.condition_3_window_10_rich_features.mean_sd
  const critic = [runStat(lbf1.runs, 'critic_rel_err'), runStat(lbf5.runs, 'critic_rel_err'), c3.window_10.mean_sd.critic, c3.window_20.mean_sd.critic, rb.critic, rr.critic]
  const mc = [runStat(lbf1.runs, 'mc32_rel_err'), runStat(lbf5.runs, 'mc32_rel_err'), c3.window_10.mean_sd.mc32, c3.window_20.mean_sd.mc32, rb.mc32, rr.mc32]
  const counts = ledger.claims.reduce<Record<string, number>>((a, c) => ({ ...a, [c.status]: (a[c.status] ?? 0) + 1 }), {})

  return (
    <div className="wrap">
      <header className="top">
        <span className="kicker">Accountable-by-Design Agent Teams · research instrument</span>
        <h1>Intervention Bench</h1>
        <p>When a team of AI agents succeeds or fails, how much did each agent cause it? Replace an agent with its declared reference, watch the team value change, and compare what three methods say about who is responsible.</p>
        <p className="muted small">Every figure in stages 2 to 4 is computed in this page by code that is tested against the project’s Python. Stage 5 reads the project’s run records (repo commit <span className="num">{COMMIT}</span>, 11 Nov 2026). All experimental results are exploratory: 3 seeds, not pre-registered.</p>
      </header>

      <nav className="steps" aria-label="Stages">
        {[['theory', 'Theory'], ['mechanism', 'Mechanism'], ['bench', 'Intervention bench'], ['experiments', 'Experiments'], ['evidence', 'Evidence']].map(([id, name], i) => (
          <span key={id} style={{ display: 'flex', gap: 6 }}>{i > 0 && <span className="arrow" aria-hidden="true">→</span>}<a href={`#${id}`}><span className="n">{i + 1}</span>{name}</a></span>
        ))}
      </nav>

      <section id="theory">
        <div className="sec-head"><span className="n">1</span><h2>Theory: what is claimed</h2></div>
        <div className="card grid">
          <p><strong>Definition.</strong> The responsibility of agent <em>i</em> at a decision is its Shapley value in the interventional game below. The reference is declared in advance: a default action or a certified version of the agent.</p>
          <div className="formula">v(S) = E[ team return | agents in S keep their actions, the others take their declared reference ]</div>
        </div>
        <div className="card scroll-x">
          <table>
            <thead><tr><th>Claim</th><th>In plain words</th><th>Formula</th><th>Status</th><th></th></tr></thead>
            <tbody>
              {ledger.claims.filter((c) => c.kind === 'theorem').map((c) => (
                <tr key={c.id}>
                  <td className="num">{c.id}</td><td style={{ minWidth: 280 }}>{c.statement}</td>
                  <td className="num small" style={{ minWidth: 220 }}>{c.formula}</td><td><Chip s={c.status} /></td>
                  <td><button className="link" onClick={() => tryIt(c.scenario)}>Try it</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">“Checked numerically” means the repo’s check script passes on random games and a reference proof exists. The reference proofs were drafted with AI; the author’s own written proofs are pending, so no theorem here is yet claimed as proved by the author.</p>
      </section>

      <section id="mechanism">
        <div className="sec-head"><span className="n">2</span><h2>Mechanism: how one responsibility number is made</h2><span className="muted small">Live numbers for the scenario selected on the bench ({sc.tab.toLowerCase()}, {g.n} agents, {B} branches).</span></div>
        <div className="stages">
          <div className="card stage"><span className="n">2.1</span><h3>Declare the reference</h3><p className="small muted">Each agent gets a stand-in fixed before any data is seen. Without it the split is not unique (T1c).</p><p className="num small">{g.roles.length} agents, reference = no-op</p></div>
          <div className="card stage"><span className="n">2.2</span><h3>Sample who is replaced</h3><p className="small muted">Coalition sizes are drawn from the Shapley kernel, which favours very small and very large groups.</p><p className="num small">{g.n === 2 ? 'size 1: 100%' : Array.from({ length: g.n - 1 }, (_, k) => `size ${k + 1}: ${((100 / ((k + 1) * (g.n - k - 1))) / sizeNorm).toFixed(0)}%`).join(' · ')}</p></div>
          <div className="card stage"><span className="n">2.3</span><h3>Execute the branches</h3><p className="small muted">The environment is replayed from the same state with those agents replaced. Branches per coalition, by size:</p><p className="num small">{bySize.map((x, k) => `${k}: ${x}`).join(' · ')}</p></div>
          <div className="card stage"><span className="n">2.4</span><h3>Read out responsibility</h3><p className="small muted">A kernel-weighted least-squares fit of the branch returns gives each agent’s share. Exact values here:</p><p className="num small">{phi.map((x, i) => `${g.names[i]}: ${f2(x)}`).join(' · ')}</p></div>
          <div className="card stage"><span className="n">2.5</span><h3>Audit</h3><p className="small muted">Random-order audits check any single number. Branches for ±0.10 at 95% confidence (Hoeffding, largest marginal {f2(R)}):</p><p className="num small">{C.hoeffdingBudget(R, 0.1, 0.05)} permutations</p></div>
        </div>
      </section>

      <section id="bench">
        <div className="sec-head"><span className="n">3</span><h2>Intervention bench</h2></div>
        <div className="tabs" role="tablist" aria-label="Scenario">
          {SCENARIOS.map((x) => <button key={x.id} role="tab" aria-selected={x.id === sid} onClick={() => pick(x.id)}>{x.tab}</button>)}
        </div>
        <p><strong>{sc.title}.</strong> {sc.blurb}</p>
        <div className="bench">
          <div className="card grid">
            <h3>Intervene</h3>
            <div>
              {g.roles.map((role, i) => (
                <div className="agent" key={role}>
                  <span><strong>{role}</strong></span>
                  <span className="switch" role="group" aria-label={`${role}: action`}>
                    <button aria-pressed={!!(s & (1 << i))} onClick={() => setCur(s | (1 << i))}>Own action</button>
                    <button className="ref" aria-pressed={!(s & (1 << i))} onClick={() => setCur(s & ~(1 << i))}>Reference</button>
                  </span>
                  {g.p && <Slider id={`p${i}`} text="Competence (how often it acts)" value={g.p[i]} min={0.05} max={0.95} step={0.05}
                    onChange={(x) => setPrm((q) => sid === 'team' ? { ...q, pTeam: q.pTeam.map((y, j) => (j === i ? x : y)) } : { ...q, pPair: q.pPair.map((y, j) => (j === i ? x : y)) })} />}
                </div>
              ))}
            </div>
            {sid === 'synergy' && <Slider id="kappa" text="Synergy κ between the two" value={prm.kappa} min={-1} max={1} step={0.1} onChange={(x) => setPrm((q) => ({ ...q, kappa: x }))} />}
            {sid === 'saboteur' && <Slider id="delta" text="δ: how often A notices an audit" value={prm.delta} min={0} max={1} step={0.05} onChange={(x) => setPrm((q) => ({ ...q, delta: x }))} />}
            {sid === 'backup' && (
              <div className="grid" style={{ gap: 6 }}><span className="small muted">Agent 2’s plan if agent 1 had not acted (never logged)</span>
                <span className="switch" role="group" aria-label="Unseen plan of agent 2">
                  <button aria-pressed={prm.compensates} onClick={() => setPrm((q) => ({ ...q, compensates: true }))}>Step in</button>
                  <button aria-pressed={!prm.compensates} onClick={() => setPrm((q) => ({ ...q, compensates: false }))}>Let it fail</button>
                </span>
                <span className="small num">Log of both teams: agent 1 acts · agent 2 confirms · success</span>
              </div>
            )}
            {g.p && (
              <div className="grid" style={{ gap: 6 }}>
                <label className="slider" htmlFor="B"><span>Executed branches for the estimate</span><span className="num">{B}</span>
                  <input id="B" type="range" min={5} max={12} step={1} value={Math.log2(B)} onChange={(e) => setB(2 ** +e.target.value)} /></label>
                <button onClick={() => setSeed((x) => x + 1)}>Execute a fresh set of branches</button>
              </div>
            )}
            <button onClick={() => { setPrm(DEFAULTS); setCur(full); setB(512) }}>Reset this bench</button>
          </div>

          <div className="card grid">
            <div className="readings">
              <div className="reading"><span className="small muted">{g.valueWord} with this intervention</span><span className="big">{f2(g.v[s])}</span></div>
              <div className="reading"><span className="small muted">Change from what really happened ({f2(g.v[full])})</span><span className="big">{signed(g.v[s] - g.v[full])}</span></div>
            </div>
            <Lattice n={g.n} names={g.names} v={g.v} current={s} focus={focus} onPick={setCur} />
            <p className="small muted">{s === full ? 'Nobody is replaced. Switch an agent to its reference, or click any box, to execute that branch.'
              : removed.length ? <>Replacing {label(g.names, full & ~s)} removes these group contributions: <span className="num">{removed.map((t) => `{${label(g.names, t)}} ${signed(g.m[t])}`).join(', ')}</span>.</> : 'This intervention removes no group contribution.'}</p>
            <div className="grid" style={{ gap: 8 }}>
              <div className="tabs" role="group" aria-label="Agent to explain"><span className="small muted" style={{ alignSelf: 'center' }}>Explain the share of</span>
                {g.roles.map((r, i) => <button key={r} aria-pressed={focus === i} onClick={() => setFocus(i)}>{g.names[i]}</button>)}</div>
              <div className="scroll-x"><table>
                <thead><tr><th>Others kept</th><th>Without {g.names[focus]}</th><th>With {g.names[focus]}</th><th>It adds</th><th>Weight</th></tr></thead>
                <tbody>{C.marginals(g.v, g.n, focus).map((x) => (
                  <tr key={x.without}><td className="num">{label(g.names, x.without)}</td><td className="num">{f2(g.v[x.without])}</td><td className="num">{f2(g.v[x.without | (1 << focus)])}</td><td className="num">{signed(x.delta)}</td><td className="num">{x.weight.toFixed(3)}</td></tr>))}
                  <tr><td colSpan={3}><strong>{g.shareWord} of {g.names[focus]} = weighted average of what it adds</strong> (thick edges in the diagram)</td><td className="num"><strong>{f2(phi[focus])}</strong></td><td className="num">1.000</td></tr>
                </tbody></table></div>
            </div>
          </div>

          <div className="card grid">
            <h3>{sid === 'saboteur' ? 'Who is blamed' : sid === 'backup' ? 'Who is responsible' : 'What each method reports'}</h3>
            <Bars groups={g.roles.map((r) => r.replace(/ \(.*\)/, ''))} series={series} yLabel={g.shareWord} />
            {g.p && beta && (
              <div className="readings">
                <div className="reading"><span className="small muted">On-policy credit, relative error</span><span className="big">{f2(C.relErr(beta, phi))}</span></div>
                <div className="reading"><span className="small muted">Estimate from branches, relative error</span><span className="big">{f2(C.relErr(est, phi))}</span></div>
                <div className="reading"><span className="small muted">Over-credit of the weakest ({g.names[weak]}) against the strongest ({g.names[strong]})</span><span className="big">{signed(gap)}</span>
                  {sid === 'synergy' && <span className="small num">κ (p_S − p_W) = {signed(prm.kappa * (prm.pPair[1] - prm.pPair[0]))}</span>}</div>
              </div>
            )}
            {sid === 'team' && (
              <div className="grid" style={{ gap: 6 }}><div className="tabs" role="group" aria-label="Readout order"><span className="small muted" style={{ alignSelf: 'center' }}>Readout from groups of at most</span>
                {[1, 2, 3].map((k) => <button key={k} aria-pressed={order === k} onClick={() => setOrder(k)}>{k}</button>)}</div>
                <span className="small num">error per agent: {ro.map((x, i) => signed(x - phi[i])).join(' · ')} (exactly the dropped tail, T2)</span></div>
            )}
            {sid === 'saboteur' && (
              <div className="readings">
                <div className="reading"><span className="small muted">Blame shifted onto honest X (δ/2)</span><span className="big">{f2(phi[0])}</span></div>
                <div className="reading"><span className="small muted">Most any strategy can shift if audits are indistinguishable up to δ (2δ, capped at 1)</span><span className="big">{f2(Math.min(1, 2 * prm.delta))}</span></div>
              </div>
            )}
            {sid === 'backup' && <p className="small muted">Both bars are consistent with the same log. Switch agent 1 to its reference on the left: the value of that branch is the only thing that differs between the two teams.</p>}
            <details className="predict"><summary>{sc.question}</summary><p className="small">{sc.answer}</p></details>
          </div>
        </div>
      </section>

      <section id="experiments">
        <div className="sec-head"><span className="n">4</span><h2>Experiments run in this page</h2><span className="muted small">Seeded simulations of the estimators on planted games. They test the theory’s predictions; they are not the learned-critic results of stage 5.</span></div>
        <div className="three">
          <div className="card grid">
            <h3>Branch-design law (T3)</h3>
            <p className="small muted">Variance of agent 1’s estimate × B / σ², predicted and measured over {e1.reps} repetitions at B = {e1.Bx}.</p>
            <div className="tabs" role="group" aria-label="Number of agents"><span className="small muted" style={{ alignSelf: 'center' }}>Agents</span>{[4, 6, 8].map((k) => <button key={k} aria-pressed={nE1 === k} onClick={() => setNE1(k)}>{k}</button>)}</div>
            <Bars groups={['Kernel design', 'Order-2 groups', 'Permutation audit']} series={[{ name: 'Predicted by T3', color: col.grey, values: e1.predicted }, { name: 'Measured', color: col.branch, values: e1.measured }]} yLabel="noise factor" />
            <p className="small num">bias of the mean: {e1.bias.map(signed).join(' · ')} (order 2 drops the triple terms)</p>
          </div>
          <div className="card grid">
            <h3>Plateau against convergence (T1b)</h3>
            <p className="small muted">Four-agent team, return noise σ = {SIGMA}. Same number of returns for both: on-policy samples, or executed branches.</p>
            {e2 ? <Lines x={e2.x} logX xLabel="returns used" yLabel="relative error" refLine={{ y: e2.bias, label: `on-policy bias ${f2(e2.bias)}` }}
              series={[{ name: 'Additive fit on team data', color: col.on, values: e2.onp }, { name: 'Kernel design on executed branches', color: col.branch, values: e2.kern }]} /> : <p className="small muted">Running…</p>}
          </div>
          <div className="card grid">
            <h3>Fairness gap (T4)</h3>
            <p className="small muted">Two equally responsible agents, κ = 1, strong agent at 0.9. Over-credit of the weak agent from a fit on 40,000 on-policy returns.</p>
            {e3 ? <Lines x={e3.x} xLabel="competence of the weak agent" yLabel="over-credit" yMax={0.9}
              series={[{ name: 'Predicted κ (p_S − p_W)', color: col.grey, values: e3.pred, dash: true }, { name: 'Measured from the fit', color: col.on, values: e3.meas }]} /> : <p className="small muted">Running…</p>}
          </div>
        </div>
      </section>

      <section id="evidence">
        <div className="sec-head"><span className="n">5</span><h2>Evidence: what the project’s runs show so far</h2></div>
        <div className="readings card">
          {Object.keys(STATUS).map((k) => <div className="reading" key={k}><Chip s={k} /><span className="big">{counts[k] ?? 0}</span><span className="small muted">of {ledger.claims.length} claims</span></div>)}
        </div>
        <div className="two">
          <div className="card grid">
            <h3>Planted games: error against budget</h3>
            <p className="small muted">4 agents, order-3 games, learned critics, mean ± sd over 3 seeds. Lower is better.</p>
            <Lines x={[5000, 20000, 80000]} logX xLabel="executed returns" yLabel="relative error" yMax={0.85} series={[
              { name: 'Additive, on-policy', color: col.on, values: pick3('additive', 'mae'), err: pick3('additive', 'mae_sd') },
              { name: 'Order-2 critic', color: col.o2, values: pick3('order-2', 'mae'), err: pick3('order-2', 'mae_sd') },
              { name: 'Kernel critic (executed branches)', color: col.branch, values: pick3('kernel', 'mae'), err: pick3('kernel', 'mae_sd') }]} />
          </div>
          <div className="card grid">
            <h3>Simulated tasks: the amortised critic against 32 branches per decision</h3>
            <p className="small muted">Scripted teams, intervention window in steps, mean ± sd over 3 seeds. The critic must beat Monte Carlo to pass the gate.</p>
            <Bars height={270} groups={['LBF 1', 'LBF 5', 'LBF 10', 'LBF 20', 'RWARE 10', 'RWARE 10+']} refLine={{ y: 1, label: 'predicting zero = 1.00' }} yLabel="relative error" series={[
              { name: 'Kernel critic', color: col.branch, values: critic.map((x) => x[0]), err: critic.map((x) => x[1]) },
              { name: 'Monte Carlo, 32 branches', color: col.mc, values: mc.map((x) => x[0]), err: mc.map((x) => x[1]) }]} />
            <p className="small muted">LBF = Level-Based Foraging; RWARE = RobotWarehouse; 10+ = richer critic inputs. The gate passes on LBF at 20 steps and fails on RWARE.</p>
          </div>
        </div>
        <div className="card scroll-x">
          <table>
            <thead><tr><th>Claim</th><th>Status</th><th>Statement</th><th>Numbers in the records</th></tr></thead>
            <tbody>{ledger.claims.filter((c) => c.kind === 'experiment').map((c) => (
              <tr key={c.id}><td className="num">{c.id}</td><td><Chip s={c.status} /></td><td style={{ minWidth: 300 }}>{c.statement}</td>
                <td className="num small" style={{ minWidth: 240 }}>{c.evidence.length ? c.evidence.map((e) => <div key={e.label}>{e.label}: {e.value}</div>) : '—'}</td></tr>))}
            </tbody>
          </table>
        </div>
        <p className="muted small">The table lists the experimental claims. The theorems and their status are in stage 1. Both come from the same ledger.</p>
        <div className="card grid">
          <h3>What this evidence does not show</h3>
          <ul className="plain small">
            <li>No result is confirmatory. Every run has 3 seeds; the pre-registered tests need 10 seeds, bootstrap intervals and Holm correction.</li>
            <li>All teams so far are planted or scripted. Nothing has been tested on learned (MAPPO) teams or on the LLM-team testbed.</li>
            <li>The amortised critic loses to plain Monte Carlo on RobotWarehouse. Where it is worth using is an open question, not a result.</li>
            <li>The Phase 2 claims about audits (H6) have no experiment yet. The saboteur scenario above is the theorem’s own two-agent example.</li>
            <li>Two tier-2 records were transcribed by hand from Colab output; they need one machine-written re-run.</li>
          </ul>
        </div>
      </section>

      <footer>
        <span>Code and records: <a href="https://github.com/yashwanthsagar2004-dot/accountable-teams">github.com/yashwanthsagar2004-dot/accountable-teams</a>. The claims table is generated from <span className="num">claims/claims.json</span> and checked against the records by <span className="num">tools/verify_claims.py</span>.</span>
        <span>AI use: this page, the reference proofs and the starter code were drafted with Claude (Anthropic) at the author’s request. The author is responsible for every claim.</span>
      </footer>
    </div>
  )
}
