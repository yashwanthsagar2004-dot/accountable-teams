import { auditGame, backupGame, dividends, valuesFromDividends } from './core.ts'

export type ScenarioId = 'team' | 'synergy' | 'backup' | 'saboteur'
export interface Params { kappa: number; pTeam: number[]; pPair: number[]; delta: number; compensates: boolean }
export const DEFAULTS: Params = { kappa: 1, pTeam: [0.9, 0.7, 0.5, 0.2], pPair: [0.2, 0.9], delta: 0.5, compensates: true }

export interface Built { n: number; names: string[]; roles: string[]; v: Float64Array; m: Float64Array; p: number[] | null; valueWord: string; shareWord: string }

export const SCENARIOS: { id: ScenarioId; tab: string; title: string; blurb: string; question: string; answer: string }[] = [
  { id: 'team', tab: 'Four-agent team', title: 'A planted team with pair and triple synergies',
    blurb: 'Every group contribution (dividend) is planted, so true responsibility is known exactly. Competence is how often each agent acts instead of idling when the team runs on its own.',
    question: 'Predict first: on the team’s own data, which agent gets the most credit, agent 1 or agent 2?',
    answer: 'Agent 2, although agent 1 is more responsible (0.95 against 0.75). Agent 2’s synergies are with partners who act often, so the on-policy fit credits it with their work. Lower agent 1’s competence and watch agent 2’s credit fall while its responsibility stays put.' },
  { id: 'synergy', tab: 'Weak and strong', title: 'Two equally responsible agents, one synergy',
    blurb: 'The two agents contribute the same alone (0.30 each) and share one synergy. They are equally responsible by construction. Only their competence differs.',
    question: 'Predict first: you make the strong agent even more reliable. Whose on-policy credit rises?',
    answer: 'The weak agent’s. Its credit is 0.30 + κ × (the partner’s competence). Credit fitted on team data pays an agent for its partner’s reliability; the gap is κ × (p strong − p weak). Responsibility from executed branches does not move.' },
  { id: 'backup', tab: 'Same log, two teams', title: 'What agent 2 would have done is not in the log',
    blurb: 'Agent 1 does the task; agent 2 confirms. Both teams log exactly that. They differ only in agent 2’s plan for the case that never happened: step in, or let the task fail.',
    question: 'Predict first: can any method that reads only the log tell the two teams apart?',
    answer: 'No. The log is identical, yet agent 1’s responsibility is 0.5 in one team and 1.0 in the other. Only executing the branch where agent 1 is replaced reveals which team you have.' },
  { id: 'saboteur', tab: 'Saboteur under audit', title: 'An agent that behaves when it notices an audit',
    blurb: 'Agent A sabotages the real run. In an audit branch that replaces honest agent X, A notices with probability δ and behaves, so the failure seems to need X.',
    question: 'Predict first: at δ = 1 (audits always noticed), how is blame for the failure split?',
    answer: 'Half and half, although A alone caused it. The shift onto X is δ/2. If audit branches cannot be told from real runs beyond δ, no strategy can shift more than 2δ.' },
]

export function build(id: ScenarioId, prm: Params): Built {
  if (id === 'team') {
    const m = new Float64Array(16)
    m[1] = 0.4; m[2] = 0.3; m[4] = 0.2; m[8] = 0.1; m[3] = 0.5; m[12] = -0.3; m[9] = 0.2; m[7] = 0.6
    return { n: 4, names: ['1', '2', '3', '4'], roles: ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'], v: valuesFromDividends(4, m), m, p: prm.pTeam, valueWord: 'Team return', shareWord: 'Responsibility' }
  }
  if (id === 'synergy') {
    const m = Float64Array.of(0, 0.3, 0.3, prm.kappa)
    return { n: 2, names: ['W', 'S'], roles: ['Weak agent W', 'Strong agent S'], v: valuesFromDividends(2, m), m, p: prm.pPair, valueWord: 'Team return', shareWord: 'Responsibility' }
  }
  if (id === 'backup') {
    const v = backupGame(prm.compensates)
    return { n: 2, names: ['1', '2'], roles: ['Agent 1 (does the task)', 'Agent 2 (acts second)'], v, m: dividends(v, 2), p: null, valueWord: 'Task success', shareWord: 'Responsibility' }
  }
  const v = auditGame(prm.delta)
  return { n: 2, names: ['X', 'A'], roles: ['Honest agent X', 'Adversary A'], v, m: dividends(v, 2), p: null, valueWord: 'Team failure', shareWord: 'Blame' }
}

export const label = (names: string[], s: number): string => {
  const parts = names.filter((_, i) => s & (1 << i))
  return parts.length ? parts.join('·') : 'none'
}
