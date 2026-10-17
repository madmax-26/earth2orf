import { atom } from 'jotai'
import {
  loadableForest, activeGroupAtom, type GroupStats,
} from '../Forest/ForestAtoms'

// ── Forest-level metric state ────────────────────────────────────────────────
//
// Everything the Forest tab reports: aggregates ACROSS the trees, both the
// standard ones the backend computes (purity / mean / std per label group) and
// the user-defined per-tree metrics (accuracy, mae, …) summarised here.
// The Tree tab shows a single tree's own values; anything averaged over the
// forest lives in this module.

export const activeForestStatAtom = atom((get): GroupStats | null => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return null
  return state.data.forest_stats?.[get(activeGroupAtom).index] ?? null
})

// ── Custom-metric aggregates ─────────────────────────────────────────────────
//
// Custom per-tree metrics (accuracy, mae, …) get the same forest-level summary
// as the built-in structural ones (height, node count): mean, standard
// deviation and range across the individual trees. σ is the population form
// (÷ n) — these n trees are the whole population, not a sample of one.
//
// Always a PER-TREE summary: the mean of what each tree scores alone, never the
// ensemble's score (a forest averages its trees' posteriors and normally scores
// well above any of them).

export type MetricStats = { mean: number; std: number; min: number; max: number; n: number }

/** Counts (tree_age, drifts) stay bare; scores (accuracy, mae) get 3 decimals. */
export function fmtMetric(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(3)
}

function statsOf(values: number[]): MetricStats {
  const n = values.length
  let sum = 0, min = Infinity, max = -Infinity
  for (const v of values) {
    sum += v
    if (v < min) min = v
    if (v > max) max = v
  }
  const mean = sum / n
  let sq = 0
  for (const v of values) sq += (v - mean) ** 2
  return { mean, std: Math.sqrt(sq / n), min, max, n }
}

/** Column-wise stats over one metric bag per tree; keys absent on a tree are
    simply not counted for that key. */
function metricStatsFrom(perTree: Record<string, number>[]): Record<string, MetricStats> {
  const cols: Record<string, number[]> = {}
  for (const bag of perTree) {
    for (const [key, v] of Object.entries(bag)) {
      if (typeof v !== 'number' || !Number.isFinite(v)) continue
      const col = cols[key]
      if (col) col.push(v)
      else cols[key] = [v]
    }
  }
  const out: Record<string, MetricStats> = {}
  for (const [key, values] of Object.entries(cols)) {
    if (values.length) out[key] = statsOf(values)
  }
  return out
}

/** Forest-wide stats for every tree-global custom metric (accuracy, drift, …). */
export const treeMetricStatsAtom = atom((get): Record<string, MetricStats> => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return {}
  return metricStatsFrom(Object.values(state.data.trees).map(m => m.metrics ?? {}))
})

/** Same, for the custom metrics scoped to the ACTIVE label group / target. */
export const activeGroupMetricStatsAtom = atom((get): Record<string, MetricStats> => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return {}
  const name = get(activeGroupAtom).name
  return metricStatsFrom(Object.values(state.data.trees).map(m => m.group_metrics?.[name] ?? {}))
})

// Structural spread of the forest — the same mean/σ/range treatment the custom
// metrics get, so the Forest tab reports every aggregate the same way.
export const forestSizeStatsAtom = atom((get): {
  count: number
  entries: [string, { height: number; n_nodes: number }][]
  height: MetricStats
  nodes: MetricStats
} | null => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return null
  const entries = Object.entries(state.data.trees)
  if (!entries.length) return null
  return {
    count: entries.length,
    entries: entries.map(([id, m]) => [id, { height: m.height, n_nodes: m.n_nodes }]),
    height: statsOf(entries.map(([, m]) => m.height)),
    nodes: statsOf(entries.map(([, m]) => m.n_nodes)),
  }
})
