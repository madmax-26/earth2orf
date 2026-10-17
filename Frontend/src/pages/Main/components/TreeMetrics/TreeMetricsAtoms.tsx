import { atom } from 'jotai'
import { atomFamily } from 'jotai/utils'
import { loadableForest, activeGroupAtom, type GroupStats } from '../Forest/ForestAtoms'

// ── Single-tree metric state ─────────────────────────────────────────────────
//
// What the Tree tab reports about ONE tree: its own user-defined metric values
// and its backend per-group aggregate. Anything averaged over the forest lives
// in ForestMetrics/ForestMetricsAtoms.ts instead — the two tabs never show the
// same number twice.

export type TreeOwnMetrics = {
  /** tree-global user metrics (accuracy, drifts, …) */
  metrics: Record<string, number>
  /** user metrics scoped to the ACTIVE label group / regression target */
  groupMetrics: Record<string, number>
  /** backend per-group aggregate of this tree (purity / mean+std) */
  stat: GroupStats | null
}

export const treeOwnMetricsAtomFamily = atomFamily((treeId: string) =>
  atom((get): TreeOwnMetrics => {
    const state = get(loadableForest)
    if (state.state !== 'hasData') return { metrics: {}, groupMetrics: {}, stat: null }
    const group = get(activeGroupAtom)
    const meta = state.data.trees[treeId]
    return {
      metrics: meta?.metrics ?? {},
      groupMetrics: meta?.group_metrics?.[group.name] ?? {},
      stat: meta?.stats?.[group.index] ?? null,
    }
  })
)
