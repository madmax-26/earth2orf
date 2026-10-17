import { atom, type Atom } from "jotai"
import { atomFamily, unwrap } from "jotai/utils"
import { selectedIterationAtom } from "../Iteration/IterationAtoms"
// Backend address — configurable via the StatusBar settings dialog. Called per
// fetch (never captured in a const) so a changed setting takes effect at once.
import { apiBase } from "../../../../apiConfig"

// Backend-computed per-group aggregates (group-ordinal aligned):
// classification → purity (1 − sample-weighted leaf Gini); regression → mean
// (root prediction) + std (sample-weighted leaf σ).
export type GroupStats = { purity?: number; mean?: number; std?: number }

export type TreeMeta = {
  height: number
  n_nodes: number
  // Tree-global custom metrics (drifts, tree_age, …)
  metrics: Record<string, number>
  // Group-scoped custom metrics keyed by group/target NAME (accuracy per label
  // group, mae per regression target, …)
  group_metrics?: Record<string, Record<string, number>>
  // Per-tree aggregates, aligned with the group ordinals
  stats?: GroupStats[]
}
// Per-GROUP kind: even in a mixed forest every group is purely one of these.
export type TaskType = 'classification' | 'regression'
// Forest-level task; 'mixed' = class_labels dict groups first, then one
// regression target per target_labels entry, all on the same tree nodes.
export type ForestTask = TaskType | 'mixed'
// class_labels: flat list (single group) or {group_name: labels[]} (multi-group).
// Regression: target_labels lists the target names; ranges are per-target arrays
// in the multi-target case (and always per-target for mixed).
export type ForestData = {
  class_labels: string[] | Record<string, string[]>
  class_colors: string[] | Record<string, string[]>
  trees: Record<string, TreeMeta>
  task?: ForestTask
  target_range?: [number, number] | [number, number][] | null
  std_range?: [number, number] | [number, number][] | null
  target_label?: string
  target_labels?: string[] | null
  // Forest-level per-group aggregates (average over trees), group-ordinal aligned
  forest_stats?: GroupStats[] | null
}

export type TreeNodeData = {
  id: number
  depth: number
  n_node_samples: number
  // Scalar for single-group forests; one entry per label group / target otherwise
  impurity: number | number[]
  // Single group: histogram (or [mean]). Multi-group: concatenated histograms /
  // one mean per target — slice with sliceValue(value, group).
  value: number[]
}

// ── Label groups (multi-label / multi-target) ────────────────────────────────

export type GroupSpec = {
  name: string
  labels: string[]     // class labels of the group; [] for regression targets
  offset: number       // start of the group's span in TreeNodeData.value
  size: number         // span length; -1 = whole value vector (legacy unknown count)
  index: number        // group ordinal — indexes per-group impurity arrays, sent as ?group=
  kind: TaskType       // what this group renders as (pie/histogram vs gradient)
  regIndex: number     // position in target_labels/target_range/std_range; -1 for classification
}

/** The group's slice of a node's value vector. */
export function sliceValue(value: number[], g: GroupSpec): number[] {
  if (g.size < 0 || (g.offset === 0 && g.size >= value.length)) return value
  return value.slice(g.offset, g.offset + g.size)
}

/** The group's impurity (Gini / std) from a scalar-or-per-group field. */
export function impurityAt(imp: number | number[], groupIndex: number): number {
  return typeof imp === 'number' ? imp : (imp[groupIndex] ?? 0)
}

export type LoadableState<T> =
  | { state: 'loading' }
  | { state: 'hasError'; error: unknown }
  | { state: 'hasData'; data: T }


const PENDING: unique symbol = Symbol('pending')

export function createLoadable<T>(asyncAtom: Atom<T>): Atom<LoadableState<Awaited<T>>> {
  const unwrappedAtom = unwrap(asyncAtom, (): typeof PENDING => PENDING)
  return atom((get): LoadableState<Awaited<T>> => {
    try {
      const value = get(unwrappedAtom)
      if (value === PENDING) return { state: 'loading' }
      return { state: 'hasData', data: value }
    } catch (error) {
      return { state: 'hasError', error }
    }
  })
}

// Forest meta for the currently selected iteration
const asyncForestAtom = atom(async (get): Promise<ForestData> => {
  const iter = get(selectedIterationAtom)
  if (iter === null) return { class_labels: [], class_colors: [], trees: {} }
  const res = await fetch(`${apiBase()}/iteration/${iter}/meta`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
})

export const loadableForest = createLoadable(asyncForestAtom)

// Label groups of a forest, with their value-vector spans and per-group kind.
// Single-group forests (flat class_labels / one target) yield exactly one spec,
// so every consumer can treat the single- and multi-group cases uniformly.
//
// ORDINAL INVARIANT: the order produced here (classification groups in
// class_labels insertion order, then regression targets) must match the
// backend's _group_specs in treeflow.py — GroupSpec.index is sent as ?group= and
// indexes the per-node impurity array.
export function groupsFromForest(d: ForestData): GroupSpec[] {
  const task = d.task ?? 'classification'
  if (task === 'regression') {
    const names = d.target_labels ?? [d.target_label ?? '']
    return names.map((name, i) => ({
      name, labels: [], offset: i, size: 1, index: i,
      kind: 'regression' as const, regIndex: i,
    }))
  }

  const cl = d.class_labels
  const groups: GroupSpec[] = []
  let off = 0
  if (Array.isArray(cl)) {
    if (cl.length > 0 || task !== 'mixed') {
      // size -1: legacy forests may ship empty labels — span the whole value vector
      groups.push({
        name: '', labels: cl, offset: 0, size: cl.length || -1, index: 0,
        kind: 'classification', regIndex: -1,
      })
      off = cl.length
    }
  } else {
    for (const [name, labels] of Object.entries(cl)) {
      groups.push({
        name, labels, offset: off, size: labels.length, index: groups.length,
        kind: 'classification', regIndex: -1,
      })
      off += labels.length
    }
  }

  if (task === 'mixed') {
    const names = d.target_labels ?? (d.target_label ? [d.target_label] : [])
    names.forEach((name, k) => {
      groups.push({
        name, labels: [], offset: off + k, size: 1, index: groups.length,
        kind: 'regression', regIndex: k,
      })
    })
  }
  return groups
}

export const labelGroupsAtom = atom((get): GroupSpec[] => {
  const state = get(loadableForest)
  return state.state === 'hasData' ? groupsFromForest(state.data) : []
})

// Raw user selection; clamped against the current group count in activeGroupAtom
// so a stale index survives iteration switches gracefully.
export const selectedGroupAtom = atom(0)

const FALLBACK_GROUP: GroupSpec = {
  name: '', labels: [], offset: 0, size: -1, index: 0,
  kind: 'classification', regIndex: -1,
}

export const activeGroupAtom = atom((get): GroupSpec => {
  const groups = get(labelGroupsAtom)
  if (groups.length === 0) return FALLBACK_GROUP
  return groups[Math.max(0, Math.min(get(selectedGroupAtom), groups.length - 1))]
})

// Task of the ACTIVE group. In a mixed forest this flips as the user switches
// groups; for pure forests it equals the forest task. Replaces the old
// forest-wide taskAtom — rendering always follows the active group.
export const activeTaskAtom = atom((get): TaskType => get(activeGroupAtom).kind)

// Class labels of the ACTIVE group — downstream consumers stay group-agnostic.
export const classLabelsAtom = atom((get): string[] => get(activeGroupAtom).labels)

// Embedded colors of the active group (flat list for legacy single-group forests).
export const embeddedClassColorsAtom = atom((get): string[] => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return []
  const cc = state.data.class_colors
  if (!cc) return []
  if (Array.isArray(cc)) return cc
  return cc[get(activeGroupAtom).name] ?? []
})

// Per-target range picker: multi-target forests ship one [min,max] per target.
export function pickRange(
  r: [number, number] | [number, number][] | null | undefined,
  index: number,
): [number, number] | null {
  if (!r || r.length === 0) return null
  return Array.isArray(r[0]) ? ((r as [number, number][])[index] ?? null) : (r as [number, number])
}

// Regression color domains of the ACTIVE target (null for classification).
// Indexed by regIndex: in a mixed forest the target arrays are aligned with
// target_labels, not with the global group ordinal.
export const targetRangeAtom = atom((get): [number, number] | null => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return null
  const g = get(activeGroupAtom)
  if (g.kind !== 'regression') return null
  return pickRange(state.data.target_range, g.regIndex)
})

export const stdRangeAtom = atom((get): [number, number] | null => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return null
  const g = get(activeGroupAtom)
  if (g.kind !== 'regression') return null
  return pickRange(state.data.std_range, g.regIndex)
})

// Name of the active regression target.
export const targetLabelAtom = atom((get): string => {
  const state = get(loadableForest)
  if (state.state !== 'hasData') return ''
  const g = get(activeGroupAtom)
  return g.name || (state.data.target_label ?? '')
})

// Forest-level aggregates (purity, custom-metric mean/σ) live in
// ForestMetrics/ForestMetricsAtoms.ts — this module stays the raw forest data.

// Tree node data: re-fetches whenever the selected iteration or tree ID changes
export const treeDataAtomFamily = atomFamily((treeId: string) =>
  atom(async (get): Promise<TreeNodeData[]> => {
    const iter = get(selectedIterationAtom)
    if (iter === null) return []
    const res = await fetch(`${apiBase()}/iteration/${iter}/tree/${treeId}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
  })
)

export type SimilarityMetric = 'leaf_cooccurrence' | 'path_overlap'

/** Expected shared nodes / path length, root-excluded, in raw node units. */
export type PathStats = { path_shared?: number[][]; path_len?: number[] }

export type SimilarityDiagnostics = {
  class_totals: number[]
  majority_class: number | null
  imbalance_warning: boolean
}

export type SimilarityData = {
  labels: string[]
  metrics: Record<SimilarityMetric, number[][]>
  per_tree: Record<string, Record<SimilarityMetric, number[][]> & PathStats>
  // Raw units behind path_overlap, for the "share X of Y nodes" readout.
  // Optional so a response from an older backend still renders.
  path_shared?: number[][]
  path_len?: number[]
  diagnostics?: SimilarityDiagnostics
  computed_ms?: number  // backend compute time, shown after the matrix arrives
}

const asyncSimilarityAtom = atom(async (get): Promise<SimilarityData | null> => {
  const iter = get(selectedIterationAtom)
  if (iter === null) return null
  // Both metrics are classification-only (they pair up labels via per-node class
  // histograms, which a regression node does not have), so skip the request
  // rather than round-trip for an empty payload.
  const g = get(activeGroupAtom)
  if (g.kind === 'regression') return null
  const group = g.index
  const res = await fetch(`${apiBase()}/iteration/${iter}/similarity?group=${group}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
})

export const loadableSimilarity = createLoadable(asyncSimilarityAtom)


export type ForestMapData = {
  tree_ids: string[]
  distances: number[][]
  // 'none' = no correlation matrix on this forest; the map is disabled.
  // There is no structural fallback — see _compute_tree_distances in treeflow.py.
  source: 'correlation' | 'none'
}

// Whole-tree: no group dependency. A tree serves every task at once, so the map
// must not change (or refetch) when the user switches label group.
const asyncForestMapAtom = atom(async (get): Promise<ForestMapData | null> => {
  const iter = get(selectedIterationAtom)
  if (iter === null) return null
  const res = await fetch(`${apiBase()}/iteration/${iter}/tree-map`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
})

export const loadableForestMap = createLoadable(asyncForestMapAtom)

const _loadableTreeCache = new Map<string, Atom<LoadableState<TreeNodeData[]>>>()
export function loadableTreeDataAtomFamily(treeId: string): Atom<LoadableState<TreeNodeData[]>> {
  let cached = _loadableTreeCache.get(treeId)
  if (!cached) {
    cached = createLoadable(treeDataAtomFamily(treeId))
    _loadableTreeCache.set(treeId, cached)
  }
  return cached
}
