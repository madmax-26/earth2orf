import type { TreeNodeData } from '../Main/components/Forest/ForestAtoms'
import type { ReferenceMeta } from './NodeViewAtoms'

// Pure helpers that turn reference paths into per-node views. Kept free of
// React/Jotai so they are trivially unit-testable.

/**
 * nodeId → its two children (or null for leaves), rebuilt from the DFS-preorder
 * node list with the same depth-stack walk as the layout's buildHierarchy: a
 * node's parent is the nearest preceding node of strictly smaller depth; the
 * first child encountered is the left one.
 */
export function computeChildrenMap(
  nodes: TreeNodeData[],
): Map<number, { left: number; right: number } | null> {
  const out = new Map<number, { left: number; right: number } | null>()
  const firstChild = new Map<number, number>()
  const stack: number[] = []   // indices into nodes
  for (let i = 0; i < nodes.length; i++) {
    const d = nodes[i].depth
    while (stack.length && nodes[stack[stack.length - 1]].depth >= d) stack.pop()
    if (stack.length) {
      const pid = nodes[stack[stack.length - 1]].id
      const left = firstChild.get(pid)
      if (left === undefined) firstChild.set(pid, nodes[i].id)
      else out.set(pid, { left, right: nodes[i].id })
    }
    out.set(nodes[i].id, null)   // provisional leaf; overwritten when 2nd child arrives
    stack.push(i)
  }
  // Nodes with exactly one recorded child (shouldn't occur in binary trees,
  // but stay safe): treat as leaf — out already holds null for them.
  return out
}

export type NodeRefPartition = {
  /** every reference whose path visits the node */
  arriving: ReferenceMeta[]
  /** continue into the left / right child */
  toLeft: ReferenceMeta[]
  toRight: ReferenceMeta[]
  /** path terminates here (leaf endings; also truncated paths at inner nodes) */
  ending: ReferenceMeta[]
}

/** Invariant: arriving = toLeft + toRight + ending (same objects, same order). */
export function partitionRefsAtNode(
  refs: ReferenceMeta[],
  treeId: string,
  nodeId: number,
  childrenMap: Map<number, { left: number; right: number } | null>,
): NodeRefPartition {
  const arriving: ReferenceMeta[] = []
  const toLeft: ReferenceMeta[] = []
  const toRight: ReferenceMeta[] = []
  const ending: ReferenceMeta[] = []
  const children = childrenMap.get(nodeId) ?? null

  for (const r of refs) {
    const path = r.paths[treeId]
    if (!path) continue
    const i = path.indexOf(nodeId)
    if (i < 0) continue
    arriving.push(r)
    const next = i + 1 < path.length ? path[i + 1] : undefined
    if (next === undefined) ending.push(r)
    else if (children && next === children.right) toRight.push(r)
    else toLeft.push(r)
  }
  return { arriving, toLeft, toRight, ending }
}

export type LabelSplitRow = {
  key: string
  labelIndex: number | null
  /** raw label from the backend; the view prefers classLabels[labelIndex] */
  label: string
  left: number
  right: number
  total: number
  /** Every reference of this label went the same way AND there were at least
      two of them — a single reference trivially lands on one side and is no
      evidence that the split isolated anything. */
  exclusive: 'left' | 'right' | null
}

/**
 * How each label's references divided at this split. This is the actual read on
 * "did the split separate a class": a label with all its references on one side
 * (`exclusive`) was isolated by the split test, one spread across both was not.
 * References whose path ended at the node (truncated) belong to neither side and
 * are not passed in.
 */
/** The identity a reference is grouped by: class index when present, else the
    raw label text, so unlabelled references still aggregate sensibly. */
export function refLabelKey(r: ReferenceMeta): string {
  return r.label_index !== null ? `i${r.label_index}` : `n${r.label ?? ''}`
}

/** What the split did with a label: separated it, failed to, or had too few
    references to tell. */
export type SplitOutcome = 'isolated' | 'mixed' | 'few'

export function outcomeOf(row: LabelSplitRow): SplitOutcome {
  return row.exclusive !== null ? 'isolated' : row.total < 2 ? 'few' : 'mixed'
}

export function labelSplit(
  toLeft: ReferenceMeta[],
  toRight: ReferenceMeta[],
): LabelSplitRow[] {
  const rows = new Map<string, LabelSplitRow>()
  const add = (r: ReferenceMeta, side: 'left' | 'right') => {
    const key = refLabelKey(r)
    let row = rows.get(key)
    if (!row) {
      row = {
        key, labelIndex: r.label_index, label: r.label ?? '',
        left: 0, right: 0, total: 0, exclusive: null,
      }
      rows.set(key, row)
    }
    row[side] += 1
    row.total += 1
  }
  for (const r of toLeft) add(r, 'left')
  for (const r of toRight) add(r, 'right')

  const out = [...rows.values()]
  for (const row of out) {
    row.exclusive = row.total < 2 ? null
      : row.left > 0 && row.right === 0 ? 'left'
      : row.right > 0 && row.left === 0 ? 'right'
      : null
  }
  // Grouped by outcome so the table reads as blocks: everything the split sent
  // left, then everything it sent right, then the labels it failed to separate,
  // then the ones with too few references to say. Isolated rows lead because
  // they answer "did this split separate a class?" and the view caps the list.
  // Within a group: most-represented first, with a stable tiebreak so the order
  // can't flicker between renders.
  const rank = (r: LabelSplitRow) =>
    r.exclusive === 'left' ? 0 :
    r.exclusive === 'right' ? 1 :
    r.total >= 2 ? 2 : 3
  return out.sort((a, b) =>
    rank(a) - rank(b) || b.total - a.total || a.key.localeCompare(b.key))
}

/** Per-node pass-through counts of one tree (for the hover-widget hint). */
export function refCountsForTree(refs: ReferenceMeta[], treeId: string): Map<number, number> {
  const counts = new Map<number, number>()
  for (const r of refs) {
    const path = r.paths[treeId]
    if (!path) continue
    for (const id of path) counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  return counts
}
