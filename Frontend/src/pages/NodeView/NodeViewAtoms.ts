import { atom } from 'jotai'
import { createLoadable } from '../Main/components/Forest/ForestAtoms'
import { selectedIterationAtom } from '../Main/components/Iteration/IterationAtoms'
import { apiBase } from '../../apiConfig'

// One visual reference as served by /iteration/{i}/references: a sample the
// user registered during training, its displayable image, and its root→leaf
// node path per tree — re-routed at every push, so paths always match the
// iteration's snapshot.
export type ReferenceMeta = {
  id: string
  name: string | null
  label: string | null
  label_index: number | null
  step: number | null
  /** tree id (string) → node ids, root..terminal */
  paths: Record<string, number[]>
  /** backend-relative URL; prefix with apiBase() */
  image_url: string
}

export const refImageUrl = (r: ReferenceMeta): string => `${apiBase()}${r.image_url}`

// The node the user opened NodeView for. Global (not Tree-local state) because
// Tree instances mount in several places while the NodeView window mounts
// exactly once at App level.
export const selectedNodeAtom = atom<{ treeId: string; nodeId: number } | null>(null)

// References whose root→leaf path is highlighted in a tree render, per tree
// (treeId → reference ids). Set from NodeView ("show path in tree"), cleared
// from the tree header. Reference ids — not node ids — so the highlight follows
// the patch across iterations: each iteration re-routes every reference, and
// the tree re-reads that iteration's path for it.
export const highlightedRefsAtom = atom<Record<string, readonly string[]>>({})

// References of the selected iteration. Old/tabular iterations (or a backend
// without the feature) degrade to an empty list — the whole NodeView feature
// stays dormant then.
const asyncReferencesAtom = atom(async (get): Promise<ReferenceMeta[]> => {
  const iter = get(selectedIterationAtom)
  if (iter === null) return []
  try {
    const res = await fetch(`${apiBase()}/iteration/${iter}/references`)
    if (!res.ok) return []
    return await res.json()
  } catch {
    return []
  }
})

export const loadableReferencesAtom = createLoadable(asyncReferencesAtom)
