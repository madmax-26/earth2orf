import { sliceValue, type TreeNodeData, type GroupSpec } from '../Forest/ForestAtoms'

// Per-node role in a Decisive Node Analysis result:
//  'decisive' — a mixed split node that separates a pure-A region from a pure-B
//               region beneath it (the "decisive" node);
//  'A' / 'B'  — a node on a pure path descending from a decisive node toward a
//               pure endpoint of label A / label B.
export type DecisiveTag = 'A' | 'B' | 'decisive'
export type DecisiveMap = Map<number, DecisiveTag>

/**
 * Decisive Node Analysis (see plan). Purity is measured over the two chosen
 * labels only: pA = A/(A+B). A node is pure-A if pA ≥ t, pure-B if pB ≥ t, and
 * "mixed" if it holds both but neither dominates. Walking up from every pure
 * node to its first mixed ancestor identifies the split that separated it; a
 * split is decisive only if BOTH a pure-A and a pure-B region reach it. The
 * returned map tags every decisive split and the full pure paths beneath it.
 *
 * `nodes` must be the tree's full DFS-preorder node list (as served by the
 * backend). `group` selects the active label group's value slice.
 */
export function computeDecisiveNodes(
  nodes: TreeNodeData[],
  group: GroupSpec,
  labelA: number,
  labelB: number,
  threshold: number,
): DecisiveMap {
  const out: DecisiveMap = new Map()
  const n = nodes.length
  if (n === 0 || labelA === labelB) return out

  // ── Classify each node from its A-vs-B split (0 empty, 1 pureA, 2 pureB, 3 mixed) ──
  const EMPTY = 0, PURE_A = 1, PURE_B = 2, MIXED = 3
  const kind = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const v = sliceValue(nodes[i].value, group)
    const a = v[labelA] ?? 0
    const b = v[labelB] ?? 0
    const tot = a + b
    if (tot <= 0) { kind[i] = EMPTY; continue }
    if (a / tot >= threshold) kind[i] = PURE_A
    else if (b / tot >= threshold) kind[i] = PURE_B
    else kind[i] = MIXED
  }

  // ── Parent pointers from DFS preorder (a node's parent is the nearest
  //    preceding node of strictly smaller depth — depth-stack walk) ──
  const parent = new Int32Array(n).fill(-1)
  const stack: number[] = []
  for (let i = 0; i < n; i++) {
    const d = nodes[i].depth
    while (stack.length && nodes[stack[stack.length - 1]].depth >= d) stack.pop()
    parent[i] = stack.length ? stack[stack.length - 1] : -1
    stack.push(i)
  }

  // ── For each pure node, walk up to its first mixed ancestor and record the
  //    pure chain plus which labels reach each candidate split ──
  //
  //    `via` is the child of that split the region arrives through. It matters
  //    because purity is not closed downward: a 90%-pure-A node can still hold
  //    a pure-B pocket, and the walk passes straight through it (it isn't
  //    MIXED). Both regions can therefore reach the same split via the SAME
  //    child — in which case that split separated nothing and must not count.
  const firstMixed = new Int32Array(n).fill(-2)   // -2 = uncomputed, -1 = none
  const viaChild = new Int32Array(n).fill(-1)     // child of firstMixed on the path
  const walkToMixed = (start: number): number => {
    if (firstMixed[start] !== -2) return firstMixed[start]
    const chain: number[] = []
    let cur = parent[start]
    let found = -1
    let via = start                 // node entered from, i.e. the child of `found`
    while (cur !== -1) {
      if (firstMixed[cur] !== -2) { found = firstMixed[cur]; via = viaChild[cur]; break }
      if (kind[cur] === MIXED) { found = cur; break }
      chain.push(cur)
      via = cur
      cur = parent[cur]
    }
    // Every node on the chain reaches `found` through the same child.
    firstMixed[start] = found
    viaChild[start] = via
    for (const c of chain) { firstMixed[c] = found; viaChild[c] = via }
    return found
  }

  // Which children of split D does a pure-A / pure-B region arrive through?
  // pathOfD: the pure nodes on the way (each tagged by its OWN kind), grouped
  // by their decisive split D.
  const reachedA = new Map<number, Set<number>>()
  const reachedB = new Map<number, Set<number>>()
  const pathOfD = new Map<number, { node: number; tag: 'A' | 'B' }[]>()
  const record = (m: Map<number, Set<number>>, D: number, via: number) => {
    const s = m.get(D)
    if (s) s.add(via)
    else m.set(D, new Set([via]))
  }

  const addPath = (leaf: number, D: number) => {
    let arr = pathOfD.get(D)
    if (!arr) { arr = []; pathOfD.set(D, arr) }
    // Every node from `leaf` up to (excluding) D is an ancestor of a pure leaf,
    // hence non-empty and non-mixed → pure-A or pure-B; tag by its own kind.
    let cur = leaf
    while (cur !== -1 && cur !== D) {
      arr.push({ node: cur, tag: kind[cur] === PURE_B ? 'B' : 'A' })
      cur = parent[cur]
    }
  }

  for (let i = 0; i < n; i++) {
    if (kind[i] !== PURE_A && kind[i] !== PURE_B) continue
    const D = walkToMixed(i)
    if (D < 0) continue
    record(kind[i] === PURE_A ? reachedA : reachedB, D, viaChild[i])
    addPath(i, D)
  }

  // A split separates the two labels only if their regions come out of
  // DIFFERENT children — both arriving through the same child means the parting
  // happened deeper, and this split just passed them along together.
  const separates = (D: number): boolean => {
    const a = reachedA.get(D)
    const b = reachedB.get(D)
    if (!a || !b) return false
    for (const x of a) for (const y of b) if (x !== y) return true
    return false
  }

  // ── A split is decisive only if BOTH labels purify below it, on opposite
  //    sides. Emit it plus its full pure paths. Keys are node.id (NOT array
  //    index — River ids aren't sequential), matching how the Tree looks up
  //    node positions. ──
  for (const [D, path] of pathOfD) {
    if (!separates(D)) continue
    out.set(nodes[D].id, 'decisive')
    for (const { node, tag } of path) {
      const id = nodes[node].id
      if (!out.has(id)) out.set(id, tag)   // decisive (mixed) and path (pure) ids are disjoint
    }
  }

  return out
}
