import { useMemo, memo } from 'react'
import { useAtomValue } from 'jotai'
import * as d3 from 'd3'
import {
  loadableTreeDataAtomFamily, activeTaskAtom, targetRangeAtom,
  activeGroupAtom, impurityAt, type TreeNodeData,
} from '../Forest/ForestAtoms'
import { meanColorsAtom } from '../Legend/LegendAtoms'
import { treeRanges } from '../Tree/regressionColor'

type HNode = TreeNodeData & { children: HNode[] }

function buildHierarchy(nodes: TreeNodeData[]): HNode | null {
  if (!nodes.length) return null
  const root: HNode = { ...nodes[0], children: [] }
  const stack: HNode[] = [root]
  for (let i = 1; i < nodes.length; i++) {
    const hNode: HNode = { ...nodes[i], children: [] }
    while (stack.length && stack[stack.length - 1].depth >= nodes[i].depth) stack.pop()
    if (stack.length) stack[stack.length - 1].children.push(hNode)
    stack.push(hNode)
  }
  return root
}

interface ThumbBranch {
  x1: number; y1: number; x2: number; y2: number
  fraction: number; color: string
}

// Detail cap: at thumbnail size (48–64px) branches deeper than this or thinner
// than this fraction of the root are sub-pixel — skip them so a 3k+ node tree
// costs a few hundred lines per thumbnail instead of thousands.
const THUMB_MAX_DEPTH = 8
const THUMB_MIN_FRACTION = 1 / 128

function computeThumbLayout(root: HNode, colorOf: (node: HNode) => string) {
  const INIT = 100
  const SHORTEN = 0.9
  const MIN_LEN = 4
  const rootSamples = root.n_node_samples

  const branches: ThumbBranch[] = []
  let bx0 = 0, by0 = 0, bx1 = 0, by1 = 0

  const stack: Array<{ node: HNode; x: number; y: number; angle: number; len: number; depth: number }> = [
    { node: root, x: 0, y: 0, angle: 0, len: INIT, depth: 0 },
  ]

  while (stack.length) {
    const { node, x, y, angle, len, depth } = stack.pop()!
    const ex = x + len * Math.sin(angle)
    const ey = y - len * Math.cos(angle)

    if (ex < bx0) bx0 = ex; if (ex > bx1) bx1 = ex
    if (ey < by0) by0 = ey; if (ey > by1) by1 = ey

    branches.push({
      x1: x, y1: y, x2: ex, y2: ey,
      fraction: node.n_node_samples / rootSamples,
      color: colorOf(node),
    })

    const left = node.children[0]
    const right = node.children[1]
    const descend = (child: HNode) =>
      depth < THUMB_MAX_DEPTH && child.n_node_samples / rootSamples >= THUMB_MIN_FRACTION

    if (!left) {
      // leaf — nothing to push
    } else if (!right) {
      if (descend(left)) {
        stack.push({ node: left, x: ex, y: ey, angle, len: Math.max(MIN_LEN, SHORTEN * len), depth: depth + 1 })
      }
    } else {
      const PL = left.n_node_samples / node.n_node_samples
      const PR = right.n_node_samples / node.n_node_samples
      if (descend(right)) {
        stack.push({
          node: right, x: ex, y: ey,
          angle: angle + (1 - PR),
          len: Math.max(MIN_LEN, Math.min(PR, SHORTEN) * len),
          depth: depth + 1,
        })
      }
      if (descend(left)) {
        stack.push({
          node: left, x: ex, y: ey,
          angle: angle - (1 - PL),
          len: Math.max(MIN_LEN, Math.min(PL, SHORTEN) * len),
          depth: depth + 1,
        })
      }
    }
  }

  return { branches, bounds: { x0: bx0, y0: by0, x1: bx1, y1: by1 } }
}

type Props = { treeId: string; size?: number }

const TreeThumbnail = memo(function TreeThumbnail({ treeId, size = 64 }: Props) {
  const atom = useMemo(() => loadableTreeDataAtomFamily(treeId), [treeId])
  const treeState = useAtomValue(atom)
  const task = useAtomValue(activeTaskAtom)
  const targetRange = useAtomValue(targetRangeAtom)
  const meanColors = useAtomValue(meanColorsAtom)
  const group = useAtomValue(activeGroupAtom)

  const layout = useMemo(() => {
    if (treeState.state !== 'hasData' || !treeState.data.length) return null
    const root = buildHierarchy(treeState.data)
    if (!root) return null

    // Classification: green→brown by Gini. Regression: mean-prediction gradient
    // over the forest range (a quick "prediction landscape" overview).
    // Both read the active label group / target in multi-output forests.
    let colorOf: (node: HNode) => string
    if (task === 'regression') {
      const interp = d3.interpolateRgb(meanColors.low, meanColors.high)
      const [lo, hi] = targetRange ?? treeRanges(treeState.data, group).mean
      const span = hi > lo ? hi - lo : 1
      colorOf = (node) => interp(Math.max(0, Math.min(1, ((node.value[group.offset] ?? lo) - lo) / span)))
    } else {
      const scale = d3.scaleSequential(d3.interpolateRgb('#3A8C3A', '#7B3F00')).domain([0, 1])
      colorOf = (node) => scale(impurityAt(node.impurity, group.index))
    }

    const { branches, bounds } = computeThumbLayout(root, colorOf)
    const PAD = 4
    const w = (bounds.x1 - bounds.x0) || 1
    const h = (bounds.y1 - bounds.y0) || 1
    const scale = Math.min((size - PAD * 2) / w, (size - PAD * 2) / h)
    const tx = size / 2 - ((bounds.x0 + bounds.x1) / 2) * scale
    const ty = size / 2 - ((bounds.y0 + bounds.y1) / 2) * scale
    return { branches, scale, tx, ty }
  }, [treeState, size, task, targetRange, meanColors, group])

  return (
    <svg width={size} height={size} style={{ flexShrink: 0, display: 'block' }}>
      {layout ? (
        <g transform={`translate(${layout.tx},${layout.ty}) scale(${layout.scale})`}>
          {layout.branches.map((b, i) => (
            <line
              key={i}
              x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2}
              stroke={b.color}
              strokeWidth={Math.max(0.4, b.fraction * 1.6) / layout.scale}
              strokeLinecap="round"
              style={{ pointerEvents: 'none' }}
            />
          ))}
        </g>
      ) : (
        <circle
          cx={size / 2} cy={size / 2} r={size * 0.3}
          fill="#8B6040" opacity={0.25}
        />
      )}
    </svg>
  )
})

export default TreeThumbnail
