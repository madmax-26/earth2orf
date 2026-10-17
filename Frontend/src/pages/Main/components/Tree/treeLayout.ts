import * as d3 from 'd3'
import { sliceValue, impurityAt, type TreeNodeData, type GroupSpec } from '../Forest/ForestAtoms'
import { classColor } from '../classColors'
import { makeRegressionLeafColor, type ColorRanges, type LeafColors } from './regressionColor'

const INITIAL_LENGTH = 140
const SHORTENING     = 0.9
const MIN_LENGTH     = 6
const MAX_WIDTH      = 14

export type HNode = TreeNodeData & { children: HNode[] }

export interface BranchDatum {
  x1: number; y1: number; x2: number; y2: number
  width: number; color: string
  nodeId: number; hasChildren: boolean; data: HNode
}

export interface LeafDatum {
  x: number; y: number; data: HNode; samples: number
  isCollapsed: boolean
}

export interface LayoutResult {
  branches: BranchDatum[]
  leaves: LeafDatum[]
  bounds: { x0: number; y0: number; x1: number; y1: number }
}

export function buildHierarchy(nodes: TreeNodeData[]): HNode | null {
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

export function computeBotanicalLayout(
  root: HNode,
  collapsed: ReadonlySet<number> = new Set(),
  branchNorm = 1,
  groupIndex = 0,
): LayoutResult {
  const branches: BranchDatum[] = []
  const leaves: LeafDatum[] = []
  const rootSamples = root.n_node_samples
  const branchColor = d3.scaleSequential(
    d3.interpolateRgb('#3A8C3A', '#7B3F00')
  ).domain([0, 1])
  const norm = branchNorm > 0 ? branchNorm : 1

  let x0 = 0, y0 = 0, x1 = 0, y1 = 0
  const stack: Array<{ node: HNode; x: number; y: number; angle: number; length: number }> = [
    { node: root, x: 0, y: 0, angle: 0, length: INITIAL_LENGTH },
  ]

  while (stack.length) {
    const { node, x, y, angle, length } = stack.pop()!
    const ex = x + length * Math.sin(angle)
    const ey = y - length * Math.cos(angle)
    if (ex < x0) x0 = ex; if (ex > x1) x1 = ex
    if (ey < y0) y0 = ey; if (ey > y1) y1 = ey

    const left  = node.children[0]
    const right = node.children[1]
    const hasChildren = left !== undefined
    branches.push({
      x1: x, y1: y, x2: ex, y2: ey,
      width: Math.max(0.4, (node.n_node_samples / rootSamples) * MAX_WIDTH),
      color: branchColor(impurityAt(node.impurity, groupIndex) / norm),
      nodeId: node.id, hasChildren, data: node,
    })

    const isCollapsed = hasChildren && collapsed.has(node.id)
    if (!left || isCollapsed) {
      leaves.push({ x: ex, y: ey, data: node, samples: node.n_node_samples, isCollapsed })
    } else if (!right) {
      stack.push({ node: left, x: ex, y: ey, angle, length: Math.max(MIN_LENGTH, SHORTENING * length) })
    } else {
      const PL = left.n_node_samples  / node.n_node_samples
      const PR = right.n_node_samples / node.n_node_samples
      stack.push({ node: right, x: ex, y: ey, angle: angle + (1 - PR), length: Math.max(MIN_LENGTH, Math.min(PR, SHORTENING) * length) })
      stack.push({ node: left,  x: ex, y: ey, angle: angle - (1 - PL), length: Math.max(MIN_LENGTH, Math.min(PL, SHORTENING) * length) })
    }
  }
  return { branches, leaves, bounds: { x0, y0, x1, y1 } }
}

function purityFill(impurity: number): string {
  const t = Math.max(0, Math.min(1, impurity))
  const r = Math.round(58 + (198 - 58) * t)
  const g = Math.round(140 + (40  - 140) * t)
  const b = Math.round(58 + (40  - 58) * t)
  return `rgb(${r},${g},${b})`
}

export type LeafMode = 'pie' | 'purity'

// Regression render options for the SVG export (mirrors the live render).
export type RegressionRenderOpts = { ranges: ColorRanges; colors: LeafColors }

export function treeToSVGString(
  nodes: TreeNodeData[],
  customColors: Record<number, string> = {},
  collapsed: ReadonlySet<number> = new Set(),
  leafMode: LeafMode = 'pie',
  reg?: RegressionRenderOpts,
  group?: GroupSpec,
  /** The tree view's leaf-size slider, so the file matches what's on screen. */
  leafScale = 1,
): string {
  const root = buildHierarchy(nodes)
  if (!root) return ''
  const gIdx = group?.index ?? 0
  const gOff = group?.offset ?? 0
  const slice = (v: number[]) => (group ? sliceValue(v, group) : v)
  const branchNorm = reg ? Math.max(1e-9, ...nodes.map(n => impurityAt(n.impurity, gIdx))) : 1
  const { branches, leaves, bounds } = computeBotanicalLayout(root, collapsed, branchNorm, gIdx)
  const regColor = reg ? makeRegressionLeafColor(leafMode, reg.ranges, reg.colors) : null

  const PAD = 30
  const vx = bounds.x0 - PAD
  const vy = bounds.y0 - PAD
  const vw = (bounds.x1 - bounds.x0) + PAD * 2
  const vh = (bounds.y1 - bounds.y0) + PAD * 2

  // Leaf area ∝ samples (see Tree.tsx): radius = Rmax·√(s/max) with a small
  // visibility floor — no additive offset, so sample ratios stay readable.
  const maxLeafSamples = Math.max(1, ...leaves.map(l => l.samples))
  const leafR = (s: number) => Math.max(1.5, 14 * Math.sqrt(s / maxLeafSamples)) * leafScale
  const pieGen = d3.pie<number>().sort(null)

  const branchLines = branches.map(b =>
    `<line x1="${b.x1.toFixed(2)}" y1="${b.y1.toFixed(2)}" x2="${b.x2.toFixed(2)}" y2="${b.y2.toFixed(2)}" ` +
    `stroke="${b.color}" stroke-width="${b.width.toFixed(2)}" stroke-linecap="round"/>`
  ).join('\n  ')

  const leafGroups = leaves.map(({ x, y, data, samples, isCollapsed }) => {
    const r  = leafR(samples)
    const rStr = r.toFixed(2)
    let fill: string
    if (regColor) {
      fill = `<circle r="${rStr}" fill="${regColor(data.value[gOff] ?? 0, impurityAt(data.impurity, gIdx))}"/>`
    } else if (leafMode === 'purity') {
      fill = `<circle r="${rStr}" fill="${purityFill(impurityAt(data.impurity, gIdx))}"/>`
    } else {
      const value = slice(data.value)
      const arcGen = d3.arc<d3.PieArcDatum<number>>().innerRadius(0).outerRadius(r)
      const total = value.reduce((a, v) => a + v, 0)
      const n = value.length
      fill = total === 0
        ? `<circle r="${rStr}" fill="#ddd"/>`
        : pieGen(value).map(s =>
            `<path d="${arcGen(s) ?? ''}" fill="${customColors[s.index] ?? classColor(s.index, n)}"/>`
          ).join('') +
          `<circle r="${rStr}" fill="none" stroke="#333" stroke-width="0.1"/>`
    }
    const ring = isCollapsed
      ? `<circle r="${(r + 1).toFixed(2)}" fill="none" stroke="#555" stroke-width="0.5" stroke-dasharray="1 1.5" opacity="0.8"/>`
      : ''
    return `<g transform="translate(${x.toFixed(2)},${y.toFixed(2)})">${fill}${ring}</g>`
  }).join('\n  ')

  return [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vx.toFixed(2)} ${vy.toFixed(2)} ${vw.toFixed(2)} ${vh.toFixed(2)}">`,
    `  <rect x="${vx.toFixed(2)}" y="${vy.toFixed(2)}" width="${vw.toFixed(2)}" height="${vh.toFixed(2)}" fill="white"/>`,
    `  ${branchLines}`,
    `  ${leafGroups}`,
    `</svg>`,
  ].join('\n')
}
