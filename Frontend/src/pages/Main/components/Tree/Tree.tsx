import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { useAtom, useAtomValue, useSetAtom } from 'jotai'
import {
  classLabelsAtom, loadableTreeDataAtomFamily,
  activeTaskAtom, targetRangeAtom, stdRangeAtom,
  activeGroupAtom, sliceValue, impurityAt,
  type TreeMeta, type TreeNodeData,
} from '../Forest/ForestAtoms'
import {
  activeGroupMetricStatsAtom, treeMetricStatsAtom, fmtMetric,
} from '../ForestMetrics/ForestMetricsAtoms'
import TreeNode from '../TreeNode/TreeNode'
import NodeWidget from '../NodeWidget/NodeWidget'
import {
  selectedClassAtom, meanColorsAtom, stdColorsAtom, effectiveClassColorsAtom,
} from '../Legend/LegendAtoms'
import { treeToSVGString } from './treeLayout'
import {
  leafModeAtom, colorRangeModeAtom, leafScaleAtom, LEAF_SCALE_MIN, LEAF_SCALE_MAX,
} from '../Customizer/CustomizerAtoms'
import { makeRegressionLeafColor, treeRanges, type ColorRanges } from './regressionColor'
import { decisiveAtom } from '../Analysis/DecisiveAtoms'
import { computeDecisiveNodes } from './decisiveNodes'
import {
  loadableReferencesAtom, selectedNodeAtom, highlightedRefsAtom,
} from '../../../NodeView/NodeViewAtoms'
import { refCountsForTree } from '../../../NodeView/referencePartition'
import { HEADING_FONT } from '../../../../theme'

export type TreeProps = {
  treeId: string
  meta: TreeMeta
}

const INITIAL_LENGTH = 140
const SHORTENING = 0.9
const MIN_LENGTH = 6
const MAX_WIDTH = 14

// Auto-collapse frontier (adjustable): subtrees deeper than this many levels
// below the root — or below the nearest user-expanded node — render as a single
// collapsed node (dashed ring). Expanding a frontier node grants its subtree a
// fresh budget of this many levels, so arbitrarily deep trees stay explorable
// while the visible element count remains bounded.
const AUTO_COLLAPSE_DEPTH = 12

// Stable empty set — the "no nodes forced open" case must keep a constant
// identity so the layout memo doesn't recompute on every render.
const NO_IDS: ReadonlySet<number> = new Set()
const NO_REF_IDS: readonly string[] = []

// Culling thresholds: below these counts the tree is small enough that culling
// overhead isn't worth it and everything renders.
const CULL_MIN_BRANCHES = 300
const OCCLUDE_MIN_LEAVES = 400

// Leaf radii are proportional to √samples so leaf AREA ∝ samples — a leaf with
// 10× the samples covers 10× the area. MAX is the radius of the largest leaf;
// MIN is only a visibility/hover floor for tiny leaves (no offset — an additive
// floor would compress the ratios and make sizes read nearly uniform).
const MIN_NODE_PX = 3
const MAX_NODE_PX = 24

// THE leaf radius (before zoom ink scaling). Used by the render AND the
// occlusion culling, so what is culled always matches what is drawn. `scale`
// is the user's leaf-size slider: a plain multiplier on the result, which keeps
// every size ratio (area ∝ samples) intact.
function leafRadius(samples: number, maxLeafSamples: number, scale: number): number {
  return Math.max(MIN_NODE_PX, MAX_NODE_PX * Math.sqrt(samples / maxLeafSamples)) * scale
}

// Exponent for zoom-damped "ink": at relative zoom z = k/fit, geometry magnifies
// by z while ink (leaf radii + branch widths) grows only z^ZOOM_DAMP. 1 = plain
// SVG scaling, 0 = constant screen size. Both leaves and branches share the one
// factor, so leaf-size ratios and the node/branch proportion hold at every zoom.
const ZOOM_DAMP = 0.5

// Minimum hit-target width for branches in screen pixels.
// Divided by transform.k when used so the hit area stays constant regardless of zoom.
const CLICK_AREA_PX = 12

type HNode = TreeNodeData & { children: HNode[] }

function buildHierarchy(nodes: TreeNodeData[]): HNode | null {
  if (!nodes.length) return null
  const root: HNode = { ...nodes[0], children: [] }
  const stack: HNode[] = [root]
  for (let i = 1; i < nodes.length; i++) {
    const hNode: HNode = { ...nodes[i], children: [] }
    while (stack.length && stack[stack.length - 1].depth >= nodes[i].depth) {
      stack.pop()
    }
    if (stack.length) stack[stack.length - 1].children.push(hNode)
    stack.push(hNode)
  }
  return root
}

interface BranchDatum {
  x1: number; y1: number; x2: number; y2: number
  width: number; color: string
  nodeId: number       // node placed at (x2, y2)
  hasChildren: boolean // clicking this branch will collapse/expand
  collapsed: boolean   // effective state (manual override or auto frontier)
  data: HNode          // full node data for hover widget
}

interface LeafDatum {
  x: number; y: number; data: HNode; samples: number
  isCollapsed: boolean // true when this node was stopped early (manual or auto frontier)
}

interface LayoutResult {
  branches: BranchDatum[]
  leaves: LeafDatum[]
  bounds: { x0: number; y0: number; x1: number; y1: number }
  maxLeafSamples: number
}

// Paper Section 4.1: botanical layout via angle + length derived from sample fractions.
// α_L/R = α_parent ± |P_L/R − 1|   (Eq. 4)
// l_L/R = max(min(P_L/R, s) * l, c) (Eq. 5)
// Collapse model: a node renders as a single summary node when the user
// collapsed it manually (`collapsed`), or when it sits on the auto frontier —
// `sinceExpand` levels below the root / nearest user-expanded node reaching
// AUTO_COLLAPSE_DEPTH — and wasn't explicitly expanded (`expanded`) and isn't
// held open by an active analysis (`keepOpen`, e.g. the DCA paths). The walk
// never descends past a collapsed node, so the visible element count stays
// bounded no matter how deep the tree is.
// branchNorm divides each node's impurity before the green→brown scale so the
// gradient spans [0,1]. Classification passes 1 (Gini is already [0,1]);
// regression passes the tree's max std (impurity is an unbounded std dev).
// groupIndex selects the entry of per-group impurity arrays (multi-label forests).
function computeBotanicalLayout(
  root: HNode,
  collapsed: ReadonlySet<number>,
  expanded: ReadonlySet<number>,
  keepOpen: ReadonlySet<number> = NO_IDS,
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
  let maxLeafSamples = 1

  interface Frame {
    node: HNode; x: number; y: number; angle: number; length: number
    sinceExpand: number   // depth since root / last user-expanded ancestor
  }

  const stack: Frame[] = [{ node: root, x: 0, y: 0, angle: 0, length: INITIAL_LENGTH, sinceExpand: 0 }]

  while (stack.length) {
    const { node, x, y, angle, length, sinceExpand } = stack.pop()!
    const ex = x + length * Math.sin(angle)
    const ey = y - length * Math.cos(angle)

    if (ex < x0) x0 = ex; if (ex > x1) x1 = ex
    if (ey < y0) y0 = ey; if (ey > y1) y1 = ey

    const left = node.children[0]
    const right = node.children[1]
    const hasChildren = left !== undefined

    // keepOpen (analysis-driven, e.g. DCA) suspends the AUTO frontier only —
    // a manual collapse is an explicit user choice and still wins.
    const isCollapsed = hasChildren && (
      collapsed.has(node.id) ||
      (sinceExpand >= AUTO_COLLAPSE_DEPTH && !expanded.has(node.id) && !keepOpen.has(node.id))
    )

    branches.push({
      x1: x, y1: y, x2: ex, y2: ey,
      width: Math.max(0.4, (node.n_node_samples / rootSamples) * MAX_WIDTH),
      color: branchColor(impurityAt(node.impurity, groupIndex) / norm),
      nodeId: node.id,
      hasChildren,
      collapsed: isCollapsed,
      data: node,
    })

    if (!left || isCollapsed) {
      if (node.n_node_samples > maxLeafSamples) maxLeafSamples = node.n_node_samples
      leaves.push({ x: ex, y: ey, data: node, samples: node.n_node_samples, isCollapsed })
    } else {
      // A user-expanded node grants its subtree a fresh depth budget
      const childSince = expanded.has(node.id) ? 1 : sinceExpand + 1
      if (!right) {
        stack.push({
          node: left, x: ex, y: ey, angle,
          length: Math.max(MIN_LENGTH, SHORTENING * length),
          sinceExpand: childSince,
        })
      } else {
        const PL = left.n_node_samples / node.n_node_samples
        const PR = right.n_node_samples / node.n_node_samples
        stack.push({
          node: right, x: ex, y: ey,
          angle: angle + (1 - PR),
          length: Math.max(MIN_LENGTH, Math.min(PR, SHORTENING) * length),
          sinceExpand: childSince,
        })
        stack.push({
          node: left, x: ex, y: ey,
          angle: angle - (1 - PL),
          length: Math.max(MIN_LENGTH, Math.min(PL, SHORTENING) * length),
          sinceExpand: childSince,
        })
      }
    }
  }

  return { branches, leaves, bounds: { x0, y0, x1, y1 }, maxLeafSamples }
}

function dominantClassOf(value: number[]): number {
  return value.reduce((best, v, i, arr) => v > arr[best] ? i : best, 0)
}

export default function Tree({ treeId, meta }: TreeProps) {
  const treeState = useAtomValue(
    useMemo(() => loadableTreeDataAtomFamily(treeId), [treeId])
  )

  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  // Imperative zoom plumbing: the content <g> gets its transform set directly
  // by the zoom handler; React state follows at ≤1 update per animation frame.
  const gRef = useRef<SVGGElement>(null)
  const pendingTransform = useRef<d3.ZoomTransform | null>(null)
  const rafId = useRef(0)
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity)
  // Culling reference transform — updated only when a gesture ENDS, so the
  // visible set stays frozen while dragging/zooming (no mid-gesture popping)
  // and is recomputed exactly once at rest. Outside gestures it follows the
  // live transform (see the rAF commit), so it can never be left stale.
  const [cullTransform, setCullTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity)
  const interactingRef = useRef(false)
  const [svgSize, setSvgSize] = useState({ w: 600, h: 500 })
  const fitDone = useRef(false)
  const [ctrlHeld, setCtrlHeld] = useState(false)
  const selectedClass = useAtomValue(selectedClassAtom)
  const classLabels = useAtomValue(classLabelsAtom)
  const leafMode = useAtomValue(leafModeAtom)
  const [leafScale, setLeafScale] = useAtom(leafScaleAtom)
  const task = useAtomValue(activeTaskAtom)
  const targetRange = useAtomValue(targetRangeAtom)
  const stdRange = useAtomValue(stdRangeAtom)
  const colorRangeMode = useAtomValue(colorRangeModeAtom)
  const meanColors = useAtomValue(meanColorsAtom)
  const stdColors = useAtomValue(stdColorsAtom)
  // Class colors as rendered on screen (forest-supplied class_colors merged with
  // the Legend's overrides) — the SVG export must use these, not the defaults.
  const effectiveColors = useAtomValue(effectiveClassColorsAtom)
  const group = useAtomValue(activeGroupAtom)
  const groupMetricStats = useAtomValue(activeGroupMetricStatsAtom)
  const treeMetricStats = useAtomValue(treeMetricStatsAtom)
  const decisive = useAtomValue(decisiveAtom)
  // Visual references (NodeView): gestures/hints only activate when the
  // current iteration actually has references — tabular / pre-feature forests
  // keep the exact previous interactions (leaf click no-op, Alt ignored).
  const refsState = useAtomValue(loadableReferencesAtom)
  const setSelectedNode = useSetAtom(selectedNodeAtom)
  const refs = useMemo(
    () => (refsState.state === 'hasData' ? refsState.data : []),
    [refsState],
  )
  const hasRefs = refs.length > 0
  const refCounts = useMemo(
    () => (hasRefs ? refCountsForTree(refs, treeId) : null),
    [hasRefs, refs, treeId],
  )
  const openNode = useCallback(
    (id: number) => setSelectedNode({ treeId, nodeId: id }),
    [treeId, setSelectedNode],
  )
  // Custom metrics scoped to the active label group / regression target
  const groupMetrics = meta.group_metrics?.[group.name] ?? {}

  // Co-locate treeId with hover so the leaf clears automatically on tree change —
  // no setState-in-effect needed.
  const [hoverState, setHoverState] = useState<{
    treeId: string
    target: { data: HNode; x: number; y: number } | null
  }>({ treeId, target: null })
  const hoverTarget = hoverState.treeId === treeId ? hoverState.target : null

  // Pinned node: clicking a node keeps its widget open (and makes it
  // interactive) until the next click that isn't on a node or the widget.
  // Only the id is stored — the position is re-derived from the live layout
  // each render, so the widget follows collapse/expand, zoom and pan.
  const [pinnedState, setPinnedState] = useState<{ treeId: string; nodeId: number | null }>(
    { treeId, nodeId: null }
  )
  const pinnedId = pinnedState.treeId === treeId ? pinnedState.nodeId : null
  const pinNode = useCallback((id: number, e: { stopPropagation: () => void }) => {
    // Keep the click from reaching the window listener that clears the pin.
    e.stopPropagation()
    setPinnedState({ treeId, nodeId: id })
  }, [treeId])
  const unpin = useCallback(() => setPinnedState({ treeId, nodeId: null }), [treeId])

  // Any click that isn't on a node or inside the widget (both stopPropagation)
  // dismisses the pin; Escape does too.
  useEffect(() => {
    if (pinnedId === null) return
    const onClick = () => unpin()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') unpin() }
    window.addEventListener('click', onClick)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('click', onClick)
      window.removeEventListener('keydown', onKey)
    }
  }, [pinnedId, unpin])

  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.key === 'Control') setCtrlHeld(true) }
    const up = () => setCtrlHeld(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', up)
    }
  }, [])

  // Collapse overrides, co-located with treeId so they auto-reset on tree change.
  // `collapsed` = manually collapsed nodes; `expanded` = auto-frontier nodes the
  // user drilled into (each grants its subtree a fresh AUTO_COLLAPSE_DEPTH budget).
  const [collapsedState, setCollapsedState] = useState<{
    treeId: string
    collapsed: ReadonlySet<number>
    expanded: ReadonlySet<number>
  }>({ treeId, collapsed: new Set(), expanded: new Set() })

  const { collapsed, expanded } = useMemo(() => (
    collapsedState.treeId === treeId
      ? { collapsed: collapsedState.collapsed, expanded: collapsedState.expanded }
      : { collapsed: new Set<number>(), expanded: new Set<number>() }
  ), [collapsedState, treeId])

  // isCollapsed = the node's CURRENT effective state (from the layout walk).
  // Expanding lifts a manual collapse first, else marks a frontier expansion;
  // collapsing removes a frontier expansion first, else collapses manually.
  const toggleCollapsed = useCallback((nodeId: number, isCollapsed: boolean) => {
    setCollapsedState(prev => {
      const same = prev.treeId === treeId
      const col = new Set(same ? prev.collapsed : [])
      const exp = new Set(same ? prev.expanded : [])
      if (isCollapsed) {
        if (col.has(nodeId)) col.delete(nodeId)
        else exp.add(nodeId)
      } else {
        if (exp.has(nodeId)) exp.delete(nodeId)
        else col.add(nodeId)
      }
      return { treeId, collapsed: col, expanded: exp }
    })
  }, [treeId])

  // Track container dimensions
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) {
        setSvgSize({ w: Math.floor(width), h: Math.floor(height) })
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Only a ref mutation — hoverLeaf auto-clears via co-location, no setState needed.
  useEffect(() => {
    fitDone.current = false
  }, [treeId])

  // Attach D3 zoom — initial scaleExtent is a wide placeholder;
  // the bounds-aware effect below narrows it once layout is known.
  //
  // Performance: the geometry transform is applied IMPERATIVELY to the content
  // <g> on every tick (zero React work — pan/zoom stays smooth at any element
  // count). React state, which drives derived values only (ink scale, hit
  // widths, culling, hover-overlay position), commits at most once per
  // animation frame. React never renders the <g> transform attribute, so the
  // two writers can't fight.
  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.0001, 40])
      .on('start', () => { interactingRef.current = true })
      .on('zoom', (e: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        gRef.current?.setAttribute('transform', e.transform.toString())
        pendingTransform.current = e.transform
        if (!rafId.current) {
          rafId.current = requestAnimationFrame(() => {
            rafId.current = 0
            if (pendingTransform.current) {
              setTransform(pendingTransform.current)
              // Outside an active gesture (programmatic fits, odd event paths)
              // culling must follow immediately — it can never be left stale.
              if (!interactingRef.current) setCullTransform(pendingTransform.current)
            }
          })
        }
      })
      // Culling only re-evaluates when the gesture settles — while dragging or
      // wheel-zooming the visible set stays frozen so nothing pops mid-view.
      .on('end', (e: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        interactingRef.current = false
        setCullTransform(e.transform)
      })
    zoomRef.current = zoom
    // Double-click is the collapse/expand gesture — drop d3's default
    // double-click-to-zoom. It listens on the <svg> itself, so it would fire
    // BEFORE React's handler on the node and can't be stopPropagation'd away.
    d3.select(el).call(zoom).on('dblclick.zoom', null)
    return () => {
      if (rafId.current) cancelAnimationFrame(rafId.current)
      rafId.current = 0
      d3.select(el).on('.zoom', null)
    }
  }, [])

  // Decisive Node Analysis highlight map (id → 'A'|'B'|'decisive'), computed on
  // the FULL tree data. Active only for a valid distinct label pair on a
  // classification group (set from the Analysis tab's DCA section).
  // Declared before the layout: it also decides where the layout may auto-collapse.
  const decisiveMap = useMemo(() => {
    if (task !== 'classification' || treeState.state !== 'hasData') return null
    const { labelA, labelB, threshold } = decisive
    if (labelA === null || labelB === null || labelA === labelB) return null
    const m = computeDecisiveNodes(treeState.data, group, labelA, labelB, threshold)
    return m.size ? m : null
  }, [task, treeState, group, decisive])

  // Every DCA-highlighted node plus all of its ancestors. The auto-collapse
  // frontier must not swallow a decisive split: collapsed, it renders as ONE
  // summary node whose pie is the aggregate of the whole subtree — you'd see a
  // single mixed pie instead of the decisive split and the pure branches under
  // it. Keeping just these ids open reveals the analysed structure while every
  // off-path subtree still collapses at the frontier as usual.
  const dcaKeepOpen = useMemo((): ReadonlySet<number> => {
    if (decisiveMap === null || treeState.state !== 'hasData') return NO_IDS
    const nodes = treeState.data
    const parent = new Map<number, number>()
    const stack: TreeNodeData[] = []
    for (const n of nodes) {
      while (stack.length && stack[stack.length - 1].depth >= n.depth) stack.pop()
      if (stack.length) parent.set(n.id, stack[stack.length - 1].id)
      stack.push(n)
    }
    const keep = new Set<number>()
    for (const id of decisiveMap.keys()) {
      let cur: number | undefined = id
      while (cur !== undefined && !keep.has(cur)) {
        keep.add(cur)
        cur = parent.get(cur)
      }
    }
    return keep
  }, [decisiveMap, treeState])

  // Reference paths highlighted from NodeView ("show path in tree"). Stored as
  // reference ids and resolved against THIS iteration's routing, so the
  // highlight follows a patch as it migrates between iterations. A reference
  // that doesn't exist in the current iteration simply contributes nothing.
  const highlightedRefs = useAtomValue(highlightedRefsAtom)
  const setHighlightedRefs = useSetAtom(highlightedRefsAtom)
  const myHighlightedRefIds = highlightedRefs[treeId] ?? NO_REF_IDS
  const pathNodeIds = useMemo((): ReadonlySet<number> | null => {
    if (myHighlightedRefIds.length === 0) return null
    const wanted = new Set(myHighlightedRefIds)
    const ids = new Set<number>()
    for (const r of refs) {
      if (!wanted.has(r.id)) continue
      for (const id of r.paths[treeId] ?? []) ids.add(id)
    }
    return ids.size ? ids : null
  }, [myHighlightedRefIds, refs, treeId])

  // Analysis-driven nodes the auto-collapse frontier must not swallow: the DCA
  // structure plus every highlighted path (a path is its own ancestor chain).
  const keepOpen = useMemo((): ReadonlySet<number> => {
    if (pathNodeIds === null) return dcaKeepOpen
    if (dcaKeepOpen.size === 0) return pathNodeIds
    return new Set([...dcaKeepOpen, ...pathNodeIds])
  }, [dcaKeepOpen, pathNodeIds])

  const layout = useMemo((): LayoutResult | null => {
    if (treeState.state !== 'hasData') return null
    const root = buildHierarchy(treeState.data)
    if (!root) return null
    let branchNorm = 1
    if (task === 'regression') {
      branchNorm = 1e-9
      for (const n of treeState.data) {
        const s = impurityAt(n.impurity, group.index)
        if (s > branchNorm) branchNorm = s
      }
    }
    return computeBotanicalLayout(root, collapsed, expanded, keepOpen, branchNorm, group.index)
  }, [treeState, collapsed, expanded, keepOpen, task, group])

  // Active regression color ranges (per-tree vs forest-wide) + per-leaf color fn.
  const activeRanges: ColorRanges | null = useMemo(() => {
    if (task !== 'regression' || treeState.state !== 'hasData') return null
    const tr = treeRanges(treeState.data, group)
    if (colorRangeMode === 'tree') return tr
    return { mean: targetRange ?? tr.mean, std: stdRange ?? tr.std }
  }, [task, treeState, colorRangeMode, targetRange, stdRange, group])

  const leafColorFn = useMemo(() => {
    if (!activeRanges) return null
    return makeRegressionLeafColor(leafMode, activeRanges, { mean: meanColors, std: stdColors })
  }, [activeRanges, leafMode, meanColors, stdColors])

  // Rendered leaf/collapsed node ids — the nodes the legend filter visually
  // fades (internal branches keep their color, hence stay interactive).
  const layoutLeafIds = useMemo(
    () => (layout ? new Set(layout.leaves.map(l => l.data.id)) : null),
    [layout],
  )

  // Is this node picked out by an active selection (legend dominant class /
  // Decisive Node Analysis)? Null when nothing is selected, which lets the
  // culling below skip the check entirely in the common case.
  const isHighlighted = useMemo(() => {
    const legendOn = task === 'classification' && selectedClass !== null
    if (!legendOn && decisiveMap === null && pathNodeIds === null) return null
    return (d: TreeNodeData): boolean => {
      if (decisiveMap !== null && decisiveMap.has(d.id)) return true
      if (pathNodeIds !== null && pathNodeIds.has(d.id)) return true
      if (legendOn && dominantClassOf(sliceValue(d.value, group)) === selectedClass) return true
      return false
    }
  }, [task, selectedClass, decisiveMap, pathNodeIds, group])

  // If a highlight (legend class / DCA) activates while the cursor already sits
  // on a node it fades, that node's hover disc unmounts without firing
  // onMouseLeave — derive the hover away (no setState-in-effect) so the
  // NodeWidget can't stick to a node that is no longer interactive. Stale
  // hoverState self-clears on the next mouse move.
  const hoverTargetFaded = hoverTarget !== null && (
    (decisiveMap !== null && !decisiveMap.has(hoverTarget.data.id)) ||
    (pathNodeIds !== null && !pathNodeIds.has(hoverTarget.data.id)) ||
    (task === 'classification' && selectedClass !== null &&
      layoutLeafIds?.has(hoverTarget.data.id) === true &&
      dominantClassOf(sliceValue(hoverTarget.data.value, group)) !== selectedClass)
  )
  const visibleHoverTarget = hoverTargetFaded ? null : hoverTarget

  // Live position of every rendered node, keyed by id — lets the pinned widget
  // track its node through collapse/expand instead of freezing at click-time
  // coordinates. (Leaves overwrite the branch entry; both carry the same point.)
  const nodePositions = useMemo(() => {
    if (!layout) return null
    const m = new Map<number, { data: HNode; x: number; y: number }>()
    for (const b of layout.branches) m.set(b.nodeId, { data: b.data, x: b.x2, y: b.y2 })
    for (const l of layout.leaves) m.set(l.data.id, { data: l.data, x: l.x, y: l.y })
    return m
  }, [layout])

  // The two sides of each decisive split — the children of a decisive node,
  // i.e. the roots of its pure-A / pure-B branches.
  //
  // Purity is NOT closed downward (a 90%-A node can hold a mixed child), so a
  // pure-branch root is frequently an INTERNAL node, and internal nodes draw
  // only a branch line. Without this, a decisive split can show no distribution
  // at all — the two pies that answer "what did this split separate?" are
  // exactly what's missing. Roots already rendered as leaves are skipped: they
  // draw their own pie.
  const dcaSummaries = useMemo(() => {
    if (decisiveMap === null || !layout || !nodePositions) return null
    const renderedLeaves = new Set(layout.leaves.map(l => l.data.id))
    const out: { id: number; data: HNode; x: number; y: number }[] = []
    for (const [id, tag] of decisiveMap) {
      if (tag !== 'decisive') continue
      const entry = nodePositions.get(id)
      if (!entry) continue
      for (const child of entry.data.children) {
        const childTag = decisiveMap.get(child.id)
        if (childTag !== 'A' && childTag !== 'B') continue
        if (renderedLeaves.has(child.id)) continue
        const pos = nodePositions.get(child.id)
        if (pos) out.push({ id: child.id, data: child, x: pos.x, y: pos.y })
      }
    }
    return out.length ? out : null
  }, [decisiveMap, layout, nodePositions])

  // A pinned node wins over hover, and stays visible even under a highlight
  // filter — pinning is explicit, so it shouldn't be faded away.
  const pinnedTarget = pinnedId !== null ? nodePositions?.get(pinnedId) ?? null : null
  const widgetTarget = pinnedTarget ?? visibleHoverTarget
  const widgetPinned = pinnedTarget !== null

  // Zoom at which the whole layout fits the viewport — the reference for
  // resetView, the scaleExtent, and the damped ink scaling in the render.
  const fitScale = useMemo(() => {
    if (!layout) return 1
    const pad = 24
    const w = layout.bounds.x1 - layout.bounds.x0 || 1
    const h = layout.bounds.y1 - layout.bounds.y0 || 1
    return Math.min((svgSize.w - pad * 2) / w, (svgSize.h - pad * 2) / h) * 0.9
  }, [layout, svgSize])

  // ── Culling — evaluated only when a gesture SETTLES (cullTransform updates on
  // d3 'end' / at rest), so the visible set is frozen while dragging/zooming and
  // nothing pops mid-gesture. Pure O(n) arithmetic, active only above thresholds. ──
  // 1. Viewport rejection: skip branches/leaves outside the visible rect
  //    (generous padding so moderate drags don't reveal blank edges before the
  //    end-of-gesture recompute fills them in).
  // 2. Occlusion cover test (z-buffer semantics): a leaf is skipped ONLY when it
  //    is FULLY covered on screen by an already-kept bigger leaf
  //    (dist + rSmall ≤ rBig) — never merely for being near one. Kept leaves are
  //    indexed in a world-space grid so each candidate checks just its 3×3
  //    neighbourhood; world-space keys mean panning can never change the result.
  // 3. Selection exemption: leaves picked out by the legend class filter or
  //    Decisive Node Analysis bypass the occlusion test (2) — hiding a selected
  //    leaf behind a bigger unselected one defeats the selection. They are not
  //    exempt from the viewport test (1): off-screen is off-screen either way.
  const visible = useMemo(() => {
    if (!layout) return null
    const { branches, leaves, maxLeafSamples } = layout
    const { k, x: tx, y: ty } = cullTransform
    const { w, h } = svgSize
    const inkScale = Math.pow(k / fitScale, ZOOM_DAMP - 1)

    let vBranches = branches
    if (branches.length > CULL_MIN_BRANCHES) {
      const pad = Math.max(160, w * 0.25)
      vBranches = branches.filter(b => {
        const ax = b.x1 * k + tx, ay = b.y1 * k + ty
        const bx = b.x2 * k + tx, by = b.y2 * k + ty
        return Math.max(ax, bx) >= -pad && Math.min(ax, bx) <= w + pad &&
               Math.max(ay, by) >= -pad && Math.min(ay, by) <= h + pad
      })
    }

    let vLeaves = leaves
    if (leaves.length > CULL_MIN_BRANCHES) {
      const maxScreenR = MAX_NODE_PX * leafScale * inkScale * k
      const pad = Math.max(160, w * 0.25) + maxScreenR
      const inView = leaves.filter(l => {
        const sx = l.x * k + tx, sy = l.y * k + ty
        return sx >= -pad && sx <= w + pad && sy >= -pad && sy <= h + pad
      })
      if (inView.length > OCCLUDE_MIN_LEAVES && maxScreenR > 0) {
        const rScreen = (samples: number) =>
          leafRadius(samples, maxLeafSamples, leafScale) * inkScale * k
        // Any covering leaf lies within maxScreenR of the covered one, i.e.
        // within one grid cell — the 3×3 neighbourhood scan is exhaustive.
        const cellWorld = maxScreenR / k
        // Big leaves first: only already-kept leaves can cover a candidate.
        // Deterministic order → identical result on every recompute (no popping);
        // smaller leaves also end up later in the DOM, i.e. painted on top.
        const sorted = [...inView].sort((a, b) => b.samples - a.samples)
        const grid = new Map<number, LeafDatum[]>()
        const kept: LeafDatum[] = []
        for (const l of sorted) {
          const cx = Math.round(l.x / cellWorld)
          const cy = Math.round(l.y / cellWorld)
          const rC = rScreen(l.samples)
          // Selected leaves (legend class / DCA) are exempt from the occlusion
          // test: a small highlighted leaf tucked behind a big neighbour is
          // precisely what the selection exists to surface, so it must never be
          // dropped for being covered. Nothing is selected → no check at all.
          let covered = false
          if (!isHighlighted?.(l.data)) {
            for (let dx = -1; dx <= 1 && !covered; dx++) {
              for (let dy = -1; dy <= 1 && !covered; dy++) {
                const arr = grid.get((cx + dx) * 65536 + (cy + dy))
                if (!arr) continue
                for (const kp of arr) {
                  const dist = Math.hypot(kp.x - l.x, kp.y - l.y) * k
                  if (dist + rC <= rScreen(kp.samples) + 0.25) { covered = true; break }
                }
              }
            }
          }
          if (!covered) {
            kept.push(l)
            const key = cx * 65536 + cy
            const arr = grid.get(key)
            if (arr) arr.push(l)
            else grid.set(key, [l])
          }
        }
        vLeaves = kept
      } else {
        vLeaves = inView
      }
    }

    return { branches: vBranches, leaves: vLeaves }
    // isHighlighted changes identity only when a selection changes, so the
    // culling recomputes exactly when the exemption set does.
  }, [layout, cullTransform, svgSize, fitScale, isHighlighted, leafScale])

  // Stable reset-view callback — also called by the home button.
  const resetView = useCallback(() => {
    if (!layout || !svgRef.current || !zoomRef.current) return
    const { bounds } = layout
    const tx = svgSize.w / 2 - ((bounds.x0 + bounds.x1) / 2) * fitScale
    const ty = svgSize.h / 2 - ((bounds.y0 + bounds.y1) / 2) * fitScale
    zoomRef.current.scaleExtent([fitScale * 0.75, 10])
    d3.select(svgRef.current).call(
      zoomRef.current.transform,
      d3.zoomIdentity.translate(tx, ty).scale(fitScale)
    )
  }, [layout, svgSize, fitScale])

  const downloadSVG = useCallback(() => {
    if (treeState.state !== 'hasData') return
    const reg = task === 'regression' && activeRanges
      ? { ranges: activeRanges, colors: { mean: meanColors, std: stdColors } }
      : undefined
    const svg = treeToSVGString(treeState.data, effectiveColors, collapsed, leafMode, reg, group, leafScale)
    const blob = new Blob([svg], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `tree-${treeId}.svg`
    a.click()
    URL.revokeObjectURL(url)
  }, [treeState, treeId, collapsed, leafMode, task, activeRanges, meanColors, stdColors, group,
      effectiveColors, leafScale])

  // Auto-fit once per tree; also update scaleExtent whenever layout changes.
  useEffect(() => {
    if (!layout || !zoomRef.current) return
    zoomRef.current.scaleExtent([fitScale * 0.75, 10])
    if (!fitDone.current) {
      resetView()
      fitDone.current = true
    }
  }, [layout, fitScale, resetView])

  // Loading / error messages live OUTSIDE the zoomed <g> (screen coordinates).
  let overlayText: ReactNode = null
  let content: ReactNode = null

  if (treeState.state === 'loading') {
    overlayText = (
      <text x={svgSize.w / 2} y={svgSize.h / 2} textAnchor="middle" fill="#999" fontSize={14}>
        Loading tree {treeId}…
      </text>
    )
  } else if (treeState.state === 'hasError') {
    overlayText = (
      <text x={svgSize.w / 2} y={svgSize.h / 2} textAnchor="middle" fill="red" fontSize={14}>
        Error loading tree {treeId}
      </text>
    )
  } else if (layout && visible) {
    const { branches, leaves } = visible
    const { maxLeafSamples } = layout
    // Damped ink scaling (see ZOOM_DAMP): the rendered SVG attribute is
    // base · (k/fit)^(DAMP−1), so on screen ink grows as (k/fit)^DAMP while the
    // geometry grows as k. One shared factor — no per-leaf clamping, so leaves
    // with different sample counts never collapse to a common size.
    // Leaves receive the BASE radius + inkScale separately so their props stay
    // memo-stable during zoom (only the cheap group scale() changes).
    const inkScale = Math.pow(transform.k / fitScale, ZOOM_DAMP - 1)
    const nodeRBase = (samples: number) => leafRadius(samples, maxLeafSamples, leafScale)
    // Decisive Node Analysis focus: when a valid pair is chosen, fade everything
    // that isn't part of a decisive split / its pure paths — the same phase-out
    // treatment as the Legend class filter and the Ctrl-held branch mode.
    // Highlighted reference paths fade everything off them the same way.
    const focusFaded = (id: number) =>
      (decisiveMap !== null && !decisiveMap.has(id)) ||
      (pathNodeIds !== null && !pathNodeIds.has(id))
    // Legend class highlight: leaves whose dominant class isn't the selected one.
    const legendFiltered = (data: TreeNodeData) =>
      task === 'classification' && selectedClass !== null &&
      dominantClassOf(sliceValue(data.value, group)) !== selectedClass
    // Faded nodes (either highlight mode) are also INERT — no hover disc, no
    // click — so they can never steal focus from the highlighted nodes they
    // visually sit behind.
    const fadedOut = (data: TreeNodeData) => legendFiltered(data) || focusFaded(data.id)
    content = (
      <>
        {/* Pass 1 — visual strokes only, no pointer events */}
        {branches.map(b => (
          <line
            key={b.nodeId}
            x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2}
            stroke={b.color}
            strokeWidth={b.width * inkScale}
            strokeLinecap="round"
            opacity={focusFaded(b.nodeId) ? 0.1 : 1}
            style={{ pointerEvents: 'none' }}
          />
        ))}

        {/* Pass 2 — true leaf nodes (never interactive, purely visual) */}
        <g style={{ pointerEvents: 'none' }}>
          {leaves.filter(l => !l.isCollapsed).map(leaf => {
            return (
              <g key={leaf.data.id} opacity={ctrlHeld ? 0.08 : fadedOut(leaf.data) ? 0.1 : 1}>
                <TreeNode
                  data={leaf.data}
                  x={leaf.x}
                  y={leaf.y}
                  radius={nodeRBase(leaf.samples)}
                  inkScale={inkScale}
                  isCollapsed={false}
                  highlighted={visibleHoverTarget?.data.id === leaf.data.id}
                  regressionColor={leafColorFn ? leafColorFn(leaf.data.value[group.offset] ?? 0, impurityAt(leaf.data.impurity, group.index)) : undefined}
                />
              </g>
            )
          })}
        </g>

        {/* Pass 3 — ghost interaction lines. Single click pins the info widget;
            DOUBLE click collapses/expands (direction from the branch's effective
            state). The same double-click affordance exists on the collapsed node
            ring + hover disc (Pass 4/5), which sit above these. */}
        {branches.map(b => b.hasChildren ? (
          <line
            key={b.nodeId}
            x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2}
            stroke="transparent"
            strokeWidth={Math.max(CLICK_AREA_PX / transform.k, b.width * inkScale)}
            style={{
              cursor: ctrlHeld && b.collapsed ? 'default' : 'pointer',
              // Faded elements are inert — they must not soak up hovers or
              // clicks meant for the highlighted paths crossing them. Legend
              // fading applies to collapsed endpoints only (internal branches
              // keep their color under the legend filter and stay interactive).
              pointerEvents:
                (ctrlHeld && b.collapsed) || focusFaded(b.nodeId) ||
                (b.collapsed && legendFiltered(b.data))
                  ? 'none' : 'auto',
            }}
            onClick={(e) => {
              pinNode(b.nodeId, e)
              if (hasRefs && e.altKey) openNode(b.nodeId)
            }}
            onDoubleClick={(e) => {
              e.stopPropagation()
              toggleCollapsed(b.nodeId, b.collapsed)
            }}
            onMouseEnter={() => setHoverState({ treeId, target: { data: b.data, x: b.x2, y: b.y2 } })}
            onMouseLeave={() => setHoverState({ treeId, target: null })}
          >
            <title>
              {'Click: pin info · '
                + (b.collapsed ? 'Double-click: expand' : 'Double-click: collapse')
                + (hasRefs ? ' · Alt+click: inspect references' : '')}
            </title>
          </line>
        ) : null)}

        {/* Pass 4 — collapsed node rings on top; their double-click-to-expand
            wins over any branch line underneath. Faded + non-interactive in
            Ctrl mode; highlight-faded nodes are inert too. */}
        <g style={{ pointerEvents: ctrlHeld ? 'none' : 'auto' }}>
          {leaves.filter(l => l.isCollapsed).map(leaf => {
            const faded = fadedOut(leaf.data)
            return (
              <g
                key={leaf.data.id}
                opacity={ctrlHeld ? 0.08 : faded ? 0.1 : 1}
                style={faded ? { pointerEvents: 'none' } : undefined}
              >
                <TreeNode
                  data={leaf.data}
                  x={leaf.x}
                  y={leaf.y}
                  radius={nodeRBase(leaf.samples)}
                  inkScale={inkScale}
                  isCollapsed={true}
                  highlighted={visibleHoverTarget?.data.id === leaf.data.id}
                  onClick={(e) => {
                    pinNode(leaf.data.id, e)
                    if (hasRefs && e.altKey) openNode(leaf.data.id)
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation()
                    toggleCollapsed(leaf.data.id, true)
                  }}
                  regressionColor={leafColorFn ? leafColorFn(leaf.data.value[group.offset] ?? 0, impurityAt(leaf.data.impurity, group.index)) : undefined}
                />
              </g>
            )
          })}
        </g>

        {/* Pass 4.5 — Decisive Node Analysis: an aggregate pie at each side of a
            decisive split whose root is an internal node (which would otherwise
            draw no distribution at all). They behave like any other node —
            hover for the widget, click to pin, double-click to collapse. */}
        {dcaSummaries && (
          <>
            {/* Ctrl (branch view) fades node glyphs out of the way, exactly as
                it does for leaves in Pass 2/4 — these are node glyphs too. A
                highlighted reference path fades the ones off it as well. */}
            <g style={{ pointerEvents: 'none' }} opacity={ctrlHeld ? 0.08 : 1}>
              {dcaSummaries.map(s => (
                <g key={`dca-sum-${s.id}`} opacity={focusFaded(s.id) ? 0.1 : 1}>
                <TreeNode
                  data={s.data}
                  x={s.x}
                  y={s.y}
                  radius={Math.min(MAX_NODE_PX * leafScale, nodeRBase(s.data.n_node_samples))}
                  inkScale={inkScale}
                  isCollapsed={false}
                  highlighted={visibleHoverTarget?.data.id === s.id}
                />
                </g>
              ))}
            </g>
            {/* Hit targets: these nodes aren't in `leaves`, so Pass 5 never
                gives them a hover disc — without this they'd be inert. */}
            {!ctrlHeld && dcaSummaries.filter(s => !focusFaded(s.id)).map(s => (
              <circle
                key={`dca-hit-${s.id}`}
                cx={s.x} cy={s.y}
                r={(Math.min(MAX_NODE_PX * leafScale, nodeRBase(s.data.n_node_samples)) + 6) * inkScale}
                fill="transparent"
                style={{ pointerEvents: 'all', cursor: 'pointer' }}
                onMouseEnter={() => setHoverState({ treeId, target: { data: s.data, x: s.x, y: s.y } })}
                onMouseLeave={() => setHoverState({ treeId, target: null })}
                onClick={(e) => {
                  pinNode(s.id, e)
                  if (hasRefs && e.altKey) openNode(s.id)
                }}
                onDoubleClick={(e) => {
                  e.stopPropagation()
                  toggleCollapsed(s.id, false)   // rendered expanded by definition
                }}
              />
            ))}
          </>
        )}

        {/* Pass 5 — transparent hover discs, topmost z-order.
            Hidden in Ctrl mode so branch interaction stays clean.
            Collapsed discs forward double-clicks to toggleCollapsed; otherwise
            the disc would silently eat the gesture Pass 3/4 expand on.
            Highlight-faded leaves get NO disc at all: hovers and clicks fall
            straight through to the highlighted nodes/branches beneath them. */}
        {!ctrlHeld && leaves.filter(l => !fadedOut(l.data)).map(leaf => (
          <circle
            key={leaf.data.id}
            cx={leaf.x}
            cy={leaf.y}
            r={(nodeRBase(leaf.samples) + 6) * inkScale}
            fill="transparent"
            style={{ pointerEvents: 'all', cursor: 'pointer' }}
            onMouseEnter={() => setHoverState({ treeId, target: { data: leaf.data, x: leaf.x, y: leaf.y } })}
            onMouseLeave={() => setHoverState({ treeId, target: null })}
            // Plain click pins the widget (every node is clickable for that);
            // Alt+click inspects references — the same modifier as on branches;
            // double-click expands a collapsed node (see Pass 3).
            onClick={(e) => {
              pinNode(leaf.data.id, e)
              if (hasRefs && e.altKey) openNode(leaf.data.id)
            }}
            onDoubleClick={leaf.isCollapsed ? (e) => {
              e.stopPropagation()
              toggleCollapsed(leaf.data.id, true)
            } : undefined}
          />
        ))}
      </>
    )
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden' }}>
      <div style={{
        padding: '4px 8px',
        borderBottom: '1px solid #ccc',
        flexShrink: 0,
        background: '#fff',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontFamily: HEADING_FONT, fontWeight: 700, fontSize: 12 }}>
            Tree {treeId} — height {meta.height} · {meta.n_nodes} nodes
          </span>
          {(collapsed.size > 0 || expanded.size > 0) && (
            <span
              onClick={() => setCollapsedState({ treeId, collapsed: new Set(), expanded: new Set() })}
              style={{ fontSize: 11, color: '#3A8C3A', cursor: 'pointer', fontWeight: 400, marginLeft: 'auto' }}
            >
              reset collapse ({collapsed.size + expanded.size} override{collapsed.size + expanded.size !== 1 ? 's' : ''})
            </span>
          )}
          {myHighlightedRefIds.length > 0 && (
            <span
              onClick={() => setHighlightedRefs(prev => {
                const next = { ...prev }
                delete next[treeId]
                return next
              })}
              title="Stop highlighting the reference paths selected in NodeView"
              style={{
                fontSize: 11, color: '#3A8C3A', cursor: 'pointer', fontWeight: 400,
                // shares the right edge with "reset collapse" when both show
                marginLeft: collapsed.size > 0 || expanded.size > 0 ? 0 : 'auto',
              }}
            >
              reset path highlight ({myHighlightedRefIds.length} path{myHighlightedRefIds.length !== 1 ? 's' : ''})
            </span>
          )}
        </div>
        {(Object.keys(meta.metrics).length > 0 || Object.keys(groupMetrics).length > 0) && (
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', marginTop: 3 }}>
            {/* Scope marker: every metric on this row is for THIS tree alone,
                never the ensemble — a forest averages its trees' posteriors and
                normally scores well above any single tree. */}
            <span
              style={{ fontSize: 9, color: '#bbb', textTransform: 'uppercase', letterSpacing: 0.6 }}
              title="These values describe this tree on its own — not the forest. The ensemble combines all trees and normally scores higher."
            >
            </span>
            {/* Tree-global custom metrics, with the forest ⌀ ± σ for context */}
            {Object.entries(meta.metrics).map(([k, v]) => {
              const s = treeMetricStats[k]
              return (
                <span key={k} style={{ fontSize: 10, color: '#888' }}
                  title={`${k} of tree ${treeId} alone — not the forest's ${k}`
                       + (s && s.n > 1
                         ? `. Across the ${s.n} trees: mean ${fmtMetric(s.mean)}, σ ${fmtMetric(s.std)}, `
                           + `range ${fmtMetric(s.min)} – ${fmtMetric(s.max)} — still per-tree values, `
                           + `not the ensemble score.`
                         : '')}>
                  <span style={{ color: '#bbb' }}>{k}</span>
                  {' '}
                  <span style={{ fontWeight: 600, color: '#555' }}>
                    {Number.isInteger(v) ? v : v.toFixed(3)}
                  </span>
                  {s && s.n > 1 && (
                    <span style={{ color: '#c4c4c4' }}>
                      {' '}(⌀ {fmtMetric(s.mean)} ± {fmtMetric(s.std)})
                    </span>
                  )}
                </span>
              )
            })}
            {/* Custom metrics of the ACTIVE label group / target, same treatment */}
            {Object.entries(groupMetrics).map(([k, v]) => {
              const s = groupMetricStats[k]
              return (
                <span key={`g-${k}`} style={{ fontSize: 10, color: '#3A7C3A' }}
                  title={`${k} of ${group.name || 'this group'} for tree ${treeId} alone`
                       + (s && s.n > 1
                         ? `. Across the ${s.n} trees: mean ${fmtMetric(s.mean)}, σ ${fmtMetric(s.std)}, `
                           + `range ${fmtMetric(s.min)} – ${fmtMetric(s.max)} — means of the individual `
                           + `trees, not the ensemble score.`
                         : '')}>
                  <span style={{ color: '#8fbc8f' }}>{group.name ? `${group.name} · ` : ''}{k}</span>
                  {' '}
                  <span style={{ fontWeight: 600 }}>
                    {Number.isInteger(v) ? v : v.toFixed(3)}
                  </span>
                  {s && s.n > 1 && (
                    <span style={{ color: '#9cba9c' }}>
                      {' '}(⌀ {fmtMetric(s.mean)} ± {fmtMetric(s.std)})
                    </span>
                  )}
                </span>
              )
            })}
          </div>
        )}
      </div>
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <svg
          ref={svgRef}
          width={svgSize.w}
          height={svgSize.h}
          style={{ cursor: 'grab', display: 'block', overflow: 'hidden' }}
        >
          {overlayText}
          {/* transform is set imperatively by the zoom handler — React never
              writes it, so zoom ticks cost zero React work */}
          <g ref={gRef} transform={transform.toString()}>
            {content}
          </g>
        </svg>
        {widgetTarget && (
          <NodeWidget
            // Remount per node so transient widget state (expanded class list)
            // never carries over to a different node.
            key={widgetTarget.data.id}
            data={widgetTarget.data}
            x={widgetTarget.x * transform.k + transform.x}
            y={widgetTarget.y * transform.k + transform.y}
            containerW={svgSize.w}
            containerH={svgSize.h}
            classLabels={classLabels}
            // Reference hint — only when this iteration has references
            refCount={refCounts ? refCounts.get(widgetTarget.data.id) ?? 0 : undefined}
            // A true leaf has no double-click action; anything with children
            // does (collapsed subtrees included — they keep their children).
            canCollapse={widgetTarget.data.children.length > 0}
            pinned={widgetPinned}
            onClose={unpin}
          />
        )}

        {/* Leaf-size slider — one multiplier on every leaf radius (see
            leafRadius); shared by all tree views. Double-click resets to 1×. */}
        <div
          title="Leaf size — double-click to reset"
          onDoubleClick={() => setLeafScale(1)}
          style={{
            position: 'absolute', bottom: 14, left: 14, height: 32,
            display: 'flex', alignItems: 'center', gap: 6, padding: '0 10px',
            borderRadius: 6, border: '1px solid #ccc', background: '#fff',
            boxShadow: '0 1px 4px rgba(0,0,0,0.15)', opacity: 0.85,
            fontSize: 11, color: '#666',
          }}
        >
          <span>Leaf size</span>
          <input
            type="range"
            min={LEAF_SCALE_MIN}
            max={LEAF_SCALE_MAX}
            step={0.05}
            value={leafScale}
            onChange={e => setLeafScale(Number(e.target.value))}
            style={{ width: 90, cursor: 'pointer', accentColor: '#3A8C3A' }}
          />
          <span style={{ width: 34, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
            {leafScale.toFixed(2)}×
          </span>
        </div>

        <button
          onClick={downloadSVG}
          title="Download as SVG"
          style={{
            position: 'absolute',
            bottom: 14,
            right: 80,
            width: 72,
            height: 32,
            borderRadius: 6,
            border: '1px solid #ccc',
            background: '#fff',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 14,
            boxShadow: '0 1px 4px rgba(0,0,0,0.15)',
            opacity: 0.85,
          }}
        >
          Download
        </button>
        <button
          onClick={resetView}
          title="Reset view"
          style={{
            position: 'absolute',
            bottom: 14,
            right: 14,
            width: 64,
            height: 32,
            borderRadius: 6,
            border: '1px solid #ccc',
            background: '#fff',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 16,
            boxShadow: '0 1px 4px rgba(0,0,0,0.15)',
            opacity: 0.85,
          }}
        >
          Reset
        </button>
      </div>
    </div>
  )
}
