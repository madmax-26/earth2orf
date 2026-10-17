import { useState, useLayoutEffect, useRef, useMemo, useCallback, type CSSProperties } from 'react'
import * as d3 from 'd3'
import { useAtom, useAtomValue } from 'jotai'
import { loadableForestMap } from '../Forest/ForestAtoms'
import { selectedTreeIdAtom } from '../ForestView/ForestViewAtoms'
import TreeThumbnail from '../TreeThumbnail/TreeThumbnail'

// ── Classical MDS — used only to seed the force layout ─────────────────────────

function classicalMDS(D: number[][]): [number, number][] {
  const n = D.length
  if (n === 0) return []
  if (n === 1) return [[0, 0]]
  if (n === 2) return [[-0.5, 0], [0.5, 0]]

  const D2 = D.map(row => row.map(d => d * d))
  const rowMean = D2.map(row => row.reduce((a, b) => a + b, 0) / n)
  const totalMean = rowMean.reduce((a, b) => a + b, 0) / n
  const B = D2.map((row, i) =>
    row.map((d2, j) => -0.5 * (d2 - rowMean[i] - rowMean[j] + totalMean))
  )

  const matVec = (M: number[][], v: number[]) =>
    M.map(row => row.reduce((s, m, j) => s + m * v[j], 0))
  const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
  const norm = (v: number[]) => {
    const len = Math.sqrt(dot(v, v))
    return len > 1e-12 ? v.map(x => x / len) : v
  }

  let v1 = norm(Array.from({ length: n }, (_, i) => Math.sin((i + 1) * 1.3)))
  for (let k = 0; k < 200; k++) v1 = norm(matVec(B, v1))
  const λ1 = dot(v1, matVec(B, v1))

  const B2 = B.map((row, i) => row.map((b, j) => b - λ1 * v1[i] * v1[j]))
  let v2 = norm(Array.from({ length: n }, (_, i) => Math.cos((i + 1) * 1.7)))
  for (let k = 0; k < 200; k++) v2 = norm(matVec(B2, v2))
  const λ2 = dot(v2, matVec(B, v2))

  const s1 = Math.sqrt(Math.max(0, λ1))
  const s2 = Math.sqrt(Math.max(0, λ2))
  return Array.from({ length: n }, (_, i) => [s1 * v1[i], s2 * v2[i]])
}

const BOTTOM_LABEL = 18

function thumbSize(T: number): number {
  return T <= 6 ? 34 : T <= 12 ? 26 : T <= 20 ? 18 : 0
}

// ── ForestMap ─────────────────────────────────────────────────────────────────

export default function ForestMap() {
  const mapState    = useAtomValue(loadableForestMap)
  const [selectedId, setSelectedId] = useAtom(selectedTreeIdAtom)
  const [hov, setHov] = useState<string | null>(null)

  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [size, setSize] = useState({ w: 260, h: 300 })
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity)

  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSize({ w: Math.max(100, width), h: Math.max(100, height) })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // Attach D3 zoom + pan via a callback ref rather than a mount-only effect.
  // The <svg> only exists once mapState reaches 'hasData' (earlier states
  // render a placeholder <div>), so an effect with an empty dep array can run
  // before the <svg> ever mounts and then never fire again — leaving zoom
  // permanently unbound on the first open. A callback ref instead fires
  // exactly when the node is actually attached/detached, however late.
  const bindSvg = useCallback((el: SVGSVGElement | null) => {
    if (el) {
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.4, 8])
        .on('zoom', (e: d3.D3ZoomEvent<SVGSVGElement, unknown>) => setTransform(e.transform))
      zoomRef.current = zoom
      d3.select(el).call(zoom)
    } else if (svgRef.current) {
      d3.select(svgRef.current).on('.zoom', null)
      zoomRef.current = null
    }
    svgRef.current = el
  }, [])

  const resetView = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return
    d3.select(svgRef.current).call(zoomRef.current.transform, d3.zoomIdentity)
  }, [])

  // Distance-faithful layout: spread freely so spacing tracks the similarity
  // distance, then fit-to-fill the panel and relax overlaps. Recomputed when
  // the distance matrix or the panel size changes.
  const layout = useMemo(() => {
    if (mapState.state !== 'hasData' || !mapState.data) return null
    const { tree_ids, distances } = mapState.data
    const T = tree_ids.length
    // `distances` is EMPTY when source === 'none' (no correlation matrix) while
    // tree_ids is still populated, so T alone is not enough to guard on — this
    // memo is a hook and runs before the disabled-state early return below.
    if (T < 2 || distances.length !== T) return null

    const THUMB = thumbSize(T)
    const HALF  = THUMB > 0 ? THUMB / 2 : 6
    const PAD   = HALF + 16

    const usableTop = PAD
    const usableBot = size.h - PAD - BOTTOM_LABEL
    const availW = Math.max(40, size.w - PAD * 2)
    const availH = Math.max(40, usableBot - usableTop)

    type N = d3.SimulationNodeDatum & { idx: number }

    // Seed from MDS (scaled up to roughly the target spread)
    const mds = classicalMDS(distances)
    const nodes: N[] = mds.map(([x, y], i) => ({ idx: i, x: x * 300, y: y * 300 }))

    const links: { source: number; target: number; d: number }[] = []
    for (let i = 0; i < T; i++)
      for (let j = i + 1; j < T; j++)
        links.push({ source: i, target: j, d: distances[i][j] })

    // Phase 1 — distance-faithful spread in free space.
    // Link rest length is proportional to the actual distance, so distinct
    // trees sit far apart and similar trees stay close.
    d3.forceSimulation(nodes)
      .force('link', d3.forceLink<N, (typeof links)[number]>(links)
        .id(d => d.idx)
        .distance(l => 30 + l.d * 620)
        .strength(0.55))
      .force('charge', d3.forceManyBody().strength(-220))
      .force('collide', d3.forceCollide(HALF + 4))
      .stop()
      .tick(400)

    // Phase 2 — uniform fit-to-fill so the farthest pair spans the panel
    // (preserves relative distances; just scales + centres).
    const xs = nodes.map(n => n.x ?? 0), ys = nodes.map(n => n.y ?? 0)
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1
    const scale = Math.min(availW / spanX, availH / spanY) * 0.92
    const srcCX = (minX + maxX) / 2, srcCY = (minY + maxY) / 2
    const dstCX = PAD + availW / 2, dstCY = usableTop + availH / 2

    const fitted: N[] = nodes.map(n => ({
      idx: n.idx,
      x: dstCX + ((n.x ?? 0) - srcCX) * scale,
      y: dstCY + ((n.y ?? 0) - srcCY) * scale,
    }))

    // Phase 3 — collision-only relax in pixel space, anchored to the fitted
    // positions so overlaps separate without disturbing the global structure.
    const anchorX = fitted.map(p => p.x ?? 0), anchorY = fitted.map(p => p.y ?? 0)
    d3.forceSimulation(fitted)
      .force('collide', d3.forceCollide(HALF + 3).strength(0.9))
      .force('x', d3.forceX<N>(d => anchorX[d.idx]).strength(0.15))
      .force('y', d3.forceY<N>(d => anchorY[d.idx]).strength(0.15))
      .stop()
      .tick(90)

    const coords = fitted.map(n => [n.x ?? dstCX, n.y ?? dstCY] as [number, number])
    return { tree_ids, coords, THUMB, HALF }
  }, [mapState, size.w, size.h])

  // ── States ──
  if (mapState.state === 'loading') {
    return <div ref={containerRef} style={fillStyle}><span style={msgStyle}>Computing…</span></div>
  }
  if (mapState.state === 'hasError' || !mapState.data) {
    return <div ref={containerRef} style={fillStyle}><span style={{ ...msgStyle, color: '#e53935' }}>Map error</span></div>
  }
  // source 'none' = the forest carries no correlation matrix. The map is
  // disabled rather than fenced with a structural fallback: a fingerprint built
  // from node statistics was measured against real per-tree prediction
  // agreement on three datasets and tracked it at ~zero every time, so a map
  // drawn from it would look authoritative while meaning nothing.
  if (mapState.data.source === 'none' && mapState.data.tree_ids.length >= 2) {
    return (
      <div ref={containerRef} style={{ ...fillStyle, flexDirection: 'column', gap: 6, padding: '0 18px' }}>
        <span style={{ ...msgStyle, color: '#ad6800', fontWeight: 600 }}>
          Forest Map unavailable
        </span>
        <span style={{ ...msgStyle, fontSize: 10, lineHeight: 1.5, textAlign: 'center' }}>
          This iteration has no tree correlation matrix, so there is no way to
          tell how similarly the trees behave.
        </span>
        <span style={{ ...msgStyle, fontSize: 10, lineHeight: 1.5, textAlign: 'center', color: '#bbb' }}>
          Pass evaluation samples to the converter when pushing — e.g.
          <code style={{ fontSize: 10 }}> convert_arf(arf, eval_data=…)</code> or
          <code style={{ fontSize: 10 }}> convert_polsar_rf(rf, eval_patches=…, eval_labels=…)</code>
          — and the map appears.
        </span>
      </div>
    )
  }
  if (!layout) {
    return <div ref={containerRef} style={fillStyle}><span style={msgStyle}>Need ≥ 2 trees</span></div>
  }

  const { tree_ids, coords, THUMB, HALF } = layout
  const k = transform.k

  return (
    <div ref={containerRef} style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'hidden' }}>
      <svg
        ref={bindSvg}
        width={size.w}
        height={size.h}
        style={{ display: 'block', cursor: 'grab' }}
      >
        {/* All content inside the zoom transform group */}
        <g transform={transform.toString()}>


          {/* Nodes — center follows zoom; thumbnail counter-scaled to stay constant screen size */}
          {tree_ids.map((treeId, i) => {
            const cx = coords[i][0]
            const cy = coords[i][1]
            const inv = 1 / k
            const isSelected = treeId === selectedId
            const isHov = treeId === hov

            return (
              <g key={treeId}
                transform={`translate(${cx},${cy})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedId(isSelected ? null : treeId)}
                onMouseEnter={() => setHov(treeId)}
                onMouseLeave={() => setHov(null)}
              >
                {/* counter-scale: thumbnail stays constant screen px */}
                <g transform={`scale(${inv})`}>
                  {THUMB > 0 ? (
                    <>
                      <rect
                        x={-HALF - 3} y={-HALF - 3}
                        width={THUMB + 6} height={THUMB + 6}
                        rx={6} fill="none"
                        stroke={isSelected ? '#3A8C3A' : isHov ? '#bbb' : 'none'}
                        strokeWidth={isSelected ? 2 : 1}
                      />
                      <clipPath id={`fmclip-${treeId}`}>
                        <rect x={-HALF} y={-HALF} width={THUMB} height={THUMB} rx={3} />
                      </clipPath>
                      <rect x={-HALF} y={-HALF} width={THUMB} height={THUMB} fill="#fafafa" rx={3} />
                      <foreignObject
                        x={-HALF} y={-HALF}
                        width={THUMB} height={THUMB}
                        clipPath={`url(#fmclip-${treeId})`}
                      >
                        <TreeThumbnail treeId={treeId} size={THUMB} />
                      </foreignObject>
                      <rect
                        x={-HALF} y={-HALF} width={THUMB} height={THUMB}
                        rx={3} fill="none"
                        stroke={isSelected ? '#3A8C3A' : '#ddd'}
                        strokeWidth={isSelected ? 1.5 : 1}
                      />
                    </>
                  ) : (
                    <circle r={isHov || isSelected ? HALF + 2 : HALF}
                      fill={isSelected ? '#3A8C3A' : '#8B6040'}
                      stroke={isSelected ? '#2a6a2a' : isHov ? '#555' : '#fff'}
                      strokeWidth={1.5} opacity={0.85}
                    />
                  )}
                  <text
                    y={HALF + 11}
                    textAnchor="middle" fontSize={9}
                    fontWeight={isSelected ? 700 : 400}
                    fill={isSelected ? '#3A8C3A' : '#999'}
                  >
                    T{treeId}
                  </text>
                </g>
              </g>
            )
          })}
        </g>
      </svg>

      {/* Reset view button */}
      <button
        onClick={resetView}
        title="Reset view"
        style={{
          position: 'absolute', top: 6, right: 6,
          width: 22, height: 22, lineHeight: '20px',
          padding: 0, fontSize: 12,
          border: '1px solid #ddd', borderRadius: 4,
          background: '#fff', color: '#888', cursor: 'pointer',
        }}
      >
        ⌂
      </button>

      {/* Source badge */}
      <div style={{
        position: 'absolute', bottom: 4, right: 8,
        fontSize: 8, color: '#ccc', fontStyle: 'italic', pointerEvents: 'none',
      }}>
        tree correlation
      </div>
    </div>
  )
}

const fillStyle: CSSProperties = {
  flex: 1, minHeight: 0, display: 'flex',
  alignItems: 'center', justifyContent: 'center',
}
const msgStyle: CSSProperties = { fontSize: 11, color: '#bbb' }
