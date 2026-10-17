import { useState, useMemo, useEffect, useRef, type RefObject, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useAtomValue } from 'jotai'
import { loadableSimilarity, activeTaskAtom, type SimilarityMetric, type PathStats } from '../Forest/ForestAtoms'
import { selectedTreeIdAtom } from '../ForestView/ForestViewAtoms'
import { classColor } from '../classColors'
import { effectiveClassColorsAtom } from '../Legend/LegendAtoms'
import { downloadSvgElement, sanitizeFilename } from '../svgExport'
import DecisiveNodeAnalysis from './DecisiveNodeAnalysis'
import { HEADING_FONT } from '../../../../theme'

// ── Classical MDS ─────────────────────────────────────────────────────────────

/** Classical MDS plus the share of the structure the 2D map actually shows.
 *
 *  `coverage` = (λ1 + λ2) / Σλ. The denominator is trace(B), because for a
 *  Euclidean D every eigenvalue of B is ≥ 0 and the trace is therefore exactly
 *  the sum of the positive ones — so the figure is exact without ever running a
 *  full eigendecomposition (verified to 4e-15 against one). It matters: with 24
 *  labels spread over 23 positive dimensions this lands near 32%, meaning most
 *  pairwise structure cannot survive the projection and on-screen adjacency is
 *  only a rough guide. */
type MDSResult = { coords: [number, number][]; coverage: number }

function classicalMDS(D: number[][]): MDSResult | null {
  const n = D.length
  if (n === 0) return { coords: [], coverage: 1 }
  if (n === 1) return { coords: [[0, 0]], coverage: 1 }

  const D2 = D.map(row => row.map(d => d * d))
  const rowMean = D2.map(row => row.reduce((a, b) => a + b, 0) / n)
  const totalMean = rowMean.reduce((a, b) => a + b, 0) / n

  const B = D2.map((row, i) =>
    row.map((d2, j) => -0.5 * (d2 - rowMean[i] - rowMean[j] + totalMean))
  )

  const matVec = (M: number[][], v: number[]): number[] =>
    M.map(row => row.reduce((s, m, j) => s + m * v[j], 0))
  const dot = (a: number[], b: number[]): number =>
    a.reduce((s, x, i) => s + x * b[i], 0)
  const normalize = (v: number[]): number[] => {
    const len = Math.sqrt(dot(v, v))
    return len > 1e-12 ? v.map(x => x / len) : v
  }

  let v1 = normalize(Array.from({ length: n }, (_, i) => Math.sin((i + 1) * 1.3)))
  for (let k = 0; k < 100; k++) v1 = normalize(matVec(B, v1))
  const λ1 = dot(v1, matVec(B, v1))

  const B2 = B.map((row, i) => row.map((b, j) => b - λ1 * v1[i] * v1[j]))
  let v2 = normalize(Array.from({ length: n }, (_, i) => Math.cos((i + 1) * 1.7)))
  for (let k = 0; k < 100; k++) v2 = normalize(matVec(B2, v2))
  const λ2 = dot(v2, matVec(B, v2))

  // A NON-EUCLIDEAN input (most commonly a distance matrix with a nonzero
  // diagonal) makes B indefinite, so the leading eigenvalues come out negative
  // and both scales clamp to 0 — every point would land on the origin and the
  // map would look like a single stacked dot with no hint why. Signal it
  // instead: callers render an explanation rather than a pile.
  if (λ1 <= 0) return null
  const s1 = Math.sqrt(Math.max(0, λ1))
  const s2 = Math.sqrt(Math.max(0, λ2))
  // trace(B) = Σλ — see the note above on why this is the exact denominator.
  const trace = B.reduce((s, row, i) => s + row[i], 0)
  return {
    coords: Array.from({ length: n }, (_, i) => [s1 * v1[i], s2 * v2[i]]),
    coverage: trace > 0 ? Math.max(0, Math.min(1, (λ1 + Math.max(0, λ2)) / trace)) : 1,
  }
}

// ── Color scale: blue (least overlap) → white → red (most overlap) ──────────
// Matrices are SIMILARITIES, so high = alike = red. Ends are NAMED per metric
// (METRIC_INFO.low / .high) because what "alike" means differs between them.

/** Off-diagonal extent of an overlap matrix, used to map the colour ramp onto
 *  the values actually present. A fixed [0,1] ramp wastes most of the scale — a
 *  30-class forest lands in roughly [0.01, 0.80], so the top of the ramp would
 *  never be reached. MIN_SPAN keeps a near-uniform matrix from being stretched
 *  into looking structured. The diagonal (always 1) is excluded. */
function distExtent(matrix: number[][]): [number, number] {
  const MIN_SPAN = 0.05
  let lo = Infinity, hi = -Infinity
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      if (i === j) continue          // diagonal is trivially 1 (self-overlap)
      const v = matrix[i][j]
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1]
  if (hi - lo < MIN_SPAN) {
    const mid = (lo + hi) / 2
    return [mid - MIN_SPAN / 2, mid + MIN_SPAN / 2]
  }
  return [lo, hi]
}

function distColorRgb(v: number, lo = 0, hi = 1): [number, number, number] {
  // v is an OVERLAP (similarity): high = the labels behave alike, so high = red.
  const t = Math.max(0, Math.min(1, (v - lo) / ((hi - lo) || 1)))
  const RED   = [215, 48,  39]  // most overlap
  const WHITE = [247, 247, 247]
  const BLUE  = [69,  117, 180] // least overlap
  const [from, to, s] = t <= 0.5
    ? [BLUE,  WHITE, t * 2]
    : [WHITE, RED,   (t - 0.5) * 2]
  return [
    Math.round(from[0] + s * (to[0] - from[0])),
    Math.round(from[1] + s * (to[1] - from[1])),
    Math.round(from[2] + s * (to[2] - from[2])),
  ]
}

function distColor(d: number, lo = 0, hi = 1): string {
  const [r, g, b] = distColorRgb(d, lo, hi)
  return `rgb(${r},${g},${b})`
}

// Contrast-appropriate text color for a value label drawn on top of distColor(d)
function distTextColor(d: number, lo = 0, hi = 1): string {
  const [r, g, b] = distColorRgb(d, lo, hi)
  const luminance = 0.299 * r + 0.587 * g + 0.114 * b
  return luminance > 150 ? '#333' : '#fff'
}

function abbrev(s: string, max = 8): string {
  return s.length <= max ? s : s.slice(0, max - 1) + '…'
}

// Human-readable duration for the "computed in …" note shown after the matrix
// arrives. The computation is O(C²·n) per tree (C labels, n nodes).
function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(1, Math.round(ms))}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  return `${m}m ${rem}s`
}

// ── Heatmap ───────────────────────────────────────────────────────────────────

/** The raw reading behind path_overlap: "80% of path shared (6.0 of 7.5 levels)".
 *  Two labels that ride most of the tree together before the forest can split
 *  them are similar even when the leaves separate them perfectly — which is the
 *  signal leaf co-occurrence structurally cannot see. Root-excluded, so these
 *  count shared decisions BELOW the root.
 *
 *  The percentage leads deliberately. Phrased the other way round ("share 6.0 of
 *  7.5 nodes") the counts get read against the tree total — 6 out of a 1300-node
 *  tree looks tiny — when the denominator is actually the root→leaf PATH LENGTH.
 *  "levels" says depth, not node count, and the percent is the part that has to
 *  agree with the cell colour. null when the backend did not send the stats. */
function pathReadout(i: number, j: number, path?: PathStats): string | null {
  const shared = path?.path_shared?.[i]?.[j]
  const len = path?.path_len
  if (shared === undefined || !len) return null
  const avg = (len[i] + len[j]) / 2
  if (!(avg > 0)) return null
  const pct = Math.round((shared / avg) * 100)
  return `${pct}% of path shared (${shared.toFixed(1)} of ${avg.toFixed(1)} levels)`
}

type HeatmapProps = {
  matrix: number[][]
  labels: string[]
  /** Only supplied while Path overlap is the selected metric. */
  path?: PathStats
  /** Names for the two ramp ends, from METRIC_INFO — what red/blue MEAN here. */
  scaleEnds?: [string, string]
  // Maximized/download mode: bigger cells, values written in every cell, and
  // the color scale embedded in the SVG itself (so a downloaded file is
  // self-contained). The minimized panel view never sets this — it keeps the
  // hover-only caption and the separately-rendered legend swatch below.
  expanded?: boolean
  svgRef?: RefObject<SVGSVGElement | null>
}

function Heatmap({ matrix, labels, path, scaleEnds = ['least overlap', 'most overlap'], expanded = false, svgRef }: HeatmapProps) {
  const [hovered, setHovered] = useState<[number, number] | null>(null)
  const N = labels.length
  if (N === 0) return null

  const LABEL_AREA = expanded ? 90 : 44
  const CELL = expanded
    ? Math.max(16, Math.min(44, Math.floor(560 / N)))
    : Math.max(6, Math.min(24, Math.floor(176 / N)))
  const GRID = N * CELL
  const LEGEND_H = expanded ? 46 : 20
  const SVG_W = Math.max(LABEL_AREA + GRID, expanded ? 320 : 0)
  const [cLo, cHi] = distExtent(matrix)
  const readout = hovered ? pathReadout(hovered[0], hovered[1], path) : null
  const SVG_H = LABEL_AREA + GRID + LEGEND_H + (readout ? 11 : 0)
  const fontSize = Math.max(7, Math.min(expanded ? 12 : 10, CELL - 1))
  const valueFontSize = Math.max(7, Math.min(11, CELL / 3))
  const labelMax = expanded ? 20 : 8

  return (
    <svg ref={svgRef} width={SVG_W} height={SVG_H} style={{ display: 'block', overflow: 'visible' }}>
      {labels.map((lbl, j) => (
        <text key={j}
          transform={`translate(${LABEL_AREA + j * CELL + CELL / 2},${LABEL_AREA - 4}) rotate(-45)`}
          textAnchor="start" fontSize={fontSize} fill="#999">
          {abbrev(lbl, labelMax)}
        </text>
      ))}
      {labels.map((lbl, i) => (
        <text key={i}
          x={LABEL_AREA - 4} y={LABEL_AREA + i * CELL + CELL / 2}
          textAnchor="end" dominantBaseline="middle" fontSize={fontSize} fill="#999">
          {abbrev(lbl, labelMax)}
        </text>
      ))}
      {matrix.map((row, i) =>
        row.map((d, j) => {
          const isHov = hovered?.[0] === i && hovered?.[1] === j
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={LABEL_AREA + j * CELL} y={LABEL_AREA + i * CELL}
                width={CELL} height={CELL}
                fill={distColor(d, cLo, cHi)}
                stroke={isHov ? '#333' : '#fff'} strokeWidth={isHov ? 1.5 : 0.5}
                onMouseEnter={() => setHovered([i, j])}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: 'default' }}
              />
              {expanded && (
                <text
                  x={LABEL_AREA + j * CELL + CELL / 2} y={LABEL_AREA + i * CELL + CELL / 2}
                  textAnchor="middle" dominantBaseline="middle"
                  fontSize={valueFontSize} fill={distTextColor(d, cLo, cHi)}
                  style={{ pointerEvents: 'none' }}
                >
                  {d.toFixed(2)}
                </text>
              )}
            </g>
          )
        })
      )}
      {hovered && (
        <text x={LABEL_AREA + GRID / 2} y={LABEL_AREA + GRID + 14}
          textAnchor="middle" fontSize={9} fill="#555">
          {abbrev(labels[hovered[0]], 12)} ↔ {abbrev(labels[hovered[1]], 12)}: {matrix[hovered[0]][hovered[1]].toFixed(3)}
        </text>
      )}
      {hovered && readout && (
        <text x={LABEL_AREA + GRID / 2} y={LABEL_AREA + GRID + 24}
          textAnchor="middle" fontSize={8} fill="#999">
          {readout}
        </text>
      )}
      {/* Embedded color scale — only in expanded/download mode, so the exported
          SVG is self-contained without relying on the surrounding page. */}
      {expanded && (
        <g transform={`translate(${LABEL_AREA},${LABEL_AREA + GRID + 26})`}>
          <text x={0} y={10} fontSize={11} fill="#3a6fad">{cLo.toFixed(2)}</text>
          <defs>
            <linearGradient id="simHeatmapEmbedGrad" x1="0" x2="1">
              <stop offset="0%"   stopColor={distColor(0)} />
              <stop offset="50%"  stopColor={distColor(0.5)} />
              <stop offset="100%" stopColor={distColor(1)} />
            </linearGradient>
          </defs>
          <rect x={52} y={2} width={140} height={12} rx={3} fill="url(#simHeatmapEmbedGrad)" stroke="#ddd" strokeWidth={0.5} />
          <text x={200} y={10} fontSize={11} fill="#c0392b">{cHi.toFixed(2)}</text>
          <text x={96} y={26} textAnchor="middle" fontSize={9} fill="#aaa">{scaleEnds[0]} → {scaleEnds[1]}</text>
        </g>
      )}
    </svg>
  )
}

// ── Dot Map (MDS scatter) ─────────────────────────────────────────────────────

type DotMapProps = {
  matrix: number[][]
  labels: string[]
  /** Only supplied while Path overlap is the selected metric. */
  path?: PathStats
  expanded?: boolean
  svgRef?: RefObject<SVGSVGElement | null>
}

function DotMap({ matrix, labels, path, expanded = false, svgRef }: DotMapProps) {
  // matrix holds OVERLAPS; classical MDS needs a Euclidean distance. sqrt(1 - s)
  // is exactly that (Hellinger for leaf co-occurrence; the feature-space distance
  // of the histogram-intersection kernel for path overlap), which is what makes
  // the embedding exact rather than approximate.
  const mds = useMemo(
    () => classicalMDS(matrix.map(row => row.map(v => Math.sqrt(Math.max(0, 1 - v))))),
    [matrix],
  )
  const embedFailed = mds === null
  const coords = mds?.coords ?? []
  const coverage = mds?.coverage ?? 1
  const customColors = useAtomValue(effectiveClassColorsAtom)
  const [hov, setHov] = useState<number | null>(null)
  const N = labels.length
  if (N === 0) return null
  // Non-embeddable distances (see classicalMDS). Say so rather than drawing a
  // pile of overlapping dots at the origin — the usual cause is a backend that
  // predates the overlap/distance switch, which the panel banner also flags.
  if (embedFailed || coords.length === 0) {
    return (
      <svg ref={svgRef} width={expanded ? 640 : 200} height={expanded ? 560 : 185}>
        <text x={(expanded ? 640 : 200) / 2} y={(expanded ? 560 : 185) / 2}
          textAnchor="middle" fontSize={11} fill="#c0392b">
          Cannot place these labels on a map
        </text>
        <text x={(expanded ? 640 : 200) / 2} y={(expanded ? 560 : 185) / 2 + 16}
          textAnchor="middle" fontSize={9} fill="#999">
          the matrix is not a valid distance matrix
        </text>
      </svg>
    )
  }

  const MAP_W = expanded ? 640 : 200
  const MAP_H = expanded ? 560 : 185
  const PAD = expanded ? 50 : 20
  const IW = MAP_W - PAD * 2, IH = MAP_H - PAD * 2
  const R = expanded ? Math.max(6, Math.min(14, 200 / N)) : Math.max(4, Math.min(8, 72 / N))
  const labelMax = expanded ? 24 : 8

  const xs = coords.map(c => c[0]), ys = coords.map(c => c[1])
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const spanX = maxX - minX || 1, spanY = maxY - minY || 1

  const sx = (x: number) => PAD + ((x - minX) / spanX) * IW
  const sy = (y: number) => PAD + ((y - minY) / spanY) * IH
  const labelFs = expanded ? Math.max(10, Math.min(14, R + 3)) : Math.max(7, Math.min(10, R + 2))

  // Hovering one label answers "what is this most like?": its highest-overlap
  // partner, plus how much of the tree the pair actually travels together.
  const CAPTION_H = 18   // reserved always, so hovering never shifts the layout
  let caption: string | null = null
  if (hov !== null) {
    let best = -1, bestV = -Infinity
    for (let j = 0; j < N; j++) {
      if (j !== hov && matrix[hov][j] > bestV) { bestV = matrix[hov][j]; best = j }
    }
    if (best >= 0) {
      const r = pathReadout(hov, best, path)
      caption = `most overlap: ${abbrev(labels[best], 14)} (${bestV.toFixed(3)})${r ? ` — ${r}` : ''}`
    }
  }
  // Idle state: say how much of the structure this projection can actually show.
  // Squeezing many dimensions into two discards the rest, so two dots sitting
  // together need not be alike — without this the map reads as more literal than
  // it is. Hover replaces it with the true highest-overlap partner, which is the
  // answer adjacency alone cannot give.
  const pctShown = Math.round(coverage * 100)
  const fidelity = `2D shows ${pctShown}% of the structure`

  return (
    <svg ref={svgRef} width={MAP_W} height={MAP_H + CAPTION_H} style={{ display: 'block', overflow: 'visible' }}>
      {/* Pass 1 — visuals only, never interactive. A label sprawls well past its
          own dot and is drawn after earlier dots, so if it could take the hover it
          would pin the selection to itself no matter which dot the cursor is over.
          Hit targets live in pass 2 instead, so every dot is equally reachable. */}
      <g style={{ pointerEvents: 'none' }}>
        {coords.map(([cx, cy], i) => {
          const px = sx(cx), py = sy(cy)
          const color = customColors[i] ?? classColor(i, N)
          const isHov = hov === i
          const anchor = px > MAP_W * 0.65 ? 'end' : 'start'
          const lx = px + (anchor === 'end' ? -(R + 3) : R + 3)
          return (
            <g key={i}>
              <circle cx={px} cy={py} r={isHov ? R + 2 : R}
                fill={color} stroke={isHov ? '#333' : '#fff'} strokeWidth={isHov ? 1.5 : 1}
                opacity={0.9} />
              <text x={lx} y={py} textAnchor={anchor} dominantBaseline="middle"
                fontSize={labelFs} fill={isHov ? '#333' : '#888'} fontWeight={isHov ? 700 : 400}>
                {abbrev(labels[i], labelMax)}
              </text>
            </g>
          )
        })}
      </g>
      {/* Pass 2 — invisible dot-sized hit targets, topmost. Slightly larger
          than the dot so small points stay grabbable. fill="none" (not
          "transparent", which is not valid SVG 1.1 paint: standalone viewers
          render it BLACK, and these circles sit on top of every dot in a
          downloaded map). pointer-events: all keeps them hoverable anyway. */}
      {coords.map(([cx, cy], i) => (
        <circle key={`hit-${i}`}
          cx={sx(cx)} cy={sy(cy)} r={R + 4}
          fill="none"
          style={{ pointerEvents: 'all', cursor: 'default' }}
          onMouseEnter={() => setHov(i)}
          onMouseLeave={() => setHov(null)}
        />
      ))}
      <text x={MAP_W / 2} y={MAP_H + 12} textAnchor="middle" fontSize={9}
        fill={caption ? '#888' : (pctShown < 50 ? '#ad6800' : '#bbb')}>
        {caption ?? fidelity}
      </text>
    </svg>
  )
}

// ── Expanded chart modal (maximize + download as SVG) ─────────────────────────
// Same chrome as ExpandedBarChart/ExpandedStackedBarChart: fixed overlay, click
// outside or Esc to close. The download button serializes whatever <svg> the
// caller renders inside via svgRef — the heatmap/dot map are drawn once, in
// their normal expanded={true} form, so the file matches what's on screen.

function ExpandedChartModal({ title, onClose, svgRef, children }: {
  title: string
  onClose: () => void
  svgRef: RefObject<SVGSVGElement | null>
  children: ReactNode
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const modal = (
    <div
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.45)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        // Above the MUI AppBar (z-index 1100) — otherwise resizing the box taller
        // (it grows both up and down from its centered position) lets the top
        // slide behind the status bar instead of over it.
        zIndex: 2000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#fff', borderRadius: 12,
          padding: '18px 24px 14px',
          maxWidth: '95vw', maxHeight: '88vh', minHeight: 200,
          overflow: 'auto', resize: 'both',
          boxShadow: '0 8px 36px rgba(0,0,0,0.18)',
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <span style={{ fontFamily: HEADING_FONT, fontWeight: 700, fontSize: 14, color: '#333', flex: 1 }}>
            {title}
          </span>
          <button
            onClick={() => downloadSvgElement(svgRef.current, `${sanitizeFilename(title)}.svg`)}
            title="Download as SVG"
            style={{
              fontSize: 11, fontWeight: 600, padding: '4px 10px',
              border: '1px solid #ccc', borderRadius: 4,
              background: '#fff', color: '#555', cursor: 'pointer',
            }}
          >
            ⬇ Download
          </button>
          <button
            onClick={onClose}
            style={{
              border: 'none', background: 'none',
              fontSize: 20, lineHeight: 1, padding: '0 2px',
              cursor: 'pointer', color: '#aaa',
            }}
          >
            ×
          </button>
        </div>

        {children}

        <div style={{ fontSize: 10, color: '#ddd', marginTop: 10, textAlign: 'right' }}>
          click outside or press Esc to close
        </div>
      </div>
    </div>
  )

  return createPortal(modal, document.body)
}

// ── Metric / source metadata ──────────────────────────────────────────────────

// low/high name the two ends of the colour ramp: low = least overlap (blue),
// high = most overlap (red). They are per-metric on purpose:
// a shared pair like "similar → distinct" hides WHY each metric says so, which is
// the whole distinction between the two.
const METRIC_INFO: Record<SimilarityMetric, {
  label: string; desc: string; title: string; low: string; high: string
}> = {
  leaf_cooccurrence: {
    label: 'Leaf co-occurrence',
    desc: 'Do the two labels still land in the same leaves? Residual confusion the forest failed to resolve — fades to no signal as leaves get pure.',
    title: 'Bhattacharyya coefficient between the two leaf-occupancy profiles. '
         + 'Adapted from Shi & Horvath 2006 / Zhu et al. 2014 Variant I.',
    low: 'own leaves', high: 'same leaves',
  },
  path_overlap: {
    label: 'Path overlap',
    desc: 'How much of their root→leaf journey the two labels share before the forest separates them. Keeps working when leaves are pure.',
    title: 'Overlap of the two node-occupancy profiles, over every node except the root. '
         + 'The numerator is exactly the expected number of nodes a random sample of each '
         + 'label shares. Zhu, Loy & Gong 2014, Eq. (11).',
    low: 'splits at root', high: 'splits deep',
  },
}

const METRICS = Object.keys(METRIC_INFO) as SimilarityMetric[]

// ── Shared button style helpers ───────────────────────────────────────────────

function pillBtn(active: boolean, color = '#3A8C3A') {
  return {
    padding: '3px 7px',
    border: '1px solid',
    borderColor: active ? color : '#ddd',
    borderRadius: 4,
    background: active ? color : '#fff',
    color: active ? '#fff' : '#888',
    fontSize: 10,
    fontWeight: active ? 700 : 400,
    cursor: 'pointer',
    lineHeight: 1.4,
  } as const
}

// ── Main component ────────────────────────────────────────────────────────────

type Source = 'forest' | 'pertree'

export default function Similarity() {
  const [metric, setMetric]   = useState<SimilarityMetric>('path_overlap')
  const [view, setView]       = useState<'heatmap' | 'dotmap'>('heatmap')
  const [source, setSource]   = useState<Source>('forest')
  // Maximize state — declared before any early return (rules of hooks): a
  // classification/regression forest switch must never change the hook count.
  const [heatmapExpanded, setHeatmapExpanded] = useState(false)
  const [dotmapExpanded, setDotmapExpanded]   = useState(false)
  const heatSvgRef = useRef<SVGSVGElement>(null)
  const dotSvgRef  = useRef<SVGSVGElement>(null)

  const simState     = useAtomValue(loadableSimilarity)
  const selectedTree = useAtomValue(selectedTreeIdAtom)
  const task         = useAtomValue(activeTaskAtom)

  // The similarity computation is expensive, but Decisive Node Analysis is
  // independent of it — so the similarity view is rendered into `body` and DCA
  // is always appended below, visible while the matrix is still computing.
  const body = renderSimilarityBody()

  return (
    <div style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 9 }}>
      {body}
      {/* Decisive Node Analysis — pick two labels + a threshold, highlight the
          splits that separate them in the selected tree's render */}
      <DecisiveNodeAnalysis />
    </div>
  )

  function renderSimilarityBody(): ReactNode {
  // Both metrics pair up labels via per-node class histograms, which a
  // regression node (a mean plus a spread) does not have. The fetch is skipped
  // for regression groups, so this also covers the null payload that produces.
  if (task === 'regression') {
    return (
      <div style={{ fontSize: 10, color: '#bbb', lineHeight: 1.5, padding: '4px 2px' }}>
        Available for classification forests only.
      </div>
    )
  }
  if (simState.state === 'loading') {
    return (
      <div style={{ padding: '18px 2px', fontSize: 11, color: '#bbb', textAlign: 'center' }}>
        Computing…
      </div>
    )
  }
  if (simState.state === 'hasError' || !simState.data) {
    return <div style={{ padding: '18px 2px', fontSize: 11, color: '#e53935', textAlign: 'center' }}>Error loading similarity</div>
  }

  const per_tree = simState.data.per_tree ?? {}
  const { labels, metrics } = simState.data
  const diagnostics = simState.data.diagnostics
  const treeCount = Object.keys(per_tree).length

  const majorityLabel =
    diagnostics?.imbalance_warning && diagnostics.majority_class !== null
      ? labels[diagnostics.majority_class]
      : null

  const matrix: number[][] | null =
    source === 'forest'
      ? (metrics[metric] ?? null)
      : (selectedTree !== null ? (per_tree[selectedTree]?.[metric] ?? null) : null)

  // Raw "share X of Y nodes" units, from whichever source is on screen. Only
  // handed to the charts while Path overlap is selected — it is that metric's
  // own reading, and would misdescribe a leaf co-occurrence number.
  const pathSource: PathStats | undefined =
    source === 'forest'
      ? simState.data
      : (selectedTree !== null ? per_tree[selectedTree] : undefined)
  const pathStats = metric === 'path_overlap' ? pathSource : undefined

  if (!matrix || labels.length < 2) {
    const msg = source === 'pertree' && selectedTree === null
      ? 'Select a tree in the forest view'
      : 'Need at least 2 class labels'
    return <div style={{ padding: '18px 2px', fontSize: 11, color: '#bbb', textAlign: 'center' }}>{msg}</div>
  }

  // Same extent the Heatmap uses, so the panel legend labels match the cells.
  const [heatLo, heatHi] = distExtent(matrix)

  const scaleEnds: [string, string] = [METRIC_INFO[metric].low, METRIC_INFO[metric].high]

  return (
    <>

      {/* Source toggle */}
      <div>
        <div style={{ fontSize: 9, fontWeight: 700, color: '#aaa', fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 5 }}>
          Source
        </div>
        <div style={{ display: 'flex', gap: 5 }}>
          <button style={{ ...pillBtn(source === 'forest'), flex: 1 }} onClick={() => setSource('forest')}>
            Forest Avg
          </button>
          <button style={{ ...pillBtn(source === 'pertree'), flex: 1 }} onClick={() => setSource('pertree')}>
            Per Tree
          </button>
        </div>
      </div>

      {/* Context bar */}
      <div style={{
        background: source === 'pertree' ? '#f0f7f0' : '#f5f5f5',
        border: '1px solid',
        borderColor: source === 'pertree' ? '#c3e0c3' : '#e8e8e8',
        borderRadius: 4,
        padding: '4px 8px',
        fontSize: 10,
        color: source === 'pertree' ? '#3A7C3A' : '#888',
        fontWeight: 600,
      }}>
        {source === 'forest'
          ? `Forest average — ${treeCount} trees`
          : `Tree ${selectedTree} only`}
        {simState.data.computed_ms !== undefined && (
          <span style={{ fontWeight: 400, opacity: 0.7 }}>
            {' '}· computed in {formatDuration(simState.data.computed_ms)}
          </span>
        )}
      </div>

      {/* Imbalance degeneracy warning */}
      {source === 'forest' && majorityLabel && (
        <div style={{
          background: '#fff7e6',
          border: '1px solid #ffd591',
          borderRadius: 4,
          padding: '5px 8px',
          fontSize: 10,
          color: '#ad6800',
          lineHeight: 1.4,
        }}>
          ⚠ Majority class <strong>{majorityLabel.trim()}</strong> is the most-overlapping
          neighbour of most other classes — rankings may reflect class imbalance
          rather than semantics.
        </div>
      )}

      {/* Metric selector */}
      <div>
        <div style={{ fontSize: 9, fontWeight: 700, color: '#aaa', fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 5 }}>
          Overlap Metric
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {METRICS.map(m => (
            <button
              key={m}
              title={`${METRIC_INFO[m].desc}

${METRIC_INFO[m].title}`}
              onClick={() => setMetric(m)}
              style={pillBtn(metric === m)}
            >
              {METRIC_INFO[m].label}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 9, color: '#bbb', marginTop: 4, lineHeight: 1.4 }}>
          {METRIC_INFO[metric].desc}
        </div>
      </div>

      {/* View toggle */}
      <div style={{ display: 'flex', gap: 5 }}>
        <button style={{ ...pillBtn(view === 'heatmap', '#5B8DB8'), flex: 1 }} onClick={() => setView('heatmap')}>
          Heatmap
        </button>
        <button style={{ ...pillBtn(view === 'dotmap', '#5B8DB8'), flex: 1 }} onClick={() => setView('dotmap')}>
          Dot Map (MDS)
        </button>
      </div>

      {/* Visualization — click to maximize, like the other expandable charts */}
      <div style={{ overflowX: 'auto' }}>
        <div
          onClick={() => view === 'heatmap' ? setHeatmapExpanded(true) : setDotmapExpanded(true)}
          title="Click to expand"
          style={{ cursor: 'pointer', display: 'inline-block' }}
        >
          {view === 'heatmap'
            ? <Heatmap matrix={matrix} labels={labels} path={pathStats} scaleEnds={scaleEnds} />
            : <DotMap  matrix={matrix} labels={labels} path={pathStats} />
          }
        </div>
      </div>

      {/* Color legend (heatmap only). The ramp is scaled to the values actually
          present, so the endpoint numbers — not fixed 0/1 — say what the colors mean. */}
      {view === 'heatmap' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 9, color: '#aaa' }}
          title={`Color scale spans this matrix's own range (${heatLo.toFixed(3)} – ${heatHi.toFixed(3)}), not a fixed 0–1, so the full ramp stays usable. Hover a cell for its exact distance.`}>
          <span style={{ color: '#3a6fad' }}>{heatLo.toFixed(2)}</span>
          <svg width={80} height={8}>
            <defs>
              <linearGradient id="simLegGrad" x1="0" x2="1">
                <stop offset="0%"   stopColor={distColor(0)} />
                <stop offset="50%"  stopColor={distColor(0.5)} />
                <stop offset="100%" stopColor={distColor(1)} />
              </linearGradient>
            </defs>
            <rect width={80} height={8} rx={2} fill="url(#simLegGrad)" />
          </svg>
          <span style={{ color: '#c0392b' }}>{heatHi.toFixed(2)}</span>
          <span style={{ opacity: 0.8 }}>{METRIC_INFO[metric].low} → {METRIC_INFO[metric].high}</span>
        </div>
      )}

      {/* Confusability framing — supervised forest measures confusability, not geometry */}
      {/* Maximized views — bigger canvas, values written in every heatmap cell,
          embedded color scale, and a Download-as-SVG button */}
      {heatmapExpanded && (
        <ExpandedChartModal
          title={`Heatmap — ${METRIC_INFO[metric].label} (${source === 'forest' ? `Forest avg, ${treeCount} trees` : `Tree ${selectedTree}`})`}
          onClose={() => setHeatmapExpanded(false)}
          svgRef={heatSvgRef}
        >
          <Heatmap matrix={matrix} labels={labels} path={pathStats} scaleEnds={scaleEnds} expanded svgRef={heatSvgRef} />
        </ExpandedChartModal>
      )}
      {dotmapExpanded && (
        <ExpandedChartModal
          title={`Dot Map (MDS) — ${METRIC_INFO[metric].label} (${source === 'forest' ? `Forest avg, ${treeCount} trees` : `Tree ${selectedTree}`})`}
          onClose={() => setDotmapExpanded(false)}
          svgRef={dotSvgRef}
        >
          <DotMap matrix={matrix} labels={labels} path={pathStats} expanded svgRef={dotSvgRef} />
        </ExpandedChartModal>
      )}

    </>
  )
  }
}
