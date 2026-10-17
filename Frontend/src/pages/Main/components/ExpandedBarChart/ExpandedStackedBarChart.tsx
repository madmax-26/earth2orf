import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { downloadSvgElement, sanitizeFilename } from '../svgExport'
import { HEADING_FONT } from '../../../../theme'

export interface StackedBarSpec {
  label: string
  internal: number
  leaf: number
}

export interface ExpandedStackedBarChartProps {
  title: string
  bars: StackedBarSpec[]
  xLabel?: string
  yLabel?: string
  onClose: () => void
}

// ── Axis helpers (same strategy as ExpandedBarChart) ─────────────────────────

function niceMax(v: number): number {
  if (v <= 0) return 1
  const mag = Math.pow(10, Math.floor(Math.log10(v)))
  const n = v / mag
  if (n <= 1) return mag
  if (n <= 2) return 2 * mag
  if (n <= 5) return 5 * mag
  return 10 * mag
}

function yTicks(ceiling: number, steps = 5): number[] {
  const step = ceiling / steps
  return Array.from({ length: steps + 1 }, (_, i) => Math.round(step * i))
}

// ── Layout constants ──────────────────────────────────────────────────────────

const ML = 60
const MR = 20
const MT = 32
const MB = 56
const CH = 360
const SVG_W = 760
const SVG_H = MT + CH + MB

const C_INTERNAL = '#5B8DB8'
const C_LEAF     = '#E08050'

export default function ExpandedStackedBarChart({
  title, bars, xLabel, yLabel, onClose,
}: ExpandedStackedBarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const maxVal = Math.max(...bars.map(b => b.internal + b.leaf), 1)
  const ceiling = niceMax(maxVal)
  const ticks   = yTicks(ceiling)
  const cw      = SVG_W - ML - MR
  const barW    = cw / Math.max(bars.length, 1)
  const rotateLabels = bars.length > 18

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
          width: '80vw', minWidth: 480, maxWidth: '95vw',
          maxHeight: '88vh', minHeight: 320,
          overflow: 'auto',
          resize: 'both',
          boxShadow: '0 8px 36px rgba(0,0,0,0.18)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
          <span style={{ fontFamily: HEADING_FONT, fontWeight: 700, fontSize: 14, color: '#333', flex: 1 }}>{title}</span>
          {/* Legend */}
          <span style={{ display: 'flex', alignItems: 'center', gap: 12, marginRight: 16, fontSize: 11, color: '#666' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <svg width={10} height={10}><rect width={10} height={10} rx={2} fill={C_INTERNAL} opacity={0.8} /></svg>
              Internal
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <svg width={10} height={10}><rect width={10} height={10} rx={2} fill={C_LEAF} opacity={0.8} /></svg>
              Leaf
            </span>
          </span>
          <button
            onClick={() => downloadSvgElement(svgRef.current, `${sanitizeFilename(title)}.svg`)}
            title="Download as SVG"
            style={{
              fontSize: 11, fontWeight: 600, padding: '4px 10px', marginRight: 10,
              border: '1px solid #ccc', borderRadius: 4,
              background: '#fff', color: '#555', cursor: 'pointer',
            }}
          >
            ⬇ Download
          </button>
          <button
            onClick={onClose}
            style={{ border: 'none', background: 'none', fontSize: 20, lineHeight: 1, padding: '0 2px', cursor: 'pointer', color: '#aaa' }}
          >
            ×
          </button>
        </div>

        {/* Chart */}
        <svg ref={svgRef} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ display: 'block', width: '100%', maxWidth: SVG_W }}>
          <g transform={`translate(${ML},${MT})`}>

            {/* Y-axis label */}
            {yLabel && (
              <text
                transform={`translate(${-46},${CH / 2}) rotate(-90)`}
                textAnchor="middle" fontSize={10} fill="#bbb"
              >{yLabel}</text>
            )}

            {/* Gridlines + Y tick labels */}
            {ticks.map(t => {
              const y = CH - (t / ceiling) * CH
              return (
                <g key={t}>
                  <line x1={0} y1={y} x2={cw} y2={y}
                    stroke={t === 0 ? '#ccc' : '#f0f0f0'} strokeWidth={1} />
                  <text x={-8} y={y} textAnchor="end" dominantBaseline="middle"
                    fontSize={10} fill="#aaa">{t}</text>
                </g>
              )
            })}

            {/* Bars */}
            {bars.map((bar, i) => {
              const total      = bar.internal + bar.leaf
              const totalH     = Math.max(total > 0 ? 2 : 0, (total / ceiling) * CH)
              const internalH  = total > 0 ? (bar.internal / total) * totalH : 0
              const leafH      = totalH - internalH
              const bx         = i * barW + 1
              const bw         = Math.max(1, barW - 2)

              return (
                <g key={i}>
                  {/* Internal segment (bottom) */}
                  {internalH > 0 && (
                    <rect x={bx} y={CH - internalH} width={bw} height={internalH}
                      fill={C_INTERNAL} rx={2} opacity={0.8}>
                      <title>{bar.label}: {bar.internal} internal</title>
                    </rect>
                  )}
                  {/* Leaf segment (top) */}
                  {leafH > 0 && (
                    <rect x={bx} y={CH - totalH} width={bw} height={leafH}
                      fill={C_LEAF} rx={2} opacity={0.8}>
                      <title>{bar.label}: {bar.leaf} leaf</title>
                    </rect>
                  )}

                  {/* Value labels: internal inside bar (if tall enough), leaf above bar */}
                  {internalH >= 14 && (
                    <text x={bx + bw / 2} y={CH - internalH / 2}
                      textAnchor="middle" dominantBaseline="middle"
                      fontSize={9} fill="#fff" fontWeight="600" style={{ pointerEvents: 'none' }}>
                      {bar.internal}
                    </text>
                  )}
                  {leafH >= 14 && (
                    <text x={bx + bw / 2} y={CH - totalH + leafH / 2}
                      textAnchor="middle" dominantBaseline="middle"
                      fontSize={9} fill="#fff" fontWeight="600" style={{ pointerEvents: 'none' }}>
                      {bar.leaf}
                    </text>
                  )}
                  {/* Total above bar */}
                  {total > 0 && (
                    <text x={bx + bw / 2} y={CH - totalH - 4}
                      textAnchor="middle" fontSize={9} fill="#555" fontWeight="600">
                      {total}
                    </text>
                  )}

                  {/* X-axis label */}
                  {rotateLabels ? (
                    <text
                      transform={`translate(${bx + bw / 2},${CH + 8}) rotate(45)`}
                      textAnchor="start" fontSize={9} fill="#888">
                      {bar.label}
                    </text>
                  ) : (
                    <text x={bx + bw / 2} y={CH + 14}
                      textAnchor="middle" fontSize={10} fill="#888">
                      {bar.label}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        </svg>

        {xLabel && (
          <div style={{ textAlign: 'center', fontSize: 10, color: '#bbb', marginTop: 2 }}>{xLabel}</div>
        )}

        <div style={{ fontSize: 10, color: '#ddd', marginTop: 10, textAlign: 'right' }}>
          click outside or press Esc to close
        </div>
      </div>
    </div>
  )

  return createPortal(modal, document.body)
}
