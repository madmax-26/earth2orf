import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { downloadSvgElement, sanitizeFilename } from '../svgExport'
import { HEADING_FONT } from '../../../../theme'

export interface BarSpec {
  label: string
  value: number
  color: string
}

export interface ExpandedBarChartProps {
  title: string
  bars: BarSpec[]
  xLabel?: string
  yLabel?: string
  onClose: () => void
}

// ── Axis helpers ──────────────────────────────────────────────────────────────

/** Round up to the next "nice" number for a clean axis ceiling. */
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

const ML = 60   // left margin (Y-axis labels)
const MR = 20   // right margin
const MT = 32   // top margin (value labels above tallest bar)
const MB = 56   // bottom margin (X-axis labels)
const CH = 360  // chart area height
const SVG_W = 760
const SVG_H = MT + CH + MB

export default function ExpandedBarChart({
  title, bars, xLabel, yLabel, onClose,
}: ExpandedBarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const maxVal = Math.max(...bars.map(b => b.value), 1)
  const ceiling = niceMax(maxVal)
  const ticks = yTicks(ceiling)
  const cw = SVG_W - ML - MR          // chart area width
  const barW = cw / Math.max(bars.length, 1)
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

        {/* Chart */}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          style={{ display: 'block', width: '100%', maxWidth: SVG_W }}
        >
          <g transform={`translate(${ML},${MT})`}>

            {/* Y-axis label */}
            {yLabel && (
              <text
                transform={`translate(${-40},${CH / 2}) rotate(-90)`}
                textAnchor="middle" fontSize={10} fill="#bbb"
              >
                {yLabel}
              </text>
            )}

            {/* Gridlines + Y tick labels */}
            {ticks.map(t => {
              const y = CH - (t / ceiling) * CH
              return (
                <g key={t}>
                  <line
                    x1={0} y1={y} x2={cw} y2={y}
                    stroke={t === 0 ? '#ccc' : '#f0f0f0'} strokeWidth={1}
                  />
                  <text
                    x={-8} y={y}
                    textAnchor="end" dominantBaseline="middle"
                    fontSize={10} fill="#aaa"
                  >
                    {t}
                  </text>
                </g>
              )
            })}

            {/* Bars */}
            {bars.map((bar, i) => {
              const bh = Math.max(1, (bar.value / ceiling) * CH)
              const bx = i * barW
              const by = CH - bh

              return (
                <g key={i}>
                  {/* Bar */}
                  <rect
                    x={bx + 1} y={by}
                    width={Math.max(1, barW - 2)} height={bh}
                    fill={bar.color} rx={2} opacity={0.85}
                  >
                    <title>{bar.label}: {bar.value}</title>
                  </rect>

                  {/* Value label above bar */}
                  <text
                    x={bx + barW / 2} y={by - 4}
                    textAnchor="middle" fontSize={9} fill="#555" fontWeight="600"
                  >
                    {bar.value}
                  </text>

                  {/* X-axis label */}
                  {rotateLabels ? (
                    <text
                      transform={`translate(${bx + barW / 2},${CH + 8}) rotate(45)`}
                      textAnchor="start" fontSize={9} fill="#888"
                    >
                      {bar.label}
                    </text>
                  ) : (
                    <text
                      x={bx + barW / 2} y={CH + 14}
                      textAnchor="middle" fontSize={10} fill="#888"
                    >
                      {bar.label}
                    </text>
                  )}
                </g>
              )
            })}

          </g>
        </svg>

        {/* X-axis label */}
        {xLabel && (
          <div style={{ textAlign: 'center', fontSize: 10, color: '#bbb', marginTop: 2 }}>
            {xLabel}
          </div>
        )}

        <div style={{ fontSize: 10, color: '#ddd', marginTop: 10, textAlign: 'right' }}>
          click outside or press Esc to close
        </div>
      </div>
    </div>
  )

  return createPortal(modal, document.body)
}
