import { useState } from 'react'
import { useAtomValue } from 'jotai'
import {
  activeTaskAtom, targetLabelAtom, activeGroupAtom, sliceValue, impurityAt,
  type TreeNodeData,
} from '../Forest/ForestAtoms'
import { classColor } from '../classColors'
import { effectiveClassColorsAtom } from '../Legend/LegendAtoms'
import { HEADING_FONT } from '../../../../theme'

export type NodeWidgetProps = {
  data: TreeNodeData
  /** Container-relative pixel position of the node centre */
  x: number
  y: number
  containerW: number
  containerH: number
  classLabels?: string[]
  /** Visual references passing through this node. undefined = feature dormant
      (no references in this iteration) → widget renders exactly as before. */
  refCount?: number
  /** Node has children or is a collapsed subtree — i.e. double-click does
      something here (a true leaf can neither collapse nor expand). */
  canCollapse?: boolean
  /** Pinned by a node click: the widget stays open and becomes interactive
      (only then can the class list be expanded/scrolled). */
  pinned?: boolean
  onClose?: () => void
}

const WIDGET_W = 164        // bar-chart layout
const WIDGET_W_TABLE = 212  // table layout — room for label + share + count
// Chart geometry
const BAR_H = 54           // maximum bar height
const PCT_PAD = 28         // space above bars reserved for rotated percentage label
const LABEL_FS = 8         // class-label font size
const LABEL_LINE_H = 8.5   // baseline step between wrapped label lines
const LABEL_TOP = 13       // first label baseline below the axis
const LABEL_BOTTOM = 3     // breathing room under the last line
const LABEL_MAX_LINES = 2  // beyond this the label is ellipsized instead
// SVG text does not wrap, so labels are split by hand. Average glyph advance
// for this sans-serif is ~0.55em — close enough to decide where to break, and
// erring narrow (breaking early) is harmless while erring wide would collide.
const LABEL_CHAR_W = LABEL_FS * 0.55

// From this many classes on, the vertical bars get too thin to read (and their
// labels collide) — switch to a table that names each class next to its share.
const TABLE_MIN_CLASSES = 6
const ROW_H = 15
// With 30+ classes only a handful ever reach a given node; the rest are folded
// into a single summary row so the widget can't outgrow the viewport. Clicking
// that row (on a pinned widget) reveals the rest inside a scroll area capped at
// the same height, so the widget never grows past its collapsed size.
const MAX_ROWS = 10
const EXPANDED_MAX_H = MAX_ROWS * ROW_H

function InfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}>
      <span style={{ color: '#999' }}>{label}</span>
      <span style={{ fontWeight: 600, color: '#333' }}>{value}</span>
    </div>
  )
}

// Tiny non-zero shares must not render as a flat "0.0" — that reads as absent.
function fmtPct(frac: number): string {
  const p = frac * 100
  if (p > 0 && p < 0.1) return '<0.1'
  return p.toFixed(1)
}

const NUM = { fontVariantNumeric: 'tabular-nums' } as const

// Opt-in for the few widget controls that must receive the mouse (the widget
// itself is pointer-events: none so clicks reach the tree underneath).
const INTERACTIVE = { pointerEvents: 'auto' } as const

/**
 * Greedy word-wrap for a bar's class label: "Natural meadow" under a 36px bar
 * becomes ["Natural", "meadow"] instead of running into its neighbours. A word
 * that cannot fit on its own (no space to break at) is ellipsized — the full
 * name stays available in the bar's tooltip.
 */
function wrapLabel(text: string, maxChars: number, maxLines = LABEL_MAX_LINES): string[] {
  const fit = Math.max(3, maxChars)
  const clip = (w: string) => (w.length > fit ? w.slice(0, fit - 1) + '…' : w)
  if (text.length <= fit) return [text]

  const lines: string[] = []
  let cur = ''
  for (const word of text.split(/\s+/)) {
    if (!cur) { cur = word; continue }
    if (cur.length + 1 + word.length <= fit) cur += ' ' + word
    else { lines.push(cur); cur = word }
  }
  if (cur) lines.push(cur)

  if (lines.length <= maxLines) return lines.map(clip)
  // Too many lines: keep the first ones, mark the last as truncated.
  const kept = lines.slice(0, maxLines)
  kept[maxLines - 1] = clip(kept[maxLines - 1] + '…')
  return kept
}

export default function NodeWidget({ data, x, y, containerW, containerH, classLabels = [], refCount, canCollapse = false, pinned = false, onClose }: NodeWidgetProps) {
  const [expanded, setExpanded] = useState(false)
  const effectiveColors = useAtomValue(effectiveClassColorsAtom)
  const task = useAtomValue(activeTaskAtom)
  const targetLabel = useAtomValue(targetLabelAtom)
  const group = useAtomValue(activeGroupAtom)
  const isReg = task === 'regression'
  const value = sliceValue(data.value, group)
  const impurity = impurityAt(data.impurity, group.index)
  const classTotal = value.reduce((a, b) => a + b, 0)
  const fractions = classTotal > 0
    ? value.map(v => v / classTotal)
    : value.map(() => 1 / Math.max(1, value.length))

  // Many classes → table instead of bars (see TABLE_MIN_CLASSES).
  const nClasses = value.length
  const useTable = !isReg && nClasses >= TABLE_MIN_CLASSES
  const widgetW = useTable ? WIDGET_W_TABLE : WIDGET_W

  // Table entries: only classes actually present, largest share first — with a
  // 30-class forest that's the handful this node is about.
  const present = useTable
    ? value
        .map((count, i) => ({ i, count, frac: classTotal > 0 ? count / classTotal : 0 }))
        .filter(e => e.count > 0)
        .sort((a, b) => b.count - a.count)
    : []
  const shown = present.slice(0, MAX_ROWS)
  const rest = present.slice(MAX_ROWS)
  const restCount = rest.reduce((s, e) => s + e.count, 0)
  const restFrac = rest.reduce((s, e) => s + e.frac, 0)

  // The full list is only reachable on a pinned widget (a hover widget can't be
  // clicked or scrolled), so expansion collapses again if the pin is dropped.
  const showAll = expanded && pinned
  const listRows = showAll ? present : shown
  const listH = showAll
    ? Math.min(present.length * ROW_H, EXPANDED_MAX_H)   // scrolls past the cap
    : shown.length * ROW_H
  const footerH = rest.length > 0 ? ROW_H : 0        // "+N more" / "show less"
  const tableH = present.length === 0
    ? 16                                              // "no samples" line
    : listH + footerH
  // Bar geometry + wrapped labels. The chart's height depends on how many
  // lines the longest label needs, so both are resolved before the widget
  // height estimate that drives the viewport clamping below.
  const barAreaW = WIDGET_W - 20
  const barW = barAreaW / Math.max(1, fractions.length)
  const labelLines = useTable
    ? []
    : fractions.map((_, i) =>
        wrapLabel(classLabels[i] ?? `C${i}`, Math.floor(barW / LABEL_CHAR_W)))
  const maxLabelLines = labelLines.reduce((m, l) => Math.max(m, l.length), 1)
  const chartH = PCT_PAD + BAR_H + LABEL_TOP
    + (maxLabelLines - 1) * LABEL_LINE_H + LABEL_BOTTOM
  const distH = useTable ? tableH : chartH

  // Header (28) + stats (48) + section label (18) + distribution + padding (16)
  // + reference hint (24) when the feature is active + gesture legend (20)
  // while pinned — the estimate drives the viewport clamping below.
  const hintH = refCount !== undefined ? 24 : 0
  const legendH = pinned ? 20 : 0
  const widgetH = (isReg ? 28 + 48 + 24 : 28 + 48 + 18 + distH + 16) + hintH + legendH

  const offX = x + 18 + widgetW > containerW ? -(widgetW + 14) : 18
  const rawTop = y - widgetH / 2
  const offY = rawTop < 4
    ? 4 - y
    : rawTop + widgetH > containerH - 4
      ? containerH - 4 - widgetH - y
      : -widgetH / 2

  return (
    <div style={{
      position: 'absolute',
      left: x + offX,
      top: y + offY,
      width: widgetW,
      background: pinned ? '#fff' : 'rgba(255,255,255,0.97)',
      border: '1px solid',
      borderColor: pinned ? '#3A8C3A' : '#ddd',
      borderRadius: 7,
      padding: '8px 10px',
      fontSize: 11,
      boxShadow: pinned ? '0 4px 16px rgba(0,0,0,0.2)' : '0 3px 10px rgba(0,0,0,0.13)',
      // The widget itself is click-through, pinned or not: it often sits over
      // neighbouring nodes/branches, and clicks (collapse, pin, inspect) must
      // reach the tree beneath it. Only its own controls opt back in below
      // (INTERACTIVE), and they stop propagation so using them doesn't count as
      // the outside click that dismisses the pin.
      pointerEvents: 'none',
      zIndex: 20,
    }}>
      {/* Header */}
      <div style={{
        fontFamily: HEADING_FONT, fontWeight: 700, color: '#333', marginBottom: 5,
        paddingBottom: 4, borderBottom: '1px solid #eee',
        display: 'flex', alignItems: 'center', gap: 6,
      }}>
        <span style={{ flex: 1 }}>Node {data.id}</span>
        {pinned && (
          <>
            <span style={{ fontSize: 9, fontWeight: 600, color: '#3A8C3A' }}>pinned</span>
            <span
              onClick={(e) => { e.stopPropagation(); onClose?.() }}
              title="Close (Esc, or click anywhere else)"
              style={{ ...INTERACTIVE, cursor: 'pointer', color: '#bbb', fontSize: 14, lineHeight: 1 }}
            >
              ×
            </span>
          </>
        )}
      </div>

      {/* Stats */}
      <div style={{ display: 'flex', flexDirection: 'column', marginBottom: isReg ? 0 : 8 }}>
        <InfoRow label="Depth" value={data.depth} />
        <InfoRow label="Samples" value={data.n_node_samples} />
        <InfoRow label={isReg ? 'Std dev' : 'Impurity'} value={impurity.toFixed(3)} />
        {isReg && (
          <InfoRow label={targetLabel ? `Mean (${targetLabel})` : 'Mean'} value={(value[0] ?? 0).toFixed(3)} />
        )}
      </div>

      {/* Class distribution — section label shared by both layouts */}
      {!isReg && (
        <div style={{
          fontSize: 10, color: '#bbb', marginBottom: 4,
          display: 'flex', justifyContent: 'space-between', gap: 6,
        }}>
          <span>class distribution</span>
          {useTable && present.length > 0 && (
            <span style={NUM}>{present.length}/{nClasses} present</span>
          )}
        </div>
      )}

      {/* Table layout — many classes: name + share, sorted by share */}
      {useTable && (
        present.length === 0 ? (
          <div style={{ fontSize: 10, color: '#ccc', fontStyle: 'italic', height: 16 }}>
            no samples at this node
          </div>
        ) : (
          <div>
            {/* The expanded list must take the wheel to scroll; collapsed it
                stays click-through like the rest of the widget. */}
            <div
              onClick={showAll ? (e) => e.stopPropagation() : undefined}
              style={{
                maxHeight: showAll ? EXPANDED_MAX_H : undefined,
                overflowY: showAll ? 'auto' : 'visible',
                ...(showAll ? INTERACTIVE : null),
              }}
            >
            {listRows.map(e => {
              const color = effectiveColors[e.i] ?? classColor(e.i, nClasses)
              return (
                <div
                  key={e.i}
                  style={{ position: 'relative', height: ROW_H, borderRadius: 2, overflow: 'hidden' }}
                >
                  {/* Faint share bar behind the row: keeps the visual scale of
                      the old chart while the numbers stay the primary read.
                      A separate layer (not a gradient with an alpha-suffixed
                      hex) — classColor emits hsl() above 12 classes. */}
                  
                  <div style={{
                    position: 'relative', height: '100%',
                    display: 'flex', alignItems: 'center', gap: 5,
                  }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: 2,
                      background: color, flexShrink: 0, marginLeft: 1,
                    }} />
                    <span style={{
                      flex: 1, minWidth: 0, color: '#555',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {classLabels[e.i] ?? `Class ${e.i}`}
                    </span>
                    <span style={{ ...NUM, fontWeight: 700, color: '#333' }}>
                      {fmtPct(e.frac)}%
                    </span>
                    <span style={{
                      ...NUM, color: '#bbb', fontSize: 9.5,
                      width: 36, textAlign: 'right', marginRight: 1,
                    }}>
                      {e.count}
                    </span>
                  </div>
                </div>
              )
            })}
            </div>

            {/* Footer: reveal the remaining classes (pinned widget only, since
                a hover widget can be neither clicked nor scrolled). */}
            {rest.length > 0 && (
              <div
                onClick={pinned ? (e) => { e.stopPropagation(); setExpanded(v => !v) } : undefined}
                style={{
                  ...(pinned ? INTERACTIVE : null),
                  display: 'flex', alignItems: 'center', gap: 5, height: ROW_H,
                  color: pinned ? '#3A8C3A' : '#aaa', fontSize: 10,
                  cursor: pinned ? 'pointer' : 'default',
                  fontWeight: pinned ? 600 : 400,
                  borderTop: '1px solid #f0f0f0',
                }}
              >
                <span style={{ flex: 1, minWidth: 0, marginLeft: 14 }}>
                  {showAll ? '▴ show less' : `▾ +${rest.length} more`}
                </span>
                {!showAll && <>
                  <span style={{ ...NUM, fontWeight: 600 }}>{fmtPct(restFrac)}%</span>
                  <span style={{ ...NUM, fontSize: 9.5, width: 36, textAlign: 'right', marginRight: 1 }}>
                    {restCount}
                  </span>
                </>}
              </div>
            )}
          </div>
        )
      )}

      {/* Vertical bar chart — few enough classes for readable bars */}
      {!isReg && !useTable && <>
      <svg width={barAreaW} height={chartH} style={{ display: 'block', overflow: 'visible' }}>
        {/* Baseline */}
        <line
          x1={0} y1={PCT_PAD + BAR_H}
          x2={barAreaW} y2={PCT_PAD + BAR_H}
          stroke="#e8e8e8" strokeWidth={1}
        />

        {fractions.map((f, i) => {
          const bh = Math.max(1, f * (BAR_H - 1))
          const barTop = PCT_PAD + (BAR_H - bh)
          const cx = i * barW + barW / 2
          const pct = `${(f * 100).toFixed(1)}`
          const color = effectiveColors[i] ?? classColor(i, fractions.length)

          return (
            <g key={i}>
              {/* Bar */}
              <rect
                x={i * barW + 1}
                y={barTop}
                width={Math.max(1, barW - 2)}
                height={bh}
                fill={color}
                opacity={0.82}
                rx={1}
              >
                <title>{classLabels[i] ?? `Class ${i}`}: {pct} ({value[i]} samples)</title>
              </rect>

              {/* Percentage — rotated –90° so it reads bottom-to-top above the bar.
                  textAnchor="start" puts the first character closest to the bar top. */}
              <text
                transform={`translate(${cx - 6}, ${barTop - 6}) rotate(0)`}
                textAnchor="start"
                dominantBaseline="middle"
                fontSize={9}
                fontWeight={600}
                fill="#444"
              >
                {pct}
              </text>

              {/* Color dot — directly under the axis, so it cannot collide
                  with a label that wrapped onto a second line. */}
              <rect
                x={cx - 3}
                y={PCT_PAD + BAR_H + 3}
                width={6} height={2}
                rx={1}
                fill={color}
                opacity={0.7}
              />

              {/* Class label, wrapped to the bar's width (one tspan per line) */}
              <text
                x={cx}
                y={PCT_PAD + BAR_H + LABEL_TOP}
                textAnchor="middle"
                fontSize={LABEL_FS}
                fill="#888"
              >
                {labelLines[i].map((line, li) => (
                  <tspan key={li} x={cx} dy={li === 0 ? 0 : LABEL_LINE_H}>{line}</tspan>
                ))}
              </text>
            </g>
          )
        })}
      </svg>
      </>}

      {/* Visual-reference hint (only when the iteration has references) */}
      {refCount !== undefined && (
        <div style={{
          borderTop: '1px solid #eee', marginTop: 6, paddingTop: 5,
          fontSize: 10, color: '#888',
          display: 'flex', justifyContent: 'space-between', gap: 8,
        }}>
          <span style={{ fontWeight: 600 }}>
            {refCount} reference{refCount !== 1 ? 's' : ''}
          </span>
          {refCount > 0 && (
            <span style={{ color: '#bbb' }}>Alt+click to inspect</span>
          )}
        </div>
      )}

      {/* Gesture legend — shown while pinned, where there's room for it and the
          user has stopped to read. Keeps double-click-to-collapse discoverable
          without a tooltip hunt. */}
      {pinned && (
        <div style={{
          borderTop: '1px solid #eee', marginTop: 6, paddingTop: 5,
          fontSize: 9, color: '#bbb', lineHeight: 1.5,
        }}>
          {canCollapse && 'Double-click node: collapse / expand · '}Esc: close
        </div>
      )}
    </div>
  )
}
