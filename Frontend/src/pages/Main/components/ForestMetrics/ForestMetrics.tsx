import { useState, type CSSProperties, type ReactNode } from 'react'
import { useAtomValue } from 'jotai'
import { activeGroupAtom, activeTaskAtom } from '../Forest/ForestAtoms'
import { selectedTreeIdAtom } from '../ForestView/ForestViewAtoms'
import ExpandedBarChart from '../ExpandedBarChart/ExpandedBarChart'
import {
  activeForestStatAtom, activeGroupMetricStatsAtom, forestSizeStatsAtom,
  treeMetricStatsAtom, fmtMetric, type MetricStats,
} from './ForestMetricsAtoms'
import { HEADING_FONT } from '../../../../theme'

const PANEL_W = 220

// Accents: structural facts of the forest (trees, height, node counts) vs the
// user-defined metrics, so a glance tells the two kinds apart.
const STRUCTURE_ACCENT = '#8B6040'
const METRIC_ACCENT = '#3A8C3A'
const GROUP_METRIC_ACCENT = '#3A7C3A'

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: '#aaa',
        fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1.2,
        paddingBottom: 5, borderBottom: '1px solid #eee', marginBottom: 6,
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

// `accent`: a string tints the LABEL, the way MetricStatRow tints its metric
// name, so a row reads as part of that group. The boolean form is the original
// behaviour and still highlights the value.
// `style`: free-form overrides for the label text (size, weight, color, …) —
// merged last, so it wins over `accent`.
function Row({ label, value, accent, style }:
  { label: string; value: string | number; accent?: boolean | string; style?: CSSProperties }) {
  const labelColor = typeof accent === 'string' ? accent : '#999'
  const valueColor = accent === true ? METRIC_ACCENT : '#333'
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
      <span style={{ fontSize: 11, color: labelColor, ...style }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: valueColor }}>
        {value}
      </span>
    </div>
  )
}

// ── Metric display ───────────────────────────────────────────────────────────

const BAR_W = 44
const BAR_H = 6

// Horizontal bar on a shared domain (same idea as the tree-balance bar).
// `from` lets it draw a span rather than a fill, so the mean bar and the range
// bar are directly comparable: the mean's end sits inside the range's span.
function StatBar({ from = 0, to, lo, hi, color, opacity = 0.75 }: {
  from?: number; to: number; lo: number; hi: number; color: string; opacity?: number
}) {
  // Domain [lo, hi] rather than [0, max]: a metric that can go negative still
  // lands inside the track, and its bar then grows leftwards from zero.
  const span = hi - lo || 1
  const at = (v: number) => Math.max(0, Math.min(BAR_W, ((v - lo) / span) * BAR_W))
  const a = at(from), b = at(to)
  const w = Math.max(2, Math.abs(b - a))            // keep a zero-width span visible…
  const x = Math.min(Math.min(a, b), BAR_W - w)     // …without pushing it off the track
  return (
    <svg width={BAR_W} height={BAR_H} style={{ display: 'inline-block', verticalAlign: 'middle', marginLeft: 6 }}>
      <rect x={0} y={0} width={BAR_W} height={BAR_H} rx={3} fill="#eee" />
      <rect x={x} y={0} width={w} height={BAR_H} rx={3} fill={color} opacity={opacity} />
    </svg>
  )
}

function StatLine({ label, value, bar }: { label: string; value: string; bar?: ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1px 0' }}>
      <span style={{ fontSize: 11, color: '#999' }}>{label}</span>
      <span style={{ display: 'flex', alignItems: 'center' }}>
        {bar}
        <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 6, textAlign: 'right' }}>
          {value}
        </span>
      </span>
    </div>
  )
}

// One metric summarised over the trees: mean, σ and range each as their own
// entry. Used for the user-defined metrics and the structural ones alike, so
// every forest aggregate reads the same way.
function MetricStatRow({ name, m, accent = METRIC_ACCENT, fmt = fmtMetric }:
  { name: string; m: MetricStats; accent?: string; fmt?: (v: number) => string }) {
  // Bars share one domain so the mean's end reads inside the range's span.
  // Zero is kept as the baseline whenever the metric doesn't go below it.
  const lo = Math.min(0, m.min)
  const hi = m.max > lo ? m.max : lo + 1
  return (
    <div
      title={`${name} of each tree on its own, over ${m.n} tree${m.n !== 1 ? 's' : ''}: `
           + `mean ${fmt(m.mean)}, σ ${fmt(m.std)}, range ${fmt(m.min)} – ${fmt(m.max)}. `
           + `Bars are scaled to the largest value (${fmt(m.max)}). `
           + `This is NOT the forest's ${name}.`}
      style={{ padding: '3px 0' }}
    >
      <div style={{ fontSize: 10, fontWeight: 700, color: accent, marginBottom: 1 }}>{name}</div>
      <StatLine label="mean" value={fmt(m.mean)}
        bar={<StatBar to={m.mean} lo={lo} hi={hi} color={accent} />} />
      <StatLine label="σ" value={fmt(m.std)} />
      <StatLine label="range" value={`${fmt(m.min)} – ${fmt(m.max)}`}
        bar={<StatBar from={m.min} to={m.max} lo={lo} hi={hi} color={accent} opacity={0.4} />} />
    </div>
  )
}

// Vertical bar chart: trees sorted by node count, selected tree highlighted green.
function SizeChart({
  entries,
  selectedId,
}: {
  // Only the node count is charted — no need for the full TreeMeta here.
  entries: [string, { n_nodes: number }][]
  selectedId: string | null
}) {
  const sorted = [...entries].sort((a, b) => a[1].n_nodes - b[1].n_nodes)
  const maxNodes = sorted[sorted.length - 1]?.[1].n_nodes || 1
  const W = PANEL_W - 32
  const H = 52
  const barW = W / sorted.length

  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      {sorted.map(([id, meta], i) => {
        const bh = Math.max(1, (meta.n_nodes / maxNodes) * (H - 2))
        const isSelected = id === selectedId
        return (
          <rect
            key={id}
            x={i * barW}
            y={H - bh}
            width={Math.max(1, barW - 1)}
            height={bh}
            rx={1}
            fill={isSelected ? METRIC_ACCENT : STRUCTURE_ACCENT}
            opacity={isSelected ? 1 : 0.65}
          />
        )
      })}
    </svg>
  )
}

export default function ForestMetrics() {
  const selectedId = useAtomValue(selectedTreeIdAtom)
  const [sizeExpanded, setSizeExpanded] = useState(false)
  // Every aggregate on this panel comes from ForestMetricsAtoms — the Tree tab
  // reports single-tree values only, so no figure appears in both places.
  const stats            = useAtomValue(forestSizeStatsAtom)
  const forestStat       = useAtomValue(activeForestStatAtom)
  const treeMetricStats  = useAtomValue(treeMetricStatsAtom)
  const groupMetricStats = useAtomValue(activeGroupMetricStatsAtom)
  const group            = useAtomValue(activeGroupAtom)
  const isReg            = useAtomValue(activeTaskAtom) === 'regression'

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        padding: '12px 16px',
      }}
    >

      {stats && (
        <>
          {/* Built-in aggregates over the trees: structure, then the backend's
              per-group figures for the active label group / target. Each row is
              independent — a regression forest reports mean AND leaf σ. */}
          <Section title="Metrics">
            <Row label="Trees (T)" value={stats.count}
              style={{ fontSize: 10, fontWeight: 700, color: STRUCTURE_ACCENT, marginBottom: 1 }} />
            <MetricStatRow name="Height (D)" m={stats.height} accent={STRUCTURE_ACCENT}
              fmt={(v) => (Number.isInteger(v) ? String(v) : v.toFixed(2))} />
            <MetricStatRow name="Nodes (S)" m={stats.nodes} accent={STRUCTURE_ACCENT}
              fmt={(v) => (Number.isInteger(v) ? String(v) : v.toFixed(1))} />
            {forestStat?.purity !== undefined && (
              <Row label="Purity" value={`${(forestStat.purity * 100).toFixed(1)}%`} style={{ fontSize: 10, fontWeight: 700, color: STRUCTURE_ACCENT, marginBottom: 1 }} />
            )}
            {isReg && forestStat?.mean !== undefined && (
              <Row label="Root mean" value={forestStat.mean.toFixed(3)} style={{ fontSize: 10, fontWeight: 700, color: STRUCTURE_ACCENT, marginBottom: 1 }} />
            )}
            {forestStat?.std !== undefined && (
              <Row label="Leaf σ" value={forestStat.std.toFixed(3)} style={{ fontSize: 10, fontWeight: 700, color: STRUCTURE_ACCENT, marginBottom: 1 }} />
            )}
          </Section>
          {/* Metrics the user attached to each tree — named explicitly so they
              are never mistaken for the built-in structural ones above. */}
          {Object.keys(treeMetricStats).length > 0 && (
            <Section title="User-defined metrics">
              <div style={{ fontSize: 9, color: '#bbb', lineHeight: 1.35, marginBottom: 5, fontStyle: 'italic' }}>
                Mean over the {stats.count} individual trees — <strong>not</strong> the
                ensemble score.
              </div>
              {Object.entries(treeMetricStats).map(([key, m]) => (
                <MetricStatRow key={key} name={key} m={m} />
              ))}
            </Section>
          )}

          {/* The same, for metrics scoped to the active label group / target */}
          {Object.keys(groupMetricStats).length > 0 && (
            <Section title={group.name
              ? `User-defined metrics · ${group.name}`
              : 'User-defined metrics · group'}>
              {Object.entries(groupMetricStats).map(([key, m]) => (
                <MetricStatRow key={key} name={key} m={m} accent={GROUP_METRIC_ACCENT} />
              ))}
            </Section>
          )}

          {/* Size distribution chart */}
          <Section title="Size distribution">
            <div
              onClick={() => setSizeExpanded(true)}
              title="Click to expand"
              style={{ cursor: 'pointer', display: 'inline-block' }}
            >
              <SizeChart entries={stats.entries} selectedId={selectedId} />
            </div>
            <div style={{ fontSize: 10, color: '#ccc', marginTop: 3, textAlign: 'right' }}>
              ↑ nodes · sorted · click to expand
            </div>
          </Section>
        </>
      )}

      {sizeExpanded && stats && (
        <ExpandedBarChart
          title="Node Count Distribution"
          xLabel="Trees (sorted by node count)"
          yLabel="Nodes"
          bars={[...stats.entries]
            .sort((a, b) => a[1].n_nodes - b[1].n_nodes)
            .map(([id, meta]) => ({
              label: id,
              value: meta.n_nodes,
              color: id === selectedId ? METRIC_ACCENT : STRUCTURE_ACCENT,
            }))
          }
          onClose={() => setSizeExpanded(false)}
        />
      )}

    </div>
  )
}
