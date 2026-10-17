import { useMemo, useState, type ReactNode } from 'react'
import { useAtomValue } from 'jotai'
import {
  classLabelsAtom, loadableTreeDataAtomFamily, activeTaskAtom, targetLabelAtom,
  activeGroupAtom, sliceValue, impurityAt,
  type TreeNodeData, type GroupSpec,
} from '../Forest/ForestAtoms'
import { fmtMetric } from '../ForestMetrics/ForestMetricsAtoms'
import { treeOwnMetricsAtomFamily } from './TreeMetricsAtoms'
import { classColor } from '../classColors'
import { effectiveClassColorsAtom } from '../Legend/LegendAtoms'
import ExpandedBarChart from '../ExpandedBarChart/ExpandedBarChart'
import ExpandedStackedBarChart from '../ExpandedBarChart/ExpandedStackedBarChart'
import { HEADING_FONT } from '../../../../theme'

function mean(arr: number[]) {
  return arr.reduce((a, b) => a + b, 0) / arr.length
}
function std(arr: number[], avg: number) {
  return Math.sqrt(arr.reduce((s, v) => s + (v - avg) ** 2, 0) / arr.length)
}

// In DFS preorder, node[i] is a leaf iff it is last OR node[i+1].depth <= node[i].depth
function partitionNodes(nodes: TreeNodeData[]) {
  const leaves: TreeNodeData[] = []
  const internals: TreeNodeData[] = []
  for (let i = 0; i < nodes.length; i++) {
    const isLeaf = i === nodes.length - 1 || nodes[i + 1].depth <= nodes[i].depth
    if (isLeaf) leaves.push(nodes[i])
    else internals.push(nodes[i])
  }
  return { leaves, internals }
}

function computeMetrics(nodes: TreeNodeData[], group: GroupSpec) {
  if (!nodes.length) return null
  const root = nodes[0]
  const total = root.n_node_samples
  // Active label group / regression target: impurity and value are per-group
  // in multi-output forests; imp()/val() reduce them to the selected group.
  const imp = (n: TreeNodeData) => impurityAt(n.impurity, group.index)
  const val = (n: TreeNodeData) => sliceValue(n.value, group)

  const { leaves, internals } = partitionNodes(nodes)

  // ── Depth ────────────────────────────────────────────
  const maxDepth = Math.max(...nodes.map(n => n.depth))
  const leafDepths = leaves.map(l => l.depth)
  const meanLeafDepth = mean(leafDepths)
  const stdLeafDepth = std(leafDepths, meanLeafDepth)
  const minLeafDepth = Math.min(...leafDepths)

  // Nodes per depth level (for histograms)
  const byDepth = new Map<number, number>()
  const byDepthLeaf = new Map<number, number>()
  for (let i = 0; i < nodes.length; i++) {
    const d = nodes[i].depth
    byDepth.set(d, (byDepth.get(d) ?? 0) + 1)
    const isLeaf = i === nodes.length - 1 || nodes[i + 1].depth <= nodes[i].depth
    if (isLeaf) byDepthLeaf.set(d, (byDepthLeaf.get(d) ?? 0) + 1)
  }
  const depthCounts = Array.from({ length: maxDepth + 1 }, (_, d) => byDepth.get(d) ?? 0)
  const depthBreakdown = Array.from({ length: maxDepth + 1 }, (_, d) => {
    const t = byDepth.get(d) ?? 0
    const l = byDepthLeaf.get(d) ?? 0
    return { internal: t - l, leaf: l }
  })
  const maxWidth = Math.max(...depthCounts)

  // Balance: 1 − coefficient of variation of leaf depths (1 = perfectly balanced)
  const balance = meanLeafDepth > 0 ? Math.max(0, 1 - stdLeafDepth / meanLeafDepth) : 1

  // ── Samples (S from paper) ────────────────────────────
  const leafSamples = leaves.map(l => l.n_node_samples)
  const avgLeafSamples = mean(leafSamples)
  const stdLeafSamples = std(leafSamples, avgLeafSamples)

  // Concentration: fraction of all samples held by the top-25% leaves (by sample count)
  const sorted25 = [...leafSamples].sort((a, b) => b - a)
  const top25n = Math.max(1, Math.ceil(leaves.length * 0.25))
  const concentration = sorted25.slice(0, top25n).reduce((a, b) => a + b, 0) / total

  // ── Purity ────────────────────────────────────────────
  // Weighted Gini: Σ(n_samples × impurity) / total — analogous to weighted impurity before a split
  const weightedImpurity = nodes.reduce((s, n) => s + n.n_node_samples * imp(n), 0) / total
  const leafImpurities = leaves.map(imp)
  const meanLeafImpurity = mean(leafImpurities)
  const pureLeafFrac = leaves.filter(l => imp(l) < 0.01).length / leaves.length

  // ── Classes ───────────────────────────────────────────
  const rootValue = val(root)
  const classTotal = rootValue.reduce((a, b) => a + b, 0)
  const classFractions = classTotal > 0
    ? rootValue.map(v => v / classTotal)
    : rootValue.map(() => 1 / rootValue.length)
  const dominantClass = classFractions.indexOf(Math.max(...classFractions))

  // ── Regression: Target + Deviation ────────────────────
  // value[offset] is the node mean of the active target; imp() its std.
  const leafMeans = leaves.map(l => val(l)[0] ?? 0)
  const rootMean = rootValue[0] ?? 0
  const leafMeanMin = Math.min(...leafMeans)
  const leafMeanMax = Math.max(...leafMeans)
  const leafMeanAvg = mean(leafMeans)
  const leafMeanStd = std(leafMeans, leafMeanAvg)
  const rootStd = imp(root)
  const maxLeafStd = Math.max(...leafImpurities, 1e-9)
  const lowVarThresh = rootStd > 0 ? 0.1 * rootStd : 0
  const lowVarLeafFrac = leaves.filter(l => imp(l) <= lowVarThresh).length / leaves.length
  const NBINS = 12
  const meanSpan = leafMeanMax > leafMeanMin ? leafMeanMax - leafMeanMin : 1
  const meanHist = new Array(NBINS).fill(0)
  for (const m of leafMeans) {
    const b = Math.min(NBINS - 1, Math.max(0, Math.floor((m - leafMeanMin) / meanSpan * NBINS)))
    meanHist[b] += 1
  }

  return {
    nodeCount: nodes.length,
    leafCount: leaves.length,
    internalCount: internals.length,
    maxDepth,
    maxWidth,
    balance,
    depthCounts,
    depthBreakdown,
    meanLeafDepth,
    stdLeafDepth,
    minLeafDepth,
    total,
    avgLeafSamples,
    stdLeafSamples,
    minLeafSamples: Math.min(...leafSamples),
    maxLeafSamples: Math.max(...leafSamples),
    concentration,
    rootImpurity: imp(root),
    weightedImpurity,
    meanLeafImpurity,
    pureLeafFrac,
    classFractions,
    dominantClass,
    classCount: rootValue.length,
    // Regression fields (only meaningful / rendered when task === 'regression')
    rootMean,
    leafMeanMin,
    leafMeanMax,
    leafMeanAvg,
    leafMeanStd,
    rootStd,
    maxLeafStd,
    lowVarLeafFrac,
    meanHist,
  }
}

// ── Small reusable visual components ─────────────────────────────────────────

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
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

function Row({ label, value, accent, title }: { label: string; value: string | number; accent?: boolean; title?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }} title={title}>
      <span style={{ fontSize: 11, color: '#999' }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: accent ? '#3A8C3A' : '#333' }}>
        {value}
      </span>
    </div>
  )
}

// Horizontal proportion bar (0–1)
function PropBar({ value, color = '#3A8C3A', bg = '#eee' }: { value: number; color?: string; bg?: string }) {
  return (
    <svg width={64} height={6} style={{ display: 'inline-block', verticalAlign: 'middle', marginLeft: 6 }}>
      <rect x={0} y={0} width={64} height={6} rx={3} fill={bg} />
      <rect x={0} y={0} width={Math.max(2, value * 64)} height={6} rx={3} fill={color} opacity={0.75} />
    </svg>
  )
}

// Stacked histogram: internal nodes (bottom) vs leaf nodes (top) per depth level
function DepthStackedHistogram({ breakdown }: { breakdown: { internal: number; leaf: number }[] }) {
  const W = 148
  const H = 52
  const max = Math.max(...breakdown.map(b => b.internal + b.leaf), 1)
  const barW = W / Math.max(breakdown.length, 1)

  return (
    <svg width={W} height={H + 18} style={{ display: 'block', marginTop: 4 }}>
      {breakdown.map(({ internal, leaf }, i) => {
        const total = internal + leaf
        if (total === 0) return null
        const totalH = Math.max(2, (total / max) * (H - 2))
        const internalH = (internal / total) * totalH
        const leafH = totalH - internalH
        const bx = i * barW + 0.5
        const bw = Math.max(1, barW - 1)
        return (
          <g key={i}>
            <rect x={bx} y={H - internalH} width={bw} height={internalH}
              rx={1} fill="#5B8DB8" opacity={0.75}>
              <title>depth {i}: {internal} internal</title>
            </rect>
            {leafH > 0 && (
              <rect x={bx} y={H - totalH} width={bw} height={leafH}
                rx={1} fill="#E08050" opacity={0.75}>
                <title>depth {i}: {leaf} leaf</title>
              </rect>
            )}
          </g>
        )
      })}
      {/* Legend */}
      <g transform={`translate(0,${H + 5})`}>
        <rect x={0} y={0} width={7} height={7} rx={1} fill="#5B8DB8" opacity={0.75} />
        <text x={10} y={7} fontSize={7} fill="#888">internal</text>
        <rect x={52} y={0} width={7} height={7} rx={1} fill="#E08050" opacity={0.75} />
        <text x={62} y={7} fontSize={7} fill="#888">leaf</text>
      </g>
    </svg>
  )
}

// Horizontal stacked bar for class distribution
function ClassBar({ fractions, labels, customColors }: {
  fractions: number[]
  labels: string[]
  customColors: Record<number, string>
}) {
  const W = 148
  const H = 10
  const offsets = fractions.reduce<number[]>((acc, f) => {
    acc.push((acc[acc.length - 1] ?? 0) + Math.max(0, f * W))
    return acc
  }, [])
  return (
    <svg width={W} height={H} style={{ display: 'block', marginTop: 4, borderRadius: 3, overflow: 'hidden' }}>
      {fractions.map((f, i) => (
        <rect key={i} x={i === 0 ? 0 : offsets[i - 1]} y={0}
          width={Math.max(0, f * W)} height={H}
          fill={customColors[i] ?? classColor(i, fractions.length)}
          opacity={0.8}
        >
          <title>{labels[i] ?? `Class ${i}`}: {(f * 100).toFixed(1)}%</title>
        </rect>
      ))}
    </svg>
  )
}

// Simple bar histogram of leaf mean predictions (regression Target section)
function MeanHistogram({ hist, min, max }: { hist: number[]; min: number; max: number }) {
  const W = 148
  const H = 44
  const maxC = Math.max(...hist, 1)
  const barW = W / Math.max(hist.length, 1)
  return (
    <svg width={W} height={H + 14} style={{ display: 'block', marginTop: 4 }}>
      {hist.map((c, i) => {
        const h = c === 0 ? 0 : Math.max(2, (c / maxC) * (H - 2))
        return (
          <rect key={i} x={i * barW + 0.5} y={H - h} width={Math.max(1, barW - 1)} height={h}
            rx={1} fill="#5B8DB8" opacity={0.78}>
            <title>{c} leaves</title>
          </rect>
        )
      })}
      <text x={0} y={H + 11} fontSize={8} fill="#aaa">{min.toFixed(2)}</text>
      <text x={W} y={H + 11} fontSize={8} fill="#aaa" textAnchor="end">{max.toFixed(2)}</text>
    </svg>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export type TreeMetricsProps = { treeId: string }

export default function TreeMetrics({ treeId }: TreeMetricsProps) {
  const treeState = useAtomValue(
    useMemo(() => loadableTreeDataAtomFamily(treeId), [treeId])
  )
  const classLabels  = useAtomValue(classLabelsAtom)
  const customColors = useAtomValue(effectiveClassColorsAtom)
  const task         = useAtomValue(activeTaskAtom)
  const targetLabel  = useAtomValue(targetLabelAtom)
  const group        = useAtomValue(activeGroupAtom)
  const isReg        = task === 'regression'
  // This tree's own values. Forest-wide averages deliberately live only in the
  // Forest tab, so no number is reported in two places.
  const own = useAtomValue(useMemo(() => treeOwnMetricsAtomFamily(treeId), [treeId]))

  // Backend per-tree aggregate for the active group (purity / mean+std)
  const treeStat = own.stat

  const metrics = useMemo(() => {
    if (treeState.state !== 'hasData') return null
    return computeMetrics(treeState.data, group)
  }, [treeState, group])

  const [depthExpanded, setDepthExpanded] = useState(false)
  const [depthStackedExpanded, setDepthStackedExpanded] = useState(false)
  const [meanExpanded, setMeanExpanded] = useState(false)

  const label = (i: number) => classLabels[i] ?? `C${i}`

  return (
    <div style={{ padding: '12px 16px' }}>

      {/* Panel header */}
      <div style={{
        fontSize: 10, fontWeight: 700, color: '#888',
        fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1,
        paddingBottom: 8, borderBottom: '1px solid #eee', marginBottom: 12,
      }}>
        Tree {treeId}
      </div>

      {treeState.state === 'loading' && (
        <div style={{ fontSize: 12, color: '#bbb' }}>Loading…</div>
      )}
      {treeState.state === 'hasError' && (
        <div style={{ fontSize: 12, color: '#e53935' }}>Error loading tree</div>
      )}

      {depthExpanded && metrics && (
        <ExpandedBarChart
          title={`Nodes per Depth Level — Tree ${treeId}`}
          xLabel="Depth"
          yLabel="Node count"
          bars={metrics.depthCounts.map((c, i) => ({
            label: `d${i}`,
            value: c,
            color: '#8B6040',
          }))}
          onClose={() => setDepthExpanded(false)}
        />
      )}

      {metrics && (
        <>
          {/* Metrics the user attached to this tree. Named explicitly so they
              are never mistaken for the built-in structural ones. */}
          {(Object.keys(own.metrics).length > 0 || Object.keys(own.groupMetrics).length > 0) && (
            <Section title="User-defined metrics">
              {Object.entries(own.metrics).map(([key, v]) => (
                <Row key={key} label={key} value={fmtMetric(v)}
                  title={`${key} of tree ${treeId} alone — the forest-wide mean and σ are in the Forest tab`} />
              ))}
              {Object.entries(own.groupMetrics).map(([key, v]) => (
                <Row key={`g-${key}`} label={group.name ? `${group.name} · ${key}` : key}
                  value={fmtMetric(v)} accent
                  title={`${key} of ${group.name || 'this group'} for tree ${treeId} alone`} />
              ))}
            </Section>
          )}

          {/* Structure */}
          <Section title="Structure">
            <Row label="Nodes" value={metrics.nodeCount} />
            <Row label="Leaves" value={metrics.leafCount} />
            <Row label="Internal" value={metrics.internalCount} />
            <Row label="Max width" value={metrics.maxWidth}
              title="Maximum number of nodes at any single depth level" />
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
              title="1 − coefficient of variation of leaf depths. Higher = more balanced.">
              <span style={{ fontSize: 11, color: '#999' }}>Balance</span>
              <span style={{ display: 'flex', alignItems: 'center' }}>
                <PropBar value={metrics.balance} color={
                  metrics.balance > 0.7 ? '#3A8C3A' : metrics.balance > 0.4 ? '#f28e2b' : '#e15759'
                } />
                <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                  {(metrics.balance * 100).toFixed(0)}%
                </span>
              </span>
            </div>
          </Section>

          {/* Depth distribution — D in RF distance paper */}
          <Section title="Depth (D)">
            <Row label="Height" value={metrics.maxDepth} />
            <Row label="Mean leaf depth" value={metrics.meanLeafDepth.toFixed(1)} />
            <Row label="σ leaf depth" value={metrics.stdLeafDepth.toFixed(2)} />
            <Row label="Min leaf depth" value={metrics.minLeafDepth} />
            <div style={{ fontSize: 10, color: '#bbb', marginTop: 8 }}>Node types per depth</div>
            <div
              onClick={() => setDepthStackedExpanded(true)}
              title="Click to expand"
              style={{ cursor: 'pointer', display: 'inline-block' }}
            >
              <DepthStackedHistogram breakdown={metrics.depthBreakdown} />
            </div>

            {depthStackedExpanded && (
              <ExpandedStackedBarChart
                title={`Node types per depth — Tree ${treeId}`}
                xLabel="Depth"
                yLabel="Node count"
                bars={metrics.depthBreakdown.map((b, i) => ({ label: `d${i}`, ...b }))}
                onClose={() => setDepthStackedExpanded(false)}
              />
            )}
          </Section>

          {/* Sample distribution — S in RF distance paper */}
          <Section title="Samples (S)">
            <Row label="Root samples" value={metrics.total} />
            <Row label="Mean leaf" value={metrics.avgLeafSamples.toFixed(1)} />
            <Row label="σ leaf" value={metrics.stdLeafSamples.toFixed(1)} />
            <Row label="Min / Max leaf" value={`${metrics.minLeafSamples} / ${metrics.maxLeafSamples}`} />
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
              title="Fraction of all samples held by the top-25% leaves by size. Higher = more uneven coverage.">
              <span style={{ fontSize: 11, color: '#999' }}>Top-25% coverage</span>
              <span style={{ display: 'flex', alignItems: 'center' }}>
                <PropBar value={metrics.concentration}
                  color={metrics.concentration > 0.7 ? '#e15759' : '#4e79a7'} />
                <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                  {(metrics.concentration * 100).toFixed(0)}%
                </span>
              </span>
            </div>
          </Section>

          {isReg ? (
            <>
              {/* Deviation (regression: impurity == target std dev) */}
              <Section title="Deviation">
                {(() => {
                  const denom = metrics.rootStd > 0 ? metrics.rootStd : 1
                  const bar = (v: number) => Math.min(1, v / denom)
                  return (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                        title="Std dev of the target at the root node (before any splits).">
                        <span style={{ fontSize: 11, color: '#999' }}>Root σ</span>
                        <span style={{ display: 'flex', alignItems: 'center' }}>
                          <PropBar value={bar(metrics.rootStd)} color="#e15759" />
                          <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                            {metrics.rootStd.toFixed(3)}
                          </span>
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                        title="Σ(n_samples × σ) / total — sample-weighted std across all nodes.">
                        <span style={{ fontSize: 11, color: '#999' }}>Weighted σ</span>
                        <span style={{ display: 'flex', alignItems: 'center' }}>
                          <PropBar value={bar(metrics.weightedImpurity)} color="#f28e2b" />
                          <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                            {metrics.weightedImpurity.toFixed(3)}
                          </span>
                        </span>
                      </div>
                      {treeStat?.std !== undefined && (
                        <Row label="Leaf σ (tree)" value={treeStat.std.toFixed(3)}
                          title="Sample-weighted std across this tree's leaves — residual spread after splits." />
                      )}
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                        title="Average σ across leaf nodes only.">
                        <span style={{ fontSize: 11, color: '#999' }}>Mean leaf σ</span>
                        <span style={{ display: 'flex', alignItems: 'center' }}>
                          <PropBar value={bar(metrics.meanLeafImpurity)} color="#b07aa1" />
                          <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                            {metrics.meanLeafImpurity.toFixed(3)}
                          </span>
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                        title="Fraction of leaves with σ below 10% of the root σ (very consistent leaves).">
                        <span style={{ fontSize: 11, color: '#999' }}>Low-variance leaves</span>
                        <span style={{ display: 'flex', alignItems: 'center' }}>
                          <PropBar value={metrics.lowVarLeafFrac} color="#3A8C3A" />
                          <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                            {(metrics.lowVarLeafFrac * 100).toFixed(0)}%
                          </span>
                        </span>
                      </div>
                    </>
                  )
                })()}
              </Section>

              {/* Target (regression prediction summary) */}
              <Section title="Target">
                {targetLabel && <Row label="Variable" value={targetLabel} accent />}
                <Row label="Root mean" value={metrics.rootMean.toFixed(3)} accent />
                <Row label="Leaf range"
                  value={`${metrics.leafMeanMin.toFixed(2)} – ${metrics.leafMeanMax.toFixed(2)}`} />
                <Row label="Mean leaf pred."
                  value={`${metrics.leafMeanAvg.toFixed(2)} ± ${metrics.leafMeanStd.toFixed(2)}`} />
                <div style={{ fontSize: 10, color: '#bbb', marginTop: 6 }}>leaf prediction distribution</div>
                <div
                  onClick={() => setMeanExpanded(true)}
                  title="Click to expand"
                  style={{ cursor: 'pointer', display: 'inline-block' }}
                >
                  <MeanHistogram hist={metrics.meanHist} min={metrics.leafMeanMin} max={metrics.leafMeanMax} />
                </div>

                {meanExpanded && (
                  <ExpandedBarChart
                    title={`Leaf prediction distribution — Tree ${treeId}`}
                    xLabel={targetLabel ? `Predicted ${targetLabel}` : 'Predicted value'}
                    yLabel="Leaf count"
                    bars={metrics.meanHist.map((c, i) => {
                      const binW = (metrics.leafMeanMax - metrics.leafMeanMin) / metrics.meanHist.length
                      return {
                        label: (metrics.leafMeanMin + (i + 0.5) * binW).toFixed(1),
                        value: c,
                        color: '#5B8DB8',
                      }
                    })}
                    onClose={() => setMeanExpanded(false)}
                  />
                )}
              </Section>
            </>
          ) : (
            <>
              {/* Purity / impurity */}
              <Section title="Purity">
                {treeStat?.purity !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                    title="1 − sample-weighted leaf Gini for the active label group. 100% = every leaf pure.">
                    <span style={{ fontSize: 11, color: '#999' }}>Purity (tree)</span>
                    <span style={{ display: 'flex', alignItems: 'center' }}>
                      <PropBar value={treeStat.purity} color="#3A8C3A" />
                      <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                        {(treeStat.purity * 100).toFixed(1)}%
                      </span>
                    </span>
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                  title="Gini impurity at the root node (before any splits).">
                  <span style={{ fontSize: 11, color: '#999' }}>Root impurity</span>
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    <PropBar value={metrics.rootImpurity} color="#e15759" />
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                      {metrics.rootImpurity.toFixed(3)}
                    </span>
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                  title="Σ(n_samples × impurity) / total — weighted average Gini across all nodes.">
                  <span style={{ fontSize: 11, color: '#999' }}>Weighted Gini</span>
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    <PropBar value={metrics.weightedImpurity} color="#f28e2b" />
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                      {metrics.weightedImpurity.toFixed(3)}
                    </span>
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                  title="Average Gini impurity across leaf nodes only.">
                  <span style={{ fontSize: 11, color: '#999' }}>Mean leaf imp.</span>
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    <PropBar value={metrics.meanLeafImpurity} color="#b07aa1" />
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                      {metrics.meanLeafImpurity.toFixed(3)}
                    </span>
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}
                  title="Fraction of leaf nodes with Gini < 0.01 (effectively pure leaves).">
                  <span style={{ fontSize: 11, color: '#999' }}>Pure leaves</span>
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    <PropBar value={metrics.pureLeafFrac} color="#3A8C3A" />
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#333', marginLeft: 4 }}>
                      {(metrics.pureLeafFrac * 100).toFixed(0)}%
                    </span>
                  </span>
                </div>
              </Section>

              {/* Class distribution */}
              <Section title="Classes">
                <Row
                  label="Dominant"
                  value={`${label(metrics.dominantClass)} (${(metrics.classFractions[metrics.dominantClass] * 100).toFixed(1)}%)`}
                  accent
                />
                <Row label="Count" value={metrics.classCount} />
                <div style={{ fontSize: 10, color: '#bbb', marginTop: 6 }}>root sample distribution</div>
                <ClassBar fractions={metrics.classFractions} labels={classLabels} customColors={customColors} />
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px 8px', marginTop: 5 }}>
                  {metrics.classFractions.map((f, i) => (
                    <span key={i} style={{ fontSize: 10, color: '#888', display: 'flex', alignItems: 'center', gap: 3 }}>
                      <svg width={8} height={8}>
                        <rect width={8} height={8} rx={2}
                          fill={customColors[i] ?? classColor(i, metrics.classFractions.length)} opacity={0.8} />
                      </svg>
                      {label(i)} {(f * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </Section>
            </>
          )}
        </>
      )}
    </div>
  )
}
