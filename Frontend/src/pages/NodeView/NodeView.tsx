import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { useAtom, useAtomValue, useSetAtom } from 'jotai'
import { classLabelsAtom, loadableTreeDataAtomFamily } from '../Main/components/Forest/ForestAtoms'
import { effectiveClassColorsAtom } from '../Main/components/Legend/LegendAtoms'
import {
  loadableReferencesAtom, refImageUrl, selectedNodeAtom, highlightedRefsAtom,
  type ReferenceMeta,
} from './NodeViewAtoms'
import {
  computeChildrenMap, partitionRefsAtNode, labelSplit, outcomeOf, refLabelKey,
  type LabelSplitRow, type SplitOutcome,
} from './referencePartition'
import { classColor } from '../Main/components/classColors'
import RefThumb from './RefThumb'
import MeanImage from './MeanImage'
import { HEADING_FONT } from '../../theme'

// NodeView — a full window (opened by clicking a node in a tree render) that
// shows the visual references passing through that node. For an internal node
// the references are partitioned by the child they continue into: seeing the
// two groups side by side is exactly "what did this split function separate".
// Leaves show the references whose paths end there.
//
// Mounted once at App level; renders nothing until selectedNodeAtom is set.

const HEADER_STAT = { fontSize: 11, color: '#888' } as const

const LEFT_ACCENT = '#3A8C3A'
const RIGHT_ACCENT = '#5B8DB8'
const MAX_SPLIT_ROWS = 10

// What the label-split table currently focuses. Filters are sticky toggles:
// clicking a label name or a result adds it, clicking it again removes it, and
// several can be active at once. A reference is highlighted if it matches ANY
// active filter (its label is selected, or its label's outcome is selected).
type SplitFilters = {
  labels: ReadonlySet<string>          // refLabelKey values
  outcomes: ReadonlySet<SplitOutcome>
}
const NO_FILTERS: SplitFilters = { labels: new Set(), outcomes: new Set() }

function toggle<T>(set: ReadonlySet<T>, v: T): Set<T> {
  const next = new Set(set)
  if (next.has(v)) next.delete(v)
  else next.add(v)
  return next
}

// Label-split table cells
const TH = { fontWeight: 600, padding: '0 6px 4px 0', borderBottom: '1px solid #eee' } as const
const TD = { padding: '2px 6px 2px 0', verticalAlign: 'middle' } as const
const NUM_CELL = { textAlign: 'right', width: 62 } as const

// Per-label view of how the references divided at this split: a plain table,
// one row per label, with the two sides as named columns. A label whose count
// sits entirely in one column was isolated by the split — the question the
// reference paths exist to answer. Named columns rather than color coding, so
// which side is which needs no legend.
function LabelSplitPanel({
  rows, labels, colors, filters, onFilters,
}: {
  rows: LabelSplitRow[]
  labels: string[]
  colors: Record<number, string>
  filters: SplitFilters
  onFilters: (f: SplitFilters) => void
}) {
  const [showAll, setShowAll] = useState(false)
  if (rows.length === 0) return null

  const isolated = rows.filter(r => r.exclusive !== null).length
  const single = rows.filter(r => r.total < 2).length
  const mixed = rows.length - isolated - single
  // A 30-class forest produces 30 rows at the root — keep the summary brief and
  // let the user open the rest.
  const shown = showAll ? rows : rows.slice(0, MAX_SPLIT_ROWS)
  const activeCount = filters.labels.size + filters.outcomes.size

  return (
    <div style={{
      border: '1px solid #e5e5e5', borderRadius: 8,
      padding: '9px 12px', marginBottom: 12,
    }}>
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 7,
      }}>
        <span style={{ fontFamily: HEADING_FONT, fontSize: 11, fontWeight: 700, color: '#444' }}>Label split</span>
        <span style={{ fontSize: 10, color: '#999' }}>
          {isolated} isolated · {mixed} mixed
          {single > 0 && ` · ${single} single-ref`}
        </span>
        <span style={{ flex: 1 }} />
        {activeCount > 0 && (
          <span
            onClick={() => onFilters(NO_FILTERS)}
            title="Remove every active filter"
            style={{ fontSize: 9.5, color: '#5B8DB8', cursor: 'pointer', fontWeight: 600 }}
          >
            {activeCount} filter{activeCount !== 1 ? 's' : ''} active · clear all ×
          </span>
        )}
      </div>

      <table style={{
        width: '100%', borderCollapse: 'collapse',
        fontSize: 10.5, fontVariantNumeric: 'tabular-nums',
      }}>
        <thead>
          <tr style={{ color: '#aaa', fontSize: 9.5, textAlign: 'left' }}>
            <th style={TH}>label</th>
            <th style={{ ...TH, ...NUM_CELL }}>left</th>
            <th style={{ ...TH, ...NUM_CELL }}>right</th>
            <th style={{ ...TH, textAlign: 'right' }}>result</th>
          </tr>
        </thead>
        <tbody>
          {shown.map(r => {
            const name = (r.labelIndex !== null ? labels[r.labelIndex] : undefined) ?? r.label ?? '—'
            const swatch = r.labelIndex !== null
              ? colors[r.labelIndex] ?? classColor(r.labelIndex, Math.max(labels.length, r.labelIndex + 1))
              : '#ccc'
            const outcome = outcomeOf(r)
            const verdict = outcome === 'isolated' ? 'isolated' : outcome === 'few' ? '1 ref' : 'both'
            const active = filters.outcomes.has(outcome)
            const labelActive = filters.labels.has(r.key)
            return (
              <tr
                key={r.key}
                title={`${name}: ${r.left} left, ${r.right} right`
                     + (r.exclusive ? ` — all ${r.total} references went ${r.exclusive}, the split isolated this label`
                        : r.total < 2 ? ' — only one reference here, too few to tell'
                        : ' — present on both sides, the split did not separate it')}
                style={{ background: active || labelActive ? '#f4f8f4' : undefined }}
              >
                <td style={TD}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      width: 9, height: 9, borderRadius: 2, background: swatch, flexShrink: 0,
                    }} />
                    {/* Clicking the name toggles a sticky highlight of this
                        label's references in the columns below. */}
                    <span
                      onClick={() => onFilters({ ...filters, labels: toggle(filters.labels, r.key) })}
                      title={labelActive ? `Click to stop highlighting ${name}`
                                         : `Highlight the ${name} references below`}
                      style={{
                        cursor: 'pointer',
                        color: labelActive ? '#3A8C3A' : '#555',
                        fontWeight: labelActive ? 700 : 400,
                        textDecoration: labelActive ? 'underline' : undefined,
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      }}
                    >
                      {name}
                    </span>
                  </span>
                </td>
                {/* Zero is muted so the side that actually holds the label pops
                    out of the column without needing a color code. */}
                <td style={{ ...TD, ...NUM_CELL, color: r.left ? '#333' : '#ddd', fontWeight: r.left ? 600 : 400 }}>
                  {r.left}
                </td>
                <td style={{ ...TD, ...NUM_CELL, color: r.right ? '#333' : '#ddd', fontWeight: r.right ? 600 : 400 }}>
                  {r.right}
                </td>
                {/* Clicking a result toggles a sticky highlight of every label
                    with that outcome; combines with any other active filter. */}
                <td style={{ ...TD, textAlign: 'right', fontSize: 9.5 }}>
                  <span
                    onClick={() => onFilters({ ...filters, outcomes: toggle(filters.outcomes, outcome) })}
                    title={active ? `Click to stop highlighting ${verdict} labels`
                                  : `Show only the ${verdict} references below`}
                    style={{
                      cursor: 'pointer',
                      padding: '1px 5px', borderRadius: 3,
                      border: '1px solid',
                      borderColor: active ? '#3A8C3A' : 'transparent',
                      background: active ? '#3A8C3A' : 'transparent',
                      color: active ? '#fff' : r.exclusive ? '#555' : '#bbb',
                      fontWeight: active || r.exclusive ? 700 : 400,
                    }}
                  >
                    {verdict}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {rows.length > MAX_SPLIT_ROWS && (
        <div
          onClick={() => setShowAll(v => !v)}
          style={{
            fontSize: 10, color: '#5B8DB8', cursor: 'pointer',
            marginTop: 4, fontWeight: 600,
          }}
        >
          {showAll ? '▴ show fewer' : `▾ ${rows.length - MAX_SPLIT_ROWS} more label${rows.length - MAX_SPLIT_ROWS !== 1 ? 's' : ''}`}
        </div>
      )}
    </div>
  )
}

function Column({
  title, accent, refs, mean, labels, colors, onEnlarge, isDimmed,
}: {
  title: string
  accent: string
  refs: ReferenceMeta[]
  /** Show one averaged image per column instead of the thumbnail grid. */
  mean: boolean
  labels: string[]
  colors: Record<number, string>
  onEnlarge: (r: ReferenceMeta) => void
  /** Set while a label-split outcome is selected in the table above. */
  isDimmed?: (r: ReferenceMeta) => boolean
}) {
  // With filters active the mean image summarises the HIGHLIGHTED references —
  // averaging the faded-out ones in would defeat the selection.
  const inFocus = isDimmed ? refs.filter(r => !isDimmed(r)) : refs
  return (
    <div style={{
      flex: 1, minWidth: 0,
      border: '1px solid #e5e5e5', borderTop: `3px solid ${accent}`,
      borderRadius: 8, padding: '10px 12px',
      display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#444' }}>
        {title}
        <span style={{ fontWeight: 400, color: '#999' }}> — {refs.length} reference{refs.length !== 1 ? 's' : ''}</span>
      </div>
      {refs.length === 0 ? (
        <div style={{ fontSize: 10, color: '#bbb', fontStyle: 'italic' }}>none</div>
      ) : mean ? (
        inFocus.length === 0
          ? <div style={{ fontSize: 10, color: '#bbb', fontStyle: 'italic' }}>none selected</div>
          : <MeanImage urls={inFocus.map(refImageUrl)} />
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignContent: 'flex-start' }}>
          {refs.map(r => (
            <RefThumb
              key={r.id} r={r} labels={labels} colors={colors}
              onClick={() => onEnlarge(r)}
              dimmed={isDimmed?.(r) ?? false}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function NodeViewWindow({ treeId, nodeId }: { treeId: string; nodeId: number }) {
  const setSelected = useSetAtom(selectedNodeAtom)
  const close = () => setSelected(null)

  const treeState = useAtomValue(useMemo(() => loadableTreeDataAtomFamily(treeId), [treeId]))
  const refsState = useAtomValue(loadableReferencesAtom)
  const labels = useAtomValue(classLabelsAtom)
  const colors = useAtomValue(effectiveClassColorsAtom)

  const [enlarged, setEnlarged] = useState<ReferenceMeta | null>(null)
  const [filters, setFilters] = useState<SplitFilters>(NO_FILTERS)
  const [mean, setMean] = useState(false)
  const [highlightedRefs, setHighlightedRefs] = useAtom(highlightedRefsAtom)
  // "Show path in tree": add this reference to the tree's highlighted paths
  // (several can stack) and return to the tree render, where everything off the
  // highlighted paths fades out.
  const showPathInTree = (r: ReferenceMeta) => {
    setHighlightedRefs(prev => {
      const cur = prev[treeId] ?? []
      return cur.includes(r.id) ? prev : { ...prev, [treeId]: [...cur, r.id] }
    })
    setEnlarged(null)
    close()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (enlarged) setEnlarged(null)
        else close()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enlarged])

  const nodes = treeState.state === 'hasData' ? treeState.data : null
  const refs = refsState.state === 'hasData' ? refsState.data : []
  const node = nodes?.find(n => n.id === nodeId) ?? null
  const childrenMap = useMemo(() => (nodes ? computeChildrenMap(nodes) : null), [nodes])
  const children = childrenMap?.get(nodeId) ?? null
  const isLeaf = children === null

  const partition = useMemo(
    () => (childrenMap ? partitionRefsAtNode(refs, treeId, nodeId, childrenMap) : null),
    [refs, treeId, nodeId, childrenMap],
  )
  const splitRows = useMemo(
    () => (partition ? labelSplit(partition.toLeft, partition.toRight) : []),
    [partition],
  )
  // Selecting an outcome in the split table focuses those references in both
  // columns (label → outcome, then reference → its label).
  const outcomeByLabel = useMemo(
    () => new Map(splitRows.map(r => [r.key, outcomeOf(r)])),
    [splitRows],
  )
  const isDimmed = useMemo(() => {
    if (filters.labels.size === 0 && filters.outcomes.size === 0) return undefined
    // Union: lit if ANY active filter matches.
    return (r: ReferenceMeta) => {
      const key = refLabelKey(r)
      if (filters.labels.has(key)) return false
      const outcome = outcomeByLabel.get(key)
      return !(outcome !== undefined && filters.outcomes.has(outcome))
    }
  }, [filters, outcomeByLabel])

  let body: React.ReactNode
  if (!nodes || !partition) {
    body = <div style={{ padding: 40, textAlign: 'center', color: '#bbb', fontSize: 12 }}>Loading tree data…</div>
  } else if (refs.length === 0) {
    body = (
      <div style={{ padding: 40, textAlign: 'center', color: '#bbb', fontSize: 12, lineHeight: 1.6 }}>
        No references registered for this iteration.<br />
        Call <code style={{ fontSize: 11 }}>treeflow.add_reference(sample, image=...)</code> during training.
      </div>
    )
  } else if (partition.arriving.length === 0) {
    body = (
      <div style={{ padding: 40, textAlign: 'center', color: '#bbb', fontSize: 12 }}>
        No references pass through this node in this iteration.
      </div>
    )
  } else if (isLeaf) {
    body = (
      <Column
        title="References ending at this leaf"
        accent="#3A8C3A"
        refs={partition.ending}
        mean={mean}
        labels={labels}
        colors={colors}
        onEnlarge={setEnlarged}
      />
    )
  } else {
    body = (
      <>
        {/* Which labels the split actually pulled apart */}
        <LabelSplitPanel
          rows={splitRows}
          labels={labels}
          colors={colors}
          filters={filters}
          onFilters={setFilters}
        />
        <div style={{ display: 'flex', gap: 12, alignItems: 'stretch' }}>
          <Column
            title={`→ left child (node ${children!.left})`}
            accent={LEFT_ACCENT}
            refs={partition.toLeft}
                mean={mean}
                labels={labels}
            colors={colors}
            onEnlarge={setEnlarged}
            isDimmed={isDimmed}
          />
          <Column
            title={`→ right child (node ${children!.right})`}
            accent={RIGHT_ACCENT}
            refs={partition.toRight}
                mean={mean}
                labels={labels}
            colors={colors}
            onEnlarge={setEnlarged}
            isDimmed={isDimmed}
          />
        </div>
        {partition.ending.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <Column
              title="Path ends here (unevaluable split for these references)"
              accent="#c99a00"
              refs={partition.ending}
                    mean={mean}
                    labels={labels}
              colors={colors}
              onEnlarge={setEnlarged}
              // Dims along with the rest, so a selection doesn't leave this
              // strip as the only lit block (these took neither side).
              isDimmed={isDimmed}
            />
          </div>
        )}
      </>
    )
  }

  const modal = (
    <div style={{
      position: 'fixed', inset: 0, background: '#fff', zIndex: 2000,
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 16,
        padding: '12px 20px', borderBottom: '1px solid #e5e5e5', flexShrink: 0,
      }}>
        <span style={{ fontFamily: HEADING_FONT, fontWeight: 700, fontSize: 15, color: '#333' }}>
          Node {nodeId} <span style={{ fontWeight: 400, color: '#999' }}>— Tree {treeId}</span>
        </span>
        {node && (
          <>
            <span style={HEADER_STAT}>depth {node.depth}</span>
            <span style={HEADER_STAT}>{node.n_node_samples} samples</span>
            <span style={HEADER_STAT}>{isLeaf ? 'leaf' : 'split node'}</span>
          </>
        )}
        {partition && partition.arriving.length > 0 && (
          <span style={HEADER_STAT}>
            {partition.arriving.length} reference{partition.arriving.length !== 1 ? 's' : ''} arriving
          </span>
        )}
        <span style={{ flex: 1 }} />
        {partition && partition.arriving.length > 0 && (
          <button
            onClick={() => setMean(m => !m)}
            title="Replace each column's thumbnails by the average of its references"
            style={{
              fontSize: 11, fontWeight: 600, padding: '4px 10px',
              border: '1px solid', borderColor: mean ? '#3A8C3A' : '#ccc',
              borderRadius: 4, background: mean ? '#3A8C3A' : '#fff',
              color: mean ? '#fff' : '#555', cursor: 'pointer',
            }}
          >
            {mean ? 'Mean image ✓' : 'Mean image'}
          </button>
        )}
        <button
          onClick={close}
          style={{
            border: 'none', background: 'none', fontSize: 22, lineHeight: 1,
            padding: '0 2px', cursor: 'pointer', color: '#aaa',
          }}
        >
          ×
        </button>
      </div>

      {/* Explainer */}
      {!isLeaf && partition && partition.arriving.length > 0 && (
        <div style={{
          padding: '8px 20px', fontSize: 10.5, color: '#999',
          borderBottom: '1px solid #f2f2f2', flexShrink: 0, lineHeight: 1.5,
        }}>
          References arriving at this node, split by the child their path continues into —
          the two groups show what this split function separated visually.
        </div>
      )}

      {/* Body */}
      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {body}
      </div>

      {/* Enlarged reference overlay */}
      {enlarged && (
        <div
          onClick={() => setEnlarged(null)}
          style={{
            position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.55)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer',
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: '#fff', borderRadius: 10, padding: 20,
              maxWidth: '80vw', boxShadow: '0 8px 36px rgba(0,0,0,0.25)',
              cursor: 'default',
            }}
          >
            <img
              src={refImageUrl(enlarged)}
              crossOrigin="anonymous"
              style={{
                display: 'block', width: 280, height: 280,
                imageRendering: 'pixelated', borderRadius: 6, border: '1px solid #ddd',
              }}
            />
            <div style={{ marginTop: 10, fontSize: 12, color: '#333', fontWeight: 700 }}>
              {enlarged.name ?? enlarged.id}
            </div>
            <div style={{ fontSize: 10.5, color: '#888', marginTop: 3, lineHeight: 1.6 }}>
              {enlarged.label !== null && (
                <div>label: {labels[enlarged.label_index ?? -1] ?? enlarged.label}</div>
              )}
              {enlarged.step !== null && <div>registered at step {enlarged.step}</div>}
              {(() => {
                const path = enlarged.paths[treeId] ?? []
                const already = (highlightedRefs[treeId] ?? []).includes(enlarged.id)
                if (path.length === 0) {
                  return <div>path (tree {treeId}): —</div>
                }
                // Boxed so it reads as a button: clicking it highlights this
                // path in the tree render and switches back to it.
                return (
                  <div
                    onClick={() => showPathInTree(enlarged)}
                    title="Highlight this path in the tree render"
                    style={{
                      marginTop: 6, padding: '6px 9px',
                      border: '1px solid', borderColor: already ? '#3A8C3A' : '#cfd8cf',
                      borderRadius: 6, background: already ? '#f4f8f4' : '#fafafa',
                      cursor: 'pointer', wordBreak: 'break-all',
                    }}
                  >
                    <div style={{ fontSize: 9.5, fontWeight: 700, color: '#3A8C3A', marginBottom: 2 }}>
                      {already ? '✓ highlighted — show in tree →' : 'Show path in tree →'}
                    </div>
                    <span style={{ color: '#666' }}>
                      tree {treeId}: {path.join(' → ')}
                    </span>
                  </div>
                )
              })()}
            </div>
            <div style={{ fontSize: 9, color: '#ccc', marginTop: 8, textAlign: 'right' }}>
              click outside or press Esc to close
            </div>
          </div>
        </div>
      )}
    </div>
  )

  return createPortal(modal, document.body)
}

export default function NodeView() {
  const selected = useAtomValue(selectedNodeAtom)
  if (!selected) return null
  return (
    <NodeViewWindow
      key={`${selected.treeId}:${selected.nodeId}`}
      treeId={selected.treeId}
      nodeId={selected.nodeId}
    />
  )
}
