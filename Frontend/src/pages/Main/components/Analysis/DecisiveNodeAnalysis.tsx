import { useEffect, useRef } from 'react'
import { useAtom, useAtomValue } from 'jotai'
import { classLabelsAtom, activeTaskAtom, activeGroupAtom } from '../Forest/ForestAtoms'
import { selectedTreeIdAtom } from '../ForestView/ForestViewAtoms'
import { effectiveClassColorsAtom } from '../Legend/LegendAtoms'
import { classColor } from '../classColors'
import { decisiveAtom } from './DecisiveAtoms'
import { HEADING_FONT } from '../../../../theme'

const HEADER = {
  fontSize: 9, fontWeight: 700, color: '#aaa',
  fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 5,
} as const

function LabelList({
  labels, colors, selected, onPick, accent,
}: {
  labels: string[]
  colors: Record<number, string>
  selected: number | null
  onPick: (i: number) => void
  accent: string
}) {
  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
      {labels.map((label, i) => {
        const color = colors[i] ?? classColor(i, labels.length)
        const isActive = selected === i
        return (
          <div
            key={i}
            onClick={() => onPick(i)}
            title={label}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '2px 4px', borderRadius: 4, cursor: 'pointer',
              background: isActive ? `${accent}18` : 'transparent',
              outline: isActive ? `1.5px solid ${accent}` : '1.5px solid transparent',
              transition: 'background 0.12s, outline-color 0.12s',
            }}
          >
            <svg width={13} height={13} style={{ flexShrink: 0 }}>
              <rect width={13} height={13} rx={3} fill={color} />
              <rect width={13} height={13} rx={3} fill="none" stroke="#555" strokeWidth={1.2} />
            </svg>
            <span style={{
              fontSize: 10.5,
              color: isActive ? '#222' : '#666',
              fontWeight: isActive ? 700 : 400,
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {label}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default function DecisiveNodeAnalysis() {
  const task = useAtomValue(activeTaskAtom)
  const labels = useAtomValue(classLabelsAtom)
  const colors = useAtomValue(effectiveClassColorsAtom)
  const selectedTree = useAtomValue(selectedTreeIdAtom)
  const group = useAtomValue(activeGroupAtom)
  const [dca, setDca] = useAtom(decisiveAtom)

  // Label indices are group-relative — clear the pair when the active group
  // changes so a stale index can't point at the wrong (or a nonexistent) label.
  const prevGroup = useRef(group.index)
  useEffect(() => {
    if (prevGroup.current !== group.index) {
      prevGroup.current = group.index
      setDca(p => ({ ...p, labelA: null, labelB: null }))
    }
  }, [group.index, setDca])

  const section = (children: React.ReactNode) => (
    <div style={{ borderTop: '1px solid #eee', paddingTop: 10, marginTop: 4 }}>
      <div style={HEADER}>Decisive Node Analysis</div>
      {children}
    </div>
  )

  if (task === 'regression') {
    return section(
      <div style={{ fontSize: 10, color: '#bbb', lineHeight: 1.5 }}>
        Available for classification forests only.
      </div>
    )
  }
  if (labels.length < 2) {
    return section(
      <div style={{ fontSize: 10, color: '#bbb', lineHeight: 1.5 }}>
        Needs at least 2 class labels.
      </div>
    )
  }

  // Picking a label clears it from the other list if it would duplicate.
  const pickA = (i: number) =>
    setDca(p => ({ ...p, labelA: p.labelA === i ? null : i, labelB: p.labelB === i ? null : p.labelB }))
  const pickB = (i: number) =>
    setDca(p => ({ ...p, labelB: p.labelB === i ? null : i, labelA: p.labelA === i ? null : p.labelA }))

  const active = dca.labelA !== null && dca.labelB !== null

  return section(
    <>
      <div style={{ fontSize: 9.5, color: '#aaa', lineHeight: 1.5, marginBottom: 7 }}>
        Highlights the nodes in the selected tree that separate two classes — the split where they
        last mix, plus the pure branches beneath it.
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: '#999', marginBottom: 4 }}>Label A</div>
          <LabelList labels={labels} colors={colors} selected={dca.labelA} onPick={pickA} accent="#3A8C3A" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: '#999', marginBottom: 4 }}>Label B</div>
          <LabelList labels={labels} colors={colors} selected={dca.labelB} onPick={pickB} accent="#5B8DB8" />
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 9 }}>
        <span style={{ fontSize: 10, color: '#888' }}>Purity ≥</span>
        <input
          type="number"
          min={0.5} max={1} step={0.05}
          value={dca.threshold}
          onChange={e => {
            const v = Number(e.target.value)
            if (!Number.isNaN(v)) setDca(p => ({ ...p, threshold: Math.min(1, Math.max(0.5, v)) }))
          }}
          title="A node counts as pure for a label when its share of the two selected labels is at least this."
          style={{
            width: 58, fontSize: 11, padding: '3px 4px',
            border: '1px solid #ddd', borderRadius: 4, color: '#555', outline: 'none',
          }}
        />
        <span style={{ fontSize: 9, color: '#bbb' }}>of A + B</span>
      </div>

      {active && !selectedTree && (
        <div style={{ fontSize: 10, color: '#c99a00', marginTop: 7, fontStyle: 'italic' }}>
          Select a tree in the forest view to see the highlights.
        </div>
      )}
      {active && selectedTree && (
        <div style={{ fontSize: 9, color: '#aaa', marginTop: 8, lineHeight: 1.6 }}>
          The tree fades everything except the decisive split and the pure branches beneath it.
          Both sides of a split are drawn as pie charts — hover or click one to inspect it.
        </div>
      )}
    </>
  )
}
