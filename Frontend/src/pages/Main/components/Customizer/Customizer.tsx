import { useAtom, useAtomValue } from 'jotai'
import { leafModeAtom, colorRangeModeAtom, type LeafMode, type ColorRangeMode } from './CustomizerAtoms'
import { activeTaskAtom } from '../Forest/ForestAtoms'
import { HEADING_FONT } from '../../../../theme'

const LABEL = { fontSize: 10, fontWeight: 700, color: '#aaa', fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1.2, marginBottom: 6 } as const
const SELECT = {
  width: '100%', fontSize: 11, padding: '3px 5px',
  border: '1px solid #ddd', borderRadius: 4, background: '#fff',
  color: '#555', cursor: 'pointer', outline: 'none',
} as const

export default function Customizer() {
  const [leafMode, setLeafMode] = useAtom(leafModeAtom)
  const [rangeMode, setRangeMode] = useAtom(colorRangeModeAtom)
  const task = useAtomValue(activeTaskAtom)
  const isReg = task === 'regression'

  const leafOptions: { value: LeafMode; label: string }[] = isReg
    ? [{ value: 'pie', label: 'Mean Prediction' }, { value: 'purity', label: 'Std Deviation' }]
    : [{ value: 'pie', label: 'Class Distribution' }, { value: 'purity', label: 'Purity' }]

  return (
    <div style={{ borderTop: '1px solid #e0e0e0', padding: '8px 10px', background: '#fafafa', flexShrink: 0 }}>
      <div style={LABEL}>Leaf View</div>
      <select value={leafMode} onChange={e => setLeafMode(e.target.value as LeafMode)} style={SELECT}>
        {leafOptions.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>

      {isReg && (
        <>
          <div style={{ ...LABEL, marginTop: 10 }}>Color Range</div>
          <select value={rangeMode} onChange={e => setRangeMode(e.target.value as ColorRangeMode)} style={SELECT}>
            <option value="forest">Per-forest</option>
            <option value="tree">Per-tree</option>
          </select>
        </>
      )}
    </div>
  )
}
