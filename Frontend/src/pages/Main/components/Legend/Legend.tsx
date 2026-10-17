import { useMemo, useRef } from 'react'
import { useAtom, useAtomValue } from 'jotai'
import { selectedTreeIdAtom } from '../ForestView/ForestViewAtoms'
import {
  classLabelsAtom, loadableTreeDataAtomFamily,
  activeTaskAtom, targetRangeAtom, stdRangeAtom, targetLabelAtom,
  labelGroupsAtom, selectedGroupAtom, activeGroupAtom,
} from '../Forest/ForestAtoms'
import { classColor, BASE_CLASS_COUNT } from '../classColors'
import {
  selectedClassAtom, customClassColorsAtom, effectiveClassColorsAtom,
  meanColorsAtom, stdColorsAtom, type GradientColors,
} from './LegendAtoms'
import { leafModeAtom, colorRangeModeAtom } from '../Customizer/CustomizerAtoms'
import { treeRanges } from '../Tree/regressionColor'
import { HEADING_FONT } from '../../../../theme'

function toPickerHex(color: string): string {
  return /^#[0-9a-f]{6}$/i.test(color) ? color : '#888888'
}

export default function Legend() {
  const selectedId      = useAtomValue(selectedTreeIdAtom)
  const [selectedClass, setSelectedClass] = useAtom(selectedClassAtom)
  const [, setCustomColors] = useAtom(customClassColorsAtom)
  const customColors        = useAtomValue(effectiveClassColorsAtom)
  const classLabels = useAtomValue(classLabelsAtom)
  const task        = useAtomValue(activeTaskAtom)
  const leafMode    = useAtomValue(leafModeAtom)
  const colorRangeMode = useAtomValue(colorRangeModeAtom)
  const targetRange = useAtomValue(targetRangeAtom)
  const stdRange    = useAtomValue(stdRangeAtom)
  const targetLabel = useAtomValue(targetLabelAtom)
  const [meanColors, setMeanColors] = useAtom(meanColorsAtom)
  const [stdColors, setStdColors]   = useAtom(stdColorsAtom)
  const groups = useAtomValue(labelGroupsAtom)
  const [, setSelectedGroup] = useAtom(selectedGroupAtom)
  const group = useAtomValue(activeGroupAtom)

  const treeState = useAtomValue(
    useMemo(() => loadableTreeDataAtomFamily(selectedId ?? 'none'), [selectedId])
  )

  // Multi-label / multi-target: dimension picker shown above both legend styles.
  // Switching groups clears the class filter — indices refer to the new group.
  const groupPicker = groups.length > 1 ? (
    <select
      value={group.index}
      onChange={e => {
        setSelectedGroup(Number(e.target.value))
        setSelectedClass(null)
      }}
      style={{
        width: '100%', fontSize: 11, padding: '3px 5px', marginBottom: 7,
        border: '1px solid #ddd', borderRadius: 4, background: '#fff',
        color: '#555', cursor: 'pointer', outline: 'none',
      }}
    >
      {groups.map(g => (
        <option key={g.index} value={g.index}>
          {g.name || `Group ${g.index}`}
        </option>
      ))}
    </select>
  ) : null

  // All hooks must run before any early return, so the classification-path hooks
  // (labels / color-picker refs) are declared here too — otherwise switching
  // between a classification and a regression forest changes the hook count and
  // React throws "Rendered fewer hooks than expected".
  const labels = useMemo(() => {
    if (classLabels.length > 0) return classLabels
    if (treeState.state === 'hasData' && treeState.data.length > 0) {
      const count = treeState.data[0].value.length
      if (count > 0) return Array.from({ length: count }, (_, i) => `Class ${i}`)
    }
    return Array.from({ length: BASE_CLASS_COUNT }, (_, i) => `Class ${i}`)
  }, [classLabels, treeState])
  const colorInputRef   = useRef<HTMLInputElement>(null)
  const editingClassRef = useRef<number>(-1)

  // ── Regression: gradient legend with editable endpoints ──
  const regInputRef = useRef<HTMLInputElement>(null)
  const regEditRef  = useRef<{ mode: 'mean' | 'std'; which: 'low' | 'high' }>({ mode: 'mean', which: 'low' })

  if (task === 'regression') {
    const isStd = leafMode === 'purity'
    const colors: GradientColors = isStd ? stdColors : meanColors
    const setColors = isStd ? setStdColors : setMeanColors
    const perTree = treeState.state === 'hasData' ? treeRanges(treeState.data, group) : null
    const range: [number, number] =
      colorRangeMode === 'tree'
        ? (isStd ? (perTree?.std ?? [0, 1]) : (perTree?.mean ?? [0, 1]))
        : (isStd ? (stdRange ?? perTree?.std ?? [0, 1]) : (targetRange ?? perTree?.mean ?? [0, 1]))
    const caption = isStd ? 'Std deviation' : (targetLabel || 'Prediction')

    const openReg = (e: React.MouseEvent, which: 'low' | 'high') => {
      e.stopPropagation()
      regEditRef.current = { mode: isStd ? 'std' : 'mean', which }
      if (regInputRef.current) {
        regInputRef.current.value = toPickerHex(which === 'low' ? colors.low : colors.high)
        regInputRef.current.click()
      }
    }

    const swatch = (which: 'low' | 'high') => (
      <div onClick={e => openReg(e, which)} title="Click to change color"
        style={{ cursor: 'pointer', lineHeight: 0 }}>
        <svg width={16} height={16}>
          <rect width={16} height={16} rx={3} fill={which === 'low' ? colors.low : colors.high} />
          <rect width={16} height={16} rx={3} fill="none" stroke="#555" strokeWidth={1.5} />
        </svg>
      </div>
    )

    return (
      <div style={{ borderTop: '1px solid #e0e0e0', padding: '8px 10px 10px', background: '#fafafa', flexShrink: 0 }}>
        <input
          ref={regInputRef} type="color"
          onChange={e => {
            const { which } = regEditRef.current
            setColors(prev => ({ ...prev, [which]: e.target.value }))
          }}
          style={{ position: 'fixed', opacity: 0, pointerEvents: 'none', width: 0, height: 0 }}
        />
        <div style={{ fontSize: 10, fontWeight: 700, color: '#aaa', fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1.2, marginBottom: 6 }}>
          Legend — {caption}
        </div>
        {groupPicker}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {swatch('low')}
          <svg width="100%" height={14} style={{ flex: 1, display: 'block' }} preserveAspectRatio="none">
            <defs>
              <linearGradient id="regLegGrad" x1="0" x2="1">
                <stop offset="0%" stopColor={colors.low} />
                <stop offset="100%" stopColor={colors.high} />
              </linearGradient>
            </defs>
            <rect width="100%" height={14} rx={2} fill="url(#regLegGrad)" stroke="#ddd" strokeWidth={0.5} />
          </svg>
          {swatch('high')}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#999', marginTop: 3 }}>
          <span>{range[0].toFixed(2)}</span>
          <span style={{ color: '#bbb' }}>{colorRangeMode === 'tree' ? 'per-tree' : 'forest'}</span>
          <span>{range[1].toFixed(2)}</span>
        </div>
        {!selectedId && (
          <div style={{ fontSize: 10, color: '#ccc', marginTop: 5, fontStyle: 'italic' }}>
            select a tree to see its range
          </div>
        )}
      </div>
    )
  }

  const classCount = labels.length
  const anyActive  = selectedClass !== null

  const openColorPicker = (e: React.MouseEvent, classIndex: number) => {
    e.stopPropagation()
    editingClassRef.current = classIndex
    const defaultColor = classColor(classIndex, classCount)
    const hexDefault   = customColors[classIndex] ?? toPickerHex(defaultColor)
    if (colorInputRef.current) {
      colorInputRef.current.value = hexDefault
      colorInputRef.current.click()
    }
  }

  return (
    <div style={{
      borderTop: '1px solid #e0e0e0',
      padding: '8px 10px 10px',
      background: '#fafafa',
      flexShrink: 0,
    }}>
      {/* Hidden color picker — shared across all rows */}
      <input
        ref={colorInputRef}
        type="color"
        onChange={e => {
          const idx = editingClassRef.current
          if (idx >= 0) setCustomColors(prev => ({
            ...prev,
            [group.index]: { ...(prev[group.index] ?? {}), [idx]: e.target.value },
          }))
        }}
        style={{ position: 'fixed', opacity: 0, pointerEvents: 'none', width: 0, height: 0 }}
      />

      <div style={{
        fontSize: 10, fontWeight: 700, color: '#aaa',
        fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1.2,
        marginBottom: 6,
      }}>
        Legend{groups.length > 1 && group.name ? ` — ${group.name}` : ''}
      </div>

      {groupPicker}

      {/* Height-capped so many labels can't squeeze the tree list/map above;
          scrolls when the cap is hit */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        maxHeight: 'min(240px, 30vh)',
        overflowY: 'auto',
        overscrollBehavior: 'contain',
      }}>
        {labels.map((label, i) => {
          const color    = customColors[i] ?? classColor(i, classCount)
          const isActive = selectedClass === i
          const isDimmed = anyActive && !isActive

          return (
            <div
              key={i}
              onClick={() => setSelectedClass(isActive ? null : i)}
              title={isActive ? 'Click to clear filter' : `Click to highlight majority-${label} nodes`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '2px 4px',
                borderRadius: 4,
                cursor: 'pointer',
                opacity: isDimmed ? 0.35 : 1,
                background: isActive ? `${color}18` : 'transparent',
                outline: isActive ? `1.5px solid ${color}` : '1.5px solid transparent',
                transition: 'opacity 0.15s, background 0.15s',
              }}
            >
              {/* Color swatch — click opens color picker, stopPropagation so it doesn't select */}
              <div
                title="Click to change color"
                onClick={e => openColorPicker(e, i)}
                style={{ flexShrink: 0, cursor: 'pointer', lineHeight: 0 }}
              >
                <svg width={16} height={16}>
                  <rect width={16} height={16} rx={3} fill={color} />
                  <rect width={16} height={16} rx={3} fill="none" stroke="#555" strokeWidth={1.5} />
                </svg>
              </div>

              <span style={{
                fontSize: 11,
                color: isActive ? '#222' : '#555',
                fontWeight: isActive ? 700 : 400,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {label}
              </span>

              {isActive && (
                <svg width={8} height={8} style={{ marginLeft: 'auto', flexShrink: 0 }}>
                  <polyline points="1,4 3,7 7,1" fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
          )
        })}
      </div>

      {anyActive && (
        <div style={{ fontSize: 10, color: '#aaa', marginTop: 6, fontStyle: 'italic' }}>
          click label to clear filter
        </div>
      )}

      {!selectedId && !anyActive && (
        <div style={{ fontSize: 10, color: '#ccc', marginTop: 5, fontStyle: 'italic' }}>
          select a tree to confirm class count
        </div>
      )}
    </div>
  )
}
