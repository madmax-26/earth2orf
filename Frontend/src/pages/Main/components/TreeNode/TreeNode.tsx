import { memo, useMemo, type MouseEvent as ReactMouseEvent } from 'react'
import * as d3 from 'd3'
import { useAtomValue } from 'jotai'
import { activeGroupAtom, sliceValue, impurityAt, type TreeNodeData } from '../Forest/ForestAtoms'
import { classColor } from '../classColors'
import { leafModeAtom } from '../Customizer/CustomizerAtoms'
import { effectiveClassColorsAtom } from '../Legend/LegendAtoms'

function purityColor(impurity: number): string {
  const t = Math.max(0, Math.min(1, impurity))
  const GREEN = [58, 140, 58]
  const RED   = [198, 40, 40]
  const r = Math.round(GREEN[0] + t * (RED[0] - GREEN[0]))
  const g = Math.round(GREEN[1] + t * (RED[1] - GREEN[1]))
  const b = Math.round(GREEN[2] + t * (RED[2] - GREEN[2]))
  return `rgb(${r},${g},${b})`
}

export type TreeNodeProps = {
  data: TreeNodeData
  x: number
  y: number
  // BASE radius (ink-free). Zoom damping is applied via inkScale so this prop —
  // and therefore the pie-arc memo — stays stable while zooming.
  radius: number
  // Zoom-damped ink factor from Tree.tsx; applied as an outer scale() so a
  // change only touches the group transform, not the arc geometry.
  inkScale?: number
  isCollapsed?: boolean
  highlighted?: boolean
  onClick?: (e: ReactMouseEvent<SVGGElement>) => void
  onDoubleClick?: (e: ReactMouseEvent<SVGGElement>) => void
  // Regression: precomputed solid-fill color for this leaf. When set, the node
  // renders as a single filled circle (overrides the class pie / purity views).
  regressionColor?: string
}

function TreeNodeImpl({ data, x, y, radius, inkScale = 1, isCollapsed = false, highlighted = false, onClick, onDoubleClick, regressionColor }: TreeNodeProps) {
  const leafMode    = useAtomValue(leafModeAtom)
  const customColors = useAtomValue(effectiveClassColorsAtom)
  const group        = useAtomValue(activeGroupAtom)

  const slices = useMemo(() => {
    if (leafMode !== 'pie') return []
    const value = sliceValue(data.value, group)
    const total = value.reduce((a, b) => a + b, 0)
    if (total === 0) return []
    const pie = d3.pie<number>().sort(null)
    const arcGen = d3.arc().innerRadius(0).outerRadius(radius)
    return pie(value).map((slice) => ({
      path: arcGen(slice as unknown as d3.DefaultArcObject) ?? '',
      color: customColors[slice.index] ?? classColor(slice.index, value.length),
    }))
  }, [data.value, radius, leafMode, customColors, group])

  return (
    <g
      transform={`translate(${x},${y}) scale(${inkScale})`}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      {regressionColor !== undefined ? (
        <circle r={radius} fill={regressionColor} />
      ) : leafMode === 'purity' ? (
        <circle r={radius} fill={purityColor(impurityAt(data.impurity, group.index))} />
      ) : slices.length === 0 ? (
        <circle r={radius} fill="#ddd" />
      ) : (
        slices.map((s, i) => (
          <path key={i} d={s.path} fill={s.color} />
        ))
      )}
      {highlighted && (
        <circle r={radius + 1} fill="none" stroke="#FFB300" strokeWidth={1.5} opacity={0.9} />
      )}
      <circle r={radius} fill="none" stroke="#333" strokeWidth={0.1} />
      {isCollapsed && (
        <circle
          r={radius + 1}
          fill="none"
          stroke="#555"
          strokeWidth={0.5}
          strokeDasharray="1 1.5"
          opacity={0.8}
        />
      )}
    </g>
  )
}

// Memoized: during zoom only inkScale changes (a cheap group-transform update);
// hover/ctrl re-renders skip every untouched leaf entirely.
const TreeNode = memo(TreeNodeImpl)
export default TreeNode
