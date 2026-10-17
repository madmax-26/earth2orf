import { classColor } from '../Main/components/classColors'
import { refImageUrl, type ReferenceMeta } from './NodeViewAtoms'

// One reference thumbnail: the (small) patch image scaled up with hard pixels,
// plus a class-color badge. Colors follow the Legend's overrides.
export default function RefThumb({
  r, size = 72, labels, colors, onClick, dimmed = false,
}: {
  r: ReferenceMeta
  size?: number
  labels: string[]
  colors: Record<number, string>
  onClick?: () => void
  /** Outside the active label-split filter — faded, same as the tree's
      highlight modes, so the selected references read as the focus. */
  dimmed?: boolean
}) {
  const badge = r.label_index !== null
    ? colors[r.label_index] ?? classColor(r.label_index, Math.max(labels.length, r.label_index + 1))
    : null
  const title = [
    r.name ?? r.id,
    r.label !== null ? `label: ${labels[r.label_index ?? -1] ?? r.label}` : null,
    r.step !== null ? `step ${r.step}` : null,
  ].filter(Boolean).join(' · ')

  return (
    <div
      onClick={onClick}
      title={title}
      style={{
        position: 'relative', width: size, flexShrink: 0,
        cursor: onClick ? 'pointer' : 'default',
        opacity: dimmed ? 0.2 : 1,
        transition: 'opacity 0.12s',
      }}
    >
      <img
        src={refImageUrl(r)}
        crossOrigin="anonymous"
        width={size}
        height={size}
        style={{
          display: 'block',
          imageRendering: 'pixelated',
          borderRadius: 4,
          border: '1px solid #ddd',
          background: '#f5f5f5',
        }}
      />
      {badge && (
        <span style={{
          position: 'absolute', right: 3, bottom: 3,
          width: 11, height: 11, borderRadius: 3,
          background: badge, border: '1px solid rgba(255,255,255,0.9)',
          boxShadow: '0 0 2px rgba(0,0,0,0.4)',
        }} />
      )}
      <div style={{
        fontSize: 9, color: '#888', marginTop: 2, textAlign: 'center',
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {r.name ?? r.id}
      </div>
    </div>
  )
}
