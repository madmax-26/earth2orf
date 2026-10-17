import { useAtom, useAtomValue } from 'jotai'
import { loadableForest } from '../Forest/ForestAtoms'
import { selectedTreeIdAtom, forestViewModeAtom } from './ForestViewAtoms'
import TreeThumbnail from '../TreeThumbnail/TreeThumbnail'
import ForestMap from '../ForestMap/ForestMap'
import { HEADING_FONT } from '../../../../theme'

const THUMB_SIZE = 64

export default function ForestView() {
  const forestState = useAtomValue(loadableForest)
  const [selectedId, setSelectedId] = useAtom(selectedTreeIdAtom)
  const [mode, setMode] = useAtom(forestViewModeAtom)

  const renderList = () => {
    if (forestState.state === 'loading') {
      return <div style={{ fontSize: 12, color: '#999', padding: 8 }}>Loading...</div>
    }
    if (forestState.state === 'hasError') {
      return <div style={{ fontSize: 12, color: 'red', padding: 8 }}>Error</div>
    }

    const entries = Object.entries(forestState.data.trees)
    return entries.map(([id, meta]) => {
      const isSelected = id === selectedId
      return (
        <div
          key={id}
          onClick={() => setSelectedId(isSelected ? null : id)}
          title={`Tree ${id} — height ${meta.height}, ${meta.n_nodes} nodes`}
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '6px 8px',
            borderRadius: 6,
            background: isSelected ? '#f0f7f0' : 'transparent',
            outline: isSelected ? '1.5px solid #3A8C3A' : '1.5px solid transparent',
            transition: 'background 0.15s, outline-color 0.15s',
          }}
        >
          <div style={{
            width: THUMB_SIZE,
            height: THUMB_SIZE,
            flexShrink: 0,
            borderRadius: 4,
            overflow: 'hidden',
            background: '#fafafa',
            border: '1px solid #e8e8e8',
          }}>
            <TreeThumbnail treeId={id} size={THUMB_SIZE} />
          </div>

          <div style={{ fontSize: 11, color: '#555', lineHeight: 1.7 }}>
            <div style={{ fontWeight: 700, color: isSelected ? '#3A8C3A' : '#333', marginBottom: 1 }}>
              Tree {id}
            </div>
            <div>h = {meta.height}</div>
            <div>n = {meta.n_nodes}</div>
          </div>
        </div>
      )
    })
  }

  return (
    <div style={{
      flex: 1,
      minHeight: 0,
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Tab strip */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid #eee',
        background: '#fafafa',
        flexShrink: 0,
      }}>
        {(['list', 'map'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setMode(tab)}
            style={{
              flex: 1,
              padding: '7px 0',
              fontSize: 10,
              fontWeight: mode === tab ? 700 : 400,
              color: mode === tab ? '#3A8C3A' : '#aaa',
              background: 'none',
              border: 'none',
              borderBottom: mode === tab ? '2px solid #3A8C3A' : '2px solid transparent',
              cursor: 'pointer',
              fontFamily: HEADING_FONT, textTransform: 'uppercase',
              letterSpacing: 1,
            }}
          >
            {tab === 'list' ? 'Forest' : 'Map'}
          </button>
        ))}
      </div>

      {/* List body */}
      {mode === 'list' && (
        <div style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          padding: '8px 4px',
          gap: 4,
        }}>
          {renderList()}
        </div>
      )}

      {/* Map scatter — fills remaining height of the panel */}
      {mode === 'map' && <ForestMap />}
    </div>
  )
}
