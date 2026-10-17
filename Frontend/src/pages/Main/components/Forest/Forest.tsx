import { useAtomValue } from 'jotai'
import { loadableForest } from './ForestAtoms'
import Tree from '../Tree/Tree'

export default function Forest() {
  const state = useAtomValue(loadableForest)

  if (state.state === 'loading') return <div>Loading forest...</div>
  if (state.state === 'hasError') return <div>Error: {String(state.error)}</div>

  return (
    <div style={{ display: 'flex', gap: 16, padding: 16, overflow: 'auto' }}>
      {Object.entries(state.data.trees).map(([id, meta]) => (
        <Tree key={id} treeId={id} meta={meta} />
      ))}
    </div>
  )
}
