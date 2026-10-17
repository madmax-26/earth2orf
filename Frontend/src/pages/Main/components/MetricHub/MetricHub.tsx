import { useState } from 'react'
import { useAtomValue } from 'jotai'
import { selectedTreeIdAtom } from '../ForestView/ForestViewAtoms'
import ForestMetrics from '../ForestMetrics/ForestMetrics'
import TreeMetrics from '../TreeMetrics/TreeMetrics'
import Similarity from '../Analysis/Similarity'

type Tab = 'forest' | 'tree' | 'analysis'

const TABS: { id: Tab; label: string; title?: string }[] = [
  { id: 'forest',   label: 'Forest' },
  { id: 'tree',     label: 'Tree' },
  // Holds the similarity map AND the Decisive Node Analysis, hence the broader name.
  { id: 'analysis', label: 'Analysis', title: 'Similarity map & Decisive Node Analysis' },
]

const METRIC_HUB_W = 240

export default function MetricHub() {
  const [activeTab, setActiveTab] = useState<Tab>('forest')
  const selectedId = useAtomValue(selectedTreeIdAtom)

  return (
    <div style={{
      width: METRIC_HUB_W,
      flexShrink: 0,
      borderLeft: '1px solid #ddd',
      background: '#fafafa',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '2px solid #e8e8e8',
        background: '#fff',
        flexShrink: 0,
      }}>
        {TABS.map(tab => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              title={tab.title ?? tab.label}
              style={{
                flex: 1,
                minWidth: 0,
                padding: '7px 4px 6px',
                border: 'none',
                borderBottom: isActive ? '2px solid #3A8C3A' : '2px solid transparent',
                marginBottom: -2,
                background: 'transparent',
                fontSize: 10,
                fontWeight: isActive ? 700 : 400,
                color: isActive ? '#3A8C3A' : '#999',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                letterSpacing: 0.2,
                transition: 'color 0.12s, border-color 0.12s',
              }}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflowY: 'auto' }}>

        {activeTab === 'forest' && <ForestMetrics />}

        {activeTab === 'tree' && (
          selectedId
            ? <TreeMetrics treeId={selectedId} />
            : (
              <div style={{
                padding: '32px 16px',
                textAlign: 'center',
                color: '#ccc',
                fontSize: 12,
              }}>
                Select a tree to view its metrics
              </div>
            )
        )}

        {activeTab === 'analysis' && <Similarity />}

      </div>
    </div>
  )
}
