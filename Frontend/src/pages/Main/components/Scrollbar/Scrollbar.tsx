import { useEffect, useLayoutEffect, useRef } from 'react'
import { useAtom, useAtomValue, useSetAtom } from 'jotai'
import {
  availableIterationsAtom,
  iterationRefreshAtom,
  selectedIterationAtom,
} from '../Iteration/IterationAtoms'
import { selectedTreeIdAtom, forestViewModeAtom } from '../ForestView/ForestViewAtoms'
import { loadableForest } from '../Forest/ForestAtoms'
import Iteration from '../Iteration/Iteration'
import ForestView from '../ForestView/ForestView'
import Tree from '../Tree/Tree'
import Legend from '../Legend/Legend'
import MetricHub from '../MetricHub/MetricHub'
import Customizer from '../Customizer/Customizer'
import DownloadAll from './DownloadAll'
import AnimateButton from '../Animate/AnimateButton'
import { HEADING_FONT } from '../../../../theme'

const STRIP_H = 62
const LEFT_W_LIST = 160
const LEFT_W_MAP  = 290

const Scrollbar = () => {
  const iterations = useAtomValue(availableIterationsAtom)
  const [selectedIter, setSelectedIter] = useAtom(selectedIterationAtom)
  const triggerRefresh = useSetAtom(iterationRefreshAtom)
  const selectedId = useAtomValue(selectedTreeIdAtom)
  const forestViewMode = useAtomValue(forestViewModeAtom)
  const forestState = useAtomValue(loadableForest)

  // Poll the backend for new iterations every 2 s
  useEffect(() => {
    const id = setInterval(() => triggerRefresh(n => n + 1), 2000)
    return () => clearInterval(id)
  }, [triggerRefresh])

  // Auto-select the first iteration as soon as one arrives
  useEffect(() => {
    if (selectedIter === null && iterations.length > 0) {
      setSelectedIter(iterations[0])
    }
  }, [iterations, selectedIter, setSelectedIter])

  const selectedMeta =
    selectedId && forestState.state === 'hasData'
      ? forestState.data.trees[selectedId]
      : null

  const leftW = forestViewMode === 'map' ? LEFT_W_MAP : LEFT_W_LIST

  // Each backend poll can briefly re-render the pill list (e.g. a transient
  // empty result before it repopulates), which collapses the strip's content
  // width. Browsers clamp scrollLeft to fit when that happens and do not
  // restore it once content returns, so we track the last position ourselves
  // and re-apply it after every re-render of the pill list.
  const pillStripRef = useRef<HTMLDivElement>(null)
  const scrollLeftRef = useRef(0)
  useLayoutEffect(() => {
    const el = pillStripRef.current
    if (el) el.scrollLeft = scrollLeftRef.current
  }, [iterations])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>

      {/* Iteration strip */}
      <div
        style={{
          height: STRIP_H,
          minHeight: STRIP_H,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '0 12px 0 16px',
          borderBottom: '1px solid #e0e0e0',
          background: '#fafafa',
          overflow: 'hidden',
        }}
      >
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            color: '#aaa',
            fontFamily: HEADING_FONT, textTransform: 'uppercase',
            letterSpacing: 1.2,
            flexShrink: 0,
            marginRight: 6,
          }}
        >
          Iterations
        </span>

        {/* Scrollable pill area — stretched to full strip height with the bubbles
            pinned to the top, so the horizontal scrollbar has clear space below
            them and never overlaps the bubbles */}
        <div
          ref={pillStripRef}
          onScroll={e => { scrollLeftRef.current = e.currentTarget.scrollLeft }}
          style={{
            flex: 1,
            alignSelf: 'stretch',
            display: 'flex',
            alignItems: 'flex-start',
            paddingTop: 16,
            gap: 8,
            overflowX: 'auto',
            overflowY: 'hidden',
            minWidth: 0,
          }}>
          {iterations.length === 0 ? (
            <span style={{ fontSize: 12, color: '#ccc', fontStyle: 'italic' }}>
              Waiting for data…
            </span>
          ) : (
            iterations.map(idx => <Iteration key={idx} index={idx} />)
          )}
        </div>

        <AnimateButton iterations={iterations} />
        <DownloadAll iterations={iterations} />
      </div>

      {/* Main content area */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>

        {/* Left column — widens when Map tab is active */}
        <div style={{
          width: leftW,
          flexShrink: 0,
          borderRight: '1px solid #ddd',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          transition: 'width 0.18s ease',
        }}>
          <ForestView />
          <Customizer />
          <Legend />
        </div>

        {/* Centre: tree render or placeholder */}
        {selectedId && selectedMeta ? (
          <Tree treeId={selectedId} meta={selectedMeta} />
        ) : (
          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#bbb',
            fontSize: 14,
          }}>
            Select a tree from the forest panel
          </div>
        )}

        {/* Right: tabbed metric hub */}
        <MetricHub />

      </div>

    </div>
  )
}

export default Scrollbar
