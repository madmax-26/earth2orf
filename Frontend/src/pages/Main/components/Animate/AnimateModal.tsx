import { apiBase } from '../../../../apiConfig'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useAtomValue } from 'jotai'
import { GIFEncoder, quantize, applyPalette } from 'gifenc'
import {
  loadableForest, selectedGroupAtom, groupsFromForest, pickRange,
  type TreeNodeData, type ForestData, type GroupSpec,
} from '../Forest/ForestAtoms'
import { effectiveClassColorsAtom, meanColorsAtom, stdColorsAtom } from '../Legend/LegendAtoms'
import { leafModeAtom } from '../Customizer/CustomizerAtoms'
import { treeToSVGString } from '../Tree/treeLayout'
import { treeRanges } from '../Tree/regressionColor'
import TreeThumbnail from '../TreeThumbnail/TreeThumbnail'
import { HEADING_FONT } from '../../../../theme'

const THUMB = 48

// Fixed GIF canvas — each frame's own viewBox scales to fit (preserveAspectRatio
// meet), matching the on-screen per-frame refit, so the tree grows in place.
const GIF_W = 720
const GIF_H = 560

// Rasterize one standalone tree-SVG string to RGBA pixels at a fixed size.
// treeToSVGString emits pure SVG (no <foreignObject>/external refs), so the
// canvas stays untainted and getImageData works. White background fills the
// letterbox around the aspect-fit content.
async function svgToRGBA(svgStr: string, w: number, h: number): Promise<Uint8ClampedArray> {
  const sized = svgStr
    .replace(/^<\?xml[^>]*\?>\s*/, '')
    .replace('<svg ', `<svg width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet" `)
  const url = URL.createObjectURL(new Blob([sized], { type: 'image/svg+xml' }))
  try {
    const img = new Image()
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve()
      img.onerror = () => reject(new Error('SVG frame failed to load'))
      img.src = url
    })
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('no 2d canvas context')
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, w, h)
    ctx.drawImage(img, 0, 0, w, h)
    return ctx.getImageData(0, 0, w, h).data
  } finally {
    URL.revokeObjectURL(url)
  }
}

// One animation frame: the tree's nodes plus ITS OWN iteration meta. Task /
// label groups / ranges are resolved per iteration, not from the currently
// selected one — a directory can mix classification and regression iterations.
type Frame = { nodes: TreeNodeData[] | null; meta: ForestData | null }

// Fetched frames co-located with their tree id so a slow fetch for a previously
// selected tree can never overwrite the current one.
type Frames = { treeId: string; byIter: Map<number, Frame> }

const FALLBACK_GROUP: GroupSpec = {
  name: '', labels: [], offset: 0, size: -1, index: 0,
  kind: 'classification', regIndex: -1,
}

export type AnimateModalProps = {
  iterations: number[]
  onClose: () => void
}

export default function AnimateModal({ iterations, onClose }: AnimateModalProps) {
  const forestState = useAtomValue(loadableForest)
  const leafMode = useAtomValue(leafModeAtom)
  const customColors = useAtomValue(effectiveClassColorsAtom)
  const meanColors = useAtomValue(meanColorsAtom)
  const stdColors = useAtomValue(stdColorsAtom)
  const selectedGroup = useAtomValue(selectedGroupAtom)

  const [selectedTree, setSelectedTree] = useState<string | null>(null)
  const [frames, setFrames] = useState<Frames | null>(null)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [intervalMs, setIntervalMs] = useState(500)
  const [gifStatus, setGifStatus] = useState<'idle' | 'rendering'>('idle')
  const selectedRef = useRef<string | null>(null)
  // Iteration metas don't depend on the tree — cache them across selections
  const metaCache = useRef(new Map<number, ForestData | null>())

  // Escape closes the player
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Playback: advance one frame per tick until the last iteration, then stop.
  useEffect(() => {
    if (!playing) return
    if (frameIdx >= iterations.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrameIdx(i => i + 1), Math.max(50, intervalMs))
    return () => clearTimeout(t)
  }, [playing, frameIdx, iterations.length, intervalMs])

  const selectTree = (treeId: string) => {
    if (treeId === selectedTree) return
    setSelectedTree(treeId)
    selectedRef.current = treeId
    setFrames(null)
    setFrameIdx(0)
    setPlaying(false)
    // Prefetch this tree's data AND each iteration's meta in parallel — the
    // meta drives per-frame task/group/range resolution for mixed directories.
    Promise.all(
      iterations.map(async (iter): Promise<[number, Frame]> => {
        const metaP: Promise<ForestData | null> = metaCache.current.has(iter)
          ? Promise.resolve(metaCache.current.get(iter) ?? null)
          : fetch(`${apiBase()}/iteration/${iter}/meta`)
              .then(r => (r.ok ? (r.json() as Promise<ForestData>) : null))
              .catch(() => null)
              .then(meta => { metaCache.current.set(iter, meta); return meta })
        const nodesP = fetch(`${apiBase()}/iteration/${iter}/tree/${treeId}`)
          .then(r => (r.ok ? (r.json() as Promise<TreeNodeData[]>) : null))
          .catch(() => null)
        const [meta, nodes] = await Promise.all([metaP, nodesP])
        return [iter, { meta, nodes: nodes && nodes.length ? nodes : null }]
      })
    ).then(entries => {
      if (selectedRef.current !== treeId) return  // stale fetch — a newer tree was picked
      setFrames({ treeId, byIter: new Map(entries) })
    })
  }

  const framesReady = frames !== null && frames.treeId === selectedTree
  const currentIter = iterations[Math.min(frameIdx, Math.max(0, iterations.length - 1))]
  const currentFrame = framesReady ? frames.byIter.get(currentIter) ?? null : null

  // Build one frame's standalone SVG via the shared export pipeline — identical
  // styling to the live render (leaf mode, group slicing, custom colors,
  // regression gradients). Task + group + ranges come from THAT frame's own
  // iteration meta, so regression iterations color correctly even when a
  // classification one is selected. Used by both the live view and the GIF.
  const buildFrameSvg = useCallback((frame: Frame | null | undefined): string | null => {
    if (!frame?.nodes) return null
    const { nodes, meta } = frame
    const groups = meta ? groupsFromForest(meta) : []
    const g = groups.length
      ? groups[Math.max(0, Math.min(selectedGroup, groups.length - 1))]
      : FALLBACK_GROUP
    const reg = g.kind === 'regression' && meta
      ? {
          ranges: {
            // Iteration-wide range keeps colors stable across frames; fall back
            // to the frame's own node ranges when the meta doesn't carry any
            mean: pickRange(meta.target_range, g.regIndex) ?? treeRanges(nodes, g).mean,
            std: pickRange(meta.std_range, g.regIndex) ?? treeRanges(nodes, g).std,
          },
          colors: { mean: meanColors, std: stdColors },
        }
      : undefined
    return treeToSVGString(nodes, customColors, new Set(), leafMode, reg, g) || null
  }, [selectedGroup, meanColors, stdColors, customColors, leafMode])

  const svgMarkup = useMemo(() => {
    const svg = buildFrameSvg(currentFrame)
    if (!svg) return null
    // Inline into the page: drop the XML declaration, make the svg fill its box
    return svg
      .replace(/^<\?xml[^>]*\?>\s*/, '')
      .replace('<svg ', '<svg width="100%" height="100%" style="display:block" ')
  }, [buildFrameSvg, currentFrame])

  const trees = forestState.state === 'hasData' ? Object.entries(forestState.data.trees) : []

  const onPlay = () => {
    if (!framesReady || iterations.length === 0) return
    setFrameIdx(0)
    setPlaying(true)
  }

  // Encode the animation to an (infinitely looping) GIF: rasterize each
  // iteration's frame to the fixed GIF canvas, quantize per frame, and write it
  // with the same inter-frame delay the player uses. Iterations where the tree
  // has no data are skipped (a blank frame would read as a glitch).
  const exportGif = useCallback(async () => {
    if (!framesReady || !frames || gifStatus !== 'idle') return
    setPlaying(false)
    setGifStatus('rendering')
    try {
      const gif = GIFEncoder()
      let count = 0
      for (const iter of iterations) {
        const svg = buildFrameSvg(frames.byIter.get(iter))
        if (!svg) continue
        const rgba = await svgToRGBA(svg, GIF_W, GIF_H)
        const palette = quantize(rgba, 256)
        const index = applyPalette(rgba, palette)
        gif.writeFrame(index, GIF_W, GIF_H, { palette, delay: Math.max(50, intervalMs) })
        count++
      }
      if (count === 0) return
      gif.finish()
      const url = URL.createObjectURL(new Blob([gif.bytes()], { type: 'image/gif' }))
      const a = document.createElement('a')
      a.href = url
      a.download = `tree-${frames.treeId}-animation.gif`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('GIF export failed:', err)
    } finally {
      setGifStatus('idle')
    }
  }, [framesReady, frames, gifStatus, iterations, buildFrameSvg, intervalMs])

  const modal = (
    <div style={{
      position: 'fixed', inset: 0,
      // Above the MUI AppBar (z-index 1100) — otherwise the status bar draws
      // over the top strip of the player, hiding the ✕ and the first tree row
      background: '#fff', zIndex: 2000,
      display: 'flex',
    }}>
      {/* ── Left column: tree list + player controls ── */}
      <div style={{
        width: 210, flexShrink: 0,
        borderRight: '1px solid #ddd',
        display: 'flex', flexDirection: 'column',
        background: '#fcfcfc',
      }}>
        <div style={{
          fontSize: 10, fontWeight: 700, color: '#888',
          fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1.2,
          padding: '14px 14px 10px', borderBottom: '1px solid #eee',
        }}>
          Animate — Trees
        </div>

        {/* Tree list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '10px 6px', display: 'flex', flexDirection: 'column', gap: 4 }}>
          {trees.length === 0 && (
            <span style={{ fontSize: 11, color: '#bbb', padding: 8 }}>No trees available</span>
          )}
          {trees.map(([id, meta]) => {
            const isSelected = id === selectedTree
            return (
              <div
                key={id}
                onClick={() => selectTree(id)}
                title={`Tree ${id} — height ${meta.height}, ${meta.n_nodes} nodes`}
                style={{
                  cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '5px 8px', borderRadius: 6,
                  background: isSelected ? '#f0f7f0' : 'transparent',
                  outline: isSelected ? '1.5px solid #3A8C3A' : '1.5px solid transparent',
                  transition: 'background 0.15s, outline-color 0.15s',
                }}
              >
                <div style={{
                  width: THUMB, height: THUMB, flexShrink: 0,
                  borderRadius: 4, overflow: 'hidden',
                  background: '#fafafa', border: '1px solid #e8e8e8',
                }}>
                  <TreeThumbnail treeId={id} size={THUMB} />
                </div>
                <div style={{
                  fontSize: 11, fontWeight: isSelected ? 700 : 400,
                  color: isSelected ? '#3A8C3A' : '#555',
                }}>
                  Tree {id}
                </div>
              </div>
            )
          })}
        </div>

        {/* Player controls — bottom left */}
        <div style={{
          borderTop: '1px solid #e0e0e0',
          padding: '10px 12px',
          display: 'flex', flexDirection: 'column', gap: 8,
          background: '#fafafa',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#888' }}>
              <input
                type="number"
                min={50}
                step={50}
                value={intervalMs}
                onChange={e => setIntervalMs(Math.max(50, Number(e.target.value) || 50))}
                title="Pause between frames (ms)"
                style={{
                  width: 60, fontSize: 11, padding: '3px 4px',
                  border: '1px solid #ddd', borderRadius: 4,
                  color: '#555', outline: 'none',
                }}
              />
              ms
            </label>
            <button
              onClick={onPlay}
              disabled={!framesReady || iterations.length === 0}
              title={playing ? 'Restart animation' : 'Play from first iteration'}
              style={{
                flex: 1,
                padding: '5px 10px',
                fontSize: 12, fontWeight: 600,
                border: '1px solid',
                borderColor: framesReady ? '#3A8C3A' : '#ddd',
                borderRadius: 4,
                background: framesReady ? '#3A8C3A' : '#f5f5f5',
                color: framesReady ? '#fff' : '#aaa',
                cursor: framesReady ? 'pointer' : 'default',
              }}
            >
              {playing ? '↻ Restart' : '▶ Play'}
            </button>
          </div>
          <button
            onClick={exportGif}
            disabled={!framesReady || gifStatus !== 'idle'}
            title="Render the animation to a looping GIF using the current frame delay"
            style={{
              width: '100%',
              padding: '5px 10px',
              fontSize: 11, fontWeight: 600,
              border: '1px solid #ccc',
              borderRadius: 4,
              background: (framesReady && gifStatus === 'idle') ? '#fff' : '#f5f5f5',
              color: (framesReady && gifStatus === 'idle') ? '#555' : '#aaa',
              cursor: (framesReady && gifStatus === 'idle') ? 'pointer' : 'default',
            }}
          >
            {gifStatus === 'rendering' ? 'Rendering GIF…' : '⬇ Save as GIF'}
          </button>
        </div>
      </div>

      {/* ── Main pane: frame render ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Caption */}
        <div style={{
          padding: '12px 20px',
          fontSize: 12, color: '#666',
          display: 'flex', alignItems: 'center', gap: 14,
          borderBottom: '1px solid #f0f0f0',
        }}>
          {selectedTree === null ? (
            <span style={{ color: '#bbb' }}>Select a tree to animate</span>
          ) : (
            <>
              <span style={{ fontWeight: 700, color: '#333' }}>Tree {selectedTree}</span>
              <span>
                Iteration <strong>{currentIter ?? '–'}</strong>
                {' '}({Math.min(frameIdx + 1, iterations.length)} / {iterations.length})
              </span>
              {playing && <span style={{ color: '#3A8C3A', fontWeight: 600 }}>playing…</span>}
            </>
          )}
        </div>

        {/* Frame */}
        <div style={{ flex: 1, minHeight: 0, padding: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {selectedTree === null ? (
            <span style={{ fontSize: 14, color: '#ccc' }}>← Pick a tree from the list</span>
          ) : !framesReady ? (
            <span style={{ fontSize: 13, color: '#bbb' }}>Loading iterations…</span>
          ) : svgMarkup ? (
            <div
              style={{ width: '100%', height: '100%' }}
              dangerouslySetInnerHTML={{ __html: svgMarkup }}
            />
          ) : (
            <span style={{ fontSize: 13, color: '#bbb' }}>
              No data for tree {selectedTree} in iteration {currentIter}
            </span>
          )}
        </div>
      </div>

      {/* Close */}
      <button
        onClick={onClose}
        title="Close (Esc)"
        style={{
          position: 'absolute', top: 10, right: 14,
          border: 'none', background: 'none',
          fontSize: 26, lineHeight: 1,
          color: '#aaa', cursor: 'pointer', padding: 4,
        }}
      >
        ×
      </button>
    </div>
  )

  return createPortal(modal, document.body)
}
