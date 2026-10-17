import { apiBase } from '../../../../apiConfig'
import { useState } from 'react'
import { useAtomValue } from 'jotai'
import { zipSync, strToU8 } from 'fflate'
import {
  groupsFromForest, pickRange, selectedGroupAtom,
  type TreeNodeData, type ForestData, type GroupSpec,
} from '../Forest/ForestAtoms'
import { treeToSVGString } from '../Tree/treeLayout'
import { treeRanges } from '../Tree/regressionColor'
import { leafModeAtom } from '../Customizer/CustomizerAtoms'
import { meanColorsAtom, stdColorsAtom } from '../Legend/LegendAtoms'


type IterMeta = {
  iter: number
  treeIds: string[]
  group: GroupSpec                        // this iteration's spec for the selected group
  isReg: boolean
  meanRange: [number, number] | null      // iteration-wide ranges (null → derive per tree)
  stdRange: [number, number] | null
}

type Props = { iterations: number[] }

export default function DownloadAll({ iterations }: Props) {
  const [status, setStatus] = useState<'idle' | 'fetching' | 'packing'>('idle')
  const leafMode = useAtomValue(leafModeAtom)
  const meanColors = useAtomValue(meanColorsAtom)
  const stdColors = useAtomValue(stdColorsAtom)
  const selectedGroup = useAtomValue(selectedGroupAtom)

  const handleClick = async () => {
    if (status !== 'idle' || iterations.length === 0) return
    try {
      setStatus('fetching')

      // Fetch all forest metas in parallel: tree IDs, the selected label group
      // resolved against each iteration's own group layout (offsets can differ),
      // and per-iteration regression ranges for the active target.
      const metas: IterMeta[] = await Promise.all(
        iterations.map(iter =>
          fetch(`${apiBase()}/iteration/${iter}/meta`)
            .then(r => { if (!r.ok) throw new Error(`meta ${iter}: ${r.status}`); return r.json() })
            .then((data: ForestData): IterMeta => {
              const groups = groupsFromForest(data)
              const group = groups[Math.max(0, Math.min(selectedGroup, groups.length - 1))]
                ?? { name: '', labels: [], offset: 0, size: -1, index: 0, kind: 'classification' as const, regIndex: -1 }
              return {
                iter,
                treeIds: Object.keys(data.trees),
                group,
                // Per-GROUP kind, not forest task: in a mixed iteration only the
                // regression groups render with the gradient pipeline.
                isReg: group.kind === 'regression',
                meanRange: pickRange(data.target_range, group.regIndex),
                stdRange: pickRange(data.std_range, group.regIndex),
              }
            })
        )
      )
      const metaByIter = new Map(metas.map(m => [m.iter, m]))

      // Fetch all tree node data in parallel
      const treeJobs = metas.flatMap(({ iter, treeIds }) =>
        treeIds.map(treeId =>
          fetch(`${apiBase()}/iteration/${iter}/tree/${treeId}`)
            .then(r => { if (!r.ok) throw new Error(`tree ${iter}/${treeId}: ${r.status}`); return r.json() as Promise<TreeNodeData[]> })
            .then(nodes => ({ iter, treeId, nodes }))
        )
      )
      const trees = await Promise.all(treeJobs)

      setStatus('packing')

      // Generate SVGs and pack into ZIP
      const files: Record<string, Uint8Array> = {}
      for (const { iter, treeId, nodes } of trees) {
        const m = metaByIter.get(iter)
        // Iteration-wide ranges when the meta carries them (stable colors across
        // the iteration's trees); otherwise derive from this tree's own nodes
        const reg = m?.isReg
          ? {
              ranges: {
                mean: m.meanRange ?? treeRanges(nodes, m.group).mean,
                std: m.stdRange ?? treeRanges(nodes, m.group).std,
              },
              colors: { mean: meanColors, std: stdColors },
            }
          : undefined
        const svg = treeToSVGString(nodes, {}, new Set(), leafMode, reg, m?.group)
        files[`Iteration${iter}/Tree${treeId}.svg`] = strToU8(svg)
      }

      const zipped = zipSync(files, { level: 1 })
      const blob = new Blob([zipped], { type: 'application/zip' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'trees.zip'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Download All failed:', err)
    } finally {
      setStatus('idle')
    }
  }

  const label =
    status === 'fetching' ? 'Fetching…' :
    status === 'packing'  ? 'Packing…' :
    '⬇ All'

  return (
    <button
      onClick={handleClick}
      disabled={status !== 'idle' || iterations.length === 0}
      title={`Download all trees from ${iterations.length} iteration(s) as SVG ZIP`}
      style={{
        flexShrink: 0,
        padding: '4px 9px',
        fontSize: 10,
        fontWeight: 600,
        border: '1px solid #ccc',
        borderRadius: 4,
        background: status !== 'idle' ? '#f5f5f5' : '#fff',
        color: status !== 'idle' ? '#aaa' : '#555',
        cursor: status !== 'idle' ? 'default' : 'pointer',
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </button>
  )
}
