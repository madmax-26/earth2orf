import { apiBase } from '../../../../apiConfig'
import { atom } from 'jotai'
import { unwrap } from 'jotai/utils'


// Increment to trigger a re-fetch of /iterations (driven by Scrollbar's interval)
export const iterationRefreshAtom = atom(0)

const asyncAvailableIterationsAtom = atom(async (get): Promise<number[]> => {
  get(iterationRefreshAtom)
  try {
    const res = await fetch(`${apiBase()}/iterations`)
    if (!res.ok) return []
    return res.json()
  } catch {
    return []
  }
})

// Sync with [] fallback so consumers can use useAtomValue without suspense.
// Keeps showing the last resolved list while a refetch is in flight — falling
// back to [] on every poll would flash-unmount the iteration pills and reset
// the strip's scroll position each refresh.
export const availableIterationsAtom = unwrap(asyncAvailableIterationsAtom, (prev) => prev ?? [])

export const selectedIterationAtom = atom<number | null>(null)
