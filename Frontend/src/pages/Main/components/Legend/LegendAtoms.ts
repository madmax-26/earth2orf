import { atom } from 'jotai'
import { embeddedClassColorsAtom, activeGroupAtom } from '../Forest/ForestAtoms'

/** Index of the class being highlighted in the tree view (within the active
 *  label group), or null for no filter. Reset when the group changes. */
export const selectedClassAtom = atom<number | null>(null)

/**
 * Regression gradient endpoints (low → high), editable from the Legend.
 * `mean` colors the Mean-Prediction leaf view; `std` colors the Std-Deviation view.
 * Defaults: mean blue→red; std green→red (matches the classification purity scale).
 */
export type GradientColors = { low: string; high: string }
export const meanColorsAtom = atom<GradientColors>({ low: '#4575b4', high: '#d73027' })
export const stdColorsAtom  = atom<GradientColors>({ low: '#3a8c3a', high: '#c62828' })

/** User-overridden colors keyed by label-group ordinal, then class index within
 *  that group. Values are #rrggbb hex strings. */
export const customClassColorsAtom = atom<Record<number, Record<number, string>>>({})

/**
 * Merged color map for the ACTIVE label group: forest.json embedded colors as
 * base, user picks on top. Consumers stay group-agnostic — they receive a plain
 * classIndex → color map for whatever group is currently selected.
 * Resolution order: user pick > embedded color > classColor() palette fallback.
 */
export const effectiveClassColorsAtom = atom((get): Record<number, string> => {
  const embedded = get(embeddedClassColorsAtom)   // already active-group sliced
  const custom   = get(customClassColorsAtom)[get(activeGroupAtom).index] ?? {}
  const merged: Record<number, string> = {}
  embedded.forEach((color, i) => { if (color) merged[i] = color })
  Object.assign(merged, custom)
  return merged
})
