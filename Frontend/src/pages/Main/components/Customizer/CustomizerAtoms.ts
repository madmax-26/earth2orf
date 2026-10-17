import { atom } from 'jotai'

export type LeafMode = 'pie' | 'purity'
export const leafModeAtom = atom<LeafMode>('pie')

// Regression only: whether leaf colors are normalized to each tree's own
// value/std range ('tree') or the forest-wide range ('forest').
export type ColorRangeMode = 'tree' | 'forest'
export const colorRangeModeAtom = atom<ColorRangeMode>('forest')

// Visual leaf-size multiplier (Tree view slider). Applied as one factor on top
// of the √samples radius, so relative sizes (area ∝ samples), zoom damping and
// culling all keep working unchanged. 1 = the default size.
export const LEAF_SCALE_MIN = 0.25
export const LEAF_SCALE_MAX = 1.2
export const leafScaleAtom = atom(1)
