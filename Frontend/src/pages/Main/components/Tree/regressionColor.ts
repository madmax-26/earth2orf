import * as d3 from 'd3'
import type { LeafMode } from '../Customizer/CustomizerAtoms'
import type { GradientColors } from '../Legend/LegendAtoms'
import { impurityAt, type GroupSpec } from '../Forest/ForestAtoms'

export type ColorRanges = { mean: [number, number]; std: [number, number] }
export type LeafColors  = { mean: GradientColors; std: GradientColors }

function norm(v: number, lo: number, hi: number): number {
  if (!(hi > lo)) return 0.5
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)))
}

/**
 * Build the per-leaf color function for a regression tree. In 'purity' mode the
 * leaf is colored by its std (impurity) over the std range; otherwise by its mean
 * prediction (value[0]) over the mean range. Endpoints come from the (editable)
 * gradient colors. Build once per render and apply per leaf.
 */
export function makeRegressionLeafColor(
  leafMode: LeafMode,
  ranges: ColorRanges,
  colors: LeafColors,
): (mean: number, std: number) => string {
  if (leafMode === 'purity') {
    const interp = d3.interpolateRgb(colors.std.low, colors.std.high)
    const [lo, hi] = ranges.std
    return (_mean, std) => interp(norm(std, lo, hi))
  }
  const interp = d3.interpolateRgb(colors.mean.low, colors.mean.high)
  const [lo, hi] = ranges.mean
  return (mean) => interp(norm(mean, lo, hi))
}

/** Per-tree [min,max] of leaf means and [0,max] of node stds for the active
 *  regression target: value[group.offset] is the mean, impurity[group.index]
 *  the std (both scalar in the single-target case). */
export function treeRanges(
  nodes: { value: number[]; impurity: number | number[] }[],
  group?: GroupSpec,
): ColorRanges {
  const off = group?.offset ?? 0
  const gi  = group?.index ?? 0
  let mlo = Infinity, mhi = -Infinity, shi = 0
  for (const n of nodes) {
    const m = n.value[off] ?? 0
    if (m < mlo) mlo = m
    if (m > mhi) mhi = m
    const s = impurityAt(n.impurity, gi)
    if (s > shi) shi = s
  }
  if (!isFinite(mlo)) { mlo = 0; mhi = 1 }
  if (mhi <= mlo) mhi = mlo + 1
  return { mean: [mlo, mhi], std: [0, shi || 1] }
}
