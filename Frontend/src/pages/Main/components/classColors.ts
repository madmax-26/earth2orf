// Curated palette — visually distinct across a wide range of backgrounds
const BASE = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
  '#b07aa1', '#ff9da7', '#9c755f', '#499894', '#86bcb6',
  '#bab0ac', '#d4a6c8',
]

/**
 * Color for class index `i` when there are `total` classes.
 * Uses the curated palette for up to 12 classes so they look great out of
 * the box; above 12 it distributes hues evenly around the color wheel with
 * alternating lightness to maximise perceptual separation.
 */
export function classColor(i: number, total: number): string {
  if (total <= BASE.length) return BASE[i % BASE.length]
  const hue = (i / total) * 360
  const lightness = 45 + (i % 2) * 10   // alternates 45 % / 55 %
  return `hsl(${hue.toFixed(1)}, 65%, ${lightness}%)`
}

// Number of hand-picked colors — useful as a fallback count when the real
// class count is not yet known (e.g. before the first tree loads).
export const BASE_CLASS_COUNT = BASE.length
