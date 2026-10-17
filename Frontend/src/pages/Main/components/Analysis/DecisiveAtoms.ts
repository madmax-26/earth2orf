import { atom } from 'jotai'

/**
 * Decisive Node Analysis selection, set from the Analysis tab and read by the
 * Tree render (same cross-panel pattern as selectedClassAtom). labelA/labelB are
 * class indices within the ACTIVE label group; threshold is the A-vs-B purity
 * cutoff (fraction, 0.5–1.0). Highlighting is active only when both labels are
 * chosen and distinct.
 */
export type DecisiveParams = {
  labelA: number | null
  labelB: number | null
  threshold: number
}

export const decisiveAtom = atom<DecisiveParams>({
  labelA: null,
  labelB: null,
  threshold: 0.9,
})
