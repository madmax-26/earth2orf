// Minimal typings for gifenc (ships no .d.ts). Covers only the API we use.
declare module 'gifenc' {
  export type Palette = number[][]  // [r,g,b] or [r,g,b,a] entries

  export interface WriteFrameOpts {
    palette?: Palette
    delay?: number        // milliseconds (converted to centiseconds internally)
    repeat?: number       // 0 = loop forever (default)
    transparent?: boolean
    transparentIndex?: number
    dispose?: number
    first?: boolean
    colorDepth?: number
  }

  export interface GifEncoderInstance {
    writeFrame(index: Uint8Array, width: number, height: number, opts?: WriteFrameOpts): void
    finish(): void
    // .slice() internally → backed by a plain ArrayBuffer (BlobPart-compatible)
    bytes(): Uint8Array<ArrayBuffer>
    bytesView(): Uint8Array
    reset(): void
  }

  export function GIFEncoder(opts?: { auto?: boolean; initialCapacity?: number }): GifEncoderInstance

  export function quantize(
    rgba: Uint8Array | Uint8ClampedArray,
    maxColors: number,
    opts?: { format?: 'rgb565' | 'rgb444' | 'rgba4444'; oneBitAlpha?: boolean | number; clearAlpha?: boolean },
  ): Palette

  export function applyPalette(
    rgba: Uint8Array | Uint8ClampedArray,
    palette: Palette,
    format?: 'rgb565' | 'rgb444' | 'rgba4444',
  ): Uint8Array
}
