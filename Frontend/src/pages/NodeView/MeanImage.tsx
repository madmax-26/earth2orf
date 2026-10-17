import { useEffect, useRef, useState } from 'react'

// Client-side average of a set of reference images: every image is drawn onto
// a common canvas size, RGBA-accumulated, divided, and put back. Averaging is
// deliberately NOT precomputed on the backend — it's cheap here and keeps the
// stored data inspectable per reference. Images are fetched as blobs (the CORS
// middleware covers the image route), so the canvas never taints.
export default function MeanImage({ urls, size = 96 }: { urls: string[]; size?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'empty'>('loading')
  const key = urls.join('|')

  useEffect(() => {
    let cancelled = false
    setState('loading')
    ;(async () => {
      const acc = new Float64Array(size * size * 4)
      let n = 0
      for (const url of urls) {
        try {
          const blob = await (await fetch(url)).blob()
          const bmp = await createImageBitmap(blob)
          const tmp = document.createElement('canvas')
          tmp.width = size
          tmp.height = size
          const ctx = tmp.getContext('2d')!
          ctx.imageSmoothingEnabled = false
          ctx.drawImage(bmp, 0, 0, size, size)
          const data = ctx.getImageData(0, 0, size, size).data
          for (let i = 0; i < acc.length; i++) acc[i] += data[i]
          n++
        } catch {
          // unreachable image → simply not part of the mean
        }
      }
      if (cancelled) return
      if (n === 0) { setState('empty'); return }
      const out = new Uint8ClampedArray(acc.length)
      for (let i = 0; i < acc.length; i++) out[i] = acc[i] / n
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = size
      canvas.height = size
      canvas.getContext('2d')!.putImageData(new ImageData(out, size, size), 0, 0)
      setState('ready')
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, size])

  if (urls.length === 0 || state === 'empty') return null
  return (
    <div style={{ position: 'relative', width: size }}>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        style={{
          display: 'block',
          imageRendering: 'pixelated',
          borderRadius: 4,
          border: '1px solid #ddd',
          background: '#f5f5f5',
          opacity: state === 'ready' ? 1 : 0.3,
        }}
      />
      <div style={{ fontSize: 9, color: '#888', marginTop: 2, textAlign: 'center' }}>
        {state === 'ready' ? `mean of ${urls.length}` : 'averaging…'}
      </div>
    </div>
  )
}
