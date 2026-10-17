/** Serializes a live SVG element to a standalone, downloadable .svg file.
 * Inlines an explicit white background (matching the page) and concrete
 * width/height so the file renders correctly in any standalone viewer. */
export function downloadSvgElement(svg: SVGSVGElement | null, filename: string): void {
  if (!svg) return
  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

  const vb = clone.viewBox.baseVal
  const w = vb && vb.width ? vb.width : clone.width.baseVal.value
  const h = vb && vb.height ? vb.height : clone.height.baseVal.value
  const x = vb && vb.width ? vb.x : 0
  const y = vb && vb.height ? vb.y : 0

  if (w && h) {
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    bg.setAttribute('x', String(x))
    bg.setAttribute('y', String(y))
    bg.setAttribute('width', String(w))
    bg.setAttribute('height', String(h))
    bg.setAttribute('fill', '#ffffff')
    clone.insertBefore(bg, clone.firstChild)
  }
  // `transparent` is a CSS color, not valid SVG 1.1 paint — browsers accept it,
  // but standalone viewers (Inkscape, image viewers, editors) fall back to the
  // default fill, BLACK. Invisible hit targets drawn on top of real marks then
  // cover them in black. Normalise to `none`, which every renderer understands.
  for (const attr of ['fill', 'stroke'] as const) {
    clone.querySelectorAll(`[${attr}="transparent"]`).forEach(el => el.setAttribute(attr, 'none'))
  }
  if (!clone.getAttribute('width') && w) clone.setAttribute('width', String(w))
  if (!clone.getAttribute('height') && h) clone.setAttribute('height', String(h))

  const xml = new XMLSerializer().serializeToString(clone)
  const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${xml}`], { type: 'image/svg+xml' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/** Turns an arbitrary title into a safe lowercase-dashed filename stem. */
export function sanitizeFilename(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'chart'
}
