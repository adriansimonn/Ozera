/**
 * Utilities for client-side SVG and image export.
 */

export type ExportFormat = 'png' | 'svg' | 'pdf'

export interface ExportOptions {
  filename?: string
  format?: ExportFormat
  scale?: number
  backgroundColor?: string
}

/**
 * Serialize an SVG element to a string.
 */
export function serializeSvg(svgElement: SVGSVGElement): string {
  const serializer = new XMLSerializer()
  let svgString = serializer.serializeToString(svgElement)

  // Add XML declaration and namespace if missing
  if (!svgString.includes('xmlns=')) {
    svgString = svgString.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
  }

  return svgString
}

/**
 * Convert SVG element to a data URL.
 */
export function svgToDataUrl(svgElement: SVGSVGElement): string {
  const svgString = serializeSvg(svgElement)
  const encoded = encodeURIComponent(svgString)
  return `data:image/svg+xml;charset=utf-8,${encoded}`
}

/**
 * Convert SVG element to PNG blob.
 */
export async function svgToPngBlob(
  svgElement: SVGSVGElement,
  scale: number = 2,
  backgroundColor?: string
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const svgString = serializeSvg(svgElement)
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    if (!ctx) {
      reject(new Error('Failed to get canvas context'))
      return
    }

    const img = new Image()
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(svgBlob)

    img.onload = () => {
      // Get dimensions from SVG
      const width = svgElement.width.baseVal.value || svgElement.clientWidth || 800
      const height = svgElement.height.baseVal.value || svgElement.clientHeight || 600

      canvas.width = width * scale
      canvas.height = height * scale

      // Fill background if specified
      if (backgroundColor) {
        ctx.fillStyle = backgroundColor
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }

      ctx.scale(scale, scale)
      ctx.drawImage(img, 0, 0)
      URL.revokeObjectURL(url)

      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob)
          } else {
            reject(new Error('Failed to create blob'))
          }
        },
        'image/png',
        1.0
      )
    }

    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Failed to load SVG image'))
    }

    img.src = url
  })
}

/**
 * Download a blob as a file.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Download SVG element as a file.
 */
export function downloadSvg(svgElement: SVGSVGElement, filename: string = 'export.svg'): void {
  const svgString = serializeSvg(svgElement)
  const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' })
  downloadBlob(blob, filename)
}

/**
 * Download SVG element as PNG.
 */
export async function downloadSvgAsPng(
  svgElement: SVGSVGElement,
  filename: string = 'export.png',
  scale: number = 2,
  backgroundColor?: string
): Promise<void> {
  const blob = await svgToPngBlob(svgElement, scale, backgroundColor)
  downloadBlob(blob, filename)
}

/**
 * Export SVG element in specified format.
 */
export async function exportSvgElement(
  svgElement: SVGSVGElement,
  options: ExportOptions = {}
): Promise<void> {
  const { filename = 'export', format = 'png', scale = 2, backgroundColor = 'white' } = options

  const fullFilename = filename.includes('.') ? filename : `${filename}.${format}`

  switch (format) {
    case 'svg':
      downloadSvg(svgElement, fullFilename)
      break
    case 'png':
      await downloadSvgAsPng(svgElement, fullFilename, scale, backgroundColor)
      break
    case 'pdf':
      // PDF export requires server-side rendering
      throw new Error('PDF export requires server-side rendering. Use the API endpoint.')
    default:
      throw new Error(`Unsupported format: ${format}`)
  }
}

/**
 * Get dimensions of an SVG element.
 */
export function getSvgDimensions(svgElement: SVGSVGElement): { width: number; height: number } {
  const width = svgElement.width.baseVal.value || svgElement.clientWidth || 800
  const height = svgElement.height.baseVal.value || svgElement.clientHeight || 600
  return { width, height }
}

/**
 * Clone SVG element with styles inlined.
 * Useful for exporting styled SVGs.
 */
export function cloneSvgWithStyles(svgElement: SVGSVGElement): SVGSVGElement {
  const clone = svgElement.cloneNode(true) as SVGSVGElement

  // Get computed styles from original and apply to clone
  const originalElements = svgElement.querySelectorAll('*')
  const cloneElements = clone.querySelectorAll('*')

  originalElements.forEach((el, i) => {
    const computedStyle = window.getComputedStyle(el)
    const cloneEl = cloneElements[i] as SVGElement

    // Copy important style properties
    const styleProps = ['fill', 'stroke', 'stroke-width', 'opacity', 'font-family', 'font-size']
    styleProps.forEach((prop) => {
      const value = computedStyle.getPropertyValue(prop)
      if (value && value !== 'none' && value !== '') {
        cloneEl.style.setProperty(prop, value)
      }
    })
  })

  return clone
}
