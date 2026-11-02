// The upscaling pipeline, free of the DOM and of ONNX Runtime: tiles go through the model (with TTA and
// the alpha pass) and, one row of tiles at a time, into the resampler that makes the requested size.

import { NATIVE_SCALE as S } from './models'
import { RowResampler } from './resize'
import { addUntransformed, readWindow, spans, tileSize, transformPlane } from './tiles'

import type { Filter } from './resize'

// Runs the network on planar RGB in [0, 1] and returns its planar output, S times larger
export type RunModel = (input: Float32Array, width: number, height: number) => Promise<Float32Array>

export type Job = {
  pixels: Uint8ClampedArray
  width: number
  height: number
  outWidth: number
  outHeight: number
  filter: Filter
  tta: boolean
  // How a transparent image's alpha grows: through the model like the colors, or resampled
  alpha: 'model' | 'resize'
  tile: number
  overlap: number
}

export type Hooks = {
  progress?: (done: number, total: number) => void
  // A finished tile at the native scale: RGBA of the region [x, x + width) × [y, y + height)
  tile?: (x: number, y: number, width: number, height: number, pixels: Uint8ClampedArray<ArrayBuffer>) => void
  // Checked before each tile
  cancelled?: () => boolean
}

export class CancelledError extends Error {}

export const hasTransparency = (pixels: Uint8ClampedArray) => {
  for (let i = 3; i < pixels.length; i += 4) if (pixels[i] !== 255) return true
  return false
}

const unit = (value: number) => Math.min(1, Math.max(0, value))

export const upscale = async (job: Job, run: RunModel, hooks: Hooks = {}): Promise<Uint8ClampedArray<ArrayBuffer>> => {
  const { pixels, width, height, outWidth, outHeight, overlap } = job
  const transparent = hasTransparency(pixels)
  const alphaPass = transparent && job.alpha === 'model'
  let tileWidth = tileSize(width, job.tile)
  let tileHeight = tileSize(height, job.tile)
  // TTA transposes the tile, so it must be square
  if (job.tta) tileWidth = tileHeight = Math.max(tileWidth, tileHeight)
  const columns = spans(width, tileWidth)
  const rows = spans(height, tileHeight)
  const total = columns.length * rows.length * (job.tta ? 8 : 1) * (alphaPass ? 2 : 1)
  let done = 0
  hooks.progress?.(0, total)

  const inWidth = tileWidth + 2 * overlap
  const inHeight = tileHeight + 2 * overlap
  const input = new Float32Array(3 * inWidth * inHeight)
  const nativeWidth = width * S
  const out = new Uint8ClampedArray(outWidth * outHeight * 4)
  const resampler = new RowResampler(
    nativeWidth,
    height * S,
    outWidth,
    outHeight,
    job.filter,
    out,
    0,
    alphaPass ? 4 : 3,
  )
  // The native rows of the current row of tiles
  const band = new Uint8ClampedArray(nativeWidth * tileHeight * S * 4).fill(255)

  const infer = async (): Promise<Float32Array> => {
    if (!job.tta) {
      const output = await run(input, inWidth, inHeight)
      hooks.progress?.(++done, total)
      return output
    }
    // The average of the model over the 8 flips and transposes of the tile
    const n = inWidth
    const m = n * S
    const variant = new Float32Array(input.length)
    const sum = new Float32Array(3 * m * m)
    for (let t = 0; t < 8; t++) {
      for (let c = 0; c < 3; c++)
        transformPlane(input.subarray(c * n * n, (c + 1) * n * n), variant.subarray(c * n * n, (c + 1) * n * n), n, t)
      const output = await run(variant, n, n)
      for (let c = 0; c < 3; c++)
        addUntransformed(output.subarray(c * m * m, (c + 1) * m * m), sum.subarray(c * m * m, (c + 1) * m * m), m, t)
      hooks.progress?.(++done, total)
    }
    for (let i = 0; i < sum.length; i++) sum[i]! /= 8
    return sum
  }

  for (const row of rows) {
    const top = row.start - overlap
    for (const column of columns) {
      if (hooks.cancelled?.()) throw new CancelledError()
      const left = column.start - overlap
      const outputWidth = inWidth * S
      const plane = outputWidth * inHeight * S

      readWindow(pixels, width, height, left, top, input, inWidth, inHeight, [0, 1, 2])
      let output = await infer()
      for (let y = row.from * S; y < row.to * S; y++) {
        const source = (y - top * S) * outputWidth - left * S
        const target = (y - row.from * S) * nativeWidth
        for (let x = column.from * S; x < column.to * S; x++) {
          // Uint8ClampedArray rounds half to even and clamps, as Real-ESRGAN's numpy round does
          band[(target + x) * 4] = output[source + x]! * 255
          band[(target + x) * 4 + 1] = output[plane + source + x]! * 255
          band[(target + x) * 4 + 2] = output[2 * plane + source + x]! * 255
        }
      }

      if (alphaPass) {
        // Alpha goes through as a gray image and comes back as OpenCV's BGR2GRAY of the output
        readWindow(pixels, width, height, left, top, input, inWidth, inHeight, [3, 3, 3])
        output = await infer()
        for (let y = row.from * S; y < row.to * S; y++) {
          const source = (y - top * S) * outputWidth - left * S
          const target = (y - row.from * S) * nativeWidth
          for (let x = column.from * S; x < column.to * S; x++) {
            const i = source + x
            band[(target + x) * 4 + 3] =
              (0.299 * unit(output[i]!) + 0.587 * unit(output[plane + i]!) + 0.114 * unit(output[2 * plane + i]!)) * 255
          }
        }
      }

      if (hooks.tile) {
        const x = column.from * S
        const w = (column.to - column.from) * S
        const h = (row.to - row.from) * S
        const region = new Uint8ClampedArray(w * h * 4)
        for (let y = 0; y < h; y++)
          region.set(band.subarray((y * nativeWidth + x) * 4, (y * nativeWidth + x + w) * 4), y * w * 4)
        hooks.tile(x, row.from * S, w, h, region)
      }
    }
    resampler.push(band, row.from * S, (row.to - row.from) * S)
  }

  if (alphaPass) return out
  if (transparent) new RowResampler(width, height, outWidth, outHeight, job.filter, out, 3, 1).push(pixels, 0, height)
  else for (let i = 3; i < out.length; i += 4) out[i] = 255
  return out
}
