import { expect, test } from 'bun:test'

import { RowResampler } from './resize'
import { CancelledError, upscale } from './upscale'

import type { Job, RunModel } from './upscale'

// A stand-in network: nearest 4x, which commutes with flips and transposes like the real one should
const nearest: RunModel = async (input, width, height) => {
  const out = new Float32Array(3 * width * height * 16)
  for (let c = 0; c < 3; c++)
    for (let y = 0; y < height * 4; y++)
      for (let x = 0; x < width * 4; x++)
        out[c * width * height * 16 + y * width * 4 + x] = input[c * width * height + (y >> 2) * width + (x >> 2)]!
  return out
}

const image = (width: number, height: number, alpha?: number) =>
  Uint8ClampedArray.from({ length: width * height * 4 }, (_, i) =>
    i % 4 === 3 ? (alpha ?? 255) : Math.floor(Math.random() * 256),
  )

const nearestUpscale = (pixels: Uint8ClampedArray, width: number, height: number) => {
  const out = new Uint8ClampedArray(width * height * 64)
  for (let y = 0; y < height * 4; y++)
    for (let x = 0; x < width * 4; x++)
      for (let c = 0; c < 4; c++) out[(y * width * 4 + x) * 4 + c] = pixels[((y >> 2) * width + (x >> 2)) * 4 + c]!
  return out
}

const job = (pixels: Uint8ClampedArray, width: number, height: number, patch: Partial<Job> = {}): Job => ({
  pixels,
  width,
  height,
  outWidth: width * 4,
  outHeight: height * 4,
  filter: 'lanczos3',
  tta: false,
  alpha: 'model',
  tile: 7,
  overlap: 3,
  ...patch,
})

test('tiles, overlaps and TTA reassemble the image without seams', async () => {
  const pixels = image(23, 17)
  const expected = nearestUpscale(pixels, 23, 17)
  for (const patch of [{}, { overlap: 0 }, { tile: 64 }, { tile: 5, tta: true }, { tile: 16, overlap: 9, tta: true }])
    expect(await upscale(job(pixels, 23, 17, patch), nearest)).toEqual(expected)
})

test('resampling band by band matches resampling the whole native image', async () => {
  const pixels = image(19, 26)
  const native = nearestUpscale(pixels, 19, 26)
  const expected = new Uint8ClampedArray(41 * 57 * 4).fill(255)
  new RowResampler(76, 104, 41, 57, 'mitchell', expected, 0, 3).push(native, 0, 104)
  expect(
    await upscale(job(pixels, 19, 26, { outWidth: 41, outHeight: 57, filter: 'mitchell', tile: 6 }), nearest),
  ).toEqual(expected)
})

test('alpha goes through the model or is resampled', async () => {
  const pixels = image(9, 11, 128)
  pixels[3] = 0
  const viaModel = await upscale(job(pixels, 9, 11), nearest)
  expect(viaModel).toEqual(nearestUpscale(pixels, 9, 11))

  pixels[3] = 128
  const resampled = await upscale(job(pixels, 9, 11, { alpha: 'resize' }), nearest)
  for (let i = 3; i < resampled.length; i += 4) expect(resampled[i]).toBe(128)
})

test('tiles reported for the preview cover the native image once', async () => {
  const covered = new Uint8Array(40 * 28)
  await upscale(job(image(10, 7), 10, 7, { tile: 4 }), nearest, {
    tile: (x, y, width, height, region) => {
      expect(region.length).toBe(width * height * 4)
      for (let j = y; j < y + height; j++) for (let i = x; i < x + width; i++) covered[j * 40 + i]!++
    },
  })
  expect(covered.every(count => count === 1)).toBe(true)
})

test('cancelling stops before the next tile', async () => {
  let runs = 0
  const counting: RunModel = (input, width, height) => {
    runs++
    return nearest(input, width, height)
  }
  const pending = upscale(job(image(20, 20), 20, 20, { tile: 5 }), counting, { cancelled: () => runs >= 2 })
  await expect(pending).rejects.toBeInstanceOf(CancelledError)
  expect(runs).toBe(2)
})
