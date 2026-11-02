import { expect, test } from 'bun:test'

import { contributions, RowResampler } from './resize'

import type { Filter } from './resize'

const FILTERS: Filter[] = ['lanczos3', 'mitchell', 'catmull-rom']

const random = (length: number) => Uint8ClampedArray.from({ length }, () => Math.floor(Math.random() * 256))

const resample = (
  pixels: Uint8ClampedArray,
  w: number,
  h: number,
  ow: number,
  oh: number,
  filter: Filter,
  rows = h,
) => {
  const out = new Uint8ClampedArray(ow * oh * 4)
  const resampler = new RowResampler(w, h, ow, oh, filter, out, 0, 4)
  for (let y = 0; y < h; y += rows) {
    const count = Math.min(rows, h - y)
    resampler.push(pixels.subarray(y * w * 4, (y + count) * w * 4), y, count)
  }
  return out
}

test('weights sum to one', () => {
  for (const filter of FILTERS)
    for (const [inSize, outSize] of [
      [100, 37],
      [37, 100],
      [64, 64],
    ] as const) {
      const { count, weights, taps } = contributions(inSize, outSize, filter)
      for (let o = 0; o < outSize; o++) {
        let sum = 0
        for (let i = 0; i < count[o]!; i++) sum += weights[o * taps + i]!
        expect(sum).toBeCloseTo(1, 5)
      }
    }
})

test('the same size is an exact copy', () => {
  const pixels = random(13 * 9 * 4)
  for (const filter of FILTERS) expect(resample(pixels, 13, 9, 13, 9, filter)).toEqual(pixels)
})

test('a flat image stays flat', () => {
  const pixels = new Uint8ClampedArray(20 * 12 * 4).fill(173)
  for (const filter of FILTERS)
    for (const [ow, oh] of [
      [7, 5],
      [61, 33],
    ])
      expect(resample(pixels, 20, 12, ow, oh, filter).every(value => value === 173)).toBe(true)
})

test('rows pushed in bands give the same result as all at once', () => {
  const pixels = random(40 * 30 * 4)
  for (const filter of FILTERS) {
    const whole = resample(pixels, 40, 30, 17, 11, filter)
    expect(resample(pixels, 40, 30, 17, 11, filter, 7)).toEqual(whole)
    expect(resample(pixels, 40, 30, 17, 11, filter, 1)).toEqual(whole)
  }
})
