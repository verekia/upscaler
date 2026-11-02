import { expect, test } from 'bun:test'

import { addUntransformed, reflect, spans, tileSize, transformPlane } from './tiles'

test('tiles are as few as possible and equal', () => {
  expect(tileSize(1000, 192)).toBe(167)
  expect(tileSize(192, 192)).toBe(192)
  expect(tileSize(100, 192)).toBe(100)
  expect(tileSize(193, 192)).toBe(97)
})

test('spans write every pixel once, from tiles inside the image', () => {
  // The last case is a square TTA tile wider than the image
  for (const [size, tile] of [
    [1000, 167],
    [193, 97],
    [7, 3],
    [64, 64],
    [50, 80],
  ] as const) {
    const written = new Uint8Array(size)
    for (const { start, from, to } of spans(size, tile)) {
      expect(start).toBeGreaterThanOrEqual(0)
      expect(start <= from && to <= start + tile).toBe(true)
      for (let i = from; i < to; i++) written[i]!++
    }
    expect([...written].every(count => count === 1)).toBe(true)
  }
})

test('reflect mirrors without repeating the edge, however far out', () => {
  expect(reflect(-1, 10)).toBe(1)
  expect(reflect(10, 10)).toBe(8)
  expect(reflect(-25, 10)).toBe(7)
  expect(reflect(4, 10)).toBe(4)
  expect(reflect(-3, 1)).toBe(0)
})

test('every TTA transform is undone', () => {
  const n = 5
  const source = Float32Array.from({ length: n * n }, (_, i) => i)
  for (let t = 0; t < 8; t++) {
    const transformed = new Float32Array(n * n)
    const back = new Float32Array(n * n)
    transformPlane(source, transformed, n, t)
    addUntransformed(transformed, back, n, t)
    expect([...back]).toEqual([...source])
  }
})
