import { expect, test } from 'bun:test'

import { fromHalf, toHalf } from './half'

test('rounds to the nearest half, ties to even', () => {
  expect(toHalf(1)).toBe(0x3c00)
  expect(toHalf(-2)).toBe(0xc000)
  expect(toHalf(65504)).toBe(0x7bff)
  expect(toHalf(65520)).toBe(0x7c00)
  expect(toHalf(1 + 2 ** -11)).toBe(0x3c00)
  expect(toHalf(1 + 3 * 2 ** -11)).toBe(0x3c02)
  expect(toHalf(2 ** -24)).toBe(0x0001)
  expect(toHalf(2 ** -25)).toBe(0)
  expect(toHalf(3 * 2 ** -25)).toBe(0x0002)
  expect(toHalf(Infinity)).toBe(0x7c00)
  expect(toHalf(NaN) & 0x7c00).toBe(0x7c00)
})

test('matches Float16Array on random values, and round-trips', () => {
  const values = Float32Array.from({ length: 20000 }, (_, i) => (Math.random() - 0.5) * 2 ** ((i % 40) - 26))
  const reference = new Uint16Array(new Float16Array(values).buffer)
  for (let i = 0; i < values.length; i++) {
    expect(toHalf(values[i]!)).toBe(reference[i]!)
    expect(toHalf(fromHalf(reference[i]!))).toBe(reference[i]!)
  }
})
