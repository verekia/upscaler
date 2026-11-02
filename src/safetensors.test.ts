import { expect, test } from 'bun:test'

import { halfArray } from './half'
import { readSafetensors, toFloat32, writeSafetensors } from './safetensors'

import type { Tensors } from './safetensors'

test('round-trips tensors and metadata, with the data 8-byte aligned', () => {
  const kernel = Float32Array.from([0.5, -1.25, 3, 0.125, 2, -0.75])
  const bias = Float32Array.from([0.1, -0.2])
  const tensors: Tensors = new Map([
    ['conv.weight', { dtype: 'F16', shape: [2, 1, 1, 3], data: new Uint8Array(halfArray(kernel).buffer) }],
    ['conv.bias', { dtype: 'F32', shape: [2], data: new Uint8Array(bias.buffer) }],
  ])
  const bytes = writeSafetensors(tensors, { source: 'test' })
  expect((8 + Number(new DataView(bytes.buffer).getBigUint64(0, true))) % 8).toBe(0)

  const { tensors: read, metadata } = readSafetensors(bytes.buffer as ArrayBuffer)
  expect(metadata).toEqual({ source: 'test' })
  expect(read.get('conv.weight')).toMatchObject({ dtype: 'F16', shape: [2, 1, 1, 3] })
  expect([...toFloat32(read.get('conv.weight')!)]).toEqual([...kernel])
  expect([...toFloat32(read.get('conv.bias')!)]).toEqual([...bias])
})
