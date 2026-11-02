import { expect, test } from 'bun:test'
import * as ort from 'onnxruntime-web'

import { buildModel, checkpointWeights } from './graph'

import type { Tensors } from './safetensors'

// Planar tensors and the reference ops of the two architectures, as PyTorch computes them
type Planes = { c: number; h: number; w: number; data: Float32Array }

const random = (length: number) => Float32Array.from({ length }, () => Math.random() - 0.5)

const stateDict = (shapes: [string, number[]][]): Tensors =>
  new Map(
    shapes.map(([name, shape]) => {
      const values = random(shape.reduce((a, b) => a * b, 1))
      return [name, { dtype: 'F32', shape, data: new Uint8Array(values.buffer) }]
    }),
  )

const weightOf = (tensors: Tensors, name: string) => new Float32Array(tensors.get(name)!.data.slice().buffer)

const conv = (x: Planes, tensors: Tensors, layer: string): Planes => {
  const weight = weightOf(tensors, `${layer}.weight`)
  const bias = weightOf(tensors, `${layer}.bias`)
  const out = new Float32Array(bias.length * x.h * x.w)
  for (let o = 0; o < bias.length; o++)
    for (let i = 0; i < x.h; i++)
      for (let j = 0; j < x.w; j++) {
        let sum = bias[o]!
        for (let c = 0; c < x.c; c++)
          for (let ki = 0; ki < 3; ki++)
            for (let kj = 0; kj < 3; kj++) {
              const y = i + ki - 1
              const z = j + kj - 1
              if (y >= 0 && y < x.h && z >= 0 && z < x.w)
                sum += weight[((o * x.c + c) * 3 + ki) * 3 + kj]! * x.data[(c * x.h + y) * x.w + z]!
            }
        out[(o * x.h + i) * x.w + j] = sum
      }
  return { c: bias.length, h: x.h, w: x.w, data: out }
}

const each = (x: Planes, f: (value: number, channel: number) => number): Planes => ({
  ...x,
  data: x.data.map((value, i) => f(value, Math.floor(i / (x.h * x.w)))),
})
const add = (a: Planes, b: Planes, scale = 1): Planes => ({ ...a, data: a.data.map((v, i) => v * scale + b.data[i]!) })
const concat = (xs: Planes[]): Planes => ({
  c: xs.reduce((sum, x) => sum + x.c, 0),
  h: xs[0]!.h,
  w: xs[0]!.w,
  data: Float32Array.from(xs.flatMap(x => [...x.data])),
})
const nearest = (x: Planes, s: number): Planes => {
  const out = new Float32Array(x.c * x.h * s * x.w * s)
  for (let c = 0; c < x.c; c++)
    for (let i = 0; i < x.h * s; i++)
      for (let j = 0; j < x.w * s; j++)
        out[(c * x.h * s + i) * x.w * s + j] = x.data[(c * x.h + Math.floor(i / s)) * x.w + Math.floor(j / s)]!
  return { c: x.c, h: x.h * s, w: x.w * s, data: out }
}
const pixelShuffle = (x: Planes, s: number): Planes => {
  const c = x.c / (s * s)
  const out = new Float32Array(x.data.length)
  for (let k = 0; k < c; k++)
    for (let i = 0; i < x.h * s; i++)
      for (let j = 0; j < x.w * s; j++)
        out[(k * x.h * s + i) * x.w * s + j] =
          x.data[((k * s * s + (i % s) * s + (j % s)) * x.h + Math.floor(i / s)) * x.w + Math.floor(j / s)]!
  return { c, h: x.h * s, w: x.w * s, data: out }
}
const lrelu = (x: Planes) => each(x, v => (v < 0 ? 0.2 * v : v))

const run = async (tensors: Tensors, input: Planes) => {
  const { shapes, weights } = checkpointWeights([tensors], 1)
  const session = await ort.InferenceSession.create(buildModel(shapes, weights, 'float32'), {
    executionProviders: ['wasm'],
  })
  const outputs = await session.run({ input: new ort.Tensor('float32', input.data, [1, 3, input.h, input.w]) })
  return outputs[session.outputNames[0]!]!.data as Float32Array
}

const expectClose = (actual: Float32Array, expected: Float32Array) => {
  expect(actual.length).toBe(expected.length)
  const scale = Math.max(...expected.map(Math.abs))
  for (let i = 0; i < actual.length; i++) expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(1e-4 * scale)
}

ort.env.wasm.numThreads = 1
const input: Planes = { c: 3, h: 5, w: 6, data: random(90) }

test('SRVGGNetCompact: conv and PReLU pairs, pixel shuffle, nearest base', async () => {
  const tensors = stateDict([
    ['body.0.weight', [4, 3, 3, 3]],
    ['body.0.bias', [4]],
    ['body.1.weight', [4]],
    ['body.2.weight', [4, 4, 3, 3]],
    ['body.2.bias', [4]],
    ['body.3.weight', [4]],
    ['body.4.weight', [12, 4, 3, 3]],
    ['body.4.bias', [12]],
  ])
  const prelu = (x: Planes, layer: string) => {
    const slope = weightOf(tensors, `${layer}.weight`)
    return each(x, (v, c) => (v < 0 ? slope[c]! * v : v))
  }
  const body = conv(
    prelu(conv(prelu(conv(input, tensors, 'body.0'), 'body.1'), tensors, 'body.2'), 'body.3'),
    tensors,
    'body.4',
  )
  expectClose(await run(tensors, input), add(pixelShuffle(body, 2), nearest(input, 2)).data)
})

test('RRDBNet: dense blocks, scaled residuals, two nearest 2x upsamplings', async () => {
  const shapes: [string, number[]][] = [['conv_first', [4, 3]]]
  for (const rdb of ['rdb1', 'rdb2', 'rdb3'])
    for (let i = 1; i <= 5; i++) shapes.push([`body.0.${rdb}.conv${i}`, [i === 5 ? 4 : 2, 4 + 2 * (i - 1)]])
  for (const layer of ['conv_body', 'conv_up1', 'conv_up2', 'conv_hr']) shapes.push([layer, [4, 4]])
  shapes.push(['conv_last', [3, 4]])
  const tensors = stateDict(
    shapes.flatMap(([layer, [out, inC]]): [string, number[]][] => [
      [`${layer}.weight`, [out!, inC!, 3, 3]],
      [`${layer}.bias`, [out!]],
    ]),
  )
  const dense = (x: Planes, prefix: string) => {
    const features = [x]
    for (let i = 1; i <= 4; i++) features.push(lrelu(conv(concat(features), tensors, `${prefix}.conv${i}`)))
    return add(conv(concat(features), tensors, `${prefix}.conv5`), x, 0.2)
  }
  const first = conv(input, tensors, 'conv_first')
  const block = add(dense(dense(dense(first, 'body.0.rdb1'), 'body.0.rdb2'), 'body.0.rdb3'), first, 0.2)
  let features = add(conv(block, tensors, 'conv_body'), first)
  features = lrelu(conv(nearest(features, 2), tensors, 'conv_up1'))
  features = lrelu(conv(nearest(features, 2), tensors, 'conv_up2'))
  const expected = conv(lrelu(conv(features, tensors, 'conv_hr')), tensors, 'conv_last')
  expectClose(await run(tensors, input), expected.data)
})

test('the denoise strength blends two checkpoints', () => {
  const a = stateDict([['w', [3]]])
  const b = stateDict([['w', [3]]])
  const blended = new Float32Array(checkpointWeights([a, b], 0.25).weights('w', 'float32').slice().buffer)
  const [wa, wb] = [weightOf(a, 'w'), weightOf(b, 'w')]
  for (let i = 0; i < 3; i++) expect(blended[i]!).toBeCloseTo(0.25 * wa[i]! + 0.75 * wb[i]!, 6)
})
