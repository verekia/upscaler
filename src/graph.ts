// Real-ESRGAN's two architectures as ONNX graphs, built from a checkpoint's state dict: RRDBNet (basicsr)
// for the x4plus models and SRVGGNetCompact for the v3 ones. The input and output are float32 NCHW in
// [0, 1], any size; an fp16 graph casts at both ends.

import { halfArray, toHalf } from './half'
import { Graph } from './onnx'
import { toFloat32 } from './safetensors'

import type { Attribute, DataType } from './onnx'
import type { Tensors } from './safetensors'

// A state dict tensor's bytes in the graph's data type
export type WeightSource = (name: string, dataType: DataType) => Uint8Array

const CONV: Record<string, Attribute> = { kernel_shape: { ints: [3, 3] }, pads: { ints: [1, 1, 1, 1] } }
// PyTorch's F.interpolate(mode='nearest')
const NEAREST: Record<string, Attribute> = {
  mode: { string: 'nearest' },
  coordinate_transformation_mode: { string: 'asymmetric' },
  nearest_mode: { string: 'floor' },
}

const floats = (values: number[]) => new Uint8Array(Float32Array.from(values).buffer)

// The weights of one checkpoint, or of two blended by the denoise strength (Real-ESRGAN's network
// interpolation: strength × the first + (1 - strength) × the second)
export const checkpointWeights = (checkpoints: Tensors[], denoise: number) => {
  const [a, b] = checkpoints as [Tensors, Tensors?]
  const shapes = new Map([...a].map(([name, { shape }]) => [name, shape]))
  const weights: WeightSource = (name, dataType) => {
    const tensor = a.get(name)!
    if (!b && tensor.dtype === (dataType === 'float16' ? 'F16' : 'F32')) return tensor.data
    const values = toFloat32(tensor)
    if (b) {
      const other = toFloat32(b.get(name)!)
      for (let i = 0; i < values.length; i++) values[i] = denoise * values[i]! + (1 - denoise) * other[i]!
    }
    return dataType === 'float16' ? new Uint8Array(halfArray(values).buffer) : new Uint8Array(values.buffer)
  }
  return { shapes, weights }
}

export const buildModel = (shapes: Map<string, number[]>, weights: WeightSource, dataType: DataType): Uint8Array => {
  const g = new Graph()
  const names = new Set(shapes.keys())
  const weight = (name: string, dims = shapes.get(name)!) =>
    g.initializer(name, dataType, dims, weights(name, dataType))
  const conv = (input: string, layer: string) =>
    g.op('Conv', [input, weight(`${layer}.weight`), weight(`${layer}.bias`)], CONV)
  const upsample = (input: string, factor: number) =>
    g.op(
      'Resize',
      [input, '', g.initializer(`scales${factor}`, 'float32', [4], floats([1, 1, factor, factor]))],
      NEAREST,
    )

  let x = 'input'
  if (dataType === 'float16') x = g.op('Cast', [x], { to: { int: 10 } })

  let y: string
  if (names.has('conv_first.weight')) {
    // RRDBNet: residual-in-residual dense blocks at the input size, then two nearest 2x upsamplings
    const scale = g.initializer(
      'residual_scale',
      dataType,
      [],
      dataType === 'float16' ? new Uint8Array(Uint16Array.of(toHalf(0.2)).buffer) : floats([0.2]),
    )
    const lrelu = (input: string) => g.op('LeakyRelu', [input], { alpha: { float: 0.2 } })
    const scaledAdd = (residual: string, input: string) => g.op('Add', [g.op('Mul', [residual, scale]), input])
    const dense = (input: string, prefix: string) => {
      const features = [input]
      for (let i = 1; i <= 4; i++) {
        const joined = i === 1 ? input : g.op('Concat', features, { axis: { int: 1 } })
        features.push(lrelu(conv(joined, `${prefix}.conv${i}`)))
      }
      return scaledAdd(conv(g.op('Concat', features, { axis: { int: 1 } }), `${prefix}.conv5`), input)
    }
    const first = conv(x, 'conv_first')
    let body = first
    for (let b = 0; names.has(`body.${b}.rdb1.conv1.weight`); b++) {
      let out = body
      for (const rdb of ['rdb1', 'rdb2', 'rdb3']) out = dense(out, `body.${b}.${rdb}`)
      body = scaledAdd(out, body)
    }
    let features = g.op('Add', [first, conv(body, 'conv_body')])
    features = lrelu(conv(upsample(features, 2), 'conv_up1'))
    features = lrelu(conv(upsample(features, 2), 'conv_up2'))
    y = conv(lrelu(conv(features, 'conv_hr')), 'conv_last')
  } else {
    // SRVGGNetCompact: body.N alternates conv and PReLU, its last conv feeds a pixel shuffle, and the
    // input upsampled by nearest is added back
    let out = x
    let last = ''
    for (let i = 0; names.has(`body.${i}.weight`); i++) {
      if (names.has(`body.${i}.bias`)) {
        out = conv(out, `body.${i}`)
        last = `body.${i}`
      } else {
        // One slope per channel, broadcast over the image
        out = g.op('PRelu', [out, weight(`body.${i}.weight`, [shapes.get(`body.${i}.weight`)![0]!, 1, 1])])
      }
    }
    const factor = Math.round(Math.sqrt(shapes.get(`${last}.weight`)![0]! / 3))
    out = g.op('DepthToSpace', [out], { blocksize: { int: factor }, mode: { string: 'CRD' } })
    y = g.op('Add', [out, upsample(x, factor)])
  }

  if (dataType === 'float16') y = g.op('Cast', [y], { to: { int: 1 } })
  return g.encode(
    { name: 'input', dataType: 'float32', dims: [1, 3, 'height', 'width'] },
    { name: y, dataType: 'float32', dims: [1, 3, 'out_height', 'out_width'] },
  )
}
