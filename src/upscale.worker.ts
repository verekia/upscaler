// The upscale worker: ONNX Runtime on WebGPU or wasm, the model graphs, the pipeline and the encoders.
// ONNX Runtime is not bundled: its ESM build and wasm are copied to public/ort by scripts/assets.ts and
// imported at run time, so its threads load its own glue file and not this bundle.

import { buildModel, checkpointWeights } from './graph'
import { modelById } from './models'
import { encodePng } from './png'
import { readSafetensors } from './safetensors'
import { CancelledError, upscale } from './upscale'

import type * as Ort from 'onnxruntime-web'
import type { CheckpointName } from './models'
import type { Tensors } from './safetensors'
import type { Backend, Stage, UpscaleOptions, WorkerRequest, WorkerResponse } from './worker-protocol'

let ort: typeof Ort
let backend: Backend
let loaded: { key: string; session: Ort.InferenceSession } | undefined
const buffers = new Map<CheckpointName, ArrayBuffer>()
const cancelled = new Set<number>()
// One job at a time: a cancelled job finishes its current tile before the next one starts
let queue = Promise.resolve()

const reply = (response: WorkerResponse, transfer: Transferable[] = []) => postMessage(response, { transfer })
const progress = (id: number, stage: Stage, done = 0, total = 1) => reply({ type: 'progress', id, stage, done, total })

const init = async (requested: Backend) => {
  let device = ''
  let fp16 = false
  if (requested === 'webgpu') {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) return reply({ type: 'unavailable', message: 'WebGPU is not available in this browser' })
    fp16 = adapter.features.has('shader-f16')
    const { vendor, architecture, description } = adapter.info
    device = description || [vendor.charAt(0).toUpperCase() + vendor.slice(1), architecture].filter(Boolean).join(' ')
  }
  const file = requested === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'
  ort = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ `${location.origin}/ort/${file}`)
  // Threads need SharedArrayBuffer, so the page must be cross-origin isolated (nginx.conf, next.config.js)
  const threads = crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`
  ort.env.wasm.numThreads = threads
  ort.env.logLevel = 'error'
  backend = requested
  reply({ type: 'ready', backend, device, fp16, threads })
}

const sum = (map: Map<string, number>) => [...map.values()].reduce((a, b) => a + b, 0)

// Fetches the safetensors once per worker, reporting the bytes received across all of a model's files
const download = (id: number, names: readonly CheckpointName[]): Promise<Tensors[]> => {
  const received = new Map<string, number>()
  const sizes = new Map<string, number>()
  let reported = 0
  const report = () => {
    if (performance.now() - reported < 100) return
    reported = performance.now()
    progress(id, 'download', sum(received), sum(sizes))
  }
  return Promise.all(
    names.map(async name => {
      let buffer = buffers.get(name)
      if (!buffer) {
        const response = await fetch(`/models/${name}.safetensors`)
        if (!response.ok || !response.body) throw new Error(`Could not download ${name} (HTTP ${response.status})`)
        sizes.set(name, Number(response.headers.get('content-length')) || 0)
        const chunks: Uint8Array[] = []
        let size = 0
        const reader = response.body.getReader()
        for (let read = await reader.read(); !read.done; read = await reader.read()) {
          chunks.push(read.value)
          size += read.value.byteLength
          received.set(name, size)
          report()
        }
        const bytes = new Uint8Array(size)
        let offset = 0
        for (const chunk of chunks) {
          bytes.set(chunk, offset)
          offset += chunk.byteLength
        }
        buffer = bytes.buffer
        buffers.set(name, buffer)
      }
      return readSafetensors(buffer).tensors
    }),
  )
}

const sessionFor = async (id: number, options: UpscaleOptions) => {
  const model = modelById(options.model)
  const precision = backend === 'webgpu' ? options.precision : 'fp32'
  const key = `${model.id}:${precision}:${model.checkpoints.length > 1 ? options.denoise : ''}`
  if (loaded?.key === key) return loaded.session
  const checkpoints = await download(id, model.checkpoints)
  progress(id, 'compile')
  const { shapes, weights } = checkpointWeights(checkpoints, options.denoise)
  const graph = buildModel(shapes, weights, precision === 'fp16' ? 'float16' : 'float32')
  await loaded?.session.release()
  loaded = undefined
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  })
  loaded = { key, session }
  return session
}

const encode = async (pixels: Uint8ClampedArray<ArrayBuffer>, options: UpscaleOptions): Promise<Blob> => {
  const { format, outWidth: width, outHeight: height } = options
  if (format === 'png') return encodePng(pixels, width, height)
  // JPEG has no alpha: the colors are kept as they are, as OpenCV and sharp write them
  if (format === 'jpeg') for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255
  const canvas = new OffscreenCanvas(width, height)
  canvas.getContext('2d')!.putImageData(new ImageData(pixels, width, height), 0, 0)
  const type = `image/${format}`
  const blob = await canvas.convertToBlob({ type, quality: options.quality / 100 })
  if (blob.type !== type) throw new Error(`This browser cannot encode ${format.toUpperCase()}, pick PNG or JPEG`)
  return blob
}

const run = async (id: number, file: Blob, options: UpscaleOptions) => {
  const timings: Record<string, number> = {}
  let mark = performance.now()
  const lap = (name: string) => {
    timings[name] = performance.now() - mark
    mark = performance.now()
  }

  progress(id, 'decode')
  const bitmap = await createImageBitmap(file, { premultiplyAlpha: 'none' })
  const { width, height } = bitmap
  const context = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true })!
  context.drawImage(bitmap, 0, 0)
  bitmap.close()
  const pixels = context.getImageData(0, 0, width, height).data
  lap('decode')

  const session = await sessionFor(id, options)
  const [input] = session.inputNames
  const [output] = session.outputNames
  lap('model')

  const result = await upscale(
    { ...options, pixels, width, height },
    async (data, w, h) => {
      const outputs = await session.run({ [input!]: new ort.Tensor('float32', data, [1, 3, h, w]) })
      return outputs[output!]!.data as Float32Array
    },
    {
      progress: (done, total) => progress(id, 'upscale', done, total),
      tile: (x, y, w, h, region) =>
        reply({ type: 'tile', id, x, y, width: w, height: h, pixels: region }, [region.buffer]),
      cancelled: () => cancelled.has(id),
    },
  )
  lap('upscale')

  progress(id, 'encode')
  const blob = await encode(result, options)
  lap('encode')
  reply({ type: 'done', id, blob, width: options.outWidth, height: options.outHeight, timings })
}

const describe = (error: unknown) => (error instanceof Error ? error.message : String(error))

addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data
  if (message.type === 'init') {
    init(message.backend).catch(error => reply({ type: 'unavailable', message: describe(error) }))
  } else if (message.type === 'cancel') {
    cancelled.add(message.id)
  } else {
    const { id, file, options } = message
    queue = queue
      .then(() => (cancelled.has(id) ? Promise.reject(new CancelledError()) : run(id, file, options)))
      .catch(error =>
        reply(
          error instanceof CancelledError ? { type: 'cancelled', id } : { type: 'error', id, message: describe(error) },
        ),
      )
      .finally(() => cancelled.delete(id))
  }
})
