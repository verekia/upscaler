// Messages between the main thread and the upscale worker (kept apart from the worker module so importing
// the types never pulls the worker file into the main bundle).

import type { ModelId } from './models'
import type { Filter } from './resize'

export type Backend = 'webgpu' | 'wasm'
export type Precision = 'fp16' | 'fp32'
export type Format = 'png' | 'jpeg' | 'webp'

export type UpscaleOptions = {
  model: ModelId
  // realesr-general-x4v3 only: 0 keeps the noise, 1 removes it
  denoise: number
  outWidth: number
  outHeight: number
  filter: Filter
  tta: boolean
  alpha: 'model' | 'resize'
  format: Format
  // JPEG and WebP, 1-100
  quality: number
  precision: Precision
  tile: number
  overlap: number
}

export type Stage = 'decode' | 'download' | 'compile' | 'upscale' | 'encode'

export type WorkerRequest =
  | { type: 'init'; backend: Backend }
  | { type: 'upscale'; id: number; file: Blob; options: UpscaleOptions }
  | { type: 'cancel'; id: number }

export type WorkerResponse =
  | { type: 'ready'; backend: Backend; device: string; fp16: boolean; threads: number }
  | { type: 'unavailable'; message: string }
  | { type: 'progress'; id: number; stage: Stage; done: number; total: number }
  // A finished tile in native 4x pixels, for the live preview
  | {
      type: 'tile'
      id: number
      x: number
      y: number
      width: number
      height: number
      pixels: Uint8ClampedArray<ArrayBuffer>
    }
  | { type: 'done'; id: number; blob: Blob; width: number; height: number; timings: Record<string, number> }
  | { type: 'cancelled'; id: number }
  | { type: 'error'; id: number; message: string }
