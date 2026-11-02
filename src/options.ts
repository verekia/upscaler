// The settings the UI exposes, their defaults (manablade's `bun upscale`: the anime model, 2x, Lanczos3,
// PNG or JPEG q95), their persistence in localStorage, and the output size they ask for.

import { MODELS } from './models'

import type { Capabilities } from './engine'
import type { ModelId } from './models'
import type { Filter } from './resize'
import type { Backend, Format, Precision, UpscaleOptions } from './worker-protocol'

export type SizeMode = 'scale' | 'width' | 'height'

export type Options = {
  model: ModelId
  denoise: number
  sizeMode: SizeMode
  scale: number
  width: number
  height: number
  filter: Filter
  tta: boolean
  alpha: 'model' | 'resize'
  format: Format
  quality: number
  backend: Backend | 'auto'
  precision: Precision | 'auto'
  // 0 picks the model's default
  tile: number
  overlap: number
}

export const DEFAULT_OPTIONS: Options = {
  model: 'realesrgan-x4plus-anime',
  denoise: 0.5,
  sizeMode: 'scale',
  scale: 2,
  width: 3840,
  height: 2160,
  filter: 'lanczos3',
  tta: false,
  alpha: 'model',
  format: 'png',
  quality: 95,
  backend: 'auto',
  precision: 'auto',
  tile: 0,
  overlap: 16,
}

// The output is one RGBA buffer and an encoder's working copy: beyond this, tabs run out of memory
export const MAX_SIDE = 16384
export const MAX_PIXELS = 140e6

const STORAGE_KEY = 'upscaler:options'

export const loadOptions = (): Options => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as Partial<Options>
    const options = { ...DEFAULT_OPTIONS, ...saved }
    return MODELS.some(model => model.id === options.model) ? options : DEFAULT_OPTIONS
  } catch {
    return DEFAULT_OPTIONS
  }
}

export const saveOptions = (options: Options) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(options))
  } catch {
    // Private mode or storage full: settings just don't persist
  }
}

export const outputSize = (options: Options, width: number, height: number) => {
  if (options.sizeMode === 'width')
    return { width: options.width, height: Math.max(1, Math.round((options.width * height) / width)) }
  if (options.sizeMode === 'height')
    return { width: Math.max(1, Math.round((options.height * width) / height)), height: options.height }
  return {
    width: Math.max(1, Math.round(width * options.scale)),
    height: Math.max(1, Math.round(height * options.scale)),
  }
}

export const sizeProblem = (width: number, height: number) => {
  if (width > MAX_SIDE || height > MAX_SIDE) return `Over ${MAX_SIDE} px on a side`
  if (width * height > MAX_PIXELS) return `Over ${MAX_PIXELS / 1e6} megapixels`
  return undefined
}

export const workerOptions = (
  options: Options,
  capabilities: Capabilities,
  outWidth: number,
  outHeight: number,
): UpscaleOptions => ({
  model: options.model,
  denoise: options.denoise,
  outWidth,
  outHeight,
  filter: options.filter,
  tta: options.tta,
  alpha: options.alpha,
  format: options.format,
  quality: options.quality,
  precision: resolvedPrecision(options, capabilities),
  tile: options.tile || MODELS.find(model => model.id === options.model)!.tile,
  overlap: options.overlap,
})

export const resolvedPrecision = (options: Options, capabilities: Capabilities): Precision => {
  if (capabilities.backend === 'wasm' || !capabilities.fp16) return 'fp32'
  return options.precision === 'auto' ? 'fp16' : options.precision
}
