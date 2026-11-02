// The Real-ESRGAN models. scripts/assets.ts converts each pinned upstream checkpoint into
// public/models/<checkpoint>.safetensors, which the worker turns into an ONNX graph.

const RELEASES = 'https://github.com/xinntao/Real-ESRGAN/releases/download'

export const CHECKPOINTS = {
  'realesrgan-x4plus-anime': {
    url: `${RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth`,
    sha256: 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da',
  },
  'realesrgan-x4plus': {
    url: `${RELEASES}/v0.1.0/RealESRGAN_x4plus.pth`,
    sha256: '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
  },
  'realesr-animevideov3': {
    url: `${RELEASES}/v0.2.5.0/realesr-animevideov3.pth`,
    sha256: 'b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d',
  },
  'realesr-general-x4v3': {
    url: `${RELEASES}/v0.2.5.0/realesr-general-x4v3.pth`,
    sha256: '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
  },
  'realesr-general-wdn-x4v3': {
    url: `${RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth`,
    sha256: '1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca',
  },
} as const

export type CheckpointName = keyof typeof CHECKPOINTS

export type ModelId = 'realesrgan-x4plus-anime' | 'realesrgan-x4plus' | 'realesr-animevideov3' | 'realesr-general-x4v3'

export type Model = {
  id: ModelId
  label: string
  description: string
  architecture: string
  // The general model blends its two checkpoints by the denoise strength: 1 is the first, 0 the second
  checkpoints: readonly [CheckpointName] | readonly [CheckpointName, CheckpointName]
  megabytes: number
  // Multiply-accumulates per input pixel, which sets the relative speed
  macsPerPixel: number
  // Default input tile side: the RRDBNets hold 64 channels at 4x the tile, the compact nets only at 1x
  tile: number
}

export const MODELS: readonly Model[] = [
  {
    id: 'realesrgan-x4plus-anime',
    label: 'Anime',
    description: 'Anime, illustrations and game art: crisp lines, clean flat colors.',
    architecture: 'RRDBNet · 6 blocks',
    checkpoints: ['realesrgan-x4plus-anime'],
    megabytes: 8.9,
    macsPerPixel: 5.7e6,
    tile: 256,
  },
  {
    id: 'realesrgan-x4plus',
    label: 'Photo',
    description: 'Photos and realistic renders. The heaviest model, about 3x slower than Anime.',
    architecture: 'RRDBNet · 23 blocks',
    checkpoints: ['realesrgan-x4plus'],
    megabytes: 33.4,
    macsPerPixel: 17.9e6,
    tile: 256,
  },
  {
    id: 'realesr-animevideov3',
    label: 'Anime video',
    description: 'Tiny and about 9x faster than Anime, a little softer. Made for video frames.',
    architecture: 'SRVGGNet · 16 convs',
    checkpoints: ['realesr-animevideov3'],
    megabytes: 1.2,
    macsPerPixel: 0.62e6,
    tile: 512,
  },
  {
    id: 'realesr-general-x4v3',
    label: 'General',
    description: 'Fast general-purpose model for photos and mixed content, with a denoise strength.',
    architecture: 'SRVGGNet · 32 convs',
    checkpoints: ['realesr-general-x4v3', 'realesr-general-wdn-x4v3'],
    megabytes: 4.9,
    macsPerPixel: 1.21e6,
    tile: 512,
  },
]

// Every model upscales 4x; any other output size is resampled from that
export const NATIVE_SCALE = 4

export const modelById = (id: ModelId) => MODELS.find(model => model.id === id)!
