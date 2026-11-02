// Prepares public/: the Real-ESRGAN weights as safetensors and the ONNX Runtime files the worker imports, each
// with its license. The pinned downloads go once into ~/.cache/upscaler, never into the repo.
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

import { halfArray } from '../src/half'
import { CHECKPOINTS } from '../src/models'
import { writeSafetensors } from '../src/safetensors'
import { readPth } from './pth'

import type { Tensors } from '../src/safetensors'

// Bump to convert the checkpoints again
const CONVERTER = '1'
// The WebGPU build (native WebGPU EP, asyncify) and the plain CPU build, each with its Emscripten glue and wasm
const ORT_FILES = [
  'ort.webgpu.min.mjs',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort.wasm.min.mjs',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
]
// The version the ONNX Runtime notices below come from
const ORT_VERSION = '1.31.1'
// Served next to what they cover and shown by pages/licenses.tsx. ONNX Runtime's npm package ships neither its
// license nor the notices of the third-party code compiled into its wasm, so they come from its release tag.
const NOTICES = [
  {
    file: 'realesrgan-LICENSE.txt',
    url: 'https://raw.githubusercontent.com/xinntao/Real-ESRGAN/v0.2.5.0/LICENSE',
    sha256: '4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2',
    target: 'models/LICENSE.txt',
  },
  {
    file: `onnxruntime-${ORT_VERSION}-LICENSE.txt`,
    url: `https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/LICENSE`,
    sha256: '2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c',
    target: 'ort/LICENSE.txt',
  },
  {
    file: `onnxruntime-${ORT_VERSION}-ThirdPartyNotices.txt`,
    url: `https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/ThirdPartyNotices.txt`,
    sha256: '143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3',
    target: 'ort/ThirdPartyNotices.txt',
  },
]

const cacheDir = join(process.env.XDG_CACHE_HOME ?? join(homedir(), '.cache'), 'upscaler')
const publicDir = join(import.meta.dir, '..', 'public')
const ortDist = dirname(Bun.resolveSync('onnxruntime-web/ort-wasm-simd-threaded.wasm', import.meta.dir))

const sha256 = (bytes: Uint8Array) => new Bun.CryptoHasher('sha256').update(bytes).digest('hex')

const download = async (name: string, url: string, expected: string) => {
  const file = Bun.file(join(cacheDir, name))
  if (await file.exists()) {
    const bytes = new Uint8Array(await file.arrayBuffer())
    if (sha256(bytes) === expected) return bytes
  }
  console.log(`assets: downloading ${url}`)
  const response = await fetch(url)
  if (!response.ok) throw new Error(`assets: ${url} answered ${response.status}`)
  const bytes = new Uint8Array(await response.arrayBuffer())
  const actual = sha256(bytes)
  if (actual !== expected) throw new Error(`assets: ${url} has sha256 ${actual}, not ${expected}`)
  await Bun.write(file, bytes)
  return bytes
}

const metadataOf = async (path: string) => {
  const file = Bun.file(path)
  if (!(await file.exists())) return undefined
  const length = Number(new DataView(await file.slice(0, 8).arrayBuffer()).getBigUint64(0, true))
  const header = JSON.parse(await file.slice(8, 8 + length).text()) as Record<string, Record<string, string>>
  return header['__metadata__']
}

// Convolution kernels ship as fp16, like the official ncnn release; biases and PReLU slopes stay fp32
const convert = (pth: Uint8Array<ArrayBuffer>, metadata: Record<string, string>) => {
  const tensors: Tensors = new Map()
  for (const [name, { shape, data }] of readPth(pth)) {
    tensors.set(
      name,
      shape.length === 4
        ? { dtype: 'F16', shape, data: new Uint8Array(halfArray(data).buffer) }
        : { dtype: 'F32', shape, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) },
    )
  }
  return writeSafetensors(tensors, metadata)
}

mkdirSync(cacheDir, { recursive: true })
mkdirSync(join(publicDir, 'models'), { recursive: true })
mkdirSync(join(publicDir, 'ort'), { recursive: true })

for (const [name, { url, sha256: expected }] of Object.entries(CHECKPOINTS)) {
  const path = join(publicDir, 'models', `${name}.safetensors`)
  const metadata = { source: url, sha256: expected, converter: CONVERTER }
  const existing = await metadataOf(path)
  if (existing?.sha256 === expected && existing.converter === CONVERTER) continue
  await Bun.write(path, convert(await download(`${name}.pth`, url, expected), metadata))
  console.log(`assets: ${path}`)
}

// node:fs, as Bun.write copies nothing to a file whose size was read while it did not exist
for (const name of ORT_FILES) {
  const source = join(ortDist, name)
  const target = join(publicDir, 'ort', name)
  if (!existsSync(target) || statSync(target).size !== statSync(source).size) copyFileSync(source, target)
}

const installed = (JSON.parse(readFileSync(join(ortDist, '..', 'package.json'), 'utf8')) as { version: string }).version
if (installed !== ORT_VERSION)
  throw new Error(
    `assets: onnxruntime-web is ${installed}, its notices are ${ORT_VERSION}'s: update ORT_VERSION and their sha256`,
  )

for (const { file, url, sha256: expected, target } of NOTICES) {
  const path = join(publicDir, target)
  if (existsSync(path) && sha256(readFileSync(path)) === expected) continue
  await Bun.write(path, await download(file, url, expected))
}
