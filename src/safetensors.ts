// The safetensors format: an 8-byte little-endian header length, a JSON header, then every tensor's bytes back to back

import { floatArray } from './half'

export type Dtype = 'F16' | 'F32'
export type Tensor = { dtype: Dtype; shape: number[]; data: Uint8Array }
export type Tensors = Map<string, Tensor>

type HeaderEntry = { dtype: Dtype; shape: number[]; data_offsets: [number, number] }

const BYTES: Record<Dtype, number> = { F16: 2, F32: 4 }

export const readSafetensors = (buffer: ArrayBuffer): { tensors: Tensors; metadata: Record<string, string> } => {
  const headerLength = Number(new DataView(buffer).getBigUint64(0, true))
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength))) as Record<
    string,
    HeaderEntry | Record<string, string>
  >
  const tensors: Tensors = new Map()
  let metadata: Record<string, string> = {}
  for (const [name, entry] of Object.entries(header)) {
    if (name === '__metadata__') {
      metadata = entry as Record<string, string>
      continue
    }
    const {
      dtype,
      shape,
      data_offsets: [begin, end],
    } = entry as HeaderEntry
    if (!(dtype in BYTES)) throw new Error(`safetensors: ${name} has unsupported dtype ${dtype}`)
    tensors.set(name, { dtype, shape, data: new Uint8Array(buffer, 8 + headerLength + begin, end - begin) })
  }
  return { tensors, metadata }
}

export const writeSafetensors = (tensors: Tensors, metadata: Record<string, string>): Uint8Array => {
  const header: Record<string, HeaderEntry | Record<string, string>> = { __metadata__: metadata }
  let offset = 0
  for (const [name, { dtype, shape, data }] of tensors) {
    header[name] = { dtype, shape, data_offsets: [offset, offset + data.byteLength] }
    offset += data.byteLength
  }
  const json = new TextEncoder().encode(JSON.stringify(header))
  // Spaces pad the header so the tensor bytes start 8-byte aligned
  const headerLength = Math.ceil(json.byteLength / 8) * 8
  const out = new Uint8Array(8 + headerLength + offset).fill(0x20, 8, 8 + headerLength)
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true)
  out.set(json, 8)
  offset = 8 + headerLength
  for (const { data } of tensors.values()) {
    out.set(data, offset)
    offset += data.byteLength
  }
  return out
}

// Copies, so the typed array is aligned whatever the tensor's offset in the file
export const toFloat32 = ({ dtype, data }: Tensor): Float32Array =>
  dtype === 'F32'
    ? new Float32Array(data.slice().buffer)
    : floatArray(new Uint16Array(data.slice().buffer, 0, data.byteLength / 2))
