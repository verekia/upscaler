// Reads the state dict of a PyTorch checkpoint (.pth): a zip holding a pickle and the raw tensor storages.
// Only the pickle opcodes and globals torch.save writes for a dict of float tensors are understood.

import { floatArray } from '../src/half'

export type StateDict = Map<string, { shape: number[]; data: Float32Array }>

type Storage = { dtype: 'F32' | 'F16'; key: string }
type RebuiltTensor = { storage: Storage; offset: number; shape: number[]; stride: number[] }

class Global {
  constructor(
    readonly module: string,
    readonly name: string,
  ) {}
}

const MARK = Symbol('mark')
const decoder = new TextDecoder()

const readZip = (bytes: Uint8Array<ArrayBuffer>): Map<string, Uint8Array> => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let end = bytes.length - 22
  while (end >= 0 && view.getUint32(end, true) !== 0x06054b50) end--
  if (end < 0) throw new Error('pth: not a zip file (no end of central directory)')
  let count = view.getUint16(end + 10, true)
  let offset = view.getUint32(end + 16, true)
  if (offset === 0xffffffff || count === 0xffff) {
    const zip64 = Number(view.getBigUint64(end - 12, true))
    count = Number(view.getBigUint64(zip64 + 32, true))
    offset = Number(view.getBigUint64(zip64 + 48, true))
  }
  const entries = new Map<string, Uint8Array>()
  for (let i = 0; i < count; i++) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error('pth: corrupt central directory')
    const method = view.getUint16(offset + 10, true)
    let compressed = view.getUint32(offset + 20, true)
    const uncompressed = view.getUint32(offset + 24, true)
    const nameLength = view.getUint16(offset + 28, true)
    const extraLength = view.getUint16(offset + 30, true)
    const commentLength = view.getUint16(offset + 32, true)
    let local = view.getUint32(offset + 42, true)
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength))
    // A zip64 extra field carries the 64-bit values of the fields set to 0xffffffff, in order
    for (let p = offset + 46 + nameLength; p < offset + 46 + nameLength + extraLength; ) {
      const id = view.getUint16(p, true)
      const length = view.getUint16(p + 2, true)
      if (id === 0x0001) {
        let q = p + 4
        if (uncompressed === 0xffffffff) q += 8
        if (compressed === 0xffffffff) {
          compressed = Number(view.getBigUint64(q, true))
          q += 8
        }
        if (local === 0xffffffff) local = Number(view.getBigUint64(q, true))
      }
      p += 4 + length
    }
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true)
    const data = bytes.subarray(start, start + compressed)
    entries.set(name, method === 0 ? data : Bun.inflateSync(data))
    offset += 46 + nameLength + extraLength + commentLength
  }
  return entries
}

const call = (fn: Global, args: unknown[]): unknown => {
  const name = `${fn.module}.${fn.name}`
  if (name === 'collections.OrderedDict') return new Map()
  if (name === 'torch._utils._rebuild_tensor_v2') {
    const [storage, offset, shape, stride] = args as [Storage, number, number[], number[]]
    return { storage, offset, shape, stride } satisfies RebuiltTensor
  }
  throw new Error(`pth: unsupported pickle global ${name}`)
}

const persistentLoad = (pid: unknown): Storage => {
  const [kind, type, key] = pid as [string, Global, string]
  if (kind !== 'storage') throw new Error(`pth: unsupported persistent id ${kind}`)
  if (type.name === 'FloatStorage') return { dtype: 'F32', key }
  if (type.name === 'HalfStorage') return { dtype: 'F16', key }
  throw new Error(`pth: unsupported storage ${type.name}`)
}

const unpickle = (bytes: Uint8Array): unknown => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const stack: unknown[] = []
  const memo = new Map<number, unknown>()
  let pos = 0
  const popMark = () => {
    const mark = stack.lastIndexOf(MARK)
    const items = stack.splice(mark + 1)
    stack.pop()
    return items
  }
  const line = () => {
    const end = bytes.indexOf(10, pos)
    const text = decoder.decode(bytes.subarray(pos, end))
    pos = end + 1
    return text
  }
  for (;;) {
    const op = bytes[pos++]
    switch (op) {
      case 0x80: // PROTO
        pos++
        break
      case 0x7d: // EMPTY_DICT
        stack.push(new Map())
        break
      case 0x29: // EMPTY_TUPLE
      case 0x5d: // EMPTY_LIST
        stack.push([])
        break
      case 0x28: // MARK
        stack.push(MARK)
        break
      case 0x71: // BINPUT
        memo.set(bytes[pos++]!, stack.at(-1))
        break
      case 0x72: // LONG_BINPUT
        memo.set(view.getUint32(pos, true), stack.at(-1))
        pos += 4
        break
      case 0x68: // BINGET
        stack.push(memo.get(bytes[pos++]!))
        break
      case 0x6a: // LONG_BINGET
        stack.push(memo.get(view.getUint32(pos, true)))
        pos += 4
        break
      case 0x58: // BINUNICODE
      case 0x8c: {
        // SHORT_BINUNICODE
        const length = op === 0x58 ? view.getUint32(pos, true) : bytes[pos]!
        pos += op === 0x58 ? 4 : 1
        stack.push(decoder.decode(bytes.subarray(pos, pos + length)))
        pos += length
        break
      }
      case 0x63: // GLOBAL
        stack.push(new Global(line(), line()))
        break
      case 0x4a: // BININT
        stack.push(view.getInt32(pos, true))
        pos += 4
        break
      case 0x4b: // BININT1
        stack.push(bytes[pos++])
        break
      case 0x4d: // BININT2
        stack.push(view.getUint16(pos, true))
        pos += 2
        break
      case 0x8a: {
        // LONG1: little-endian two's complement
        const length = bytes[pos++]!
        let value = 0n
        for (let i = length - 1; i >= 0; i--) value = (value << 8n) | BigInt(bytes[pos + i]!)
        if (length && bytes[pos + length - 1]! & 0x80) value -= 1n << BigInt(length * 8)
        stack.push(Number(value))
        pos += length
        break
      }
      case 0x47: // BINFLOAT
        stack.push(view.getFloat64(pos, false))
        pos += 8
        break
      case 0x74: // TUPLE
        stack.push(popMark())
        break
      case 0x85: // TUPLE1
        stack.push([stack.pop()])
        break
      case 0x86: // TUPLE2
        stack.push(stack.splice(-2))
        break
      case 0x87: // TUPLE3
        stack.push(stack.splice(-3))
        break
      case 0x88: // NEWTRUE
      case 0x89: // NEWFALSE
        stack.push(op === 0x88)
        break
      case 0x4e: // NONE
        stack.push(null)
        break
      case 0x51: // BINPERSID
        stack.push(persistentLoad(stack.pop()))
        break
      case 0x52: {
        // REDUCE
        const args = stack.pop() as unknown[]
        stack.push(call(stack.pop() as Global, args))
        break
      }
      case 0x73: {
        // SETITEM
        const [key, value] = stack.splice(-2)
        ;(stack.at(-1) as Map<unknown, unknown>).set(key, value)
        break
      }
      case 0x75: {
        // SETITEMS
        const items = popMark()
        const dict = stack.at(-1) as Map<unknown, unknown>
        for (let i = 0; i < items.length; i += 2) dict.set(items[i], items[i + 1])
        break
      }
      case 0x61: // APPEND
        ;(stack.at(-2) as unknown[]).push(stack.pop())
        break
      case 0x65: {
        // APPENDS
        const items = popMark()
        ;(stack.at(-1) as unknown[]).push(...items)
        break
      }
      case 0x62: // BUILD: the state dict's _metadata, unused
        stack.pop()
        break
      case 0x2e: // STOP
        return stack.pop()
      default:
        throw new Error(`pth: unsupported pickle opcode 0x${op?.toString(16)} at ${pos - 1}`)
    }
  }
}

// Real-ESRGAN checkpoints keep the weights under params_ema (the EMA of training) or params
export const readPth = (bytes: Uint8Array<ArrayBuffer>): StateDict => {
  const entries = readZip(bytes)
  const pickle = [...entries.keys()].find(name => name.endsWith('/data.pkl'))
  if (!pickle) throw new Error('pth: no data.pkl in the archive')
  const root = pickle.slice(0, -'data.pkl'.length)
  const top = unpickle(entries.get(pickle)!) as Map<string, unknown>
  const params = (top.get('params_ema') ?? top.get('params') ?? top) as Map<string, RebuiltTensor>
  const state: StateDict = new Map()
  for (const [name, { storage, offset, shape, stride }] of params) {
    let expected = 1
    for (let i = shape.length - 1; i >= 0; i--) {
      if (shape[i]! > 1 && stride[i] !== expected) throw new Error(`pth: ${name} is not contiguous`)
      expected *= shape[i]!
    }
    const raw = entries.get(`${root}data/${storage.key}`)
    if (!raw) throw new Error(`pth: missing storage ${storage.key} for ${name}`)
    const bytesPer = storage.dtype === 'F32' ? 4 : 2
    const slice = raw.slice(offset * bytesPer, (offset + expected) * bytesPer)
    if (slice.byteLength !== expected * bytesPer) throw new Error(`pth: ${name} storage is too short`)
    const data = storage.dtype === 'F32' ? new Float32Array(slice.buffer) : floatArray(new Uint16Array(slice.buffer))
    state.set(name, { shape, data })
  }
  return state
}
