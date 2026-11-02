// PNG encoder: 8-bit RGB when fully opaque, else RGBA; libpng's adaptive filter heuristic; deflate by
// CompressionStream. Unlike a canvas, it keeps the exact unpremultiplied pixels and has no size limit.

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

const chunk = (type: string, data: Uint8Array): Uint8Array[] => {
  const head = new Uint8Array(8)
  new DataView(head.buffer).setUint32(0, data.byteLength)
  for (let i = 0; i < 4; i++) head[4 + i] = type.charCodeAt(i)
  let crc = 0xffffffff
  for (const part of [head.subarray(4), data])
    for (let i = 0; i < part.length; i++) crc = CRC_TABLE[(crc ^ part[i]!) & 0xff]! ^ (crc >>> 8)
  const tail = new Uint8Array(4)
  new DataView(tail.buffer).setUint32(0, (crc ^ 0xffffffff) >>> 0)
  return [head, data, tail]
}

// A filter residual as a signed byte's magnitude, the heuristic's cost
const cost = (value: number) => {
  value &= 0xff
  return value < 128 ? value : 256 - value
}

const paeth = (a: number, b: number, c: number) => {
  const pa = Math.abs(b - c)
  const pb = Math.abs(a - c)
  const pc = Math.abs(a + b - 2 * c)
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c
}

// Writes the filter type byte and the filtered row, picking the filter with the smallest residuals
const filterRow = (row: Uint8Array, prev: Uint8Array, bpp: number, out: Uint8Array, offset: number) => {
  const costs = [0, 0, 0, 0, 0]
  for (let i = 0; i < row.length; i++) {
    const x = row[i]!
    const a = i >= bpp ? row[i - bpp]! : 0
    const b = prev[i]!
    const c = i >= bpp ? prev[i - bpp]! : 0
    costs[0]! += cost(x)
    costs[1]! += cost(x - a)
    costs[2]! += cost(x - b)
    costs[3]! += cost(x - ((a + b) >> 1))
    costs[4]! += cost(x - paeth(a, b, c))
  }
  const type = costs.indexOf(Math.min(...costs))
  out[offset] = type
  for (let i = 0; i < row.length; i++) {
    const x = row[i]!
    const a = i >= bpp ? row[i - bpp]! : 0
    const b = prev[i]!
    const predicted =
      type === 0
        ? 0
        : type === 1
          ? a
          : type === 2
            ? b
            : type === 3
              ? (a + b) >> 1
              : paeth(a, b, i >= bpp ? prev[i - bpp]! : 0)
    out[offset + 1 + i] = x - predicted
  }
}

export const encodePng = async (rgba: Uint8ClampedArray, width: number, height: number): Promise<Blob> => {
  let opaque = true
  for (let i = 3; i < rgba.length && opaque; i += 4) opaque = rgba[i] === 255
  const bpp = opaque ? 3 : 4
  const stride = width * bpp

  const stream = new CompressionStream('deflate')
  const compressed = (async () => {
    const parts: Uint8Array[] = []
    const reader = stream.readable.getReader()
    for (let read = await reader.read(); !read.done; read = await reader.read()) parts.push(read.value)
    return parts
  })()
  const writer = stream.writable.getWriter()
  let row = new Uint8Array(stride)
  let prev = new Uint8Array(stride)
  const rowsPerBatch = Math.max(1, Math.floor(4_000_000 / (stride + 1)))
  for (let y = 0; y < height; y += rowsPerBatch) {
    const batch = new Uint8Array(Math.min(rowsPerBatch, height - y) * (stride + 1))
    for (let r = 0; r * (stride + 1) < batch.length; r++) {
      const source = (y + r) * width * 4
      if (bpp === 4) row.set(rgba.subarray(source, source + stride))
      else
        for (let x = 0; x < width; x++) {
          row[x * 3] = rgba[source + x * 4]!
          row[x * 3 + 1] = rgba[source + x * 4 + 1]!
          row[x * 3 + 2] = rgba[source + x * 4 + 2]!
        }
      filterRow(row, prev, bpp, batch, r * (stride + 1))
      ;[row, prev] = [prev, row]
    }
    await writer.write(batch)
  }
  await writer.close()

  const header = new Uint8Array(13)
  const view = new DataView(header.buffer)
  view.setUint32(0, width)
  view.setUint32(4, height)
  header[8] = 8
  header[9] = opaque ? 2 : 6
  const signature = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)
  const idat = (await compressed).flatMap(part => chunk('IDAT', part))
  return new Blob([signature, ...chunk('IHDR', header), ...idat, ...chunk('IEND', new Uint8Array())] as BlobPart[], {
    type: 'image/png',
  })
}
