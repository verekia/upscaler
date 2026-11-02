import { expect, test } from 'bun:test'
import { inflateSync } from 'node:zlib'

import { encodePng } from './png'

// Decodes what encodePng writes: checks the signature and every chunk's CRC, then undoes the filters
const decode = async (blob: Blob) => {
  const bytes = new Uint8Array(await blob.arrayBuffer())
  expect([...bytes.subarray(0, 8)]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const view = new DataView(bytes.buffer)
  const idat: Uint8Array[] = []
  let header = new Uint8Array()
  for (let offset = 8; offset < bytes.length; ) {
    const length = view.getUint32(offset)
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8))
    const data = bytes.subarray(offset + 8, offset + 8 + length)
    expect(view.getUint32(offset + 8 + length)).toBe(Bun.hash.crc32(bytes.subarray(offset + 4, offset + 8 + length)))
    if (type === 'IHDR') header = data
    if (type === 'IDAT') idat.push(data)
    offset += 12 + length
  }
  const width = new DataView(header.buffer, header.byteOffset).getUint32(0)
  const height = new DataView(header.buffer, header.byteOffset).getUint32(4)
  const bpp = header[9] === 6 ? 4 : 3
  const raw = inflateSync(Buffer.concat(idat))
  const stride = width * bpp
  const pixels = new Uint8Array(height * stride)
  for (let y = 0; y < height; y++) {
    const type = raw[y * (stride + 1)]!
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? pixels[y * stride + i - bpp]! : 0
      const b = y ? pixels[(y - 1) * stride + i]! : 0
      const c = i >= bpp && y ? pixels[(y - 1) * stride + i - bpp]! : 0
      const p = a + b - c
      const paeth =
        Math.abs(p - a) <= Math.abs(p - b) && Math.abs(p - a) <= Math.abs(p - c)
          ? a
          : Math.abs(p - b) <= Math.abs(p - c)
            ? b
            : c
      const predicted = [0, a, b, (a + b) >> 1, paeth][type]!
      pixels[y * stride + i] = (raw[y * (stride + 1) + 1 + i]! + predicted) & 0xff
    }
  }
  return { width, height, bpp, pixels }
}

test('round-trips RGBA, and writes RGB when every pixel is opaque', async () => {
  // Smooth with noise, so every filter type gets picked somewhere
  const rgba = Uint8ClampedArray.from({ length: 37 * 23 * 4 }, (_, i) =>
    i % 4 === 3 ? 255 - (i % 7) : ((i >> 2) % 37) * 5 + (i % 13),
  )
  const decoded = await decode(await encodePng(rgba, 37, 23))
  expect(decoded).toMatchObject({ width: 37, height: 23, bpp: 4 })
  expect([...decoded.pixels]).toEqual([...rgba])

  for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255
  const opaque = await decode(await encodePng(rgba, 37, 23))
  expect(opaque.bpp).toBe(3)
  expect([...opaque.pixels]).toEqual([...rgba].filter((_, i) => i % 4 !== 3))
})
