// Separable resampling from the model's native 4x to the requested size, streamed by rows: the tiles
// arrive band by band, so only a few horizontally resampled rows are ever held, never the native image.

export type Filter = 'lanczos3' | 'mitchell' | 'catmull-rom'

type Kernel = { support: number; weight: (x: number) => number }

// Mitchell–Netravali cubics: B = C = 1/3 is Mitchell, B = 0, C = 1/2 is Catmull-Rom
const cubic = (b: number, c: number): Kernel => ({
  support: 2,
  weight: x => {
    x = Math.abs(x)
    if (x < 1) return ((12 - 9 * b - 6 * c) * x ** 3 + (-18 + 12 * b + 6 * c) * x ** 2 + (6 - 2 * b)) / 6
    if (x < 2)
      return ((-b - 6 * c) * x ** 3 + (6 * b + 30 * c) * x ** 2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6
    return 0
  },
})

const KERNELS: Record<Filter, Kernel> = {
  lanczos3: {
    support: 3,
    weight: x => {
      if (x === 0) return 1
      if (Math.abs(x) >= 3) return 0
      const px = Math.PI * x
      return (3 * Math.sin(px) * Math.sin(px / 3)) / (px * px)
    },
  },
  mitchell: cubic(1 / 3, 1 / 3),
  'catmull-rom': cubic(0, 1 / 2),
}

type Contributions = { first: Int32Array; count: Int32Array; weights: Float32Array; taps: number }

// For each output pixel, the input pixels it reads and their weights. Shrinking widens the kernel by the
// ratio (no aliasing); the window is clipped at the borders and renormalized. One size is a plain copy.
export const contributions = (inSize: number, outSize: number, filter: Filter): Contributions => {
  const { support, weight } = KERNELS[filter]
  const scale = Math.max(1, inSize / outSize)
  const reach = inSize === outSize ? 0.5 : support * scale
  const taps = Math.ceil(reach) * 2 + 1
  const first = new Int32Array(outSize)
  const count = new Int32Array(outSize)
  const weights = new Float32Array(outSize * taps)
  for (let o = 0; o < outSize; o++) {
    const center = ((o + 0.5) * inSize) / outSize
    const lo = Math.max(0, Math.floor(center - reach + 0.5))
    const hi = Math.min(inSize, Math.ceil(center + reach - 0.5))
    let sum = 0
    for (let i = lo; i < hi; i++) {
      const w = inSize === outSize ? 1 : weight((i + 0.5 - center) / scale)
      weights[o * taps + i - lo] = w
      sum += w
    }
    for (let i = 0; i < hi - lo; i++) weights[o * taps + i]! /= sum
    first[o] = lo
    count[o] = hi - lo
  }
  return { first, count, weights, taps }
}

export class RowResampler {
  private horizontal: Contributions
  private vertical: Contributions
  private ring: Float32Array[]
  private sum: Float32Array
  private next = 0

  // Resamples RGBA channels [offset, offset + channels) of the pushed rows into the same channels of `out`
  constructor(
    private inWidth: number,
    inHeight: number,
    private outWidth: number,
    private outHeight: number,
    filter: Filter,
    private out: Uint8ClampedArray,
    private offset: number,
    private channels: number,
  ) {
    this.horizontal = contributions(inWidth, outWidth, filter)
    this.vertical = contributions(inHeight, outHeight, filter)
    const rows = this.vertical.count.reduce((max, count) => Math.max(max, count), 1)
    this.ring = Array.from({ length: rows }, () => new Float32Array(outWidth * channels))
    this.sum = new Float32Array(outWidth * channels)
  }

  // Takes input rows [y, y + count), in order, from an RGBA buffer `inWidth` wide
  push(rows: Uint8ClampedArray, y: number, count: number) {
    const { first, count: taps } = this.vertical
    for (let r = 0; r < count; r++) {
      this.resampleRow(rows, r * this.inWidth * 4, this.ring[(y + r) % this.ring.length]!)
      while (this.next < this.outHeight && first[this.next]! + taps[this.next]! <= y + r + 1) this.emitRow(this.next++)
    }
  }

  private resampleRow(rows: Uint8ClampedArray, offset: number, target: Float32Array) {
    const { first, count, weights, taps } = this.horizontal
    const channels = this.channels
    for (let x = 0; x < this.outWidth; x++) {
      const base = offset + first[x]! * 4 + this.offset
      for (let c = 0; c < channels; c++) {
        let sum = 0
        for (let i = 0; i < count[x]!; i++) sum += rows[base + i * 4 + c]! * weights[x * taps + i]!
        target[x * channels + c] = sum
      }
    }
  }

  private emitRow(y: number) {
    const { first, count, weights, taps } = this.vertical
    const sum = this.sum.fill(0)
    for (let i = 0; i < count[y]!; i++) {
      const row = this.ring[(first[y]! + i) % this.ring.length]!
      const w = weights[y * taps + i]!
      for (let j = 0; j < sum.length; j++) sum[j]! += row[j]! * w
    }
    const channels = this.channels
    const offset = y * this.outWidth * 4 + this.offset
    for (let x = 0; x < this.outWidth; x++)
      for (let c = 0; c < channels; c++) this.out[offset + x * 4 + c] = sum[x * channels + c]!
  }
}
