// Tiling and test-time augmentation. Every tile has one size, so the model runs on one input shape: the
// last tile of a row or column is moved back inside the image and only writes what the previous one did not.

export type Span = { start: number; from: number; to: number }

// The fewest tiles of at most `max` px that cover `size`, made equal
export const tileSize = (size: number, max: number) => Math.ceil(size / Math.ceil(size / max))

// A tile reads [start, start + tile) plus its overlap and writes [from, to)
export const spans = (size: number, tile: number): Span[] => {
  const result: Span[] = []
  for (let i = 0; i < Math.ceil(size / tile); i++) {
    const start = Math.min(i * tile, Math.max(0, size - tile))
    result.push({ start, from: i ? result[i - 1]!.to : 0, to: Math.min(start + tile, size) })
  }
  return result
}

// Mirrors a coordinate into [0, n) without repeating the edge (reflect-101), however far outside it is
export const reflect = (i: number, n: number) => {
  if (n === 1) return 0
  const period = 2 * (n - 1)
  const j = ((i % period) + period) % period
  return j < n ? j : period - j
}

// Copies a window of the image into planar float channels in [0, 1], mirrored past the borders like
// realesrgan-ncnn-vulkan's preprocessing. `channels` picks the RGBA channel read into each plane.
export const readWindow = (
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  left: number,
  top: number,
  out: Float32Array,
  windowWidth: number,
  windowHeight: number,
  channels: readonly number[],
) => {
  const plane = windowWidth * windowHeight
  const columns = Int32Array.from({ length: windowWidth }, (_, x) => reflect(left + x, width) * 4)
  for (let y = 0; y < windowHeight; y++) {
    const row = reflect(top + y, height) * width * 4
    for (let x = 0; x < windowWidth; x++) {
      const source = row + columns[x]!
      const target = y * windowWidth + x
      for (let c = 0; c < channels.length; c++) out[c * plane + target] = pixels[source + channels[c]!]! / 255
    }
  }
}

// TTA transform t of an n×n plane: bit 2 transposes, then bit 0 flips x and bit 1 flips y
export const transformPlane = (source: Float32Array, target: Float32Array, n: number, t: number) => {
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let tx = t & 4 ? y : x
      let ty = t & 4 ? x : y
      if (t & 1) tx = n - 1 - tx
      if (t & 2) ty = n - 1 - ty
      target[ty * n + tx] = source[y * n + x]!
    }
  }
}

// Adds the output of a transformed input back in the original orientation: model(T(x)) ≈ T(model(x))
export const addUntransformed = (output: Float32Array, sum: Float32Array, n: number, t: number) => {
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let tx = t & 4 ? y : x
      let ty = t & 4 ? x : y
      if (t & 1) tx = n - 1 - tx
      if (t & 2) ty = n - 1 - ty
      sum[y * n + x]! += output[ty * n + tx]!
    }
  }
}
