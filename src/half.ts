// IEEE 754 half <-> single precision, round-to-nearest-even (the weights ship as fp16)

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

export const toHalf = (value: number): number => {
  f32[0] = value
  const bits = u32[0]!
  const sign = (bits >>> 16) & 0x8000
  const exponent = (bits >>> 23) & 0xff
  const mantissa = bits & 0x7fffff
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0)
  const e = exponent - 112
  if (e >= 0x1f) return sign | 0x7c00
  if (e <= 0) {
    // Subnormal half: the implicit bit joins the mantissa, shifted by how far below the normal range it is
    const shift = 14 - e
    if (shift > 24) return sign
    const full = mantissa | 0x800000
    const rest = full & ((1 << shift) - 1)
    const halfway = 1 << (shift - 1)
    let half = full >>> shift
    if (rest > halfway || (rest === halfway && half & 1)) half++
    return sign | half
  }
  let half = sign | (e << 10) | (mantissa >>> 13)
  const rest = mantissa & 0x1fff
  // A carry out of the mantissa bumps the exponent, which is the correct rounding
  if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++
  return half
}

export const fromHalf = (half: number): number => {
  const sign = half & 0x8000 ? -1 : 1
  const exponent = (half >>> 10) & 0x1f
  const mantissa = half & 0x3ff
  if (exponent === 0) return sign * mantissa * 2 ** -24
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15)
}

export const halfArray = (values: Float32Array): Uint16Array => {
  const out = new Uint16Array(values.length)
  for (let i = 0; i < values.length; i++) out[i] = toHalf(values[i]!)
  return out
}

export const floatArray = (halves: Uint16Array): Float32Array => {
  const out = new Float32Array(halves.length)
  for (let i = 0; i < halves.length; i++) out[i] = fromHalf(halves[i]!)
  return out
}
