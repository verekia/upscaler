// A minimal ONNX writer: the protobuf messages of a ModelProto holding one graph of standard ops.
// Field numbers are onnx.proto's; repeated scalars are written unpacked, which every parser accepts.

export type DataType = 'float32' | 'float16'

const DATA_TYPES: Record<DataType, number> = { float32: 1, float16: 10 }
const IR_VERSION = 8
const OPSET = 17

const encoder = new TextEncoder()

// Sub-messages and tensor bytes are kept as parts and only copied once, by the outermost encode()
class Message {
  private parts: Uint8Array[] = []
  private pending: number[] = []
  private size = 0

  private flush() {
    if (!this.pending.length) return
    this.parts.push(Uint8Array.from(this.pending))
    this.size += this.pending.length
    this.pending = []
  }

  private varint(value: number) {
    // int64 fields: negative numbers take the 10-byte two's complement form
    let v = BigInt.asUintN(64, BigInt(value))
    while (v > 0x7fn) {
      this.pending.push(Number(v & 0x7fn) | 0x80)
      v >>= 7n
    }
    this.pending.push(Number(v))
  }

  private append(parts: Uint8Array[], size: number) {
    this.flush()
    // No spread: a big graph has more parts than an engine accepts as call arguments
    for (const part of parts) this.parts.push(part)
    this.size += size
  }

  int(field: number, value: number) {
    this.varint(field << 3)
    this.varint(value)
    return this
  }

  float(field: number, value: number) {
    this.varint((field << 3) | 5)
    const bytes = new Uint8Array(4)
    new DataView(bytes.buffer).setFloat32(0, value, true)
    this.pending.push(...bytes)
    return this
  }

  raw(field: number, data: Uint8Array) {
    this.varint((field << 3) | 2)
    this.varint(data.byteLength)
    this.append([data], data.byteLength)
    return this
  }

  string(field: number, value: string) {
    return this.raw(field, encoder.encode(value))
  }

  message(field: number, message: Message) {
    message.flush()
    this.varint((field << 3) | 2)
    this.varint(message.size)
    this.append(message.parts, message.size)
    return this
  }

  encode(): Uint8Array {
    this.flush()
    const out = new Uint8Array(this.size)
    let offset = 0
    for (const part of this.parts) {
      out.set(part, offset)
      offset += part.byteLength
    }
    return out
  }
}

export type Attribute = { float: number } | { int: number } | { ints: number[] } | { string: string }

const attribute = (name: string, value: Attribute) => {
  const message = new Message().string(1, name)
  if ('float' in value) return message.float(2, value.float).int(20, 1)
  if ('int' in value) return message.int(3, value.int).int(20, 2)
  if ('string' in value) return message.string(4, value.string).int(20, 3)
  for (const i of value.ints) message.int(8, i)
  return message.int(20, 7)
}

// A dimension is a size, or a name for a size only known at run time
const valueInfo = (name: string, dataType: DataType, dims: (number | string)[]) => {
  const shape = new Message()
  for (const dim of dims)
    shape.message(1, typeof dim === 'number' ? new Message().int(1, dim) : new Message().string(2, dim))
  const tensorType = new Message().int(1, DATA_TYPES[dataType]).message(2, shape)
  return new Message().string(1, name).message(2, new Message().message(1, tensorType))
}

export class Graph {
  private nodes: Message[] = []
  private initializers: Message[] = []
  private names = new Set<string>()
  private count = 0

  // Adds a node and returns the name of its single output
  op(type: string, inputs: string[], attributes: Record<string, Attribute> = {}): string {
    const output = `${type}_${this.count++}`
    const node = new Message()
    for (const input of inputs) node.string(1, input)
    node.string(2, output).string(3, output).string(4, type)
    for (const [name, value] of Object.entries(attributes)) node.message(5, attribute(name, value))
    this.nodes.push(node)
    return output
  }

  // Adds a constant tensor once per name, from its little-endian bytes
  initializer(name: string, dataType: DataType, dims: number[], data: Uint8Array): string {
    if (this.names.has(name)) return name
    this.names.add(name)
    const tensor = new Message()
    for (const dim of dims) tensor.int(1, dim)
    this.initializers.push(tensor.int(2, DATA_TYPES[dataType]).string(8, name).raw(9, data))
    return name
  }

  encode(
    input: { name: string; dataType: DataType; dims: (number | string)[] },
    output: { name: string; dataType: DataType; dims: (number | string)[] },
  ): Uint8Array {
    const graph = new Message()
    for (const node of this.nodes) graph.message(1, node)
    graph.string(2, 'upscaler')
    for (const tensor of this.initializers) graph.message(5, tensor)
    graph.message(11, valueInfo(input.name, input.dataType, input.dims))
    graph.message(12, valueInfo(output.name, output.dataType, output.dims))
    return new Message()
      .int(1, IR_VERSION)
      .string(2, 'upscaler')
      .message(7, graph)
      .message(8, new Message().string(1, '').int(2, OPSET))
      .encode()
  }
}
