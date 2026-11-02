// The main thread's handle on the upscale worker: one worker per backend, `auto` falling back from WebGPU
// to wasm. Switching backend replaces the worker, which also frees its GPU memory.

import type { Backend, Stage, UpscaleOptions, WorkerRequest, WorkerResponse } from './worker-protocol'

export type Capabilities = { backend: Backend; device: string; fp16: boolean; threads: number }

export type JobEvents = {
  progress: (stage: Stage, done: number, total: number) => void
  tile: (x: number, y: number, width: number, height: number, pixels: Uint8ClampedArray<ArrayBuffer>) => void
}

export type JobResult = { blob: Blob; width: number; height: number; timings: Record<string, number> }

export class CancelledJob extends Error {}

type Pending = { events: JobEvents; resolve: (result: JobResult) => void; reject: (error: Error) => void }

export class Engine {
  private worker: Worker | undefined
  private jobs = new Map<number, Pending>()
  private nextId = 0

  async start(preference: Backend | 'auto'): Promise<Capabilities> {
    this.stop()
    const reasons: string[] = []
    for (const backend of preference === 'auto' ? (['webgpu', 'wasm'] as const) : [preference]) {
      const started = await this.spawn(backend)
      if (typeof started !== 'string') return started
      reasons.push(started)
    }
    throw new Error(reasons.join('; '))
  }

  // Resolves with the capabilities, or the reason the backend is unavailable
  private spawn(backend: Backend) {
    const worker = new Worker(new URL('./upscale.worker.ts', import.meta.url), { type: 'module' })
    return new Promise<Capabilities | string>(resolve => {
      const onMessage = ({ data }: MessageEvent<WorkerResponse>) => {
        if (data.type === 'ready') {
          worker.removeEventListener('message', onMessage)
          worker.addEventListener('message', this.onMessage)
          this.worker = worker
          resolve(data)
        } else if (data.type === 'unavailable') {
          worker.terminate()
          resolve(data.message)
        }
      }
      worker.addEventListener('message', onMessage)
      worker.addEventListener('error', event => {
        worker.terminate()
        resolve(event.message || 'The worker failed to start')
      })
      worker.postMessage({ type: 'init', backend } satisfies WorkerRequest)
    })
  }

  private onMessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (!('id' in data)) return
    const job = this.jobs.get(data.id)
    if (!job) return
    if (data.type === 'progress') job.events.progress(data.stage, data.done, data.total)
    else if (data.type === 'tile') job.events.tile(data.x, data.y, data.width, data.height, data.pixels)
    else {
      this.jobs.delete(data.id)
      if (data.type === 'done') job.resolve(data)
      else job.reject(data.type === 'cancelled' ? new CancelledJob() : new Error(data.message))
    }
  }

  upscale(file: Blob, options: UpscaleOptions, events: JobEvents) {
    const id = ++this.nextId
    const result = new Promise<JobResult>((resolve, reject) => {
      if (!this.worker) return reject(new Error('The upscaler is not ready'))
      this.jobs.set(id, { events, resolve, reject })
      this.worker.postMessage({ type: 'upscale', id, file, options } satisfies WorkerRequest)
    })
    return { id, result }
  }

  cancel(id: number) {
    this.worker?.postMessage({ type: 'cancel', id } satisfies WorkerRequest)
  }

  stop() {
    this.worker?.terminate()
    this.worker = undefined
    for (const job of this.jobs.values()) job.reject(new CancelledJob())
    this.jobs.clear()
  }
}
