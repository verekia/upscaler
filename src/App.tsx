import { useEffect, useRef, useState } from 'react'
import Head from 'next/head'

import { Settings } from './components/Settings'
import { Viewer } from './components/Viewer'
import { CancelledJob, Engine } from './engine'
import { formatBytes, formatDuration } from './format'
import { modelById, NATIVE_SCALE } from './models'
import { loadOptions, outputSize, resolvedPrecision, saveOptions, sizeProblem, workerOptions } from './options'
import { hasTransparency } from './upscale'

import type { Capabilities } from './engine'
import type { Format, Stage } from './worker-protocol'

type Source = { file: File; url: string; name: string; width: number; height: number; transparent: boolean }
type Job = {
  id: number
  stage: Stage
  done: number
  total: number
  started: number
  // When the first and the latest tiles came back: the ETA leaves the first out, as it compiles the shaders
  firstTile?: number
  lastTile?: number
}
type Result = {
  url: string
  blob: Blob
  width: number
  height: number
  filename: string
  format: Format
  elapsed: number
  timings: Record<string, number>
  summary: string
}
type EngineState =
  | { status: 'starting' }
  | { status: 'ready'; capabilities: Capabilities }
  | { status: 'failed'; message: string }

const STAGES: Record<Stage, string> = {
  decode: 'Reading the image',
  download: 'Downloading the model',
  compile: 'Preparing the model',
  upscale: 'Upscaling',
  encode: 'Encoding',
}
const EXTENSIONS: Record<Format, string> = { png: 'png', jpeg: 'jpg', webp: 'webp' }
// The live preview canvas is the output size, up to this many px on its long side
const PREVIEW_MAX = 4096

const Logo = () => (
  <svg viewBox="0 0 24 24" aria-hidden className="text-accent size-5">
    <path d="M3 21V11h4v6h6v4z" fill="currentColor" opacity="0.45" />
    <path d="M3 9V3h18v18h-6v-4h2V7H7v2z" fill="currentColor" />
  </svg>
)

const EngineChip = ({ state }: { state: EngineState }) => {
  if (state.status === 'starting')
    return (
      <span className="text-fg-faint flex items-center gap-2 text-[11.5px]">
        <span className="bg-fg-faint size-1.5 animate-pulse rounded-full" />
        Starting…
      </span>
    )
  if (state.status === 'failed')
    return (
      <span className="text-danger flex items-center gap-2 text-[11.5px]" title={state.message}>
        <span className="bg-danger size-1.5 rounded-full" />
        Unavailable
      </span>
    )
  const { backend, device, fp16, threads } = state.capabilities
  return (
    <span
      className="text-fg-muted flex min-w-0 items-center gap-2 text-[11.5px]"
      title={
        backend === 'webgpu'
          ? `WebGPU on ${device || 'the GPU'}${fp16 ? ', half precision available' : ''}`
          : `WebAssembly on the CPU, ${threads} thread${threads > 1 ? 's' : ''}: much slower than WebGPU`
      }
    >
      <span className={`size-1.5 shrink-0 rounded-full ${backend === 'webgpu' ? 'bg-ok' : 'bg-[#f5b94a]'}`} />
      <span className="truncate">
        {backend === 'webgpu'
          ? `WebGPU${device ? ` · ${device}` : ''}`
          : `CPU · ${threads} thread${threads > 1 ? 's' : ''}`}
      </span>
    </span>
  )
}

export const App = () => {
  const [options, setOptions] = useState(loadOptions)
  const [source, setSource] = useState<Source>()
  const [engineState, setEngineState] = useState<EngineState>({ status: 'starting' })
  const [job, setJob] = useState<Job>()
  const [result, setResult] = useState<Result>()
  const [error, setError] = useState<string>()
  const [dragging, setDragging] = useState(false)
  const [now, setNow] = useState(() => performance.now())
  const engine = useRef<Engine>(undefined)
  const preview = useRef<HTMLCanvasElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  // The file on display, so a job that finishes after another image was opened drops its result
  const shown = useRef<File>(undefined)

  const capabilities = engineState.status === 'ready' ? engineState.capabilities : undefined
  const output = source && outputSize(options, source.width, source.height)
  const problem = output && sizeProblem(output.width, output.height)

  useEffect(() => saveOptions(options), [options])

  // A worker per backend choice; switching backend replaces it (and cancels a running job)
  useEffect(() => {
    const instance = new Engine()
    engine.current = instance
    let active = true
    setEngineState({ status: 'starting' })
    instance.start(options.backend).then(
      started => active && setEngineState({ status: 'ready', capabilities: started }),
      (reason: Error) => active && setEngineState({ status: 'failed', message: reason.message }),
    )
    return () => {
      active = false
      instance.stop()
    }
  }, [options.backend])

  // Ticks the elapsed time and ETA while a job runs
  const running = job?.id
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(performance.now()), 250)
    return () => clearInterval(timer)
  }, [running])

  const open = async (file: File | undefined) => {
    if (!file) return
    if (job) engine.current?.cancel(job.id)
    setError(undefined)
    try {
      const bitmap = await createImageBitmap(file)
      const { width, height } = bitmap
      const context = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true })!
      context.drawImage(bitmap, 0, 0)
      bitmap.close()
      const transparent = hasTransparency(context.getImageData(0, 0, width, height).data)
      if (source) URL.revokeObjectURL(source.url)
      if (result) URL.revokeObjectURL(result.url)
      shown.current = file
      setSource({ file, url: URL.createObjectURL(file), name: file.name, width, height, transparent })
      setResult(undefined)
    } catch {
      setError(`${file.name} is not an image this browser can read.`)
    }
  }

  // Drop anywhere, or paste
  useEffect(() => {
    const hasFiles = (event: DragEvent) => event.dataTransfer?.types.includes('Files')
    const onDragOver = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      setDragging(true)
    }
    const onDragLeave = (event: DragEvent) => event.relatedTarget === null && setDragging(false)
    const onDrop = (event: DragEvent) => {
      event.preventDefault()
      setDragging(false)
      void open(event.dataTransfer?.files[0])
    }
    const onPaste = (event: ClipboardEvent) =>
      void open([...(event.clipboardData?.files ?? [])].find(file => file.type.startsWith('image/')))
    window.addEventListener('dragover', onDragOver)
    window.addEventListener('dragleave', onDragLeave)
    window.addEventListener('drop', onDrop)
    window.addEventListener('paste', onPaste)
    return () => {
      window.removeEventListener('dragover', onDragOver)
      window.removeEventListener('dragleave', onDragLeave)
      window.removeEventListener('drop', onDrop)
      window.removeEventListener('paste', onPaste)
    }
  })

  const start = async () => {
    if (!source || !capabilities || !output || problem || job || !engine.current) return
    const settings = workerOptions(options, capabilities, output.width, output.height)
    const canvas = preview.current!
    const k = Math.min(1, PREVIEW_MAX / Math.max(output.width, output.height))
    canvas.width = Math.max(1, Math.round(output.width * k))
    canvas.height = Math.max(1, Math.round(output.height * k))
    const context = canvas.getContext('2d')!
    // Tiles come in native 4x pixels
    const toCanvas = canvas.width / (source.width * NATIVE_SCALE)

    setError(undefined)
    const started = performance.now()
    const { id, result: finished } = engine.current.upscale(source.file, settings, {
      progress: (stage, done, total) =>
        setJob(current =>
          current?.id === id
            ? {
                ...current,
                stage,
                done,
                total,
                firstTile: current.firstTile ?? (done ? performance.now() : undefined),
                lastTile: performance.now(),
              }
            : current,
        ),
      tile: (x, y, width, height, pixels) =>
        void createImageBitmap(new ImageData(pixels, width, height)).then(bitmap => {
          context.drawImage(bitmap, x * toCanvas, y * toCanvas, width * toCanvas, height * toCanvas)
          bitmap.close()
        }),
    })
    setNow(started)
    setJob({ id, stage: 'decode', done: 0, total: 1, started })
    try {
      const done = await finished
      if (shown.current !== source.file) return
      if (result) URL.revokeObjectURL(result.url)
      const model = modelById(settings.model)
      setResult({
        url: URL.createObjectURL(done.blob),
        blob: done.blob,
        width: done.width,
        height: done.height,
        filename: `${source.name.replace(/\.[^.]+$/, '')}-up.${EXTENSIONS[settings.format]}`,
        format: settings.format,
        elapsed: performance.now() - started,
        timings: done.timings,
        summary: [
          model.label,
          model.checkpoints.length > 1 && `denoise ${settings.denoise.toFixed(2)}`,
          settings.tta && 'TTA',
          capabilities.backend === 'webgpu' ? `WebGPU ${settings.precision.toUpperCase()}` : 'CPU',
        ]
          .filter(Boolean)
          .join(' · '),
      })
    } catch (reason) {
      if (!(reason instanceof CancelledJob)) setError((reason as Error).message)
    } finally {
      setJob(current => (current?.id === id ? undefined : current))
    }
  }

  const notice =
    error ?? (engineState.status === 'failed' ? `The upscaler could not start: ${engineState.message}` : undefined)
  const progress = job && (job.stage === 'download' || job.stage === 'upscale') && job.total ? job.done / job.total : 0
  const eta =
    job?.stage === 'upscale' && job.firstTile && job.lastTile && job.done > 1
      ? Math.max(0, ((job.lastTile - job.firstTile) / (job.done - 1)) * (job.total - job.done) - (now - job.lastTile))
      : undefined

  return (
    <>
      <Head>
        <title>Upscaler — Real-ESRGAN in your browser</title>
        <meta name="description" content="AI image upscaling with Real-ESRGAN, running entirely in your browser." />
      </Head>
      <div className="flex h-full flex-col">
        <header className="border-line bg-ink-1 flex h-12 shrink-0 items-center justify-between gap-4 border-b px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <Logo />
            <h1 className="text-[14px] font-semibold tracking-tight">Upscaler</h1>
            <span className="text-fg-faint hidden truncate text-[12px] sm:inline">
              Real-ESRGAN, entirely in your browser
            </span>
          </div>
          <div className="flex min-w-0 items-center gap-4">
            <EngineChip state={engineState} />
            <a
              href="/licenses"
              target="_blank"
              rel="noreferrer"
              className="text-fg-faint hover:text-fg shrink-0 text-[11.5px]"
            >
              Licenses
            </a>
            <a
              href="https://github.com/verekia/upscaler"
              target="_blank"
              rel="noreferrer"
              className="text-fg-faint hover:text-fg shrink-0 text-[11.5px]"
            >
              GitHub ↗
            </a>
          </div>
        </header>

        <div className="flex min-h-0 flex-1 flex-col-reverse md:flex-row">
          <aside className="border-line bg-ink-1 flex min-h-0 flex-1 flex-col md:w-[324px] md:flex-none md:border-r">
            <div className="border-line flex items-center gap-3 border-b px-4 py-3">
              {source ? (
                <>
                  <img
                    src={source.url}
                    alt=""
                    className="border-line size-10 shrink-0 rounded border object-cover"
                    style={{ imageRendering: 'auto' }}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[12.5px] font-medium" title={source.name}>
                      {source.name}
                    </div>
                    <div className="text-fg-faint font-mono text-[11px]">
                      {source.width} × {source.height} · {formatBytes(source.file.size)}
                      {source.transparent && ' · alpha'}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-fg-faint min-w-0 flex-1 text-[12px]">No image yet</div>
              )}
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                className="border-line bg-ink-2 text-fg-muted hover:border-line-strong hover:text-fg shrink-0 rounded-md border px-2.5 py-1 text-[12px]"
              >
                {source ? 'Replace' : 'Open…'}
              </button>
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={event => {
                  void open(event.target.files?.[0])
                  event.target.value = ''
                }}
              />
            </div>

            <div className="min-h-0 flex-1 scrollbar-thin overflow-y-auto">
              <Settings
                options={options}
                onChange={patch => setOptions(current => ({ ...current, ...patch }))}
                source={source}
                capabilities={capabilities}
                busy={!!job}
              />
              <p className="text-fg-faint px-4 py-4 text-[11px] leading-relaxed">
                The image never leaves your device: the model weights download once, then everything runs locally with
                ONNX Runtime ({capabilities?.backend === 'wasm' ? 'WebAssembly' : 'WebGPU or WebAssembly'}). Real-ESRGAN
                by Xintao Wang et al., under the BSD-3-Clause{' '}
                <a
                  href="/licenses"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-fg underline underline-offset-2"
                >
                  license
                </a>
                .
              </p>
            </div>

            <div className="border-line bg-ink-1 border-t p-4">
              {job ? (
                <div>
                  <div className="mb-2 flex items-baseline justify-between gap-2 text-[12px]">
                    <span className="text-fg">
                      {STAGES[job.stage]}
                      {job.stage === 'upscale' && job.total > 1 && (
                        <span className="text-fg-muted">
                          {' '}
                          · tile {Math.min(job.done + 1, job.total)} of {job.total}
                        </span>
                      )}
                    </span>
                    <span className="text-fg-faint font-mono text-[11px] tabular-nums">
                      {formatDuration(Math.max(0, now - job.started))}
                    </span>
                  </div>
                  <div className="bg-ink-4 h-1.5 overflow-hidden rounded-full">
                    {job.stage === 'download' || job.stage === 'upscale' ? (
                      <div
                        className="bg-accent h-full rounded-full transition-[width] duration-300"
                        style={{ width: `${Math.max(2, progress * 100)}%` }}
                      />
                    ) : (
                      <div className="bg-accent/70 h-full w-1/3 animate-[indeterminate_1.1s_ease-in-out_infinite] rounded-full" />
                    )}
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-2">
                    <span className="text-fg-faint font-mono text-[11px] tabular-nums">
                      {job.stage === 'download'
                        ? `${formatBytes(job.done)} of ${formatBytes(job.total)}`
                        : eta !== undefined
                          ? `about ${formatDuration(eta)} left`
                          : ''}
                    </span>
                    <button
                      type="button"
                      onClick={() => engine.current?.cancel(job.id)}
                      className="border-line text-fg-muted hover:border-line-strong hover:text-fg rounded-md border px-2.5 py-1 text-[12px]"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {notice && <p className="text-danger mb-3 text-[12px] leading-snug">{notice}</p>}
                  {result && !notice && (
                    <p className="text-fg-faint mb-3 text-[11.5px] leading-snug">
                      <span className="text-ok">Done in {formatDuration(result.elapsed)}</span> · {result.summary}
                      <span className="block font-mono text-[10.5px]">
                        model {formatDuration(result.timings.model ?? 0)} · upscale{' '}
                        {formatDuration(result.timings.upscale ?? 0)} · encode{' '}
                        {formatDuration(result.timings.encode ?? 0)}
                      </span>
                    </p>
                  )}
                  <button
                    type="button"
                    disabled={!source || !capabilities || !!problem}
                    onClick={() => void start()}
                    className="bg-accent shadow-accent/60 hover:bg-accent-strong disabled:bg-ink-4 disabled:text-fg-faint flex h-10 w-full items-center justify-center gap-2 rounded-lg text-[13px] font-semibold text-[#0d0b1f] shadow-[0_8px_24px_-8px] transition disabled:cursor-not-allowed disabled:shadow-none"
                  >
                    {!source
                      ? 'Open an image to start'
                      : engineState.status === 'starting'
                        ? 'Starting…'
                        : `Upscale${output ? ` to ${output.width} × ${output.height}` : ''}`}
                  </button>
                  {capabilities && source && (
                    <p className="text-fg-faint mt-2 text-center font-mono text-[10.5px]">
                      {modelById(options.model).label} ·{' '}
                      {capabilities.backend === 'webgpu'
                        ? `WebGPU ${resolvedPrecision(options, capabilities).toUpperCase()}`
                        : 'CPU (slow)'}
                      {options.tta && ' · TTA ×8'}
                    </p>
                  )}
                </>
              )}
            </div>
          </aside>

          <main className="flex h-[48vh] min-w-0 flex-col md:h-auto md:flex-1">
            <Viewer
              size={result ? { width: result.width, height: result.height } : output}
              source={source}
              result={result}
              preview={preview}
              previewing={!!job}
              empty={
                <button
                  type="button"
                  onClick={() => fileInput.current?.click()}
                  className="group border-line-strong hover:border-accent/70 absolute inset-4 grid place-items-center rounded-2xl border-2 border-dashed text-center transition-colors md:inset-8"
                >
                  <span className="max-w-sm px-6">
                    <svg
                      viewBox="0 0 48 48"
                      aria-hidden
                      className="text-fg-faint group-hover:text-accent mx-auto mb-4 size-12"
                    >
                      <path
                        d="M8 34V14a4 4 0 0 1 4-4h24a4 4 0 0 1 4 4v20a4 4 0 0 1-4 4H12a4 4 0 0 1-4-4zm0-4 9-9 7 7 5-5 11 11M30 18h.01"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    <span className="text-fg block text-[15px] font-medium">Drop an image here</span>
                    <span className="text-fg-muted mt-1 block text-[12.5px]">or paste it, or click to browse</span>
                    <span className="text-fg-faint mt-4 block text-[11.5px] leading-relaxed">
                      PNG, JPEG, WebP, AVIF… Upscaled on your device: nothing is uploaded.
                    </span>
                  </span>
                </button>
              }
              actions={
                result &&
                !job && (
                  <a
                    href={result.url}
                    download={result.filename}
                    className="bg-accent hover:bg-accent-strong flex items-center gap-2 rounded-lg px-3.5 py-2 text-[12.5px] font-semibold text-[#0d0b1f] shadow-lg transition"
                  >
                    <svg viewBox="0 0 16 16" aria-hidden className="size-4">
                      <path
                        d="M8 2v8m0 0L4.5 6.5M8 10l3.5-3.5M3 13h10"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.7"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    {result.format.toUpperCase()} · {result.width} × {result.height} · {formatBytes(result.blob.size)}
                  </a>
                )
              }
            />
          </main>
        </div>
      </div>

      {dragging && (
        <div className="bg-ink-0/70 pointer-events-none fixed inset-0 z-50 grid place-items-center backdrop-blur-sm">
          <div className="border-accent text-fg rounded-2xl border-2 border-dashed px-10 py-8 text-[15px] font-medium">
            Drop to open
          </div>
        </div>
      )}
    </>
  )
}
