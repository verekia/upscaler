import { useEffect, useRef, useState } from 'react'

import type { ReactNode, RefObject } from 'react'

type Layer = { url: string }
type Mode = 'split' | 'original' | 'result'
type View = { zoom: number; x: number; y: number }

const MIN_ZOOM = 0.02
const MAX_ZOOM = 32

const ToolButton = ({
  onClick,
  pressed,
  title,
  children,
}: {
  onClick: () => void
  pressed?: boolean
  title: string
  children: ReactNode
}) => (
  <button
    type="button"
    title={title}
    aria-label={title}
    aria-pressed={pressed}
    onClick={onClick}
    className={`grid h-7 min-w-7 place-items-center rounded-md px-1.5 text-[12px] transition-colors ${
      pressed ? 'bg-ink-4 text-fg' : 'text-fg-muted hover:bg-ink-3 hover:text-fg'
    }`}
  >
    {children}
  </button>
)

// The image stage: the original stretched to the output size, the live tiles over it while upscaling, then
// the result, split against the original by a draggable divider. Wheel zooms around the cursor, drag pans.
export const Viewer = ({
  size,
  source,
  result,
  preview,
  previewing,
  empty,
  actions,
}: {
  size: { width: number; height: number } | undefined
  source: Layer | undefined
  result: Layer | undefined
  preview: RefObject<HTMLCanvasElement | null>
  previewing: boolean
  empty: ReactNode
  actions?: ReactNode
}) => {
  const viewport = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>({ zoom: 1, x: 0, y: 0 })
  const [split, setSplit] = useState(0.5)
  const [mode, setMode] = useState<Mode>('split')
  const [smooth, setSmooth] = useState(true)
  const [viewportWidth, setViewportWidth] = useState(0)
  const fitted = useRef(true)
  const drag = useRef<{ x: number; y: number; view: View } | undefined>(undefined)

  const width = size?.width ?? 1
  const height = size?.height ?? 1

  const fit = () => {
    const element = viewport.current
    if (!element) return
    const { width: w, height: h } = element.getBoundingClientRect()
    setViewportWidth(w)
    const margin = Math.min(32, w * 0.05)
    const zoom = Math.max(MIN_ZOOM, Math.min((w - 2 * margin) / width, (h - 2 * margin - 48) / height))
    setView({ zoom, x: (w - width * zoom) / 2, y: (h - 48 - height * zoom) / 2 })
    fitted.current = true
  }

  const zoomAt = (factor: number, cx: number, cy: number) => {
    setView(current => {
      const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.zoom * factor))
      const k = zoom / current.zoom
      return { zoom, x: cx - (cx - current.x) * k, y: cy - (cy - current.y) * k }
    })
    fitted.current = false
  }

  const zoomCentered = (factor: number) => {
    const rect = viewport.current?.getBoundingClientRect()
    if (rect) zoomAt(factor, rect.width / 2, (rect.height - 48) / 2)
  }

  // Refit when the content changes size, and on viewport resizes until the user zooms or pans
  useEffect(fit, [width, height, source?.url])
  useEffect(() => {
    const element = viewport.current
    if (!element) return
    const observer = new ResizeObserver(() => {
      setViewportWidth(element.clientWidth)
      if (fitted.current) fit()
    })
    observer.observe(element)
    return () => observer.disconnect()
  })

  useEffect(() => {
    const element = viewport.current
    if (!element) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = element.getBoundingClientRect()
      // Pinches arrive as ctrl+wheel with small deltas
      const speed = event.ctrlKey ? 0.01 : event.deltaMode === 1 ? 0.05 : 0.0015
      zoomAt(Math.exp(-event.deltaY * speed), event.clientX - rect.left, event.clientY - rect.top)
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    return () => element.removeEventListener('wheel', onWheel)
  })

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return
      if (event.metaKey || event.ctrlKey || event.altKey || !source) return
      if (event.key === 'f') fit()
      else if (event.key === '1') zoomCentered(1 / view.zoom)
      else if (event.key === '=' || event.key === '+') zoomCentered(1.25)
      else if (event.key === '-') zoomCentered(0.8)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  if (!source || !size) return <div className="checker relative min-h-0 flex-1">{empty}</div>

  const clip = Math.min(width, Math.max(0, (split * viewportWidth - view.x) / view.zoom))
  const showSplit = result && mode === 'split' && !previewing

  return (
    <div
      ref={viewport}
      className="checker relative min-h-0 flex-1 cursor-grab touch-none overflow-hidden select-none active:cursor-grabbing"
      onPointerDown={event => {
        if (event.button !== 0 && event.button !== 1) return
        event.currentTarget.setPointerCapture(event.pointerId)
        drag.current = { x: event.clientX, y: event.clientY, view }
      }}
      onPointerMove={event => {
        const start = drag.current
        if (!start) return
        setView({ ...start.view, x: start.view.x + event.clientX - start.x, y: start.view.y + event.clientY - start.y })
        fitted.current = false
      }}
      onPointerUp={() => (drag.current = undefined)}
      onPointerCancel={() => (drag.current = undefined)}
      onDoubleClick={fit}
    >
      <div
        className="absolute top-0 left-0 origin-top-left shadow-[0_0_0_1px_rgba(255,255,255,0.06),0_20px_60px_rgba(0,0,0,0.5)]"
        style={{ width, height, transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})` }}
      >
        <img
          src={source.url}
          alt="Original"
          draggable={false}
          className="absolute inset-0 size-full max-w-none"
          style={{ imageRendering: smooth ? 'auto' : 'pixelated' }}
        />
        <canvas ref={preview} className={`absolute inset-0 size-full ${previewing ? '' : 'hidden'}`} />
        {result && !previewing && (
          <img
            src={result.url}
            alt="Upscaled"
            draggable={false}
            className="absolute inset-0 size-full max-w-none"
            style={{
              clipPath: showSplit ? `inset(0 0 0 ${clip}px)` : undefined,
              visibility: mode === 'original' ? 'hidden' : 'visible',
              imageRendering: view.zoom >= 2 ? 'pixelated' : 'auto',
            }}
          />
        )}
      </div>

      {showSplit && (
        <>
          <div
            className="absolute inset-y-0 z-10 w-px bg-white/80 shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
            style={{ left: `${split * 100}%` }}
          >
            <button
              type="button"
              aria-label="Drag to compare"
              className="bg-ink-1/90 text-fg absolute top-1/2 left-1/2 grid size-8 -translate-1/2 cursor-ew-resize place-items-center rounded-full border border-white/70 shadow-lg backdrop-blur"
              onPointerDown={event => {
                event.stopPropagation()
                event.currentTarget.setPointerCapture(event.pointerId)
              }}
              onPointerMove={event => {
                if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
                const rect = viewport.current!.getBoundingClientRect()
                setSplit(Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)))
              }}
              onKeyDown={event => {
                if (event.key === 'ArrowLeft') setSplit(value => Math.max(0, value - 0.02))
                if (event.key === 'ArrowRight') setSplit(value => Math.min(1, value + 0.02))
              }}
            >
              <svg viewBox="0 0 16 16" aria-hidden className="size-4">
                <path d="M6 4L2 8l4 4M10 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </button>
          </div>
          <span className="pointer-events-none absolute top-3 left-3 rounded bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white/85 backdrop-blur">
            Original
          </span>
          <span className="pointer-events-none absolute top-3 right-3 rounded bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white/85 backdrop-blur">
            Upscaled
          </span>
        </>
      )}

      <div
        className="absolute inset-x-3 bottom-3 z-20 flex flex-wrap items-center justify-between gap-2"
        onPointerDown={event => event.stopPropagation()}
        onDoubleClick={event => event.stopPropagation()}
      >
        <div className="border-line bg-ink-1/90 flex items-center gap-0.5 rounded-lg border p-0.5 shadow-lg backdrop-blur">
          {result && !previewing && (
            <>
              <ToolButton title="Split view" pressed={mode === 'split'} onClick={() => setMode('split')}>
                Split
              </ToolButton>
              <ToolButton title="Original only" pressed={mode === 'original'} onClick={() => setMode('original')}>
                Original
              </ToolButton>
              <ToolButton title="Upscaled only" pressed={mode === 'result'} onClick={() => setMode('result')}>
                Upscaled
              </ToolButton>
              <span className="bg-line mx-1 h-4 w-px" />
            </>
          )}
          <ToolButton title="Zoom out (−)" onClick={() => zoomCentered(0.8)}>
            −
          </ToolButton>
          <span className="text-fg w-12 text-center font-mono text-[11.5px] tabular-nums">
            {Math.round(view.zoom * 100)}%
          </span>
          <ToolButton title="Zoom in (+)" onClick={() => zoomCentered(1.25)}>
            +
          </ToolButton>
          <ToolButton title="Fit (F)" onClick={fit}>
            Fit
          </ToolButton>
          <ToolButton title="Actual size (1)" onClick={() => zoomCentered(1 / view.zoom)}>
            1:1
          </ToolButton>
          <span className="bg-line mx-1 h-4 w-px" />
          <ToolButton
            title={smooth ? 'Original: smooth (click for pixels)' : 'Original: pixels (click for smooth)'}
            pressed={!smooth}
            onClick={() => setSmooth(value => !value)}
          >
            <svg viewBox="0 0 16 16" aria-hidden className="size-4">
              <path
                d="M2 2h4v4H2zM10 2h4v4h-4zM6 6h4v4H6zM2 10h4v4H2zM10 10h4v4h-4z"
                fill="currentColor"
                opacity={smooth ? 0.45 : 1}
              />
            </svg>
          </ToolButton>
        </div>
        {actions}
      </div>
    </div>
  )
}
