import { MODELS, modelById, NATIVE_SCALE } from '../models'
import { DEFAULT_OPTIONS, MAX_SIDE, outputSize, sizeProblem } from '../options'
import { Field, NumberInput, Row, Section, Segmented, Slider, Toggle } from './ui'

import type { Capabilities } from '../engine'
import type { Model } from '../models'
import type { Options } from '../options'

const SCALES = [1, 1.5, 2, 3, 4, 6, 8]
const TILES = [64, 96, 128, 192, 256, 384, 512, 768, 1024]
const FILTER_NAMES = { lanczos3: 'Lanczos3', mitchell: 'Mitchell', 'catmull-rom': 'Catmull-Rom' }

// 5 for the lightest model, one less per 3x the work
const speed = (model: Model) => 5 - Math.round(Math.log(model.macsPerPixel / 0.62e6) / Math.log(3))

const SpeedBars = ({ level }: { level: number }) => (
  <span className="flex items-center gap-1" title={`Speed ${level}/5`}>
    <span className="sr-only">Speed {level} of 5</span>
    {[1, 2, 3, 4, 5].map(i => (
      <span key={i} aria-hidden className={`h-2 w-1 rounded-[1px] ${i <= level ? 'bg-accent' : 'bg-ink-4'}`} />
    ))}
  </span>
)

export type SourceInfo = { width: number; height: number; transparent: boolean }

export const Settings = ({
  options,
  onChange,
  source,
  capabilities,
  busy,
}: {
  options: Options
  onChange: (patch: Partial<Options>) => void
  source: SourceInfo | undefined
  capabilities: Capabilities | undefined
  busy: boolean
}) => {
  const model = modelById(options.model)
  const output = source && outputSize(options, source.width, source.height)
  const problem = output && sizeProblem(output.width, output.height)
  const resampled = output && source && output.width !== source.width * NATIVE_SCALE

  return (
    <>
      <Section title="Model">
        <fieldset className="space-y-1.5">
          <legend className="sr-only">Model</legend>
          {MODELS.map(item => (
            <label
              key={item.id}
              className="border-line bg-ink-2 hover:border-line-strong hover:bg-ink-3 has-checked:border-accent/60 has-checked:bg-accent/10 has-focus-visible:outline-accent block cursor-pointer rounded-lg border px-3 py-2.5 transition-colors has-focus-visible:outline-2 has-focus-visible:outline-offset-2"
            >
              <input
                type="radio"
                name="model"
                className="sr-only"
                checked={item.id === options.model}
                onChange={() => onChange({ model: item.id })}
              />
              <span className="flex items-baseline justify-between gap-2">
                <span className="text-fg text-[13px] font-medium">{item.label}</span>
                <span className="text-fg-faint font-mono text-[10.5px]">{item.megabytes} MB</span>
              </span>
              <span className="text-fg-muted mt-0.5 block text-[11.5px] leading-snug">{item.description}</span>
              <span className="text-fg-faint mt-2 flex items-center justify-between gap-2 font-mono text-[10.5px]">
                <span className="truncate">{item.architecture}</span>
                <SpeedBars level={speed(item)} />
              </span>
            </label>
          ))}
        </fieldset>
        {model.checkpoints.length > 1 && (
          <div className="mt-3.5">
            <Slider
              label="Denoise strength"
              hint="Blends the model with its weak-denoise twin: 0 keeps grain and fine noise, 1 removes it"
              value={options.denoise}
              min={0}
              max={1}
              step={0.05}
              display={options.denoise.toFixed(2)}
              onChange={denoise => onChange({ denoise })}
            />
          </div>
        )}
      </Section>

      <Section
        title="Output size"
        aside={
          output && (
            <span className={`font-mono text-[11.5px] tabular-nums ${problem ? 'text-danger' : 'text-fg'}`}>
              {output.width} × {output.height}
            </span>
          )
        }
      >
        <Segmented
          label="Size by"
          value={options.sizeMode}
          choices={[
            { value: 'scale', label: 'Scale' },
            { value: 'width', label: 'Width' },
            { value: 'height', label: 'Height' },
          ]}
          onChange={sizeMode => onChange({ sizeMode })}
        />
        <div className="mt-2.5">
          {options.sizeMode === 'scale' ? (
            <div className="grid grid-cols-[repeat(7,minmax(0,1fr))_4.5rem] gap-1">
              {SCALES.map(scale => (
                <button
                  key={scale}
                  type="button"
                  aria-pressed={options.scale === scale}
                  onClick={() => onChange({ scale })}
                  className={`rounded-md border py-1 font-mono text-[11.5px] transition-colors ${
                    options.scale === scale
                      ? 'border-accent/60 bg-accent/10 text-fg'
                      : 'border-line bg-ink-2 text-fg-muted hover:text-fg'
                  }`}
                >
                  {scale}×
                </button>
              ))}
              <NumberInput
                label="Scale"
                value={options.scale}
                min={1}
                max={16}
                step={0.05}
                suffix="×"
                onChange={scale => onChange({ scale })}
              />
            </div>
          ) : (
            <NumberInput
              label={options.sizeMode === 'width' ? 'Output width' : 'Output height'}
              value={options.sizeMode === 'width' ? options.width : options.height}
              min={1}
              max={MAX_SIDE}
              step={1}
              suffix={options.sizeMode === 'width' ? 'px wide' : 'px high'}
              onChange={value => onChange(options.sizeMode === 'width' ? { width: value } : { height: value })}
            />
          )}
        </div>
        {problem ? (
          <p className="text-danger mt-2 text-[11.5px]">{problem}: pick a smaller size.</p>
        ) : (
          resampled && (
            <p className="text-fg-faint mt-2 text-[11.5px] leading-snug">
              The model upscales 4×, then {FILTER_NAMES[options.filter]} resamples to this size.
            </p>
          )
        )}
      </Section>

      <Section title="Format">
        <Segmented
          label="Format"
          value={options.format}
          choices={[
            { value: 'png', label: 'PNG' },
            { value: 'jpeg', label: 'JPEG' },
            { value: 'webp', label: 'WebP' },
          ]}
          onChange={format => onChange({ format })}
        />
        {options.format === 'png' ? (
          <p className="text-fg-faint mt-2 text-[11.5px] leading-snug">Lossless, keeps transparency.</p>
        ) : (
          <div className="mt-3">
            <Slider
              label="Quality"
              value={options.quality}
              min={50}
              max={100}
              step={1}
              display={String(options.quality)}
              onChange={quality => onChange({ quality })}
            />
          </div>
        )}
      </Section>

      <Section title="Refinement">
        <div className="space-y-3.5">
          <Toggle
            label="Test-time augmentation"
            description="Averages the model over 8 flips and rotations of each tile: a touch cleaner, 8× slower."
            checked={options.tta}
            onChange={tta => onChange({ tta })}
          />
          {source?.transparent && (
            <Row label="Transparency">
              <div className="w-40">
                <Segmented
                  label="Transparency"
                  value={options.alpha}
                  choices={[
                    {
                      value: 'model',
                      label: 'Model',
                      title: 'Upscale the alpha channel with the model too (2× the work)',
                    },
                    { value: 'resize', label: 'Resample', title: 'Resample the alpha channel, fast' },
                  ]}
                  onChange={alpha => onChange({ alpha })}
                />
              </div>
            </Row>
          )}
        </div>
      </Section>

      <details className="group border-line border-b">
        <summary className="text-fg-faint hover:text-fg-muted flex cursor-pointer list-none items-center justify-between px-4 py-3.5 text-[11px] font-semibold tracking-[0.12em] uppercase [&::-webkit-details-marker]:hidden">
          Advanced
          <svg viewBox="0 0 16 16" aria-hidden className="size-3.5 transition-transform group-open:rotate-180">
            <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </summary>
        <div className="space-y-3.5 px-4 pb-4">
          <Field label="Backend">
            <Segmented
              label="Backend"
              value={options.backend}
              choices={[
                { value: 'auto', label: 'Auto', disabled: busy },
                { value: 'webgpu', label: 'WebGPU', disabled: busy },
                {
                  value: 'wasm',
                  label: 'CPU',
                  title: 'WebAssembly on the CPU: works everywhere, much slower',
                  disabled: busy,
                },
              ]}
              onChange={backend => onChange({ backend })}
            />
          </Field>
          <Field label="Precision">
            <Segmented
              label="Precision"
              value={options.precision}
              choices={[
                { value: 'auto', label: 'Auto' },
                {
                  value: 'fp16',
                  label: 'FP16',
                  title: 'Half precision on WebGPU (shader-f16): faster, half the memory',
                  disabled: capabilities?.backend !== 'webgpu' || !capabilities.fp16,
                },
                { value: 'fp32', label: 'FP32' },
              ]}
              onChange={precision => onChange({ precision })}
            />
          </Field>
          <Field label="Tile size">
            <select
              aria-label="Tile size"
              value={options.tile}
              onChange={event => onChange({ tile: Number(event.target.value) })}
              className="border-line bg-ink-2 text-fg focus:border-accent w-full rounded-md border px-2 py-1 font-mono text-[12px] outline-none"
            >
              <option value={0}>Auto ({model.tile} px)</option>
              {TILES.map(tile => (
                <option key={tile} value={tile}>
                  {tile} px
                </option>
              ))}
            </select>
          </Field>
          <Slider
            label="Tile overlap"
            hint="Context each tile reads past its edges: more hides seams better, at some speed"
            value={options.overlap}
            min={0}
            max={48}
            step={2}
            display={`${options.overlap} px`}
            onChange={overlap => onChange({ overlap })}
          />
          <Field label="Resampling">
            <Segmented
              label="Resampling filter"
              value={options.filter}
              choices={[
                { value: 'lanczos3', label: 'Lanczos3', title: 'The sharpest, as in sharp' },
                { value: 'mitchell', label: 'Mitchell', title: 'Softer, no ringing' },
                { value: 'catmull-rom', label: 'Catmull-Rom', title: 'Bicubic' },
              ]}
              onChange={filter => onChange({ filter })}
            />
          </Field>
          <button
            type="button"
            onClick={() => onChange({ ...DEFAULT_OPTIONS, backend: busy ? options.backend : DEFAULT_OPTIONS.backend })}
            className="text-fg-faint decoration-line-strong hover:text-fg-muted text-[11.5px] underline underline-offset-4"
          >
            Reset all settings
          </button>
        </div>
      </details>
    </>
  )
}
