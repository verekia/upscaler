import { useId, useState } from 'react'

import type { CSSProperties, ReactNode } from 'react'

export const Section = ({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) => (
  <section className="border-line border-b px-4 py-4">
    <div className="mb-3 flex min-h-5 items-center justify-between gap-2">
      <h2 className="text-fg-faint text-[11px] font-semibold tracking-[0.12em] uppercase">{title}</h2>
      {aside}
    </div>
    {children}
  </section>
)

type Choice<T> = { value: T; label: ReactNode; title?: string; disabled?: boolean }

// Native radios, so arrow keys move within the group
export const Segmented = <T extends string | number>({
  value,
  choices,
  onChange,
  label,
}: {
  value: T
  choices: Choice<T>[]
  onChange: (value: T) => void
  label: string
}) => {
  const name = useId()
  return (
    <fieldset className="border-line bg-ink-2 flex rounded-md border p-0.5">
      <legend className="sr-only">{label}</legend>
      {choices.map(choice => (
        <label
          key={String(choice.value)}
          title={choice.title}
          className="text-fg-muted hover:text-fg has-checked:bg-ink-4 has-checked:text-fg has-focus-visible:outline-accent has-disabled:hover:text-fg-muted min-w-0 flex-1 cursor-pointer truncate rounded-[5px] px-2 py-1 text-center text-[12px] font-medium transition-colors has-checked:shadow-sm has-focus-visible:outline-2 has-disabled:cursor-not-allowed has-disabled:opacity-35"
        >
          <input
            type="radio"
            name={name}
            className="sr-only"
            checked={choice.value === value}
            disabled={choice.disabled}
            onChange={() => onChange(choice.value)}
          />
          {choice.label}
        </label>
      ))}
    </fieldset>
  )
}

export const Slider = ({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
  disabled,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  display: string
  onChange: (value: number) => void
  disabled?: boolean
  hint?: string
}) => (
  <label className="block" title={hint}>
    <span className="mb-1 flex items-baseline justify-between text-[12px]">
      <span className="text-fg-muted">{label}</span>
      <span className="text-fg font-mono text-[11.5px] tabular-nums">{display}</span>
    </span>
    <input
      type="range"
      className="range"
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={event => onChange(Number(event.target.value))}
      style={{ '--fill': `${((value - min) / (max - min)) * 100}%` } as CSSProperties}
    />
  </label>
)

export const Toggle = ({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description?: string
  checked: boolean
  onChange: (checked: boolean) => void
}) => (
  <label className="flex cursor-pointer items-start justify-between gap-3">
    <span>
      <span className="text-fg block text-[12.5px]">{label}</span>
      {description && <span className="text-fg-faint mt-0.5 block text-[11.5px] leading-snug">{description}</span>}
    </span>
    <input
      type="checkbox"
      className="peer sr-only"
      checked={checked}
      onChange={event => onChange(event.target.checked)}
    />
    <span
      aria-hidden
      className="bg-ink-4 peer-checked:bg-accent peer-focus-visible:outline-accent after:bg-fg relative mt-0.5 h-[18px] w-8 shrink-0 rounded-full transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 after:absolute after:top-[3px] after:left-[3px] after:size-3 after:rounded-full after:transition-transform peer-checked:after:translate-x-[14px]"
    />
  </label>
)

export const Field = ({ label, children }: { label: ReactNode; children: ReactNode }) => (
  <div>
    <div className="text-fg-muted mb-1.5 text-[12px]">{label}</div>
    {children}
  </div>
)

export const Row = ({ label, children }: { label: ReactNode; children: ReactNode }) => (
  <div className="flex items-center justify-between gap-3 text-[12px]">
    <span className="text-fg-muted">{label}</span>
    {children}
  </div>
)

// Keeps what is typed while it is out of range (a width on its way to 3840 starts at 3); only valid
// values reach onChange, and leaving the field shows the current value again
export const NumberInput = ({
  value,
  min,
  max,
  step,
  suffix,
  onChange,
  label,
}: {
  value: number
  min: number
  max: number
  step: number
  suffix?: string
  onChange: (value: number) => void
  label: string
}) => {
  const [draft, setDraft] = useState<string>()
  return (
    <span className="border-line bg-ink-2 focus-within:border-accent flex items-center rounded-md border">
      <input
        type="number"
        aria-label={label}
        className="text-fg w-full min-w-0 bg-transparent px-2 py-1 font-mono text-[12px] tabular-nums outline-none"
        value={draft ?? value}
        min={min}
        max={max}
        step={step}
        onChange={event => {
          setDraft(event.target.value)
          const next = Number(event.target.value)
          if (event.target.value !== '' && next >= min && next <= max) onChange(next)
        }}
        onBlur={() => setDraft(undefined)}
      />
      {suffix && <span className="text-fg-faint pr-2 font-mono text-[11px]">{suffix}</span>}
    </span>
  )
}
