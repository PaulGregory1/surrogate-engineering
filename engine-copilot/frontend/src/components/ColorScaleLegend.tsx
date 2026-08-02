import { cn } from '@/lib/utils'

/**
 * The colour key for a field heatmap, drawn in the DOM rather than by Plotly.
 *
 * Plotly's own colorbar is unusable at this size: it picks a "nice" tick step from
 * `nticks` as a *hint*, then suppresses any label that would overflow the bar's
 * ends — which on a short bar routinely leaves a single tick in the middle, and on
 * a horizontal one rotates whatever survives. A CSS gradient with the two endpoints
 * (and the zero crossing, where there is one) is fully determined, always legible,
 * and costs the plot no width — which matters because these fields are aspect-locked,
 * so every pixel the key takes would shrink the image in both dimensions.
 *
 * Values always go through `toFixed`/`toExponential`: raw JS floats print
 * `0.30000000000000004`, and a dirty number costs an engineering tool more
 * credibility than a colour mistake ever will.
 */
export function ColorScaleLegend({
  stops,
  min,
  max,
  diverging = false,
  className,
}: {
  stops: [number, string][]
  min: number
  max: number
  diverging?: boolean
  className?: string
}) {
  const gradient = `linear-gradient(90deg, ${stops.map(([t, c]) => `${c} ${(t * 100).toFixed(1)}%`).join(', ')})`

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      <div className="h-[3px] w-full rounded-[2px]" style={{ background: gradient }} />
      <div className="relative flex justify-between font-mono text-[9px] tracking-[0.08em] text-ink3 tabular-nums">
        <span>{formatValue(min)}</span>
        {diverging && <span className="absolute left-1/2 -translate-x-1/2">0</span>}
        <span>{formatValue(max)}</span>
      </div>
    </div>
  )
}

/** Enough significant digits to distinguish the ends, without a wall of zeros. */
function formatValue(v: number): string {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 100) return v.toFixed(0)
  if (a >= 1) return v.toFixed(1)
  if (a >= 0.01) return v.toFixed(3)
  return v.toExponential(1)
}
