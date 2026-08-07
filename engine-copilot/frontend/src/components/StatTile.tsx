import { AnimatedNumber } from '@/components/AnimatedNumber'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

/**
 * A panel's principal value: a micro-label, one figure, and the interval under it.
 *
 * The figure is the charte's single display slot — Instrument Serif **roman** at
 * 40px, never italic, never sans, never bold, never larger. It reads as a
 * breathing space rather than a headline precisely because everything around it
 * sits between 9 and 12.5px; enlarge it and the whole screen changes register.
 *
 * The interval is not decoration and not optional — a surrogate prediction shown
 * without its variance is a substantive error, not a stylistic one. Callers derive
 * it from the model's measured test error; `interval` is only ever omitted where
 * the number being shown is a measurement rather than a prediction.
 */
export function StatTile({
  label,
  value,
  unit,
  decimals = 2,
  loading = false,
  interval,
  className,
}: {
  label: string
  value: number | null
  /** rendered as the `<i>` of the 40px figure — 16px, italic, ink3 */
  unit?: string
  decimals?: number
  loading?: boolean
  /** e.g. `± 0.006 · CONSTANT PROJECTED AREA` — mono orange, under the figure */
  interval?: string
  className?: string
}) {
  return (
    <div className={cn('flex min-w-0 flex-col', className)}>
      <div className="cq-k">{label}</div>
      {loading || value === null ? (
        <Skeleton className="mt-2.5 h-9 w-28" />
      ) : (
        // AnimatedNumber counts to the new value in place, so the figure runs on
        // tabular numerals: without them every frame re-measures and the whole row
        // twitches sideways for the length of the transition.
        <div className="cq-value mt-2.5">
          <AnimatedNumber value={value} decimals={decimals} />
          {unit && <i>{unit}</i>}
        </div>
      )}
      {interval && !loading && value !== null && <div className="cq-band mt-2">{interval}</div>}
    </div>
  )
}
