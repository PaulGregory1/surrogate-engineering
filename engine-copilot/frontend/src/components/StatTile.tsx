import { AnimatedNumber } from '@/components/AnimatedNumber'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

/**
 * The stat-tile contract: a sentence-case label, one value, and optional supporting
 * marks. The value keeps the font's proportional figures — `tabular-nums` gives every
 * digit the width of a zero, which reads loose at display sizes — but the readout
 * animates in place, so it is measured in a fixed-width box to stop the row shifting.
 */
export function StatTile({
  label,
  value,
  unit,
  decimals = 2,
  loading = false,
  hint,
  trailing,
  className,
}: {
  label: string
  value: number | null
  unit?: string
  decimals?: number
  loading?: boolean
  hint?: string
  trailing?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex min-w-0 flex-col gap-1', className)}>
      <div className="text-xs text-muted-foreground">{label}</div>
      {loading || value === null ? (
        <Skeleton className="h-8 w-24" />
      ) : (
        <div className="flex items-baseline gap-1.5">
          <span className="text-3xl leading-none font-semibold tracking-tight">
            <AnimatedNumber value={value} decimals={decimals} />
          </span>
          {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
          {trailing}
        </div>
      )}
      {hint && <div className="text-[11px] text-muted-foreground">{hint}</div>}
    </div>
  )
}
