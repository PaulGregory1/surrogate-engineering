export function MetricsBadge({ label, errorPct }: { label: string; errorPct: number }) {
  const good = errorPct < 2
  const color = good ? 'var(--ok)' : 'var(--warn)'
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ color, background: `color-mix(in srgb, ${color} 15%, transparent)`, border: `1px solid color-mix(in srgb, ${color} 40%, transparent)` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {label}: erreur attendue &plusmn;{errorPct.toFixed(2)}%
    </span>
  )
}
