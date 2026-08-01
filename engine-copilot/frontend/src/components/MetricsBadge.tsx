import { CircleCheck, TriangleAlert } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/** Every predicted number in this app ships with the model's measured error beside it.
 *  Status is carried by an icon *and* a label, never by hue alone. */
export function MetricsBadge({ label, errorPct }: { label: string; errorPct: number }) {
  const good = errorPct < 2
  const Icon = good ? CircleCheck : TriangleAlert

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className={cn(
            'cursor-default gap-1.5 py-1 font-normal',
            good ? 'border-good/40 bg-good/10' : 'border-warning/40 bg-warning/10',
          )}
        >
          <Icon className={cn('size-3', good ? 'text-good' : 'text-warning')} />
          <span className="font-mono text-[11px] text-foreground">{label}</span>
          <span className="text-muted-foreground">±{errorPct.toFixed(2)}%</span>
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        Erreur relative moyenne mesurée de <span className="font-mono">{label}</span> sur le jeu de
        test, face au solveur haute-fidélité. La valeur affichée est une prédiction, pas une mesure.
      </TooltipContent>
    </Tooltip>
  )
}
