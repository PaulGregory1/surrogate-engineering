import { motion } from 'framer-motion'
import { RotateCcw, Shuffle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { COEFF_RANGE, boundaryXY, r0ForArea } from '@/geometry'
import { cn } from '@/lib/utils'

const LABELS = ['a2', 'b2', 'a3', 'b3', 'a4', 'b4']

/** The shape preview alone — reused read-only in the optimizer's before/after pair,
 *  where the sliders and the surrounding card would only be noise. */
export function ShapePreview({ coeffs, targetArea, className }: { coeffs: number[]; targetArea: number; className?: string }) {
  const r0 = r0ForArea(coeffs, targetArea)
  const { x, y } = boundaryXY(r0, coeffs)
  const path = x.map((xi, i) => `${i === 0 ? 'M' : 'L'} ${xi.toFixed(2)} ${y[i].toFixed(2)}`).join(' ') + ' Z'
  const extent = r0 * 1.6

  return (
    <svg
      viewBox={`${-extent} ${-extent} ${extent * 2} ${extent * 2}`}
      className={cn('aspect-square w-full rounded-lg border border-border bg-background', className)}
    >
      <line x1={-extent} y1="0" x2={extent} y2="0" stroke="var(--border)" strokeWidth={extent / 220} />
      <line x1="0" y1={-extent} x2="0" y2={extent} stroke="var(--border)" strokeWidth={extent / 220} />
      <motion.path
        fill="color-mix(in srgb, var(--ring) 16%, transparent)"
        stroke="var(--ring)"
        strokeWidth={extent / 90}
        strokeLinejoin="round"
        initial={false}
        animate={{ d: path }}
        transition={{ type: 'spring', stiffness: 220, damping: 26 }}
      />
    </svg>
  )
}

function CoeffSlider({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-5 shrink-0 font-mono text-[11px] text-muted-foreground">{label}</span>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={-COEFF_RANGE}
        max={COEFF_RANGE}
        step={0.005}
        className="flex-1"
      />
      <span className="tnum w-11 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
        {value >= 0 ? '+' : '−'}
        {Math.abs(value).toFixed(3)}
      </span>
    </div>
  )
}

export function ShapeEditor({
  coeffs,
  onChange,
  reynolds,
  onReynoldsChange,
  targetArea,
}: {
  coeffs: number[]
  onChange: (coeffs: number[]) => void
  reynolds: number
  onReynoldsChange: (re: number) => void
  targetArea: number
}) {
  function setCoeff(i: number, v: number) {
    const next = [...coeffs]
    next[i] = v
    onChange(next)
  }

  return (
    <Card className="gap-4 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm">Géométrie</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 px-4">
        <ShapePreview coeffs={coeffs} targetArea={targetArea} />

        <div className="flex flex-col gap-2.5">
          <div className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            Coefficients de Fourier
          </div>
          {LABELS.map((lab, i) => (
            <CoeffSlider key={lab} label={lab} value={coeffs[i]} onChange={(v) => setCoeff(i, v)} />
          ))}
        </div>

        <Separator />

        <div className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
              Nombre de Reynolds
            </span>
            <span className="tnum font-mono text-sm">{reynolds.toFixed(0)}</span>
          </div>
          <Slider
            value={[reynolds]}
            onValueChange={([v]) => onReynoldsChange(v)}
            min={10}
            max={200}
            step={1}
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>10</span>
            <span>entraîné sur 20–150</span>
            <span>200</span>
          </div>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="flex-1" onClick={() => onChange([0, 0, 0, 0, 0, 0])}>
            <RotateCcw />
            Cercle
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={() => onChange(coeffs.map(() => (Math.random() * 2 - 1) * COEFF_RANGE * 0.7))}
          >
            <Shuffle />
            Aléatoire
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
