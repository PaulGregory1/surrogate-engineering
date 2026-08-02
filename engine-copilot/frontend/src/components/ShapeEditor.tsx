import { motion } from 'framer-motion'
import { RotateCcw, Shuffle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { COEFF_RANGE, boundaryXY, r0ForArea } from '@/geometry'
import { cn } from '@/lib/utils'

const LABELS = ['a2', 'b2', 'a3', 'b3', 'a4', 'b4']

/**
 * The shape preview alone — reused read-only in the optimizer's before/after pair,
 * where the sliders and the surrounding card would only be noise.
 *
 * `tone` is the whole before/after story in one prop: the starting geometry is cut
 * paper under an ink outline, the optimised one is the same silhouette in orange,
 * because *that* one is the result of a measurement.
 */
export function ShapePreview({
  coeffs,
  targetArea,
  tone = 'ink',
  className,
}: {
  coeffs: number[]
  targetArea: number
  tone?: 'ink' | 'measure'
  className?: string
}) {
  const r0 = r0ForArea(coeffs, targetArea)
  const { x, y } = boundaryXY(r0, coeffs)
  const path = x.map((xi, i) => `${i === 0 ? 'M' : 'L'} ${xi.toFixed(2)} ${y[i].toFixed(2)}`).join(' ') + ' Z'
  const extent = r0 * 1.6
  const measure = tone === 'measure'

  return (
    <svg
      viewBox={`${-extent} ${-extent} ${extent * 2} ${extent * 2}`}
      className={cn('aspect-square w-full rounded-[12px] border border-line bg-paper', className)}
    >
      {/* The crosshair belongs to the same one-pixel family as the table rules and
          the background arcs — it locates the body, it does not decorate it. */}
      <line x1={-extent} y1="0" x2={extent} y2="0" stroke="var(--line)" strokeWidth={extent / 240} />
      <line x1="0" y1={-extent} x2="0" y2={extent} stroke="var(--line)" strokeWidth={extent / 240} />
      <motion.path
        fill={measure ? 'rgba(255,106,0,.10)' : 'rgba(13,12,11,.05)'}
        stroke={measure ? 'var(--or)' : 'var(--ink)'}
        strokeWidth={extent / 110}
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
    <div className="flex items-center gap-3">
      <span className="w-5 shrink-0 cq-num text-[10px] text-ink3">{label}</span>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={-COEFF_RANGE}
        max={COEFF_RANGE}
        step={0.005}
        className="flex-1"
      />
      <span className="w-11 shrink-0 text-right cq-num text-[10px] text-ink2">
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

  const outOfEnvelope = reynolds > 150 || reynolds < 20

  return (
    <Card className="gap-5 py-5">
      <CardHeader className="px-5">
        <CardTitle className="cq-title">Géométrie</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5 px-5">
        <ShapePreview coeffs={coeffs} targetArea={targetArea} />

        <div className="flex flex-col gap-3">
          <div className="cq-k">COEFFICIENTS DE FOURIER</div>
          {LABELS.map((lab, i) => (
            <CoeffSlider key={lab} label={lab} value={coeffs[i]} onChange={(v) => setCoeff(i, v)} />
          ))}
        </div>

        <Separator />

        <div className="flex flex-col gap-2.5">
          <div className="flex items-baseline justify-between">
            <span className="cq-k">NOMBRE DE REYNOLDS</span>
            <span
              className={cn(
                'cq-num text-band',
                outOfEnvelope ? 'font-medium text-burnt' : 'text-ink',
              )}
            >
              {reynolds.toFixed(0)}
            </span>
          </div>
          <Slider
            value={[reynolds]}
            onValueChange={([v]) => onReynoldsChange(v)}
            min={10}
            max={200}
            step={1}
          />
          <div className="flex justify-between font-mono text-[9px] tracking-[0.1em] text-ink3">
            <span>10</span>
            {/* A point outside the training envelope is flagged burnt, never red: the
                charte has no red, and introducing one would break the ramp's reading. */}
            <span className={cn(outOfEnvelope && 'text-burnt')}>ENTRAÎNÉ 20–150</span>
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
