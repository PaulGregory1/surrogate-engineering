import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { TriangleAlert } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { HEIGHT, MAX_R, MIN_R, WIDTH, isValid } from '@/stressGeometry'

const DEFAULT_R = 0.12

export function HoleEditor({
  holes,
  onChange,
  loadAngleDeg,
  onLoadAngleChange,
}: {
  holes: number[][]
  onChange: (holes: number[][]) => void
  loadAngleDeg: number
  onLoadAngleChange: (deg: number) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [invalid, setInvalid] = useState(false)

  function flashInvalid() {
    setInvalid(true)
    window.setTimeout(() => setInvalid(false), 350)
  }

  function toDomain(clientX: number, clientY: number): [number, number] {
    const rect = svgRef.current!.getBoundingClientRect()
    return [((clientX - rect.left) / rect.width) * WIDTH, ((clientY - rect.top) / rect.height) * HEIGHT]
  }

  function handleBackgroundClick(e: React.MouseEvent) {
    if (holes.length >= 4) return
    const [x, y] = toDomain(e.clientX, e.clientY)
    const next = [...holes, [x, y, DEFAULT_R]]
    if (isValid(next)) onChange(next)
    else flashInvalid()
  }

  function handleHoleDoubleClick(i: number, e: React.MouseEvent) {
    e.stopPropagation()
    if (holes.length <= 2) return
    onChange(holes.filter((_, j) => j !== i))
  }

  function handlePointerDown(i: number, e: React.PointerEvent) {
    e.stopPropagation()
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
    setDragIndex(i)
  }

  function handlePointerMove(e: React.PointerEvent) {
    if (dragIndex === null) return
    const [x, y] = toDomain(e.clientX, e.clientY)
    const next = holes.map((h, j) => (j === dragIndex ? [x, y, h[2]] : h))
    if (isValid(next)) {
      onChange(next)
      setInvalid(false)
    } else {
      setInvalid(true)
    }
  }

  function handlePointerUp() {
    setDragIndex(null)
    setInvalid(false)
  }

  function handleWheel(i: number, e: React.WheelEvent) {
    e.preventDefault()
    e.stopPropagation()
    const delta = e.deltaY > 0 ? -0.006 : 0.006
    const next = holes.map((h, j) => (j === i ? [h[0], h[1], Math.min(MAX_R, Math.max(MIN_R, h[2] + delta))] : h))
    if (isValid(next)) onChange(next)
    else flashInvalid()
  }

  return (
    <Card className="gap-4 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm">Plaque perforée</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 px-4">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full rounded-lg border border-border bg-background"
          style={{ aspectRatio: `${WIDTH} / ${HEIGHT}`, touchAction: 'none' }}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          <rect
            x={0}
            y={0}
            width={WIDTH}
            height={HEIGHT}
            fill="color-mix(in srgb, var(--foreground) 4%, transparent)"
            stroke="var(--border)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
            className="cursor-crosshair"
            onClick={handleBackgroundClick}
          />
          {/* encastrement (gauche) */}
          <line x1={0.004} y1={0} x2={0.004} y2={HEIGHT} stroke="var(--good)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
          {/* bord chargé (droite) */}
          <line
            x1={WIDTH - 0.004}
            y1={0}
            x2={WIDTH - 0.004}
            y2={HEIGHT}
            stroke="var(--warning)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />

          {holes.map(([x, y, r], i) => (
            <motion.circle
              key={i}
              fill="color-mix(in srgb, var(--ring) 20%, transparent)"
              stroke={dragIndex === i ? 'var(--good)' : 'var(--ring)'}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
              style={{ cursor: dragIndex === i ? 'grabbing' : 'grab' }}
              onPointerDown={(e) => handlePointerDown(i, e)}
              onDoubleClick={(e) => handleHoleDoubleClick(i, e)}
              onWheel={(e) => handleWheel(i, e)}
              initial={false}
              animate={{ cx: x, cy: y, r }}
              transition={{ type: 'spring', stiffness: 320, damping: 30 }}
            />
          ))}
        </svg>

        <div className="flex items-start gap-4 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded-full bg-good" />
            encastré
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded-full bg-warning" />
            chargé
          </span>
          <span className="ml-auto tabular-nums">{holes.length}/4 trous</span>
        </div>

        <div className="min-h-[2.5em] text-[11px] leading-relaxed">
          {invalid ? (
            <span className="flex items-start gap-1.5 text-warning">
              <TriangleAlert className="mt-px size-3 shrink-0" />
              Position invalide : chevauchement, ou trop proche du bord ou d’un autre trou.
            </span>
          ) : (
            <span className="text-muted-foreground">
              Clic pour ajouter · glisser pour déplacer · molette pour redimensionner · double-clic
              pour retirer.
            </span>
          )}
        </div>

        <Separator />

        <div className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
              Angle de charge
            </span>
            <span className="tnum font-mono text-sm">{loadAngleDeg.toFixed(0)}°</span>
          </div>
          <Slider
            value={[loadAngleDeg]}
            onValueChange={([v]) => onLoadAngleChange(v)}
            min={-90}
            max={0}
            step={1}
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>−90° (cisaillement)</span>
            <span>0° (traction)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
