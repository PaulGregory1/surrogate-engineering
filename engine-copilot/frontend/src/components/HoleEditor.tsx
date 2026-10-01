import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
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
    <Card className="gap-5 py-5">
      <CardHeader className="px-5">
        <CardTitle className="cq-title">Perforated plate</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5 px-5">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full rounded-[12px] border border-line bg-paper"
          style={{ aspectRatio: `${WIDTH} / ${HEIGHT}`, touchAction: 'none' }}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          <rect
            x={0}
            y={0}
            width={WIDTH}
            height={HEIGHT}
            fill="var(--white)"
            stroke="var(--line2)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
            className="cursor-crosshair"
            onClick={handleBackgroundClick}
          />
          {/* The two boundary conditions, told apart by what they are rather than by
              a status hue: the clamp is structure, so it is ink; the load is an
              applied quantity, so it is the one thing here allowed to be orange. */}
          <line
            x1={0.004}
            y1={0}
            x2={0.004}
            y2={HEIGHT}
            stroke="var(--ink)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1={WIDTH - 0.004}
            y1={0}
            x2={WIDTH - 0.004}
            y2={HEIGHT}
            stroke="var(--or)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />

          {holes.map(([x, y, r], i) => (
            <motion.circle
              key={i}
              fill="var(--paper)"
              stroke={dragIndex === i ? 'var(--or)' : 'var(--ink)'}
              strokeWidth={dragIndex === i ? 2 : 1}
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

        <div className="flex items-center gap-4">
          <span className="cq-k flex items-center gap-1.5">
            <span className="h-0.5 w-3 bg-ink" />
            CLAMPED
          </span>
          <span className="cq-k flex items-center gap-1.5">
            <span className="h-0.5 w-3 bg-or" />
            LOADED
          </span>
          <span className="cq-k ml-auto">{holes.length}/4 HOLES</span>
        </div>

        <div className="min-h-[3em] text-band leading-[1.7]">
          {invalid ? (
            // An out-of-envelope state is burnt, never red: the charte has no red,
            // and introducing one would break the reading of the ramp.
            <div className="cq-error py-0">
              <div className="text-body font-medium">Invalid position.</div>
              <div className="cq-muted mt-0.5 text-band">
                Overlapping, or too close to the edge or to another hole.
              </div>
            </div>
          ) : (
            <span className="text-ink2">
              Click to add · drag to move · scroll to resize · double-click to remove.
            </span>
          )}
        </div>

        <Separator />

        <div className="flex flex-col gap-2.5">
          <div className="flex items-baseline justify-between">
            <span className="cq-k">LOAD ANGLE</span>
            <span className="cq-num text-band">{loadAngleDeg.toFixed(0)}°</span>
          </div>
          <Slider
            value={[loadAngleDeg]}
            onValueChange={([v]) => onLoadAngleChange(v)}
            min={-90}
            max={0}
            step={1}
          />
          <div className="flex justify-between font-mono text-[9px] tracking-[0.1em] text-ink3">
            <span>−90° SHEAR</span>
            <span>0° TENSION</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
