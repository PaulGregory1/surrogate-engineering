import { useEffect, useMemo, useRef } from 'react'
import { STREAK, sampleScale } from '@/lib/chartTheme'
import { rk4Step } from '@/geometry'

const N_PARTICLES = 56
// Long enough that each particle draws a continuous streak rather than a dash: at the
// slow advection speed below, a short trail covers so little ground per frame that the
// field reads as scattered flecks instead of flowing lines.
const TRAIL_LEN = 28
// Index-units advected per 30fps frame at unit velocity. At Re=80 the inlet is
// ~0.08 lattice units, so this puts a particle's crossing of the ~96-wide downscaled
// domain at roughly 25s — slow enough to read the wake structure and follow a single
// trace with your eye. Raising it past ~90 makes the traces flicker rather than flow,
// because each rAF step then jumps further than the trail segments can describe.
const ADVECT_SCALE = 42
// Fraction of a particle's life spent fading in/out, so traces dissolve at the
// outflow and appear at the inlet instead of popping.
const FADE = 0.12

interface Particle {
  x: number
  y: number
  trail: { x: number; y: number }[]
  /** 0→1 over the particle's traversal, used only for the fade envelope */
  life: number
}

// Client-side RK4 streamlines over the predicted (u, v) field — instantaneous,
// no round trip. Particles are animated by mutating pre-allocated SVG elements
// directly via refs on every requestAnimationFrame tick instead of going through
// React state, so 56 particles x 27 trail segments doesn't trigger a re-render 60
// times a second.
//
// Each trace is coloured by the local flow speed on the same blue→red ramp as the
// field panels above it: slow water blue, freestream red. Colour goes on the <g>
// (one write per particle per frame) rather than on each <line>, and the segments
// inherit it — the per-segment work stays the four coordinates plus opacity.
export function StreamlineOverlay({
  u,
  v,
  width,
  height,
}: {
  u: number[][]
  v: number[][]
  width: number
  height: number
}) {
  const lineRefs = useRef<(SVGLineElement | null)[][]>([])
  const groupRefs = useRef<(SVGGElement | null)[]>([])
  const particlesRef = useRef<Particle[]>([])
  const fieldRef = useRef({ u, v })
  fieldRef.current = { u, v }

  // Normalising speed against the field's own maximum keeps the ramp meaningful
  // across Reynolds numbers, where the absolute inlet speed changes by an order of
  // magnitude but the wake structure is what the reader is after.
  const speedMax = useMemo(() => {
    let m = 0
    for (let j = 0; j < u.length; j++) {
      for (let i = 0; i < u[j].length; i++) {
        const s = Math.hypot(u[j][i], v[j][i])
        if (s > m) m = s
      }
    }
    return m || 1
  }, [u, v])

  const speedMaxRef = useRef(speedMax)
  speedMaxRef.current = speedMax

  // 24 pre-sampled steps: interpolating the ramp per particle per frame would be
  // 56 string builds every 16ms for a difference no eye can resolve.
  const swatches = useMemo(
    () => Array.from({ length: 24 }, (_, i) => sampleScale(STREAK, i / 23)),
    [],
  )

  useEffect(() => {
    function spawn(seeded = false): Particle {
      const y0 = Math.random() * (height - 1)
      // On first mount, scatter the seeds across the domain instead of releasing
      // them all from the inlet — otherwise the first few seconds show one dense
      // vertical curtain of traces marching across rather than a settled field.
      const x0 = seeded ? Math.random() * (width - 1) : 0
      return { x: x0, y: y0, trail: [{ x: x0, y: y0 }], life: 0 }
    }
    particlesRef.current = Array.from({ length: N_PARTICLES }, () => spawn(true))

    let raf = 0
    let last = performance.now()

    function tick(now: number) {
      const dtReal = Math.min((now - last) / 1000, 0.05)
      last = now
      const { u: uf, v: vf } = fieldRef.current
      const dt = dtReal * 30 * ADVECT_SCALE

      particlesRef.current.forEach((p, pi) => {
        const [nx, ny] = rk4Step(uf, vf, p.x, p.y, dt)
        if (nx >= width - 1 || nx < 0 || ny < 0 || ny >= height - 1) {
          Object.assign(p, spawn())
        } else {
          p.x = nx
          p.y = ny
          p.life = nx / (width - 1)
          p.trail.push({ x: nx, y: ny })
          if (p.trail.length > TRAIL_LEN) p.trail.shift()
        }

        // Ramp in over the first FADE of the traversal and out over the last.
        const envelope = Math.min(1, p.life / FADE, (1 - p.life) / FADE)

        const g = groupRefs.current[pi]
        if (g) {
          const j = Math.min(height - 1, Math.max(0, Math.round(p.y)))
          const i = Math.min(width - 1, Math.max(0, Math.round(p.x)))
          const speed = Math.hypot(uf[j]?.[i] ?? 0, vf[j]?.[i] ?? 0) / speedMaxRef.current
          g.setAttribute('stroke', swatches[Math.min(swatches.length - 1, (speed * swatches.length) | 0)])
        }

        const segs = lineRefs.current[pi]
        if (!segs) return
        for (let s = 0; s < TRAIL_LEN - 1; s++) {
          const el = segs[s]
          if (!el) continue
          const a = p.trail[s]
          const b = p.trail[s + 1]
          if (a && b) {
            el.setAttribute('x1', String(a.x))
            el.setAttribute('y1', String(a.y))
            el.setAttribute('x2', String(b.x))
            el.setAttribute('y2', String(b.y))
            // Linear fade from the head of the trail to nothing at the tail. Both
            // the near-opaque ceiling and the linear (rather than eased) falloff are
            // there because these are hairlines on white paper: anything gentler and
            // the traces read as scattered flecks instead of continuous lines.
            el.setAttribute('stroke-opacity', String(Math.max(0, envelope * 0.95 * (s / TRAIL_LEN))))
          } else {
            el.setAttribute('stroke-opacity', '0')
          }
        }
      })
      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // `swatches` is memoised on [] and never changes; listing it keeps the deps
    // honest without ever re-seeding the particles.
  }, [width, height, swatches])

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" style={{ pointerEvents: 'none' }}>
      {/* Stroke is set on the group each frame from the local speed; the segments
          inherit it and only carry geometry plus their own trail-age opacity. */}
      {Array.from({ length: N_PARTICLES }).map((_, pi) => (
        <g
          key={pi}
          ref={(el) => {
            groupRefs.current[pi] = el
          }}
          stroke={swatches[0]}
        >
          {Array.from({ length: TRAIL_LEN - 1 }).map((_, s) => (
            <line
              key={s}
              ref={(el) => {
                if (!lineRefs.current[pi]) lineRefs.current[pi] = []
                lineRefs.current[pi][s] = el
              }}
              strokeWidth={0.55}
              strokeLinecap="round"
              strokeOpacity={0}
            />
          ))}
        </g>
      ))}
    </svg>
  )
}
