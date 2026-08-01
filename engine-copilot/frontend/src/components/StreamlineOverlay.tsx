import { useEffect, useRef } from 'react'
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
// no round trip. Particles are animated by mutating pre-allocated SVG <line>
// elements directly via refs on every requestAnimationFrame tick instead of
// going through React state, so 56 particles x 27 trail segments doesn't trigger
// a re-render 60 times a second.
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
  const particlesRef = useRef<Particle[]>([])
  const fieldRef = useRef({ u, v })
  fieldRef.current = { u, v }

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
            // Brightest at the head of the trail, fading to nothing at the tail.
            el.setAttribute('stroke-opacity', String(Math.max(0, envelope * 0.75 * (s / TRAIL_LEN) ** 1.6)))
          } else {
            el.setAttribute('stroke-opacity', '0')
          }
        }
      })
      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [width, height])

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" style={{ pointerEvents: 'none' }}>
      {Array.from({ length: N_PARTICLES }).map((_, pi) => (
        <g key={pi}>
          {Array.from({ length: TRAIL_LEN - 1 }).map((_, s) => (
            <line
              key={s}
              ref={(el) => {
                if (!lineRefs.current[pi]) lineRefs.current[pi] = []
                lineRefs.current[pi][s] = el
              }}
              stroke="var(--ring)"
              strokeWidth={0.35}
              strokeLinecap="round"
              strokeOpacity={0}
            />
          ))}
        </g>
      ))}
    </svg>
  )
}
