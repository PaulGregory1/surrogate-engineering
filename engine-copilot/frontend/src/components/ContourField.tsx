import { useMemo } from 'react'

// Grid resolution of the scalar field the contours are traced from. 96x64 is the
// point where the curves stop showing polygonal corners at full-screen width;
// finer costs path length for nothing, since the lines are drawn at 5% ink.
const NX = 96
const NY = 64
const LEVELS = 20

/** Three fixed Gaussian bumps and one broad saddle — a smooth, closed-contour field
 *  with a couple of local optima, which is what a surrogate response surface
 *  actually looks like. Positions are hand-placed so the busiest part of the map
 *  falls in the page gutters rather than behind the content column. */
function field(x: number, y: number): number {
  const bump = (cx: number, cy: number, s: number) =>
    Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
  return (
    1.0 * bump(0.16, 0.24, 0.22) +
    0.85 * bump(0.86, 0.7, 0.26) +
    0.55 * bump(0.55, 0.12, 0.18) -
    0.45 * bump(0.62, 0.52, 0.3)
  )
}

/**
 * Marching squares, emitting one SVG path per iso-level.
 *
 * Segments are concatenated into a single `M…L…` string per level rather than
 * stitched into continuous polylines: joining them would need an adjacency pass and
 * buys nothing here, because at 1px with butt caps a chain of abutting segments is
 * indistinguishable from the polyline it approximates.
 *
 * The saddle cases (5 and 10) are resolved by the cell-centre average, the standard
 * disambiguation — left arbitrary they produce visible X crossings exactly where the
 * eye is drawn.
 */
function contourPaths(): string[] {
  const g: number[][] = []
  for (let j = 0; j <= NY; j++) {
    const row: number[] = []
    for (let i = 0; i <= NX; i++) row.push(field(i / NX, j / NY))
    g.push(row)
  }

  let lo = Infinity
  let hi = -Infinity
  for (const row of g) for (const v of row) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }

  const paths: string[] = []
  for (let k = 1; k <= LEVELS; k++) {
    const level = lo + ((hi - lo) * k) / (LEVELS + 1)
    let d = ''

    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const tl = g[j][i]
        const tr = g[j][i + 1]
        const br = g[j + 1][i + 1]
        const bl = g[j + 1][i]

        let code = 0
        if (tl > level) code |= 8
        if (tr > level) code |= 4
        if (br > level) code |= 2
        if (bl > level) code |= 1
        if (code === 0 || code === 15) continue

        // Linear interpolation of the crossing along each edge, in grid units.
        const t = (a: number, b: number) => (level - a) / (b - a)
        const top = { x: i + t(tl, tr), y: j }
        const right = { x: i + 1, y: j + t(tr, br) }
        const bottom = { x: i + t(bl, br), y: j + 1 }
        const left = { x: i, y: j + t(tl, bl) }

        const seg = (a: { x: number; y: number }, b: { x: number; y: number }) => {
          d += `M${a.x.toFixed(2)} ${a.y.toFixed(2)}L${b.x.toFixed(2)} ${b.y.toFixed(2)}`
        }

        switch (code) {
          case 1:
          case 14:
            seg(left, bottom)
            break
          case 2:
          case 13:
            seg(bottom, right)
            break
          case 3:
          case 12:
            seg(left, right)
            break
          case 4:
          case 11:
            seg(top, right)
            break
          case 6:
          case 9:
            seg(top, bottom)
            break
          case 7:
          case 8:
            seg(left, top)
            break
          case 5:
          case 10: {
            const centre = (tl + tr + br + bl) / 4
            const joinTopLeft = code === 5 ? centre > level : centre <= level
            if (joinTopLeft) {
              seg(left, top)
              seg(bottom, right)
            } else {
              seg(left, bottom)
              seg(top, right)
            }
            break
          }
        }
      }
    }
    paths.push(d)
  }
  return paths
}

/**
 * The hard-edged layer of the background, as a topographic map: iso-lines of a
 * smooth scalar field, which is the shape of the thing this whole product predicts.
 *
 * Scoped to the main column on purpose — it is mounted inside the content area's
 * own stacking context, not fixed to the viewport, so it never appears behind the
 * nav rail or the copilot panel. Content panels are opaque (cards) or nearly so
 * (`.cq-calque`, .88 over a 20px backdrop blur), so nothing carrying a plot ever has
 * these running underneath it.
 *
 * Drawn in `--burnt`, not ink. Orange in this charte has to encode a quantity, and
 * iso-lines are the one background element that genuinely does — they are level sets
 * of a scalar field, which is the shape of the thing this product predicts. Burnt
 * rather than `--or` because these are hairlines: at 1px on paper the bright orange
 * goes pink and dissolves.
 */
export function ContourField() {
  const paths = useMemo(contourPaths, [])

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${NX} ${NY}`}
      preserveAspectRatio="none"
      aria-hidden
    >
      {paths.map((d, k) => (
        <path
          key={k}
          d={d}
          fill="none"
          // Faintest at the outside, a shade denser towards the peaks — the same
          // "density carries magnitude" reading the field plots use. #B03E00 is
          // --burnt.
          stroke={`rgba(176,62,0,${(0.1 + 0.09 * (k / (LEVELS - 1))).toFixed(3)})`}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  )
}
