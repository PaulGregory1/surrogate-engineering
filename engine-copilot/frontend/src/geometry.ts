// Client-side replica of flowsurrogate.geometry.radius(), so the contour
// preview updates instantly on slider drag without a round trip. Must stay
// numerically identical to the Python version (see engine-copilot backend tests).
export const MODES = [2, 3, 4]
export const COEFF_RANGE = 0.16

export function radius(theta: number[], r0: number, coeffs: number[]): number[] {
  return theta.map((t) => {
    let factor = 1
    MODES.forEach((k, j) => {
      factor += coeffs[2 * j] * Math.cos(k * t) + coeffs[2 * j + 1] * Math.sin(k * t)
    })
    return r0 * factor
  })
}

export function boundaryXY(r0: number, coeffs: number[], n = 200): { x: number[]; y: number[] } {
  const theta = Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n)
  const r = radius(theta, r0, coeffs)
  return {
    x: r.map((ri, i) => ri * Math.cos(theta[i])),
    y: r.map((ri, i) => ri * Math.sin(theta[i])),
  }
}

// r0 that gives exactly `targetArea` for these harmonics (mirrors r0_for_area).
export function r0ForArea(coeffs: number[], targetArea: number): number {
  const sumSq = coeffs.reduce((s, c) => s + c * c, 0)
  return Math.sqrt(targetArea / (Math.PI * (1 + 0.5 * sumSq)))
}

// Bilinear sample of a (ny, nx) field at fractional grid coordinates, clamped to
// the field's extent (particles that drift to the boundary just stall there
// instead of throwing).
export function bilinear(field: number[][], x: number, y: number): number {
  const ny = field.length
  const nx = field[0].length
  const xc = Math.min(Math.max(x, 0), nx - 1.001)
  const yc = Math.min(Math.max(y, 0), ny - 1.001)
  const x0 = Math.floor(xc)
  const y0 = Math.floor(yc)
  const fx = xc - x0
  const fy = yc - y0
  const v00 = field[y0][x0]
  const v10 = field[y0][x0 + 1]
  const v01 = field[y0 + 1][x0]
  const v11 = field[y0 + 1][x0 + 1]
  return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy
}

// One RK4 step of a massless particle advected by (u, v), in grid-index units.
export function rk4Step(u: number[][], v: number[][], x: number, y: number, dt: number): [number, number] {
  const k1x = bilinear(u, x, y)
  const k1y = bilinear(v, x, y)
  const k2x = bilinear(u, x + (dt / 2) * k1x, y + (dt / 2) * k1y)
  const k2y = bilinear(v, x + (dt / 2) * k1x, y + (dt / 2) * k1y)
  const k3x = bilinear(u, x + (dt / 2) * k2x, y + (dt / 2) * k2y)
  const k3y = bilinear(v, x + (dt / 2) * k2x, y + (dt / 2) * k2y)
  const k4x = bilinear(u, x + dt * k3x, y + dt * k3y)
  const k4y = bilinear(v, x + dt * k3x, y + dt * k3y)
  return [x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x), y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)]
}

// Shape boundary (shape-centred frame) -> grid-index coordinates of the downscaled
// flow fields, so the streamline overlay's obstacle silhouette lines up with the
// u/v arrays it advects particles through. Must match flowsurrogate.geometry.Domain's
// defaults (cx=56, cy=48) and config.FIELD_DOWNSCALE (2) on the backend.
export function boundaryToGrid(
  boundaryXYArr: number[][],
  domain: { cx: number; cy: number; downscale: number },
): { x: number[]; y: number[] } {
  const [xs, ys] = boundaryXYArr
  return {
    x: xs.map((xi) => (xi + domain.cx) / domain.downscale),
    y: ys.map((yi) => (yi + domain.cy) / domain.downscale),
  }
}
