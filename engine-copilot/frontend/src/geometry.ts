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
