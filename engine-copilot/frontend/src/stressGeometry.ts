// Client-side replica of stressgnn.geometry.is_valid()/constants, so HoleEditor
// can reject an invalid drag/add before it ever reaches the backend (same pattern
// as geometry.ts mirroring flowsurrogate.geometry.radius() for the Flow Lab).
export const WIDTH = 2.0
export const HEIGHT = 1.0
export const MIN_R = 0.08
export const MAX_R = 0.18
export const MARGIN = 0.07
export const GAP = 0.09

export function isValid(holes: number[][]): boolean {
  if (holes.length === 0) return true
  for (const [x, y, r] of holes) {
    if (x - r < MARGIN || x + r > WIDTH - MARGIN) return false
    if (y - r < MARGIN || y + r > HEIGHT - MARGIN) return false
  }
  for (let i = 0; i < holes.length; i++) {
    for (let j = i + 1; j < holes.length; j++) {
      const [xi, yi, ri] = holes[i]
      const [xj, yj, rj] = holes[j]
      const d = Math.hypot(xi - xj, yi - yj)
      if (d < ri + rj + GAP) return false
    }
  }
  return true
}
