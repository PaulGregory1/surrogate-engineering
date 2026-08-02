/**
 * Calque's single motion curve, shared by every entrance in the app.
 *
 * The charte fixes one breathing pattern — `opacity 0→1` with `translateY(22px)→0`
 * over `.9s` on `cubic-bezier(.2,.7,.2,1)` — and asks that it be used everywhere
 * rather than re-tuned per component. Reduced motion is honoured globally in
 * `index.css`, which collapses every animation and transition to ~0ms; these
 * helpers only decide the shape of the movement, not whether it happens.
 */
export const CALQUE_EASE: [number, number, number, number] = [0.2, 0.7, 0.2, 1]

/** Block entrance: the full 22px rise, for a card or a panel arriving on screen. */
export function reveal(delay = 0) {
  return {
    initial: { opacity: 0, y: 22 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.9, ease: CALQUE_EASE, delay },
  }
}

/**
 * In-place state change — a readout swapping value, a badge appearing beside one
 * that was already there. Short and shallow: the 22px rise is an *arrival*, and
 * using it for a number that merely changed makes the panel look like it reloaded.
 */
export function settle(delay = 0) {
  return {
    initial: { opacity: 0, y: 6 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0 },
    transition: { duration: 0.42, ease: CALQUE_EASE, delay },
  }
}

/**
 * Stagger a set of marks by where they sit on an axis rather than by array index.
 * A sweep that follows x tells you something; `i * 30ms` tells you nothing.
 */
export function axisDelay(value: number, min: number, max: number): number {
  if (!(max > min)) return 0.1
  return 0.1 + ((value - min) / (max - min)) * 1.1
}
