import type { Layout, Config } from 'plotly.js'

// Plotly draws into its own SVG and never resolves CSS custom properties, so the
// Calque tokens it needs are mirrored here as literals. These MUST stay in step
// with `src/index.css` — that file is the source of truth for the app, this one is
// the source of truth for the plots.
export const INK = {
  /** axis titles, tick labels — never a series colour */
  muted: '#9C9691',
  /** hairline gridlines: horizontal only, and barely there */
  grid: 'rgba(13,12,11,.055)',
  /** the baseline rule, a step stronger than the grid */
  axis: '#D6D0CA',
  /** the paper the plots sit on — what marks are ringed and knocked out in */
  surface: '#FFFFFF',
  /** body ink */
  ink: '#0D0C0B',
  ink2: '#57534E',
} as const

/**
 * Marks, by *role* rather than by slot.
 *
 * The charte has no categorical palette: there aren't enough hues in papier /
 * orange / encre to separate series by colour, and two oranges side by side are
 * unreadable. Series separate by value (opacity, density), by stroke (solid,
 * dashed, dotted) or by marker shape. Grey is a state — "dominated", "discarded",
 * "superseded" — never an active series.
 */
export const MARK = {
  /** the quantity being optimised / measured */
  measure: '#FF6A00',
  /** orange as *text* or as a thin mark on white: `--or` is too light under 14px */
  burnt: '#B03E00',
  /** the focal object — one per chart, and it arrives last */
  focus: '#0D0C0B',
  /** context around the focal series: population mean, baseline, envelope */
  context: '#9C9691',
  /** dominated, discarded, superseded */
  gone: '#B8B1AA',
} as const

// ── Colorscales for the physics fields ───────────────────────────────────────
//
// DELIBERATE DEVIATION FROM THE CHARTE, decided by the product owner — do not
// "fix" this back to the orange ramp without asking.
//
// Calque says the orange measure ramp is the only ramp. These two scales are the
// documented exception, and they are scoped strictly to *rasterised solver
// fields*: velocity, pressure, vorticity, von Mises. Everywhere else — convergence
// traces, intervals, progress rules, gauges, table encodings — orange remains the
// only quantity colour, so the exception stays legible as an exception.
//
// The reason it is worth the deviation: blue→red is the entrenched convention in
// CFD and FEA post-processing (ParaView, Fluent, Abaqus all ship it as default).
// An engineer reads "blue is slow / cold / unloaded, red is fast / hot / critical"
// without a legend, and these panels are precisely where the app is competing with
// the tools they already use.
//
// Both scales sit on `--white` cards, so their neutral midpoint is warmed slightly
// off pure white — a cold #f7f7f7 next to this paper reads grey and dirty.

/**
 * Unsigned magnitude, low → high (axial velocity, von Mises stress).
 *
 * Read as a ramp rather than as a divergence: there is no meaningful zero here, so
 * the pale band around 0.5 is just the crossover, not a neutral. It carries less
 * perceptual order than a single-hue ramp would, which is the price of the
 * convention — the endpoints are what the reader is actually looking for.
 */
export const SEQUENTIAL: [number, string][] = [
  [0.0, '#1B4F8F'],
  [0.18, '#3A7EBF'],
  [0.36, '#8FBEDD'],
  [0.52, '#EFE7DF'],
  [0.68, '#F0A981'],
  [0.84, '#D45A43'],
  [1.0, '#8C1B22'],
]

/**
 * Signed fields (transverse velocity, pressure, vorticity): blue below zero, red
 * above, neutral exactly on it.
 *
 * Always pair with `zmid: 0` — without that anchor Plotly centres the neutral on
 * the data's own midpoint, the crossover drifts off zero and the sign stops meaning
 * anything, which is the one way a diverging scale can actively lie.
 */
export const DIVERGING: [number, string][] = [
  [0.0, '#053061'],
  [0.12, '#2166AC'],
  [0.26, '#4393C3'],
  [0.38, '#92C5DE'],
  [0.47, '#D6E6F0'],
  [0.5, '#F4F1EE'],
  [0.53, '#FBDDCC'],
  [0.62, '#F4A582'],
  [0.74, '#D6604D'],
  [0.87, '#B2182B'],
  [1.0, '#67001F'],
]

/**
 * The same blue→red reading, for **strokes** rather than filled cells.
 *
 * `SEQUENTIAL` cannot be reused here: its crossover band sits near white, which is
 * invisible as a 2px hairline on paper, so half the streamlines would simply vanish
 * at mid speed. This keeps the two poles and routes the transition through values
 * dark enough to hold a line, at the cost of a muddier middle that a filled raster
 * could not get away with.
 */
export const STREAK: [number, string][] = [
  [0.0, '#17457E'],
  [0.3, '#3F76AE'],
  [0.52, '#8A6E82'],
  [0.72, '#B85A45'],
  [1.0, '#8E1B20'],
]

/**
 * Sample a colorscale in JS, for marks that are *painted* rather than handed to
 * Plotly — SVG strokes, DOM swatches, canvas fills. `t` is clamped to [0,1], and
 * the scale's own stop positions are honoured, so an unevenly-spaced scale
 * interpolates the way its author intended rather than by index.
 */
export function sampleScale(scale: [number, string][], t: number): string {
  const x = Math.max(0, Math.min(1, t))
  let i = 0
  while (i < scale.length - 2 && scale[i + 1][0] < x) i++
  const [t0, c0] = scale[i]
  const [t1, c1] = scale[i + 1]
  const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0)
  const rgb = (h: string) =>
    [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)) as [number, number, number]
  const a = rgb(c0)
  const b = rgb(c1)
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`
}

// ── Layout ───────────────────────────────────────────────────────────────────

export const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
  // Plotly's default double-click "reset" fights the fixed extents these physical
  // fields are drawn at, and dragging a zoom box on a field plot is never what the
  // user meant on a live-updating panel.
  staticPlot: false,
  scrollZoom: false,
  doubleClick: false,
}

/** Shared chrome: transparent surfaces (the card behind supplies the paper), muted
 *  warm ink, and a hover label built like a small calque. */
export function baseLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  return {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: INK.muted, size: 10, family: '"Instrument Sans", ui-sans-serif, sans-serif' },
    hoverlabel: {
      bgcolor: '#FFFFFF',
      bordercolor: '#D6D0CA',
      font: { color: '#0D0C0B', size: 11, family: '"IBM Plex Mono", ui-monospace, monospace' },
    },
    margin: { l: 44, r: 12, t: 8, b: 34 },
    ...overrides,
  }
}

/** Y axis: horizontal hairlines, five or six divisions, no frame, no ticks.
 *  Numbers are mono, because a measure is not a word. */
export function axisY(title?: string, overrides: Record<string, unknown> = {}) {
  return {
    title: title ? { text: title, font: { size: 9.5, color: INK.muted } } : undefined,
    gridcolor: INK.grid,
    gridwidth: 1,
    showline: false,
    zeroline: false,
    ticks: '' as const,
    nticks: 6,
    tickfont: { size: 9, color: INK.muted, family: '"IBM Plex Mono", ui-monospace, monospace' },
    ...overrides,
  }
}

/** X axis: same, minus the grid — vertical gridlines are for time series only. */
export function axisX(title?: string, overrides: Record<string, unknown> = {}) {
  return {
    ...axisY(title, overrides),
    showgrid: false,
    linecolor: INK.axis,
    showline: true,
    ...overrides,
  }
}

/** A field plot's axis: the data is an image, so ticks and grid carry nothing. */
export const FIELD_AXIS = { visible: false, showgrid: false, zeroline: false } as const

/** Field heatmaps carry no Plotly colorbar at all — see `ColorScaleLegend`, which
 *  draws the key in the DOM instead. This margin is what a field plot uses once the
 *  bar is gone: just enough to keep the image off the card edge. */
export const FIELD_MARGIN = { l: 8, r: 8, t: 4, b: 8 } as const

/** Finite range of a field that may contain nulls where the body is masked out. */
export function fieldRange(field: (number | null)[][]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const row of field) {
    for (const v of row) {
      if (v === null || !Number.isFinite(v)) continue
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (!Number.isFinite(lo)) return [0, 1]
  return [lo, hi]
}

/** A diverging field is drawn symmetrically about zero (`zmid: 0`), so its colour key
 *  has to span the same symmetric range or the two would disagree. */
export function symmetricRange(field: (number | null)[][]): [number, number] {
  const [lo, hi] = fieldRange(field)
  const m = Math.max(Math.abs(lo), Math.abs(hi))
  return [-m, m]
}
