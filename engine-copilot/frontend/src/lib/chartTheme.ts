import type { Layout, Config } from 'plotly.js'

// Plotly draws into its own SVG and never resolves CSS custom properties, so the
// tokens it needs are mirrored here as literals. These MUST stay in step with
// `src/index.css` — that file is the source of truth for the app, this one is the
// source of truth for the plots.
export const INK = {
  /** axis ticks, tick labels, legend text — muted, never a series color */
  muted: '#8b93a3',
  /** hairline gridlines, one step off the card surface */
  grid: '#1e222b',
  /** baseline / zero line, a step stronger than the grid */
  axis: '#2a3040',
  /** the card surface the plots sit on — the plane rings and gaps are drawn in */
  surface: '#121419',
} as const

/** Categorical slots, in fixed order. Never cycled — a chart past slot 5 needs
 *  facets or an "Other" bucket, not a generated hue. */
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'] as const

export const STATUS = {
  good: '#0ca30c',
  warning: '#fab219',
  critical: '#d03b3b',
} as const

// ── Colorscales ──────────────────────────────────────────────────────────────
// Two jobs, two kinds of scale:
//   * magnitude (unsigned) → sequential, perceptually uniform, light→dark
//   * polarity (signed)    → diverging, two hues around a neutral midpoint, and
//                            ALWAYS anchored with `zmid: 0` so the neutral sits on
//                            zero rather than wherever the data happens to centre.

/** Plotly 3.x bundles only a fixed built-in colorscale list and silently falls back
 *  to a default when handed an unknown name — 'Magma' is one of the missing ones, so
 *  it is spelled out (matplotlib's magma, 17 stops). */
export const MAGMA: [number, string][] = [
  [0.0, '#000004'],
  [0.0625, '#0a0822'],
  [0.125, '#1d1147'],
  [0.1875, '#36106b'],
  [0.25, '#51127c'],
  [0.3125, '#6a1c81'],
  [0.375, '#832681'],
  [0.4375, '#9c2e7f'],
  [0.5, '#b73779'],
  [0.5625, '#d0416f'],
  [0.625, '#e75263'],
  [0.6875, '#f56b5c'],
  [0.75, '#fc8961'],
  [0.8125, '#fea772'],
  [0.875, '#fec488'],
  [0.9375, '#fde2a3'],
  [1.0, '#fcfdbf'],
]

/** Diverging blue↔red through a neutral gray. Warm/cool poles read as opposite and
 *  the gray midpoint reads as "nothing", which a second hue never does. Pair with
 *  `zmid: 0`. */
export const DIVERGING: [number, string][] = [
  [0.0, '#0d366b'],
  [0.15, '#1c5cab'],
  [0.3, '#3987e5'],
  [0.42, '#9ec5f4'],
  [0.5, '#383835'],
  [0.58, '#f4b0ae'],
  [0.7, '#e34948'],
  [0.85, '#b02423'],
  [1.0, '#6d1414'],
]

/** Sequential single-hue blue, light→dark reversed for dark surfaces (low = near the
 *  surface, high = bright). Used for unsigned magnitude. */
export const SEQUENTIAL: [number, string][] = [
  [0.0, '#0d1b2e'],
  [0.25, '#184f95'],
  [0.5, '#2a78d6'],
  [0.75, '#86b6ef'],
  [1.0, '#e3eefc'],
]

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

/** Shared chrome: transparent surfaces (the Card behind it supplies the plane),
 *  muted ink, hairline grid, and a unified hover label. */
export function baseLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  return {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: INK.muted, size: 10, family: 'ui-sans-serif, system-ui, sans-serif' },
    hoverlabel: {
      bgcolor: '#161a21',
      bordercolor: '#2a3040',
      font: { color: '#e9ecf2', size: 11, family: 'ui-sans-serif, system-ui, sans-serif' },
    },
    margin: { l: 44, r: 12, t: 8, b: 34 },
    ...overrides,
  }
}

/** A quantitative axis: hairline grid, no zero-line emphasis, tabular tick text. */
export function axis(title?: string, overrides: Record<string, unknown> = {}) {
  return {
    title: title ? { text: title, font: { size: 10, color: INK.muted } } : undefined,
    gridcolor: INK.grid,
    gridwidth: 1,
    linecolor: INK.axis,
    zeroline: false,
    tickfont: { size: 9, color: INK.muted },
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
