import Plot from 'react-plotly.js'
import { INK, MARK, PLOT_CONFIG, axisX, axisY, baseLayout } from '@/lib/chartTheme'

export interface ConvergencePoint {
  it: number
  cd_best: number
  cd_mean: number
}

/**
 * Two series, separated by stroke rather than by hue — the charte has no categorical
 * palette, and two oranges side by side are unreadable.
 *
 * `cd_best` is the quantity being optimised, so it takes the measure orange, solid.
 * `cd_mean` is population context around it and stays a dotted "gone" grey. The
 * final point is the focal mark: solid ink, knocked out of the paper with a 2px
 * ring, and the only black-filled thing in the plot.
 *
 * No entrance animation: the trace is streamed live from the optimiser, so it draws
 * itself iteration by iteration, which is the thing a `stroke-dashoffset` reveal
 * would only be imitating.
 *
 * With a single start there is no population: `cd_mean` is `cd_best`, so the dotted
 * grey would sit exactly under the orange and the legend would name a series that
 * isn't there. Both are dropped — one curve needs no key.
 */
export function ConvergencePlot({ history, height = 300 }: { history: ConvergencePoint[]; height?: number }) {
  const its = history.map((h) => h.it)
  const last = history[history.length - 1]
  const hasPopulation = history.some((h) => Math.abs(h.cd_mean - h.cd_best) > 1e-9)

  return (
    <Plot
      data={[
        {
          x: its,
          y: history.map((h) => h.cd_best),
          type: 'scatter',
          mode: 'lines',
          name: hasPopulation ? 'Cd meilleur' : 'Cd',
          line: { color: MARK.measure, width: 2, shape: 'spline', smoothing: 0.4 },
          hovertemplate: 'it %{x} · Cd %{y:.4f}<extra></extra>',
        },
        ...(hasPopulation
          ? [
              {
                x: its,
                y: history.map((h) => h.cd_mean),
                type: 'scatter' as const,
                mode: 'lines' as const,
                name: 'Cd moyen (population)',
                line: { color: MARK.gone, width: 1.25, dash: 'dot' as const },
                hovertemplate: 'it %{x} · Cd %{y:.4f}<extra></extra>',
              },
            ]
          : []),
        // The endpoint carries the one number worth labelling directly; every other
        // value stays in the axis and the hover layer.
        ...(last
          ? [
              {
                x: [last.it],
                y: [last.cd_best],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { size: 7, color: MARK.focus, line: { color: INK.surface, width: 2 } },
                hoverinfo: 'skip' as const,
                showlegend: false,
              },
            ]
          : []),
      ]}
      layout={baseLayout({
        height,
        margin: { l: 52, r: 16, t: 26, b: 38 },
        xaxis: axisX('ITÉRATION'),
        yaxis: axisY('Cd', { tickformat: '.3f' }),
        hovermode: 'x unified',
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.18,
          font: { size: 9.5, color: INK.muted, family: '"Instrument Sans", ui-sans-serif, sans-serif' },
        },
        showlegend: hasPopulation,
      })}
      useResizeHandler
      config={PLOT_CONFIG}
      style={{ width: '100%' }}
    />
  )
}
