import Plot from 'react-plotly.js'
import { INK, PLOT_CONFIG, SERIES, axis, baseLayout } from '@/lib/chartTheme'

export interface ConvergencePoint {
  it: number
  cd_best: number
  cd_mean: number
}

/** cd_best is the metric being optimised, so it takes categorical slot 1 solid;
 *  cd_mean is population context around it and stays in muted ink with a dash — no
 *  second hue competes for attention. Two series, so a legend is mandatory. */
export function ConvergencePlot({ history, height = 300 }: { history: ConvergencePoint[]; height?: number }) {
  const its = history.map((h) => h.it)
  const last = history[history.length - 1]

  return (
    <Plot
      data={[
        {
          x: its,
          y: history.map((h) => h.cd_best),
          type: 'scatter',
          mode: 'lines',
          name: 'Cd meilleur',
          line: { color: SERIES[0], width: 2, shape: 'spline', smoothing: 0.4 },
          hovertemplate: 'it %{x} · Cd %{y:.4f}<extra></extra>',
        },
        {
          x: its,
          y: history.map((h) => h.cd_mean),
          type: 'scatter',
          mode: 'lines',
          name: 'Cd moyen (population)',
          line: { color: INK.muted, width: 1.5, dash: 'dot' },
          hovertemplate: 'it %{x} · Cd %{y:.4f}<extra></extra>',
        },
        // The endpoint carries the one number worth labelling directly; every other
        // value stays in the axis and the hover layer.
        ...(last
          ? [
              {
                x: [last.it],
                y: [last.cd_best],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { size: 8, color: SERIES[0], line: { color: INK.surface, width: 2 } },
                hoverinfo: 'skip' as const,
                showlegend: false,
              },
            ]
          : []),
      ]}
      layout={baseLayout({
        height,
        margin: { l: 48, r: 16, t: 24, b: 36 },
        xaxis: axis('itération'),
        yaxis: axis('Cd', { tickformat: '.3f' }),
        hovermode: 'x unified',
        legend: { orientation: 'h', x: 0, y: 1.16, font: { size: 10, color: INK.muted } },
        showlegend: true,
      })}
      useResizeHandler
      config={PLOT_CONFIG}
      style={{ width: '100%' }}
    />
  )
}
