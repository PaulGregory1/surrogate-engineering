import Plot from 'react-plotly.js'
import type { Data } from 'plotly.js'
import { INK, MARK, PLOT_CONFIG, axisX, axisY, baseLayout } from '@/lib/chartTheme'
import type { BracketData, BracketRun } from '@/lib/bracket'

/**
 * Lightest FEA-feasible mass against FEA runs spent, one step line per optimiser.
 *
 * Series separate by stroke and marker shape, not hue — the charte has no categorical
 * palette. The winning run takes the measure orange, solid; the second agent a
 * dashed burnt; Optuna, the reference it is compared with, a dotted ink2. Each FEA
 * evaluation is a marker: filled when FEA found it feasible, hollow when not. The
 * overall best design is the one black mark, knocked out of the paper with a ring.
 */
const STYLE: Record<string, { color: string; dash: 'solid' | 'dash' | 'dot'; symbol: string }> = {
  'agent-01': { color: MARK.measure, dash: 'solid', symbol: 'circle' },
  'agent-02': { color: MARK.burnt, dash: 'dash', symbol: 'square' },
  'optuna-01': { color: INK.ink2, dash: 'dot', symbol: 'diamond' },
}

export function BracketMassPlot({ data, best, height = 300 }: { data: BracketData; best: BracketRun; height?: number }) {
  const maxN = Math.max(...data.runs.map((r) => r.trace.length))
  const traces: Data[] = []

  for (const run of data.runs) {
    const s = STYLE[run.key] ?? STYLE['optuna-01']
    const pts = run.trace.filter((p) => p.best_g !== null)
    traces.push({
      x: pts.map((p) => p.n),
      y: pts.map((p) => p.best_g),
      type: 'scatter',
      mode: 'lines',
      name: run.label,
      legendgroup: run.key,
      line: { color: s.color, width: run.key === best.key ? 2 : 1.5, dash: s.dash, shape: 'hv' },
      hoverinfo: 'skip',
    })
    traces.push({
      x: run.trace.map((p) => p.n),
      y: run.trace.map((p) => p.mass_g),
      customdata: run.trace.map((p) => [p.sigma_eval, p.delta_max, p.feasible ? 'feasible' : 'infeasible']),
      type: 'scatter',
      mode: 'markers',
      name: run.label,
      legendgroup: run.key,
      showlegend: false,
      marker: {
        symbol: run.trace.map((p) => (p.feasible ? s.symbol : `${s.symbol}-open`)),
        size: 6,
        color: s.color,
        line: { color: INK.surface, width: 1 },
      },
      hovertemplate: `${run.label} · FEA %{x}<br>%{y:.2f} g · σ %{customdata[0]:.1f} MPa · δ %{customdata[1]:.3f} mm · %{customdata[2]}<extra></extra>`,
    })
  }

  const focus = best.trace.find((p) => p.mass_g === best.mass_g)
  if (focus) {
    traces.push({
      x: [focus.n],
      y: [focus.mass_g],
      type: 'scatter',
      mode: 'markers',
      marker: { size: 9, color: MARK.focus, line: { color: INK.surface, width: 2 } },
      hoverinfo: 'skip',
      showlegend: false,
    })
  }

  return (
    <Plot
      data={traces}
      layout={baseLayout({
        height,
        margin: { l: 52, r: 16, t: 30, b: 38 },
        xaxis: axisX('FEA RUN', { range: [0.5, maxN + 0.5], dtick: 1 }),
        yaxis: axisY('MASS (g)', { tickformat: '.0f' }),
        shapes: [
          {
            type: 'line',
            xref: 'paper',
            x0: 0,
            x1: 1,
            y0: data.baseline.mass_g,
            y1: data.baseline.mass_g,
            line: { color: MARK.context, width: 1, dash: 'dash' },
          },
        ],
        annotations: [
          {
            xref: 'paper',
            x: 1,
            y: data.baseline.mass_g,
            xanchor: 'right',
            yanchor: 'bottom',
            showarrow: false,
            text: `BASELINE ${data.baseline.mass_g.toFixed(1)} g${data.baseline.feasible ? '' : ' · INFEASIBLE'}`,
            font: { size: 9, color: INK.muted, family: '"IBM Plex Mono", ui-monospace, monospace' },
          },
        ],
        hovermode: 'closest',
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          font: { size: 9.5, color: INK.muted, family: '"Instrument Sans", ui-sans-serif, sans-serif' },
        },
      })}
      useResizeHandler
      config={PLOT_CONFIG}
      style={{ width: '100%' }}
    />
  )
}
