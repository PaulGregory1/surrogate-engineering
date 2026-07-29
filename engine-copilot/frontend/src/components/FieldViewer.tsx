import Plot from 'react-plotly.js'
import { Skeleton } from './Skeleton'
import type { FlowPredictResponse } from '../api'

const PANELS: { key: 'u' | 'v' | 'p'; title: string; colorscale: string }[] = [
  { key: 'u', title: 'u [–]', colorscale: 'Viridis' },
  { key: 'v', title: 'v [–]', colorscale: 'Viridis' },
  { key: 'p', title: 'p [–]', colorscale: 'RdBu' },
]

function maskField(field: number[][], mask: number[][]): (number | null)[][] {
  return field.map((row, j) => row.map((v, i) => (mask[j][i] ? null : v)))
}

export function FieldViewer({ result, loading }: { result: FlowPredictResponse | null; loading: boolean }) {
  if (loading || !result) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {PANELS.map((p) => (
          <Skeleton key={p.key} className="h-56 w-full" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
      {PANELS.map((p) => (
        <div key={p.key} className="nc-card p-2">
          <Plot
            data={[
              {
                z: maskField(result.fields[p.key], result.mask),
                type: 'heatmap',
                colorscale: p.colorscale,
                zsmooth: 'best',
                showscale: true,
                colorbar: { title: { text: p.title }, thickness: 12, len: 0.85 },
              },
            ]}
            layout={{
              margin: { l: 24, r: 8, t: 8, b: 24 },
              height: 220,
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#9ca3af', size: 10 },
              xaxis: { visible: false, scaleanchor: 'y' },
              yaxis: { visible: false, autorange: 'reversed' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      ))}
    </div>
  )
}
