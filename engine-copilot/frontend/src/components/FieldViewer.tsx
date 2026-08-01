import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { ColorScaleLegend } from '@/components/ColorScaleLegend'
import { Skeleton } from '@/components/ui/skeleton'
import {
  DIVERGING,
  FIELD_AXIS,
  FIELD_MARGIN,
  INK,
  PLOT_CONFIG,
  SEQUENTIAL,
  baseLayout,
  fieldRange,
  symmetricRange,
} from '@/lib/chartTheme'
import { boundaryToGrid } from '@/geometry'
import type { FlowPredictResponse } from '@/api'

// Scale per field, by what the field actually *is*:
//   u is dominated by the freestream and reads as a magnitude — how fast is the flow
//     here — so it takes the sequential ramp, where the wake deficit shows as a dark
//     trail. Diverging it around zero paints the whole domain one colour, because
//     almost every cell sits at the positive end and the symmetric range flattens it.
//   v and p genuinely straddle zero — the counter-rotating shear layers and the
//     front/rear pressure split ARE the signal — so they take the diverging scale,
//     pinned with `zmid: 0`. Without that anchor Plotly centres the neutral colour on
//     the data's own midpoint and the sign stops meaning anything.
const PANELS: {
  key: 'u' | 'v' | 'p'
  title: string
  caption: string
  colorscale: [number, string][]
  diverging: boolean
}[] = [
  {
    key: 'u',
    title: 'Vitesse axiale',
    caption: 'u — sombre = déficit de sillage',
    colorscale: SEQUENTIAL,
    diverging: false,
  },
  {
    key: 'v',
    title: 'Vitesse transverse',
    caption: 'v — cisaillement, signé',
    colorscale: DIVERGING,
    diverging: true,
  },
  {
    key: 'p',
    title: 'Pression',
    caption: 'p — dépression arrière, signée',
    colorscale: DIVERGING,
    diverging: true,
  },
]

function maskField(field: number[][], mask: number[][]): (number | null)[][] {
  return field.map((row, j) => row.map((v, i) => (mask[j][i] ? null : v)))
}

export function FieldViewer({ result, loading }: { result: FlowPredictResponse | null; loading: boolean }) {
  // The obstacle silhouette drawn over the masked cells, so the body reads as a solid
  // object rather than a hole in the data.
  const bodyPath = useMemo(() => {
    if (!result) return null
    const { x, y } = boundaryToGrid(result.boundary_xy, result.domain)
    return x.map((xi, i) => `${i === 0 ? 'M' : 'L'} ${xi.toFixed(2)} ${y[i].toFixed(2)}`).join(' ') + ' Z'
  }, [result])

  const panels = useMemo(() => {
    if (!result) return null
    return PANELS.map((p) => {
      const z = maskField(result.fields[p.key], result.mask)
      const [min, max] = p.diverging ? symmetricRange(z) : fieldRange(z)
      return { ...p, z, min, max }
    })
  }, [result])

  if (loading || !panels) {
    return (
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {PANELS.map((p) => (
          <Skeleton key={p.key} className="h-[212px] w-full" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {panels.map((p) => (
        <div key={p.key} className="flex flex-col overflow-hidden rounded-xl border border-border bg-card">
          <div className="px-3 pt-2.5">
            <div className="text-xs font-medium">{p.title}</div>
            <div className="text-[10px] text-muted-foreground">{p.caption}</div>
          </div>
          <Plot
            data={[
              {
                z: p.z,
                type: 'heatmap',
                colorscale: p.colorscale,
                zmid: p.diverging ? 0 : undefined,
                zsmooth: 'best',
                showscale: false,
                hovertemplate: `${p.key} = %{z:.4f}<extra></extra>`,
              },
            ]}
            layout={baseLayout({
              margin: FIELD_MARGIN,
              height: 150,
              xaxis: { ...FIELD_AXIS, scaleanchor: 'y' },
              yaxis: { ...FIELD_AXIS, autorange: 'reversed' },
              shapes: bodyPath
                ? [
                    {
                      type: 'path',
                      path: bodyPath,
                      xref: 'x',
                      yref: 'y',
                      fillcolor: INK.surface,
                      line: { color: INK.axis, width: 1 },
                    },
                  ]
                : [],
            })}
            useResizeHandler
            config={PLOT_CONFIG}
            style={{ width: '100%' }}
          />
          <ColorScaleLegend
            stops={p.colorscale}
            min={p.min}
            max={p.max}
            diverging={p.diverging}
            className="px-3 pb-2.5"
          />
        </div>
      ))}
    </div>
  )
}
