import { useEffect, useRef, useState } from 'react'
import Plot from 'react-plotly.js'
import { Skeleton } from '@/components/ui/skeleton'
import { ColorScaleLegend } from '@/components/ColorScaleLegend'
import { FIELD_AXIS, FIELD_MARGIN, MAGMA, PLOT_CONFIG, SERIES, baseLayout, fieldRange } from '@/lib/chartTheme'
import type { StressPredictResponse } from '@/api'

// The "pulse" on the peak marker is driven as a Plotly trace (marker size/opacity
// oscillating), not literal CSS on a DOM overlay: Plotly owns its own internal SVG
// layout (margins, colorbar width) and a naively-positioned absolute div drifts out
// of alignment with the true data coordinates. Two stacked star traces — a growing,
// fading halo plus a fixed-size solid core — read as a pulse without fighting that.
export function StressViewer({
  result,
  loading,
  showPeak = true,
  height = 320,
}: {
  result: StressPredictResponse | null
  loading: boolean
  showPeak?: boolean
  height?: number
}) {
  const [phase, setPhase] = useState(0)
  const lastUpdate = useRef(0)

  useEffect(() => {
    if (!result || !showPeak) return
    let raf = 0
    const start = performance.now()
    function tick(now: number) {
      if (now - lastUpdate.current > 80) {
        setPhase(((now - start) / 1400) % 1)
        lastUpdate.current = now
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [result, showPeak])

  if (loading || !result) {
    return <Skeleton className="w-full" style={{ height }} />
  }

  const [x0, x1, y0, y1] = result.field.extent
  const nx = result.field.vm[0].length
  const ny = result.field.vm.length
  const xs = Array.from({ length: nx }, (_, i) => x0 + ((i + 0.5) / nx) * (x1 - x0))
  const ys = Array.from({ length: ny }, (_, j) => y0 + ((j + 0.5) / ny) * (y1 - y0))

  const haloSize = 14 + 10 * Math.sin(phase * 2 * Math.PI)
  const haloOpacity = 0.45 * (1 - Math.abs(Math.sin(phase * Math.PI)))
  // The rasterised field is sampled off the mesh, so its brightest cell sits *below*
  // the true nodal peak the headline reports. Left alone, the colour key would top out
  // at a number lower than the stat tile right above it and read as a contradiction —
  // so the ramp is stretched to the reported peak. The raster simply never reaches the
  // top of the scale, which is the honest picture: that stress lives between samples.
  const [vmMin, vmRasterMax] = fieldRange(result.field.vm)
  const vmMax = Math.max(vmRasterMax, result.peak_vm_surrogate)

  return (
    <div className="flex flex-col">
      <Plot
      data={[
        {
          x: xs,
          y: ys,
          z: result.field.vm,
          type: 'heatmap',
          // von Mises stress is an unsigned magnitude, so this is a sequential,
          // perceptually-uniform ramp — the peak has to be findable by eye.
          colorscale: MAGMA,
          zmin: vmMin,
          zmax: vmMax,
          zsmooth: 'best',
          showscale: false,
          hovertemplate: 'σ_vM = %{z:.2f}<extra></extra>',
        },
        ...(showPeak
          ? [
              {
                x: [result.peak_location[0]],
                y: [result.peak_location[1]],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { symbol: 'circle', size: haloSize, color: SERIES[0], opacity: haloOpacity, line: { width: 0 } },
                hoverinfo: 'skip' as const,
                showlegend: false,
              },
              {
                x: [result.peak_location[0]],
                y: [result.peak_location[1]],
                type: 'scatter' as const,
                mode: 'markers' as const,
                // The 2px ring in the surface colour is what keeps the marker legible
                // wherever the magma ramp happens to be bright underneath it.
                marker: { symbol: 'circle', size: 9, color: SERIES[0], line: { color: '#ffffff', width: 2 } },
                hovertemplate: `pic σ_vM = ${result.peak_vm_surrogate.toFixed(2)}<extra></extra>`,
                showlegend: false,
              },
            ]
          : []),
      ]}
      layout={baseLayout({
        margin: FIELD_MARGIN,
        height,
        xaxis: { ...FIELD_AXIS, range: [x0, x1], scaleanchor: 'y' },
        yaxis: { ...FIELD_AXIS, range: [y0, y1] },
      })}
      useResizeHandler
      config={PLOT_CONFIG}
      style={{ width: '100%' }}
      />
      <ColorScaleLegend stops={MAGMA} min={vmMin} max={vmMax} className="px-1 pt-1" />
    </div>
  )
}
