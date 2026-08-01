import { useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import { motion } from 'framer-motion'
import { LoaderCircle, Play, TriangleAlert } from 'lucide-react'
import { StreamlineOverlay } from './StreamlineOverlay'
import { AnimatedNumber } from './AnimatedNumber'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ColorScaleLegend } from '@/components/ColorScaleLegend'
import { DIVERGING, FIELD_AXIS, FIELD_MARGIN, PLOT_CONFIG, baseLayout, symmetricRange } from '@/lib/chartTheme'
import { boundaryToGrid } from '@/geometry'
import { streamFlowSimulate, type FlowPredictResponse, type FlowSimulateFrame } from '@/api'

const TOTAL_STEPS = 6000

type Mode = 'streamlines' | 'lbm'

export function FlowAnimation({
  coeffs,
  reynolds,
  result,
}: {
  coeffs: number[]
  reynolds: number
  result: FlowPredictResponse | null
}) {
  const [mode, setMode] = useState<Mode>('streamlines')
  const [running, setRunning] = useState(false)
  const [frame, setFrame] = useState<FlowSimulateFrame | null>(null)
  const [cdFinal, setCdFinal] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const grid = useMemo(() => {
    if (!result) return null
    const { x, y } = boundaryToGrid(result.boundary_xy, result.domain)
    const path = x.map((xi, i) => `${i === 0 ? 'M' : 'L'} ${xi.toFixed(2)} ${y[i].toFixed(2)}`).join(' ') + ' Z'
    return { width: result.fields.u[0].length, height: result.fields.u.length, path }
  }, [result])

  async function runSimulation() {
    if (running) return
    setRunning(true)
    setError(null)
    setFrame(null)
    setCdFinal(null)
    try {
      await streamFlowSimulate(
        { coeffs, reynolds, n_frames: 60, steps: TOTAL_STEPS },
        (f) => setFrame(f),
        (d) => setCdFinal(d.cd_final),
      )
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  const progress = frame ? frame.t_step / TOTAL_STEPS : 0
  // Recomputed per streamed frame: the wake grows over the run, so a range frozen at
  // the first frame would clip the vortices that appear later.
  const vorticityRange = useMemo(
    () => (frame ? symmetricRange(frame.vorticity) : ([-1, 1] as [number, number])),
    [frame],
  )

  return (
    <Card className="gap-4 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm">Écoulement</CardTitle>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
            <TabsList className="h-8">
              <TabsTrigger value="streamlines" className="text-xs">
                Lignes de courant
              </TabsTrigger>
              <TabsTrigger value="lbm" className="text-xs">
                Solveur LBM
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>

      <CardContent className="px-4">
        {mode === 'streamlines' ? (
          <div className="flex flex-col gap-2">
            <div
              className="relative w-full overflow-hidden rounded-lg border border-border bg-background"
              style={{ aspectRatio: '2 / 1' }}
            >
              {grid ? (
                <>
                  <svg viewBox={`0 0 ${grid.width} ${grid.height}`} className="absolute inset-0 h-full w-full">
                    <path
                      d={grid.path}
                      fill="color-mix(in srgb, var(--foreground) 10%, transparent)"
                      stroke="var(--muted-foreground)"
                      strokeWidth={0.3}
                    />
                  </svg>
                  <StreamlineOverlay
                    u={result!.fields.u}
                    v={result!.fields.v}
                    width={grid.width}
                    height={grid.height}
                  />
                </>
              ) : (
                <Skeleton className="absolute inset-0" />
              )}
            </div>
            <p className="text-[11px] text-muted-foreground">
              Particules advectées (RK4) dans le champ prédit par le surrogate — instantané, aucune
              intégration en temps. La structure du sillage est réelle, la vitesse d’affichage ne
              l’est pas.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <Button size="sm" onClick={runSimulation} disabled={running}>
                {running ? <LoaderCircle className="animate-spin" /> : <Play />}
                {running ? 'Simulation…' : 'Lancer le solveur'}
              </Button>
              {frame && (
                <div className="flex items-center gap-2">
                  <div className="h-1 w-28 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary transition-[width] duration-200"
                      style={{ width: `${progress * 100}%` }}
                    />
                  </div>
                  <span className="tnum text-[11px] text-muted-foreground">
                    {frame.t_step} / {TOTAL_STEPS}
                  </span>
                </div>
              )}
              {cdFinal !== null && (
                <div className="ml-auto text-xs">
                  <span className="text-muted-foreground">Cd (LBM, run court) </span>
                  <span className="tnum font-mono font-medium text-good">
                    <AnimatedNumber value={cdFinal} decimals={3} />
                  </span>
                </div>
              )}
            </div>

            {error && (
              <div className="flex items-start gap-1.5 text-xs text-warning">
                <TriangleAlert className="mt-px size-3 shrink-0" />
                {error}
              </div>
            )}

            <div className="overflow-hidden rounded-lg border border-border bg-background">
              {frame ? (
                <motion.div key="plot" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <Plot
                    data={[
                      {
                        z: frame.vorticity,
                        type: 'heatmap',
                        // Vorticity is signed — the two counter-rotating shear layers
                        // ARE the story — so: diverging, pinned neutral at zero.
                        colorscale: DIVERGING,
                        zmid: 0,
                        zsmooth: 'best',
                        showscale: false,
                        hovertemplate: 'ω = %{z:.3f}<extra></extra>',
                      },
                    ]}
                    layout={baseLayout({
                      margin: FIELD_MARGIN,
                      height: 280,
                      xaxis: { ...FIELD_AXIS, scaleanchor: 'y' },
                      yaxis: { ...FIELD_AXIS, autorange: 'reversed' },
                    })}
                    useResizeHandler
                    config={PLOT_CONFIG}
                    style={{ width: '100%' }}
                  />
                  <ColorScaleLegend
                    stops={DIVERGING}
                    min={vorticityRange[0]}
                    max={vorticityRange[1]}
                    diverging
                    className="px-3 pb-2"
                  />
                </motion.div>
              ) : (
                <Skeleton className="w-full" style={{ aspectRatio: '2 / 1' }} />
              )}
            </div>
            <p className="text-[11px] text-muted-foreground">
              Vraies itérations du solveur Lattice-Boltzmann, diffusées image par image. C’est la
              référence que le surrogate approxime — et la raison pour laquelle il existe.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
