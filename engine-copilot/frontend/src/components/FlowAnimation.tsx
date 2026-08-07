import { useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import { motion } from 'framer-motion'
import { Play } from 'lucide-react'
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
    <Card className="gap-4 py-5">
      <CardHeader className="px-5">
        <CardTitle className="cq-title">Flow</CardTitle>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
            <TabsList className="h-8">
              <TabsTrigger value="streamlines" className="text-body">
                Streamlines
              </TabsTrigger>
              <TabsTrigger value="lbm" className="text-body">
                LBM solver
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>

      <CardContent className="px-5">
        {mode === 'streamlines' ? (
          <div className="flex flex-col gap-2.5">
            <div
              className="relative w-full overflow-hidden rounded-[12px] border border-line bg-paper"
              style={{ aspectRatio: '2 / 1' }}
            >
              {grid ? (
                <>
                  <svg viewBox={`0 0 ${grid.width} ${grid.height}`} className="absolute inset-0 h-full w-full">
                    {/* Cut paper under a hairline of ink, like every other body in
                        this app — the traces flow around it, they don't tint it. */}
                    <path d={grid.path} fill="var(--paper)" stroke="var(--ink)" strokeWidth={0.3} />
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
            <p className="cq-muted text-band leading-[1.7]">
              Particles advected (RK4) through the field the surrogate predicted — a snapshot, with
              no time integration. The wake structure is real, the playback speed is not.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3.5">
            <div className="flex flex-wrap items-center gap-4">
              {/* Outline, not filled: the screen's one orange flat is the optimiser's
                  button, further down. */}
              <Button variant="outline" size="sm" onClick={runSimulation} disabled={running}>
                <Play />
                {running ? 'Simulating…' : 'Run the solver'}
              </Button>
              {frame && (
                <div className="min-w-[160px] flex-1">
                  <div className="cq-k mb-1.5">
                    LBM ITERATIONS · {frame.t_step} / {TOTAL_STEPS}
                  </div>
                  <div className="cq-progress">
                    <div style={{ width: `${progress * 100}%` }} />
                  </div>
                </div>
              )}
              {cdFinal !== null && (
                <div className="ml-auto text-right">
                  <div className="cq-k">CD · LBM, SHORT RUN</div>
                  <div className="mt-1 cq-num text-[16px] text-or">
                    <AnimatedNumber value={cdFinal} decimals={3} />
                  </div>
                </div>
              )}
            </div>

            {error && (
              <div className="cq-error">
                <div className="text-body font-medium">The solver stopped.</div>
                <div className="mt-1 cq-num text-band cq-muted">{error}</div>
              </div>
            )}

            <div className="overflow-hidden rounded-[12px] border border-line bg-white">
              {frame ? (
                <motion.div key="plot" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <Plot
                    data={[
                      {
                        z: frame.vorticity,
                        type: 'heatmap',
                        // Vorticity is signed — the two counter-rotating shear layers
                        // ARE the story — so: the diverging blue↔red scale, one sign
                        // of rotation per pole, neutral pinned at zero.
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
                    className="px-4 pb-3"
                  />
                </motion.div>
              ) : (
                <Skeleton className="w-full rounded-none" style={{ aspectRatio: '2 / 1' }} />
              )}
            </div>
            <p className="cq-muted text-band leading-[1.7]">
              Real Lattice-Boltzmann solver iterations, streamed frame by frame. This is the
              reference the surrogate approximates — and the reason it exists.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
