import { useState } from 'react'
import Plot from 'react-plotly.js'
import { motion } from 'framer-motion'
import { Download, Sparkles } from 'lucide-react'
import { AnimatedNumber } from './AnimatedNumber'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { INK, MARK, PLOT_CONFIG, axisX, axisY, baseLayout } from '@/lib/chartTheme'
import { settle } from '@/lib/motion'
import { cn } from '@/lib/utils'
import {
  downloadMarkdown,
  generateReport,
  streamStressOptimize,
  type StressOptimizeIteration,
} from '@/api'

const N_ITERS = 30

export function StressOptimizePanel({
  holes,
  loadAngleDeg,
  model,
  onApply,
}: {
  holes: number[][]
  loadAngleDeg: number
  model: string
  onApply: (holes: number[][]) => void
}) {
  const [running, setRunning] = useState(false)
  const [history, setHistory] = useState<{ it: number; peak: number }[]>([])
  const [finalHoles, setFinalHoles] = useState<number[][] | null>(null)
  const [initialPeak, setInitialPeak] = useState<number | null>(null)
  const [finalPeak, setFinalPeak] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)

  async function run() {
    if (running) return
    setRunning(true)
    setError(null)
    setHistory([])
    setFinalHoles(null)
    setFinalPeak(null)
    try {
      await streamStressOptimize(
        { holes, load_angle_deg: loadAngleDeg, model, n_iters: N_ITERS, seed: 0 },
        (it: StressOptimizeIteration) => {
          setHistory((h) => [...h, { it: it.it, peak: it.peak_vm_current }])
          if (it.it === 0) setInitialPeak(it.peak_vm_current)
        },
        (d) => {
          setFinalHoles(d.final_holes)
          setFinalPeak(d.peak_vm_final)
        },
      )
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  async function runExport() {
    if (!finalHoles || exporting) return
    setExporting(true)
    setError(null)
    try {
      const markdown = await generateReport({
        project: 'stress',
        context: {
          peak_vm_initial: initialPeak,
          peak_vm_final: finalPeak,
          final_holes: finalHoles,
          load_angle_deg: loadAngleDeg,
          model,
          n_iters: N_ITERS,
        },
      })
      downloadMarkdown(`engine-copilot-stress-${finalHoles.length}holes.md`, markdown)
    } catch (e) {
      setError(String(e))
    } finally {
      setExporting(false)
    }
  }

  const improvedPct = initialPeak && finalPeak ? (100 * (initialPeak - finalPeak)) / initialPeak : null
  const last = history[history.length - 1]
  const progress = history.length ? Math.min(1, (last.it + 1) / N_ITERS) : 0

  return (
    <Card className="gap-5 py-5">
      <CardHeader className="px-5">
        <CardTitle className="cq-title">Hole placement</CardTitle>
        <CardDescription className="cq-muted text-body">
          Randomised local search over the GNN surrogate — the mesh topology changes on every
          trial, so there is no usable gradient here. Removed area stays constant.
        </CardDescription>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          {/* The single orange flat of this screen. */}
          <Button onClick={run} disabled={running}>
            <Sparkles />
            {running ? 'Searching…' : 'Optimise'}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-4 px-5">
        {error && (
          <div className="cq-error">
            <div className="text-body font-medium">The search did not complete.</div>
            <div className="mt-1 cq-num text-band cq-muted">{error}</div>
          </div>
        )}

        {/* No spinner: a rule filling along the measure ramp, with the real count. */}
        {running && (
          <div>
            <div className="cq-k mb-2">
              SEARCHING · {history.length} / {N_ITERS} TRIALS
            </div>
            <div className="cq-progress">
              <div style={{ width: `${progress * 100}%` }} />
            </div>
          </div>
        )}

        {history.length > 0 ? (
          <Plot
            data={[
              {
                x: history.map((h) => h.it),
                y: history.map((h) => h.peak),
                type: 'scatter',
                mode: 'lines',
                line: { color: MARK.measure, width: 2 },
                hovertemplate: 'it %{x} · peak %{y:.2f}<extra></extra>',
              },
              {
                x: [last.it],
                y: [last.peak],
                type: 'scatter',
                mode: 'markers',
                // The focal mark, and the only black-filled thing in the plot.
                marker: { size: 7, color: MARK.focus, line: { color: INK.surface, width: 2 } },
                hoverinfo: 'skip',
              },
            ]}
            layout={baseLayout({
              height: 220,
              margin: { l: 56, r: 16, t: 14, b: 38 },
              xaxis: axisX('TRIAL'),
              yaxis: axisY('PEAK σ_vM', { tickformat: '.2f' }),
              hovermode: 'x unified',
              // One series: the card title already names it, so a legend box with a
              // single swatch would only restate the heading.
              showlegend: false,
            })}
            useResizeHandler
            config={PLOT_CONFIG}
            style={{ width: '100%' }}
          />
        ) : (
          // An invitation, not an excuse — and exactly where the background masses
          // carry the panel.
          <div className="flex h-[220px] flex-col items-center justify-center px-6 text-center">
            <div className="cq-k mb-3">NO SEARCH RUN YET</div>
            <div className="cq-value max-w-[18ch] text-[22px]">Move the holes, lower the peak</div>
            <p className="mt-3 max-w-[40ch] cq-muted text-body leading-[1.7]">
              {N_ITERS} trials at constant removed area — the stress peak comes down trial after
              trial.
            </p>
          </div>
        )}

        {finalPeak !== null && initialPeak !== null && (
          <>
            <Separator />
            <motion.div className="flex flex-wrap items-end gap-x-8 gap-y-3" {...settle()}>
              <div>
                <div className="cq-k">PEAK σ_vM AFTER SEARCH</div>
                <div className="mt-1.5 flex items-baseline gap-3">
                  <span className="cq-value text-[26px]">
                    <AnimatedNumber value={finalPeak} decimals={2} />
                  </span>
                  {/* Grey is a state, never a series: the superseded value is gone. */}
                  <span className="cq-num text-band text-gone line-through">
                    {initialPeak.toFixed(2)}
                  </span>
                </div>
              </div>

              {improvedPct !== null && (
                <div>
                  <div className="cq-k">CHANGE</div>
                  <div
                    className={cn(
                      'cq-num mt-2 text-[16px]',
                      improvedPct > 0 ? 'text-or' : 'text-burnt',
                    )}
                  >
                    {improvedPct > 0 ? '−' : '+'}
                    {Math.abs(improvedPct).toFixed(1)} %
                  </div>
                </div>
              )}

              {finalHoles && (
                <div className="ml-auto flex flex-wrap items-center gap-3">
                  <Button variant="outline" size="sm" onClick={runExport} disabled={exporting}>
                    <Download />
                    {exporting ? 'Report…' : 'Export report'}
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => onApply(finalHoles)}>
                    Apply this design
                  </Button>
                </div>
              )}
            </motion.div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
