import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Download, Sparkles } from 'lucide-react'
import { ShapePreview } from './ShapeEditor'
import { ConvergencePlot, type ConvergencePoint } from './ConvergencePlot'
import { AnimatedNumber } from './AnimatedNumber'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { settle } from '@/lib/motion'
import { cn } from '@/lib/utils'
import {
  downloadMarkdown,
  generateReport,
  streamFlowOptimize,
  verifyFlow,
  type FlowOptimizeIteration,
} from '@/api'

const ITERS = 250
// One start, seeded with the shape on screen. The Cd landscape at fixed area is
// effectively unimodal across the whole Re slider — starts as different as blunt,
// slender, wavy and flat land on the same optimum to ~1e-4 in Cd — so the 15 extra
// random starts bought nothing, took 16x longer, and left START showing a shape the
// result had never descended from. The endpoint still accepts n_starts > 1.
const N_STARTS = 1

export function OptimizerPanel({
  coeffs,
  reynolds,
  model,
  targetArea,
  onApply,
}: {
  coeffs: number[]
  reynolds: number
  model: string
  targetArea: number
  onApply: (coeffs: number[]) => void
}) {
  const [running, setRunning] = useState(false)
  const [history, setHistory] = useState<ConvergencePoint[]>([])
  const [initialCoeffs, setInitialCoeffs] = useState<number[] | null>(null)
  const [bestCoeffs, setBestCoeffs] = useState<number[] | null>(null)
  const [finalCd, setFinalCd] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [verifying, setVerifying] = useState(false)
  const [verifyResult, setVerifyResult] = useState<{
    cd_lbm: number
    cd_surrogate: number
    gap_pct: number
    wall_time_s: number
  } | null>(null)
  const [exporting, setExporting] = useState(false)

  async function runOptimize() {
    if (running) return
    setRunning(true)
    setError(null)
    setHistory([])
    setInitialCoeffs(coeffs)
    setBestCoeffs(coeffs)
    setFinalCd(null)
    setVerifyResult(null)
    try {
      await streamFlowOptimize(
        { coeffs, reynolds, n_starts: N_STARTS, iters: ITERS, model, seed: 0 },
        (it: FlowOptimizeIteration) => {
          setHistory((h) => [...h, { it: it.it, cd_best: it.cd_best, cd_mean: it.cd_mean }])
          setBestCoeffs(it.coeffs_best)
        },
        (d) => {
          setBestCoeffs(d.final_coeffs)
          setFinalCd(d.cd_surrogate_final)
        },
      )
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  async function runVerify() {
    if (!bestCoeffs || verifying) return
    setVerifying(true)
    setError(null)
    try {
      const res = await verifyFlow({ coeffs: bestCoeffs, reynolds, target_area: targetArea })
      setVerifyResult({
        cd_lbm: res.cd_lbm,
        cd_surrogate: res.cd_surrogate,
        gap_pct: res.gap_pct,
        wall_time_s: res.wall_time_s,
      })
    } catch (e) {
      setError(String(e))
    } finally {
      setVerifying(false)
    }
  }

  async function runExport() {
    if (!bestCoeffs || exporting) return
    setExporting(true)
    setError(null)
    try {
      const markdown = await generateReport({
        project: 'flow',
        context: {
          cd_surrogate_final: finalCd,
          history: { cd_best: history.map((h) => h.cd_best) },
          reynolds,
          final_coeffs: bestCoeffs,
          target_area: targetArea,
          model,
          // Only sent once the LBM has actually been run. Without it the report says
          // in as many words that nothing confirmed the surrogate — which is the
          // honest reading of an un-verified optimum.
          ...(verifyResult ? { verify: verifyResult } : {}),
        },
      })
      downloadMarkdown(`engine-copilot-flow-Re${reynolds.toFixed(0)}.md`, markdown)
    } catch (e) {
      setError(String(e))
    } finally {
      setExporting(false)
    }
  }

  const started = history.length > 0 || finalCd !== null
  const lastIt = history.length ? history[history.length - 1].it + 1 : 0
  const progress = Math.min(1, lastIt / ITERS)

  return (
    <Card className="gap-5 py-5">
      <CardHeader className="px-5">
        <CardTitle className="cq-title">Shape optimisation</CardTitle>
        <CardDescription className="cq-muted text-body">
          Adam descent through the frozen surrogate, from the shape on screen, at constant area —
          Re&nbsp;=&nbsp;{reynolds.toFixed(0)}.
        </CardDescription>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          {/* The single orange flat of this screen. It is also the whole "one bold
              move" budget: nothing else here is filled. */}
          <Button onClick={runOptimize} disabled={running}>
            <Sparkles />
            {running ? 'Optimising…' : 'Optimise'}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-5 px-5">
        {error && (
          <div className="cq-error">
            <div className="text-body font-medium">The optimisation did not complete.</div>
            <div className="mt-1 cq-num text-band cq-muted">{error}</div>
          </div>
        )}

        {/* No spinner: a rule filling along the measure ramp, with the real cost
            printed beside it. The product sells speed — show it. */}
        {running && (
          <div>
            <div className="cq-k mb-2">
              DESCENT · {lastIt} / {ITERS} ITERATIONS · CONSTANT AREA
            </div>
            <div className="cq-progress">
              <div style={{ width: `${progress * 100}%` }} />
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="flex items-center gap-3">
            <figure className="flex-1">
              <ShapePreview coeffs={initialCoeffs ?? coeffs} targetArea={targetArea} />
              <figcaption className="cq-k mt-2 text-center">START</figcaption>
            </figure>
            <ArrowRight className="size-4 shrink-0 text-ink3" />
            <figure className="flex-1">
              <ShapePreview
                coeffs={bestCoeffs ?? coeffs}
                targetArea={targetArea}
                tone={started ? 'measure' : 'ink'}
              />
              <figcaption className={cn('cq-k mt-2 text-center', started && 'text-burnt')}>OPTIMISED</figcaption>
            </figure>
          </div>

          <div className="min-w-0">
            {started ? (
              <ConvergencePlot history={history} />
            ) : (
              // An empty state is an invitation, never an excuse — and this is
              // exactly where the background masses carry the panel.
              <div className="flex h-[300px] flex-col items-center justify-center px-6 text-center">
                <div className="cq-k mb-3">NO DESCENT RUN YET</div>
                <div className="cq-value max-w-[16ch] text-[22px]">Search for the lowest-drag shape</div>
                <p className="mt-3 max-w-[38ch] cq-muted text-body leading-[1.7]">
                  {ITERS} descent iterations from the shape on screen, at constant projected
                  area.
                </p>
              </div>
            )}
          </div>
        </div>

        {finalCd !== null && (
          <>
            <Separator />
            <motion.div className="flex flex-wrap items-end gap-x-8 gap-y-4" {...settle()}>
              <div>
                <div className="cq-k">OPTIMISED CD · SURROGATE</div>
                <div className="cq-value mt-2 text-[26px]">
                  <AnimatedNumber value={finalCd} decimals={3} />
                </div>
              </div>

              <div className="flex flex-1 flex-wrap items-center gap-4">
                <Button variant="outline" size="sm" onClick={runVerify} disabled={running || verifying}>
                  {verifying ? 'LBM solver…' : 'Verify with solver'}
                </Button>

                <Button variant="outline" size="sm" onClick={runExport} disabled={running || exporting}>
                  <Download />
                  {exporting ? 'Report…' : 'Export report'}
                </Button>

                {verifyResult ? (
                  <motion.div className="flex flex-wrap items-center gap-x-5 gap-y-1" {...settle()}>
                    <span className="cq-k">
                      SURROGATE{' '}
                      <span className="cq-num text-band text-ink tracking-normal">
                        {verifyResult.cd_surrogate.toFixed(3)}
                      </span>
                    </span>
                    <span className="cq-k">
                      LBM{' '}
                      <span className="cq-num text-band text-ink tracking-normal">
                        {verifyResult.cd_lbm.toFixed(3)}
                      </span>
                    </span>
                    {/* The gap is a measured quantity, so it gets the orange — and
                        goes burnt past 5 %, the one deviation mark the charte has. */}
                    <span
                      className={cn(
                        'cq-num text-band',
                        verifyResult.gap_pct < 5 ? 'text-or' : 'text-burnt',
                      )}
                    >
                      gap {verifyResult.gap_pct.toFixed(1)} %
                    </span>
                  </motion.div>
                ) : (
                  // No italic here: this screen already spends its one serif moment
                  // on the headline Cd, at the top of the lab.
                  <span className="max-w-[38ch] cq-muted text-body leading-[1.7]">
                    Re-runs the optimised design in the real solver — the only honest measurement.
                  </span>
                )}

                {bestCoeffs && (
                  <Button
                    variant="secondary"
                    size="sm"
                    className="ml-auto"
                    onClick={() => onApply(bestCoeffs)}
                    disabled={running}
                  >
                    Apply this design
                  </Button>
                )}
              </div>
            </motion.div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
