import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, LoaderCircle, Sparkles, TriangleAlert } from 'lucide-react'
import { ShapePreview } from './ShapeEditor'
import { ConvergencePlot, type ConvergencePoint } from './ConvergencePlot'
import { AnimatedNumber } from './AnimatedNumber'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { streamFlowOptimize, verifyFlow, type FlowOptimizeIteration } from '@/api'

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
  const [verifyResult, setVerifyResult] = useState<{ cd_lbm: number; cd_surrogate: number; gap_pct: number } | null>(
    null,
  )

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
        { reynolds, n_starts: 16, iters: 250, model, seed: 0 },
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
      setVerifyResult({ cd_lbm: res.cd_lbm, cd_surrogate: res.cd_surrogate, gap_pct: res.gap_pct })
    } catch (e) {
      setError(String(e))
    } finally {
      setVerifying(false)
    }
  }

  const started = history.length > 0 || finalCd !== null

  return (
    <Card>
      <CardHeader>
        <CardTitle>Optimisation de forme</CardTitle>
        <CardDescription>
          Descente Adam à travers le surrogate gelé, 16 démarrages, aire constante — Re&nbsp;=&nbsp;
          {reynolds.toFixed(0)}.
        </CardDescription>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          <Button onClick={runOptimize} disabled={running}>
            {running ? <LoaderCircle className="animate-spin" /> : <Sparkles />}
            {running ? 'Optimisation…' : 'Optimiser'}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-5">
        {error && (
          <div className="flex items-start gap-1.5 text-xs text-warning">
            <TriangleAlert className="mt-px size-3 shrink-0" />
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="flex items-center gap-3">
            <figure className="flex-1">
              <ShapePreview coeffs={initialCoeffs ?? coeffs} targetArea={targetArea} />
              <figcaption className="mt-1.5 text-center text-[11px] text-muted-foreground">Départ</figcaption>
            </figure>
            <ArrowRight className="size-4 shrink-0 text-muted-foreground" />
            <figure className="flex-1">
              <ShapePreview
                coeffs={bestCoeffs ?? coeffs}
                targetArea={targetArea}
                className={cn(running && 'ring-1 ring-ring/40')}
              />
              <figcaption className="mt-1.5 text-center text-[11px] text-muted-foreground">Optimisé</figcaption>
            </figure>
          </div>

          <div className="min-w-0">
            {started ? (
              <ConvergencePlot history={history} />
            ) : (
              <div className="flex h-[300px] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border">
                <Sparkles className="size-5 text-muted-foreground" />
                <p className="max-w-56 text-center text-xs text-muted-foreground">
                  Lance l’optimisation pour voir la convergence du Cd, itération par itération.
                </p>
              </div>
            )}
          </div>
        </div>

        {finalCd !== null && (
          <>
            <Separator />
            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
              <div>
                <div className="text-xs text-muted-foreground">Cd optimisé (surrogate)</div>
                <div className="text-2xl font-semibold tracking-tight">
                  <AnimatedNumber value={finalCd} decimals={3} />
                </div>
              </div>

              <Separator orientation="vertical" className="hidden h-10 sm:block" />

              <div className="flex flex-1 flex-wrap items-center gap-3">
                <Button variant="outline" size="sm" onClick={runVerify} disabled={running || verifying}>
                  {verifying ? <LoaderCircle className="animate-spin" /> : null}
                  {verifying ? 'Solveur LBM…' : 'Vérifier au solveur'}
                </Button>

                {verifyResult ? (
                  <motion.div
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs"
                  >
                    <span className="text-muted-foreground">
                      surrogate <span className="tnum font-mono text-foreground">{verifyResult.cd_surrogate.toFixed(3)}</span>
                    </span>
                    <span className="text-muted-foreground">
                      LBM <span className="tnum font-mono text-foreground">{verifyResult.cd_lbm.toFixed(3)}</span>
                    </span>
                    <Badge
                      variant="outline"
                      className={cn(
                        'font-normal',
                        verifyResult.gap_pct < 5
                          ? 'border-good/40 bg-good/10 text-good'
                          : 'border-warning/40 bg-warning/10 text-warning',
                      )}
                    >
                      écart {verifyResult.gap_pct.toFixed(1)}%
                    </Badge>
                  </motion.div>
                ) : (
                  <span className="text-[11px] text-muted-foreground">
                    Rejoue le design optimisé dans le vrai solveur — la seule mesure honnête.
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
                    Appliquer ce design
                  </Button>
                )}
              </div>
            </div>
          </>
        )}

        {running && history.length === 0 && <Skeleton className="h-4 w-40" />}
      </CardContent>
    </Card>
  )
}
