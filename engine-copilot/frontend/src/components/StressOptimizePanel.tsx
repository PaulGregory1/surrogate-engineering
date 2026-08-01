import { useState } from 'react'
import Plot from 'react-plotly.js'
import { motion } from 'framer-motion'
import { LoaderCircle, Sparkles, TriangleAlert } from 'lucide-react'
import { AnimatedNumber } from './AnimatedNumber'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { INK, PLOT_CONFIG, SERIES, axis, baseLayout } from '@/lib/chartTheme'
import { cn } from '@/lib/utils'
import { streamStressOptimize, type StressOptimizeIteration } from '@/api'

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

  async function run() {
    if (running) return
    setRunning(true)
    setError(null)
    setHistory([])
    setFinalHoles(null)
    setFinalPeak(null)
    try {
      await streamStressOptimize(
        { holes, load_angle_deg: loadAngleDeg, model, n_iters: 30, seed: 0 },
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

  const improvedPct = initialPeak && finalPeak ? (100 * (initialPeak - finalPeak)) / initialPeak : null
  const last = history[history.length - 1]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Placement des trous</CardTitle>
        <CardDescription>
          Recherche aléatoire locale sur le surrogate GNN — la topologie du maillage change à chaque
          essai, donc pas de gradient exploitable ici. Aire retirée constante.
        </CardDescription>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          <Button onClick={run} disabled={running}>
            {running ? <LoaderCircle className="animate-spin" /> : <Sparkles />}
            {running ? 'Recherche…' : 'Optimiser'}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {error && (
          <div className="flex items-start gap-1.5 text-xs text-warning">
            <TriangleAlert className="mt-px size-3 shrink-0" />
            {error}
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
                line: { color: SERIES[0], width: 2 },
                hovertemplate: 'it %{x} · pic %{y:.2f}<extra></extra>',
              },
              {
                x: [last.it],
                y: [last.peak],
                type: 'scatter',
                mode: 'markers',
                marker: { size: 8, color: SERIES[0], line: { color: INK.surface, width: 2 } },
                hoverinfo: 'skip',
              },
            ]}
            layout={baseLayout({
              height: 220,
              margin: { l: 52, r: 16, t: 12, b: 36 },
              xaxis: axis('itération'),
              yaxis: axis('pic σ_vM', { tickformat: '.2f' }),
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
          <div className="flex h-[220px] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border">
            <Sparkles className="size-5 text-muted-foreground" />
            <p className="max-w-64 text-center text-xs text-muted-foreground">
              Lance la recherche pour voir le pic de contrainte descendre au fil des essais.
            </p>
          </div>
        )}

        {finalPeak !== null && initialPeak !== null && (
          <>
            <Separator />
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-wrap items-center gap-x-4 gap-y-2"
            >
              <div className="text-xs text-muted-foreground">pic σ_vM</div>
              <div className="tnum font-mono text-sm text-muted-foreground line-through">
                {initialPeak.toFixed(2)}
              </div>
              <div className="text-xl font-semibold tracking-tight">
                <AnimatedNumber value={finalPeak} decimals={2} />
              </div>
              {improvedPct !== null && (
                <Badge
                  variant="outline"
                  className={cn(
                    'font-normal',
                    improvedPct > 0
                      ? 'border-good/40 bg-good/10 text-good'
                      : 'border-warning/40 bg-warning/10 text-warning',
                  )}
                >
                  {improvedPct > 0 ? '−' : '+'}
                  {Math.abs(improvedPct).toFixed(1)}%
                </Badge>
              )}
              {finalHoles && (
                <Button variant="secondary" size="sm" className="ml-auto" onClick={() => onApply(finalHoles)}>
                  Appliquer ce design
                </Button>
              )}
            </motion.div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
