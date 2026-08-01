import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { TriangleAlert } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { HoleEditor } from '@/components/HoleEditor'
import { StressViewer } from '@/components/StressViewer'
import { StressOptimizePanel } from '@/components/StressOptimizePanel'
import { MetricsBadge } from '@/components/MetricsBadge'
import { StatTile } from '@/components/StatTile'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { usePublishChatContext } from '@/chat/ChatProvider'
import { predictStress, type StressPredictResponse } from '@/api'

const DEFAULT_HOLES = [
  [0.5, 0.5, 0.12],
  [1.3, 0.4, 0.1],
]

const MODELS = [
  { value: 'gnn-mp10-glob', label: 'GNN + global', hint: 'défaut' },
  { value: 'gnn-mp10-glob-corr', label: 'GNN + global + coarse', hint: 'correction FEM' },
  { value: 'gnn-mp10', label: 'GNN', hint: '10 passes' },
] as const

export function StressLab() {
  const [holes, setHoles] = useState<number[][]>(DEFAULT_HOLES)
  const [loadAngleDeg, setLoadAngleDeg] = useState(-90)
  const [model, setModel] = useState<'gnn-mp10-glob' | 'gnn-mp10-glob-corr' | 'gnn-mp10'>('gnn-mp10-glob')
  const [result, setResult] = useState<StressPredictResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [compare, setCompare] = useState(false)
  const debounceRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    setLoading(true)
    setError(null)
    window.clearTimeout(debounceRef.current)
    debounceRef.current = window.setTimeout(() => {
      predictStress({ holes, load_angle_deg: loadAngleDeg, model })
        .then((r) => {
          setResult(r)
          setLoading(false)
        })
        .catch((e) => {
          setError(String(e))
          setLoading(false)
        })
    }, 300)
    return () => window.clearTimeout(debounceRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [holes, loadAngleDeg, model])

  // Toggling `compare` changes the grid from one column to two, which resizes
  // both StressViewer plots without a real window resize firing — react-plotly's
  // `useResizeHandler` only listens for that event, so nudge one synthetically
  // once the new grid has actually been laid out (double rAF: one to let React
  // commit the DOM change, one to let the browser paint the new layout).
  useLayoutEffect(() => {
    let raf1 = 0
    let raf2 = 0
    raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    })
    return () => {
      cancelAnimationFrame(raf1)
      cancelAnimationFrame(raf2)
    }
  }, [compare])

  usePublishChatContext({
    label: 'Stress Lab',
    lines: [
      `trous (x, y, r) = ${holes.map((h) => `(${h.map((v) => v.toFixed(2)).join(', ')})`).join(' ')}`,
      `angle de charge = ${loadAngleDeg.toFixed(0)}°`,
      `modèle = ${model}`,
      result ? `pic σ_vM prédit = ${result.peak_vm_surrogate.toFixed(3)}` : 'pic σ_vM prédit = (en cours)',
    ],
    suggestions: [
      'Où se situe la concentration de contrainte sur ce design ?',
      'Quelle erreur dois-je attendre sur ce pic de contrainte ?',
      'Pourquoi le FEM grossier sous-estime-t-il le pic ?',
    ],
  })

  return (
    <>
      <PageHeader
        title="Stress Lab"
        subtitle="Plaque perforée sous charge — surrogate GNN mesh-native"
        actions={
          <Select value={model} onValueChange={(v) => setModel(v as typeof model)}>
            <SelectTrigger size="sm" className="w-[275px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MODELS.map((m) => (
                <SelectItem key={m.value} value={m.value}>
                  <span>{m.label}</span>
                  <span className="text-muted-foreground">· {m.hint}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      />

      <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-4 px-6 py-6">
        <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
          <HoleEditor
            holes={holes}
            onChange={setHoles}
            loadAngleDeg={loadAngleDeg}
            onLoadAngleChange={setLoadAngleDeg}
          />

          <div className="flex min-w-0 flex-col gap-4">
            <Card className="py-4">
              <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-4 px-4">
                <StatTile
                  label="Pic de contrainte de von Mises"
                  value={result?.peak_vm_surrogate ?? null}
                  decimals={2}
                  loading={loading && !result}
                  hint="σ_vM max sur la plaque"
                />
                <Separator orientation="vertical" className="hidden h-12 sm:block" />
                <AnimatePresence mode="wait">
                  {result && (
                    <motion.div
                      key={result.model}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                    >
                      <MetricsBadge label={result.model} errorPct={result.peak_error_pct_expected} />
                    </motion.div>
                  )}
                </AnimatePresence>
                {result && (
                  <Button variant="outline" size="sm" onClick={() => setCompare((c) => !c)}>
                    {compare ? 'Masquer le FEM grossier' : 'Comparer au FEM grossier'}
                  </Button>
                )}
                {result && (
                  <div className="ml-auto text-right">
                    <div className="text-xs text-muted-foreground">Latence</div>
                    <div className="tnum font-mono text-sm">{result.latency_ms.toFixed(1)} ms</div>
                  </div>
                )}
              </CardContent>
            </Card>

            {error && (
              <Card className="border-critical/40 py-3">
                <CardContent className="flex items-start gap-2 px-4 text-sm text-critical">
                  <TriangleAlert className="mt-0.5 size-4 shrink-0" />
                  {error}
                </CardContent>
              </Card>
            )}

            <div className={`grid grid-cols-1 gap-3 ${compare ? 'md:grid-cols-2' : ''}`}>
              <Card className="gap-2 py-3">
                <CardContent className="px-3">
                  <div className="mb-1 text-xs font-medium">Surrogate GNN</div>
                  <StressViewer result={result} loading={loading && !result} />
                </CardContent>
              </Card>

              {compare && result && (
                <Card className="gap-2 py-3">
                  <CardContent className="px-3">
                    <div className="mb-1 text-xs font-medium">
                      FEM grossier{' '}
                      <span className="font-normal text-muted-foreground">
                        pic {result.peak_vm_coarse_fem.toFixed(2)}
                      </span>
                    </div>
                    <StressViewer
                      result={{
                        ...result,
                        field: { vm: result.vm_coarse_fem, extent: result.field.extent },
                        peak_vm_surrogate: result.peak_vm_coarse_fem,
                      }}
                      loading={false}
                      showPeak={false}
                    />
                    <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                      La baseline que le GNN doit battre : elle sous-estime typiquement le pic de
                      24 à 36 %, parce qu’un maillage grossier ne résout pas la concentration de
                      contrainte au bord du trou.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>

        <StressOptimizePanel holes={holes} loadAngleDeg={loadAngleDeg} model={model} onApply={setHoles} />
      </div>
    </>
  )
}
