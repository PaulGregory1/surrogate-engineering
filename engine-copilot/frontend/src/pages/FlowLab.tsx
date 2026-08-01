import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { TriangleAlert } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { ShapeEditor } from '@/components/ShapeEditor'
import { FieldViewer } from '@/components/FieldViewer'
import { MetricsBadge } from '@/components/MetricsBadge'
import { StatTile } from '@/components/StatTile'
import { FlowAnimation } from '@/components/FlowAnimation'
import { OptimizerPanel } from '@/components/OptimizerPanel'
import { Card, CardContent } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { usePublishChatContext } from '@/chat/ChatProvider'
import { predictFlow, type FlowPredictResponse } from '@/api'

const DEFAULT_TARGET_AREA = 307 // refined once the first prediction comes back

const MODELS = [
  { value: 'unet-phys', label: 'U-Net + continuité', hint: 'défaut' },
  { value: 'unet', label: 'U-Net', hint: 'sans terme physique' },
  { value: 'fno', label: 'FNO', hint: 'opérateur de Fourier' },
] as const

export function FlowLab() {
  const [coeffs, setCoeffs] = useState([0.05, -0.02, 0.03, 0.01, -0.04, 0.0])
  const [reynolds, setReynolds] = useState(80)
  const [model, setModel] = useState<'unet-phys' | 'unet' | 'fno'>('unet-phys')
  const [result, setResult] = useState<FlowPredictResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const debounceRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    setLoading(true)
    setError(null)
    window.clearTimeout(debounceRef.current)
    debounceRef.current = window.setTimeout(() => {
      predictFlow({ coeffs, reynolds, model })
        .then((r) => {
          setResult(r)
          setLoading(false)
        })
        .catch((e) => {
          setError(String(e))
          setLoading(false)
        })
    }, 250)
    return () => window.clearTimeout(debounceRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coeffs, reynolds, model])

  const targetArea = result?.target_area ?? DEFAULT_TARGET_AREA

  // Hand the design on screen to the copilot, so a question can just say "cette forme".
  usePublishChatContext({
    label: 'Flow Lab',
    lines: [
      `coefficients [a2,b2,a3,b3,a4,b4] = [${coeffs.map((c) => c.toFixed(3)).join(', ')}]`,
      `Reynolds = ${reynolds.toFixed(0)}`,
      `modèle = ${model}`,
      result ? `Cd prédit = ${result.cd_surrogate.toFixed(4)}` : 'Cd prédit = (en cours)',
    ],
    suggestions: [
      'Prédis le Cd de la forme affichée et compare-la à un cercle.',
      'Ce Reynolds est-il dans le domaine d’entraînement du modèle ?',
      'Quelle erreur dois-je attendre sur ce Cd ?',
    ],
  })

  return (
    <>
      <PageHeader
        title="Flow Lab"
        subtitle="Écoulement externe autour d’un corps immergé"
        actions={
          <Select value={model} onValueChange={(v) => setModel(v as typeof model)}>
            <SelectTrigger size="sm" className="w-[255px]">
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
          <ShapeEditor
            coeffs={coeffs}
            onChange={setCoeffs}
            reynolds={reynolds}
            onReynoldsChange={setReynolds}
            targetArea={targetArea}
          />

          <div className="flex min-w-0 flex-col gap-4">
            <Card className="py-4">
              <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-4 px-4">
                <StatTile
                  label="Coefficient de traînée"
                  value={result?.cd_surrogate ?? null}
                  decimals={3}
                  loading={loading && !result}
                  hint="Cd — surface projetée constante"
                />
                <Separator orientation="vertical" className="hidden h-12 sm:block" />
                <div className="flex flex-wrap items-center gap-3">
                  <AnimatePresence mode="wait">
                    {result && (
                      <motion.div
                        key={result.model}
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                      >
                        <MetricsBadge label={result.model} errorPct={result.cd_error_pct_expected} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                  {reynolds > 150 && (
                    <span className="flex items-center gap-1.5 text-[11px] text-warning">
                      <TriangleAlert className="size-3" />
                      Re au-delà du domaine d’entraînement (20–150)
                    </span>
                  )}
                </div>
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

            <FieldViewer result={result} loading={loading && !result} />

            <FlowAnimation coeffs={coeffs} reynolds={reynolds} result={result} />
          </div>
        </div>

        <OptimizerPanel
          coeffs={coeffs}
          reynolds={reynolds}
          model={model}
          targetArea={targetArea}
          onApply={setCoeffs}
        />
      </div>
    </>
  )
}
