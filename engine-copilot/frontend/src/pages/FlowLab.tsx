import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { PageHeader } from '@/components/PageHeader'
import { ShapeEditor } from '@/components/ShapeEditor'
import { FieldViewer } from '@/components/FieldViewer'
import { StatTile } from '@/components/StatTile'
import { FlowAnimation } from '@/components/FlowAnimation'
import { OptimizerPanel } from '@/components/OptimizerPanel'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { usePublishChatContext } from '@/chat/ChatProvider'
import { settle } from '@/lib/motion'
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

  // The prediction's own error bar, in the units the reader is looking at. A
  // surrogate value shown without its variance is a substantive error, not a
  // stylistic one — so this rides directly under the figure, never in a tooltip.
  const band = result ? (result.cd_surrogate * result.cd_error_pct_expected) / 100 : null

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
                  <span className="text-ink3">· {m.hint}</span>
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
            {/* The charte's panel, at its recipe width: the one 40px figure, its
                interval, then the supporting facts as key/value rows. Everything
                the prediction is worth saying fits here, which is why there is no
                prose beside it. */}
            <div className="flex flex-wrap items-start gap-x-8 gap-y-4">
              <div className="cq-calque w-full shrink-0 px-[21px] py-[19px] sm:w-[298px]">
                <StatTile
                  label="COEFFICIENT DE TRAÎNÉE"
                  value={result?.cd_surrogate ?? null}
                  decimals={3}
                  loading={loading && !result}
                  interval={band !== null ? `± ${band.toFixed(3)} · AIRE CONSTANTE` : undefined}
                />
                {result && (
                  <motion.div className="mt-4" key={result.model} {...settle()}>
                    <div className="cq-row">
                      <span>Modèle</span>
                      <span>{result.model}</span>
                    </div>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="cq-row cursor-default">
                          <span>Erreur mesurée</span>
                          <span>± {result.cd_error_pct_expected.toFixed(2)} %</span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-64">
                        Erreur relative moyenne de <span className="font-mono">{result.model}</span> sur le
                        jeu de test, face au solveur haute-fidélité. La valeur affichée est une
                        prédiction, pas une mesure.
                      </TooltipContent>
                    </Tooltip>
                    <div className="cq-row">
                      <span>Latence</span>
                      <span>{result.latency_ms.toFixed(1)} ms</span>
                    </div>
                    {/* The interval printed as bounds, not as a ± — it is the one
                        thing a reader actually acts on, and "between 1.677 and
                        1.712" needs no arithmetic. */}
                    {band !== null && (
                      <div className="cq-row">
                        <span>Intervalle</span>
                        <span>
                          {(result.cd_surrogate - band).toFixed(3)} — {(result.cd_surrogate + band).toFixed(3)}
                        </span>
                      </div>
                    )}
                    {reynolds > 150 && (
                      <div className="cq-row">
                        <span>Enveloppe</span>
                        <span className="cq-warn">Re {reynolds.toFixed(0)} hors 20–150</span>
                      </div>
                    )}
                  </motion.div>
                )}
              </div>
            </div>

            {error && (
              <div className="cq-error">
                <div className="text-body font-medium">La prédiction n’a pas abouti.</div>
                <div className="mt-1 cq-num text-band cq-muted">{error}</div>
              </div>
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
