import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ShapeEditor } from '../components/ShapeEditor'
import { FieldViewer } from '../components/FieldViewer'
import { MetricsBadge } from '../components/MetricsBadge'
import { AnimatedNumber } from '../components/AnimatedNumber'
import { predictFlow, type FlowPredictResponse } from '../api'

const DEFAULT_TARGET_AREA = 307 // refined once the first prediction comes back

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

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold">Flow Lab</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Écoulement externe autour d&apos;un corps immergé — surrogate U-Net + continuité.
          </p>
        </div>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as typeof model)}
          className="nc-card px-2 py-1.5 text-xs"
        >
          <option value="unet-phys">U-Net + continuité (défaut)</option>
          <option value="unet">U-Net</option>
          <option value="fno">FNO</option>
        </select>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4">
        <ShapeEditor coeffs={coeffs} onChange={setCoeffs} reynolds={reynolds} onReynoldsChange={setReynolds} targetArea={targetArea} />

        <div className="flex flex-col gap-4">
          <div className="nc-card p-4 flex flex-wrap items-center gap-4">
            <div>
              <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Coefficient de traînée
              </div>
              <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>
                {result ? <AnimatedNumber value={result.cd_surrogate} decimals={3} /> : '—'}
              </div>
            </div>
            <AnimatePresence mode="wait">
              {result && (
                <motion.div key={result.model} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <MetricsBadge label={result.model} errorPct={result.cd_error_pct_expected} />
                </motion.div>
              )}
            </AnimatePresence>
            {result && (
              <div className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>
                {result.latency_ms.toFixed(1)} ms
              </div>
            )}
          </div>

          {error && (
            <div className="nc-card p-4 text-sm" style={{ color: 'var(--warn)' }}>
              {error}
            </div>
          )}

          <FieldViewer result={result} loading={loading && !result} />
        </div>
      </div>
    </div>
  )
}
