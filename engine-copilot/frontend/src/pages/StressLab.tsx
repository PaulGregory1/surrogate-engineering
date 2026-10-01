import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { PageHeader } from '@/components/PageHeader'
import { HoleEditor } from '@/components/HoleEditor'
import { StressViewer } from '@/components/StressViewer'
import { StressOptimizePanel } from '@/components/StressOptimizePanel'
import { StatTile } from '@/components/StatTile'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { usePublishChatContext } from '@/chat/ChatProvider'
import { settle } from '@/lib/motion'
import { predictStress, type StressPredictResponse } from '@/api'

const DEFAULT_HOLES = [
  [0.5, 0.5, 0.12],
  [1.3, 0.4, 0.1],
]

const MODELS = [
  { value: 'gnn-mp10-glob', label: 'GNN + global', hint: 'default' },
  { value: 'gnn-mp10-glob-corr', label: 'GNN + global + coarse', hint: 'FEM correction' },
  { value: 'gnn-mp10', label: 'GNN', hint: '10 message-passing steps' },
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

  const band = result ? (result.peak_vm_surrogate * result.peak_error_pct_expected) / 100 : null
  // How far under the true peak the cheap baseline lands — the number that justifies
  // the whole model, and the one the serif line reads out loud.
  const coarseGapPct = result
    ? (100 * (result.peak_vm_surrogate - result.peak_vm_coarse_fem)) / result.peak_vm_surrogate
    : null

  usePublishChatContext({
    label: 'Stress Lab',
    lines: [
      `holes (x, y, r) = ${holes.map((h) => `(${h.map((v) => v.toFixed(2)).join(', ')})`).join(' ')}`,
      `load angle = ${loadAngleDeg.toFixed(0)}°`,
      `model = ${model}`,
      result ? `predicted peak σ_vM = ${result.peak_vm_surrogate.toFixed(3)}` : 'predicted peak σ_vM = (pending)',
    ],
    suggestions: [
      'Where is the stress concentration on this design?',
      'What error should I expect on this stress peak?',
      'Why does the coarse FEM underestimate the peak?',
    ],
  })

  return (
    <>
      <PageHeader
        title="Stress Lab"
        subtitle="Perforated plate under load — mesh-native GNN surrogate"
        actions={
          <Select value={model} onValueChange={(v) => setModel(v as typeof model)}>
            <SelectTrigger size="sm" className="w-[275px]">
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
          <HoleEditor
            holes={holes}
            onChange={setHoles}
            loadAngleDeg={loadAngleDeg}
            onLoadAngleChange={setLoadAngleDeg}
          />

          <div className="flex min-w-0 flex-col gap-4">
            <div className="flex flex-wrap items-start gap-x-8 gap-y-4">
              <div className="cq-calque w-full shrink-0 px-[21px] py-[19px] sm:w-[298px]">
                <StatTile
                  label="PEAK VON MISES STRESS"
                  value={result?.peak_vm_surrogate ?? null}
                  decimals={2}
                  loading={loading && !result}
                  interval={band !== null ? `± ${band.toFixed(2)} · NODAL PEAK` : undefined}
                />
                {result && (
                  <motion.div className="mt-4" key={result.model} {...settle()}>
                    <div className="cq-row">
                      <span>Model</span>
                      <span>{result.model}</span>
                    </div>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="cq-row cursor-default">
                          <span>Measured error</span>
                          <span>± {result.peak_error_pct_expected.toFixed(2)} %</span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-64">
                        Mean relative error of <span className="font-mono">{result.model}</span> on the
                        test split, against the high-fidelity solver. The value shown is a
                        prediction, not a measurement.
                      </TooltipContent>
                    </Tooltip>
                    <div className="cq-row">
                      <span>Coarse FEM</span>
                      <span>{result.peak_vm_coarse_fem.toFixed(2)}</span>
                    </div>
                    {/* The gap is the whole argument for the model, so it gets its
                        own line rather than being left for the reader to subtract. */}
                    {coarseGapPct !== null && (
                      <div className="cq-row">
                        <span>Missed by the FEM</span>
                        <span className="text-burnt">{coarseGapPct.toFixed(0)} %</span>
                      </div>
                    )}
                    <div className="cq-row">
                      <span>Latency</span>
                      <span>{result.latency_ms.toFixed(1)} ms</span>
                    </div>
                  </motion.div>
                )}
              </div>

              {result && (
                <Button variant="outline" size="sm" className="mt-1" onClick={() => setCompare((c) => !c)}>
                  {compare ? 'Hide the coarse FEM' : 'Compare to the coarse FEM'}
                </Button>
              )}
            </div>

            {error && (
              <div className="cq-error">
                <div className="text-body font-medium">The prediction did not complete.</div>
                <div className="mt-1 cq-num text-band cq-muted">{error}</div>
              </div>
            )}

            <div className={`grid grid-cols-1 gap-3 ${compare ? 'md:grid-cols-2' : ''}`}>
              <div className="rounded-[16px] border border-line2 bg-white p-4 shadow-card">
                <div className="cq-k mb-2">GNN SURROGATE</div>
                <StressViewer result={result} loading={loading && !result} />
              </div>

              {compare && result && (
                <div className="rounded-[16px] border border-line2 bg-white p-4 shadow-card">
                  <div className="cq-k mb-2">
                    COARSE FEM · PEAK{' '}
                    <span className="cq-num text-band tracking-normal text-ink2">
                      {result.peak_vm_coarse_fem.toFixed(2)}
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
                  <p className="cq-muted mt-2.5 text-band leading-[1.7]">
                    The baseline the GNN has to beat: it typically underestimates the peak by 24
                    to 36 %, because a coarse mesh does not resolve the stress concentration at
                    the hole edge.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        <StressOptimizePanel holes={holes} loadAngleDeg={loadAngleDeg} model={model} onApply={setHoles} />
      </div>
    </>
  )
}
