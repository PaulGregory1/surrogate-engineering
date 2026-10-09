import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { RotateCcw } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { StatTile } from '@/components/StatTile'
import { useChairViewer, type WebMesh } from '@/components/ChairViewer'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { reveal } from '@/lib/motion'

// Everything on this page is read from public/fcso/, written by
// fcso-chairs/scripts/export_platform.py from the study's report. No number is typed here.

type Cell = { ratio: number; mesh: number; step: number; time: number; comps: number; stray: number; final: boolean }
type Run = { stop: 'target' | 'budget'; steps: number; time: number; cells: Record<string, Cell> }
type Chair = { id: string; name: string; runs: Record<'fcso_x0' | 'gd', Run> }
type Threshold = {
  success: number
  median_time: number | null
  mean_stray_share: number | null
  whole: number
}
type Method = { key: string; name: string; n_done: number; '50': Threshold; '70': Threshold }
type Side = { mean_share: number; whole: number }
type Summary = {
  methods: Method[]
  overhead: Record<string, { n: number; median_ratio: number; min_ratio: number; max_ratio: number }>
  paired: Record<string, { n: number; fcso_x0: Side; gd: Side }>
  chairs: Chair[]
}
type Meshes = { init: WebMesh; runs: Record<string, Record<string, WebMesh>> }

const ROWS = [
  { key: 'fcso_x0', label: 'FCSO' },
  { key: 'gd', label: 'Gradient descent' },
] as const
const COLS = [
  { key: 'init', label: 'Initial' },
  { key: '30', label: '−30 %' },
  { key: '50', label: '−50 %' },
  { key: '70', label: '−70 %' },
]
const SHORT: Record<string, string> = {
  fcso_x0: 'FCSO',
  gd: 'Gradient descent',
  gd_reg: 'GD + regulariser',
  fcso: 'FCSO, noisy guidance',
  fcso_noguide: 'FCSO, no guidance',
  fmg: 'FMG',
}

const pct = (x: number | null, d = 1) => (x === null ? '—' : `${(100 * x).toFixed(d)} %`)
const secs = (s: number | null) => (s === null ? '—' : s < 600 ? `${s.toFixed(0)} s` : `${(s / 60).toFixed(1)} min`)
const label = (name: string) => name.charAt(0).toUpperCase() + name.slice(1)

export function FcsoReview() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [chairId, setChairId] = useState<string | null>(null)
  const [meshes, setMeshes] = useState<Meshes | null>(null)
  const [thr, setThr] = useState<'50' | '70'>('50')
  const [nonce, setNonce] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/fcso/summary.json')
      .then((r) => r.json())
      .then((s: Summary) => {
        setSummary(s)
        setChairId(s.chairs[0].id)
      })
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (!chairId) return
    let cancelled = false
    setMeshes(null)
    fetch(`/fcso/meshes/${chairId}.json`)
      .then((r) => r.json())
      .then((m: Meshes) => !cancelled && setMeshes(m))
      .catch((e) => setError(String(e)))
    return () => {
      cancelled = true
    }
  }, [chairId])

  const chair = summary?.chairs.find((c) => c.id === chairId) ?? null

  // cell key "<method>:<snapshot>" -> mesh; the initial shape is shared by both rows
  const cells = useMemo(() => {
    if (!meshes || !chair) return null
    const out: Record<string, WebMesh> = {}
    for (const r of ROWS)
      for (const c of COLS) {
        const m = c.key === 'init' ? meshes.init : meshes.runs[r.key]?.[c.key]
        if (m && chair.runs[r.key].cells[c.key]) out[`${r.key}:${c.key}`] = m
      }
    return out
  }, [meshes, chair])

  const gridRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useChairViewer(gridRef, canvasRef, cells, nonce)

  const paired = summary?.paired['50']
  const overhead = summary?.overhead['50']

  return (
    <>
      <PageHeader
        title="FCSO Review"
        subtitle="Volume reduction of 10 chairs: flow-corrected optimisation vs gradient descent"
        actions={
          summary && (
            <Select value={chairId ?? undefined} onValueChange={setChairId}>
              <SelectTrigger size="sm" className="w-[300px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {summary.chairs.map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {label(c.name)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )
        }
      />

      <div className="mx-auto flex w-full max-w-[1200px] flex-col gap-4 px-6 py-6">
        {error && (
          <div className="cq-error">
            <div className="text-body font-medium">The review data did not load.</div>
            <div className="cq-num cq-muted mt-1 text-band">{error}</div>
          </div>
        )}

        {/* Measured on the same 9 chairs both methods brought to −50 %. */}
        <motion.div className="grid grid-cols-1 gap-4 sm:grid-cols-3" {...reveal()}>
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label="WHOLE CHAIRS AT −50 %"
              value={paired?.fcso_x0.whole ?? null}
              decimals={0}
              unit={paired ? ` / ${paired.n}` : undefined}
              interval={paired ? `GRADIENT DESCENT ${paired.gd.whole} / ${paired.n}` : undefined}
            />
          </div>
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label="STRAY VOLUME AT −50 %"
              value={paired ? 100 * paired.fcso_x0.mean_share : null}
              decimals={1}
              unit=" %"
              interval={paired ? `GRADIENT DESCENT ${pct(paired.gd.mean_share)}` : undefined}
            />
          </div>
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label="TIME COST VS GD"
              value={overhead?.median_ratio ?? null}
              decimals={1}
              unit=" ×"
              interval={
                overhead
                  ? `MEDIAN · RANGE ${overhead.min_ratio.toFixed(1)}–${overhead.max_ratio.toFixed(1)} ×`
                  : undefined
              }
            />
          </div>
        </motion.div>

        <motion.div
          className="rounded-[16px] border border-line2 bg-white p-5 shadow-card"
          {...reveal(0.08)}
        >
          <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2">
            <div className="cq-title">Trajectory</div>
            <span className="cq-k flex items-center gap-1.5">
              <span className="inline-block size-2 rounded-[2px] bg-burnt" /> STRAY COMPONENTS
            </span>
            <span className="cq-k hidden sm:inline">DRAG TO ROTATE</span>
            <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setNonce((n) => n + 1)}>
              <RotateCcw />
              Reset view
            </Button>
          </div>

          <div className="overflow-x-auto">
            <div ref={gridRef} className="relative min-w-[640px] touch-none select-none">
              <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-10 size-full" />
              <div className="grid grid-cols-[110px_repeat(4,minmax(0,1fr))] gap-x-3 gap-y-4">
                <div />
                {COLS.map((c) => (
                  <div key={c.key} className="cq-k text-center">
                    {c.label}
                  </div>
                ))}
                {ROWS.map((r) => {
                  const run = chair?.runs[r.key]
                  return (
                    <Row key={r.key} label={r.label} run={run} loading={!cells} rowKey={r.key} />
                  )
                })}
              </div>
            </div>
          </div>
        </motion.div>

        {summary && (
          <motion.div
            className="rounded-[16px] border border-line2 bg-white p-5 shadow-card"
            {...reveal(0.16)}
          >
            <div className="mb-3 flex items-center gap-4">
              <div className="cq-title">All methods</div>
              <span className="cq-k">10 CHAIRS · 1000-STEP BUDGET</span>
              <Tabs value={thr} onValueChange={(v) => setThr(v as '50' | '70')} className="ml-auto">
                <TabsList>
                  <TabsTrigger value="50">−50 %</TabsTrigger>
                  <TabsTrigger value="70">−70 %</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-body">
                <thead>
                  <tr className="border-b border-ink text-left">
                    {['Method', 'Reached', 'Median time', 'Stray volume', 'Whole'].map((h, i) => (
                      <th key={h} className={cn('cq-k py-2 font-normal', i > 0 && 'text-right')}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {summary.methods.map((m) => {
                    const t = m[thr]
                    return (
                      <tr key={m.key} className="border-b border-line">
                        <td className={cn('py-2', m.key === 'fcso_x0' && 'font-medium')}>
                          {SHORT[m.key] ?? m.name}
                        </td>
                        <td className="cq-num py-2 text-right">
                          {t.success} / {m.n_done}
                        </td>
                        <td className="cq-num py-2 text-right">{secs(t.median_time)}</td>
                        <td className="cq-num py-2 text-right">{pct(t.mean_stray_share)}</td>
                        <td className="cq-num py-2 text-right">
                          {t.whole} / {t.success}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </div>
    </>
  )
}

function Row({ label, run, loading, rowKey }: { label: string; run?: Run; loading: boolean; rowKey: string }) {
  return (
    <>
      <div className="flex flex-col justify-center">
        <div className="text-body font-medium">{label}</div>
        {run && (
          <div className="cq-num mt-1 text-band text-ink3">
            {run.stop === 'target' ? `${run.steps} steps · ${secs(run.time)}` : 'budget stop'}
          </div>
        )}
      </div>
      {COLS.map((c) => {
        const cell = run?.cells[c.key]
        return (
          <div key={c.key} className="flex min-w-0 flex-col">
            {cell ? (
              <div
                data-cell={`${rowKey}:${c.key}`}
                className={cn('aspect-square rounded-[10px] border border-line', loading && 'animate-pulse bg-paper')}
              />
            ) : (
              <div className="cq-k flex aspect-square items-center justify-center rounded-[10px] border border-dashed border-line2">
                NOT REACHED
              </div>
            )}
            {cell && (
              <div className="mt-1.5 text-center">
                <span className="cq-num text-band">
                  {cell.final ? 'FINAL · ' : ''}V/V₀ {cell.ratio.toFixed(2)}
                </span>
                {cell.comps > 1 && <span className="cq-num ml-2 text-band text-burnt">+{cell.comps - 1}</span>}
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}
