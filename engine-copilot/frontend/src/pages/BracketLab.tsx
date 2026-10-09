import { motion } from 'framer-motion'
import { PageHeader } from '@/components/PageHeader'
import { StatTile } from '@/components/StatTile'
import { BracketMassPlot } from '@/components/BracketMassPlot'
import { Skeleton } from '@/components/ui/skeleton'
import { usePublishChatContext } from '@/chat/ChatProvider'
import { reveal } from '@/lib/motion'
import { bestRun, massDeltaPct, useBracketData, type BracketData, type BracketParam } from '@/lib/bracket'
import { cn } from '@/lib/utils'

const CARD = 'rounded-[16px] border border-line2 bg-white p-4 shadow-card'
const IMG = `${import.meta.env.BASE_URL}bracket/`

function pct(v: number) {
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)} %`
}

function paramValue(p: BracketParam, v: number) {
  if (p.name === 'rib_on') return v >= 0.5 ? 'yes' : 'no'
  if (p.name === 'cutout_d' && v === 0) return 'none'
  return `${v.toFixed(1)}${p.unit ? ` ${p.unit}` : ''}`
}

function Feasible({ ok, label = false }: { ok: boolean; label?: boolean }) {
  const text = label ? (ok ? 'feasible' : 'infeasible') : ok ? 'yes' : 'no'
  return <span className={ok ? 'text-ink' : 'text-burnt'}>{text}</span>
}

/** One line under a design render: its FEA figures, then the verdict. */
function Caption({ mass, sigma, delta, ok }: { mass: number; sigma: number; delta: number; ok: boolean }) {
  return (
    <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-line pt-2 text-body">
      <span className="cq-num">{mass.toFixed(1)} g</span>
      <span className="cq-num">{sigma.toFixed(1)} MPa</span>
      <span className="cq-num">{delta.toFixed(3)} mm</span>
      <span className="ml-auto">
        <Feasible ok={ok} label />
      </span>
    </div>
  )
}

function ComparisonTable({ data }: { data: BracketData }) {
  const best = bestRun(data)
  const rows = [
    { label: 'Baseline', ...data.baseline, fea: null as number | null, focal: false },
    ...data.runs.map((r) => ({
      label: r.label,
      mass_g: r.mass_g,
      sigma_eval: r.sigma_eval,
      delta_max: r.delta_max,
      feasible: r.feasible,
      fea: r.fea_used,
      focal: r.key === best.key,
    })),
  ]
  const th = 'cq-k py-2 pr-4 text-right font-normal first:text-left'
  const td = 'cq-num py-2 pr-4 text-right text-body first:text-left first:font-sans'

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse">
        <thead>
          <tr className="border-b border-ink">
            <th className={th}>Design</th>
            <th className={th}>Mass <span className="normal-case">g</span></th>
            <th className={th}><span className="normal-case">Δ</span> baseline</th>
            <th className={th}><span className="normal-case">σ_eval</span> MPa</th>
            <th className={th}><span className="normal-case">δ_max</span> mm</th>
            <th className={th}>Feasible</th>
            <th className={th}>FEA runs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-line last:border-b-0">
              <td className={cn(td, r.focal && 'font-medium')}>{r.label}</td>
              <td className={cn(td, r.focal && 'text-burnt')}>{r.mass_g.toFixed(1)}</td>
              <td className={td}>{r.label === 'Baseline' ? '—' : pct(massDeltaPct(data, r.mass_g))}</td>
              <td className={cn(td, r.sigma_eval > data.limits.sigma_eval && 'text-burnt')}>{r.sigma_eval.toFixed(1)}</td>
              <td className={cn(td, r.delta_max > data.limits.delta_max && 'text-burnt')}>{r.delta_max.toFixed(3)}</td>
              <td className={td}>
                <Feasible ok={r.feasible} />
              </td>
              <td className={td}>{r.fea ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function BracketLab() {
  const { data, error } = useBracketData()
  const best = data ? bestRun(data) : null

  usePublishChatContext({
    label: 'Bracket Lab',
    lines:
      data && best
        ? [
            `aluminium bracket, 7 parameters, limits σ_eval ≤ ${data.limits.sigma_eval} MPa and δ_max ≤ ${data.limits.delta_max} mm (FEA)`,
            `baseline ${data.baseline.mass_g.toFixed(1)} g, σ_eval ${data.baseline.sigma_eval.toFixed(1)} MPa, δ_max ${data.baseline.delta_max.toFixed(3)} mm, ${data.baseline.feasible ? 'feasible' : 'infeasible'}`,
            ...data.runs.map(
              (r) =>
                `${r.label}: ${r.mass_g.toFixed(2)} g, σ_eval ${r.sigma_eval.toFixed(1)} MPa, δ_max ${r.delta_max.toFixed(3)} mm, ${r.fea_used} FEA runs`,
            ),
          ]
        : [],
    suggestions: ['Why is the baseline bracket infeasible?', 'How does the agent compare with Optuna here?'],
  })

  return (
    <>
      <PageHeader title="Bracket Lab" subtitle="Mass reduction of an aluminium bracket, verified by FEA" />

      <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-4 px-6 py-6">
        {error && (
          <div className="cq-error">
            <div className="text-body font-medium">The study data did not load.</div>
            <div className="mt-1 cq-num text-band cq-muted">{error}</div>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label="BEST FEASIBLE MASS"
              value={best?.mass_g ?? null}
              unit="g"
              decimals={1}
              loading={!data}
              interval={data && best ? `${pct(massDeltaPct(data, best.mass_g))} · BASELINE ${data.baseline.mass_g.toFixed(1)}` : undefined}
            />
          </div>
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label={<>VON MISES · <span className="normal-case">σ_eval</span></>}
              value={best?.sigma_eval ?? null}
              unit="MPa"
              decimals={1}
              loading={!data}
              interval={data && best ? `LIMIT ${data.limits.sigma_eval} · FINE MESH ${best.sigma_eval_fine.toFixed(1)}` : undefined}
            />
          </div>
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label={<>DEFLECTION · <span className="normal-case">δ_max</span></>}
              value={best?.delta_max ?? null}
              unit="mm"
              decimals={3}
              loading={!data}
              interval={data ? `LIMIT ${data.limits.delta_max} · BASELINE ${data.baseline.delta_max.toFixed(2)}` : undefined}
            />
          </div>
          <div className="cq-calque px-[21px] py-[19px]">
            <StatTile
              label="FEA RUNS"
              value={best?.fea_used ?? null}
              unit={data ? `/ ${data.fea_budget}` : undefined}
              decimals={0}
              loading={!data}
              interval={data && best ? `${best.label.toUpperCase()} · BEST DESIGN` : undefined}
            />
          </div>
        </div>

        <motion.div className="grid grid-cols-1 gap-4 md:grid-cols-2" {...reveal(0.05)}>
          <div className={CARD}>
            <div className="cq-k mb-2">BASELINE</div>
            <img src={`${IMG}baseline.png`} alt="Baseline bracket, iso view" className="mx-auto h-[260px] w-full object-contain" />
            {data && (
              <Caption mass={data.baseline.mass_g} sigma={data.baseline.sigma_eval} delta={data.baseline.delta_max} ok={data.baseline.feasible} />
            )}
          </div>
          <div className={CARD}>
            <div className="cq-k mb-2">{best ? `${best.label} · ` : ''}VON MISES, FEA</div>
            <img src={`${IMG}best_stress.png`} alt="Optimised bracket, von Mises stress map from FEA" className="mx-auto h-[260px] w-full object-contain" />
            {best && (
              <Caption mass={best.mass_g} sigma={best.sigma_eval} delta={best.delta_max} ok={best.feasible} />
            )}
          </div>
        </motion.div>

        <motion.div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]" {...reveal(0.1)}>
          <div className={CARD}>
            <div className="cq-k">LIGHTEST FEASIBLE MASS · BY FEA RUN</div>
            {data && best ? <BracketMassPlot data={data} best={best} /> : <Skeleton className="mt-3 h-[300px] w-full" />}
          </div>

          <div className={CARD}>
            <div className="cq-k mb-2">{best ? `${best.label.toUpperCase()} · ` : ''}PARAMETERS · BASELINE → BEST</div>
            {data && best
              ? data.params.map((p) => (
                  <div key={p.name} className="cq-row">
                    <span className="first-letter:uppercase">{p.meaning}</span>
                    <span>
                      <span className="text-ink3">{paramValue(p, p.baseline)} → </span>
                      {paramValue(p, best.params[p.name])}
                    </span>
                  </div>
                ))
              : <Skeleton className="h-[230px] w-full" />}
          </div>
        </motion.div>

        <motion.div className={CARD} {...reveal(0.15)}>
          <div className="cq-k mb-1">COMPARISON · FEA-VERIFIED</div>
          {data ? <ComparisonTable data={data} /> : <Skeleton className="h-[150px] w-full" />}
        </motion.div>
      </div>
    </>
  )
}
