import { useEffect, useState } from 'react'

/**
 * The Bracket Lab is a static result, not a live model: its numbers are extracted
 * once from the bracket-lab study by `scripts/build_bracket_data.py` into
 * `public/bracket/bracket.json`, and every figure on the page is read from there.
 */

export interface BracketTracePoint {
  /** 1-based index of the FEA run within the optimisation */
  n: number
  mass_g: number
  sigma_eval: number
  delta_max: number
  feasible: boolean
  /** lightest FEA-feasible mass so far, null until the first feasible run */
  best_g: number | null
}

export interface BracketRun {
  key: string
  label: string
  kind: 'agent' | 'optuna'
  mass_g: number
  sigma_eval: number
  delta_max: number
  feasible: boolean
  fea_used: number
  /** σ_eval of the same design re-run on the finer mesh (h = 1.5 mm) */
  sigma_eval_fine: number
  params: Record<string, number>
  trace: BracketTracePoint[]
}

export interface BracketParam {
  name: string
  meaning: string
  unit: string
  min: number
  max: number
  baseline: number
}

export interface BracketData {
  limits: { sigma_eval: number; delta_max: number }
  fea_budget: number
  baseline: { mass_g: number; sigma_eval: number; delta_max: number; feasible: boolean }
  runs: BracketRun[]
  params: BracketParam[]
}

let cache: Promise<BracketData> | null = null

function load(): Promise<BracketData> {
  cache ??= fetch(`${import.meta.env.BASE_URL}bracket/bracket.json`).then((r) => {
    if (!r.ok) throw new Error(`bracket.json: HTTP ${r.status}`)
    return r.json() as Promise<BracketData>
  })
  return cache
}

export function useBracketData(): { data: BracketData | null; error: string | null } {
  const [data, setData] = useState<BracketData | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    load()
      .then((d) => !cancelled && setData(d))
      .catch((e) => {
        cache = null
        if (!cancelled) setError(String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])
  return { data, error }
}

/** The lightest FEA-feasible design across all runs. */
export function bestRun(d: BracketData): BracketRun {
  return d.runs.filter((r) => r.feasible).reduce((a, b) => (b.mass_g < a.mass_g ? b : a))
}

/** Mass change against the baseline, in percent (negative = lighter). */
export function massDeltaPct(d: BracketData, mass: number): number {
  return (100 * (mass - d.baseline.mass_g)) / d.baseline.mass_g
}
