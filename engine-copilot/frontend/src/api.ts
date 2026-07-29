export interface FlowPredictRequest {
  coeffs: number[]
  reynolds: number
  model: 'unet-phys' | 'unet' | 'fno'
  target_area?: number | null
}

export interface FlowPredictResponse {
  cd_surrogate: number
  cd_error_pct_expected: number
  fields: { u: number[][]; v: number[][]; p: number[][] }
  mask: number[][]
  domain: { nx: number; ny: number }
  boundary_xy: number[][]
  target_area: number
  latency_ms: number
  model: string
}

export async function predictFlow(req: FlowPredictRequest): Promise<FlowPredictResponse> {
  const res = await fetch('/api/flow/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`predictFlow failed: ${res.status}`)
  return res.json()
}

export interface HealthResponse {
  ok: boolean
  claude_cli_installed: boolean
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health')
  return res.json()
}
