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
  domain: { nx: number; ny: number; cx: number; cy: number; downscale: number }
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

// Generic reader for our own SSE endpoints (optimize/simulate/agent chat): a plain
// fetch() ReadableStream rather than EventSource, since EventSource can't send a
// POST body. Every event here carries a self-describing JSON payload (a `done`
// flag, or an explicit `type`), so the caller doesn't need the SSE `event:` line —
// only the `data:` line is parsed.
async function readSSE(res: Response, onData: (data: any) => void): Promise<void> {
  if (!res.ok || !res.body) throw new Error(`stream failed: ${res.status}`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''
    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      onData(JSON.parse(dataLine.slice(5).trim()))
    }
  }
}

export interface FlowVerifyRequest {
  coeffs: number[]
  reynolds: number
  target_area?: number | null
  steps?: number
}

export interface FlowVerifyResponse {
  cd_lbm: number
  cd_surrogate: number
  gap_pct: number
  fields: { u: number[][]; v: number[][]; p: number[][] }
  wall_time_s: number
}

export async function verifyFlow(req: FlowVerifyRequest): Promise<FlowVerifyResponse> {
  const res = await fetch('/api/flow/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`verifyFlow failed: ${res.status}`)
  return res.json()
}

export interface FlowOptimizeParams {
  reynolds: number
  n_starts?: number
  iters?: number
  model?: string
  seed?: number
}

export interface FlowOptimizeIteration {
  it: number
  cd_best: number
  cd_mean: number
  coeffs_best: number[]
  boundary_xy: number[][]
}

export interface FlowOptimizeDone {
  done: true
  final_coeffs: number[]
  history: { cd_mean: number[]; cd_best: number[] }
  cd_surrogate_final: number
  target_area: number
}

// The optimize endpoint takes query params, not a JSON body (see main.py) — it
// predates the JSON-body convention used everywhere else and streams fine either way.
export async function streamFlowOptimize(
  params: FlowOptimizeParams,
  onIteration: (it: FlowOptimizeIteration) => void,
  onDone: (d: FlowOptimizeDone) => void,
): Promise<void> {
  const qs = new URLSearchParams(
    Object.entries(params).reduce<Record<string, string>>((acc, [k, v]) => {
      if (v !== undefined) acc[k] = String(v)
      return acc
    }, {}),
  )
  const res = await fetch(`/api/flow/optimize?${qs.toString()}`, { method: 'POST' })
  await readSSE(res, (data) => (data.done ? onDone(data) : onIteration(data)))
}

export interface FlowSimulateRequest {
  coeffs: number[]
  reynolds: number
  target_area?: number | null
  n_frames?: number
  steps?: number
}

export interface FlowSimulateFrame {
  t_step: number
  u: number[][]
  v: number[][]
  vorticity: number[][]
  frame_index: number
}

export interface FlowSimulateDone {
  done: true
  cd_final: number
}

export async function streamFlowSimulate(
  req: FlowSimulateRequest,
  onFrame: (f: FlowSimulateFrame) => void,
  onDone: (d: FlowSimulateDone) => void,
): Promise<void> {
  const res = await fetch('/api/flow/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  await readSSE(res, (data) => (data.done ? onDone(data) : onFrame(data)))
}

export interface HealthResponse {
  ok: boolean
  claude_cli_installed: boolean
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health')
  return res.json()
}

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'tool_use_start'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_use_end'; id: string; result: unknown; is_error: boolean }
  | { type: 'end'; usage: Record<string, unknown>; is_error: boolean }

export async function streamAgentChat(message: string, onEvent: (e: AgentEvent) => void): Promise<void> {
  const res = await fetch('/api/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  await readSSE(res, (data) => onEvent(data as AgentEvent))
}

export interface StressPredictRequest {
  holes: number[][]
  load_angle_deg: number
  model: 'gnn-mp10-glob' | 'gnn-mp10-glob-corr' | 'gnn-mp10'
}

export interface StressPredictResponse {
  peak_vm_surrogate: number
  peak_error_pct_expected: number
  field: { vm: (number | null)[][]; extent: number[] }
  mesh: { nodes: number[][]; triangles: number[][]; vm_nodal: number[] }
  peak_location: number[]
  vm_coarse_fem: (number | null)[][]
  peak_vm_coarse_fem: number
  latency_ms: number
  model: string
}

export async function predictStress(req: StressPredictRequest): Promise<StressPredictResponse> {
  const res = await fetch('/api/stress/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `predictStress failed: ${res.status}`)
  }
  return res.json()
}

export interface StressOptimizeRequest {
  holes: number[][]
  load_angle_deg: number
  model: string
  n_iters?: number
  seed?: number
}

export interface StressOptimizeIteration {
  it: number
  peak_vm_current: number
  best_holes: number[][]
}

export interface StressOptimizeDone {
  done: true
  final_holes: number[][]
  peak_vm_initial: number
  peak_vm_final: number
}

export async function streamStressOptimize(
  req: StressOptimizeRequest,
  onIteration: (it: StressOptimizeIteration) => void,
  onDone: (d: StressOptimizeDone) => void,
): Promise<void> {
  const res = await fetch('/api/stress/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  await readSSE(res, (data) => (data.done ? onDone(data) : onIteration(data)))
}
