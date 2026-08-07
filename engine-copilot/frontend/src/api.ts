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

// SSE has no way to report failure once the 200 headers are out: a backend generator
// that raises — or a server running stale code — just closes the stream. Without this
// guard a truncated or empty stream resolves like a success and the caller sits there
// showing nothing, which is indistinguishable from "still computing".
async function readSSEUntilDone(res: Response, label: string, onData: (data: any) => void): Promise<void> {
  let sawDone = false
  await readSSE(res, (data) => {
    if (data.error) throw new Error(`${label}: ${data.error}`)
    if (data.done) sawDone = true
    onData(data)
  })
  if (!sawDone) throw new Error(`${label}: the stream ended early — backend stopped, or stale code (restart uvicorn).`)
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
  /** The shape on screen. Seeds start #0 so the run descends from the user's design. */
  coeffs?: number[]
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

export async function streamFlowOptimize(
  params: FlowOptimizeParams,
  onIteration: (it: FlowOptimizeIteration) => void,
  onDone: (d: FlowOptimizeDone) => void,
): Promise<void> {
  const res = await fetch('/api/flow/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  await readSSEUntilDone(res, 'Optimisation', (data) => (data.done ? onDone(data) : onIteration(data)))
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
  await readSSEUntilDone(res, 'Simulation', (data) => (data.done ? onDone(data) : onFrame(data)))
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

/** Forget the server-side conversation, so "clear" clears what the agent remembers
 *  and not merely what the transcript shows. */
export async function resetAgent(): Promise<void> {
  await fetch('/api/agent/reset', { method: 'POST' })
}

// `context` is deliberately untyped on the wire: the backend treats every field as
// optional and renders "n/a" for whatever is missing, so the lab sends the shape of
// result it happens to have rather than a schema both sides must agree on.
export interface ReportRequest {
  project: 'flow'
  context: Record<string, unknown>
}

export async function generateReport(req: ReportRequest): Promise<string> {
  const res = await fetch('/api/report/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`generateReport failed: ${res.status}`)
  const data: { markdown: string } = await res.json()
  return data.markdown
}

/** Save text to the user's downloads as `filename`, via a transient object URL. */
export function downloadMarkdown(filename: string, markdown: string): void {
  const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  // Revoking synchronously can race the download in Safari; one frame is enough.
  requestAnimationFrame(() => URL.revokeObjectURL(url))
}
