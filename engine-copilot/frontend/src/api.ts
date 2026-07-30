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

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'tool_use_start'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_use_end'; id: string; result: unknown; is_error: boolean }
  | { type: 'end'; usage: Record<string, unknown>; is_error: boolean }

// EventSource can't send a POST body, so this streams text/event-stream
// manually off a fetch() ReadableStream — the standard pattern for
// POST-triggered SSE.
export async function streamAgentChat(message: string, onEvent: (e: AgentEvent) => void): Promise<void> {
  const res = await fetch('/api/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  if (!res.ok || !res.body) throw new Error(`agent chat failed: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    // The stream arrives with \r\n line endings (Vite's dev proxy normalizes
    // them), so normalize before splitting on the blank-line event boundary.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')

    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''
    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      const payload = JSON.parse(dataLine.slice(5).trim())
      onEvent(payload as AgentEvent)
    }
  }
}
