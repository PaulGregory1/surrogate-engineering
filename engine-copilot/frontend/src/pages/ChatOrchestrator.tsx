import { useState } from 'react'
import { motion } from 'framer-motion'
import { ToolCallCard } from '../components/ToolCallCard'
import { streamAgentChat } from '../api'

type ToolCall = { id: string; name: string; input: unknown; result?: unknown; isError?: boolean; running: boolean }
type Turn = { role: 'user' | 'assistant'; text: string; tools: ToolCall[] }

const SUGGESTIONS = [
  "Prédis le Cd pour une forme avec a2=0.05, b2=-0.02, a3=0.03, b3=0.01, a4=-0.04, b4=0 à Re=80",
  "Quelle est la précision du modèle unet-phys ?",
  "Est-ce que Re=180 est fiable pour ce modèle ?",
]

export function ChatOrchestrator() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)

  async function send(message: string) {
    if (!message.trim() || busy) return
    setBusy(true)
    setInput('')
    setTurns((t) => [...t, { role: 'user', text: message, tools: [] }, { role: 'assistant', text: '', tools: [] }])

    await streamAgentChat(message, (event) => {
      setTurns((prev) => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.tools = [...last.tools]
        if (event.type === 'text') {
          last.text += event.delta
        } else if (event.type === 'tool_use_start') {
          last.tools.push({ id: event.id, name: event.name.replace('mcp__engine_copilot__', ''), input: event.input, running: true })
        } else if (event.type === 'tool_use_end') {
          const idx = last.tools.findIndex((tc) => tc.id === event.id)
          if (idx >= 0) {
            last.tools[idx] = { ...last.tools[idx], running: false, result: event.result, isError: event.is_error }
          }
        }
        next[next.length - 1] = last
        return next
      })
    }).catch((e) => {
      setTurns((prev) => {
        const next = [...prev]
        next[next.length - 1] = { ...next[next.length - 1], text: `Erreur: ${String(e)}` }
        return next
      })
    })
    setBusy(false)
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-8 flex flex-col" style={{ minHeight: 'calc(100vh - 52px)' }}>
      <div>
        <h1 className="text-2xl font-semibold">Chat Orchestrator</h1>
        <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
          Agent Claude (Claude Agent SDK — abonnement Claude Code local, sans clé API) qui appelle
          directement le service de prédiction du Flow Lab.
        </p>
      </div>

      <div className="flex-1 flex flex-col gap-3 overflow-y-auto py-2">
        {turns.length === 0 && (
          <div className="flex flex-wrap gap-2 mt-4">
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => send(s)} className="nc-card px-3 py-1.5 text-xs text-left hover:brightness-125 transition">
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="flex flex-col gap-2">
            <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
              {turn.role === 'user' ? 'Vous' : 'Engine-Copilot'}
            </div>
            {turn.tools.map((tc) => (
              <ToolCallCard key={tc.id} name={tc.name} input={tc.input} result={tc.result} isError={tc.isError} running={tc.running} />
            ))}
            {turn.text && (
              <div className={turn.role === 'user' ? 'nc-card px-3 py-2 text-sm self-end max-w-[80%]' : 'text-sm'}>{turn.text}</div>
            )}
          </motion.div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
        className="flex gap-2 mt-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Écris un message..."
          className="nc-card flex-1 px-3 py-2 text-sm outline-none"
          disabled={busy}
        />
        <button type="submit" disabled={busy} className="nc-card px-4 py-2 text-sm hover:brightness-125 transition disabled:opacity-50" style={{ color: 'var(--accent)' }}>
          {busy ? '...' : 'Envoyer'}
        </button>
      </form>
    </div>
  )
}
