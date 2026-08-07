import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { resetAgent, streamAgentChat } from '@/api'

export interface ToolCall {
  id: string
  name: string
  input: unknown
  result?: unknown
  isError?: boolean
  running: boolean
}

export interface Turn {
  role: 'user' | 'assistant'
  text: string
  tools: ToolCall[]
}

/** What the page the user is currently looking at wants the agent to know. Labs
 *  publish this so a question like "et si je montais Re ?" resolves against the
 *  design actually on screen instead of the agent having to ask for every number. */
export interface PageContext {
  label: string
  /** short `key: value` lines, rendered verbatim into the prompt preamble */
  lines: string[]
  suggestions: string[]
}

interface ChatState {
  turns: Turn[]
  busy: boolean
  open: boolean
  pageContext: PageContext | null
  setOpen: (open: boolean) => void
  toggle: () => void
  send: (message: string) => void
  clear: () => void
  publishContext: (ctx: PageContext | null) => void
}

const ChatCtx = createContext<ChatState | null>(null)

const OPEN_KEY = 'engine-copilot.chat.open'

const GENERIC_SUGGESTIONS = [
  'Quels modèles sont disponibles et quelle est leur erreur ?',
  'Explique-moi ce que fait le surrogate par rapport au solveur LBM.',
]

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  // An explicit choice always wins. Without one, open on a wide screen (where the panel
  // takes its own column and is the feature worth discovering) and closed on a narrow
  // one (where it overlays, and would otherwise bury the lab on first load).
  const [open, setOpenState] = useState(() => {
    const stored = localStorage.getItem(OPEN_KEY)
    if (stored !== null) return stored === '1'
    return window.matchMedia('(min-width: 1024px)').matches
  })
  const [pageContext, setPageContext] = useState<PageContext | null>(null)

  // The agent request must read the context that was live when Send was pressed,
  // not whatever a slider moved it to mid-stream, so keep a ref alongside state.
  const contextRef = useRef<PageContext | null>(null)
  contextRef.current = pageContext

  const setOpen = useCallback((next: boolean) => {
    setOpenState(next)
    localStorage.setItem(OPEN_KEY, next ? '1' : '0')
  }, [])

  const toggle = useCallback(() => setOpen(!open), [open, setOpen])

  const publishContext = useCallback((ctx: PageContext | null) => setPageContext(ctx), [])

  const send = useCallback(
    async (message: string) => {
      const text = message.trim()
      if (!text || busy) return
      setBusy(true)
      setTurns((t) => [...t, { role: 'user', text, tools: [] }, { role: 'assistant', text: '', tools: [] }])

      // The transcript shows what the user typed; the wire carries the design state
      // too, so "optimise cette forme" has a referent.
      const ctx = contextRef.current
      const wire = ctx?.lines.length
        ? `[État courant de l'interface — ${ctx.label}]\n${ctx.lines.join('\n')}\n\n${text}`
        : text

      try {
        await streamAgentChat(wire, (event) => {
          setTurns((prev) => {
            const next = [...prev]
            const last = { ...next[next.length - 1] }
            last.tools = [...last.tools]
            if (event.type === 'text') {
              last.text += event.delta
            } else if (event.type === 'tool_use_start') {
              last.tools.push({
                id: event.id,
                name: event.name.replace('mcp__engine_copilot__', ''),
                input: event.input,
                running: true,
              })
            } else if (event.type === 'tool_use_end') {
              const idx = last.tools.findIndex((tc) => tc.id === event.id)
              if (idx >= 0) {
                last.tools[idx] = { ...last.tools[idx], running: false, result: event.result, isError: event.is_error }
              }
            }
            next[next.length - 1] = last
            return next
          })
        })
      } catch (e) {
        setTurns((prev) => {
          const next = [...prev]
          next[next.length - 1] = { ...next[next.length - 1], text: `Erreur : ${String(e)}` }
          return next
        })
      } finally {
        setBusy(false)
      }
    },
    [busy],
  )

  // The transcript is only half the conversation: the other half is the SDK session
  // the backend resumes by id. Drop both, and drop the local one whatever the network
  // says — a clear that visibly fails to clear is worse than one that leaves a stale
  // session behind.
  const clear = useCallback(() => {
    setTurns([])
    void resetAgent().catch(() => {})
  }, [])

  const value = useMemo(
    () => ({ turns, busy, open, pageContext, setOpen, toggle, send, clear, publishContext }),
    [turns, busy, open, pageContext, setOpen, toggle, send, clear, publishContext],
  )

  return <ChatCtx.Provider value={value}>{children}</ChatCtx.Provider>
}

export function useChat(): ChatState {
  const ctx = useContext(ChatCtx)
  if (!ctx) throw new Error('useChat must be used inside <ChatProvider>')
  return ctx
}

export function useChatSuggestions(): string[] {
  const { pageContext } = useChat()
  return pageContext?.suggestions.length ? pageContext.suggestions : GENERIC_SUGGESTIONS
}

/**
 * Publish the current page's design state to the chat sidebar for as long as the
 * page is mounted, and withdraw it on unmount so a stale Flow Lab design never
 * gets attached to a question asked from another page.
 */
export function usePublishChatContext(ctx: PageContext | null) {
  const { publishContext } = useChat()
  const serialized = ctx ? `${ctx.label}|${ctx.lines.join('|')}|${ctx.suggestions.join('|')}` : ''

  useEffect(() => {
    publishContext(ctx)
    return () => publishContext(null)
    // `ctx` is rebuilt every render by callers; compare it by value instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serialized, publishContext])
}
