import { useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Eraser, PanelRightClose, Send, Sparkles } from 'lucide-react'
import { ToolCallCard } from '@/components/ToolCallCard'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useChat, useChatSuggestions } from './ChatProvider'

function Composer() {
  const { send, busy } = useChat()
  const ref = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const value = ref.current?.value ?? ''
    if (!value.trim() || busy) return
    send(value)
    if (ref.current) {
      ref.current.value = ''
      ref.current.style.height = 'auto'
    }
  }

  return (
    <div className="border-t border-sidebar-border p-3">
      <div className="focus-within:border-ring/60 focus-within:ring-ring/20 flex items-end gap-2 rounded-lg border border-input bg-background p-2 transition focus-within:ring-[3px]">
        <textarea
          ref={ref}
          rows={1}
          placeholder="Demande une prédiction, une optimisation…"
          disabled={busy}
          onInput={(e) => {
            // Grow with the content up to a ceiling, so a long prompt is visible
            // without the composer ever eating the transcript.
            const el = e.currentTarget
            el.style.height = 'auto'
            el.style.height = `${Math.min(el.scrollHeight, 140)}px`
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          className="max-h-[140px] flex-1 resize-none bg-transparent text-sm leading-relaxed outline-none placeholder:text-muted-foreground disabled:opacity-50"
        />
        <Button size="icon-sm" onClick={submit} disabled={busy} aria-label="Envoyer">
          <Send />
        </Button>
      </div>
      <p className="mt-1.5 text-[10px] text-muted-foreground">
        <kbd className="rounded border border-border px-1">Entrée</kbd> pour envoyer ·{' '}
        <kbd className="rounded border border-border px-1">⇧ Entrée</kbd> pour un saut de ligne
      </p>
    </div>
  )
}

function Transcript() {
  const { turns, busy, send } = useChat()
  const suggestions = useChatSuggestions()
  const scrollRef = useRef<HTMLDivElement>(null)

  // Follow the stream: every token appended should keep the newest text in view.
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [turns])

  if (turns.length === 0) {
    return (
      <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-3 pt-6">
        <div className="flex flex-col items-start gap-2">
          <div className="rounded-lg border border-sidebar-border bg-muted/40 p-2">
            <Sparkles className="size-4 text-ring" />
          </div>
          <p className="text-sm font-medium">Copilote d&apos;ingénierie</p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            L&apos;agent voit la configuration affichée à l&apos;écran et peut appeler les mêmes
            modèles que les labs. Pose-lui une question sans quitter la page.
          </p>
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
            Pour démarrer
          </div>
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="rounded-md border border-sidebar-border bg-muted/30 px-2.5 py-2 text-left text-xs leading-snug text-muted-foreground transition hover:border-ring/40 hover:bg-muted/60 hover:text-foreground"
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-3">
      {turns.map((turn, i) => (
        <div key={i} className="space-y-1.5">
          {turn.role === 'user' ? (
            <div className="flex justify-end">
              <div className="max-w-[90%] rounded-lg rounded-br-sm bg-primary px-3 py-2 text-sm leading-relaxed text-primary-foreground">
                {turn.text}
              </div>
            </div>
          ) : (
            <>
              {turn.tools.map((tc) => (
                <ToolCallCard
                  key={tc.id}
                  name={tc.name}
                  input={tc.input}
                  result={tc.result}
                  isError={tc.isError}
                  running={tc.running}
                />
              ))}
              {turn.text ? (
                <div className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">{turn.text}</div>
              ) : (
                busy &&
                i === turns.length - 1 && (
                  <div className="flex gap-1 py-1">
                    {[0, 1, 2].map((d) => (
                      <motion.span
                        key={d}
                        className="size-1.5 rounded-full bg-muted-foreground"
                        animate={{ opacity: [0.25, 1, 0.25] }}
                        transition={{ duration: 1.1, repeat: Infinity, delay: d * 0.18 }}
                      />
                    ))}
                  </div>
                )
              )}
            </>
          )}
        </div>
      ))}
    </div>
  )
}

export function ChatSidebar() {
  const { open, setOpen, turns, clear, pageContext } = useChat()

  return (
    <AnimatePresence initial={false}>
      {open && (
        <>
          {/* Below lg the panel would eat the whole lab, so it floats over it instead
              of taking a column — same component, same state, different occupancy. */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-30 bg-background/70 backdrop-blur-sm lg:hidden"
          />
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 380, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            className="no-print fixed inset-y-0 right-0 z-40 max-w-full shrink-0 overflow-hidden border-l border-sidebar-border bg-sidebar lg:static lg:z-auto"
          >
            <div className="flex h-full w-[380px] max-w-[100vw] flex-col">
              <header className="flex h-14 shrink-0 items-center gap-2 border-b border-sidebar-border px-3">
                <Sparkles className="size-4 shrink-0 text-ring" />
                <span className="text-sm font-medium">Copilote</span>
                {pageContext && (
                  <span className="truncate rounded-full border border-border px-1.5 py-0.5 text-[10px] text-muted-foreground">
                    {pageContext.label}
                  </span>
                )}
                <div className="ml-auto flex items-center gap-0.5">
                  {turns.length > 0 && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon-sm" onClick={clear} aria-label="Effacer la conversation">
                          <Eraser />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Effacer la conversation</TooltipContent>
                    </Tooltip>
                  )}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon-sm" onClick={() => setOpen(false)} aria-label="Fermer">
                        <PanelRightClose />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Fermer le copilote ⌘J</TooltipContent>
                  </Tooltip>
                </div>
              </header>

              <Transcript />
              <Separator className="bg-sidebar-border" />
              <Composer />
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
