import { useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Eraser, PanelRightClose, Send } from 'lucide-react'
import { ToolCallCard } from '@/components/ToolCallCard'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { CALQUE_EASE } from '@/lib/motion'
import { Markdown } from './Markdown'
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
    <div className="border-t border-line p-3">
      <div className="flex items-end gap-2 rounded-[10px] border border-line2 bg-white p-2 transition-shadow focus-within:shadow-[var(--focus)]">
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
          className="max-h-[140px] flex-1 resize-none bg-transparent text-ctl leading-relaxed outline-none placeholder:text-ink3 disabled:opacity-50"
        />
        {/* Outline, not filled: the lab's primary action already spends this
            screen's one orange flat. */}
        <Button variant="outline" size="icon-sm" onClick={submit} disabled={busy} aria-label="Envoyer">
          <Send />
        </Button>
      </div>
      <p className="cq-k mt-2">ENTRÉE POUR ENVOYER · ⇧ ENTRÉE POUR UN SAUT DE LIGNE</p>
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
    // An empty state is an invitation, never an excuse.
    return (
      <div className="flex flex-1 flex-col gap-6 overflow-y-auto p-4 pt-7">
        <div>
          <div className="cq-k mb-3">COPILOTE D’INGÉNIERIE</div>
          <div className="cq-value text-[20px]">Pose ta question sans quitter la page</div>
          <p className="mt-3 cq-muted text-body leading-[1.7]">
            L’agent voit la configuration affichée à l’écran et appelle les mêmes modèles que les
            labs. Chaque appel reste visible dans la conversation.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <div className="cq-k">POUR DÉMARRER</div>
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="rounded-[10px] border border-line2 border-l-2 border-l-transparent bg-white px-3 py-2.5 text-left cq-muted text-body leading-snug transition-colors duration-150 hover:border-l-or hover:text-ink"
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto p-4">
      {turns.map((turn, i) => (
        <div key={i} className="space-y-2">
          {turn.role === 'user' ? (
            // The transcript reads as a document, not as a chat app: the question is
            // a quoted sheet of paper, the answer is body text under it. A filled
            // orange bubble would be an orange flat carrying no quantity.
            <div className="rounded-[10px] rounded-tr-sm border border-line2 bg-white px-3 py-2 text-ctl leading-relaxed">
              {turn.text}
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
                <div className="px-0.5">
                  <Markdown>{turn.text}</Markdown>
                </div>
              ) : (
                busy &&
                i === turns.length - 1 && (
                  <div className="flex gap-1 px-0.5 py-1">
                    {[0, 1, 2].map((d) => (
                      <motion.span
                        key={d}
                        className="size-1 rounded-full bg-ink3"
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
            className="fixed inset-0 z-30 bg-paper/60 backdrop-blur-sm lg:hidden"
          />
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 380, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.42, ease: CALQUE_EASE }}
            className="no-print fixed inset-y-0 right-0 z-40 max-w-full shrink-0 overflow-hidden border-l border-line bg-sidebar backdrop-blur-xl backdrop-saturate-150 lg:static lg:z-auto"
          >
            <div className="flex h-full w-[380px] max-w-[100vw] flex-col">
              <header className="flex h-14 shrink-0 items-center gap-2.5 border-b border-line px-4">
                <span className="cq-k">COPILOTE</span>
                {pageContext && (
                  <span className="cq-k truncate rounded-full border border-line2 px-2 py-0.5 text-ink2">
                    {pageContext.label.toUpperCase()}
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
              <Composer />
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
