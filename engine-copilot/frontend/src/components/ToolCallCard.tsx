import { useState } from 'react'
import { motion } from 'framer-motion'
import { ChevronRight } from 'lucide-react'
import { settle } from '@/lib/motion'
import { cn } from '@/lib/utils'

function format(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/**
 * One agent tool invocation. Collapsed by default — the tool's *name* is the part
 * that earns permanent space; the JSON is there for when someone doubts the answer.
 *
 * State is carried by a word plus one mark, and the mark has no hue of its own:
 * running is an orange dot (the only thing here in motion), done is plain ink, a
 * failure is burnt. No green tick, no red cross — the charte has neither.
 */
export function ToolCallCard({
  name,
  input,
  result,
  isError,
  running,
}: {
  name: string
  input: unknown
  result?: unknown
  isError?: boolean
  running: boolean
}) {
  const [open, setOpen] = useState(false)

  return (
    <motion.div
      {...settle()}
      className={cn(
        'overflow-hidden rounded-[10px] border bg-white',
        isError ? 'border-burnt/45' : 'border-line2',
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2.5 py-2 text-left"
      >
        <ChevronRight className={cn('size-3 shrink-0 text-ink3 transition-transform', open && 'rotate-90')} />
        <span
          className={cn(
            'size-1.5 shrink-0 rounded-full',
            running ? 'animate-pulse bg-or' : isError ? 'bg-burnt' : 'bg-ink',
          )}
        />
        <span className="cq-num truncate text-band text-ink">{name}</span>
        <span className={cn('cq-k ml-auto', isError && 'text-burnt')}>
          {running ? 'EN COURS' : isError ? 'ÉCHEC' : 'TERMINÉ'}
        </span>
      </button>

      {open && (
        <div className="space-y-2 border-t border-line px-2.5 py-2 font-mono text-[10px] leading-relaxed text-ink2">
          <div>
            <div className="cq-k mb-1">ARGUMENTS</div>
            <pre className="overflow-x-auto break-all whitespace-pre-wrap">{format(input)}</pre>
          </div>
          {result !== undefined && (
            <div>
              <div className="cq-k mb-1">RÉSULTAT</div>
              <pre className="overflow-x-auto break-all whitespace-pre-wrap">{format(result)}</pre>
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}
