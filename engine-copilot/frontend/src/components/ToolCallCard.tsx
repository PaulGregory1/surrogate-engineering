import { useState } from 'react'
import { motion } from 'framer-motion'
import { ChevronRight, CircleCheck, LoaderCircle, TriangleAlert } from 'lucide-react'
import { cn } from '@/lib/utils'

function format(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/** One agent tool invocation. Collapsed by default — the tool's *name* is the part
 *  that earns permanent space; the JSON is there for when someone doubts the answer. */
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
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={cn('rounded-md border bg-muted/40 text-xs', isError ? 'border-critical/50' : 'border-border')}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left"
      >
        <ChevronRight
          className={cn('size-3 shrink-0 text-muted-foreground transition-transform', open && 'rotate-90')}
        />
        {running ? (
          <LoaderCircle className="size-3 shrink-0 animate-spin text-ring" />
        ) : isError ? (
          <TriangleAlert className="size-3 shrink-0 text-critical" />
        ) : (
          <CircleCheck className="size-3 shrink-0 text-good" />
        )}
        <span className="font-mono text-[11px] text-foreground">{name}</span>
        <span className="ml-auto text-[10px] text-muted-foreground">
          {running ? 'en cours' : isError ? 'échec' : 'terminé'}
        </span>
      </button>

      {open && (
        <div className="space-y-1.5 border-t border-border px-2.5 py-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
          <div>
            <span className="text-foreground/70">arguments</span>
            <pre className="mt-0.5 overflow-x-auto break-all whitespace-pre-wrap">{format(input)}</pre>
          </div>
          {result !== undefined && (
            <div>
              <span className="text-foreground/70">résultat</span>
              <pre className="mt-0.5 overflow-x-auto break-all whitespace-pre-wrap">{format(result)}</pre>
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}
