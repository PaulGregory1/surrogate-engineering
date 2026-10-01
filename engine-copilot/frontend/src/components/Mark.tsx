import { cn } from '@/lib/utils'

/**
 * The mark, everywhere it appears: an `<img>` on `/mark.svg`, which
 * `docs/make_logo.py` writes — never inlined, never hand-drawn, so regenerating the
 * lattice propagates to the nav, the copilot and the favicon in one command.
 *
 * Decorative by default. Every call site but one sets the mark beside the word it
 * stands for, and a screen reader announcing "Copilot" twice is noise; pass `label`
 * only where the mark carries the identity alone — the nav rail once it collapses.
 */
export function Mark({ className, label }: { className?: string; label?: string }) {
  return (
    <img
      src="/mark.svg"
      alt={label ?? ''}
      title={label}
      aria-hidden={label ? undefined : true}
      className={cn('shrink-0 select-none', className)}
      draggable={false}
    />
  )
}
