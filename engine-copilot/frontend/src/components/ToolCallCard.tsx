import { motion } from 'framer-motion'

export function ToolCallCard({ name, input, result, isError, running }: { name: string; input: unknown; result?: unknown; isError?: boolean; running: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      transition={{ duration: 0.2 }}
      className="nc-card px-3 py-2 text-xs font-mono overflow-hidden"
      style={{ borderColor: isError ? 'var(--warn)' : 'var(--border)' }}
    >
      <div className="flex items-center gap-2">
        {running ? (
          <span className="h-2.5 w-2.5 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
        ) : (
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: isError ? 'var(--warn)' : 'var(--ok)' }} />
        )}
        <span style={{ color: 'var(--accent)' }}>{name}</span>
        <span style={{ color: 'var(--text-muted)' }}>({JSON.stringify(input)})</span>
      </div>
      {result !== undefined && (
        <div className="mt-1 pl-4" style={{ color: 'var(--text-muted)' }}>
          → {typeof result === 'string' ? result : JSON.stringify(result)}
        </div>
      )}
    </motion.div>
  )
}
