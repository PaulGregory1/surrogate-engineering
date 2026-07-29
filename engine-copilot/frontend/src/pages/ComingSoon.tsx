import { motion } from 'framer-motion'

export function ComingSoon({ title, milestone, description }: { title: string; milestone: string; description: string }) {
  return (
    <div className="mx-auto max-w-2xl px-6 py-24 text-center">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="text-xs uppercase tracking-wide mb-3" style={{ color: 'var(--warn)' }}>
          {milestone}
        </div>
        <h1 className="text-2xl font-semibold mb-3">{title}</h1>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {description}
        </p>
      </motion.div>
    </div>
  )
}
