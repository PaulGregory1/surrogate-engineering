import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

const CARDS = [
  {
    to: '/flow',
    title: 'Flow Lab',
    subtitle: 'Traînée aérodynamique',
    stat: '713×',
    statLabel: 'plus rapide que le solveur LBM',
    description: 'Prédit et optimise la forme d’un corps immergé pour minimiser le coefficient de traînée, avec vérification par solveur haute-fidélité.',
    ready: true,
  },
  {
    to: '/stress',
    title: 'Stress Lab',
    subtitle: 'Contrainte mécanique',
    stat: '6.0%',
    statLabel: 'erreur pic attendue (GNN)',
    description: 'Prédit la contrainte de von Mises sur une plaque perforée et propose des positions de trous qui réduisent le pic de contrainte.',
    ready: false,
  },
]

export function Home() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-16">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="text-xs font-medium tracking-wide uppercase mb-3" style={{ color: 'var(--accent)' }}>
          Engine-Copilot
        </div>
        <h1 className="text-3xl md:text-4xl font-semibold mb-3">
          Des surrogates ML, mis entre les mains d&apos;un ingénieur.
        </h1>
        <p className="max-w-2xl text-sm md:text-base" style={{ color: 'var(--text-muted)' }}>
          Deux workflows d&apos;ingénierie — écoulement externe et plaque perforée — pilotés par des
          modèles entraînés, toujours contextualisés par leur erreur connue et vérifiables face au
          solveur haute-fidélité.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-10">
        {CARDS.map((c, i) => (
          <motion.div
            key={c.to}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.08 * i }}
          >
            <Link
              to={c.ready ? c.to : '#'}
              className={`nc-card block p-6 h-full transition ${c.ready ? 'hover:brightness-110 hover:-translate-y-0.5' : 'opacity-60 cursor-default'}`}
              style={{ transitionProperty: 'transform, filter' }}
              onClick={(e) => {
                if (!c.ready) e.preventDefault()
              }}
            >
              <div className="flex items-baseline justify-between mb-1">
                <h2 className="text-lg font-semibold">{c.title}</h2>
                {!c.ready && (
                  <span className="text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full" style={{ color: 'var(--warn)', border: '1px solid color-mix(in srgb, var(--warn) 40%, transparent)' }}>
                    bientôt
                  </span>
                )}
              </div>
              <div className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                {c.subtitle}
              </div>
              <div className="text-3xl font-bold mb-1" style={{ color: 'var(--accent)' }}>
                {c.stat}
              </div>
              <div className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                {c.statLabel}
              </div>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {c.description}
              </p>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
