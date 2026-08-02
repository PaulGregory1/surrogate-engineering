import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { ArrowRight, Grid2x2, Wind } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { usePublishChatContext } from '@/chat/ChatProvider'
import { reveal } from '@/lib/motion'

const LABS = [
  {
    to: '/flow',
    icon: Wind,
    title: 'Flow Lab',
    domain: 'AÉRODYNAMIQUE EXTERNE',
    stat: '713',
    statUnit: '×',
    statLabel: 'PLUS RAPIDE QUE LE SOLVEUR LBM',
    description:
      'Dessine un corps immergé, lis son coefficient de traînée en temps réel, puis laisse une descente de gradient à travers le surrogate chercher la forme la moins traînante — et vérifie le résultat dans le vrai solveur.',
    method: 'U-Net + contrainte de continuité',
  },
  {
    to: '/stress',
    icon: Grid2x2,
    title: 'Stress Lab',
    domain: 'MÉCANIQUE DES STRUCTURES',
    stat: '6.0',
    statUnit: '%',
    statLabel: 'ERREUR ATTENDUE SUR LE PIC',
    description:
      'Place les perçages d’une plaque sous charge, vois la carte de von Mises se recalculer à chaque déplacement, et fais chercher au modèle un placement qui abaisse le pic de contrainte.',
    method: 'MeshGraphNet, 10 passes de message',
  },
]

const PRINCIPLES = [
  {
    title: 'Chaque nombre porte son erreur',
    body: 'Une prédiction s’affiche toujours avec l’erreur mesurée de son modèle sur le jeu de test. Pas d’estimation présentée comme une mesure.',
  },
  {
    title: 'Le solveur reste accessible',
    body: 'Un clic rejoue n’importe quel design dans le solveur haute-fidélité. Le surrogate accélère la boucle, il ne la remplace pas.',
  },
  {
    title: 'Le copilote appelle les mêmes modèles',
    body: 'L’agent n’improvise pas de chiffres : il appelle les services de prédiction du backend, et chaque appel est visible dans la conversation.',
  },
]

export function Home() {
  usePublishChatContext({
    label: 'Accueil',
    lines: [],
    suggestions: [
      'Quelle est la différence entre les deux labs ?',
      'Quels modèles sont disponibles et quelle est leur erreur mesurée ?',
      'Prédis le Cd d’une forme à a2=0.05, b2=-0.02, a3=0.03, b3=0.01, a4=-0.04, b4=0, Re=80',
    ],
  })

  return (
    <>
      <PageHeader
        title="Vue d’ensemble"
        subtitle="Deux workflows d’ingénierie pilotés par des surrogates entraînés"
      />

      {/* This is an index of instruments, not a landing page — which is why nothing
          here reaches the 40px ceiling and the headline is set in serif roman at 28,
          quieter than the two measured figures it introduces. */}
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        <motion.div {...reveal()}>
          <div className="cq-k mb-4">ENGINE-COPILOT · DÉMONSTRATEUR</div>
          <h2 className="cq-value max-w-[26ch] text-[28px]">
            Des surrogates ML, mis entre les mains d’un ingénieur.
          </h2>
          <p className="cq-muted mt-5 max-w-[62ch] text-body leading-[1.7]">
            Écoulement externe et plaque perforée, résolus en millisecondes par des modèles entraînés
            sur un solveur de référence — toujours contextualisés par leur erreur connue, et
            vérifiables face à ce solveur en un clic.
          </p>
        </motion.div>

        <div className="mt-11 grid grid-cols-1 gap-4 md:grid-cols-2">
          {LABS.map((lab, i) => (
            <motion.div key={lab.to} {...reveal(0.08 * (i + 1))}>
              <Link
                to={lab.to}
                className="group block h-full rounded-[16px] outline-none focus-visible:shadow-[var(--focus)]"
              >
                <div className="flex h-full flex-col gap-5 rounded-[16px] border border-line2 bg-white p-5 shadow-card transition-colors duration-200 group-hover:border-ink3">
                  <div className="flex items-center gap-2.5">
                    <lab.icon className="size-3.5 shrink-0 text-burnt" />
                    <div className="cq-title">{lab.title}</div>
                    <span className="cq-k ml-auto">{lab.domain}</span>
                  </div>

                  <div>
                    {/* The ratio is the headline of the whole card — set big and
                        bold on purpose, above the charte's 40px ceiling. */}
                    <div className="cq-hero">
                      {lab.stat}
                      <i>{lab.statUnit}</i>
                    </div>
                    <div className="cq-k mt-3 text-burnt">{lab.statLabel}</div>
                  </div>

                  <p className="cq-muted text-body leading-[1.7]">{lab.description}</p>

                  <div className="mt-auto flex items-center gap-3 border-t border-line pt-3">
                    <span className="cq-num text-k tracking-normal text-ink3">{lab.method}</span>
                    <ArrowRight className="ml-auto size-3.5 text-ink3 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-ink" />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        <motion.div className="mt-12" {...reveal(0.24)}>
          <div className="cq-k">CE QUI TIENT LA DÉMO</div>
          <div className="mt-5 grid grid-cols-1 gap-6 md:grid-cols-3">
            {PRINCIPLES.map((p) => (
              // Ink rule over the heading, hairline everywhere else — the same
              // contrast jump that holds a data table together without stripes.
              <div key={p.title} className="border-t border-ink pt-3">
                <div className="text-body font-medium">{p.title}</div>
                <p className="cq-muted mt-2 text-body leading-[1.7]">{p.body}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </>
  )
}
