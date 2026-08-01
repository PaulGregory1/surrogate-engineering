import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { ArrowRight, Grid2x2, Wind } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { usePublishChatContext } from '@/chat/ChatProvider'

const LABS = [
  {
    to: '/flow',
    icon: Wind,
    title: 'Flow Lab',
    domain: 'Aérodynamique externe',
    stat: '713×',
    statLabel: 'plus rapide que le solveur LBM',
    description:
      'Dessine un corps immergé, lis son coefficient de traînée en temps réel, puis laisse une descente de gradient à travers le surrogate chercher la forme la moins traînante — et vérifie le résultat dans le vrai solveur.',
    method: 'U-Net + contrainte de continuité',
  },
  {
    to: '/stress',
    icon: Grid2x2,
    title: 'Stress Lab',
    domain: 'Mécanique des structures',
    stat: '6.0%',
    statLabel: 'erreur attendue sur le pic',
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
      <PageHeader title="Vue d’ensemble" subtitle="Deux workflows d’ingénierie pilotés par des surrogates entraînés" />

      <div className="mx-auto w-full max-w-5xl px-6 py-10">
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <Badge variant="outline" className="mb-4 font-normal text-muted-foreground">
            Engine-Copilot
          </Badge>
          <h2 className="max-w-2xl text-3xl font-semibold tracking-tight text-balance md:text-4xl">
            Des surrogates ML, mis entre les mains d’un ingénieur.
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted-foreground md:text-base">
            Écoulement externe et plaque perforée, résolus en millisecondes par des modèles entraînés
            sur un solveur de référence — toujours contextualisés par leur erreur connue, et
            vérifiables face à ce solveur en un clic.
          </p>
        </motion.div>

        <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-2">
          {LABS.map((lab, i) => (
            <motion.div
              key={lab.to}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.06 * (i + 1) }}
            >
              <Link to={lab.to} className="group block h-full rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
                <Card className="h-full gap-4 transition group-hover:border-ring/40 group-hover:bg-card/80">
                  <CardContent className="flex h-full flex-col gap-4">
                    <div className="flex items-center gap-2.5">
                      <div className="flex size-8 items-center justify-center rounded-md border border-border bg-muted/50">
                        <lab.icon className="size-4 text-ring" />
                      </div>
                      <div className="leading-tight">
                        <div className="font-semibold">{lab.title}</div>
                        <div className="text-[11px] text-muted-foreground">{lab.domain}</div>
                      </div>
                      <ArrowRight className="ml-auto size-4 text-muted-foreground transition group-hover:translate-x-0.5 group-hover:text-foreground" />
                    </div>

                    <div>
                      <div className="text-4xl font-semibold tracking-tight">{lab.stat}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">{lab.statLabel}</div>
                    </div>

                    <p className="text-sm leading-relaxed text-muted-foreground">{lab.description}</p>

                    <div className="mt-auto pt-1">
                      <Separator className="mb-3" />
                      <div className="font-mono text-[11px] text-muted-foreground">{lab.method}</div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            </motion.div>
          ))}
        </div>

        <div className="mt-12">
          <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Ce qui tient la démo</h3>
          <div className="mt-4 grid grid-cols-1 gap-6 md:grid-cols-3">
            {PRINCIPLES.map((p) => (
              <div key={p.title} className="border-t border-border pt-3">
                <div className="text-sm font-medium">{p.title}</div>
                <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{p.body}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
