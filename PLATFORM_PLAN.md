# Engine-Copilot — Plan d'exécution pour Claude Code

> **Document historique — lis-le comme tel.** Ce brief a été écrit quand la plateforme
> visait deux labs. Il décrit toujours le Stress Lab (§3 contrats d'API, §5.3 outils,
> jalon M4) : c'est volontairement **non réécrit**, pour ne pas falsifier ce qui a été
> demandé et livré à l'époque. Sur `main`, le Stress Lab n'existe plus — il vit sur la
> branche `stress-lab`, où ce plan décrit encore l'arbre exact. Pour l'état réel de
> `main`, la référence est `KNOWLEDGE-GRAPH.md`, pas ce fichier.

**Objectif** : construire une mini-plateforme web locale qui expose les deux surrogates
(`flow-surrogate`, `stress-gnn`) derrière une interface graphique interactive, orchestrée
par un agent Claude (**Claude Agent SDK**, branché sur l'abonnement Claude Code local —
pas de clé API) qui traduit des intentions ingénieur en appels aux modèles et à
l'optimiseur.

Ce document est un **brief d'exécution**. Tu (Claude Code) es libre sur l'implémentation
mais tu dois respecter les jalons, les contrats d'API et les critères d'acceptation.
Ne pars **jamais** en refactor des dépôts `flow-surrogate/` et `stress-gnn/` : traite-les
comme des **librairies** que la plateforme importe.

---

## 0. Contexte et positionnement

Cette plateforme sert de démo pour une candidature Forward Deployed Engineer.
Elle doit donc démontrer, dans l'ordre :

1. **Ownership produit** : rendre les modèles utilisables par un ingénieur qui ne
   connaît pas PyTorch.
2. **Rigueur ML** : les prédictions surrogate sont toujours **contextualisées** par
   leur erreur connue (numbers from `RESULTS.md`) et vérifiables via le solveur haute
   fidélité (LBM / FEM).
3. **Modernité outillage** : un agent LLM qui *orchestre* du code métier via
   **tool use**, pas un chatbot cosmétique.
4. **Qualité produit** : l'interface doit être irréprochable visuellement — voir
   §4.1. Le test : "si je montre ça en visio à un prospect, il pense produit,
   pas prototype."

**Non-goals** (à refuser explicitement si tu es tenté d'y aller) :
- pas d'authentification, pas de base de données, pas de déploiement cloud ;
- pas de génération de code Python à la volée par le LLM ;
- pas de réentraînement des modèles depuis l'UI ;
- pas de refactor des projets ML sous-jacents ;
- pas de 3D — les deux projets sont 2D, on reste 2D ;
- pas de SDK API `anthropic` + clé API pour l'agent — c'est le Claude Agent SDK
  branché sur l'abonnement Claude Code, ou rien (voir §5).

---

## 1. Architecture globale

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend  (Vite + React + TypeScript + Plotly.js + Tailwind)    │
│  ─ ShapeEditor (canvas SVG, drag des coeffs Fourier)             │
│  ─ FieldViewer (Plotly heatmap, colorbar physique)               │
│  ─ FlowAnimation (frames LBM ou surrogate, boucle streamlines)   │
│  ─ OptimizerPanel (courbes de convergence en live)               │
│  ─ ChatPanel (agent Claude, avec affichage des tool calls)       │
└─────────────────┬──────────────────────────────────┬─────────────┘
                  │ REST + SSE (stream optim)        │
                  ▼                                  ▼
┌──────────────────────────────────────┐  ┌────────────────────────┐
│  Backend FastAPI (engine_copilot/)   │  │  Agent Claude          │
│  ─ /api/flow/predict                 │◄─┤  (Claude Agent SDK,    │
│  ─ /api/flow/optimize (SSE)          │  │   via CLI local,       │
│  ─ /api/flow/verify                  │  │   abonnement Claude    │
│  ─ /api/flow/simulate (LBM frames)   │  │   Code, pas d'API key) │
│  ─ /api/stress/predict               │  └────────────────────────┘
│  ─ /api/stress/optimize              │
│  ─ /api/report/generate              │
└─────────────┬────────────────────────┘
              │ import
              ▼
┌──────────────────────────────────────┐
│  flowsurrogate.*  et  stressgnn.*    │  ← INTOUCHABLES
│  (chargés via load_model, runs/)     │
└──────────────────────────────────────┘
```

**Stack imposée** (raisons entre parenthèses) :

| Couche | Choix | Pourquoi |
|---|---|---|
| Backend | **FastAPI** + `uvicorn` | async, SSE natif, types Pydantic → tool schemas Claude |
| Frontend | **Vite + React 18 + TypeScript** | dev rapide, écosystème plot mature |
| Plots 2D scientifiques | **Plotly.js** (`react-plotly.js`) | heatmaps interactifs, colorbars propres |
| Overlay / streamlines | **SVG natif** au-dessus du canvas Plotly | contrôle total sur le contour + traçage particules |
| Style | **TailwindCSS** + `shadcn/ui` | look pro sans temps design |
| Micro-interactions | **Framer Motion** | transitions de panneaux, morphing de forme, nombres qui s'animent — ce qui fait "produit" plutôt que "démo académique" |
| Agent LLM | **Claude Agent SDK** (`claude-agent-sdk`, Python) — **PAS** le SDK API `anthropic` | tourne sur le CLI Claude Code local, authentifié via `claude login` (abonnement Claude Code) → aucune clé API, aucune facturation à la requête |
| Streaming | **SSE** (Server-Sent Events) | plus simple que WebSocket, unidirectionnel suffit |
| Package manager frontend | `pnpm` | rapide, lockfile propre |

> **Décision-clé #1** : l'agent tourne via le **Claude Agent SDK**, qui pilote le
> binaire `claude` (Claude Code CLI) déjà installé et authentifié sur cette machine.
> Pas de `ANTHROPIC_API_KEY`, pas de `.env` secret : l'auth est celle de la session
> `claude login`. C'est un produit différent du SDK API classique (`anthropic`) — ne
> pas les confondre à l'implémentation. Doc à jour : `https://code.claude.com/docs/en/agent-sdk`
> (vérifier la surface d'API exacte avant d'écrire `agent.py`).
>
> **Décision-clé #2** : chaque écran doit passer le test "vendable à un client" —
> voir §4.1 pour ce que ça implique concrètement (animations, skeletons, thème
> clair/sombre, export).

---

## 2. Structure du dépôt

Nouveau dossier **à créer à la racine du workspace** :

```
surrogate-engineering/
├── flow-surrogate/          # inchangé
├── stress-gnn/              # inchangé
└── engine-copilot/          # ← nouveau
    ├── README.md
    ├── pyproject.toml       # pas de .env secret — voir §7.5, l'auth passe par `claude login`
    ├── backend/
    │   ├── engine_copilot/
    │   │   ├── __init__.py
    │   │   ├── main.py                  # FastAPI app
    │   │   ├── config.py                # paths vers ../flow-surrogate, ../stress-gnn
    │   │   ├── schemas.py               # Pydantic models = tool schemas
    │   │   ├── flow_service.py          # wrappers autour de flowsurrogate.*
    │   │   ├── stress_service.py        # wrappers autour de stressgnn.*
    │   │   ├── lbm_frames.py            # LBM instrumenté qui yield frames
    │   │   ├── streamlines.py           # tracés particules pour l'animation
    │   │   ├── report.py                # /api/report/generate
    │   │   ├── agent.py                 # boucle tool use Claude
    │   │   └── tools.py                 # définition des tools exposés au LLM
    │   └── tests/
    │       ├── test_flow_service.py
    │       ├── test_stress_service.py
    │       └── test_agent_tools.py
    └── frontend/
        ├── package.json
        ├── vite.config.ts
        ├── tsconfig.json
        ├── tailwind.config.ts
        ├── index.html
        └── src/
            ├── main.tsx
            ├── App.tsx
            ├── api.ts                   # client typé du backend
            ├── theme.ts                 # colormaps (viridis, RdBu_r, coolwarm)
            ├── components/
            │   ├── ShapeEditor.tsx
            │   ├── FieldViewer.tsx
            │   ├── FlowAnimation.tsx
            │   ├── StreamlineOverlay.tsx
            │   ├── OptimizerPanel.tsx
            │   ├── ConvergencePlot.tsx
            │   ├── StressViewer.tsx
            │   ├── HoleEditor.tsx
            │   ├── ChatPanel.tsx
            │   ├── ToolCallCard.tsx
            │   ├── MetricsBadge.tsx
            │   ├── AnimatedNumber.tsx    # transition numérique sur Cd/erreurs/peak vM
            │   ├── Skeleton.tsx          # placeholders de chargement, réutilisé partout
            │   └── ThemeToggle.tsx       # dark (défaut) / light
            └── pages/
                ├── Home.tsx              # onglet 0, hero + les deux workflows en cartes
                ├── FlowLab.tsx           # onglet 1
                ├── StressLab.tsx         # onglet 2
                └── ChatOrchestrator.tsx  # onglet 3, agent transverse
```

### `engine-copilot/pyproject.toml` (minimal)

```toml
[project]
name = "engine-copilot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "pydantic>=2.7",
  "claude-agent-sdk",     # pas "anthropic" — voir §1, décision-clé #1
  "numpy",
  "torch",
  "sse-starlette>=2.1",
  # dev deps installées séparément via pip -e ../flow-surrogate ../stress-gnn
]

[tool.pytest.ini_options]
testpaths = ["backend/tests"]
```

Installation :
```
pip install -e ../flow-surrogate -e ../stress-gnn -e .
npm install -g @anthropic-ai/claude-code   # le SDK pilote ce binaire, il doit être sur le PATH
claude login                               # une fois — authentifie via l'abonnement Claude Code
```

**Vérifie avant M5** que `claude --version` répond et que la session est authentifiée
(commande de statut exacte à confirmer dans la doc du SDK). Si l'utilisateur n'a
jamais fait `claude login` sur cette machine, arrête-toi et demande-le lui.

---

## 3. Contrats d'API backend (source de vérité)

Toutes les routes retournent JSON sauf mention contraire. Les erreurs suivent le
schéma FastAPI par défaut. **Chaque route est aussi un tool exposé à Claude** (§ 5).

### 3.1 Flow

#### `POST /api/flow/predict`
Prédire les champs (u, v, p) et C_d pour une forme donnée.

Request body (Pydantic `FlowPredictRequest`) :
```json
{
  "coeffs": [0.05, -0.02, 0.03, 0.01, -0.04, 0.00],  // 6 floats, ordre a2,b2,a3,b3,a4,b4
  "reynolds": 80.0,
  "model": "unet",                                    // "unet" | "fno" | "unet-phys" | "pod-rbf"
  "target_area": null                                 // null = aire médiane du train set
}
```

Response :
```json
{
  "cd_surrogate": 1.72,
  "cd_error_pct_expected": 1.03,        // tiré de RESULTS.md pour le modèle
  "fields": {                            // downscaled à 96x48 pour le transport JSON
    "u": [[...]], "v": [[...]], "p": [[...]]
  },
  "mask": [[...]],                       // 1 = fluide, 0 = solide
  "domain": {"nx": 192, "ny": 96},
  "boundary_xy": [[...x], [...y]],       // contour de la forme, 400 points
  "target_area": 683.2,
  "latency_ms": 4.2
}
```

Implementation notes :
- charge les modèles une seule fois au startup (cache singleton dans `flow_service.py`) ;
- retourne les champs **en unités physiques** (utilise `Normalizer.decode_fields`) ;
- le modèle par défaut est `unet-phys` (meilleur physiquement).

#### `POST /api/flow/optimize` → **SSE stream**
Lance l'optimisation Adam à travers le surrogate et **stream chaque itération**.

Request :
```json
{
  "reynolds": 80.0,
  "n_starts": 8,
  "iters": 200,
  "model": "unet-phys",
  "seed": 0
}
```

Le stream émet des événements SSE typés :

```
event: iteration
data: {"it": 42, "cd_best": 1.51, "cd_mean": 1.63, "coeffs_best": [...], "boundary_xy": [[...],[...]]}

event: done
data: {"final_coeffs": [...], "history": {...}, "cd_surrogate_final": 1.436, "target_area": 683.2}
```

L'événement `iteration` DOIT contenir `boundary_xy` recalculé — c'est ce qui permet
au frontend d'**animer la déformation de la forme en direct**.

Implementation : ne pas ré-écrire `optimise()`. À la place, factorise le corps de la
boucle dans `flow_service.py` :

```python
def optimise_streaming(cfg: OptConfig, ...) -> Iterator[dict]:
    # même boucle Adam que flowsurrogate.optimize.optimise(),
    # mais `yield` à chaque itération au lieu d'accumuler dans history.
```

Fréquence : yield toutes les 2 itérations, pas toutes (sinon SSE saturé).

#### `POST /api/flow/verify`
Prend un jeu de coeffs et lance **le vrai LBM** pour vérifier le C_d prédit.

Request :
```json
{
  "coeffs": [...],
  "target_area": 683.2,
  "reynolds": 80.0,
  "steps": 9000
}
```

Response :
```json
{
  "cd_lbm": 1.436,
  "cd_surrogate": 1.598,
  "gap_pct": 11.3,
  "fields": {"u": [[...]], "v": [[...]], "p": [[...]]},
  "wall_time_s": 12.4
}
```

C'est le tool le plus important à exposer à Claude : il permet à l'agent de dire
« la surrogate se trompe de X %, je vérifie ».

#### `POST /api/flow/simulate` → **SSE stream de frames**
Le clou du spectacle visuel. Lance le LBM et **stream N frames intermédiaires** pour
animer le développement du sillage.

Request :
```json
{
  "coeffs": [...],
  "target_area": 683.2,
  "reynolds": 80.0,
  "n_frames": 60,
  "steps": 6000
}
```

Événements SSE :
```
event: frame
data: {"t_step": 100, "u": [[...]], "v": [[...]], "vorticity": [[...]], "frame_index": 0}
...
event: done
data: {"cd_final": 1.72}
```

Implementation : dans `lbm_frames.py`, sous-classer / wrapper `LBMSolver.run` pour
extraire l'état macroscopique tous les `steps // n_frames` pas. Downscale à 96×48
avant sérialisation. Calculer la vorticité $\omega = \partial_x v - \partial_y u$
côté serveur (une différence finie centrée), c'est ce qui donne le rendu visuel le
plus spectaculaire (rouleaux de Von Karman colorés en RdBu_r).

### 3.2 Stress

#### `POST /api/stress/predict`
Request :
```json
{
  "holes": [[0.5, 0.5, 0.12], [1.3, 0.4, 0.10]],   // (x, y, r) par trou
  "load_angle_deg": -90.0,
  "model": "gnn-mp10-glob"                          // meilleur par défaut
}
```

Response :
```json
{
  "peak_vm_surrogate": 22.4,
  "peak_error_pct_expected": 6.0,
  "field": {                                        // rendu en grille 128x64 pour l'UI
    "vm": [[...]],
    "extent": [0, 2, 0, 1]                          // xmin, xmax, ymin, ymax
  },
  "mesh": {
    "nodes": [[x, y], ...],                         // coordonnées nodales originales
    "triangles": [[i, j, k], ...],                  // pour un rendu mesh optionnel
    "vm_nodal": [...]                               // vM par nœud
  },
  "peak_location": [x, y],
  "latency_ms": 8.1
}
```

#### `POST /api/stress/optimize`
Objectif : minimiser la contrainte pic sous contrainte d'aire retirée ≥ seuil (ou
maximiser l'aire retirée sous contrainte $\sigma_{max}$ ≤ seuil).

Comme `stress-gnn` n'a **pas** encore d'optimiseur différentiable écrit, deux options :

**Option A (recommandée)** : optimisation **CMA-ES ou random search** sur les
(x, y, r) des trous, en évaluant à chaque coup le surrogate GNN. C'est honnête, ça
tourne en quelques secondes, et le rapport peut expliquer pourquoi (le GNN n'est
pas trivialement différentiable par rapport à la position des trous — cf. §7 de
`stress-gnn/BACKGROUND.md`).

**Option B** (bonus) : gradient sur les rayons uniquement, positions fixes.

Implémente A. Stream SSE identique aux autres optim : événements `iteration` avec
`peak_vm_current`, `best_holes`, etc.

### 3.3 Rapport transverse

#### `POST /api/report/generate`
Génère un rapport Markdown en langage ingénieur à partir d'un résultat d'optim.

Request :
```json
{
  "project": "flow" | "stress",
  "context": { ...réponse d'un précédent /optimize... }
}
```

Response :
```json
{
  "markdown": "## Résumé exécutif\n\nLa forme optimisée réduit le C_d de..."
}
```

Cette route **appelle Claude en interne** avec un prompt qui reçoit les numbers en
JSON et retourne du markdown. Pas de tool use ici, juste une génération.

### 3.4 Agent transverse

#### `POST /api/agent/chat` → **SSE stream**
Boucle tool use complète. Reçoit un message utilisateur + historique, retourne un
stream d'événements typés (texte, tool_use, tool_result, fin de tour). Détail en §5.

---

## 4. Frontend — spec des composants

### 4.1 Standard de qualité visuelle — "je vends ça à un client"

Le bar n'est pas "ça marche", c'est "un prospect qui voit ça en démo pense
produit, pas prototype". Concrètement :

- **Aucun état cru** : skeletons pendant le chargement (composant `Skeleton.tsx`
  réutilisé partout), transitions Framer Motion (150-250ms, `easeOut`) sur tout
  changement de panneau ou de valeur, jamais de saut brut ni de flash de contenu.
- **Chiffres qui comptent, pas qui sautent** : `AnimatedNumber.tsx` anime la
  transition sur $C_d$, erreurs %, peak vM plutôt que remplacer le texte
  instantanément.
- **Micro-interactions** : hover states partout où c'est cliquable, focus rings
  clavier visibles, boutons avec état "en cours" inline (spinner, pas un overlay
  qui bloque toute la page).
- **Export soigné** : bouton export PNG (`Plotly.toImage`) sur chaque graphique ;
  CSS `@media print` pensé dès le début pour le rapport, pas rajouté à la fin.
- **Une vraie page d'accueil** (`Home.tsx`) : hero bref + les deux workflows en
  cartes cliquables avec un chiffre clé chacune ("713× plus rapide que LBM",
  "6.0% d'erreur pic") plutôt que de tomber directement dans un formulaire vide.
- **Thème clair fonctionnel en plus du dark mode** (`ThemeToggle.tsx`) — un
  prospect qui partage son écran en journée n'a pas envie d'un fond noir.

**Avant d'écrire le moindre composant de graphique**, invoque le skill `dataviz`
de Claude Code — il donne la méthode de choix de couleurs (validée par script,
pas à l'œil), les specs de marks, et les règles d'accessibilité. Les colormaps
scientifiques ci-dessous (viridis/RdBu_r/magma) sont déjà de bons choix — garde-les —
mais applique la méthode `dataviz` pour tout le reste de la palette produit
(accents, statuts, dark/light).

Objectif visuel : **sobre, scientifique, dark mode par défaut, avec un thème
clair propre**. Références : observablehq, Linear, Plotly dark theme — pas
"tableau de bord Bootstrap".

- Fond : `#0b0f14` (background), `#141a22` (cards), `#1f2937` (borders)
- Accent : `#22d3ee` (cyan) pour actions, `#f59e0b` (amber) pour warnings surrogate,
  `#34d399` (vert) pour "vérifié / dans le training set"
- Colormaps :
  - vitesse `u` : `viridis`
  - pression `p` : `RdBu_r`
  - vorticité `omega` : `RdBu_r` (symétrique)
  - stress vM : `magma`
  - erreurs : `coolwarm` symétrique

Chaque heatmap Plotly DOIT avoir sa colorbar avec l'unité physique en label
(« $u$ [–] », « $\\sigma_{vM}$ [–] »).

### 4.2 `ShapeEditor.tsx` (Flow Lab)

Contrôle interactif de la forme. Deux modes :

1. **Sliders** : 6 sliders (a2, b2, a3, b3, a4, b4) avec bornes ±0.16 (respect
   `COEFF_RANGE`). Slider Reynolds (10-200). Affichage temps réel du contour.
2. **Drag direct** : le contour est rendu en SVG, l'utilisateur attrape un point et
   la forme se déforme. Sous le capot : projection du drag sur la base Fourier
   $(a_k, b_k)$ tronquée à k=2,3,4 (moindres carrés). Reset et randomize buttons.

Le contour est calculé côté **frontend** en JS (répliquer `radius()` de
`geometry.py` — c'est 5 lignes). Ça évite un round-trip pour chaque frame de drag.
La prédiction (`/api/flow/predict`) est déclenchée en **debounce 300ms** sur
changement de coeffs.

### 4.3 `FieldViewer.tsx` (Flow Lab)

Trois panneaux empilés horizontalement : $u$, $v$, $p$. Chacun un `Plotly.Heatmap`
avec :
- `zsmooth: 'best'` pour un rendu lisse ;
- masque solide en gris foncé (utiliser `null` dans la matrice `z`) ;
- overlay SVG du contour de la forme (superposé, `pointer-events: none`) ;
- annotations : $C_d$ affiché en haut à droite avec badge coloré (vert si erreur
  attendue < 2 %, ambre sinon).

### 4.4 `FlowAnimation.tsx` (Flow Lab)

Composant vedette pour le côté visuel demandé. Deux sous-modes :

1. **Streamlines temps réel** sur le champ prédit (`/api/flow/predict`) :
   - Sème 40 particules sur la face gauche du domaine ;
   - Chaque frame (`requestAnimationFrame`) : intègre $(u, v)$ interpolé bilinéaire
     par un pas RK4 explicite ; les particules qui sortent réapparaissent à gauche ;
   - Rendu en SVG comme des traînées (segments avec opacité décroissante sur les
     N derniers pas), au-dessus de la heatmap de vorticité ;
   - Cadence 30 fps, indépendant de la prédiction.
2. **Animation LBM authentique** via `/api/flow/simulate` SSE :
   - Bouton « Simulate », qui lance 60 frames LBM streamées ;
   - Chaque frame reçue met à jour la heatmap de vorticité ; le développement du
     sillage instationnaire (Von Karman à Re=80) est visible.

Le mode 1 est instantané (rendu client), le mode 2 est « vrai » (2-15 s selon
`steps`). Les deux doivent exister — le premier pour l'exploration, le second pour
la vérification.

### 4.5 `OptimizerPanel.tsx` + `ConvergencePlot.tsx` (Flow Lab)

Écran principal quand l'utilisateur clique « Optimize ». Layout : 60/40.

- **Gauche** : deux `ShapeEditor` en mode read-only côte à côte, labellés
  **« Design initial »** et **« Design optimisé »**. La forme optimisée
  **se déforme en direct** au fur et à mesure que les événements SSE arrivent
  (à partir des `boundary_xy` du stream). C'est exactement ce que tu voulais :
  « on voit la forme avant/après avec un morphing ».
- **Droite** : `ConvergencePlot` — courbe `cd_best` et `cd_mean` en fonction de
  l'itération, mise à jour en live (Plotly `extendTraces`).
- **En bas** : bouton **« Verify with solver »** → appelle `/api/flow/verify` sur
  le meilleur design final, affiche $C_d^{surrogate}$ vs $C_d^{LBM}$ avec le gap
  et l'interprétation textuelle (« la surrogate se trompe de 11 % ; ça se
  résorbe avec un round d'active learning, cf. RESULTS.md »).

### 4.6 `HoleEditor.tsx` + `StressViewer.tsx` (Stress Lab)

- `HoleEditor` : plaque 2:1 rendue en SVG. L'utilisateur clique pour ajouter un
  trou, drag pour déplacer, molette pour redimensionner. Contraintes de
  `stressgnn.geometry.is_valid()` répliquées en JS (marges, gap). Slider
  angle de charge.
- `StressViewer` : heatmap `magma` de $\sigma_{vM}$, avec une **étoile animée**
  (pulse CSS) sur `peak_location` — c'est le point qui compte pour la décision.
- Badge « peak error attendu : ±6.0 % » calé sur le modèle sélectionné.
- Bouton « Compare with coarse FEM » qui affiche côte à côte le champ surrogate
  et la coarse FEM baseline (à récupérer via une variante de l'endpoint predict).

### 4.7 `ChatPanel.tsx` + `ToolCallCard.tsx` (Chat Orchestrator)

Chat classique à droite, avec un **affichage riche des tool calls** au milieu du
fil de conversation :

```
User: Optimise la forme pour un Reynolds de 120 et dis-moi si c'est fiable.
Claude: [ToolCallCard: flow_optimize(reynolds=120, n_starts=8, iters=200)]
        → stream 200 iterations, cd_best 2.14 → 1.68
[ToolCallCard: flow_verify(coeffs=[...])]
        → cd_lbm=1.71, gap=1.8%
Voici le résultat : Cd optimisé 1.68 (surrogate), 1.71 (solveur), gap 1.8 %.
À Re=120 tu es dans le training set (max Re=150), la fiabilité est bonne.
Je recommande la forme ci-dessous. [inline preview de la forme]
```

Le composant `ToolCallCard` affiche :
- nom du tool + arguments (JSON collapsé) ;
- statut : « running… » avec spinner, puis résumé du résultat (2 lignes max) ;
- lien « voir dans le lab » qui bascule vers l'onglet Flow ou Stress avec l'état
  pré-rempli.

Prompts suggérés en bas du chat (chips cliquables) :
- « Optimise ma forme pour minimiser la traînée »
- « Cette poche de pression est-elle problématique ? »
- « Où sont les zones critiques de contrainte sur cette pièce ? »
- « Compare la fiabilité du GNN et du CNN sur mon design »

---

## 5. Agent Claude — spec détaillée

**Changement majeur vs une intégration API classique** : l'agent utilise le
**Claude Agent SDK** (`claude-agent-sdk`), pas le SDK API `anthropic` avec une
clé facturée à l'usage. Ce SDK pilote le CLI Claude Code local, authentifié via
`claude login` — donc branché sur l'abonnement Claude Code de l'utilisateur, sans
facturation séparée. Trois conséquences pratiques :

1. **Pas de `client.messages.create()` ni de boucle tool-use manuelle** : le
   Claude Agent SDK a son propre agentic loop intégré (le même moteur que Claude
   Code) — ne réécris pas de `while` sur `tool_use`/`tool_result`, configure des
   tools et laisse le SDK gérer l'appel, l'exécution, et le renvoi du résultat.
2. **Les tools sont des fonctions Python décorées**, exposées via un serveur MCP
   *in-process* déclaré au SDK (typiquement `create_sdk_mcp_server` + un champ
   `mcp_servers` dans les options) — le SDK dérive le schéma de la signature.
3. **L'auth n'a rien à voir avec `ANTHROPIC_API_KEY`** — vérifie `claude login`
   au démarrage du backend (§7.5), pas de `.env` secret.

**Avant d'écrire `agent.py` et `tools.py`** : WebFetch
`https://code.claude.com/docs/en/agent-sdk` (section Python) et vérifie le nom
exact du décorateur de tool, la signature de `create_sdk_mcp_server`, les champs
de `ClaudeAgentOptions` (`mcp_servers`, `allowed_tools`, `system_prompt`,
`permission_mode`, `model`), et la classe de conversation multi-tours streamée
(historiquement `ClaudeSDKClient` avec `.query()` / `.receive_response()`). Le
squelette ci-dessous décrit l'**architecture attendue**, pas la syntaxe finale —
adapte-le une fois la doc vérifiée.

### 5.1 Modèle et paramètres

- Modèle : laisse le SDK utiliser le modèle par défaut de la session Claude Code,
  sauf si `ClaudeAgentOptions` permet un override explicite — dans ce cas
  `claude-opus-5`, avec repli sur le défaut du CLI si l'override n'est pas
  supporté par la version installée.
- Streaming activé — chaque message/bloc reçu du SDK est repackagé en événement
  SSE vers le frontend.
- Cap le nombre de tours d'outils à **6** (paramètre du SDK s'il existe, sinon
  compteur manuel dans `agent.py`).
- **`permission_mode`** : les 8 tools sont des wrappers propres autour de nos
  propres endpoints (pas de shell, pas d'écriture disque arbitraire) — configure
  le SDK pour auto-approuver uniquement ces tools via `allowed_tools`, et
  **n'active aucun tool intégré** (pas de `Bash`/`Write`/`Read` génériques —
  l'agent n'a besoin que de son propre domaine).

### 5.2 System prompt (à copier tel quel, adapte les numbers si tu as mieux)

```
You are Engine-Copilot, an assistant embedded in a mechanical engineering design tool.
You help an engineer analyse and optimise 2-D shapes for two workflows:

  1. External flow around a bluff body (drag minimisation at constant area).
  2. Perforated bracket under load (peak von Mises stress).

You have access to trained surrogate models and to the underlying high-fidelity
solvers (LBM for flow, FEM for stress). ALWAYS use the tools — never guess numbers.

Ground rules:
- When the user asks for a prediction or an optimisation, CALL the appropriate tool.
  Do not invent numeric values.
- After every surrogate call, communicate the expected error (percentages you get
  back in the response), not just the point estimate.
- After every optimisation, propose to VERIFY with the high-fidelity solver, and
  explain the gap when it comes back.
- Speak like an engineer: use "drag coefficient", "wake", "stress concentration",
  "ligament", not "the number went down".
- Keep responses short (3-6 sentences) unless the user asks for a report.
- Respond in the language the user writes in (French or English).

Known model quality (from RESULTS.md, do not fabricate other numbers):
  Flow U-Net + continuity: field rel L2 = 0.078, Cd mean error 1.03%,
                           worst case 4.2%, 713x faster than LBM.
  Flow POD + kernel ridge: Cd mean error 3.78%, worst 13.8%.
  Stress GNN+global:       peak error 6.0%, field rel L2 0.042.
  Stress U-Net (grid):     peak error 5.2%, but fails off-distribution (88%).

Off-distribution is Re > 150 for flow, or hole count > 4 / atypical loads for stress.
Warn the user when they push there.
```

### 5.3 Tools exposés (fonctions Python, à définir dans `tools.py`)

**IMPORTANT** : chaque tool est une fonction Python décorée dont la
signature/schéma reflète le body Pydantic de l'endpoint FastAPI correspondant.
Le corps de la fonction appelle directement la fonction Python du service (tout
tourne dans le même process) — ne duplique pas la logique métier dans le tool.

Liste des tools (inchangée dans son contenu, sa forme change — plus de JSON
schema écrit à la main) :

1. `flow_predict` → `POST /api/flow/predict` (sans SSE)
2. `flow_optimize` → wrappe `/api/flow/optimize` en collectant le stream jusqu'au
   `done` (le LLM ne consomme pas le stream lui-même) ; retourne un résumé compact
   (init/final Cd, history downsamplée à 20 points).
3. `flow_verify` → `POST /api/flow/verify`
4. `flow_simulate` → idem, résumé après stream complet
5. `stress_predict` → `POST /api/stress/predict`
6. `stress_optimize` → idem flow_optimize
7. `get_model_card` → renvoie les métriques d'un modèle (lecture du JSON dans
   `runs/*_history.json` et `results/metrics.json`). Utile quand l'utilisateur
   demande « quelle est la précision du modèle ? »
8. `render_report` → appelle `/api/report/generate` (mais côté serveur, sans
   re-appeler Claude — utilise un template déterministe pour éviter les boucles).

### 5.4 Boucle conversationnelle (Python, `agent.py`)

Le Claude Agent SDK possède son propre agentic loop — **ne réécris pas de boucle
`while` manuelle sur `tool_use`/`tool_result` comme avec le SDK API classique**.
`agent.py` se limite à :

1. Construire les `ClaudeAgentOptions` une fois au démarrage (system prompt,
   serveur MCP in-process avec les 8 tools, `allowed_tools`, cap de tours).
2. Pour chaque message reçu sur `/api/agent/chat`, passer le message au SDK et
   consommer le flux de messages qu'il retourne (texte, appel d'outil, résultat
   d'outil, fin de tour).
3. Repackager chaque message en événement SSE vers le frontend — le contrat
   `{"type": "text", ...}` / `{"type": "tool_use_start", ...}` / etc. décrit plus
   haut reste la bonne cible côté frontend, seule la source change.
4. Logger chaque tool call avec ses arguments dans `logs/agent.jsonl` (le SDK
   expose probablement les tool calls dans le flux — sinon, hook-le dans chaque
   fonction tool).

Pas de gestion manuelle de `messages.append(...)` avec des blocs `tool_result` —
c'est exactement ce que le SDK fait à ta place, et c'est tout l'intérêt de
l'utiliser ici plutôt que le SDK API brut.

---

## 6. Jalons d'exécution (dans cet ordre)

Chaque jalon = un commit atomique avec un critère d'acceptation objectif.

### M0 — Scaffolding (env, pas encore de logique)
- Créer l'arborescence `engine-copilot/` complète (vide au besoin).
- `pyproject.toml`, `README.md` minimal (pas de `.env.example` — voir §7.5).
- `pnpm create vite@latest frontend --template react-ts` puis Tailwind + shadcn +
  `framer-motion`.
- Backend : `uvicorn engine_copilot.main:app --reload` sert `/api/health` →
  `{"ok": true, "claude_authenticated": true}`.
- Frontend : `pnpm dev` sert une page avec quatre onglets (Home / Flow / Stress /
  Chat) — Home peut être un hero statique à ce stade.
- **Critère** : les deux serveurs démarrent, `curl localhost:8000/api/health`
  montre `claude_authenticated: true`, navigateur affiche l'app.

### M1 — Flow prediction end-to-end (statique)
- `flow_service.py` : charge `unet-phys.pt` au startup, expose une fonction
  `predict(coeffs, reynolds, model_key) -> dict` en unités physiques.
- Endpoint `/api/flow/predict` implémenté et testé (`test_flow_service.py`).
- `ShapeEditor` (sliders) + `FieldViewer` fonctionnels dans le Flow Lab.
- Debounce sur changement de coeff → refresh des 3 heatmaps + Cd.
- **Critère** : bouger un slider met à jour les trois heatmaps sous 500 ms
  (mesuré au chrono).

### M2 — Animation flow (streamlines client + LBM streaming)
- `StreamlineOverlay.tsx` : particules RK4 sur le champ prédit, 30 fps stable.
- `lbm_frames.py` : LBM instrumenté qui yield 60 frames.
- `/api/flow/simulate` streame les frames en SSE.
- `FlowAnimation` bascule entre les deux modes.
- **Critère** : dans le mode streamlines, sur une forme allongée, les particules
  décrivent visiblement un sillage ; dans le mode LBM, on voit les rouleaux de Von
  Karman se développer sur 5-10 s.

### M3 — Optimizer en live (Flow)
- Factorise la boucle Adam de `flowsurrogate.optimize.optimise` en générateur.
- `/api/flow/optimize` streame les itérations.
- `OptimizerPanel` : morphing de la forme optimisée + convergence live.
- `/api/flow/verify` implémenté ; bouton « Verify » affiche le gap.
- **Critère** : lancer l'optim depuis l'UI donne bien un $C_d$ final proche de
  1.44 (± 5 %) sur Re=80, et le morphing de la forme est visible.

### M4 — Stress Lab (miroir plus léger)
- `stress_service.py` : charge `gnn-mp10-glob.pt`, expose `predict`.
- `/api/stress/predict` + `/api/stress/optimize` (CMA-ES ou random search).
- `HoleEditor` + `StressViewer` avec peak marker animé.
- **Critère** : cliquer pour placer 3 trous → heatmap magma + peak vM affiché ;
  cliquer « Optimize » réduit le peak vM observé sur au moins 3 designs sur 5.

### M5 — Agent Claude
- Vérifie `claude login` (voir §7.5) avant de commencer.
- `tools.py` avec les 8 tools définis comme fonctions Python décorées pour le
  Claude Agent SDK (voir §5.3).
- `agent.py` : `ClaudeAgentOptions` + boucle conversationnelle qui consomme le
  flux du SDK (voir §5.4 — le SDK gère l'agentic loop, pas toi).
- `/api/agent/chat` SSE.
- `ChatPanel` + `ToolCallCard` : affichage des appels d'outil en cours et
  résultats compacts, avec liens deep-link vers les labs.
- **Critère** : la commande utilisateur *« Optimise ma forme à Re=100 et vérifie »*
  déclenche exactement les tools `flow_optimize` puis `flow_verify`, et
  Claude cite les deux $C_d$ dans sa réponse finale.

### M6 — Rapport et polish
- `/api/report/generate` génère un markdown à partir d'un résultat d'optim.
- Bouton « Export report » dans les deux labs → télécharge un `.md`.
- README de `engine-copilot/` : gif animé de la plateforme, instructions de setup,
  section « architecture » avec un mermaid.
- **Critère** : `pnpm build` du frontend et `pytest` du backend passent
  sans warning.

---

## 7. Points techniques délicats (à ne pas rater)

### 7.1 Chargement des modèles au startup, pas par requête
Dans `main.py`, utiliser `lifespan` FastAPI pour instancier un dict
`{"unet-phys": model, ...}` **une fois**. Chaque endpoint pioche dedans. Sinon la
latence par requête explose à cause du `torch.load` + `torch.compile`.

### 7.2 Coût sérialisation JSON
Un champ 192×96 float32 = 73 KB × 3 = 220 KB. En SSE à 30 iterations/s c'est trop.
**Downscale à 96×48** avant envoi (moyennage 2×2) et cast en float16 → 18 KB.
Pour l'animation LBM, downscale à 128×64 et compresse en base64 d'un `np.savez`
côté serveur, décompresse côté client avec `pako.js` ou `fflate`.

### 7.3 Reproduire `radius()` en JS
Le contour est recalculé côté client pour l'interactivité. Test unitaire :
échantillonner 20 formes aléatoires, comparer numpy vs JS, tolérer 1e-6.

### 7.4 CORS
FastAPI CORS middleware, autoriser `http://localhost:5173` uniquement.

### 7.5 Auth Claude Code (pas de clé API)
Au démarrage du backend (`lifespan` FastAPI), vérifie que le CLI `claude` est
installé et authentifié (commande exacte à confirmer dans la doc du SDK). Si
l'auth est absente, `/api/health` doit répondre avec un champ explicite
(`{"ok": false, "reason": "claude_not_authenticated"}`) plutôt que de planter au
premier message du chat. Le `README.md` doit préciser ces deux commandes comme
seule étape de setup pour l'onglet Chat :
```
npm install -g @anthropic-ai/claude-code
claude login
```
**Aucune clé API, aucun `.env` secret.** N'introduis jamais de champ
`ANTHROPIC_API_KEY` nulle part dans ce projet.

### 7.6 Timeout et limites
- Chaque tour agent cappé à 30 s côté serveur.
- Boucle tool use cappée à 6 tours.
- Affiche en bas du ChatPanel un compteur d'usage de session — **pas un montant
  en dollars** (il n'y en a pas, l'abonnement Claude Code couvre l'usage) :
  quelque chose comme `12 messages · 4 tool calls cette session`.

### 7.7 Compatibilité device
Réutilise `pick_device()` de `flowsurrogate.train` — mps sur Mac Apple Silicon,
cpu sinon. Pas de CUDA à gérer.

### 7.8 Tests
Trois fichiers de tests, chacun rapide (< 5 s) :
- `test_flow_service.py` : predict donne un `cd` dans [0.5, 5.0] et un champ de
  shape (3, 96, 192).
- `test_stress_service.py` : predict donne un peak_vm > 0 et un champ non NaN.
- `test_agent_tools.py` : les JSON schemas des tools sont valides
  (`jsonschema.validate` sur des exemples).

Le test avec vrai appel Anthropic est **skip par défaut**
(`@pytest.mark.skipif(not ANTHROPIC_API_KEY)`).

---

## 8. Livrable final attendu

Une fois tous les jalons complets :

1. Démarrer avec deux commandes (documentées dans le README) :
   ```
   cd engine-copilot/backend && uvicorn engine_copilot.main:app --reload
   cd engine-copilot/frontend && pnpm dev
   ```
2. Ouvrir `http://localhost:5173`, naviguer entre les quatre onglets, tout marche
   sans configuration additionnelle (sauf avoir fait `claude login` une fois —
   pas de clé API, pas d'abonnement séparé).
3. Un enregistrement GIF de 30 s dans `engine-copilot/docs/demo.gif` montrant :
   optim flow live → verify → chat qui orchestre → stress lab.
4. Un README avec le mermaid d'architecture et une section « Design decisions »
   qui explique pourquoi ces choix (matériau pour l'entretien).

---

## 9. Ce que tu (Claude Code) ne dois PAS faire

- ~~Écrire un ORM ou une base de données~~
- ~~Ajouter de l'auth, du multi-utilisateur, du multi-session persistant~~
- ~~Refactorer `flow-surrogate/` ou `stress-gnn/` pour « améliorer » quoi que ce soit~~
- ~~Écrire un système de plugins~~
- ~~Rendre la 3D en Three.js~~
- ~~Réentraîner un modèle depuis l'UI~~
- ~~Générer du code Python via Claude et l'exécuter~~
- ~~Ajouter des dépendances non listées ici sans une bonne raison~~

Si tu es tenté par un de ces items, **arrête-toi et signale-le** dans un commentaire
avant de continuer.

---

## 10. Ordre de démarrage recommandé

1. Lis ce document en entier.
2. Vérifie que `flow-surrogate/runs/unet-phys.pt` et
   `stress-gnn/runs/gnn-mp10-glob.pt` existent (sinon avertis l'utilisateur —
   il faudra les régénérer avec les scripts de leurs projets respectifs).
3. Vérifie que `claude` (le CLI) est installé et authentifié. Si non, arrête-toi
   et demande à l'utilisateur de faire `claude login` — tout M5/M6 en dépend.
4. Exécute M0 → M6 dans l'ordre. Après **chaque** jalon :
   - lance les tests correspondants,
   - vérifie le critère d'acceptation à la main,
   - commit avec un message `M0: scaffolding`, `M1: flow prediction`, etc.
   - avant d'écrire du code de graphique (M1-M4), invoque le skill `dataviz`.
5. Après M6, ouvre un dernier PR (ou dernier commit) avec le gif de démo et la
   section « Design decisions » du README.

Bon build.
