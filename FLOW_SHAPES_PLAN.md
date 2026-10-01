# Flow Lab — étoiles en Fourier (k ≤ 6), grandeurs physiques, scénarios contrastés

> **Statut : proposé, rien n'est implémenté.** Rédigé le 1ᵉʳ octobre 2026 et réajusté le
> même jour selon les décisions du propriétaire du dépôt :
> 1. **paramétrisation de Fourier, modes k = 2..6** (5 modes, 10 coefficients) ;
> 2. **pas de coins vifs**, mais une étoile doit être atteignable par les coefficients ;
> 3. **l'ancien modèle reste sélectionnable dans l'interface** ;
> 4. **pas d'autre paramètre que U, ρ, μ, D**, avec Re calculé. Pas de paramètre
>    d'environnement (sol, cisaillement, vent tournant…) ni de tête Cl ;
> 5. **l'interface s'ouvre en grandeurs physiques** ;
> 6. **un seul scénario pour l'instant, le scénario de base** (min Cd à aire fixe). Les
>    autres sont gardés au §4.2 pour plus tard ;
> 7. **le budget de calcul peut être augmenté si nécessaire** (§5.1).
>
> Ce plan prolonge `FLOW_DESIGN_SPACE_PLAN.md` (11 août). Son palier 0 (contraintes
> réglables) sert aux scénarios du §4, et son palier 2 (modes 5–6) est remplacé par le §2.
> Les chiffres du §1 ont été mesurés au LBM le 1ᵉʳ octobre. Tout ce qui est marqué
> *hypothèse* ne l'a pas été.

## 0. La demande

1. Des coefficients qui permettent des formes **exotiques**, en particulier une étoile, avec
   des boutons « preset » dans l'interface.
2. Pouvoir régler individuellement **vitesse, viscosité, densité et taille**, et que Re soit
   calculé à partir d'elles.
3. Des **scénarios** dont les formes optimales sont franchement différentes. Pour l'instant,
   seul le scénario de base est retenu, et les autres sont mis de côté (§4.2).

---

## 1. Constat mesuré (1ᵉʳ octobre 2026)

Protocole : 13 formes r(θ) en polaire, toutes à **aire constante 306.9** (la médiane du
train, D_eq = 19.8 cellules), avec la grille de production 192×96. Un seul `LBMSolver.run`
en batch 26, `SolverConfig(n_steps=16000, n_avg=4000)`, 84 s sur MPS. Cd est rapporté à
D_eq. Les scripts sont jetables (scratchpad) et n'ont rien modifié dans le dépôt.

| forme | Cd Re=30 | Cd Re=80 | dérive Cd Re=80 |
|---|---|---|---|
| ellipse AR 3, dans l'axe | **1.832** | **1.090** | 0.01 % |
| goutte, culot arrondi à l'aval | 2.095 | 1.350 | 0.01 % |
| ellipse a2 = 0.16 (**optimum actuel**) | 2.214 | 1.450 | 0.01 % |
| cercle | 2.636 | 1.887 | 0.01 % |
| croix lisse à 4 branches | 3.025 | 2.166 | 0.01 % |
| étoile à 5 branches (pointue, superformule) | 4.398 | 4.033 | 0.54 % |
| plaque épaisse, AR 3, face au flux | **6.201** | **5.313** | 0.06 % |

(Carré, losange, triangles et fleur à 8 lobes ont aussi été mesurés. Ils sont hors sujet
depuis les décisions 1 et 2, mais confirment le point suivant.)

Ce qu'on en retient :

- **À aire égale, Cd varie d'un facteur 4.9**, alors que l'optimiseur actuel explore
  seulement entre 1.45 et ~2.3.
- **Une étoile est pire qu'un cercle pour la traînée** (×1.15 pour la croix, ×2.1 pour
  l'étoile pointue). Un objectif « min Cd » ne fera donc **jamais** naître d'étoile. Les
  étoiles n'apparaissent que dans les scénarios qui les récompensent (§4).
- **La famille actuelle ne contient pas d'étoile à 5 ou 6 branches** : il manque les modes
  5 et 6. Les amplitudes nécessaires (~0.3–0.4) sont aussi hors de la boîte ±0.16.
  **Un réentraînement est donc incontournable.**
- L'étoile lisse de la décision 2, `1 + a5 cos 5θ`, n'a **pas encore** été mesurée : celle
  du tableau est pointue. C'est l'objet de P2.

---

## 2. Paramétrisation : Fourier k = 2..6

### 2.1 L'étoile lisse est exacte

Une étoile lisse à N branches, c'est `r = r0 (1 + a_N cos Nθ)`, autrement dit un seul mode.
Elle est exacte pour N ≤ 6 et sans phénomène de Gibbs, puisqu'il n'y a pas de coin.
`MODES` passe de (2, 3, 4) à (2, 3, 4, 5, 6), et `N_COEFFS` de 6 à **10**.

| preset (bouton) | coefficients (indicatifs, calés en P1) |
|---|---|
| Cercle | tous à 0 |
| Carénage (ellipse) | a2 ≈ +0.25 |
| Plaque face au flux | a2 ≈ −0.3 |
| Goutte | a2 > 0 plus un terme a3 (l'asymétrie amont/aval passe par les modes impairs) |
| Trèfle | a3 ≈ 0.3 |
| Carré arrondi | a4 ≈ −0.06 (face au flux) ou +0.06 (sur la pointe) |
| Croix | a4 ≈ 0.3 |
| Étoile à 5 branches | a5 ≈ 0.35 |
| Étoile à 6 branches | a6 ≈ 0.3 |

Deux contrôles d'interface viennent en plus des 10 curseurs. Ils n'ajoutent aucun degré de
liberté :

- **Rotation** : tourner le corps de β fait tourner chaque couple (a_k, b_k) de kβ. C'est
  exact, et l'orientation, aujourd'hui cachée dans les b_k, devient un bouton.
- **Symétrie haut/bas** : si on la force, tous les b_k passent à 0.

### 2.2 Bornes et validité

Une étoile demande a5 ≈ 0.35, mais 10 coefficients à ±0.35 en même temps donnent des formes
pincées. La boîte par coefficient est donc remplacée par des **contraintes géométriques**,
appliquées au tirage **et pendant la descente**. Ce dernier point corrige le relevé §5.1 du
plan précédent, encore ouvert.

- **r_min ≥ 0.55·r0** (déjà `MIN_RADIUS_FACTOR`) ;
- **largeur de lobe ≥ ~4 cellules**. Avec k = 6 et r0 ≈ 10, une période vaut ~10 cellules et
  un lobe ~5 : c'est faisable, à confirmer en P2 ;
- **budget d'amplitude** Σ|c_k| ≤ ~0.5, pour ne pas cumuler plusieurs modes forts.

### 2.3 Tirage du dataset

| part | tirage | rôle |
|---|---|---|
| 25 % | ancienne distribution (k = 2..4, ±0.16, modes 5–6 nuls) | pas de régression sur la v1, comparaison directe |
| 50 % | **un ou deux modes dominants** (amplitude jusqu'à 0.45, phase aléatoire) plus un bruit de ±0.04 sur les autres | étoiles, trèfles, croix et leurs voisins |
| 25 % | uniforme large, filtré par §2.2 | remplissage entre les familles |

Environ 2 500 échantillons, Re ∈ [30, 110] comme aujourd'hui, r0 ∈ [8.5, 11]. **Split
OOD par famille** : par exemple, les étoiles à 6 branches sont retirées du train, ce qui
teste si le réseau a appris la physique ou les presets. L'OOD actuel (Re 110–150, amplitude
élargie) est conservé. Filtre sur `cd_drift`, avec la part rejetée rapportée par famille.

Pour l'honnêteté, l'UI affiche une **distance au train**, c'est-à-dire la distance au plus
proche échantillon d'entraînement dans l'espace des coefficients. Elle passe en `cq-warn`
au-delà d'un seuil, ce qui rend visible un optimiseur qui combine deux modes forts.

### 2.4 Coexistence v1 / v2

- `runs/unet-phys.pt` (v1, 6 coefficients) **reste en place et reste sélectionnable**. Le
  nouveau modèle va dans `runs/unet-phys-v2.pt`. Une POD v2 sert de baseline non neuronale,
  selon la règle du dépôt.
- Le backend charge les deux. `config.FLOW_MODEL_QUALITY` a une entrée par modèle, recopiée
  à la main comme aujourd'hui.
- Les entrées du réseau ne changent pas : 3 canaux (sdf, occ, Re). Seule la distribution de
  formes s'élargit.
- En v1, les curseurs a5, b5, a6, b6 et les presets qui en ont besoin sont **grisés avec la
  raison affichée**, sans disparaître.
- Le « 6 » codé en dur (`schemas.py` ×4, `geometry.ts`, `ShapeEditor.tsx` `LABELS`, la
  ligne de contexte du copilote, les tests) devient le `N_COEFFS` du modèle actif. Les
  schémas acceptent 6 ou 10 coefficients.
- `geometry.py` : `MODES` devient un paramètre (v1 par défaut) plutôt qu'une constante,
  pour que `radius`, `level_set` et `area` servent les deux modèles.

---

## 3. Grandeurs physiques : U, ρ, μ, D, avec Re calculé

Ce lot ne demande **aucun réentraînement** et fonctionne **avec la v1 comme avec la v2** :
le surrogate et le LBM ne consomment que Re. En écoulement incompressible et newtonien,
les quatre grandeurs n'agissent sur l'écoulement que **par Re**. Elles comptent en revanche
pour les **forces en newtons** (§3.3).

### 3.1 Entrées

Curseurs logarithmiques plus un champ numérique, avec les unités SI affichées :

| grandeur | symbole | plage du curseur | note |
|---|---|---|---|
| vitesse amont | U | 10⁻⁴ – 10² m/s | — |
| masse volumique | ρ | 0.5 – 2 000 kg/m³ | — |
| viscosité dynamique | μ | 10⁻⁵ – 10² Pa·s | une bascule permet de saisir ν = μ/ρ à la place |
| taille du corps | D | 10⁻⁴ – 10 m | **diamètre équivalent à aire égale**, la longueur de référence de Cd dans tout le dépôt (`geometry.equivalent_diameter`) |

On affiche **Re = ρ U D / μ** en grand, avec sa formule.

**Mode par défaut : grandeurs physiques** (décision 5). Au premier chargement, l'interface
part du preset **Eau** (grain de 1 mm à 8 cm/s, Re = 80). C'est exactement le Re = 80 par
défaut d'aujourd'hui, donc la forme et le Cd affichés à l'ouverture ne changent pas. Le
mode « Re direct » reste accessible par une bascule discrète, pour les comparaisons avec
les tableaux de `RESULTS.md`, qui sont en Re.

La viscosité se saisit en **μ (dynamique, Pa·s)** par défaut, avec une bascule vers ν
(m²/s). Les tables de fluides donnent μ, et ρ reste ainsi utile en mode ν, puisqu'il sert
à la force. Ce choix est réversible en une ligne si tu préfères l'inverse.

Presets de fluide, choisis pour qu'un cas réaliste tombe **dans l'enveloppe d'entraînement
Re ∈ [30, 110]** (calculé à la main, à 20 °C) :

| preset | ρ (kg/m³) | μ (Pa·s) | cas par défaut | Re |
|---|---|---|---|---|
| Air | 1.204 | 1.81·10⁻⁵ | fil de 1 mm à 1 m/s | 67 |
| Eau | 998 | 1.00·10⁻³ | grain de 1 mm à 8 cm/s | 80 |
| Huile moteur SAE 30 | 880 | 0.29 | tige de 2 cm à 1 m/s | 61 |
| Glycérine | 1 261 | 1.41 | barre de 10 cm à 1 m/s | 89 |
| Miel | 1 420 | ~10 | pieu de 50 cm à 1 m/s | 71 |

### 3.2 L'enveloppe

Les situations du quotidien sortent de loin du domaine : une voiture à 30 m/s dans l'air,
c'est Re ≈ 3·10⁶, contre un domaine entraîné de 30 à 110. L'interface doit donc :

- colorer chaque curseur avec **la plage de cette grandeur qui garde Re dans l'enveloppe**,
  les trois autres étant fixées ;
- afficher le bandeau `cq-warn` dès que Re sort de [30, 110], et passer en `burnt` au-delà
  de 150, où le surrogate n'a jamais été évalué ;
- proposer un bouton « ramener dans l'enveloppe » qui ajuste **la dernière grandeur
  touchée** ;
- ne **jamais** présenter comme fiable un résultat hors enveloppe.

### 3.3 Ce que les unités ajoutent

À Re égal, changer ρ ne change ni Cd ni la forme optimale. En revanche, cela change les
grandeurs dimensionnelles, qu'on affiche à côté de Cd :

- force de traînée par mètre d'envergure, **F′ = ½ ρ U² D · Cd** (N/m). Exemple glycérine :
  ½ · 1261 · 1² · 0.1 · 1.45 ≈ 91 N/m ;
- puissance dissipée **P′ = F′ · U** (W/m) ;
- champs en m/s (u_lat · U / u_in) et en Pa (p_lat · ρU² / u_in²) ;
- temps physique de l'animation LBM : un pas vaut (D / D_cellules) · (u_in / U) secondes ;
- L et H de la forme en mètres.

Une info-bulle le dit quand l'utilisateur modifie ρ à Re constant : **« Re fixe la forme
optimale ; ρ et U fixent ce qu'elle coûte en newtons. »**

Deux garde-fous :

- **Mach** : on avertit si U / c > 0.3, car le LBM est incompressible.
- **Deux notions de taille** : D change Re, mais **pas** le rapport entre le corps et le
  canal, qui reste celui de la grille. Le curseur d'aire (plan précédent §2.1) règle la taille
  relative dans le canal. Les deux sont séparés et nommés différemment (« Body size (m) »
  et « Relative size / blockage »).

### 3.4 Où ça vit

- **Backend** : `engine_copilot/units.py`, une seule source pour Re et les conversions.
  Les schémas acceptent soit `reynolds`, soit `{velocity, density, viscosity, size}`. Les
  deux sont exclusifs, et un validateur calcule Re. Les réponses renvoient en plus
  `drag_force_per_span` et `power_per_span` quand les unités sont fournies. C'est de la
  présentation : rien ne change dans `flowsurrogate`.
- **Frontend** : `src/lib/units.ts` en miroir, pour un affichage instantané, sur le modèle de
  `geometry.ts`. Un `FluidPanel.tsx` remplace le bloc Reynolds de `ShapeEditor.tsx`
  (~l. 127–148). `FlowLab.tsx` garde l'état `{U, ρ, μ, D}` et dérive `reynolds`.
- **Copilote** : `flow_predict`, `flow_optimize`, `flow_verify` et `flow_simulate`
  acceptent les grandeurs physiques.
- **Rapport** : `report.py` cite ρ, μ, U, D, Re et F′ quand ils ont été fournis.

**Relevé à corriger dans ce lot** : l'enveloppe de Re est écrite de quatre façons.
`schemas.py` accepte [10, 200], `tools.py:79` déclare `in_training_range` sur [10, 150],
`ShapeEditor.tsx:106` sur [20, 150] (et la ligne 148 affiche « TRAINED 20–150 »), alors
que le train couvre **[30, 110]** (`data/train_meta.json`). On la centralise en une
constante par modèle, exposée par `/api/flow/models`.

---

## 4. Scénarios

### 4.1 Retenu : le scénario de base

**Minimiser Cd à aire fixe**, c'est-à-dire l'objectif actuel, sur l'espace de formes élargi
de la v2 et avec Re calculé depuis U, ρ, μ, D.

Il faut le dire sans détour, ici et dans l'UI : **ce scénario ne fera pas naître d'étoile.**
Une étoile traîne plus qu'un cercle (§1). À la descente, l'optimum devrait rester une forme
allongée, de type ellipse ou goutte, et il bougera peu avec Re (plan précédent §1.3). Ce
qui change avec la v2 dans ce scénario :

- l'utilisateur peut **partir d'un preset exotique** (étoile, croix, trèfle) et **regarder
  l'optimiseur le déformer** jusqu'au carénage. Le chemin parcouru devient l'intérêt de la
  démo ;
- les bornes géométriques (§2.2) remplacent la boîte ±0.16. L'optimum n'est donc plus un
  coin de boîte, et le plan précédent §1.4 prévoit qu'il s'arrête de lui-même vers a2 ≈ 0.4.
  C'est à confirmer sur la v2 ;
- `is_valid` est appliqué pendant la descente ;
- un départ par preset, et l'on affiche **le meilleur de chaque famille de départ**.
  C'est ce qui montre si le paysage reste unimodal avec 10 coefficients.

Côté optimiseur, on ne modifie que les bornes, la validité et les départs.
**Pas de nouvel objectif, pas de nouvelle contrainte.**

### 4.2 Gardés pour plus tard (non planifiés)

Règle du plan précédent (§1.2) : **un objectif symétrique et mono-critère donne un optimum
unique.** Pour obtenir des formes vraiment différentes, il faudra donc changer l'objectif ou
ajouter des contraintes. Toutes celles listées ici sont géométriques et ne demandent aucune
nouvelle entrée au réseau : chacune est un ajout à `OptConfig` (`objective`, ou une pénalité
hinge), plus un sélecteur dans l'UI. Les pièges du plan précédent §2.4 s'appliquent :
paysage multimodal, et hinge qui sur-satisfait.

| scénario | objectif | contraintes | optimum attendu | statut |
|---|---|---|---|---|
| **Fourgon** | min Cd | hauteur ≥ H, longueur ≤ L | bloc arrondi | partiel (plan précédent §2.2) |
| **Aérofrein / mélangeur** | **max** Cd | aire fixe, épaisseur ≥ seuil | plaque si elle est permise, sinon croix ou étoile | mesuré : plaque 5.31 |
| **Échangeur à ailettes** | min Cd | **périmètre ≥ P** | trèfle, croix ou étoile, avec un N qui croît avec P | *hypothèse* ; le seul scénario où l'étoile peut gagner |
| **Fabricabilité** | min Cd | épaisseur ≥ seuil | de quasi-cercle à fuselé | mesuré (plan précédent §2.2) |

Critère d'acceptation, le moment venu : chaque optimum est vérifié au LBM, et deux scénarios
donnent des silhouettes qui ne se recouvrent pas (IoU < 0.7).

Pistes écartées par la décision 4, consignées pour plus tard. Coût : **A** = modèle actuel,
sans réentraînement ; **B** = nouvelles données, solveur inchangé ; **C** = modification de
`lbm.py` **et** de `lbm_frames.py`, plus de nouvelles données. Le critère reste le même : un
paramètre ne sert l'optimisation que s'il **change le classement des formes**, et ce sont
ceux qui cassent la symétrie haut/bas qui le font.

| piste | casse la symétrie ? | coût | note |
|---|---|---|---|
| Vent tournant (min du Cd moyen sur ±β) | non | A | rotation exacte des coefficients |
| Garde au sol (`cy` variable) | oui | B | la paroi glissante agit comme un sol miroir |
| Vent cisaillé ux(y) | oui | C léger | `f_in` est déjà par ligne, plus un canal d'entrée |
| Obstacle amont (tandem) | si décalé | C léger | il faut une force par corps (`links` restreint) |
| Tête Cl, profil porteur à Cl imposé | oui | B (`cl` est déjà dans les `.npz`) | plan précédent §3 |
| Transfert thermique (Nusselt) | non | C lourd | la vraie raison physique des ailettes ; voir la branche `thermal-pinn` |

---

## 5. Lots

| lot | contenu | coût | calcul | dépend de |
|---|---|---|---|---|
| **P0** | grandeurs physiques U, ρ, μ, D et Re calculé (§3) : `units.py`, `units.ts`, `FluidPanel.tsx`, presets de fluide, enveloppe centralisée, F′ et P′, outils, rapport | ~½ j | 0 | rien ; valable pour la v1 |
| **P1** | `geometry.py` : `MODES` paramétrable, rotation, symétrie, périmètre, L, H, largeur de lobe, `is_valid` étendu, presets ; NumPy et Torch, plus les tests de concordance | ~½ j | 0 | — |
| **P2** | étude au LBM avant tout apprentissage : étoiles lisses N = 3..6 à plusieurs amplitudes (Cd, dérive) et **convergence en maillage** 192×96 contre ×2 sur l'étoile à 6 branches | ~¼ j | ~20 min | P1 ; **point de décision avec toi** |
| **P3** | dataset v2 (§2.3), `unet-phys-v2`, POD v2, `RESULTS.md` par famille | ~½ j | ~1 h 30 | P2 |
| **P4** | optimiseur, scénario de base seulement (§4.1) : bornes géométriques à la place de la boîte, `is_valid` dans la descente, départs par preset | ~¼ j | 0 | P3 |
| **P5** | plateforme : sélecteur v1/v2, 10 curseurs, presets, rotation, symétrie, galerie « meilleur par famille de départ », distance au train, `report.py` qui cite le modèle et les grandeurs physiques, outils du copilote | ~1 j | 0 | P4 |

P0 est indépendant de tout le reste et peut partir en premier.

### 5.1 Budget de calcul extensible (décision 7)

Le budget de base est de ~1 h 50 de MPS (P2 + P3). Il peut monter si P2 ou les métriques de
P3 le justifient, et **c'est la mesure qui décide**, pas le confort :

| signal mesuré | réponse | coût ajouté |
|---|---|---|
| en P2, l'étoile à 6 branches bouge de plus de ~5 % en Cd entre 192×96 et la grille ×2 | générer le dataset v2 en grille ×2 (×4 cellules, environ ×8 en temps à Re constant), ou retirer le mode 6 | ~+10 h, ce qui demande alors ton accord explicite |
| dérive `cd_drift` > 1 % sur une part notable des étoiles | allonger `n_avg` et `n_steps` pour ces familles | ~+30 % sur P3 |
| MAPE v2 sur les familles exotiques nettement au-dessus de celui de la v1 sur l'ancienne famille | doubler le dataset (~5 000 échantillons), puis redemander | ~+1 h 15 |
| écart surrogate / LBM > 5 % sur les optimums vérifiés | boucle d'apprentissage actif déjà présente (`run_active_learning`) sur les optimums | ~10–20 min par tour |

---

## 6. Décisions encore ouvertes

Aucune ne bloque P0, P1 ou P2. La prochaine décision vient après P2 : faut-il monter la
résolution, ou garder le mode 6 ? Elle se prend sur la base de la mesure (§5.1).
