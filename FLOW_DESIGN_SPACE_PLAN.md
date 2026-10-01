# Flow Lab — élargir l'espace de conception

> **Statut : proposé, rien n'est implémenté.** Complète `PLATFORM_PLAN.md` (qui s'arrête à M6)
> et ne le remplace pas. Rédigé le 11 août 2026 à partir de mesures faites le 10 août 2026 sur
> `unet-phys` et sur le solveur LBM — les chiffres cités ici ont tous été mesurés, aucun n'est
> estimé. Le §1 est un constat : **ne pas le re-mesurer**, s'y référer.

## 0. Le problème

Le Flow Lab n'expose qu'un seul paramètre d'exploitation, Re, et la forme optimisée est la
**même à chaque fois** quelle que soit sa valeur. Ce n'est ni un bug de l'optimiseur ni un
défaut du surrogate : c'est la conséquence directe de la façon dont le problème est posé.
Trois contraintes fixes et invisibles décident du résultat, et Re n'en fait pas partie.

L'objectif de ce plan est de rendre le lab **réellement explorable** — que deux réglages
différents donnent deux formes différentes, chacune pour une raison défendable — au meilleur
coût, et sans jamais faire dire au surrogate autre chose que ce qu'il sait.

---

## 1. Constat mesuré

### 1.1 L'optimum ne dépend pas de Re

Descente depuis le cercle, 250 itérations, `unet-phys`, aire constante 306.9 :

| Re | Cd optimisé | coefficients | coeffs saturés | allongement L/H |
|---|---|---|---|---|
| 20 | 2.518 | a2 = **+0.160**, 5 autres ≤ 0.02 | 1/6 | 1.377 |
| 30 | 2.273 | a2 = **+0.160**, 5 autres ≤ 0.03 | 1/6 | 1.372 |
| 50 | 1.795 | a2 = **+0.160**, 5 autres ≤ 0.02 | 1/6 | 1.369 |
| 80 | 1.549 | a2 = **+0.160**, 5 autres ≤ 0.015 | 1/6 | 1.373 |
| 110 | 1.467 | a2 = **+0.160**, 5 autres ≤ 0.05 | 1/6 | 1.396 |
| 150 | 1.435 | a2 = **+0.160**, 5 autres ≤ 0.05 | 1/6 | 1.391 |

Écart maximal des coefficients entre Re = 20 et Re = 150 : **0.046**. Cosinus entre les
directions de descente au cercle : **≥ 0.97** pour tous les couples de Re testés.

Re change donc la **valeur** de Cd (−43 % de Re 20 à Re 150) mais pas l'**argmin**.

### 1.2 Pourquoi : c'est un coin de boîte, pas un optimum de physique

Une seule variable pèse — a2, l'allongement dans l'axe de l'écoulement. Son gradient est
fortement négatif partout dans le domaine, donc elle se colle à sa borne. **Une variable
saturée, c'est un coin du domaine admissible, et un coin ne bouge pas quand on change le
point de fonctionnement.** Le paysage est par ailleurs unimodal, ce qui est déjà documenté
dans `OptimizerPanel.tsx:20-26` (d'où `N_STARTS = 1`).

Corollaire à retenir : **tant que l'objectif est symétrique haut/bas et mono-critère, la forme
optimale sera symétrique et unique.** Aucun réglage du point de fonctionnement n'y changera
quoi que ce soit.

### 1.3 Élargir la plage de Re ne servirait à rien (vérifié au solveur)

Hypothèse testée : en descendant vers le régime visqueux, le classement des formes
s'inverserait-il ? Vérité terrain LBM (16 000 pas, aire constante, `cd_drift` max 6 %) :

| forme | Re=3 | Re=10 | Re=30 | Re=80 | Re=150 |
|---|---|---|---|---|---|
| a2 = +0.32 (très allongée) | **8.622** | **3.818** | **2.054** | **1.293** | **1.004** |
| a2 = +0.16 (optimum actuel) | 8.832 | 3.993 | 2.214 | 1.450 | 1.173 |
| cercle | 9.453 | 4.455 | 2.636 | 1.887 | 1.733 |
| a2 = −0.16 (aplatie, face au flux) | 10.321 | 4.999 | 3.060 | 2.267 | 2.167 |

Le classement est **identique de Re = 3 à Re = 150**. Seul l'écart se resserre (la forme
allongée gagne 16.8 % à Re = 150, plus que 2.4 % à Re = 3 — et 2.4 % est du même ordre que la
dérive de convergence, donc non concluant). **Conclusion : ne pas investir dans une plage de
Re élargie.** C'est mesuré, c'est réglé.

### 1.4 Les trois contraintes cachées

| # | Contrainte | Valeur | Où | Exposée ? |
|---|---|---|---|---|
| 1 | Aire du corps constante | 306.9 (médiane du train) | `flow_service.py:128`, `optimize.py:71` `to_params` | non |
| 2 | Boîte sur les coefficients | \|c_k\| ≤ 0.16 | `flow_service.py:130`, `optimize.py:47`, = `geometry.COEFF_RANGE` | non |
| 3 | Rayon minimal | r_min ≥ 0.55·r0 | `geometry.py:34` `MIN_RADIUS_FACTOR`, `geometry.py:106` `is_valid` | **non appliquée pendant l'optimisation** |

La #1 est nécessaire : sans elle, minimiser Cd fait simplement rétrécir le corps. La #2 est le
domaine d'entraînement du surrogate (le split `ood` va à 0.24 pour mesurer la dégradation
au-delà). La #3 n'est appliquée qu'au **tirage** des formes — l'optimiseur ne fait que clamper
la boîte, donc rien ne l'empêche de sortir du domaine de validité.

En desserrant la boîte à Re = 80 :

| borne | Cd | allongement | r_min/r0 |
|---|---|---|---|
| 0.16 (actuelle) | 1.549 | 1.37 | 0.852 |
| 0.24 | 1.511 | 1.56 | 0.769 |
| 0.35 | 1.496 | 1.78 | 0.626 |
| 0.50 | 1.493 | 1.86 | **0.496** ← hors validité |

La forme s'affine effectivement, mais elle **s'arrête d'elle-même** à a2 ≈ 0.40 (optimum
intérieur, pas une aiguille) pour seulement **−3.6 % de Cd** — à 2.5× la boîte
d'entraînement, où l'avis du surrogate ne vaut rien sans passage au LBM, et sous le seuil de
validité géométrique. Autrement dit : la boîte ne cache pas un trésor.

### Protocole de reproduction

Trois scripts jetables ont produit ces tableaux, tous exécutés avec
`flow-surrogate/.venv/bin/python`, sans rien modifier dans le dépôt :

1. **Balayage en Re** — `load_splits` pour le `Normalizer` et l'aire médiane, `load_model` sur
   `runs/unet-phys.pt`, puis Adam(lr = 8e-3) sur `render_inputs(coeffs, area, Re, domain, 0.7)`
   → `norm.decode_cd(model(x)[1])`, 250 itérations, clamp à ±borne à chaque pas. Le gradient au
   cercle s'obtient par un `backward()` unique en `coeffs = 0`.
2. **Vérité LBM** — `to_params` + `solid_mask` sur les 4 formes × 5 Re, un seul
   `LBMSolver.run` en batch 20, `SolverConfig(n_steps=16000, n_avg=4000)`. ~70 s sur MPS.
3. **Contraintes** (§2.2) — même boucle qu'en 1, plus une pénalité hinge sur les métriques
   géométriques différentiables (§2.3).

---

## 2. Palier 0 — faire des contraintes les paramètres de l'utilisateur

**Aucun réentraînement, aucune donnée nouvelle. C'est le meilleur rapport résultat/coût.**

Puisque ce sont les contraintes qui décident de l'optimum, ce sont elles qui doivent être
réglables. Chaque contrainte activée déplace l'optimum sur une autre face du domaine
admissible : la forme change, et elle change **pour une raison que l'ingénieur reconnaît**.

### 2.1 Ce qu'on ajoute

| Paramètre | Sens métier | Plage proposée | Coût |
|---|---|---|---|
| **Épaisseur minimale** r_min/r0 | fabricabilité, tenue structurelle | 0.55 – 0.95 (défaut 0.55 = `MIN_RADIUS_FACTOR`) | nul |
| **Longueur maximale** L | encombrement dans le canal | 20 – 32 cellules | nul |
| **Hauteur frontale minimale** H | volume utile embarqué | 14 – 22 cellules | nul |
| **Aire cible** | taille du corps / taux de blocage | 240 – 380 (domaine d'entraînement : r0 ∈ [8.5, 11]) | nul, déjà dans la distribution |
| **Borne de la boîte** | « jusqu'où j'autorise l'extrapolation » | 0.16 – 0.40, marquée `cq-warn` au-delà de 0.16 | nul, mais exige la mention d'extrapolation |

L'aire mérite un mot : elle est **déjà** dans le domaine d'entraînement, le platform la fige
simplement au médian (`flow_service.py:40`, `FlowLab.tsx:15`). La rendre réglable est gratuit
et honnête. À aire = 241 l'optimum n'est d'ailleurs plus le même (b4 sature aussi, allongement
1.475 au lieu de 1.373).

Ce qui **ne** sert **pas** : un angle d'attaque du corps. La famille de Fourier est fermée par
rotation — l'angle est déjà contenu dans les b_k, ce serait un slider redondant.

### 2.2 Preuve que ça produit vraiment des formes différentes

Mesuré à Re = 80, boîte desserrée à 0.45, pénalité hinge, 400 itérations :

| contrainte active | Cd | longueur | hauteur | épaisseur | forme |
|---|---|---|---|---|---|
| épaisseur ≥ 0.50 | 1.497 | 30.7 | 16.8 | 0.601 | fuselée |
| épaisseur ≥ 0.70 | 1.509 | 26.7 | 16.4 | 0.747 | intermédiaire |
| épaisseur ≥ 0.90 | 1.756 | 20.9 | 18.7 | 0.925 | quasi-cercle |
| longueur ≤ 22 | 1.577 | 21.7 | 16.5 | 0.757 | ramassée |
| longueur ≤ 28 | 1.527 | 25.1 | 16.6 | 0.795 | allongée |
| hauteur ≥ 17 | 1.977 | 22.9 | 20.9 | 0.617 | trapue, asymétrique |
| hauteur ≥ 21 | 2.974 | 19.9 | 23.8 | 0.782 | franchement bluff |

Chaque réglage donne une forme et un Cd distincts : c'est un vrai front de Pareto
(traînée contre épaisseur / encombrement / volume utile), pas un artefact numérique.

### 2.3 Découpage, fichier par fichier

Rappel de la règle du dépôt : la physique et la géométrie vivent dans `flowsurrogate`, le
backend n'y réimplémente rien (`PLATFORM_PLAN.md` §9, directive INTOUCHABLES). Les métriques
et les pénalités sont donc de la **bibliothèque**, pas du platform.

1. **`flow-surrogate/flowsurrogate/geometry.py`** — ajouter les métriques de forme
   différentiables, à côté de `area_torch`, en gardant la convention du fichier (chemin NumPy
   + chemin Torch qui doivent coïncider, cf. `tests/test_geometry.py`) :

   ```python
   def shape_metrics_torch(coeffs, target_area, n_theta=512):
       """L, H, r_min/r0, périmètre — différentiables w.r.t. les coefficients."""
       r0 = r0_for_area_torch(coeffs, target_area)[:, None]
       th = torch.linspace(0, 2 * np.pi, n_theta, device=coeffs.device)[None, :]
       sf = torch.ones_like(th).expand(coeffs.shape[0], -1).clone()
       for j, k in enumerate(MODES):
           sf = sf + coeffs[:, 2 * j, None] * torch.cos(k * th) + coeffs[:, 2 * j + 1, None] * torch.sin(k * th)
       r = r0 * sf
       x, y = r * torch.cos(th), r * torch.sin(th)
       return {
           "length": x.max(1).values - x.min(1).values,
           "height": y.max(1).values - y.min(1).values,
           "thickness": r.min(1).values / r0[:, 0],
           "perimeter": torch.sqrt(x.diff(dim=1) ** 2 + y.diff(dim=1) ** 2).sum(1),
       }
   ```

   plus l'équivalent NumPy pour l'affichage et les tests.

2. **`flow-surrogate/flowsurrogate/optimize.py`** — `OptConfig` gagne `min_thickness`,
   `max_length`, `min_height` (tous `float | None`) et `penalty_weight` (défaut 60.0, valeur
   utilisée dans les mesures ci-dessus) ; `optimise()` ajoute
   `Σ w·relu(violation)` à `cd.sum()` avant le `backward()`. Les contraintes **inactives**
   (`None`) ne coûtent pas un seul appel supplémentaire au modèle.

3. **`engine-copilot/backend/engine_copilot/flow_service.py`** — `optimise_streaming` prend
   les mêmes arguments et les transmet ; chaque itération yield en plus les métriques du
   meilleur candidat (`length`, `height`, `thickness`) et un booléen par contrainte active,
   pour que l'UI puisse dire *laquelle* mord.

4. **`schemas.py`** — les nouveaux champs sur `FlowOptimizeRequest`. **`FlowOptimizeRequest`
   n'a pas de `target_area`** alors que `predict` / `verify` / `simulate` en ont un : à ajouter
   au passage, sinon l'aire réglable ne servira à rien côté optimisation.

5. **`api.ts`** (`FlowOptimizeParams`), **`ShapeEditor.tsx`** (les sliders de contrainte, sous
   les coefficients, séparés par un `<Separator>`), **`OptimizerPanel.tsx`** (rappel des
   contraintes actives dans la `CardDescription`, qui dit aujourd'hui « at constant area » et
   devra dire la vérité complète).

6. **`frontend/src/geometry.ts`** — miroir client des métriques (L, H, épaisseur), pour
   afficher « contrainte violée » pendant le drag sans aller-retour serveur. C'est exactement
   le rôle que joue déjà `stressGeometry.ts` pour le Stress Lab ; les deux copies doivent
   rester d'accord avec le Python.

7. **`report.py`** — le rapport exporté doit énoncer **sous quelles contraintes** l'optimum a
   été trouvé. Un optimum sans ses contraintes n'est pas un résultat.

8. **`tools.py`** — `flow_optimize` expose les mêmes paramètres, pour que le copilote puisse
   répondre à « optimise-moi ça avec une épaisseur d'au moins 0.8 ».

### 2.4 Pièges identifiés

- **Le paysage cesse d'être unimodal.** Une des mesures (longueur ≤ 31) est tombée dans un
  minimum local *moins bon* que la contrainte plus serrée à 28. Il faut donc remettre
  `N_STARTS > 1` dans `OptimizerPanel.tsx` dès que des contraintes sont actives — le paramètre
  existe déjà côté endpoint, il est juste épinglé à 1 parce que le problème actuel est
  unimodal (§1.2). Le commentaire qui justifie `N_STARTS = 1` devra être corrigé.
- **La pénalité hinge sur-satisfait.** Épaisseur ≥ 0.60 a convergé à 0.776. Acceptable pour une
  démo, mais si on veut la contrainte serrée il faudra une barrière logarithmique ou une
  projection, pas un hinge.
- **Desserrer la boîte, c'est extrapoler.** Au-delà de 0.16, le bandeau d'extrapolation doit
  apparaître (même idiome `cq-warn` que le Re > 150 existant) et le bouton *Verify with solver*
  devient la seule mesure honnête. Ne jamais afficher un Cd hors boîte sans cette mention.
- **Contrainte #3 d'abord.** Appliquer `is_valid` pendant la descente est un prérequis, pas une
  option : sans elle, desserrer la boîte produit des formes hors domaine de validité (§1.4).

### 2.5 Critères d'acceptation

- Un balayage d'épaisseur 0.55 → 0.95 produit des silhouettes **visiblement** différentes dans
  le panneau START → OPTIMISED, avec un Cd monotone croissant.
- Toute forme rendue par l'optimiseur passe `geometry.is_valid`.
- *Verify with solver* confirme le classement d'au moins deux formes contraintes différentes
  (l'écart LBM/surrogate reste sous les 5 % qui font virer le badge au `burnt`).
- Le rapport exporté cite les contraintes actives et leur valeur.

---

## 3. Palier 1 — tête de portance (un réentraînement de ~4 min)

**C'est la vraie réponse à « une forme différente à chaque fois ».**

Le §1.2 dit qu'un objectif symétrique donne une forme symétrique. Pour obtenir une famille de
formes, il faut un objectif qui casse la symétrie — et le problème canonique de l'aérodynamique
est exactement celui-là : **minimiser la traînée à portance imposée**. Sweeper Cl* donne une
forme distincte, asymétrique, par valeur de consigne.

**Le coût est presque nul, parce que la donnée existe déjà.** Le solveur calcule `cl`
(`lbm.py:222`) et il est **déjà stocké dans les `.npz`** (`data.py:52`, `data.py:84`) ;
`FlowDataset` ne le charge simplement jamais (`data.py:181-183`). Aucune régénération LBM.
`unet-phys` s'entraîne en **246 s** (`runs/unet-phys_history.json`).

Travaux :

1. **`data.py`** — `FlowDataset` charge `cl` ; `Normalizer` gagne `cl_mean`/`cl_std` (+ leurs
   entrées dans `to_json`/`from_json`). Pas de problème de compatibilité : le `Normalizer` est
   recalculé depuis le split d'entraînement, il n'est pas dans les checkpoints.
2. **`models/{unet,fno,pod}.py`** — `scalar_head` passe à 2 sorties. Le `forward` renvoie
   aujourd'hui `(fields, cd)` ; **le changer casse tous les appelants** — `train._losses`,
   `evaluate.py`, `optimize.py` (deux `model(x)[1]`), `flow_service.predict` et
   `optimise_streaming`, plus les tests. Les recenser avant de commencer.
3. **`train.py`** — un `lambda_cl` à côté de `lambda_cd` (`train.py:29`). Attention : le Cl
   moyen d'une forme qui lâche des tourbillons est bruité ; filtrer sur `cd_drift` comme le
   fait déjà la génération, et s'attendre à une tête Cl **moins précise** que la tête Cd.
4. **`optimize.py`** — objectif `Cd + μ·(Cl − Cl*)²`, avec Cl* et μ dans `OptConfig`.
5. **`flow_service.verify`** — exposer le `cl` du LBM, qui est **déjà calculé et jeté**
   aujourd'hui (`flow_service.py:216`, la clé `cl` du dict `out`). C'est gratuit et ça vaut
   d'être fait même avant la tête Cl : ça donne la portance **mesurée** de l'optimum.
6. **`config.FLOW_MODEL_QUALITY`** — une entrée `cl_mape_pct`, recopiée **à la main** depuis
   `results/metrics.json` après réentraînement (rien ne se synchronise tout seul, cf.
   `KNOWLEDGE-GRAPH.md`). L'UI a déjà l'idiome pour l'afficher : la ligne « Measured error ».
7. **`RESULTS.md` / `results/metrics.json`** — régénérer via `scripts/03_report.py`.

Critère d'acceptation : trois consignes Cl* (par exemple −0.3, 0, +0.3) donnent trois formes
franchement différentes, et le LBM confirme le signe et l'ordre de grandeur de la portance.

---

## 4. Palier 2 — régénérer les données (**seulement après** le palier 1)

Coût mesuré : **1.74 s/échantillon** sur MPS (`data/train_meta.json`), soit ~19 min pour les
640 échantillons d'entraînement et ~26 min pour les quatre splits, plus 4 min d'entraînement.
Ce n'est pas cher — ce qui est cher, c'est de le faire **avant** d'avoir un objectif qui
exploite les degrés de liberté ajoutés.

- **Modes 5 et 6** (`MODES`, `N_COEFFS` 6 → 10). Inutile tant que l'objectif est symétrique :
  les modes supplémentaires resteront à zéro à l'optimum. Utile **après** le palier 1. Attention,
  le « 6 » est écrit en dur dans au moins cinq endroits : `schemas.py` (`min_length=6` ×4),
  `geometry.ts`, `ShapeEditor.tsx` (`LABELS`), la ligne de contexte du copilote dans
  `FlowLab.tsx`, et les tests.
- **Plage d'aire élargie** (r0 hors de [8.5, 11]) — n'a de sens que si le §2 montre que l'aire
  est un axe intéressant.
- **Incidence de l'écoulement amont** — demande de toucher la condition d'entrée du solveur,
  donc `lbm.py` **et** sa copie instrumentée `lbm_frames.py` (à garder synchronisées à la main).
  C'est le poste le plus lourd du plan ; à ne considérer qu'en dernier.
- **Plage de Re élargie — non.** Tranché par la mesure du §1.3.

---

## 5. Relevé en chemin (à corriger indépendamment)

1. **`geometry.is_valid` n'est pas appliqué pendant l'optimisation** (§1.4). Dès qu'on desserre
   la boîte, l'optimum sort du domaine de validité (0.496 < 0.55). Prérequis du palier 0.
2. **`ShapeEditor.tsx:148` annonce « TRAINED 20–150 »** ; les splits train/val/test sont en
   fait **Re ∈ [30, 110]** (`data/train_meta.json`, `data.py:29`). 110–150, c'est le split
   `ood` : mesuré, pas entraîné. Le seuil d'alerte de `FlowLab.tsx:154` est donc lui aussi trop
   permissif.
3. **`FlowOptimizeRequest` n'a pas de `target_area`** alors que les trois autres routes flow en
   ont un (§2.3, point 4).
4. **Le `cl` du LBM est calculé puis jeté** dans `flow_service.verify` (§3, point 5).

---

## 6. Ordre et décision

| Ordre | Lot | Coût | Ce que ça débloque |
|---|---|---|---|
| 1 | Relevé §5.1 (`is_valid` dans la descente) | ~1 h | prérequis, corrige un vrai trou |
| 2 | Palier 0 — contraintes réglables + aire | ~1 j | des formes différentes, tout de suite, sans réentraîner |
| 3 | Relevés §5.2–5.4 | ~1 h | honnêteté de l'enveloppe, portance mesurée gratuite |
| 4 | Palier 1 — tête Cl | ~½ j + 4 min GPU | la famille de formes, le vrai problème d'aéro |
| 5 | Palier 2 — modes 5–6 | ~½ j + 30 min GPU | richesse de forme, **utile seulement après le 4** |

Le palier 0 seul répond déjà à la demande initiale (« ne pas avoir la même forme à chaque
fois ») et ne demande aucun GPU. Le palier 1 est ce qui transforme le lab en outil de
conception plutôt qu'en démonstrateur.
