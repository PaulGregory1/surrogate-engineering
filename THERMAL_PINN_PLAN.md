# `thermal-pinn` — Plan d'exécution pour Claude Code

**Objectif** : ajouter un **troisième dépôt de recherche** au monorepo, frère de
`flow-surrogate/` et `stress-gnn/`, sur le seul terrain où un PINN bat franchement la
méthode traditionnelle : **le problème inverse mal posé avec une fonction inconnue en
plus de la source**.

Cas concret : **la température de surface d'un disque de frein pendant un freinage
d'urgence**, reconstruite depuis un unique thermocouple noyé à 2 mm sous la surface de
friction — là où aucun capteur ne peut être posé.

Ce document est un **brief d'exécution**. L'implémentation est libre, mais les jalons,
les protocoles de vérification et les règles d'honnêteté (§9) sont contraignants.
Ne touche **jamais** à `flow-surrogate/`, `stress-gnn/` ni `engine-copilot/` : ce dépôt
est autonome, avec son propre `.venv`.

---

## 0. Contexte et positionnement

Les deux dépôts existants posent la même question — *peut-on remplacer un solveur cher
par un modèle appris ?* — et répondent, honnêtement, « oui mais dans une enveloppe
étroite ». Ce troisième dépôt pose une question **différente**, et c'est ce qui justifie
son existence :

> On ne remplace pas le solveur. On demande au réseau de résoudre un problème que le
> solveur classique **ne sait pas poser**.

**Pourquoi ce cas et pas un PINN « solveur direct ».** Sur un problème direct (géométrie,
coefficients et conditions aux limites connus), un PINN est plus lent et moins précis
qu'un schéma aux différences finies. Écrire le contraire serait malhonnête et
contradictoire avec les deux autres dépôts. Le créneau réel du PINN est l'**inverse /
l'assimilation de données**, et son avantage devient décisif quand une des inconnues est
une **fonction** (ici la loi matériau $k(T)$) plutôt qu'une poignée de scalaires.

**La thèse du dépôt, en une phrase** (à vérifier par la mesure, pas à asserter) :

> Face à un problème inverse mal posé, la méthode classique régularisée est excellente
> tant qu'on lui donne le bon modèle ; le PINN vaut ce qu'elle vaut dans ce cas et la
> dépasse nettement dès qu'une **fonction** inconnue s'ajoute, parce qu'il ne demande ni
> paramétrisation *a priori* ni nouvelle dérivation d'adjoint.

**Non-goals** (à refuser explicitement) :

- pas de dépendance à `deepxde`, `modulus` ou `pina` — tout en **PyTorch nu**, c'est
  pédagogique et c'est le sujet ;
- pas de données expérimentales — tout est synthétique, donc vérifiable contre une
  vérité connue ;
- pas de 2D avant M7 (jalon optionnel) ;
- pas de troisième lab dans `engine-copilot` dans ce plan (c'est une suite possible,
  hors périmètre ici) ;
- pas de « PINN comme surrogate paramétrique » — c'est déjà le sujet des deux autres
  dépôts.

---

## 1. Le problème physique

### 1.1 La scène

Disque de frein en fonte grise, épaisseur 10 mm, freiné sur ses deux faces. Par symétrie
on modélise **une demi-épaisseur** $x \in [0, L]$, $L = 5$ mm : $x=0$ est la surface de
friction, $x=L$ le plan médian (adiabatique).

- Un thermocouple noyé à $x_s = 2$ mm, échantillonné à **100 Hz**, bruit gaussien
  $\sigma = 0{,}5$ K.
- Événement : freinage d'urgence depuis 160 km/h, ~4 s, puis **refroidissement jusqu'à
  t = 20 s** (la phase de refroidissement est ce qui rend $h$ et $\varepsilon$
  identifiables — ne pas la couper).

### 1.2 Les équations

$$\rho\,c(T)\,\frac{\partial T}{\partial t}
= \frac{\partial}{\partial x}\!\Big(k(T)\,\frac{\partial T}{\partial x}\Big),
\qquad x\in(0,L),\ t\in(0,t_{\text{end}})$$

$$-k(T)\,\frac{\partial T}{\partial x}\Big|_{x=0}
= q(t)\;-\;h\,(T_s-T_\infty)\;-\;\varepsilon\sigma_{\!SB}\,(T_s^4-T_\infty^4),
\qquad
\frac{\partial T}{\partial x}\Big|_{x=L}=0,
\qquad T(x,0)=T_0$$

### 1.3 Les constantes (fonte grise, à mettre dans `material.py`)

| grandeur | valeur |
|---|---|
| $\rho$ | 7200 kg/m³ |
| $k_0$ | 52 W/m·K à 20 °C |
| $c_0$ | 460 J/kg·K à 20 °C |
| $\alpha_0 = k_0/\rho c_0$ | ≈ 1,57 × 10⁻⁵ m²/s |
| $h$ | 60 W/m²·K |
| $\varepsilon$ | 0,80 |
| $T_0$, $T_\infty$ | 40 °C, 25 °C |
| $L$, $x_s$ | 5 mm, 2 mm |
| $q_{\max}$ | ~5 MW/m² (crête) |

### 1.4 Le profil de flux vrai — trois composantes, choisies exprès

$q(t) = q_{\text{rampe}}(t)\cdot\big(1 + a_{\text{ABS}}\sin(2\pi f_{\text{ABS}}t)\big) + q_{\text{creux}}(t)$

1. **la rampe** : décélération constante ⇒ $q \propto v(t)$, donc une rampe descendante
   sur 4 s puis zéro. **Récupérable** par toutes les méthodes.
2. **un creux de 0,3 s** au milieu du freinage (relâchement de pédale). **Marginalement
   récupérable** — c'est lui qui départage les régularisations.
3. **une ondulation ABS à 8 Hz**. **Physiquement ineffaçable à 2 mm** : voir §2.2.
   Aucune méthode ne la retrouvera, et le dépôt doit le dire.

### 1.5 Les trois lois matériau (les trois actes)

| loi | $k(T)$ | $c(T)$ | rayonnement |
|---|---|---|---|
| **A — linéaire** | $k_0$ | $c_0$ | non |
| **B — non linéaire lisse** | $k_0(1-4{,}5\times10^{-4}\,\Delta T)$ | $c_0(1+7{,}0\times10^{-4}\,\Delta T)$ | oui |
| **C — à coude** | B, plus une chute localisée de −12 % centrée sur 450 °C, largeur 60 °C (tanh) | idem B | oui |

La loi C n'est pas un artifice : elle représente une transition (changement de phase,
dégradation d'un revêtement, résistance de contact qui bascule). Son rôle dans le dépôt
est précis — **elle n'appartient à aucune famille paramétrique qu'un ingénieur devinerait
a priori**, et c'est exactement l'hypothèse dont la voie classique a besoin.

---

## 2. Pourquoi c'est difficile — les deux faits à établir avant toute méthode

### 2.1 L'inversion directe est explosive

Remonter le champ **vers** la surface est un problème de Cauchy pour l'équation de la
chaleur : mal posé au sens de Hadamard. Une marche en espace (`naive.py`) doit
**diverger visiblement**, et le dépôt s'ouvre là-dessus.

### 2.2 Une bande passante physique, pas un défaut de méthode

Une composante à $\omega$ mesurée à la profondeur $d$ est atténuée par
$\exp\!\big(-d\sqrt{\omega/2\alpha}\big)$. En imposant un plancher d'amplification $1/\eta$
sur l'inversion, la fréquence de coupure vaut

$$f_c=\frac{\alpha\,\big(\ln(1/\eta)\big)^2}{\pi\,d^2}$$

À 2 mm dans la fonte, $f_c \approx 7$ Hz pour $\eta = 0{,}1$ ; à 6 mm, moins de 1 Hz.
**L'ondulation ABS à 8 Hz est donc déjà détruite avant tout traitement.**

**Conséquence contraignante pour l'évaluation** : toutes les métriques sur $q(t)$ sont
calculées **dans la bande récupérable** ($f < f_c$, filtre passe-bas appliqué à la vérité
comme à l'estimation). Noter une méthode sur ce que la physique a effacé serait une
faute. La composante hors bande est reportée séparément, comme *information perdue*.

---

## 3. Les méthodes en compétition

Quatre méthodes, de la plus naïve à la plus forte. **Les baselines classiques doivent
être implémentées dans leur meilleure version**, sinon le résultat ne vaut rien.

| # | méthode | fichier | s'applique à |
|---|---|---|---|
| N | marche en espace (différences finies, aucune régularisation) | `naive.py` | A |
| B1 | **Beck** — sequential function specification, $r$ pas futurs | `classical.py` | A, (B via propriétés gelées) |
| B2 | **Tikhonov** — déconvolution de Duhamel, pénalité $\|D_2 q\|^2$, $\lambda$ par courbe en L **et** principe de discrépance | `classical.py` | A, (B via propriétés gelées) |
| B3 | **Alifanov** — gradient conjugué par l'**adjoint**, arrêt par discrépance | `classical.py` | A, B, C |
| P | **PINN** | `pinn.py` | A, B, C |

**B1/B2 reposent sur la superposition** : ils n'existent que pour la loi A. Pour B et C,
la pratique industrielle les applique quand même avec des propriétés gelées à une
température de référence — c'est ce biais-là qu'on mesure, pas un homme de paille. La
transformation de Kirchhoff est mentionnée dans `BACKGROUND.md` comme le contournement
classique de $k(T)$, avec la raison pour laquelle elle ne sauve ni $\rho c(T)$ en
transitoire ni la condition de rayonnement.

**B3 est le vrai adversaire** : il gère le non-linéaire exactement. Pour l'acte 3 il doit
estimer $q(t)$ **et** $k(T)$ conjointement, ce qui exige de choisir une **famille
paramétrique** pour $k(T)$ — on en teste trois : constante, affine, quadratique. C'est le
cœur de la comparaison : *la voie classique doit deviner la forme de l'inconnue*.

### 3.1 Le PINN, formellement

Adimensionnement obligatoire ($\hat x = x/L$, $\hat t = t/t_{\text{end}}$,
$\hat T = (T-T_0)/\Delta T_{\text{ref}}$ avec $\Delta T_{\text{ref}} = 500$ K, nombre de
Fourier $\mathrm{Fo} = \alpha_0 t_{\text{end}}/L^2 \approx 12{,}6$) — sans lui le réseau ne
converge pas, et `BACKGROUND.md` doit expliquer pourquoi.

$$\min_{\theta,\phi,\psi}\;
\underbrace{\frac{1}{N_s}\sum_n\big|\hat T_\theta(\hat x_s,\hat t_n)-\hat T^{\text{mes}}_n\big|^2}_{\text{le capteur}}
\;+\;\lambda_r\underbrace{\frac{1}{N_c}\sum_j\big|\mathcal R[\hat T_\theta;\,k_\psi]\,(\hat x_j,\hat t_j)\big|^2}_{\text{la chaleur, partout}}
\;+\;\lambda_b\underbrace{\big\|\text{BC}(x{=}0;\,q_\phi)\big\|^2+\big\|\partial_{\hat x}\hat T_\theta(1,\cdot)\big\|^2}_{\text{les bords}}$$

- $\hat T_\theta$ : MLP 4×64, tanh, avec **features de Fourier** en entrée (biais spectral).
- $q_\phi(\hat t)$ : MLP 3×32.
- $k_\psi(\hat T)$ : MLP 2×16 **contraint positif** (softplus) et normalisé à $k_0$ à
  20 °C — actif seulement à l'acte 3.
- **Condition initiale imposée en dur** : $\hat T = \hat T_0 + \hat t\,N_\theta(\hat x,\hat t)$.
  Un terme de loss en moins et une nette amélioration de la convergence.
- Optimisation : Adam 20k itérations (cosine 1e-3 → 1e-5) puis L-BFGS 2k. Collocation :
  4096 points ré-échantillonnés tous les 100 pas.
- **5 graines par configuration**, et la dispersion est publiée : un PINN est variable,
  le cacher serait le principal défaut de crédibilité du dépôt.

---

## 4. Les expériences

### Acte 1 — linéaire, $q(t)$ seule inconnue *(le match nul, publié comme tel)*

Loi A. N, B1, B2, B3, P. Attendu : N explose ; B1/B2/B3/P se tiennent. **Si le PINN perd
ici, on l'écrit dans le Headline du README.**

### Acte 2 — non linéaire lisse *(le classique commence à payer)*

Loi B. B1/B2 avec propriétés gelées (biais mesuré), B3 avec la vraie loi $k(T)$ connue
(borne haute), B3 avec $k$ gelée, P. Sépare bien deux choses : ce que coûte *ignorer* la
non-linéarité, et ce que coûte *devoir la connaître*.

### Acte 3 — $q(t)$ **et** $k(T)$ inconnues, vérité à coude *(la thèse du dépôt)*

Loi C. B3 × {famille constante, affine, quadratique} contre P avec $k_\psi$ libre.
Métriques : erreur sur $T_{\text{surf}}^{\max}$, RMSE sur $q$ en bande, **et erreur sur la
courbe $k(T)$ elle-même**. C'est ici que le gain doit être franc, et la figure de la
courbe découverte est la figure principale du dépôt.

### Études transverses (`04_studies.py`)

- **bruit** : $\sigma \in \{0{,}1;\,0{,}5;\,2{,}0\}$ K
- **profondeur** : $x_s \in \{1;\,2;\,4;\,6\}$ mm — avec $f_c$ superposée
- **nombre de capteurs** : 1, 2, 3 (à 2/3,5/5 mm)
- **ablations PINN** : $\lambda_r = 0$ (⇒ pure interpolation, doit s'effondrer), balayage
  $\lambda_r$ sur 3 décades, sans features de Fourier, sans CI en dur
- **coût** : temps mur par méthode, itérations, **lignes de code**, et *ce qu'il faut
  re-dériver* pour ajouter une inconnue (colonne qualitative assumée)

### Statistiques

**24 scénarios** (profils de flux, graines de bruit, $T_0$ variés) pour les tableaux
(moyenne + p95), **1 scénario vitrine** pour les figures. Ne pas viser 400 cas : ici le
coût est par inversion, pas par échantillon.

---

## 5. Structure du dépôt

```
thermal-pinn/
├── README.md            # le problème, les résultats, ce qu'ils veulent dire (structure des 2 autres)
├── BACKGROUND.md        # depuis zéro : mal-posé, autodiff, collocation, contrainte molle,
│                        #   biais spectral, adimensionnement, Beck/Tikhonov/adjoint, identifiabilité
├── RESULTS.md           # généré par scripts/03_report.py
├── pyproject.toml       # torch, numpy, scipy, matplotlib, tqdm  (paquet: thermalpinn)
├── run_all.py           # --quick / --force, mêmes conventions que les 2 autres
├── .gitignore           # .venv/ __pycache__/ *.egg-info/ data/ runs/
├── thermalpinn/
│   ├── material.py      # rho, k(T), c(T) pour les lois A/B/C, Kirchhoff, dataclass Material
│   ├── scenario.py      # dataclass Scenario (L, x_s, t_end, T0, h, eps, sigma, dt, nx),
│   │                    #   q_true(t), sample_scenarios(rng, n)
│   ├── solver.py        # Crank–Nicolson + Newton (loi non linéaire) -> forward(),
│   │                    #   impulse_response() (Duhamel, loi A), sensor_signal()
│   ├── verify.py        # MMS, semi-infini erfc, bilan d'énergie
│   ├── naive.py         # marche en espace + transfer_function() + cutoff_frequency()
│   ├── classical.py     # beck(), tikhonov() (+ courbe en L), alifanov() (adjoint CG)
│   ├── pinn.py          # features de Fourier, réseaux, résidu, loss, train() -> historique
│   ├── metrics.py       # erreur T_surf max, RMSE bande, erreur k(T), filtre passe-bas
│   └── plots.py
├── scripts/
│   ├── 00_generate_cases.py     -> data/cases.npz
│   ├── 01_validate_solver.py    -> results/solver_validation.json
│   ├── 02_run_methods.py        -> results/methods.json           (les 3 actes × 24 cas)
│   ├── 03_report.py             -> RESULTS.md + figs des actes
│   ├── 04_studies.py            -> results/studies.json + figs/sweeps.png
│   ├── 05_setup_figure.py       -> figs/setup.png
│   └── 06_explainer_figures.py  -> figs/illposed.png, what_the_pinn_optimises.png
├── tests/  test_solver.py, test_classical.py, test_pinn.py
├── figs/   (commité)
├── results/(commité)
└── data/   (ignoré)
```

**À mettre à jour hors du dépôt, une fois M6 fini** : le tableau à trois colonnes du
`README.md` racine, `KNOWLEDGE-GRAPH.md` (nouveau nœud + protocole de mise à jour +
baseline), `CLAUDE.md` (troisième `.venv`).

---

## 6. Les figures — le livrable le plus important

Toutes en `figs/`, commitées, régénérables. Le critère : **chacune doit se comprendre
sans lire le texte**.

| fichier | contenu |
|---|---|
| `setup.png` | le disque, la plaquette, le capteur à 2 mm, la surface marquée « ? », et le profil $q(t)$ vrai avec ses trois composantes annotées |
| `illposed.png` | **3 panneaux** : le signal capteur (parfaitement sage) · l'inversion naïve (échelle log, 10⁶ K) · la fonction de transfert vs fréquence pour $d$ = 1/2/4/6 mm, bande récupérable grisée |
| `what_the_pinn_optimises.png` | schéma : réseau → autodiff → résidu aux points de collocation + écart aux 2000 points capteur = loss ; les inconnues $q_\phi$, $k_\psi$ encadrées |
| `act1_linear.png` | $q(t)$ et $T_{\text{surf}}(t)$ : vérité / Beck / Tikhonov / adjoint / PINN. Le match nul, avec le pic annoté |
| `act2_nonlinear.png` | idem loi B : le biais des propriétés gelées apparaît |
| `act3_discovered_k.png` | **la figure principale** — gauche : $k(T)$ vraie (avec coude) vs les 3 familles classiques ajustées vs la courbe libre du PINN ; droite : l'erreur sur $T_{\text{surf}}^{\max}$ qui en résulte |
| `sweeps.png` | erreur vs bruit · vs profondeur (avec $f_c$) · vs nombre de capteurs |
| `convergence.png` | les termes de loss et la trace des inconnues identifiées au fil de l'entraînement — « ce qu'on optimise », littéralement |

---

## 7. Jalons

Chaque jalon a un critère d'acceptation **mesurable**. Ne pas passer au suivant sans lui.

### M0 — Squelette
`pyproject.toml`, `.venv` propre, `material.py`, `scenario.py`, `.gitignore`, tests
d'import verts.
**Accepté si** : `./.venv/bin/python -c "import thermalpinn"` passe et `pytest tests -q`
est vert (même trivialement).

### M1 — Solveur vérité + vérification
Crank–Nicolson, Newton pour les lois B/C, `float64`.
**Accepté si** `results/solver_validation.json` montre : (a) MMS d'ordre **2** en espace
et en temps (pentes mesurées entre 1,9 et 2,1) ; (b) écart < 0,5 % contre la solution
analytique en $\mathrm{erfc}$ du flux constant sur milieu semi-infini aux temps courts ;
(c) bilan d'énergie fermé à < 1e-8 en relatif sur la loi A.

### M2 — Le mal-posé, démontré
`naive.py`, `figs/illposed.png`, $f_c$ analytique **et** mesurée numériquement.
**Accepté si** : l'inversion naïve dépasse 10⁵ K sur un signal bruité à 0,5 K, et la
coupure mesurée est cohérente avec $f_c = \alpha\ln(1/\eta)^2/\pi d^2$ à ±20 %.

### M3 — Les baselines classiques
Beck, Tikhonov (+ courbe en L), Alifanov adjoint.
**Accepté si** : (a) **test de crime inverse** — bruit nul, même grille : chaque méthode
retrouve $q$ à < 1 % en bande ; (b) le gradient adjoint est validé contre des différences
finies sur 5 composantes tirées au hasard (accord < 1e-4 relatif) ; (c) sur bruit réaliste
les trois sont dans un facteur 2 les unes des autres. **Une baseline qui n'a pas passé
(b) invalide tous les résultats du dépôt.**

### M4 — Le PINN, acte 1
**Accepté si** : le PINN est dans un facteur 1,5 de la meilleure baseline sur
$T_{\text{surf}}^{\max}$, sur 5 graines, **ou** l'écart est mesuré et écrit tel quel dans
le README.

### M5 — Actes 2 et 3
**Accepté si** : (a) le biais des propriétés gelées est mesuré et non nul à l'acte 2 ;
(b) à l'acte 3, la courbe $k_\psi(T)$ retrouve le coude — erreur $L_2$ sur $k(T)$
strictement meilleure que la meilleure des 3 familles classiques, sur 5 graines sur 5 ;
(c) l'erreur sur $T_{\text{surf}}^{\max}$ suit. **C'est le jalon qui décide si le dépôt a
un résultat.** S'il échoue, ne pas maquiller : documenter et proposer un repli (§8.5).

### M6 — Études, tableaux, rédaction
`04_studies.py`, `RESULTS.md`, `README.md`, `BACKGROUND.md`, les 8 figures, tests
complets, `run_all.py --quick` < 5 min et complet < 30 min sur un M-series.
**Accepté si** : `run_all.py --force` régénère tout de zéro sans intervention, et
`RESULTS.md` ne contient aucun nombre écrit à la main.

### M7 — *(optionnel)* 2D axisymétrique
Carte de flux $q(r,t)$ le long du rayon depuis 3 capteurs. L'argument « le PINN ne change
pas de formulation, l'adjoint change de code » devient alors mesurable en lignes.

---

## 8. Points techniques délicats

### 8.1 Ne pas commettre le crime inverse
La vérité se génère sur une grille **fine** (nx = 801, dt = 1e-4 s) ; toutes les méthodes
inverses travaillent sur un modèle **différent** (nx = 201, dt = 1e-3 s). Sinon les
résultats sont un artefact de discrétisation partagée. Le seul endroit où le crime est
autorisé est le test de validation M3(a), et il doit être nommé comme tel dans le code.

### 8.2 L'adimensionnement n'est pas cosmétique
En variables physiques ($T \sim 10^2$, $x \sim 10^{-3}$, $\alpha \sim 10^{-5}$) le PINN ne
converge pas. Tout passe en variables réduites ; la conversion est isolée dans un seul
endroit et testée par un aller-retour.

### 8.3 La pondération des termes de loss
$\lambda_r$ et $\lambda_b$ décident du résultat. Choisir par équilibrage des normes de
gradient, **publier le balayage sur 3 décades**, et ne **jamais** régler $\lambda$ sur le
scénario vitrine ni sur les cas de test — réglage sur un scénario de validation dédié.

### 8.4 Le CPU sera plus rapide que MPS
Ces réseaux font quelques milliers de paramètres : le coût de lancement des noyaux MPS
domine. Device par défaut = **CPU**, `--device` en option. `float64` pour le solveur,
`float32` pour le PINN.

### 8.5 Si M5 échoue
Deux replis, dans cet ordre, à documenter et non à cacher : (1) donner plus
d'information (2 capteurs, ou un second freinage dans le même scénario — l'identifiabilité
conjointe de $q(t)$ et $k(T)$ depuis **un seul** capteur peut être structurellement
insuffisante, ce qui est en soi un résultat publiable) ; (2) rabattre l'acte 3 sur
« $k(T)$ inconnue, $q(t)$ connue », plus faible mais net. Ne pas basculer sur un autre cas
physique sans en parler.

### 8.6 Identifiabilité
Une section du README doit traiter la question : un capteur peut-il séparer « flux plus
fort » de « conduction plus faible » ? Réponse à donner par la mesure (courbes de
sensibilité, corrélation entre les deux sensibilités), pas par l'intuition. C'est le
pendant, pour ce dépôt, de la section « ranking ≠ accuracy » de `stress-gnn`.

### 8.7 Reproductibilité
Graine explicite partout, `torch.use_deterministic_algorithms(True)` si le coût le permet,
et **les résultats des 5 graines stockés**, pas seulement leur moyenne.

---

## 9. Règles d'honnêteté (contraignantes)

C'est ce qui donne sa valeur au dépôt, au même titre que le code.

1. **Le match nul de l'acte 1 se publie dans le Headline**, pas en note de bas de page.
2. **Les baselines sont implémentées dans leur meilleure version** ; toute simplification
   est nommée et justifiée dans le README.
3. **Aucun nombre n'est écrit à la main** dans `RESULTS.md` — tout vient de
   `03_report.py`.
4. **Rien n'est noté sur ce que la physique a effacé** : métriques en bande récupérable,
   information perdue reportée à part.
5. **La variance entre graines est publiée.**
6. **Ce que le PINN ne fait pas** est une section : plus lent qu'un FD sur le direct,
   $\lambda$ à régler, pas de garantie de convergence, pas d'estimation d'incertitude
   native.
7. Si un résultat contredit la thèse de §0, **la thèse change**, pas le résultat.

---

## 10. Ordre de démarrage

```bash
cd /Users/paul/Documents/surrogate-engineering
mkdir thermal-pinn && cd thermal-pinn
python3 -m venv .venv
./.venv/bin/pip install -e .            # après avoir écrit pyproject.toml
./.venv/bin/python run_all.py --quick   # doit tourner de bout en bout dès M1
```

Toujours `./.venv/bin/python` depuis `thermal-pinn/`, jamais un interpréteur partagé.

Ordre : **M0 → M1 → M2 → M3 → M4 → M5 → M6**, un commit par jalon (le dépôt ne commite
pas automatiquement : proposer le message, laisser l'utilisateur committer). M2 avant M3
est délibéré — la figure du mal-posé motive tout ce qui suit, et elle est peu coûteuse.

## 11. Le Headline attendu du README (gabarit à remplir par la mesure)

> Un thermocouple à 2 mm, 100 Hz, ±0,5 K. On veut le pic de température **de la surface
> de friction**.
>
> | méthode | acte 1 (linéaire) | acte 2 ($k(T)$ lisse) | acte 3 ($k(T)$ à coude, inconnue) |
> |---|---|---|---|
> | inversion naïve | *diverge* | *diverge* | *diverge* |
> | Beck / Tikhonov | `__ %` | `__ %` | non applicable |
> | adjoint (Alifanov), meilleure famille | `__ %` | `__ %` | `__ %` |
> | **PINN** | `__ %` | `__ %` | `__ %` |
>
> *(erreur sur $T_{\text{surf}}^{\max}$, moyenne sur 24 scénarios × 5 graines)*

Les trois phrases que le dépôt doit pouvoir écrire à la fin, chiffres à l'appui :
**(1)** le problème est mal posé et une bande de fréquences est perdue avant tout
traitement ; **(2)** sur le problème que la théorie classique sait poser, elle est aussi
bonne, et on le dit ; **(3)** dès qu'une **fonction** inconnue s'ajoute, la voie classique
doit deviner sa forme et se trompe, le PINN la découvre.
