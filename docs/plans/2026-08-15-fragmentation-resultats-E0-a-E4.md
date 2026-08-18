# Fragmentation des objets étendus — résultats E0 à E4

Date : 2026-08-15. Exécution de
[`2026-08-15-plan-de-tests-fragmentation-objets-etendus.md`](2026-08-15-plan-de-tests-fragmentation-objets-etendus.md).

**Résumé en une ligne : le symptôme est réel et désormais mesurable, mais le
diagnostic proposé est faux.** La fragmentation ne suit pas la taille de l'objet,
elle suit la dispersion des points de profondeur. E1, E2 et E3 échouent chacune sur
son propre critère d'arrêt, et E4 reste donc non entamée — comme le plan l'exige.

Toutes les mesures sont faites hors ligne sur les caches de candidats
(`benchmark/sweep-depth-boost-2026-08-12/cache`, `candidate_count` 1000,
`num_results` 400, `min_similarity` 0.15, `--near-m` 1.0), donc sans postgres ni
service ANN. Sorties : `benchmark/e0-fragmentation-2026-08-15/` et
`benchmark/e2-layout-2026-08-15/` sur les deux cartes.

**Note de méthode :** `git` fonctionne dans ce worktree, contrairement à ce
qu'annonçait le blocage préalable du plan. Rien n'a été commité pour autant.

---

## E0 — la métrique de fragmentation : positive, le diagnostic : négatif

`mean_clusters_per_annotation` est ajouté à `association_sweep.py` : pour chaque
annotation ayant ≥ 2 détections étiquetées dans `--near-m`, le nombre de clusters
distincts qui les contiennent. Ventilé par classe dans le JSON, avec en contrôle
`mean_detections_per_annotation` — sans lui le chiffre n'est pas lisible, une
annotation vue quarante fois ayant plus d'occasions d'être coupée.

Le symptôme existe et il est massif là où personne ne le cherchait :

| carte | config | median spread | fragmentation | pair F1 |
|---|---|---|---|---|
| vinci | multicut pivot 2.5 | 0.441 | **1.15** | 0.856 |
| vinci | eps 2.0 | 0.356 | 1.31 | 0.799 |
| bbhotel | multicut pivot 2.5 | 0.445 | **2.35** | 0.452 |
| bbhotel | eps 2.0 | 0.371 | 2.57 | 0.479 |

Sur bbhotel, à la granularité de production, une annotation bien couverte est en
moyenne éclatée en 2.6 clusters. Ce n'est pas anecdotique.

### Le critère de succès du plan n'est pas atteint

À la configuration de référence (`multicut pivot 2.5`), vinci donne :

| classe | fragmentation | détections/annot. | taille implicite (m) | dispersion (m) |
|---|---|---|---|---|
| **x ray machine** (le portique) | **1.27** | 19.7 | 2.93 | 0.40 |
| check in counter | 1.33 | 3.7 | 4.77 | 0.63 |
| **emergency power plant** (témoin « compact ») | **1.33** | 19.7 | 6.46 | 0.60 |
| FIDS | 1.09 | 6.3 | 2.24 | 0.39 |
| check in kiosk | 1.00 | 12.2 | 2.27 | 0.35 |
| e gates | 1.00 (n=3) | 3.7 | 2.10 | 0.41 |

Le portique n'est pas « nettement > 1 », et surtout il n'est pas le pire : le témoin
censé sortir à ~1 sort à 1.33, à égalité avec le plus fragmenté. À `eps` 3.0 le
portique tombe exactement à 1.00, mieux que toutes les autres classes.

### Sur bbhotel l'ordre est inversé, et c'est le résultat décisif

À `eps` 2.0 (fragmentation / taille implicite / dispersion) :

| classe | frag | taille (m) | portée (m) | dispersion (m) |
|---|---|---|---|---|
| extincteur | **4.35** | 0.82 | 2.48 | 1.05 |
| cctv | 3.48 | 0.31 | 2.11 | 1.00 |
| signe de sortie | 3.25 | 0.41 | 2.17 | 1.36 |
| ascenseur | 3.13 | 1.93 | 1.77 | 1.88 |
| detecteur de fumée | 2.91 | 0.26 | 1.37 | 1.81 |
| lampe | 2.18 | 0.55 | 2.83 | 0.48 |
| plante | 1.82 | 1.16 | 4.71 | 0.60 |
| table | 1.62 | 1.14 | 4.94 | 0.30 |
| **chaise** | **1.43** | 1.03 | 5.38 | 0.28 |

Les classes qui éclatent le plus sont les **petits objets** — extincteurs, caméras,
détecteurs de fumée — et les objets étendus sont les mieux tenus. Corrélations de
Spearman sur les classes ayant ≥ 5 annotations couvertes :

| | vinci (k=5) | bbhotel (k=10) |
|---|---|---|
| frag ~ taille implicite | **+0.90** | **−0.43** |
| frag ~ dispersion des points | +0.60 / +0.97 | +0.77 / +0.76 |
| frag ~ portée | +0.10 | −0.44 |
| frag ~ nombre de détections | +0.00 | −0.16 |

Deux lectures, et la seconde est la conclusion du document :

- **la taille de l'objet change de signe d'une carte à l'autre.** Règle n° 3 du plan :
  un indice de signe opposé sur deux cartes n'est pas un résultat ;
- **la dispersion des points de profondeur, elle, est du même signe et forte
  partout.** Ce qui fractionne un objet, ce n'est pas son étendue, c'est le bruit de
  profondeur — maximal sur les objets vus de près (détecteur de fumée à 1.37 m,
  ascenseur à 1.77 m, dispersion 1.8 m) et minimal sur les objets vus de loin
  (chaise à 5.4 m, dispersion 0.28 m).

Le contrôle `mean_detections_per_annotation` écarte l'explication paresseuse :
`extincteur` (19.0 détections → 4.35 clusters) et `plante` (18.1 → 1.82) ont le même
budget de détections pour un facteur 2.4 de fragmentation.

**Verdict.** Le portique n'étant pas à ~1, la clause littérale d'abandon total du plan
ne se déclenche pas ; mais la prédiction que le diagnostic devait vérifier — les
objets étendus fragmentent, les compacts non — est réfutée sur bbhotel et non
confirmée sur vinci. E1 et E2 sont menées quand même (indépendantes, peu coûteuses,
avec leurs propres critères d'arrêt) ; E3 et E4, qui reposent entièrement sur
« estimer la taille », perdent leur justification avant même d'être mesurées, et
sont mesurées quand même pour le gate d'étape 1.

**Les paniers manuels n'ont pas été construits.** L'onglet Matching demande une
session interactive. Ils sont remplacés en E3 par des paniers construits
programmatiquement à partir des annotations — même construction, reproductible, et
sur 249 paniers au lieu de deux.

---

## E1 — adjacence temporelle : arrêtée en phase 0

`pair_cue_separability.py` reçoit l'indice `|Δ video_keyframe_id|`, la **cohérence
d'ordre** (distance métrique médiane par bande de `|Δ id|`) et surtout l'**AUC
conditionnelle à la géométrie**, calculée maintenant pour *tous* les indices.

L'ordre est bien temporel sur les deux cartes : la distance médiane croît de façon
monotone avec `|Δ id|` (bbhotel/cctv : 0.73 m → 1.84 → 8.87 → 21.23 ; vinci/check in
kiosk : 0.89 → 0.99 → 0.93 → 2.11 → 134). L'expérience ne s'arrête donc pas là.

| indice | AUC brute bbhotel | AUC brute vinci | **AUC \| profondeur ≤ 2 m** bbhotel | vinci |
|---|---|---|---|---|
| distance des points de profondeur | 0.959 | 0.996 | 0.859 | 0.825 |
| distance rayon-rayon | 0.831 | 0.883 | 0.664 | 0.630 |
| cosinus des cutouts | 0.616 | 0.631 | 0.575 | 0.583 |
| **\|Δ keyframe id\|** | 0.649 | 0.640 | **0.523** | **0.560** |

Brute, l'adjacence temporelle bat le cosinus. Conditionnellement à la géométrie, elle
retombe à 0.523 sur bbhotel (3× plus de paires que vinci) et 0.560 sur vinci — soit
sous, et à peine au-dessus, du seuil d'arrêt de 0.55. Par prompt sur bbhotel les
valeurs vont de 0.421 à 0.736, sans signe stable.

**Arrêt en phase 0 conformément au plan. `multicut_temporal_weight` n'est pas
implémenté.** L'indice est ce que la phase 0 devait détecter : corrélé à la proximité
spatiale, et redondant avec elle.

---

## E2 — disposition intra-panorama : le meilleur indice jamais mesuré ici, pour un gain nul

Le calcul de `g_ij = θ_ij / H_ij` est extrait dans `localize.angular_gap_ratio`
(vectorisé) ; `matching._angular_gap` l'appelle désormais au lieu de le dupliquer.

### Phase 0 — l'indice passe le diagnostic haut la main

| | AUC brute | AUC \| profondeur ≤ 2 m | paires même-objet concernées |
|---|---|---|---|
| bbhotel | **0.909** | **0.824** | 2 142 / 30 489 (7.0 %) |
| vinci | **0.955** | **0.890** | 793 / 9 146 (8.7 %) |

C'est le seul indice du tableau qui survit au conditionnement sur la géométrie : sur
vinci il la bat même (0.890 contre 0.825). Par prompt : 0.999 sur `x ray machine`,
0.984 sur `chaise`, 0.98 sur `table` ; le plus bas est `extincteur` à 0.699.

### Phase 1 — implémenté, et sans effet mesurable

`multicut_layout_weight` (défaut `0.0` = éteint, et éteint exactement : le terme n'est
pas calculé), coût `layout_weight * (1 - g_ij)` sur les paires du **même keyframe**
uniquement, attractif sous 1 (les propositions doublons) et répulsif au-delà.

Lu contre la courbe `eps` interpolée au même `median_spread_m` :

| pivot | poids | vinci pair F1 (vs eps) | bbhotel pair F1 (vs eps) |
|---|---|---|---|
| 1.0 | 0 | 0.740 (+0.075) | 0.484 (+0.010) |
| 1.0 | 0.5 | 0.751 (+0.084) | 0.482 (+0.010) |
| 1.5 | 0 | 0.807 (+0.071) | 0.521 (+0.012) |
| 1.5 | 1.0 | 0.819 (+0.078) | 0.520 (+0.011) |
| 1.5 | 2.0 | 0.831 (**+0.084**) | 0.507 (**−0.002**) |
| 2.5 | 0 | 0.856 (+0.024) | 0.452 (−0.012) |
| 2.5 | 1.0 | 0.856 (+0.022) | 0.468 (−0.005) |

Écarts par rapport au poids nul, à granularité corrigée : **+0.009 à +0.013 sur
vinci, −0.014 à +0.007 sur bbhotel.** Signes opposés au même pivot, amplitude
partout sous la barre de 0.02, et bien sous la réserve de puissance de la règle n° 5.

Le second critère du plan — celui qu'il désigne comme le vrai — échoue plus
franchement encore : la fragmentation **monte** avec le poids sur bbhotel (2.57 →
2.67 à pivot 1.5, 2.35 → 2.54 à pivot 2.5), parce que le terme sépare surtout les
classes compactes (`cctv` 3.55 → 3.73, `lampe` 2.14 → 2.46, `signe de sortie`
3.16 → 3.31). Sur les classes étendues il fait ce qui était espéré, mais faiblement :
`table` 1.32 → 1.21, `chaise` 1.35 → 1.31, `x ray machine` 1.73 → 1.60.

**Négatif, avec une leçon.** L'indice le plus séparable jamais mesuré sur cette
donnée ne déplace rien, pour une raison structurelle : il ne concerne que 7–9 % des
paires même-objet, et sur celles-là la géométrie tranchait déjà correctement. Le code
est conservé, éteint par défaut, parce que le terme est propre et que le veto binaire
existant (`cannot_link_same_keyframe`) reste utile dans l'onglet Matching.

---

## E3 — extension observée par détection : négative sur le gate d'étape 1

Étape 1 implémentée : `merge_score.latent_cost` accepte `object_extent_m` par point
(`float | np.ndarray`), l'échelle devient `σ_i² + e_i²`, et un scalaire reproduit
l'ancien comportement au bit près (`test_merge_score.py`). `observed_extent_m` calcule
`e_i = 0.5 · range_i · hypot(aw, ah)` borné dans `[0.2, 5.0]`, `cluster_extent_m` prend
la **médiane** des membres, et `score_1v2` garde l'extension de chaque côté **fixe**
pendant l'évaluation de la fusion — chaque côté conserve la sienne dans l'union.

Le gate : `toolbox/benchmark/extent_baskets.py` construit deux familles de paniers à
partir des annotations — *solo* (les détections d'une annotation, bonne réponse : 1
cluster) et *pair* (deux annotations voisines à moins de 4 m, bonne réponse : 2
clusters) — puis les partitionne avec `gasp1v2` sous les deux politiques d'extension.

| carte | panier | politique | n | clusters moyens | exactement bon | fausses fusions/panier |
|---|---|---|---|---|---|---|
| vinci | solo | constante 1 m | 45 | 1.02 | **97.8 %** | — |
| vinci | solo | observée | 45 | 1.02 | 97.8 % | — |
| vinci | pair | constante | 25 | 1.44 | 44.0 % | 60.7 |
| vinci | pair | observée | 25 | 1.24 | **24.0 %** | **79.8** |
| bbhotel | solo | constante | 204 | 1.85 | **50.5 %** | — |
| bbhotel | solo | observée | 204 | **2.10** | **46.1 %** | — |
| bbhotel | pair | constante | 438 | 2.79 | 45.2 % | 10.7 |
| bbhotel | pair | observée | 438 | 3.17 | 49.3 % | 6.4 |

Le critère du plan est « le portique passe à 1 cluster sans que les deux chaises
fusionnent, sur les deux paniers ». Ni la première ni la seconde moitié n'est
obtenue :

- **il n'y avait rien à réparer sur vinci** : avec la constante, `score_1v2` met déjà
  97.8 % des paniers solo en un seul cluster, portique compris ;
- **l'extension observée fusionne les voisins sur vinci** (44 % → 24 % de paniers
  correctement séparés, fausses fusions +31 %) ;
- **et elle fragmente davantage sur bbhotel** (50.5 % → 46.1 %, 1.85 → 2.10 clusters).
  La baisse de fausses fusions sur les paniers pair y est mécanique : tout est coupé
  plus fin.

La cause est celle qu'E0 a identifiée. `e_i` est dominé par la portée, et la portée
est exactement ce qui gouverne le bruit de profondeur : à bbhotel les extincteurs et
détecteurs vus à 1.4–2.5 m obtiennent une extension minuscule (bornée vers 0.2–0.8 m),
donc une tolérance resserrée, précisément sur les classes qui éclataient déjà le plus ;
à vinci tout est vu à 5–8 m avec des boîtes larges, l'extension monte à 2–6 m et les
voisins fusionnent. **L'estimateur est anti-corrélé au besoin sur les deux cartes.**

**Étape 2 (`association="gasp1v2"`) non entamée** — c'est le gate explicite du plan, et
il est négatif.

---

## E4 — quadriques contraintes : non entamée, conformément au plan

Condition d'entrée : « E0 a confirmé le diagnostic **et** au moins une de E1–E3 est
positive ». E0 l'a réfuté, E1 s'est arrêtée en phase 0, E2 est neutre, E3 échoue son
gate. Aucune des deux conditions n'est remplie et rien n'a été écrit.

Ce n'est pas seulement une question de procédure. Une quadrique estime l'*étendue*
d'un objet, et E0 dit que l'étendue n'est pas la variable qui décide de la
fragmentation. La semaine que demande E4 achèterait une meilleure réponse à une
question qui n'est pas celle qui est posée.

---

## Ce que la session laisse dans le code

| fichier | changement | par défaut |
|---|---|---|
| `toolbox/benchmark/association_sweep.py` | `mean_clusters_per_annotation`, `covered_annotations`, `fragmentation_by_class` (JSON seul, avec le contrôle du nombre de détections) | toujours calculé |
| `toolbox/benchmark/pair_cue_separability.py` | indices `keyframe_delta` et `same_keyframe_gap`, **AUC conditionnelle** pour tous les indices, cohérence de l'ordre des keyframes, `--cache-dir`, `--conditional-m` | — |
| `toolbox/benchmark/extent_baskets.py` | nouveau : le gate paniers solo/pair | — |
| `toolbox/bricks/localize.py` | `angular_gap_ratio` (vectorisé, partagé), `multicut_layout_weight` | `0.0` = éteint, terme non calculé |
| `toolbox/bricks/matching.py` | `_angular_gap` délègue ; `observed_extent` sur `gasp1v2` | `False` |
| `toolbox/bricks/merge_score.py` | extension par point, `observed_extent_m`, `cluster_extent_m`, `EXTENT_BOUNDS_M` | scalaire = comportement identique |
| `toolbox/bricks/service.py` | `observed_extent` sur `/matching` | `False` |

Rien ne change de défaut. `pytest` 231 passés ; `ruff` propre sur les fichiers
touchés ; `mypy` ne signale que six erreurs préexistantes sur des lignes non modifiées.

---

## La question à poser à la place

Si la fragmentation suit la dispersion des points de profondeur et non la taille des
objets, l'intervention utile n'est pas d'estimer l'étendue mais de **modéliser le
bruit de profondeur en fonction de la portée** — `merge_score._sigmas` le fait déjà
linéairement (`0.5 + 0.05·r`), et les mesures ci-dessus disent que c'est le mauvais
sens : la dispersion observée *décroît* avec la portée sur bbhotel (ρ = −0.44), alors
que `σ_i` la fait croître. Un `σ(r)` recalibré sur ces données est un jour de travail
et se teste avec exactement le même harnais de paniers.

Avant ça, une vérification à mener : la dispersion des objets vus de près est-elle du
bruit de profondeur, ou de la parallaxe (l'objet occupe une grande partie du champ, le
centre de la boîte glisse) ? C'est le mécanisme du plan d'origine, mais appliqué au
rapport taille/portée plutôt qu'à la taille seule — et `angular_width` le mesure déjà.
Un `pair_cue_separability` avec l'indice « portée » séparerait les deux en quelques
minutes.
