# Plan de tests — fragmentation des objets étendus

Date : 2026-08-15. Destiné à un agent codeur. Suite conceptuelle de
[`2026-08-12-vinci-la-granularite-optimale-depend-de-la-carte.md`](2026-08-12-vinci-la-granularite-optimale-depend-de-la-carte.md).

## Le problème, formulé pour être mesurable

Deux machines à rayons X produisent plusieurs clusters avec `gasp`, un peu moins avec
`gasp1v2`. Le diagnostic de travail est que **ce n'est pas du bruit mais un biais** : le
centre d'une bbox n'est pas un point 3D fixe, il glisse sur l'objet quand le point de vue
change. Sur un portique de 3 m, deux keyframes séparés de 40° visent deux points
matériels différents. Aucun estimateur robuste ne corrige un biais — `σ_i`, l'extension
`e = 1 m` et le Huber de `merge_score.py` le traitent tous comme de l'aléa.

Corollaire : la vraie question n'est pas « fusionner plus » mais **« estimer la taille de
l'objet au lieu de la supposer »**. Un portique et une chaise ne demandent pas le même
seuil parce qu'ils n'ont pas la même taille, et cette taille est observable.

Quatre expériences testent ce diagnostic, de la moins chère à la plus lourde. **E0 est
obligatoire avant les autres** : aujourd'hui le symptôme est anecdotique, aucune métrique
du banc ne l'exprime directement.

---

## Règles valables pour toutes les expériences

1. **Rien ne change de défaut.** Tout est opt-in, comme les modes d'association
   existants. La ligne de `AI_CONTEXT/bricks.md` — « quand ils divergent, la production
   gagne » — s'applique.
2. **Aucune lecture de mAP seule.** Strict et groupé bougent en sens opposés et monotones
   avec la granularité. Toute comparaison se lit contre la courbe `eps` interpolée au même
   `median_spread_m`. C'est la méthode réutilisable des sessions du 11–12/08.
3. **Deux cartes ou rien.** `bbhotel-choisy` (hôtel, 674 annotations, 12 prompts) et
   `vinci-st-domingue` (aéroport, 258 annotations, 6 prompts, index v1 converti). Un
   résultat sur une seule carte n'a pas de valeur — l'optimum de `eps` y vaut 1.25–1.5 m
   contre ~3 m, c'est le précédent qui justifie cette règle.
4. **Diagnostic avant implémentation.** `pair_cue_separability.py` mesure l'AUC d'un
   indice par paires en quelques minutes. Il avait prédit correctement l'échec du cosinus
   sémantique (AUC 0.529) et de la géométrie par rayons (0.768 contre 0.879 pour la
   profondeur). Tout nouvel indice y passe **avant** d'être codé dans une association.
5. **Réserve de puissance.** Sur vinci, 181 détections seulement tombent à moins d'1 m
   d'une annotation. Un écart de 0.05 est crédible, un écart de 0.005 ne l'est pas.
6. `scripts/agent-lock.sh` avant toute écriture (voir `AI_RULES.md`).

**Blocage préalable :** `git` ne fonctionne pas dans ce worktree — le `gitdir` pointe sur
`.../wemap-vision-object-search-dev/.git/worktrees/toolbox-ui-sober` qui n'existe plus. À
réparer avant toute tâche demandant un commit.

---

## E0 — Rendre le symptôme mesurable (obligatoire, ~½ journée)

**Hypothèse.** Aucune métrique actuelle ne dit « cette annotation a été éclatée en N
clusters ». `pair_recall` l'encode mais moyenné sur tout, donc un portique fracturé et
dix chaises correctement séparées se compensent.

**À implémenter.** Dans `toolbox/benchmark/association_sweep.py`, le code qui étiquette
chaque détection par l'annotation la plus proche dans `--near-m` existe déjà (il alimente
`pair_precision`/`pair_recall`). Ajouter deux colonnes dérivées du même étiquetage :

- `mean_clusters_per_annotation` — pour chaque annotation ayant ≥ 2 détections
  étiquetées, le nombre de clusters distincts qui les contiennent ; moyenné.
- La même chose **ventilée par classe d'annotation**, écrite dans le JSON détaillé et pas
  dans le CSV de résumé.

**Critère de succès.** Sur vinci, `x ray machine` doit sortir avec
`mean_clusters_per_annotation` nettement > 1 à la configuration de référence
(`multicut pivot 2.5`), et `e gates` ou `emergency power plant` proches de 1. Si le
portique sort déjà à ~1, le diagnostic est faux et **tout le reste du plan est à
abandonner** — c'est le point de contrôle le plus important du document.

**Piège.** Cette métrique voit uniquement les objets bien couverts, comme
`pair_precision`. Elle classe des associations, elle ne mesure pas le rappel de la carte.

**En parallèle, à la main.** Construire dans l'onglet Matching deux paniers de référence
sur vinci — un portique complet, deux chaises voisines — les annoter, et lancer
`Compare 7 methods`. Ça donne un cas de non-régression lisible pour E1 à E4, avec
`falseMerge` et VoI split/merge déjà en place.

---

## E1 — Adjacence temporelle comme attribut d'arête (~1 jour)

**Hypothèse.** Les keyframes viennent d'une capture parcourue, donc ils sont **ordonnés**,
et `EnrichedCandidate.video_keyframe_id` est l'indice du manifeste, c'est-à-dire l'ordre
de capture. Deux détections dans des keyframes consécutifs sont plus probablement le même
objet que deux détections spatialement proches mais issues de passages différents. Rien
dans le graphe n'utilise cette information.

### Phase 0 — diagnostic, avant tout code d'association

Étendre `pair_cue_separability.py` avec l'indice `|Δ video_keyframe_id|` et produire son
AUC à côté des trois existants. **Deux vérifications indispensables** :

- l'AUC brute de l'indice temporel ;
- son **AUC conditionnelle à la géométrie** — restreinte aux paires dont la distance de
  points de profondeur est déjà sous 2 m. C'est la seule qui compte : l'adjacence
  temporelle est fortement corrélée à la proximité spatiale, et un indice qui n'ajoute
  rien une fois la géométrie connue est inutile même avec une bonne AUC brute.

Vérifier aussi que l'ordre est bien temporel sur les deux cartes : tracer la distance
métrique en fonction de `|Δ id|`. Une carte capturée en plusieurs sessions peut avoir des
sauts. Si la relation est incohérente, l'expérience s'arrête ici.

**Critère d'arrêt.** AUC conditionnelle < 0.55 → abandonner, ne rien coder de plus. C'est
le seuil auquel le cosinus sémantique a été jugé, et il s'est révélé prédictif.

### Phase 1 — implémentation

Dans `LocalizationParams` : `multicut_temporal_weight: float = 0.0` (exact zéro = off,
comme `multicut_sem_weight`) et `multicut_temporal_scale: float = 10.0` (en indices de
keyframe). Dans `cluster_detections_multicut`, ajouter au coût :

```
temporal_weight * (1 - |Δid| / temporal_scale)
```

Même forme linéaire log-odds que les deux termes existants, pour rester sommable. Le
terme doit être **exactement inerte** quand le poids est nul, y compris en ne calculant
rien — pas seulement en ajoutant zéro.

**Mesure.** Grille sur `association_sweep.py` : `temporal_weight` ∈ {0, 0.25, 0.5, 1.0} ×
`geo_pivot` ∈ {1.0, 1.5, 2.5}, deux cartes. Lecture contre la courbe `eps` au même
`median_spread_m`.

**Succès.** +0.02 de pair F1 à granularité égale, **de même signe sur les deux cartes**.
**Échec.** Signes opposés ou |écart| < 0.01 → l'indice est redondant avec la géométrie,
documenter le négatif et passer à E2.

---

## E2 — Disposition intra-panorama comme évidence positive (~1–2 jours)

**Hypothèse.** L'arrangement angulaire à l'intérieur d'un 360 est la mesure la plus propre
du pipeline : pas de profondeur, pas de pose, juste des angles sur une sphère unique. Deux
détections à 12° d'écart dans une vue sont à 12°, que la profondeur soit fausse ou non.
Aujourd'hui cette information n'existe que comme **veto binaire**
(`cannot_link_same_keyframe`, seuil 1.5) et uniquement dans l'onglet Matching.

**À implémenter.** Transformer le veto en **coût continu** et le porter dans
l'association. `matching.cannot_link_pairs` calcule déjà l'écart normalisé
`g_ij = θ_ij / H_ij` ; en extraire le calcul de `g_ij` dans une fonction réutilisable
(`matching.py` reste l'appelant, pas de duplication) et l'exposer à
`cluster_detections_multicut` via `multicut_layout_weight: float = 0.0`. Coût ajouté sur
les paires de même keyframe uniquement :

```
layout_weight * (1 - g_ij)
```

soit attractif pour des boîtes qui se recouvrent (`g < 1`, les propositions doublons
YOLO/GDINO que l'association doit fusionner) et répulsif au-delà, sans falaise. Les paires
de keyframes différents ne reçoivent rien : l'indice ne les concerne pas.

`angular_width` et `angular_height` sont déjà sur `EnrichedCandidate` — aucun chargement
supplémentaire, contrairement au terme sémantique qui exige `with_embeddings=True`.

**Mesure.** Grille `layout_weight` ∈ {0, 0.5, 1, 2} × `geo_pivot` ∈ {1.0, 2.5}, deux
cartes. Comparer aussi contre le veto dur existant à `cannot_link_same_keyframe=True`,
pour savoir si le continu bat le binaire ou seulement le remplace.

**Succès.** Gain de pair F1 à granularité égale sur les deux cartes, **et** baisse de
`mean_clusters_per_annotation` sur les classes étendues (E0). Le second critère est le
vrai : c'est celui qui dit que les portiques se recollent.

**Piège.** Multicut est le seul mode qui autorise deux détections du même keyframe à
fusionner. Ce terme ne s'applique donc qu'à lui — ne pas tenter de le porter dans
`leader_canopy`, qui n'a pas ces arêtes.

---

## E3 — Extension par détection au lieu de la constante de 1 m (~2 jours)

**Hypothèse.** `DEFAULT_OBJECT_EXTENT_M = 1.0` est le paramètre qui fractionne les
portiques (trop petit) et qui interdit de l'augmenter (les chaises fusionneraient). Or la
taille est observable : `angular_width`, `angular_height` et la portée
`‖p_i − o_i‖` donnent une taille métrique implicite par détection.

**Prérequis P1.** `score_1v2` n'existe que dans l'onglet Matching. **Valider d'abord sur
paniers** avec `Compare 7 methods` (les baskets de E0) avant de porter quoi que ce soit :
si l'extension observée ne recolle pas le portique sur un panier construit à la main, elle
ne le fera pas à l'échelle.

**À implémenter, étape 1 (paniers).** Dans `merge_score.latent_cost`, accepter un
`object_extent_m` **par point** (`float | np.ndarray`) au lieu d'un scalaire ; l'échelle
devient `σ_i² + e_i²`. Rétrocompatible : un scalaire garde le comportement actuel au bit
près, et un test doit le pincer. Dans `_agglomerate_by_score`, calculer

```
e_i = 0.5 * range_i * hypot(angular_width_i, angular_height_i)
```

borné dans un intervalle raisonnable (`[0.2, 5.0]` m, à documenter comme garde-fou contre
une profondeur aberrante, pas comme un réglage).

**Point de vigilance conceptuel.** L'extension d'un *cluster* n'est pas la moyenne des
extensions de ses membres. Prendre la **médiane** : une détection partielle (objet coupé
au bord du panorama) sous-estime la taille, et la moyenne se laisse tirer. Et garder
l'extension **fixe pendant l'évaluation d'une fusion** — la laisser recalculer sur l'union
rétablit exactement le défaut que la constante existait pour éviter, deux voisins qui
grandissent en un gros objet plausible. C'est écrit dans le docstring de `merge_score.py`,
ça reste vrai.

**À implémenter, étape 2 (association), seulement si l'étape 1 est positive.** Porter
`gasp1v2` comme mode d'association : `association: Literal[..., "gasp1v2"]` dans
`LocalizationParams`, branché sur `signed_clustering` / `merge_score`, et ajouté à la
grille du sweep. C'est le plus gros diff du plan ; ne pas l'entamer sans le feu vert de
l'étape 1.

**Succès.** Sur paniers : le portique passe à 1 cluster sans que les deux chaises
fusionnent, sur les deux paniers de référence. Sur le sweep : baisse de
`mean_clusters_per_annotation` sur les classes étendues **à `falseMerge` constant ou en
baisse**. Les deux ensemble, sinon c'est juste un déplacement de granularité.

---

## E4 — Quadriques contraintes (~1 semaine, prototype hors ligne uniquement)

**Hypothèse.** Une bbox est un **cône tangent**, pas l'observation d'un point. Une
ellipsoïde a 9 degrés de liberté et son extension est *estimée*, pas supposée — ce qui
supprime le paramètre de granularité au lieu de le régler. C'est la réparation
principielle du biais décrit en tête de document. Références : QuadricSLAM
([1804.04011](https://arxiv.org/abs/1804.04011),
[1708.00965](https://arxiv.org/abs/1708.00965)), ODAM
([2108.10165](https://arxiv.org/abs/2108.10165), le plus proche du cas présent : poses
connues, super-quadriques sous contraintes multi-vues et prior d'échelle), MCOO-SLAM
([2506.15402](https://arxiv.org/abs/2506.15402), omnidirectionnel).

**Livrable : un script d'analyse et un rapport, pas un mode d'association.** Sous
`toolbox/benchmark/`, un script qui prend un panier annoté et ajuste une quadrique
contrainte par cluster, puis rapporte l'extension estimée contre l'extension réelle
mesurée sur les annotations.

**Contraintes à imposer dès le départ, sans quoi le problème est mal conditionné.**
L'estimation de quadrique à partir de bboxes est notoirement instable à faible parallaxe —
l'ennemi de toutes les tentatives précédentes (21.9 % des paires même-objet ont un point
d'approche derrière une caméra). Trois garde-fous, tous disponibles ici :

- **axes alignés sur la gravité** — l'axe *up* EUS, `eus_xyz[:, 1]` ; ramène 9 DDL à 7 ;
- **initialisation par les points de profondeur**, ce que cette littérature peine à
  obtenir et que le pipeline a déjà ;
- **prior d'échelle** façon ODAM, régularisation plutôt que constante figée.

**Piège technique majeur, à lever en premier.** L'algèbre des quadriques duales suppose une
projection **perspective**. Les bboxes sont sphériques (θ, φ, extensions angulaires). Deux
options : travailler dans le cutout perspectif que `prepare` construit déjà pour MetaCLIP
— simple, réutilise l'existant, mais un portique vu de près subtend un angle large et le
cutout le distordra — ou reformuler le cône tangent en géométrie sphérique. **Trancher ce
point avant d'écrire l'optimiseur**, il conditionne tout le reste.

**Succès.** Extension estimée corrélée à l'extension réelle sur les deux paniers de
référence, avec le portique nettement au-dessus des chaises. Un rapport négatif est un
résultat acceptable et attendu au moins une fois.

**Ne pas entamer** avant que E0 ait confirmé le diagnostic et qu'au moins une de E1–E3 ait
donné un signe positif. E4 change la représentation : `spread_m`, `centroid_from` et
`max_cluster_spread_m` devront tous être repensés, et ce coût ne se justifie que sur un
diagnostic établi.

---

## Ordre et points de sortie

| # | Expérience | Coût | Bloque quoi | Sortie si négatif |
|---|---|---|---|---|
| E0 | métrique de fragmentation | ½ j | tout | **abandonner le plan entier** |
| E1 | adjacence temporelle | 1 j | rien | documenter, passer à E2 |
| E2 | disposition intra-panorama | 1–2 j | rien | documenter, passer à E3 |
| E3 | extension par détection | 2 j | E4 | reconsidérer le diagnostic |
| E4 | quadriques contraintes | 1 sem | — | rapport négatif publié |

E1 et E2 sont indépendantes et peuvent être menées en parallèle par deux agents (lock
oblige). E3 dépend de E0. E4 dépend de E0 et d'au moins un signe positif ailleurs.

Chaque expérience produit une section datée dans `docs/plans/`, avec le tableau trié par
`median_spread_m` et les négatifs écrits aussi explicitement que les positifs — c'est la
convention des sessions du 11–12/08 et c'est ce qui a évité de réimplémenter deux fois le
terme sémantique.
