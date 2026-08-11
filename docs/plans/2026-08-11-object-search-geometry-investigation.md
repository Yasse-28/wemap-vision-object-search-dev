# Session — boost de feedback, banc de mesure, et sept tentatives géométriques

Date : 2026-08-11. Carte : `bbhotel-choisy` (et `vinci-st-domingue` pour la première
moitié). Le travail décrit en §1 est depuis commité (`78b98a4`, branche
`wip/improve-search-strategy`). La suite — score interprétable et double porte
d'association — est dans
[`2026-08-11-scoring-ratio-and-two-gate-association.md`](2026-08-11-scoring-ratio-and-two-gate-association.md).

**`vinci-st-domingue` ne doit plus servir de banc** : le défaut de données amont
soupçonné en §2 est confirmé (mAP 0.154 contre 0.652, `emergency power plant` 0 VP
sur 6). Toute mesure sur `bbhotel-choisy`.

Ce document existe pour qu'aucune des sept expériences négatives ci-dessous ne soit
refaite. Chacune est décrite avec sa question, sa méthode, ses chiffres et son verdict.
Les scripts sont dans le scratchpad de la session (`triangulation_residuals.py`,
`triangulation_vs_depth.py`, `pairing_diagnostics.py`, `ray_clustering_benchmark.py`,
`em_association_benchmark.py`, `keypoint_identity.py`,
`graph_association_benchmark.py`, `graph_disk_lightglue.py`).

---

## 1. Ce qui a été livré (non commité)

### Boost de feedback — normalisation

`feedback_normalization` (`"none"` défaut, `"center"`, `"standardize"`) rescale les
colonnes de prototypes sur l'ensemble récupéré avant application des gains.
Implémenté dans `candidates.normalize_prototype_similarities`, propagé jusqu'aux deux
UI et au CLI du benchmark.

**Pourquoi c'est nécessaire, mesuré** : la similarité image↔image vaut **0.691 en
médiane, écart-type 0.083**, sur toute la carte, et ne bouge pas avec la distance
(0.706 à moins de 2 m, 0.691 à plus de 50 m). Un gain brut `α·pos_sim` injecte donc
~0.69·α d'offset pour ~0.08·α de signal, sur une similarité texte↔image dont l'étendue
utile fait **0.04** (mesurée sur la recherche « lampe » : 0.204 à 0.245, du rideau
hors-sujet à la lampe parfaite). `"center"` n'est pas une option, c'est la condition
pour que le boost veuille dire quelque chose.

### Courbe précision-rappel

`precision_recall_curve` balaye le seuil sur les scores des prédictions en une passe.
Chaque prompt reçoit `average_precision`, `best_f1` et le seuil qui l'atteint ; le
résumé porte `mean_average_precision`, **moyenné par prompt et non mis en commun**
(`match_score` est normalisé par le meilleur cluster de sa propre requête, donc deux
scores de deux prompts ne sont pas sur la même échelle).

L'appariement de la courbe est **ordonné par score** (convention PASCAL/COCO), alors
que `match_predictions` reste ordonné par distance : sous l'appariement par distance,
abaisser le seuil laisse entrer une prédiction qui vole sa cible à une autre, donc le
rappel n'est pas monotone et la courbe est illisible. Les deux matchers coexistent.

### Corrections d'outillage

- l'onglet Benchmark expose α, β, `feedback_normalization`, `min_keyframes_per_cluster`
  et `default_accuracy` ; chaque run stocké affiche ses paramètres (`config-summary.ts`) ;
- `min_similarity` est envoyé explicitement par le panneau et transmis à
  « Score this prompt » (0.2 partout au lieu de 0.2 / 0.15) ;
- la branche image de `/localize` valide le même modèle que la branche texte
  (`LocalizeParams`), donc `min_similarity`, `clustering_eps_m` et
  `min_keyframes_per_cluster` y arrivent enfin ;
- quatre boutons morts supprimés du panneau (`use_stored_positions`, `robust_centroid`,
  `embedding_similarity_threshold`, `include_debug` — vestiges de `legacy/`) ;
- `min_keyframes_per_cluster` aligné à 2 partout ;
- regroupement d'annotations activé par défaut au rayon de clustering, avec
  avertissement quand les deux divergent.

---

## 2. Défauts de données trouvés

### `vinci-st-domingue` — scission de prompt (corrigé)

152 FIDS annotés, dont 122 sous le prompt `flight information display system` et 30
sans `prompt`, retombant donc sur leur classe `FIDS`. Les deux groupes désignent des
écrans **différents** (distance médiane 20.6 m). Le benchmark constituant sa vérité
terrain par prompt, évaluer `FIDS` ne retenait que 30 des 152 écrans : les 122 autres,
correctement trouvés, comptaient faux positifs. **Précision plafonnée à ~0.20.**

Corrigé dans `ground_truth_point` (30 lignes) — la source de vérité, puisque
`regenerateGroundTruth` réécrit `annotations.geojson` avant chaque run. Sauvegardes
horodatées à côté des originaux.

Reste ouvert : `tableau électrique ouvert` compte 7 annotations sous ce prompt et 1
sous `tableau électrique`. Même défaut, plus petit.

### `vinci-st-domingue` — couverture

Sur 266 annotations, 18 n'ont **aucun cluster à moins de 5 m** alors que le
regroupement portait sur *tous* les cut-outs de la carte. `emergency power plant` :
**0 sur 6**. Ce n'est pas un défaut de recherche, c'est en amont — détection ou
profondeur. (L'utilisateur a confirmé indépendamment un problème de données/parquet
sur cette carte.)

### `bbhotel-choisy` — profondeur sur rayons montants

197 des 674 annotations (29 %) ont une altitude hors de la bande de leur niveau
déclaré. Mécanisme établi :

| | profondeur stockée | profondeur **impliquée** par l'altitude |
|---|---|---|
| annotations cohérentes | 2.47 m | 2.47 m |
| annotations incohérentes | 1.16 m | **36.29 m** (p90 103 m, p10 −39 m) |

La reconstruction `alt_keyframe + depth·sin(φ)` retombe à **2 cm** sur les cohérentes.
Direction du rayon : **33 % d'incohérentes vers le haut, 7 % vers le bas**. Par classe :
ascenseur 71 %, détecteur de fumée 60 %, signe de sortie 57 %, extincteur 55 %, cctv
51 % — contre chaise 1 %, plante 0 %, table 4 %.

**Les annotations elles-mêmes sont bonnes** (posées et vérifiées à la main). C'est la
position 3D dérivée du clic qui est fausse, là où la depth map explose. Note : la
profondeur stockée dans l'enregistrement n'est pas celle ayant servi à calculer sa
position, donc l'enregistrement n'est pas auto-réparable en l'état.

**Correctif à faire** : l'outil d'annotation doit prendre le niveau depuis la pose du
keyframe source, comme `localize.py` le fait déjà (« Level is a keyframe-pose property
(depth-independent) »). Il ne le fait pas.

Autre point : 182 des 674 `source_keyframe_id` n'existent pas dans le manifeste courant
(annotations du 9 juillet, manifeste du 5 août).

### Reviews (`detection_review`)

610 reviews, dont **458 résolvent** encore vers un candidat de ce georef ; contrôle
visuel : elles sont valides (les ✓ « plante » sont des plantes). Les 152 restantes ont
des `target_id` hors plage et correspondent exactement aux requêtes **en anglais**
(`lamp`, `extinguisher`). Pour le tuning du boost : utiliser les requêtes françaises
(`plante` 205 ✓, `poubelle`, `TV`, `lampe`, `extincteur`).

---

## 3. Les sept tentatives géométriques

Référence à battre, sur `bbhotel-choisy`, prompts `lampe` / `table` / `extincteur`,
vérité terrain complète, appariement groupé à 2 m :

**profondeur (pipeline actuel) — F1@0.9 0.397, mAP 0.593, meilleur F1 0.658.**

### 3.1 Résidu de triangulation comme validateur — réfuté

958 clusters sur `bbhotel-choisy`. Distribution **unimodale** : médiane 0.51 m, p90
0.82, max 1.39. Aucune séparation vrais/faux.

Biais de sélection à connaître : les clusters étant formés à eps = 2 m, le résidu est
mécaniquement plafonné par le rayon de clustering — le clustering a déjà écarté les
rayons qui ne se croisent pas. Et 20 % des clusters n'ont que 2 keyframes, où le résidu
ne prouve rien (deux droites presque coplanaires se croisent toujours de peu).
**Une grille de validation doit exiger ≥ 3 keyframes distincts.**

Chiffre utile : erreur angulaire de visée `atan(résidu/portée)` **médiane 5.3°**, p90
12.8°. Plancher de ~2° à baseline nulle — donc pas de la dérive de pose, mais le centre
de bbox qui se déplace d'une vue à l'autre.

### 3.2 Triangulation comme estimateur de position — indistinguable (4 mesures)

| jeu | n | profondeur | triangulé | tri. meilleure | p |
|---|---|---|---|---|---|
| vinci, appariement horizontal | 248 | 0.98 m | 0.93 m | 47 % | 0.12 |
| bbhotel, appariement 3D | 619 | 1.16 m | 1.21 m | 43 % | 1.5e-3 |
| bbhotel, sous-ensemble propre | 83 | 1.45 m | 1.45 m | 36 % | 0.04 |
| **banc complet, 3 prompts** | — | **F1 0.397 / mAP 0.593** | **0.397 / 0.601** | TP/FP/FN identiques | — |

Piège méthodologique rencontré : l'appariement horizontal (haversine) apparie des
annotations avec des clusters d'**autres étages** dans un hôtel de 9 niveaux — écart
d'altitude p75 10.6 m, p90 19.4 m. Toute comparaison de positions sur cette carte doit
être en 3D.

Gain net par classe : la triangulation gagne sur les objets **compacts** (kiosques
vinci 84 %) et perd sur les **étendus** (tableau électrique 17 %, comptoirs 35 %).

### 3.3 Clustering sur croisement de rayons — pire à tous les réglages

Balayage `eps_ray` de 0.5 à 5 m. Meilleur mAP **0.503** contre 0.593. `table` donne
**0 vrai positif à toutes les valeurs**. Précision excellente quand ça groupe (1.00 sur
`lampe`, 0.78 sur `extincteur` contre 0.41) mais rappel effondré.

Bug corrigé en cours de route : paramètres de plus courte approche intervertis, ce qui
projetait toutes les intersections derrière les caméras. Validé ensuite sur cas
synthétique (rayons parfaits → résidu 8e-16).

### 3.4 et 3.5 Réassignation EM (avec puis sans création d'hypothèses) — pire

Assignation des rayons à des hypothèses par résidu **angulaire**, re-triangulation,
fusion, puis création d'hypothèses pour les détections non réclamées.

Sans création : 1000 détections → 32 objets (les détections éjectées sont perdues).
Avec création : 234 objets, F1@0.9 0.41 ≈ existant, mais **mAP 0.439** contre 0.593.
Résidus minuscules (0.02–0.33 m) : l'association converge parfaitement sur des points
qui ne sont pas les objets.

### 3.6 Graphe d'association, SIFT — gagne sur 1 classe / 3

Arête = ≥ N appariements de keypoints dont les rayons se croisent à moins de 15 cm.
Objets = composantes connexes. Position = médiane des points appariés (**pas** les
centres de boîte).

Validation préalable du signal, sur paires étiquetées par le critère de rayon :
**AUC 0.707 / 0.725 / 0.745** (lampe / extincteur / table) contre 0.546–0.613 pour les
appariements **non vérifiés**. C'est la vérification 3D qui crée le signal. Point de
fonctionnement `extincteur` : ≥1 appariement vérifié → rappel 46 %, **faux 1 %**.

Résultat au banc, seuil ≥3 :

| | lampe | table | extincteur | mAP |
|---|---|---|---|---|
| profondeur | 0.19 / 0.400 | 0.45 / 0.611 | 0.55 / 0.769 | 0.593 |
| graphe SIFT | 0.12 / 0.270 | 0.13 / 0.071 | **0.67 / 0.814** | 0.385 |

**Seul cas de toute la session où une méthode géométrique bat l'existant** :
`extincteur`, sur F1 et AP. Chaînage massif ailleurs : composante de **449 détections**
sur `table`, 145 sur `lampe`.

### 3.7 Graphe d'association, DISK + LightGlue — pire que SIFT

DISK trouve 60 à 100 % d'appariements vérifiés en plus. **Et le résultat se dégrade**,
y compris sur `extincteur` (AP 0.814 → 0.696).

Composante maximale sur `table` : **486 détections sur 500**. Sur `extincteur` : 303
contre 71 avec SIFT.

**Un meilleur matcher densifie le graphe, et un graphe dense se referme en une
composante géante. Le goulot n'était pas la qualité du matching, c'était la connexité.**

Filtre « le point apparié doit tomber dans les deux boîtes » : aide marginalement au
seuil le plus strict (lampe AP 0.262 → 0.331, table 0.071 → 0.157) et coûte sur
`extincteur`. Il ne casse pas le chaînage parce que le contenu partagé est **dans** les
boîtes : une boîte de table contient la table voisine, une boîte de lampe contient la
baie vitrée derrière.

---

## 4. Ce qui est établi

**Cause commune aux sept échecs** : rien ne garantit qu'une détection — ni son centre,
ni son contenu — corresponde à l'objet. Le centre de bbox n'est pas un point physique
stable (plancher de 2° à baseline nulle) ; les points appariés dans la boîte ne sont pas
forcément sur l'objet ; et deux objets voisins partagent assez de contenu pour que
toute méthode par connexité les fusionne.

**Axe discriminant, constant sur toutes les mesures** : compact et texturé → la
géométrie fonctionne (extincteurs, kiosques). Lisse, étendu ou vu vers le haut → elle
échoue (tables, lampes, machines à rayons X).

**Ce que le clustering des clusters produit** : sur un regroupement sans requête, une
boule de 2 m dans un salon contient un groupe de mobilier entier — 347 détections
mélangeant tables, chaises, banquettes, plantes et lampe. Les clusters d'une **requête**
sont en revanche propres, la recherche faisant le tri sémantique. C'est un argument
contre le regroupement hors-ligne à eps = 2 m.

**Ce qui n'est pas la cause** : ni les poses (résidus bornés, plancher non lié à la
dérive), ni les boîtes du détecteur (les cutouts sont serrés et montrent bien un objet),
ni la qualité du matching (DISK > SIFT et le résultat empire).

---

## 5. À faire, par rapport gain/effort décroissant

1. ~~**Optimiser `acceptance_threshold` par prompt.**~~ *Repris et dépassé.* Sur les 12
   classes, le plafond par prompt vaut +26.7 points — mais **plus de la moitié du gain
   s'obtient avec un seuil unique**, et seule cette moitié survit à la validation
   croisée. Le fit par prompt ne transfère pas, par construction. Voir le document de
   suite.
2. ~~**Agrégateur `cluster_best_sim` : `max` → quantile.**~~ *Rendu sans objet côté
   score* : le `max` était l'un des trois endroits où la taille du cluster était comptée,
   et le score n'a plus de terme de taille. Reste pertinent pour le boost.
3. ~~**Séparer le filtre du score.**~~ *Fait* (`filter_clusters_by_geometry`). Le plafond
   à 0.883 était réel mais marginal : sur les 191 prédictions de la bande (0.776, 0.9],
   15 seulement étaient des clusters à 2 keyframes.
4. **Rejet des profondeurs aberrantes sur rayons montants** à l'ingestion (ou médiane sur
   patch dans `prepare_postprocess.sample_depths`).
5. **Corriger l'outil d'annotation** : niveau depuis la pose du keyframe.
6. **Regroupement d'annotations en leader/canopy** au lieu du single-linkage, qui chaîne
   (42 lampes → 7 cibles).

Hors cadre mais seule piste géométrique non épuisée : un **partitionnement** du graphe
au lieu de composantes connexes (détection de communautés, coupes normalisées, ou modèle
à nombre d'objets explicite). C'est un sujet en soi, pas un réglage.

*Mise à jour* : la cause commune identifiée en §4 a reçu une réponse partielle. La
**double porte** (géométrie ET sémantique, conjonctives — la règle de ConceptGraphs)
bat l'existant en moyenne sur les 12 classes, et gagne précisément sur les objets
étendus qui fusionnaient. C'est la première méthode d'association de la série à y
arriver. Le partitionnement de graphe reste ouvert et a lui aussi une littérature
(C-DOG, 2025).

---

## 6. Méthode — ce qui a marché et ce qui a raté

Ce qui a marché : réutiliser le code de production plutôt que le réimplémenter ;
identifier le confondant **avant** de regarder le résultat ; normaliser pour rendre un
nombre interprétable (0.5 m → 5.3° de visée) ; vérifier comment la règle a été fabriquée
avant de s'en servir.

Ce qui a raté, et qui a coûté : **deux bugs de géométrie non testés sur cas synthétique
avant usage** (plus courte approche intervertie, initialisation EM dégénérée) ; une
mesure d'AUC d'apparence **invalidée** par des pseudo-étiquettes bruitées, découverte
seulement en regardant les vignettes ; un filtre de « cohérence » appliqué à la vérité
terrain sur un critère que le benchmark n'utilise même pas (il apparie en horizontal, il
ne lit ni l'altitude ni le niveau).

Leçon transférable : **regarder les images avant de croire un chiffre**. Les deux
retournements majeurs de la session sont venus des vignettes, pas des statistiques.
