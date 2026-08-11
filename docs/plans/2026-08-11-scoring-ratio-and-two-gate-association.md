# Session — score interprétable et double porte d'association

Date : 2026-08-11, suite de
[`2026-08-11-object-search-geometry-investigation.md`](2026-08-11-object-search-geometry-investigation.md).
Branche `wip/improve-search-strategy`. Commits `b33b742` (score) et `a0a9a7c` (porte).

Toutes les mesures sur `bbhotel-choisy` : 674 annotations, 12 prompts, 1287 clusters.
`vinci-st-domingue` a été écartée — défaut de données amont confirmé, mAP 0.154 contre
0.652, `emergency power plant` 0 VP sur 6. Une carte cassée ressemble à une validation
inter-cartes tout en ne portant aucun signal.

---

## 1. Ce qui a été livré

**Le score.** `match_score = cluster_best_sim / meilleur cluster de la requête`. Un
terme, aucun paramètre libre. Remplace
`0.50·norm_sim + 0.15·confidence + 0.35·min(1, kf/3)` — supprimé, pas commuté.

**La géométrie en filtres.** `filter_clusters_by_geometry`
(`min_observations_per_cluster`, `max_cluster_spread_m`), appliqué *avant* le ranking
pour que le dénominateur du ratio soit un cluster réellement renvoyé. `spread_m` remonte
dans la réponse à côté de `confidence` et `observation_count`.

**La double porte.** `semantic_gate_threshold` : rayon spatial **et** cosinus
cutout↔cutout contre le seed — la règle conjonctive de ConceptGraphs. Off par défaut.
Embeddings chargés seulement quand la porte est active.

---

## 2. Le tableau par classe

Vérité terrain stricte, seuil d'acceptation réoptimisé par colonne (pondéré 0.776,
ratio 0.905, g0.80 0.888, g0.85 0.901, g0.90 0.884).

| classe | GT | pondéré | ratio | porte 0.80 | porte 0.85 | porte 0.90 | meilleur | gain |
|---|---|---|---|---|---|---|---|---|
| chaise | 213 | 0.140 | 0.202 | 0.310 | 0.359 | **0.793** | g0.90 | +0.653 |
| table | 77 | 0.277 | 0.400 | 0.540 | **0.702** | 0.688 | g0.85 | +0.425 |
| poubelle | 2 | 0.222 | 0.444 | 0.308 | **0.500** | 0.333 | g0.85 | +0.278 |
| lampe | 55 | 0.417 | 0.494 | **0.690** | 0.617 | 0.588 | g0.80 | +0.273 |
| plante | 22 | 0.514 | 0.650 | **0.745** | 0.741 | 0.642 | g0.80 | +0.231 |
| detecteur de fumée | 119 | 0.747 | **0.845** | 0.804 | 0.767 | 0.512 | ratio | +0.097 |
| extincteur | 29 | 0.879 | **0.935** | 0.935 | 0.829 | 0.761 | ratio | +0.057 |
| cctv | 41 | 0.800 | **0.822** | 0.817 | 0.708 | 0.706 | ratio | +0.022 |
| ascenseur | 21 | **0.927** | 0.927 | 0.900 | 0.833 | 0.500 | — | 0.000 |
| defibrillateur | 2 | **0.667** | 0.667 | 0.667 | 0.667 | 0.308 | — | 0.000 |
| signe de sortie | 88 | **0.865** | 0.768 | 0.775 | 0.758 | 0.782 | pondéré | −0.097 |
| TV | 5 | **0.727** | 0.435 | 0.294 | 0.364 | 0.333 | pondéré | −0.292 |
| **macro** | 674 | **0.598** | **0.632** | **0.649** | **0.654** | 0.579 | | |

Agrégats correspondants :

| | mAP | macro F1 | leave-one-prompt-out | mAP groupé | LOO groupé |
|---|---|---|---|---|---|
| pondéré | 0.652 | 0.598 | 0.533 | 0.715 | 0.552 |
| ratio | 0.653 | 0.632 | 0.611 | 0.713 | 0.627 |
| ratio + porte 0.80 | 0.694 | 0.649 | 0.628 | 0.692 | 0.610 |
| ratio + porte 0.85 | 0.698 | 0.654 | 0.629 | 0.639 | 0.565 |
| ratio + porte 0.90 | 0.630 | 0.579 | 0.556 | 0.559 | 0.457 |

### Trois groupes, selon l'axe compact/étendu

**Les fusionneurs** (chaise, table, lampe, plante, poubelle) gagnent massivement et de
façon monotone avec la sévérité de la porte. Le rappel est ce qui bouge : `table` passe
de 0.273 à 0.688 à précision quasi constante (0.750 → 0.716). La porte n'améliore pas la
recherche, elle arrête de coller plusieurs objets ensemble.

**Les compacts** (extincteur, cctv, détecteur de fumée, ascenseur) ont le ratio seul
pour optimum et la porte les dégrade, en précision d'abord : `extincteur` 0.879 → 0.707
de précision à rappel 1.000. Découper un cluster déjà correct fabrique des doublons.

**Deux régressions dues au score lui-même**, indépendantes de la porte : `signe de
sortie` (−0.097, 88 annotations — donc pas du bruit : précision 0.691 à rappel 0.864, le
ratio y admet trop) et `TV` (−0.292, 5 annotations, seuil propre 0.972 contre 0.905
partagé).

---

## 3. Ce qui est établi

- **Les termes de taille du score pondéré ne classaient rien.** La similarité seule vaut
  mAP 0.653 contre 0.652 pour le mélange ; `min(1, kf/3)` seule vaut 0.318. Saturés chez
  65 % (kf ≥ 3) et 53 % (n_obs ≥ 5) des clusters, donc constants. Et la taille était
  comptée trois fois : `kf/3`, `min(1, n_obs/5)` dans `confidence`, et le `max` sur N.
- **L'ancienne normalisation dépendait d'un paramètre de filtrage** (`best −
  min_similarity`). Le ratio découple : `min_similarity` est le plancher absolu, le ratio
  le portail relatif.
- **Un seuil sur la similarité brute ne marche pas.** Les optima par prompt s'étalent sur
  0.084 quand la dynamique intra-prompt fait 0.025 à 0.099 — `plante` tient entre 0.207
  et 0.231, l'optimum de `defibrillateur` est 0.290. Aucun point de fonctionnement commun.
- **La double porte marche, et exactement là où l'enquête géométrie prédisait qu'elle
  marcherait.** C'est le premier résultat de la série où une méthode d'association bat
  l'existant en moyenne. La raison est celle de la littérature : le graphe SIFT/DISK
  n'avait qu'une porte, la géométrique, et le contenu partagé dans la boîte suffisait à
  la franchir.
- **Le rescoring hors-ligne est fidèle** : reconstruction du `match_score` à 1.55e-15 sur
  2148 clusters, et le run bout-en-bout reproduit la prédiction (0.632 contre 0.632).

## 4. Ce qui n'est pas établi

- **Qu'un seuil transfère entre cartes.** Le LOO ne couvre que les prompts d'une carte.
- **Si la porte doit être activée.** Les deux vérités terrain se contredisent
  structurellement : la stricte (une cible par annotation) récompense le découpage, la
  groupée (single-linkage à 2 m, 674 annotations → 118 cibles) récompense la fusion.
  Ce n'est pas un seuil à trouver, c'est le défaut de regroupement.
- **Si le gain de `chaise` est réel.** Son optimum est le seuil le plus agressif, et à
  0.793 pour un rappel de 0.221 la métrique récompense peut-être un découpage que l'œil
  jugerait faux. À vérifier sur vignettes avant d'y croire — la leçon de la §6 du
  document précédent.
- **Aucune colonne ne domine.** `g0.85` est le meilleur réglage global et n'est le
  meilleur que pour 3 classes sur 12. La moyenne progresse parce que les gains des
  fusionneurs écrasent les pertes des compacts.

## 5. Suite, par rapport gain/effort

1. **Regroupement des annotations en leader/canopy** au lieu du single-linkage. Débloque
   la décision sur la porte et rend la vue groupée utilisable. C'est le prérequis, pas
   un nettoyage.
2. **Inspection visuelle de `chaise` et `table`** aux seuils de porte élevés.
3. **Temperature scaling** comme concurrent du ratio : calibration à un paramètre,
   apprenable, et remède publié à la non-comparabilité du cosinus CLIP entre requêtes.
   Cible le cas `TV`.
4. **Rejet des profondeurs aberrantes sur rayons montants**, prérequis pour qu'une
   deuxième carte devienne un banc.
5. Une porte réglée par classe atteindrait ~0.75 de macro — mais c'est du fit par classe,
   et on sait qu'il ne transfère pas. À ne pas confondre avec un gain.

## 6. Littérature

Ce que fait l'object search s'appelle **open-vocabulary 3D instance retrieval**.
OVIR-3D (CoRL 2023) est l'analogue structurel le plus proche mais fusionne sur la
similarité de features, pas sur la géométrie. ConceptGraphs (ICRA 2024) est la source de
la double porte. HOV-SG (RSS 2024) et KeySG (2025) font du niveau une structure d'index
au lieu d'un veto tardif — piste pour les 9 étages. C-DOG (2025) est le partitionnement
de graphe que la §4 du document précédent laissait ouvert : arêtes par consistance
épipolaire puis clustering par δ-recouvrement, pas des composantes connexes. FroDO
formule le clustering de rayons comme problème ouvert, ce qui recadre l'échec 3.3.
Aucun de ces travaux ne tourne sans reconstruction dense — c'est là qu'on est seuls.
