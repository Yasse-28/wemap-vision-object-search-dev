# La granularité optimale dépend de la carte — les méthodes du 11–12/08 mesurées sur vinci

**Date :** 2026-08-12. **Carte :** `vinci-st-domingue` (aéroport, 74 988 keyframes).
**Index :** le v1 converti (`toolbox.bricks.v1_index_convert`, 1 063 142 lignes sur
11 340 keyframes, ingéré sans re-thinning : `--min-distance 0.05` → 1 046 404
candidats). **Vérité terrain :** `benchmark/annotations.geojson`, 258 annotations,
6 prompts.

Toutes les méthodes du 11 et 12/08 sont rejouées ici sur une **deuxième carte**. Le
tableau de référence de [`AI_CONTEXT/bricks.md`](../../AI_CONTEXT/bricks.md) a été
établi sur `bbhotel-choisy` seul, et sa question ouverte était explicitement « est-ce
qu'un seuil transfère d'une carte à l'autre ». Réponse : **non pour `eps`, oui pour le
signe de tout le reste.**

Artefacts : `{map}/benchmark/association-sweep-2026-08-12/` (balayage hors ligne,
30 configurations) et `{map}/benchmark/gt-2026-08-12/<config>/` (6 runs HTTP complets
contre la vérité terrain).

## 1. Balayage hors ligne, trié par granularité

`association_sweep.py`, `num_results 400`, `min_similarity 0.15`,
`candidate_count 1000`, `--near-m 1.0`, `--group-annotation-radius-m 2.0`.
Lecture obligatoire par `median_spread_m` : le banc ne sait pas classer deux
granularités, seulement deux méthodes à granularité égale.

| configuration | spread méd. | pairP | pairR | **pairF1** | mAP stricte | mAP groupée |
|---|---|---|---|---|---|---|
| `eps` 0.55 | 0.116 | 0.956 | 0.366 | 0.476 | 0.403 | 0.397 |
| multicut `pivot` 1.0 | 0.196 | 0.904 | 0.661 | 0.740 | 0.421 | 0.428 |
| incremental sum δ 1.35 | 0.222 | 0.875 | 0.798 | 0.809 | 0.405 | 0.473 |
| `eps` 1.25 | 0.246 | 0.872 | 0.622 | 0.691 | 0.429 | 0.475 |
| **multicut `pivot` 1.5** | 0.278 | 0.860 | 0.828 | **0.807** | 0.444 | 0.485 |
| multicut `pivot` 1.5 + `sem` 0.5 | 0.280 | 0.862 | 0.827 | 0.809 | 0.451 | 0.490 |
| incremental sum δ 1.2 | 0.283 | 0.851 | 0.860 | 0.816 | 0.413 | 0.507 |
| `eps` 1.5 | 0.288 | 0.850 | 0.741 | 0.750 | 0.435 | 0.493 |
| porte 0.85 | 0.301 | 0.847 | 0.275 | 0.405 | 0.440 | 0.453 |
| incremental sum δ 1.05 | 0.324 | 0.824 | 0.890 | 0.817 | 0.411 | 0.515 |
| incremental conj. 0.80 | 0.335 | 0.828 | 0.466 | 0.582 | **0.456** | 0.489 |
| porte 0.80 | 0.341 | 0.825 | 0.500 | 0.612 | 0.453 | 0.483 |
| `eps` 2.0 *(défaut)* | 0.356 | 0.810 | 0.859 | 0.799 | 0.422 | 0.502 |
| **multicut `pivot` 2.5** | 0.441 | 0.825 | 0.965 | **0.856** | 0.420 | **0.536** |
| `eps` 3.0 | 0.518 | 0.765 | 0.949 | 0.820 | 0.391 | 0.519 |
| cdog δ 0 / δ 0.5 | 0.574 / 0.576 | 0.540 / 0.657 | 0.905 / 0.789 | 0.623 / 0.638 | 0.228 / 0.273 | 0.381 / 0.416 |
| multicut `pivot` 3.5 | 0.576 | 0.783 | 0.989 | 0.842 | 0.395 | 0.539 |
| `eps` 4.0 | 0.661 | 0.670 | 0.952 | 0.757 | 0.385 | 0.530 |
| multicut ray `pivot` 1.0 | 0.754 | 0.533 | 0.896 | 0.624 | 0.319 | 0.468 |
| `eps` 5.0 / 6.0 / 8.0 | 0.798–1.082 | 0.627 → 0.583 | ~0.95 | 0.737 → 0.668 | 0.332 → 0.282 | 0.483 → 0.434 |

`centroid_from="rays"` ne change aucun cluster, donc aucune colonne de paires : il
apparaît au §3. `level_strategy="median"` donne des résultats **identiques au bit**
au défaut sur cette carte.

## 2. Ce que ça dit, et qui contredit bbhotel

**L'optimum de granularité est à `eps` ≈ 3 m ici, contre 1.25–1.5 m sur
bbhotel-choisy.** Le pic de pair F1 est encadré des deux côtés (0.799 à 2 m, 0.820 à
3 m, 0.757 à 4 m), donc c'est bien un optimum intérieur et pas une borne du balayage.
La cause est le contenu : portiques, banques d'enregistrement et murs d'écrans FIDS
sont des objets étendus, et les keyframes sont espacés de 1,8 m (médiane) contre
beaucoup moins dans un hôtel.

Conséquence directe : la recommandation « `clustering_eps_m` 2.0 → 1.5 » issue de
bbhotel **ne transfère pas** — elle coûte 0.049 de pair F1 ici (0.799 → 0.750). Un
défaut global unique n'est pas le bon objet ; c'est un paramètre par carte, ou dérivé
du `venue_type` / de l'espacement des keyframes. **Ne pas porter ce changement en
production sur la base d'une seule carte.**

**À granularité égale, multicut et incremental battent leader/canopy — et beaucoup
plus nettement qu'ailleurs.** Contre la courbe `eps` interpolée au même spread :

| | spread | pair F1 | courbe `eps` au même spread | écart |
|---|---|---|---|---|
| multicut `pivot` 1.5 | 0.278 | 0.807 | ~0.745 | **+0.062** |
| incremental sum δ 1.2 | 0.283 | 0.816 | ~0.748 | **+0.068** |
| multicut `pivot` 2.5 | 0.441 | 0.856 | ~0.810 | **+0.046** |
| multicut `pivot` 3.5 | 0.576 | 0.842 | ~0.786 | **+0.056** |

Sur bbhotel le même écart valait +0.013 à +0.021, à la limite du bruit. Ici il est
d'un ordre de grandeur au-dessus et va dans le même sens pour les deux familles, dont
le point commun est l'**assignation au meilleur cluster** plutôt qu'au premier trouvé.
C'est le seul résultat de la journée qui mérite une deuxième carte de confirmation.

Le poids sémantique continu de multicut est **neutre** ici (+0.002 à `pivot` 1.5,
−0.005 à 3.5), là où il était monotonement nuisible sur bbhotel. Cohérent avec l'AUC
de 0.529 du cosinus cutout↔cutout : une caractéristique quasi sans information ne
change pas de signe, elle oscille autour de zéro.

**Les portes conjonctives restent un piège de mesure, en plus net.** `porte 0.80` et
`incremental conj. 0.80` donnent les **deux meilleures mAP strictes** du tableau
(0.453 / 0.456) et les **deux pires rappels de paires** (0.500 / 0.466) : elles
fragmentent les objets, et la vue stricte les paie pour le fragment le plus proche de
l'annotation. Sur cette carte, mAP groupée et pair F1 sont en revanche presque
d'accord (elles culminent toutes deux sur multicut 2.5–3.5), ce qui n'était pas le cas
sur bbhotel — le défaut de chaînage par lien simple pèse moins sur 258 annotations
majoritairement distinctes.

**C-DOG est négatif ici aussi** (0.62–0.64 à spread 0.57 contre ~0.79 sur la courbe
`eps`), et la géométrie par rayons de multicut coûte 0.23 de pair F1 (0.624 contre
0.856). Deux cartes, même verdict : la distance entre points projetés par la
profondeur bat la géométrie de rayons à faible base.

## 3. Runs HTTP complets contre la vérité terrain

`acceptance_threshold 0.5`, `min_similarity 0.15`, `candidate_count 1000`,
`num_results 100`, `group_annotation_radius_m 2.0`, index HNSW valide.

| configuration | P | R | F1 | **mAP** | mean best F1 | F1 groupé | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| `eps` 2.0 *(défaut)* | 0.228 | 0.469 | 0.307 | **0.405** | **0.501** | 0.286 | 121 | 409 | 137 |
| `eps` 3.0 | 0.235 | 0.461 | 0.311 | 0.372 | 0.479 | 0.314 | 119 | 388 | 139 |
| multicut `pivot` 2.5 | 0.236 | 0.469 | 0.314 | 0.398 | 0.497 | 0.309 | 121 | 391 | 137 |
| **multicut 2.5 + centroïde rayons** | **0.244** | **0.484** | **0.325** | 0.398 | 0.496 | **0.315** | **125** | **387** | **133** |
| incremental sum δ 1.2 | 0.239 | 0.453 | 0.313 | 0.393 | 0.497 | 0.301 | 117 | 372 | 141 |
| porte 0.80 | 0.202 | 0.442 | 0.277 | 0.428 | 0.493 | 0.237 | 114 | 451 | 144 |

AP par prompt :

| prompt | `eps` 2.0 | `eps` 3.0 | multicut 2.5 | + rayons | incr. δ1.2 | porte 0.80 |
|---|---|---|---|---|---|---|
| x ray machine | 0.559 | 0.625 | **0.637** | **0.637** | 0.581 | 0.544 |
| check in counter | 0.191 | 0.210 | 0.218 | **0.223** | 0.142 | 0.127 |
| check in kiosk | 0.566 | 0.327 | 0.377 | 0.343 | 0.484 | **0.830** |
| flight information display system | 0.186 | 0.191 | 0.202 | **0.234** | 0.217 | 0.142 |
| e gates | 0.061 | 0.058 | 0.063 | 0.063 | **0.070** | 0.064 |
| emergency power plant | 0.864 | 0.821 | **0.889** | **0.889** | 0.864 | 0.861 |

Trois lectures :

- **le centroïde par rayons est le seul gain gratuit** : mêmes clusters, donc mêmes
  colonnes de paires, et pourtant +0.011 de F1 au seuil, +4 TP et −4 FP contre
  multicut 2.5 seul. Même signe et même petitesse que sur bbhotel (+0.010 de mAP
  stricte) — c'est la seule intervention mesurée jusqu'ici qui bouge la métrique sans
  bouger la granularité ;
- **la mAP au seuil et la mAP sans seuil se contredisent**, exactement comme au §2 :
  `porte 0.80` a la meilleure mAP (0.428) et le pire F1 (0.277), parce qu'elle produit
  beaucoup de petits clusters bien classés. `check in kiosk` explique à lui seul cet
  écart (AP 0.830 contre 0.566 au défaut) : ses 25 annotations sont serrées, la
  fragmentation les sert ;
- **`eps` 3.0 gagne en groupé (0.314 vs 0.286) et perd en mAP (0.372 vs 0.405).**
  Multicut 2.5 prend les deux (0.309 / 0.398), ce qui est cohérent avec le §2 : à
  granularité comparable il domine la courbe `eps`.

## 4. Réserves

- **Base de mesure mince.** 181 détections seulement tombent à moins d'1 m d'une
  annotation, sur 6 prompts (bbhotel : ~17 000 paires). Les écarts de 0.05 sont
  crédibles, ceux de 0.005 ne le sont pas.
- **La précision absolue ne veut rien dire ici.** 409 FP pour P = 0.228 sur 258
  annotations : l'essentiel de ces « faux positifs » sont des objets réels non
  annotés. Seules les comparaisons entre lignes ont un sens, et seules mAP / mean best
  F1 / pair F1 sont des métriques de comparaison.
- **`e gates` est inexploitable** (4 annotations, AP ≈ 0.06 partout) et devrait être
  retiré du GT ou complété.
- Le contrôle « incremental best-match sans terme sémantique » a été **mal encodé**
  (`association_sim_threshold: 0.0` accepte tout : 22 clusters, spread 34 m, mAP 0).
  La ligne est écartée du §1 ; l'ablation reste à refaire.
- Mesuré sur l'index v1 converti, dont les rangées n'ont ni `detection_score` ni
  vignette. Ni l'un ni l'autre n'entre dans une métrique ci-dessus.

## 5. Ce qu'il faudrait faire ensuite

1. **Ne pas** changer `clustering_eps_m` en production : deux cartes, deux optima
   (1.25–1.5 m et ~3 m). Chercher plutôt de quoi il dépend — espacement des
   keyframes, `venue_type`, taille angulaire médiane des propositions — et le dériver.
2. Confirmer l'avantage à granularité égale de multicut / incremental sur une
   **troisième** carte. C'est +0.05 de pair F1 ici contre +0.013 sur bbhotel ; si ça
   tient, c'est le premier changement d'association qui vaut d'être porté.
3. Refaire l'ablation sans sémantique, correctement encodée, pour attribuer ce gain
   entre l'assignation au meilleur cluster et le terme sémantique.
4. Réparer le GT vinci avant de lire un chiffre absolu : compléter `e gates`, et
   décider si les annotations FIDS désignent des écrans ou des murs d'écrans (152
   annotations, 102 FN au seuil dans tous les runs).
