# Session 2026-08-11 — index et fil conducteur

Quatre volets, une seule carte (`bbhotel-choisy`), une branche
(`wip/improve-search-strategy`). Ce fichier est l'entrée : il dit dans quel ordre lire,
ce qui est décidé, et ce qui a été **rétracté en cours de route**.

| volet | document | commits |
|---|---|---|
| 1. Sept tentatives géométriques, défauts de données | [`…-object-search-geometry-investigation.md`](2026-08-11-object-search-geometry-investigation.md) | `78b98a4` |
| 2. Score interprétable, double porte | [`…-scoring-ratio-and-two-gate-association.md`](2026-08-11-scoring-ratio-and-two-gate-association.md) | `b33b742`, `a0a9a7c`, `0a69315` |
| 3. MetaCLIP2 vs SigLIP2, langue du prompt | [`…-embedding-model-comparison.md`](2026-08-11-embedding-model-comparison.md) | `1cf34f6` |
| 4. Rayon de clustering, métrique dégénérée | [`…-clustering-radius-and-a-degenerate-metric.md`](2026-08-11-clustering-radius-and-a-degenerate-metric.md) | ce commit |

---

## Le fil

Le point de départ était le volet 1 : sept méthodes géométriques d'association essayées,
sept échecs, avec une cause commune identifiée — rien ne garantit qu'une détection, ni son
centre ni son contenu, corresponde à l'objet.

Le volet 2 est parti d'ailleurs, du **score**. Constat mesuré : la moitié du budget de
poids de `match_score` (0.15 confidence + 0.35 keyframes) ne classait rien, les termes
étaient saturés chez deux tiers des clusters, la taille était comptée trois fois, et la
normalisation dépendait d'un paramètre de filtrage. Remplacé par un ratio à la meilleure
similarité de la requête — un terme, aucun paramètre libre. Gain cross-validé sur les
prompts : +0.078 de macro F1. La géométrie est sortie du score pour devenir des filtres.

Ce changement a eu une conséquence non prévue et décisive : **le score étant sans échelle,
deux modèles d'embedding devenaient comparables**. C'est ce qui a rendu le volet 3
possible. Verdict : trois modèles de 86 M à 630 M de paramètres tiennent dans 0.019 de
mAP, la capacité ne paie pas (+0.005 pour 4,6× les paramètres), la fusion n'apporte que du
bruit, et la **langue du prompt** vaut plus que tout changement de modèle (+0.039 à
+0.076). Décision : traduire les prompts canoniques, ne pas intégrer SigLIP2, arrêter
d'optimiser la représentation.

Le volet 4 devait être une vérification préalable de vingt minutes avant d'investir dans
l'IoU de boîtes 3D. Il a produit le résultat le plus important de la journée, et **le plus
gênant** : `clustering_eps_m` vaut +0.135 de mAP, plus que tout le reste réuni — mais la
vue stricte et la vue groupée bougent en **sens opposés et monotones** sur toute la plage.
La métrique est dégénérée le long de l'axe découper/fusionner.

## Ce qui est décidé

1. **Le score en ratio est livré et gardé** (`b33b742`). Mesurable, cross-validé,
   sans paramètre libre. Il rend `min_similarity` et le seuil d'acceptation indépendants.
2. **La géométrie est du filtrage, pas du score** — `min_observations_per_cluster`,
   `max_cluster_spread_m`, tous deux à off faute de seuil qui se paie.
3. **Traduire les prompts canoniques en anglais** : +0.039 de mAP, aucun changement de
   modèle, d'index ni de schéma. Sous réserve de vérifier les requêtes libres.
4. **Ne pas intégrer SigLIP2**, ne pas activer la porte sémantique, ne pas tester l'IoU 3D,
   ne pas passer `eps` à 0.5 — toutes ces décisions pour la même raison : soit le gain est
   du bruit, soit il n'est pas mesurable avec la vérité terrain actuelle.
5. **Le regroupement des annotations en leader/canopy est la dépendance bloquante** de
   toute la suite géométrique.

## Ce qui a été rétracté en cours de session

Trois fois, et à chaque fois pour la même famille de raisons. C'est le vrai enseignement
méthodologique de la journée.

**Le tableau par classe en F1 à seuil global est confondu.** Découvert en diagnostiquant
`lampe`, qui semblait perdre la moitié de sa performance entre « lampe » et « lamp » alors
que l'anglais est *meilleur* (AP 0.594 contre 0.552, 78 % de recouvrement des top-1000,
mêmes vignettes). Le seuil global est choisi pour maximiser la macro et pénalise
mécaniquement toute classe dont l'optimum est ailleurs. **Règle : les comparaisons par
classe se rapportent en AP.**

**« Les objets compacts perdent avec la porte sémantique » était largement faux.**
Recalculé en AP à la porte 0.80 : `extincteur` 0.999 → 0.996, `cctv` 0.908 → 0.909,
`ascenseur` 0.929 → 0.970. Seules vraies pertes : `detecteur de fumée` et `TV`.

**« La double porte est la première méthode d'association à battre l'existant » ne tient
qu'à `eps` figé.** À `eps` = 0.5 la porte dégrade le strict. Elle défaisait un
sur-regroupement créé par le rayon ; deux mécanismes de découpage pour un besoin.

Le point commun : **trois conclusions sur quatre reposaient sur un paramètre tenu fixe
sans justification** (le seuil d'acceptation, puis `eps`). Un résultat obtenu à paramètre
figé n'est pas un résultat sur la méthode, c'est un résultat sur le réglage.

## Suite, par ordre

1. **Inspecter visuellement** une dizaine de clusters à `eps` = 0.5 contre 2 m sur
   `chaise` et `extincteur`. Le moins cher, et la métrique est muette là où l'œil ne l'est
   pas. Détermine ce qu'est « un objet », donc la vérité terrain à construire.
2. **Regrouper les annotations en leader/canopy.** Débloque toute évaluation de granularité.
3. **Traduire les prompts canoniques.** Indépendant du reste, mesurable, +0.039.
4. **Rejet des profondeurs aberrantes sur rayons montants** — prérequis pour qu'une
   deuxième carte devienne un banc, `vinci-st-domingue` étant inutilisable en l'état.
5. Seulement ensuite : reconsidérer `eps`, la porte, et le partitionnement de graphe
   (C-DOG) avec une métrique capable de les classer.
