# Résultats T0 → T8 — la profondeur est fausse, et les rayons la corrigent

Date : 2026-08-17. Exécution de
[`2026-08-17-ordre-de-tests-raffinement-des-points-de-profondeur.md`](2026-08-17-ordre-de-tests-raffinement-des-points-de-profondeur.md).

**Une phrase.** Le point de profondeur est faux **radialement**, aucune politique
d'échantillonnage sur la boîte ne le rattrape (−9 %), et le seul geste qui paie est de
ne plus s'en servir pour la position : re-placer chaque détection sur son propre rayon,
à la profondeur que triangule son cluster, retire **53 %** de la dispersion radiale et
fait passer les paniers solo de 42,9 % à 57,1 % d'exactitude. T7 (POM) reste non
justifié : son propre indice R2 n'est pas décidable sur la vérité terrain disponible.

## Mise à jour en fin de session — le jeu d'annotations a grandi

Les tableaux ci-dessous sont mesurés sur **7 groupes / 161 étiquettes**. Des paniers ont
été ajoutés pendant la session : **16 groupes / 670 étiquettes** à la fin. Les mesures
clés rejouées dessus tiennent, en direction comme en ordre de grandeur :

| | 7 groupes | 16 groupes |
|---|---|---|
| radiale / tangentielle | 0,870 / 0,570 | 0,982 / 0,680 |
| \|cos(penchant, rayon)\| médian | 0,824 | 0,806 |
| ρ(tangentiel, extension angulaire) | +0,02 | −0,10 |
| biais par vue, z (≥ 3 résidus, ≥ 2 groupes) | +1,99 | +2,94 |
| **T4 : radiale** | 0,870 → 0,406 (−53 %) | 0,982 → 0,554 (**−44 %**) |
| **T4 : tangentielle** | 0,570 → 0,530 | 0,680 → 0,618 |
| **T4 : solo exact** | 42,9 → 57,1 % | 25,0 → **37,5 %** |
| T4 : clusters écartés (parallaxe) | 0 / 13 | 0 / 40 |

Le taux de départ baisse (25 % contre 42,9 %) parce que les groupes ajoutés sont plus
gros — jusqu'à 92 détections. Le gain de T4 reste de +12 points.

## Une découverte qui conditionne la lecture de tout le reste

**La vérité terrain du benchmark n'est pas indépendante de la profondeur mesurée.**
Une annotation est créée par `workbench-index.ts` (route de résolution d'un point ERP)
comme :

```
point_annoté = rayon(u, v) × depth_map_sample(u, v)
```

C'est la construction exacte d'`object_position` pour une détection. L'humain fournit
la **direction** ; c'est la depth map qui fournit la **distance**. Les 258 annotations
vinci et les 674 bbhotel héritent donc de l'erreur qu'elles servent à mesurer, et
détection par détection elles y sont corrélées.

Conséquence : **aucune mesure absolue de justesse positionnelle n'est possible avec les
données actuelles** — uniquement des comparaisons relatives. Casser cette circularité
(annoter le même objet depuis deux keyframes éloignés et trianguler, ou reconstruire un
nuage MVS de référence) est le verrou méthodologique du dossier, avant tout choix
d'algorithme.

## L'assiette de mesure, et sa limite principale

`detection_group_label` est **vide sur bbhotel** et contient 161 lignes sur vinci —
9 groupes, dont 7 au-dessus de 4 détections, sur 2 prompts (`x ray machine`,
`check in counter`). Aucun négatif explicite (`group_name IS NULL`), donc **la famille
« mixte » du plan n'existe pas**. `row_index` est NULL partout : la résolution passe
par la jointure angulaire sur `(keyframe_id, θ, φ)`, comme `matching.resolve_items`.
158/161 étiquettes résolvent, 12 (7,6 %) ont une boîte rivale à moins de 1e-3 rad —
c'est le taux de doublons intra-vue que T1b demandait de mesurer avant de s'interpréter.

**Toutes les prédictions R7 du plan opposent vinci à bbhotel. Aucune n'est testée.**
C'est le trou du dossier, et il est bouché par de l'annotation, pas par du code : sans
paniers bbhotel, la moitié des verdicts ci-dessous est un demi-verdict.

## T1 — la dispersion est radiale, et le « systématique » n'est pas la pose

| composante | mesure (7 groupes, parallaxe ≥ 5°) |
|---|---|
| radiale | **0,870 m** |
| tangentielle | 0,570 m |
| rapport t/r | 0,68 — radial pour 6 groupes sur 7 |

Le gate du plan (« radial domine → T2, T3, puis T7 ; **pas de T5** ») s'applique. T5 et
T6 ne sont donc pas exécutés, et le contrôle n° 4 le confirme indépendamment : la part
tangentielle **ne croît pas** avec l'extension angulaire des boîtes (ρ = +0,02). Le
glissement de centre de bbox n'est pas le mécanisme.

**Correction apportée au test du plan.** Le rapport `|résidu moyen| / moyenne des
|résidus|` ne vaut pas 0 pour du bruit isotrope : il vaut environ `1/√n` pour un
keyframe portant `n` résidus. Mesuré tel quel il annonçait 0,93, ce qui aurait fait
conclure au biais de pose. Comparé à un **null par permutation** (résidus rebattus
entre keyframes, tailles de blocs conservées) :

| périmètre | k | observé | null | z |
|---|---|---|---|---|
| ≥ 2 résidus | 35 | 0,782 | 0,625 ± 0,037 | +4,28 |
| ≥ 2 résidus, ≥ 2 groupes | 22 | 0,731 | 0,616 ± 0,048 | +2,40 |
| ≥ 3 résidus, ≥ 2 groupes | 14 | 0,685 | 0,590 ± 0,048 | +1,99 |

Il y a bien un excès, mais modeste — et **ce n'est pas la pose**. Le penchant d'une vue
est aligné sur son propre rayon de visée (|cos| moyen 0,73, médian 0,82 ; une erreur
isotrope donnerait ~0,5). Une translation de caméra déplacerait les points dans
n'importe quelle direction ; une erreur de profondeur ne peut les déplacer que le long
du rayon. C'est donc de l'erreur de profondeur partagée à l'intérieur d'une vue.

## T1b — l'indice d'absence de co-visibilité : non décidable

Population : les paires de **fragments** (les clusters que `gasp1v2` découpe dans un
groupe annoté), à moins de 6 m — 22 paires, 8 « même objet », 14 « objets distincts ».
Visibilité lue dans les depth maps comme oracles d'occultation, ce qui est le bon usage
et il fonctionne.

| lecture | n | AUC absence |
|---|---|---|
| brute | 22 | **0,696** |
| conditionnelle à 2 m | 4 | 0,667 |

Intervalle bootstrap à 90 % : **[0,504 ; 0,875]**. Le seuil de 0,70 est dedans, et le
seuil d'abandon de 0,55 aussi. L'indicatrice ne se déclenche que sur 9 paires sur 22
(40,9 %).

**Verdict : le gate R2 de T7 n'est ni franchi ni refusé — il est indécidable à cette
taille d'échantillon.** Construire POM (4 à 6 jours) sur cette base violerait la règle
R2 du plan lui-même. Ce qui débloque T7 n'est pas du code : ce sont des paniers
supplémentaires, bbhotel en priorité.

## T2 — la profondeur robuste sur la boîte : échec de son propre critère

| politique | radiale | tangentielle | pixels valides | solo exact (gasp1v2) |
|---|---|---|---|---|
| `center` (référence) | 0,899 | 0,582 | 100 % | 42,9 % |
| `median` | 0,822 (−8,6 %) | 0,563 | 100 % | 42,9 % |
| `nearest_mode` | **1,275 (+42 %)** | 0,598 | 100 % | 42,9 % |
| `trimmed` | 0,818 (−9,0 %) | 0,565 | 100 % | 42,9 % |

Le critère demandait −30 % de radiale **et** des solo en hausse **et** des pair
inchangés. Aucun des trois. `nearest_mode`, la politique annoncée gagnante, est la
pire : le mode proche attrape l'avant-plan. Et **100 % de pixels valides** — l'hypothèse
du « trou dans la depth map » ne s'applique pas ici : la carte est localement cohérente
et fausse à l'échelle de l'objet. C'est ce qui explique que médiane et moyenne tronquée
ne gagnent presque rien : toute la boîte est d'accord sur la mauvaise valeur.

**Trouvaille annexe, qui vaut pour tout rejeu hors ligne.** `theta_center` et
`phi_center` stockés sont exactement des float16 — soit ~0,9 px à une largeur ERP de
5760. Relire un pixel unique hors ligne tombe donc jusqu'à un pixel à côté de ce que
l'ingestion avait lu : écart **purement radial** (tangentiel exactement nul, la chaîne
pose/rayon est exacte), médian 8 cm, p95 73 cm, max 1,35 m. Du même ordre que la
dispersion qu'on cherche à réduire.

## T4 — la re-triangulation par rayons : le seul geste qui paie

Une passe : associer avec le prior de profondeur, trianguler chaque cluster depuis les
rayons de ses membres, glisser chaque membre au pied de ce point sur **son propre
rayon**, ré-associer. 13 clusters raffinés, **0 écarté** faute de parallaxe — le gain
n'est pas obtenu en rétrécissant la population.

| | total | radiale | tangentielle |
|---|---|---|---|
| avant | 1,050 | 0,870 | 0,570 |
| après | **0,712** | **0,406 (−53 %)** | 0,530 (−7 %) |

| famille | avant | après |
|---|---|---|
| solo (n=7) | 42,9 % exact | **57,1 % exact** |
| pair (n=1) | 0 % exact, 9 fausses fusions | 0 % exact, **0 fausse fusion** |

Le plan exigeait de dire d'où vient le gain quand seule la radiale baisse : **il vient
de ce que la triangulation ignore la profondeur**, pas d'une correction du biais. C'est
cohérent avec T1 (radial), T2 (irrattrapable par pixel) et T8 (pas la pose). La famille
pair est à n=1 — cette moitié du critère n'est pas testée.

## T8 — la correction de pose conjointe : nulle hors échantillon

Offset de translation par keyframe, alternance à associations fixées, ridge vers zéro,
**validation croisée par groupe**.

| ridge | held-out avant | held-out après | gain | (entraînement après) | médiane \|offset\| |
|---|---|---|---|---|---|
| 1 | 1,050 | 1,050 | −0,0 % | 0,642 | 0,507 |
| 4 | 1,050 | 1,034 | **−1,5 %** | 0,846 | 0,223 |
| 16 | 1,050 | 1,042 | −0,7 % | 0,980 | 0,068 |

Le piège annoncé par le plan est exactement celui qu'on observe : −39 % en
entraînement, ~0 hors échantillon. Et ce n'est pas un défaut de couverture — **41,6 %
des détections viennent d'un keyframe vu dans un autre groupe**, donc l'offset avait de
quoi se transmettre. Conjugué à l'alignement |cos| = 0,82 de T1, le verdict est net :
**il n'y a pas d'erreur de pose à corriger ici**. T8 s'arrête, et la précondition
« calibration » de T7 n'est pas le blocage qu'on croyait — c'est la profondeur qui l'est.

## T3 — σ(r) a la mauvaise pente

Ajusté sur les résidus radiaux, un point par **détection** (154 détections, 1,3 à 29 m),
`e` maintenu à 1 m comme exigé :

| | σ(r) |
|---|---|
| en place | `0,500 + 0,050·r` |
| mesuré (points stockés) | `0,802 + 0,028·r` (ρ = +0,15) |
| mesuré (après la passe T4) | `0,456 + 0,007·r` (ρ = +0,05) |

Sur vinci le signe de la pente est bon — contrairement à bbhotel le 15/08 (ρ = −0,44),
et c'est un rappel de plus que ces mesures changent de signe d'une carte à l'autre.
Mais **la pente est trop raide dans les deux régimes** et la base trop petite avant la
passe : à 12 m le modèle annonce σ = 1,09 quand la mesure dit 0,53 après T4. La
correction pointée par les données est un σ **quasi constant** (~0,5 m après T4,
~0,8–1,0 m sans). Non appliqué : le critère du plan exige la validation sur les paniers
des **deux** cartes, et bbhotel n'en a pas.

## Ce qui n'a pas été fait, et pourquoi

- **T5 / T6** — fermés par le gate de T1 (le tangentiel ne domine pas, et il ne croît
  pas avec l'extension angulaire). Le plan les conditionnait explicitement.
- **T7 (POM)** — non commencé. Son gate R2 (T1b) est indécidable, sa précondition T8
  s'avère sans objet, et son ensemble d'évaluation serait 7 groupes d'une seule carte.
  Le construire maintenant produirait un résultat non mesurable.
- **Critère de succès de T0** — « mêmes étiquettes que l'onglet, pour les sept
  méthodes » : non vérifié contre l'UI, qui exige postgres et le service ANN. Le harnais
  est couvert par 32 tests unitaires, ce n'est pas la même garantie.

## L'origine de l'erreur, mesurée

Décalage radial des points de profondeur par rapport au **centre consensuel des
rayons** (16 groupes, 647 détections) :

- **12 % — biais systématique de −0,46 m**, vers la caméra ; 66,9 % des points sont
  devant le centre. Constant en mètres et **indépendant de la portée** (ρ = +0,07),
  donc ce n'est pas une erreur d'échelle multiplicative du modèle de profondeur ;
- **88 % — dispersion à queue lourde** : 10 % des détections portent 56 % du carré
  moyen, 20 % en portent 76 %.

**La queue n'est prédictible par rien de disponible** : boîte en doublon (0,79 vs
0,72 m), taille angulaire (ρ = −0,08), similarité (−0,17), portée (+0,07). C'est la
raison pour laquelle un estimateur robuste sur l'ensemble fonctionne là où aucun
filtrage par détection ne le pourrait — on ne peut pas désigner la mauvaise à l'avance.

**Hypothèse écartée** : les points ne tapissent pas la surface visible d'un objet
étendu. CV(|résidu|) = 0,50, quand une coquille tendrait vers 0 et qu'un nuage gaussien
isotrope donne 0,42. La dispersion est un nuage à queue lourde, pas une enveloppe.

**Non établi** : la cause physique de la queue. Les surfaces métalliques et vitrées
d'un aéroport sont le suspect naturel pour un modèle de profondeur monoculaire, mais
c'est une hypothèse. Le test qui la tranche est visuel et coûte deux heures — afficher
le cutout et le crop de depth map des 65 pires détections côte à côte.

## Ce qu'il faut faire ensuite, dans l'ordre

0. **Casser la circularité de la vérité terrain** (voir plus haut). Le moyen le moins
   cher est d'annoter un même objet depuis deux keyframes éloignés et de trianguler :
   la position cesse alors de dépendre de la depth map. Sans ça, tout ce dossier reste
   relatif.
1. **Regarder les 65 pires détections** — cutout et crop de depth map côte à côte, sur
   le décile qui porte 56 % de l'erreur. Deux heures, et c'est le seul test qui rende
   une cause physique plutôt qu'une corrélation.
2. **Annoter des paniers sur bbhotel** (et quelques-uns de plus sur vinci, courte
   portée surtout). Quatre des verdicts ci-dessus sont mono-carte, et le plan prédisait
   des signes opposés.
3. **Porter la passe T4** dans `localize`, derrière un paramètre par défaut inactif, et
   la mesurer sur `association_sweep` en validation finale. C'est le seul candidat
   positif du lot.
4. **Reprendre T3 après T4**, puisque recalibrer σ sur des points qu'on va remplacer est
   à refaire — et le mesurer sur les deux cartes.
5. **Ne rouvrir T7 qu'après (0) et (2)**, avec T1b re-mesuré sur un échantillon où son
   intervalle ne couvre plus tout l'intervalle de décision.

## Reproduire

```bash
M=/media/yacine/T7/Wemap/vps-data/maps/vinci-st-domingue
C=$M/benchmark/sweep-depth-boost-2026-08-12/cache
PYTHONPATH=.:third_party/object_search ~/anaconda3/envs/wemap-vision/bin/python \
  -m toolbox.benchmark.matching_baskets --map-path $M --cache-dir $C
```

Puis `covisibility_cue`, `depth_policies`, `ray_refinement`, `pose_offsets`,
`sigma_calibration` avec les mêmes deux arguments. Tout tourne hors ligne : ni postgres
ni service ANN.
