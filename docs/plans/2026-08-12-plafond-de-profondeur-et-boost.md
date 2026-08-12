# Plafond de profondeur et boost par reviews — mesurés sur les deux cartes

**Date :** 2026-08-12. **Branche :** `wip/2026-08-12-depth-cap-boost-association`.
**Cartes :** `vinci-st-domingue` (index v1 converti, 1 046 404 candidats, 6 prompts,
258 annotations) et `bbhotel-choisy` (27 226 candidats, 12 prompts, 674 annotations).
**Artefacts :** `{map}/benchmark/sweep-depth-boost-2026-08-12/` (55 configurations hors
ligne au total). **Rapport :**
`~/Workspace/Current Work/PoC Vinci/reports/2026-08-12-plafond-profondeur-et-boost.html`.

Trois leviers, à retrieval identique : plafond sur la profondeur des détections, boost
des candidats par les reviews humaines, et les associations multicut / incremental
rejouées avec les deux précédents.

## 0. Ce qui a été mesuré avant d'implémenter

La distribution des profondeurs répond à moitié à la question sans lancer un run :

| carte | détections > 15 m | > 20 m | > 30 m | annotation la plus lointaine |
|---|---|---|---|---|
| vinci | 12 % | 5 % | ~0 % | **14,8 m** |
| bbhotel | 2 % | 1 % | 1 % | **6,2 m** |

Donc : 30 m ne peut rien couper, bbhotel n'a rien à couper, et **aucun vrai positif
annoté ne peut être retiré par un plafond à 15 m**. Réserve : c'est peut-être un biais
d'annotation, pas une preuve qu'il n'y a pas d'objets plus loin — cette vérité terrain
ne peut pas mesurer le coût en rappel d'un plafond serré.

## 1. Le plafond : petit gain, et gratuit en granularité

`max_depth_m`, appliqué aux détections sélectionnées, avant association. Écart de F1
macro LOO (vue stricte) contre le même réglage sans plafond, sur vinci :

| association | plafond 20 m | plafond 15 m |
|---|---|---|
| leader/canopy 2 m | +0,006 | +0,013 |
| leader/canopy 3 m | +0,011 | **−0,019** |
| multicut `pivot` 2,5 | +0,006 | **+0,017** |
| incremental somme 1,2 | +0,008 | +0,005 |

Sur bbhotel : 0,000 à 20 m, −0,003 à 15 m. Trois conclusions.

- **20 m améliore les quatre familles, 15 m dépend de la famille.** Le couple
  plafond × granularité se règle ensemble, il n'y a pas un plafond universel.
- **L'étendue médiane des clusters bouge de moins de 0,01 m.** C'est la propriété
  intéressante : avec `centroid_from="rays"`, c'est la seule intervention mesurée qui
  déplace le score sans déplacer la granularité, donc la seule lisible sans passer par
  la courbe `eps`.
- **C'est un paramètre de lieu**, comme `clustering_eps_m` : 12 % de détections à couper
  dans un aéroport, 2 % dans un hôtel. Ne pas le porter en production comme constante.

Le meilleur point de la session est **multicut 2,5 + plafond 15 m** (pair F1 0,856,
mAP groupée 0,544, F1 LOO 0,289 contre 0,272 sans plafond).

## 2. Le boost : le plus gros effet mesuré, et le moins concluant

Le sweep sait désormais le mesurer (`--with-feedback`) : les prototypes sont résolus une
fois avec les gains à zéro, et α, β et la normalisation sont balayés hors ligne sur ce
seul cache — le boost est affine dans les colonnes brutes et la normalisation est une
fonction pure du jeu retrouvé.

AP stricte par prompt, vinci, `eps` 3 m :

| prompt | sans | α=β=0,1 | α=β=0,2 |
|---|---|---|---|
| e gates | 0,046 | 0,165 | 0,242 |
| x ray machine | 0,621 | 0,724 | 0,746 |
| emergency power plant | 0,821 | 0,869 | 0,929 |
| FIDS | 0,272 | 0,306 | 0,311 |
| check in counter / kiosk | — | *identique au bit* | *identique au bit* |

**C'est de l'in-sample par construction.** Les reviews et la vérité terrain viennent de
la même carte et de la même requête ; aucune séparation n'est possible avec les données
actuelles. Le seul chiffre non ajusté dit d'ailleurs l'inverse de l'AP : le F1 macro en
leave-one-prompt-out **baisse** (0,313 → 0,233 à α=0,1), parce que décaler les scores de
quelques prompts seulement rend le seuil partagé plus difficile à transférer. Il remonte
à α=0,2 (0,344) — probablement un régime où tous les prompts boostés basculent ensemble,
à confirmer avant d'y croire.

**Vérifier la couverture avant de lire quoi que ce soit.** 8 prompts sur 12 à bbhotel et
2 sur 6 à vinci ne résolvent **aucun** prototype : les `target_id` sont des BIGSERIAL
que la réingestion ne préserve pas. `_log_feedback_coverage` le journalise à chaque run ;
sans ça, un boost inerte et un boost inutile produisent la même ligne.

## 3. Associations — rien de nouveau, et c'est le résultat attendu

Conforme au 12/08 : sur vinci multicut `pivot` 2,5 domine en pair F1 (0,856 contre 0,799
pour le défaut), incremental somme 1,2 fait aussi bien à granularité plus fine, et les
deux se composent avec le plafond sans interférence.

## 4. Ce qu'il reste à faire

1. Ne pas porter le plafond en production comme constante — le dériver du `venue_type`
   ou de l'espacement des keyframes, décidé sur au moins trois cartes.
2. Mesurer le boost hors échantillon : annoter la requête A et évaluer sur la requête B
   du même objet, ou réserver deux prompts de vinci dont les reviews ne servent jamais.
3. Refaire la mesure du plafond quand la vérité terrain contiendra des objets lointains.
