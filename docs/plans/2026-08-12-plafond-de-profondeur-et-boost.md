# Plafond de profondeur et boost par reviews — mesurés sur les deux cartes

**Date :** 2026-08-12. **Branche :** `wip/2026-08-12-depth-cap-boost-association`.
**Cartes :** `vinci-st-domingue` (index v1 converti, 1 046 404 candidats, 6 prompts,
258 annotations) et `bbhotel-choisy` (27 226 candidats, 12 prompts, 674 annotations).
**Artefacts :** `{map}/benchmark/sweep-depth-boost-2026-08-12/` et
`.../sweep-rescorers-2026-08-12/` (99 configurations hors ligne au total). **Rapport :**
`~/Workspace/Current Work/PoC Vinci/reports/2026-08-12-plafond-profondeur-et-boost.html`.

Cinq leviers, à retrieval identique : plafond sur la profondeur des détections, boost
des candidats par les reviews humaines, quatre méthodes de rescoring concurrentes
portées depuis les worktrees `feat/rescoring-*`, et les associations multicut /
incremental rejouées avec les précédents, et une gate de validation Qwen3-VL.

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

## 3. Cinq façons d'exploiter les mêmes reviews

Les quatre méthodes des worktrees `feat/rescoring-*` ont été portées sur cette branche
et rejouées sur le même jeu retrouvé, la même association et le même score. Leur branche
d'origine est antérieure au score en ratio, donc leurs chiffres d'alors ne sont pas
comparables à ceux-ci. Artefacts : `{map}/benchmark/sweep-rescorers-2026-08-12/`,
grilles `toolbox/benchmark/grids/rescorer-comparison-*.json`.

| méthode | vinci mAP / F1 LOO | bbhotel mAP / F1 LOO |
|---|---|---|
| sans reviews | 0,391 / 0,313 | 0,701 / 0,639 |
| `max_prototype` α=β=0,1 *(actuel)* | 0,441 / **0,233** | 0,736 / 0,653 |
| `max_prototype` **α=0,2 β=0,5 (défauts UI)** | **0,097** / 0,143 | **0,487** / 0,474 |
| `multi_prototype` k=2 | 0,438 / 0,297 | 0,734 / 0,684 |
| **`knn_cache` k=15 γ=0,2** | 0,473 / **0,410** | **0,751** / **0,711** |
| `knn_cache` k=5 | 0,468 / **0,430** | 0,695 / 0,692 |
| `linear_probe` w=1,0 | **0,524** / 0,156 | **0,761** / 0,536 |
| `graph_propagation` γ=0,2 | 0,484 / 0,390 | 0,738 / 0,658 |

- **Le vote kNN est la seule méthode qui améliore ce qui transfère** (+0,117 et +0,072
  de F1 LOO), là où le boost actuel le dégrade sur vinci. Même signe sur les deux cartes.
- **Les défauts affichés par l'UI sont nocifs** : α=0,2 avec β=0,5 et sans
  normalisation divisent la mAP par quatre sur vinci. À corriger indépendamment.
- **La sonde linéaire a la meilleure mAP et le pire transfert** : elle classe bien à
  l'intérieur d'une requête et produit des scores non comparables entre requêtes. Le
  mélange standardisé de sa note « round 5 » en récupère la moitié — le diagnostic de
  cette note était le bon.
- **Regrouper les prototypes n'apporte rien** : `multi_prototype` fait jeu égal avec
  `max_prototype`, à k=2 comme à k=4.

Deux contrôles structurels passent : `identity` reproduit la ligne de base au chiffre
près — l'artefact de normalisation que la branche d'origine avait dû corriger n'existe
plus avec le score en ratio — et `max_prototype` reproduit exactement le boost SQL. Le
pair F1 est identique pour toutes les méthodes à association fixée.

## 4. La gate de validation par VLM

Qwen3-VL-4B en NF4 4 bits (≈3 Go, cohabite avec le service MetaCLIP sur 8 Go) lit
`p(oui)` dans les logits, sans génération. Le score est mis en cache par
(prompt, candidat), donc une passe (~12 découpes/s, 15 min pour bbhotel) sert toutes les
configurations, et **les deux niveaux de gate partagent la même table**.

**L'indice d'abord**, sur les découpes qu'un humain avait déjà jugées :

| indice, mêmes données | AUC |
|---|---|
| distance entre points projetés | 0,879 |
| cosinus découpe↔découpe (MetaCLIP) | **0,529** |
| **Qwen3-VL `p(oui)`** | **0,925** (0,877 à 0,991 selon le prompt) |

Premier signal sémantique non trivial mesuré ici. Il explique rétrospectivement pourquoi
la porte sémantique, le descripteur accumulé et le terme sémantique du multicut sont tous
revenus neutres : la limite était l'espace d'embedding.

L'indice **ne transfère pas** : 0,594 groupé sur vinci, avec deux prompts *sous* 0,5 —
le modèle y préfère les faux positifs de l'humain. Les images l'expliquent : les faux
positifs de « checkin counter » sont des bornes libre-service **portant le mot
« Check-in » imprimé**, ceux d'« emergency power plant » sont le bâtiment qui abrite le
groupe électrogène. Négatifs difficiles d'une classe voisine, sur un vocabulaire de
métier dont l'annotateur a tranché la frontière. Reformuler la question récupère la
moitié de l'écart (0,594 → 0,683), au prix d'un hyperparamètre réglé sur les données
d'évaluation.

À noter : cette AUC est une borne **pessimiste**. Les reviews n'existent que pour les cas
limites qu'un humain a pris la peine de juger, alors que la gate s'applique à tout le jeu
retrouvé, dont l'essentiel est facile à rejeter — et vinci gagne autant que bbhotel bout
en bout malgré une AUC bien pire.

**Puis la gate appliquée**, bbhotel, `eps` 1,5 :

| configuration | mAP stricte | F1 LOO | pair F1 |
|---|---|---|---|
| sans gate | 0,701 | 0,639 | 0,508 |
| **détection, poids 0,75–1,0** | 0,716 | **0,686** | 0,508 |
| détection, poids 4,0 | **0,720** | 0,669 | 0,508 |
| cluster, moyenne, poids 0,25 | 0,718 | 0,618 | 0,508 |
| cluster, maximum, poids 0,5 | 0,704 | 0,638 | 0,508 |
| vote kNN seul | **0,751** | **0,711** | 0,508 |
| vote kNN puis gate détection | 0,758 | 0,660 | 0,508 |

Et sur vinci, `eps` 3,0, une fois ses découpes rendues :

| configuration | mAP stricte | F1 LOO | pair F1 |
|---|---|---|---|
| sans gate | 0,391 | 0,313 | 0,820 |
| **détection, poids 1,0** | 0,425 | **0,361** | 0,820 |
| cluster, moyenne, poids 0,25 | 0,404 | 0,244 | 0,820 |
| vote kNN seul | 0,473 | 0,410 | 0,820 |
| **vote kNN puis gate détection** | **0,490** | **0,417** | 0,820 |

- **Le niveau détection gagne sur les deux cartes (+0,047 et +0,048 de F1 LOO), le niveau
  cluster perd sur les deux** — l'inverse de la littérature du grounding 3D. Hypothèse non
  testée : les observations d'un cluster sont des vues quasi identiques, donc la moyenne
  n'ajoute pas d'évidence indépendante.
- **La composition avec le vote kNN dépend de la carte** : redondante sur bbhotel (F1 LOO
  0,711 → 0,660), additive sur vinci (0,410 → 0,417), ce qui est cohérent avec un indice
  VLM plus faible et donc moins redondant là-bas.
- **Le pair F1 est identique partout à association fixée** : la gate est un score, elle ne
  déplace aucun cluster.

**Le déblocage de vinci.** Son index v1 converti ne stocke aucune découpe. Les 5 547 que
le banc utilise ont été rendues une fois par `render_benchmark_cutouts`, avec la *même*
fonction de projection que l'indexeur — 2 385 keyframes décodés, 9 minutes, 79 Mo sur le
disque local (`~/.cache/wemap-object-search/cutouts/<map_id>`), contre 12,6 Go pour
l'index entier dont le banc n'a pas besoin.

Reste le coût en ligne : 1 000 appels VLM par requête. Hors ligne c'est acceptable
uniquement parce que le score se met en cache et sert toutes les configurations.

## 4b. Prompts négatifs — mesurés, et négatifs sauf confusion nommée

`negative_prompt_cue.py`, même protocole sur les découpes revues. Score relatif au lieu
de la similarité brute : `margin` (requête moins le meilleur négatif) et `softmax`
(`log p(requête | {requête} ∪ négatifs)`, sans échelle par requête — c'était l'argument).
Trois sources de négatifs : `benchmark` (les autres prompts de la carte : closed-set,
borne supérieure), `venue` (le vocabulaire de détection de `prepare` : la version
open-set honnête), `manual`.

| AUC groupée | bbhotel (brut 0,851) | vinci (brut 0,558) |
|---|---|---|
| manual, softmax | 0,854 | **0,616** |
| benchmark, softmax | 0,815 | 0,606 |
| **venue, softmax** | 0,833 | **0,462** |
| venue, margin | **0,730** | 0,447 |

**Le vocabulaire générique nuit, et le mécanisme est vérifié** : sur les *vrais positifs*
de « x ray machine », le négatif gagnant est `baggage x ray machine`, puis
`oversized baggage scanner` ; sur « e gates », `access control gate`. Tout vocabulaire
assez riche pour décrire un lieu contient des synonymes de la requête, et le max sur les
négatifs soustrait alors exactement le signal cherché.

**Par prompt, une confusion nommée vaut beaucoup** : « checkin counter » passe de 0,467 à
0,890 dès que la borne libre-service est un négatif, « emergency power plant » de 0,700 à
0,851, « lampe » de 0,667 à 0,857. Là où il n'y a pas de confusion, ça coûte (« x ray
machine » 0,574 → 0,426). La moyenne est donc nulle par compensation. `softmax` bat
`margin` partout.

Aucun rescorer n'a été écrit : c'est un instrument ciblé pour une confusion nommée, à
exposer comme champ « exclure » par requête, pas un mécanisme à activer.

## 5. Associations — rien de nouveau, et c'est le résultat attendu

Conforme au 12/08 : sur vinci multicut `pivot` 2,5 domine en pair F1 (0,856 contre 0,799
pour le défaut), incremental somme 1,2 fait aussi bien à granularité plus fine, et les
deux se composent avec le plafond sans interférence.

## 5b. L'échelle d'ablation du stack complet (13/08)

`toolbox/benchmark/grids/stack-ablation-*.json`, artefacts dans
`{map}/benchmark/stack-2026-08-13/`. Écart de F1 macro LOO contre la ligne de base :

| étape | vinci ΔLOO | bbhotel ΔLOO |
|---|---|---|
| multicut | **−0,042** | +0,001 |
| + plafond 20 m | −0,036 | +0,003 |
| + kNN des reviews | **+0,081** | **+0,069** |
| + gate VLM, détection | +0,086 | +0,030 |
| + gate VLM, **cluster** | **+0,132** | +0,030 |
| kNN seul | +0,097 | **+0,072** |

- **Correction d'un résultat antérieur** : la gate au niveau cluster perd appliquée seule,
  mais devient le meilleur point de vinci empilée sur multicut + plafond + kNN (0,445 de
  LOO contre 0,400 au niveau détection). Elle y arrive en *baissant* la mAP, donc elle
  améliore la comparabilité entre prompts, pas le classement interne.
- **La dernière brique est un choix par carte** : +0,051 sur vinci contre le kNN seul,
  −0,038 sur bbhotel. Critère apparent : la force de l'indice VLM relative à la couverture
  des reviews, mesurable à l'avance.
- **Multicut baisse le LOO en améliorant la partition** (spread 0,518 → 0,441, pair F1
  0,820 → 0,856) : l'étape d'association se lit sur le pair F1, pas sur le LOO.

Réserve : +0,132 sur 6 prompts en LOO, variance élevée. À confirmer sur une 3ᵉ carte.

## 5c. Plafond avant ou après la troncature (13/08)

Question posée : ne récupérer que 1 000 candidats *déjà* sous le plafond, au lieu d'en
prendre 1 000 puis d'en jeter. Mesuré avec un classement exact du top-2000
(`--exact-retrieval`), artefacts dans `{map}/benchmark/depth-stage-2026-08-13/`.

| configuration | vinci mAP / LOO / pair F1 | bbhotel mAP / LOO / pair F1 |
|---|---|---|
| k 1000, sans plafond | 0,397 / 0,324 / 0,808 | 0,705 / 0,639 / 0,508 |
| **k 2000, sans plafond** *(contrôle)* | 0,423 / **0,290** / **0,788** | 0,723 / 0,638 / **0,496** |
| plafond 20 m, après troncature | 0,397 / 0,330 / 0,808 | 0,703 / 0,641 / 0,510 |
| **plafond 20 m, avant troncature** | 0,398 / **0,330** / 0,809 | 0,703 / **0,641** / 0,508 |
| multicut + kNN, plafond après | 0,499 / 0,393 / 0,857 | 0,754 / 0,706 / 0,522 |
| multicut + kNN, plafond avant | 0,501 / 0,384 / 0,857 | 0,754 / 0,706 / 0,520 |

- **Les deux étapes sont indiscernables** à 20 m : même LOO à trois décimales sur les deux
  cartes. Le rappel de candidats proches restaure le *nombre* (564 → 613 clusters sur
  vinci) sans restaurer quoi que ce soit que les métriques voient.
- **Le contrôle est la ligne instructive** : passer de 1 000 à 2 000 candidats monte la
  mAP (+0,026) et **baisse** le F1 transférable (−0,034) et le pair F1 (−0,020). Les
  candidats supplémentaires sont ceux de la queue de classement : ils ajoutent des
  clusters au lieu d'améliorer ceux qui existent. C'est aussi pourquoi le rappel ne peut
  pas aider — c'est exactement cette queue qu'il fait entrer.
- **Un filtre en production serait donc pour la latence et la taille d'index, pas pour la
  qualité.**

Deux faits sur le chemin de retrieval, découverts en route : le service en ligne **ne
peut pas répondre à k > 1000** (`hnsw.ef_search = max(k, 1000)`, plafonné à 1000 par
pgvector), et le classement exact ne vaut que ~+0,01 de LOO contre le HNSW — l'index
approché ne perd presque rien.

## 6. Ce qu'il reste à faire

1. Corriger les défauts `feedback_alpha`/`feedback_beta` du toolbox (0,2 / 0,5) : ils
   sont nocifs sur les deux cartes.
2. Ne pas porter le plafond en production comme constante — le dériver du `venue_type`
   ou de l'espacement des keyframes, décidé sur au moins trois cartes.
3. Mesurer le boost hors échantillon : annoter la requête A et évaluer sur la requête B
   du même objet, ou réserver deux prompts de vinci dont les reviews ne servent jamais.
4. Refaire la mesure du plafond quand la vérité terrain contiendra des objets lointains.
5. Traiter la question posée au VLM comme un paramètre par carte, pas comme une
   constante : elle vaut 0,09 d'AUC sur vinci, et la régler proprement demande un jeu de
   découpes revues qui ne serve pas aussi à mesurer.
