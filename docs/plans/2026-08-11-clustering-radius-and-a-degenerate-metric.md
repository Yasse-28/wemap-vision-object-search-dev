# Session — le rayon de clustering, et une métrique dégénérée

Date : 2026-08-11, quatrième volet. Carte : `bbhotel-choisy`. Scripts et sorties dans le
scratchpad de la session (`eps_{0.5,1.0,2.0,3.0}_gate_{off,0.80,0.85}/`,
`eps_per_class_ap.json`).

**Ce document existe pour deux raisons : `clustering_eps_m` est le facteur le plus
puissant mesuré de toute la journée, et le banc ne sait pas dire si le bouger est une
amélioration.** La seconde raison est la plus importante.

---

## 1. Le balayage

Question posée : maintenant que la porte sémantique existe, la porte géométrique
(rayon de 2 m) est-elle encore contraignante ? Si la mAP est plate en `eps`, il est
inutile de raffiner le critère géométrique — IoU de boîtes 3D, rayon adaptatif à
l'étendue de l'objet, etc.

Prior annoncé avant la mesure : plate. **Faux.**

| `eps`            | porte off |           | porte 0.80 |        | porte 0.85 |        |
| ---------------- | --------- | --------- | ---------- | ------ | ---------- | ------ |
|                  | strict    | groupé    | strict     | groupé | strict     | groupé |
| **0.5 m**        | **0.788** | 0.675     | 0.752      | 0.659  | 0.703      | 0.628  |
| 1.0 m            | 0.729     | 0.687     | 0.735      | 0.660  | 0.715      | 0.638  |
| 2.0 m *(défaut)* | 0.653     | 0.713     | 0.694      | 0.692  | 0.698      | 0.639  |
| 3.0 m            | 0.607     | **0.731** | 0.677      | 0.700  | 0.679      | 0.645  |

De 2.0 à 0.5 m, la mAP stricte gagne **+0.135**. À comparer aux autres leviers de la
journée : porte sémantique +0.045, langue du prompt +0.039, changement de score +0.001
en mAP, changement de modèle d'embedding +0.019 d'étendue sur trois modèles.

## 2. Le vrai résultat : les deux vues sont anti-corrélées et monotones

En strict, plus on découpe, mieux c'est (0.607 → 0.788 quand `eps` diminue). En groupé,
exactement l'inverse (0.731 → 0.675). **Sur toute la plage, sans exception.**

Ce n'est pas « deux vérités terrain qui divergent un peu ». C'est une **métrique
dégénérée le long de l'axe découper/fusionner** : le banc ne classe pas deux
granularités, il rapporte laquelle des deux vérités terrain on vient de choisir.

Ni l'une ni l'autre ne décrit ce qu'un utilisateur appellerait « un objet » :

- la vue **stricte** traite un banc de chaises comme 213 objets à retrouver séparément ;
- la vue **groupée** écrase les 674 annotations en 118 cibles par single-linkage à 2 m —
  213 chaises en 5 cibles.

### Règle de tri qui en découle

| type d'intervention                         | mesurable ? | vérifié                                                                                                         |
| ------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------- |
| change le **classement** à granularité fixe | oui         | score en ratio et comparaison de modèles ont bougé dans le **même sens** sur les deux vues                      |
| change la **granularité**                   | **non**     | `eps`, porte sémantique, `min_keyframes`, IoU 3D, rayon adaptatif — signe décidé par le choix de vérité terrain |

## 3. AP par classe, `eps` (porte off)

| classe             | GT  | 0.5       | 1.0   | 2.0   | 3.0   | 0.5 + porte 0.85 | 0.5 − 2.0  |
| ------------------ | --- | --------- | ----- | ----- | ----- | ---------------- | ---------- |
| chaise             | 213 | 0.620     | 0.288 | 0.138 | 0.082 | **0.831**        | **+0.482** |
| table              | 77  | 0.740     | 0.453 | 0.246 | 0.139 | 0.754            | **+0.494** |
| poubelle           | 2   | 0.667     | 0.393 | 0.393 | 0.393 | 0.450            | +0.274     |
| lampe              | 55  | **0.805** | 0.745 | 0.552 | 0.485 | 0.565            | +0.253     |
| plante             | 22  | 0.597     | 0.668 | 0.461 | 0.282 | 0.526            | +0.136     |
| signe de sortie    | 88  | 0.857     | 0.877 | 0.816 | 0.818 | 0.815            | +0.041     |
| ascenseur          | 21  | 0.959     | 0.987 | 0.929 | 0.914 | 0.781            | +0.030     |
| extincteur         | 29  | **1.000** | 0.993 | 0.999 | 1.000 | 0.987            | +0.001     |
| cctv               | 41  | 0.902     | 0.914 | 0.908 | 0.914 | 0.838            | −0.006     |
| detecteur de fumée | 119 | 0.832     | 0.924 | 0.925 | 0.812 | 0.682            | **−0.094** |
| **mAP**            | 674 | **0.788** | 0.729 | 0.653 | 0.607 | 0.703            |            |

Le +0.135 global vient presque entièrement de `chaise` (+0.482) et `table` (+0.494) —
les deux classes que le volet 3 avait montrées **plates sous les trois modèles
d'embedding**. Confirmation par un troisième chemin que leur problème est la
granularité, pas la représentation ni le modèle.

À noter que `extincteur` et `cctv` sont **insensibles** à `eps` sur toute la plage
(1.000 / 0.999 / 1.000 et ~0.91 partout) : pour un petit objet compact, aucun rayon
entre 0.5 et 3 m ne change quoi que ce soit. Le rayon unique n'est pas « trop grand »,
il est *sans objet* pour la moitié des classes et déterminant pour l'autre.

## 4. Correction : la porte sémantique était redondante avec `eps`

Le volet 2 conclut que la double porte est « la première méthode d'association de la
série à battre l'existant ». **C'est vrai à `eps` figé à 2 m, et faux si on a le droit de
bouger `eps`.** À `eps` = 0.5, la porte *dégrade* le strict : 0.788 → 0.752 (0.80) →
0.703 (0.85).

La porte n'aidait à 2 m que parce qu'elle **défaisait un sur-regroupement créé par le
rayon**. Deux mécanismes de découpage, dont un seul est nécessaire — et le moins cher est
le rayon : ni embeddings supplémentaires par requête, ni second critère à régler.

Une exception intéressante, et c'est la seule combinaison où les deux se cumulent :
`chaise` 0.620 → **0.831** avec la porte à `eps` = 0.5. Sur 213 annotations dans quelques
salons, découper spatialement *et* sémantiquement paie encore. Sur les autres classes,
non.

## 5. Ce qui n'est pas recommandé, et pourquoi

**Ne pas passer `eps` à 0.5 malgré les +0.135.** Le gain est probablement en partie réel —
2 m est manifestement trop large pour un extincteur — mais rien ne permet de séparer la
part réelle de la part de complaisance envers la vue stricte. Faire ce changement sur ce
banc serait optimiser la métrique, pas le produit.

**Ne pas tester l'IoU de boîtes 3D.** Trois raisons, dans l'ordre de force :

1. c'est une intervention sur la granularité, donc non mesurable ici ;
2. nous n'avons pas d'étendue 3D — un point par détection plus une étendue *angulaire*.
   La fabriquer suppose d'inventer la profondeur de la boîte, et sa taille serait
   proportionnelle à `d`, c'est-à-dire à la profondeur dont le volet 1 établit qu'elle
   explose sur les rayons montants (1.16 m stockée contre 36 m impliquée). L'IoU étant
   très non linéaire, une erreur de facteur 2 fait passer un recouvrement de 0.5 à 0.03 ;
   la distance, elle, est linéaire en l'erreur ;
3. ConceptGraphs utilise un seuil d'IoU de **0.03** — fonctionnellement « est-ce que ces
   deux choses se touchent ». Leur porte géométrique est permissive et c'est la
   sémantique qui trie. Le raffinement géométrique n'est pas là que le signal se trouve.

**Ne pas activer la porte sémantique** : redondante avec un rayon plus petit, et plus
coûteuse.

## 6. La dépendance bloquante

Le regroupement des annotations en **leader/canopy** au lieu du single-linkage n'est plus
l'item 6 d'une liste de six : c'est la dépendance bloquante de **toute la ligne
géométrique**. Sans une vérité terrain qui définisse ce qu'est un objet, aucune
intervention sur la granularité ne peut être évaluée, et c'est là que se trouvent tous les
gros gains apparents (`eps` +0.135, porte +0.045, et les deux classes à 0.13 d'AP).

Étape intermédiaire moins chère et plus honnête, dans l'esprit du volet 1 (« regarder les
images avant de croire un chiffre ») : **inspecter visuellement une dizaine de clusters à
`eps` = 0.5 contre `eps` = 2 m sur `chaise` et `extincteur`**. Un avis humain là où la
métrique est muette dira si les 213 chaises sont 213 objets, 5 objets, ou autre chose —
et cette réponse détermine la vérité terrain à construire.
