# Une passe de fusion après le multicut — spécification et gates

Date : 2026-08-17. Suite de
[`2026-08-17-resultats-T0-a-T8.md`](2026-08-17-resultats-T0-a-T8.md).

## Ce qui motive l'ajout

Mesuré sur les paniers annotés de vinci (16 solo, 4 pair) :

| configuration | pair précision | pair rappel | pair F1 |
|---|---|---|---|
| `gasp1v2` @ 8° — proche de l'association en place | **97,6 %** | 53,7 % | 69,3 % |
| T-linkage @ 20° | 90,5 % | **99,3 %** | **94,7 %** |
| `gasp1v2` @ 8° + passe par rayons (T4) | 100,0 % | 73,8 % | 84,9 % |

Le diagnostic tient en une ligne : **l'association en place ne se trompe presque jamais
en fusionnant, et elle ne fusionne pas assez.** 97,6 % de précision pour 53,7 % de
rappel, c'est un sous-ajustement, et c'est exactement la fragmentation que le dossier
poursuit depuis le 11/08.

Une linkage permissive attrape le rappel manquant, mais elle paie en précision dès que
le voisinage se densifie — 90,5 % à deux objets, **72,8 % à trois**, le rappel collé à
100 %. C'est la signature de la clôture transitive, le mécanisme qui a produit la
composante de 486 détections sur 500. Un seul pool à trois objets existe sur vinci, donc
c'est un signal, pas un résultat.

D'où la forme proposée : garder la précision du multicut, récupérer une part du rappel,
et **rendre le chaînage inexprimable** plutôt que de le borner par un seuil.

## L'ajout

Une passe optionnelle, **après** l'association, **avant** les statistiques et le
ranking — dans `localize_from_enriched_candidates`, à la sortie de la branche
`params.association`, là où les labels de cluster existent et où rien n'a encore été
calculé dessus.

Trois propriétés qui définissent la passe :

1. **Les clusters sont des atomes.** La passe décide de fusions entre clusters, jamais
   entre détections. Il n'existe donc aucun chemin A–B–C au niveau des détections, et la
   densité du voisinage ne peut pas fabriquer de pont. C'est ce qui la distingue de ce
   qui a été mesuré à 94,7 % — voir la réserve en fin de document.
2. **Fusion seulement.** Elle ne redécoupe rien. Une découpe fausse du multicut reste
   fausse ; en échange, la passe ne peut pas dégrader ce que le multicut avait raison de
   séparer, et son effet est borné par construction.
3. **Le critère est réévalué sur l'union à chaque étape.** Une fusion n'est acceptée que
   si l'ensemble fusionné satisfait encore le critère — pas seulement la paire qui l'a
   déclenchée. C'est le point qui transforme une linkage en test d'hypothèse, et c'est
   déjà la logique de `score_1v2` / `_agglomerate_by_score`.

### Le déroulé

```
labels = <association en place>                      # multicut par défaut
candidats = paires de clusters (i, j) telles que
      distance(centroïde_i, centroïde_j) <= merge_pass_radius_m
  et  niveaux compatibles (_levels_compatible)
  et  pas de cannot-link hérité (cannot_link_same_keyframe)
tant qu'il reste un candidat de score positif :
      prendre le meilleur
      recalculer le critère sur l'UNION i ∪ j
      si l'union tient -> fusionner, hériter les cannot-link, réévaluer les voisins
      sinon            -> retirer ce candidat
relabeliser, puis compute_cluster_statistics / rank_localization_clusters
```

Le critère sur une union : les rayons de tous les membres admettent-ils un point de
consensus unique à `merge_pass_inlier_deg` près (`triangulate_rays`, déjà là) ? À 20°,
qui est le réglage d'où vient le gain mesuré.

### Les paramètres, et ce qu'ils sont

| paramètre | défaut | nature |
|---|---|---|
| `merge_pass` | `False` | interrupteur ; la passe est inerte par défaut |
| `merge_pass_radius_m` | 6,0 | **paramètre de coût**, pas de vérité : il décide qui est *candidat*, pas combien il y a d'objets. C'est ce qui le distingue de `clustering_eps_m` |
| `merge_pass_inlier_deg` | 20,0 | tolérance angulaire du consensus |
| `merge_pass_max_pool` | garde-fou | borne le coût sur un voisinage pathologique |

Le rayon reste un paramètre de lieu et il faut le nommer comme tel. La différence avec
`clustering_eps_m` est réelle mais étroite : il ne fixe pas la cardinalité, il fixe une
liste de candidats — un candidat non retenu par le critère ne coûte rien.

### Le coût

`O(K²)` paires dans le rayon, `K` = nombre de clusters d'une requête (quelques dizaines),
chacune un petit moindres carrés sur les rayons mis en commun. Négligeable devant
l'appel ANN. Aucune lecture de depth map, aucun accès disque supplémentaire.

## Ce que la passe ne fait pas

- elle ne répare pas une découpe fausse (fusion seulement) ;
- elle ne touche ni au ranking, ni au score de similarité, ni aux positions — les
  centroïdes sont recalculés, rien d'autre ;
- elle ne remplace pas la passe par rayons (T4). **Les deux sont substituables, pas
  cumulables** : T4 fait passer `gasp1v2` de 69,3 à 84,9 de pair F1 et *dégrade*
  T-linkage @ 20° de 94,7 à 92,1. Si la passe de fusion tient ses promesses, T4 ne se
  justifie plus.

## La réserve qui compte

**Le 94,7 % ne mesure pas l'algorithme spécifié ici.** Il mesure une T-linkage libre sur
les détections mises en commun — la variante exposée au chaînage. La passe décrite
ci-dessus est plus contrainte sur les deux axes qui importent (atomes, réévaluation sur
l'union), donc son résultat attendu se situe **entre** la précision du multicut (97,6 %)
et le rappel de la linkage (99,3 %), sans qu'aucun chiffre ne le garantisse.

Autrement dit : le tableau justifie d'écrire la passe, il ne prédit pas sa valeur.

## Les gates, dans l'ordre

1. **Mesurer la passe sur les paniers existants** — pair P/R/F1, en partant des clusters
   du multicut et non de la vérité, sinon le test est circulaire. Une demi-journée.
   Critère : pair F1 au-dessus de 84,9 (le meilleur composé actuel) **sans** que la
   précision descende sous 95 %.
2. **bbhotel.** Le chaînage ne se mesure pas sur vinci : un seul pool à trois objets y
   existe, à n'importe quel rayon. Les chaises et les détecteurs de fumée de bbhotel
   donnent des voisinages de 5 à 10 objets de même classe, qui sont le régime où la
   passe peut échouer. **Sans paniers bbhotel, la propriété anti-chaînage est un
   argument de conception, pas une mesure.**
3. **`association_sweep` sur carte complète**, en validation finale — avec la réserve
   habituelle que le rattachement par `--near-m` dépend des positions.
4. **Le patch va dans `wemap-vision-backend`**, pas ici : `toolbox/bricks/localize.py`
   est un miroir de dev (`AI_CONTEXT.md`). Prototyper ici, porter là-bas, resynchroniser.

## Le point de départ pour l'implémentation

- `toolbox/bricks/localize.py:1501-1544` — la branche `params.association`, où insérer
  l'appel ;
- `toolbox/bricks/matching.py:_agglomerate_by_score` — la boucle « meilleur score,
  réévalué sur l'union » existe déjà, sur des détections ; la passe en est la version
  sur clusters ;
- `toolbox/bricks/triangulate.py:triangulate_rays` — le critère de consensus ;
- `toolbox/bricks/matching.py:cannot_link_pairs` — la contrainte dure à hériter ;
- `toolbox/benchmark/matching_baskets.py:pair_counts` / `pair_f1` — la métrique du
  gate 1, insensible à la chance sur le compte de clusters.
