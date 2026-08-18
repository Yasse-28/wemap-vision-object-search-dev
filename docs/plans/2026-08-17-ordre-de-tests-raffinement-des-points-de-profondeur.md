# Ordre de tests — de la correction des points de profondeur au modèle génératif d'occupation

Date : 2026-08-17 (révision 2, après revue bibliographique). Destiné à un agent codeur.
Suite de [`2026-08-15-fragmentation-resultats-E0-a-E4.md`](2026-08-15-fragmentation-resultats-E0-a-E4.md)
et de [`2026-08-13-etat-des-lieux.html`](2026-08-13-etat-des-lieux.html).

## Ce qui a changé depuis la révision 1 de ce matin

La révision 1 proposait six tests de correction du point de profondeur. Elle reste valide
mais elle est désormais **subordonnée à un verdict architectural** issu de la revue de sept
papiers (bibliographie en fin de document) :

> **Ne jamais associer les détections entre elles. Faire concourir des objets hypothétiques
> pour expliquer les détections.**

Quatre papiers indépendants — POM (Fleuret 2008), la loss PCM du comptage de foule 3D, la
réassociation hongroise du comptage de fruits, le front-end frame-to-model d'ODAM —
implémentent ce principe sous quatre habillages. Il règle structurellement les deux faits
les plus coûteux du dossier :

- **le chaînage transitif** (composante de 486 détections sur 500 avec SIFT puis
  DISK+LightGlue) est une propriété de la clôture transitive d'un graphe de paires. Dans un
  modèle détection → objet, il n'y a pas de clôture transitive à prendre : le chaînage
  devient inexprimable, pas seulement rare ;
- **le seuil qui décide du nombre d'objets** (0,5 m sépare deux machines à 3 m, 1,5 m les
  fusionne) est remplacé par une compétition entre hypothèses, tranchée par un point fixe.

Le multicut sur graphe de détections fait exactement l'inverse. C'est donc un verdict
d'architecture, pas un réglage — mais il ne justifie pas de tout jeter : T2, T3 et T8
améliorent les entrées de n'importe quelle architecture, et T7 a besoin d'elles.

Cinq modifications concrètes :

1. **T1 devient critique et gagne une troisième composante** (erreur systématique par
   keyframe). Ce n'est plus un diagnostic parmi d'autres : c'est le prérequis dur de T7.
2. **Nouveau T1b** — AUC du signal d'absence de co-visibilité, gate R2 bon marché avant T7.
3. **T7 change de nature.** La grille de densité par splatting envisagée initialement est du
   vote sans explication mutuelle : elle produit un pic fantôme entre deux objets réels.
   Remplacée par POM, qui n'a pas ce défaut.
4. **Nouveau T8** — correction de pose conjointe façon DAF-SLAM, conditionnée à T1.
5. **E3 et E4 sont reconfirmés morts** par ODAM, qui est la même idée aboutie.

---

## Règles

R1 à R5 du plan du 15/08 restent en vigueur, ainsi que les trois ajoutées ce matin. Rappel
des quatre qui vont mordre le plus souvent :

- **R1 — un changement de granularité ne se lit ni sur la mAP ni sur le LOO.** Toute
  modification des positions 3D change la granularité. Sur les paniers, la lecture est
  « nombre de clusters obtenu vs attendu » et « fausses fusions ».
- **R2 — mesurer l'indice avant d'implémenter la méthode.** T1 et T1b existent pour ça et
  conditionnent T2, T5, T7 et T8.
- **R3 — un score, jamais un filtre.** POM y satisfait par construction : rien n'est retiré,
  chaque cellule reçoit une probabilité.
- **Règle nouvelle (rév. 1, n° 4) — rien ne se mesure sur les clusters existants.** Les
  détections à profondeur fausse sont précisément celles qui ont *échoué* à rejoindre leur
  cluster. Les mesurer sur les clusters formés, c'est conditionner sur le résultat. Les
  paniers annotés à la main sont la parade, et c'est leur raison d'être ici.

Une règle nouvelle propre à la révision 2 :

- **R7 — énoncer la prédiction avant de mesurer.** Toutes les mesures du dossier changent de
  signe entre vinci et bbhotel. Chaque test ci-dessous porte donc une prédiction explicite
  et différenciée par carte, écrite avant l'exécution. Un test qui « marche » sans qu'on ait
  su dire où il devait marcher n'apprend rien.

`scripts/agent-lock.sh` avant toute écriture (`AI_RULES.md`).

---

## L'ordre

| # | test | coût | prérequis | ce qu'il débloque |
|---|---|---|---|---|
| **T0** | harnais de paniers | ½ j | — | tout le reste |
| **T1** | dispersion : radial / tangentiel / **systématique** | ~3 h | T0 | **arbitre T2–T8** |
| **T1b** | AUC du signal d'absence de co-visibilité | ~3 h | T0 | gate R2 de T7 |
| **T2** | profondeur robuste sur la bbox | 1 j | T1 | entrées de tout le reste |
| **T3** | recalibrage de `σ(r)` | 1 j | T1, T2 | orthogonal, signe connu |
| **T4** | re-triangulation par rayons | 1 j | T1 | presque gratuit, attentes basses |
| **T8** | correction de pose conjointe | 2–3 j | T1 | **prérequis de T7 si systématique** |
| **T7** | **POM adapté aux ERP et aux détections** | 4–6 j | T1, T1b, T8 | la piste de fond |
| **T5** | matching de points + triangulation | 3–5 j | T1 | seulement si T1 dit tangentiel |
| **T6** | étendue mesurée (reprise d'E3) | 1 j | T5 | terminal |

T1 coûte trois heures et décide de l'ordre réel de tout ce qui suit. T7 est la destination,
mais il est inatteignable tant que T1 et T8 n'ont pas nettoyé ses entrées.

---

## T0 — Le harnais : rejouer un panier hors ligne (préalable, ~½ journée)

Inchangé depuis la révision 1, à un ajout près. `toolbox/benchmark/extent_baskets.py` fait
déjà 90 % du travail — il construit des paniers, appelle directement
`matching._agglomerate_by_score` sans passer par le service, et compte clusters et fausses
fusions. Il ne lui manque que la source de vérité, qu'il fabrique par rattachement
géométrique (`nearest_annotation_labels`), c'est-à-dire par la voie que la règle n° 4
interdit.

**À implémenter.** `toolbox/benchmark/matching_baskets.py`, calqué sur `extent_baskets.py` :

- lire `{map}/object-search-annotations.db`, table `detection_group_label`
  (`keyframe_id`, `theta_center`, `phi_center`, `row_index`, `group_name`). **`group_name IS
  NULL` signifie « pas un objet »** — négatifs explicites, à conserver comme tels ;
- résoudre vers une détection par `row_index` dans le cache de candidats
  (`association_sweep.fetch_prompt_candidates`) ; voie de secours `matching.resolve_items`,
  qui exige postgres. Les deux doivent donner le même item — c'est un test à écrire ;
- familles **solo** (1 groupe → 1 cluster), **pair** (2 groupes à ≤ 4 m → 2 clusters), et
  **mixte** (1 groupe + les négatifs du même keyframe, qui ne doivent rejoindre aucun
  cluster) ;
- rejouer les **sept méthodes** de `MatchingPanel.compareMethods()` (table `COMPARED`,
  `MatchingPanel.tsx:49-62`), pas seulement `gasp1v2`.

**Sortie obligatoire.** Pour chaque panier, la métrique qu'E0 a identifiée comme causale,
décomposée — c'est ce qui rend T1 gratuit :

- `spread_total`, `spread_radial` (projeté sur le rayon de chaque détection),
  `spread_tangential` (perpendiculaire), et **`spread_systematic`** (voir T1) ;
- `range_median`, les précédents par bande de portée, et `max_parallax_deg` par panier.

**Ajout de la révision 2.** Le harnais doit aussi exporter, par panier, la **matrice de
co-visibilité** : pour chaque paire (keyframe f, groupe g), le nombre de détections de g
produites par f. C'est la donnée brute de T1b et de T7, et elle ne coûte rien à produire ici.

**Critère de succès.** Sur un panier constitué à la main dans l'UI, mêmes étiquettes de
cluster que l'onglet, pour les sept méthodes. Écart toléré : zéro.

**Piège.** Les angles sont arrondis à 6 décimales en base (clé primaire de fait de
`detection_group_label`) et `basketKey` de l'UI utilise `toFixed(4)`. Ne pas mélanger les
deux précisions dans le join.

---

## T1 — Diagnostic de la dispersion, en trois composantes (~3 h, décide de tout)

La révision 1 en prévoyait deux. La revue en impose une troisième, et c'est probablement la
plus importante.

**Les trois mécanismes, et leurs remèdes disjoints :**

| composante | mécanisme | remède | test |
|---|---|---|---|
| **radiale** | la profondeur est fausse (mauvais pixel, quantisation, bord d'objet, trou) | mieux échantillonner, mieux modéliser | T2, T3 |
| **tangentielle** | la profondeur est juste, mais le centre de bbox ne désigne pas le même point matériel d'une vue à l'autre | vraies correspondances | T5 |
| **systématique par keyframe** | la **pose** de la vue est décalée : toutes ses détections le sont du même vecteur | correction de pose conjointe | T8 |

La troisième n'a jamais été mesurée et votre contexte annonce une « précision métrique
imparfaite » sur les poses 6-DoF. Or `latent_cost`, `σ_i` et le Huber la traitent tous comme
de l'aléa — exactement le reproche que le plan du 15/08 faisait au glissement de bbox.
**Aucun estimateur robuste ne corrige un biais.**

**À mesurer**, entièrement depuis la sortie de T0 :

1. le rapport `spread_tangential / spread_radial` par panier, ventilé par bande de portée et
   par classe. Les classes qui éclataient le plus sur bbhotel (extincteur 4,35 ; cctv 3,48 ;
   détecteur de fumée 2,91) sont vues à 1,4–2,5 m, celles qui tenaient (chaise 1,43) à
   5,4 m : le rapport doit trancher sur ces classes-là ;
2. `spread_systematic` — pour chaque keyframe f, le **résidu moyen vectoriel** de ses
   détections par rapport au centroïde de leur groupe, tous groupes confondus. Si ce vecteur
   est cohérent en direction et en norme sur les détections d'une même vue, c'est de
   l'erreur de pose et non du bruit. Test : comparer la norme du résidu **moyen** à la
   moyenne des normes ; leur rapport est nul pour du bruit isotrope et vaut 1 pour un biais
   pur ;
3. l'AUC de l'indice **portée** dans `pair_cue_separability.py`, avec l'AUC conditionnelle
   ajoutée le 15/08. Quelques minutes ;
4. contrôle : corrélation entre `spread_tangential` et `hypot(angular_width, angular_height)`.
   Si le glissement est de la parallaxe, il croît avec l'extension angulaire, qui est le
   rapport taille/portée.

**Prédictions (R7), à écrire avant l'exécution.** Sur bbhotel, courte portée et petits
objets : la composante radiale devrait dominer (la depth map est mauvaise sur les petits
objets et les bords). Sur vinci, longue portée et objets étendus : la tangentielle. La
systématique, si elle existe, devrait être du même ordre sur les deux cartes puisqu'elle ne
dépend pas des objets. **Si la systématique domine partout, tout le reste du plan est à
suspendre jusqu'à T8.**

**Le gate.**

- **radial domine** → T2, T3, puis T7. Pas de T5.
- **tangentiel domine** → T2 corrige un problème inexistant et T4 triangule un biais. Aller
  à T5, et ne garder T3 qu'en second ordre.
- **systématique domine** → **T8 d'abord**, tout le reste est ininterprétable avant.
- ordres comparables → T2 (peu cher), puis T8, puis T7.

**Piège.** La décomposition n'a de sens que si les membres du panier sont vus depuis des
positions séparées. Imprimer `max_parallax_deg` par panier et écarter ceux sous ~5°, où les
composantes sont confondues. Ce même gate sert à T4, T5 et T7 — une seule mesure, trois
usages.

---

## T1b — L'absence de co-visibilité est-elle un indice ? (~3 h, gate R2 de T7)

**L'idée.** Vos indices mesurés sont tous **métriques** sur des quantités bruitées : distance
des points de profondeur (0,879), cosinus MetaCLIP (0,529), résidu de triangulation, gap
angulaire. Le comptage par vue est **combinatoire** : le détecteur a produit trois boîtes
dans ce panorama, donc il y a au moins trois objets co-visibles. Aucun de vos cinq faits
d'échec ne s'y applique.

**La bonne forme.** Le comptage agrégé est faible — `max_f (détections co-visibles)` est une
borne **inférieure** sur K, et votre problème est une sur-estimation de K, donc elle ne mord
pas. Sa contraposée, appliquée paire de fragments par paire de fragments, mord :

- **un keyframe a produit une détection dans A *et* une dans B** → deux boîtes distinctes
  dans le même panorama → objets distincts. C'est `cannot_link_same_keyframe`, qui existe ;
- **aucun keyframe ne voit les deux, alors que les keyframes de `F(B)` couvrent
  géométriquement la position de A** → l'occasion de les voir comme deux objets existait et
  n'a pas été saisie → **fragment**. C'est le signal neuf.

**Ce n'est pas E2 rejoué.** E2 avait le meilleur AUC jamais mesuré (0,824–0,890) pour un
gain nul, parce qu'il ne concernait que 7–9 % des paires même-objet : celles du *même*
keyframe. Le signal d'absence porte sur le **complément** — les paires inter-keyframes, soit
91–93 %. La limitation structurelle qui a tué E2 ne s'applique pas.

**À mesurer.** Sur les paires de groupes des paniers de T0, l'AUC de l'indicatrice
d'absence pour discriminer « même objet » de « objets distincts ». Aucune implémentation
d'association n'est nécessaire : la matrice de co-visibilité de T0 suffit.

**La définition de « f couvre la position de A », et c'est là que vos depth maps servent
enfin.** Une cellule est couverte par f si elle est dans le champ (trivial en ERP : tout
sauf les masques), à portée, **et non occultée** — et l'occultation se lit directement dans
la depth map de f, en comparant la profondeur du pixel à la distance de la cellule. Vos
cartes de profondeur sont médiocres comme estimateurs de position ponctuelle mais
parfaitement suffisantes comme **oracles de visibilité** : savoir s'il y a un mur entre la
caméra et un point à 4 m ne demande pas 10 cm de précision. C'est le meilleur usage
disponible d'une donnée que tout le reste du dossier traite comme du bruit.

**Le gate.** AUC ≥ 0,70 → T7 est justifié. Entre 0,55 et 0,70 → T7 reste possible mais son
terme de vraisemblance sera faible, à pondérer. Sous 0,55 → **T7 est abandonné**, et la
piste se réduit à T2/T3/T5.

**Piège.** Un détecteur à vocabulaire ouvert produit des doublons et coupe parfois un grand
objet en deux boîtes. « Deux boîtes dans la même vue » n'est donc pas une preuve exacte de
deux objets — d'où la marge angulaire de `cannot_link_same_keyframe`
(`SAME_KEYFRAME_MARGIN = 1.5`) et l'AUC de 0,824–0,890 d'E2 plutôt que 1,0. Mesurer le taux
de doublons intra-vue sur les paniers avant d'interpréter.

---

## T2 — Profondeur robuste sur la bbox (~1 jour)

**Hypothèse.** `object_position` vient d'**un seul pixel** de la depth map ERP, échantillonné
au centre de la bbox (`prepare_postprocess.sample_depths` →
`ingest_cli._compute_object_positions`), TIFF uint16 sqrt-quantisé, `raw == 0` → NaN, aucune
agrégation ni rejet d'outlier. Ce pixel tombe sur l'objet la plupart du temps, mais sur un
bord, un trou ou l'arrière-plan une fois sur dix — et c'est cette queue qui fait la
dispersion.

**À implémenter hors ligne d'abord.** Dans le harnais, rejouer `sample_depths` sous
plusieurs politiques :

- `center` — l'existant, ligne de base ;
- `median` — médiane des pixels valides de la bbox ;
- `nearest_mode` — histogramme des profondeurs valides, on garde le mode **le plus proche**
  dont la masse dépasse un seuil. Un objet est devant son fond : en cas de bimodalité c'est
  le mode proche qui est l'objet. Politique attendue gagnante ;
- `trimmed` — moyenne tronquée 20/80, contrôle de robustesse bête.

Remonter la **fraction de pixels valides** : une bbox à 30 % de `raw == 0` n'est pas la même
mesure qu'une bbox pleine, et c'est probablement un critère de rejet.

**Prédiction (R7).** Gain fort sur bbhotel (petits objets, bords nombreux, depth map faible
de près), faible sur vinci (grands objets, pixels centraux fiables).

**Critère de succès.** Sur les deux cartes : `spread_radial` baisse d'au moins 30 % à courte
portée, **et** les paniers solo gagnent en exactitude, **et** les paniers pair ne perdent
rien. Les trois, pas deux — une dispersion réduite en fusionnant tout est un échec, c'est
exactement ce qui est arrivé à E3 sur vinci (44 % → 24 % de paniers correctement séparés).

**Piège.** La bbox d'une détection en ERP n'est pas un rectangle en pixels près des pôles.
Appliquer `theta_phi_to_uv` aux quatre coins, pas au centre plus une demi-taille. Vérifier
sur une détection à `phi` élevé.

**Si positif**, alors seulement : ajouter la politique dans `sample_depths` avec `center` par
défaut, réingérer une carte, vérifier que le chemin en base retrouve le résultat hors ligne.

---

## T3 — Recalibrer `σ(r)` (~1 jour, orthogonal)

**Hypothèse.** `merge_score._sigmas` vaut `0.5 + 0.05·r` : le bruit y croît avec la portée.
Les mesures du 15/08 disent l'inverse — la dispersion observée *décroît* avec la portée sur
bbhotel (ρ = −0,44). Le modèle de bruit a le mauvais signe, et il pilote toute la tolérance
de fusion de `latent_cost`.

**À implémenter.** Ajuster `σ(r)` sur `spread_radial` par bande de portée, mesuré en T1 et
**re-mesuré après T2** — recalibrer sur les anciens points serait à refaire. Affine si elle
suffit, sinon constante par bande. Exposer les coefficients, ne pas changer les défauts.

**Pourquoi c'est propre.** Score à géométrie constante : rien n'est filtré, donc R3 est
respectée et le contrôle structurel est facile — `pair F1` identique tant que le poids
est nul.

**Critère de succès.** Paniers solo en hausse sans baisse des paniers pair, sur les deux
cartes. Amplitude attendue modeste ; sous 0,02 et de signes opposés selon la carte, c'est un
échec, comme E2.

**Piège.** `σ_i² + e_i²` mélange bruit et étendue. Garder `e` à la constante 1 m pendant tout
T3, sinon on mesure la somme de deux changements.

---

## T4 — Re-triangulation par rayons (~1 jour, attentes basses)

**Hypothèse.** La profondeur sert à décider qui va avec qui, puis on n'en a plus besoin : la
position se recalcule par intersection des rayons des membres.

**Ce qui existe déjà**, et c'est ce qui rend le test bon marché : `triangulate.triangulate_rays`
(moindres carrés angulaires + RANSAC), `localize.centroid_from="rays"`,
`localize.multicut_geo_source="ray"`, le fallback de centroïdage de `localize.py:1144-1266`.

**À implémenter.** Deux passes dans le harnais : associer avec le prior de profondeur et un
seuil lâche, re-trianguler chaque cluster depuis les rayons, ré-associer. **Une seule
itération**, mesurée ; en itérer plus sans critère d'arrêt fusionne tout.

**Attentes basses, et il faut le dire avant.** Cette triangulation porte sur les **centres de
bbox**. Si T1 conclut au tangentiel, elle triangule le biais au lieu de le retirer. C'est
déjà ce qui a coulé C-DOG (−0,061 à granularité égale, 21,9 % de paires inutilisables). Le
test se fait parce qu'il coûte une journée et que le code est là.

**Critère de succès.** Paniers solo en hausse sans dégradation des pair, et
`spread_tangential` en baisse. Si seul `spread_radial` baisse, le gain vient de ce que la
triangulation ignore la profondeur, pas d'une correction du biais — dire lequel.

**Piège.** Les paires à faible parallaxe partent à l'infini. Reprendre `max_parallax_deg` et
compter les clusters rejetés ; un gain obtenu en écartant 20 % des cas n'est pas un gain.

---

## T8 — Correction de pose conjointe (~2–3 jours, conditionné à T1)

**Déclencheur.** `spread_systematic` non négligeable en T1.

**Pourquoi ça compte plus qu'il n'y paraît.** Une erreur de pose est un décalage **commun à
toutes les détections d'une vue**. Elle est donc (a) invisible aux estimateurs robustes, qui
la voient comme de l'aléa, et (b) **sur-déterminée** : une vue qui voit dix objets fournit
dix contraintes pour trois à six degrés de liberté. C'est un problème beaucoup mieux posé
que l'association elle-même, et il se résout avec les objets qu'on a déjà.

**Référence.** *Data-Association-Free Landmark-based SLAM* (Zhang, Severinsen, Leonard,
Carlone, Khosoussi, ICRA 2023, arXiv:2302.13264). Le problème est scindé en un problème
**interne** — trajectoire, positions d'amers et associations, résolu par une alternance de
type k-means — et un problème **externe** — le nombre d'amers. Ils battent une baseline
oracle qui connaît K, donc le problème externe est tractable. C'est le seul des sept papiers
qui corrige la pose *et* l'association conjointement.

**Version minimale, adaptée.** Vous n'avez pas besoin du SLAM complet : vos poses sont
connues à une erreur près, et les paniers donnent les associations. Donc, à associations
fixées, estimer un offset `Δ_f ∈ R³` (translation seule d'abord) par keyframe qui minimise la
dispersion des groupes, avec régularisation vers zéro. C'est un moindres carrés linéaire,
quelques dizaines de lignes. La rotation vient après, si la translation ne suffit pas.

**Critère de succès.** `spread_systematic` doit tomber près de zéro par construction — ce
n'est pas le critère. Le critère est que `spread_total` baisse **et** que les paniers
gagnent, sur les deux cartes. Si `spread_total` ne bouge pas, le systématique était
négligeable et T8 s'arrête là.

**Piège majeur.** À associations fixées et offsets libres, le modèle peut « expliquer »
n'importe quelle dispersion en déplaçant les vues — sur-ajustement garanti. Deux garde-fous :
régularisation forte vers l'offset nul, et surtout **validation croisée par groupe** —
estimer les offsets sur la moitié des groupes, mesurer la dispersion sur l'autre moitié. Sans
ça, T8 produira un chiffre magnifique et faux.

---

## T7 — POM adapté aux panoramas et aux détections (~4–6 jours, la piste de fond)

**Référence.** Fleuret, Berclaz, Lengagne, Fua, *Multicamera People Tracking with a
Probabilistic Occupancy Map*, TPAMI 30(2):267–282, 2008.

**Pourquoi celui-là.** Il contient déjà, et en mieux, le mécanisme de T1b. Là où T1b demande
de définir « f couvre la position de A », POM rend cette définition inutile : elle sort du
rendu. Le champ moyen donne, par cellule k :

```
q_k = 1 / (1 + exp( λ_k + Σ_c [ Ψ(B^c, E_Q(A^c|X^k=1)) − Ψ(B^c, E_Q(A^c|X^k=0)) ] ))     (27)
E_Q(A^c)(x,y) = 1 − ∏_{k : A_k^c(x,y)=1} (1 − q_k)                                       (30)
Ψ(B, A)       = (1/σ) · |B ⊗ (1−A) + (1−B) ⊗ A| / |A|                                    (19)
```

Si une cellule voisine explique déjà les pixels, forcer `q_k` à 0 ou 1 ne change plus l'image
moyenne, les deux `Ψ` s'annulent et **la cellule retombe à son prior `λ_k`**. Les auteurs
l'écrivent : *« occlusion is taken into account naturally: […] all terms vanish but `λ_k` in
the exponential. »* Propriétés obtenues d'un coup, sans seuil et sans contrainte dure :
explication mutuelle, occlusion, R3 par construction, et **cardinalité émergente** — K n'est
jamais un paramètre, le seul terme de parcimonie est `λ_k = log((1−ε_k)/ε_k)`, un coût fixe
par objet dans le logit.

Coût : ~100 itérations de point fixe, temps constant par cellule via images intégrales
(éq. 31–35), **6 fps sur 1000 cellules et 4 caméras sur une machine de 2007**. Le coût n'est
pas un sujet.

### L'adaptation, terme par terme

Les auteurs l'anticipent (§VII : remplacer la soustraction de fond par des détecteurs).

- **La grille.** 2D par étage **et par classe**, cellules de 10–20 cm, plus une **marginale
  de hauteur par cellule occupée** (2,5D). Le 2D pur fusionnerait un détecteur de fumée au
  plafond et un objet au sol de même (x, y) ; la marginale de hauteur coûte presque rien et
  les départage.
- **`B^c`, l'évidence image.** Non pas une soustraction de fond mais la carte de confiance
  des détections de la classe dans le panorama c : pour chaque pixel, le max des scores des
  boîtes qui le couvrent. Version douce plutôt que binaire — le détecteur donne un score,
  autant s'en servir.
- **`A_k^c`, le rendu.** Un objet de taille physique `s_κ` (prior par classe) placé en
  cellule k, rendu dans le panorama c. **L'ERP est ici plus commode que le sténopé** : à
  distance `r`, la demi-largeur angulaire vaut `atan(s_κ / 2r)`, et comme `theta` et `phi`
  sont linéaires en pixels dans un ERP, `A_k^c` **est** un rectangle en coordonnées pixel —
  exactement la forme dont les images intégrales de POM ont besoin. La machinerie d'éq. 31–35
  se transpose sans modification.
- **La visibilité.** POM n'avait pas de profondeur et devait tout déduire du rendu. Vous en
  avez une par keyframe : une cellule est occultée depuis c si la depth map de c donne, dans
  cette direction, une profondeur nettement inférieure à la distance de la cellule. Savoir
  s'il y a un mur entre la caméra et un point à 4 m ne demande pas 10 cm de précision.
  **C'est le meilleur usage disponible d'une donnée que tout le reste du dossier traite comme
  du bruit.**
- **`λ_k`.** Densité d'objets attendue par classe et par m². Un seul scalaire par classe, à
  fixer grossièrement — POM montre que `σ = 0.01` et le prior ne sont pas critiques.
- **La taille par classe `s_κ`.** Ce n'est **pas** E3 qui revient : E3 estimait une taille
  *par détection* depuis la portée, donc anti-corrélée au besoin. Une taille physique
  constante par classe est un fait sur l'objet. POM est robuste entre 1,7 et 2,2 m pour un
  modèle de 1,75 m, soit ±15 %. Le vocabulaire étant ouvert, dériver `s_κ` du nom de classe
  une fois, hors ligne, par table ou par LLM.

### Extraction des instances

POM rend un champ, votre vérité est une partition. Deux étapes, chacune tirée d'un papier :

1. **Champ → K instances**, sans seuil réglé à la main : arXiv:2203.15691 dérive le seuil
   analytiquement (celui pour lequel le compte normalisé d'une composante isolée vaut 1, via
   la masse gaussienne `F(r_T)`), compte par composante connexe, puis ajuste un GMM à
   K = ce compte. **Ne pas seuiller naïvement** : les composantes connexes d'un sur-niveau
   rechaînent par les crêtes, et on aurait réimporté le fait n° 4.
2. **Instances → affectation des détections** : reprojeter chaque instance dans chaque vue et
   faire une **assignation hongroise globale exclusive** contre les détections, façon
   arXiv:1811.01417 §IV-C, qui argumente explicitement contre les seuils de distance 3D. Les
   instances qui n'expliquent rien meurent.

### Prédiction (R7), à écrire avant l'exécution

POM devrait **marcher sur vinci et échouer sur bbhotel**. Les auteurs nomment eux-mêmes leur
limite : *« excessive proximity of several individuals […] represents the true limitation of
our approach »* — quand aucune vue ne résout deux objets en blobs distincts. Vos machines à
3 m vues à 5 m sont largement résolues ; vos extincteurs vus à 1,4–2,5 m, non. Si le résultat
est l'inverse de cette prédiction, ne pas s'en réjouir : chercher l'erreur.

### Les deux préconditions bloquantes

1. **La calibration.** POM suppose la pose exacte — les `A_k^c` en dérivent entièrement — et
   ne fait aucune étude de sensibilité. À 20 cm de grille, une erreur de pose métrique
   détruit le rendu. **T8 est un prérequis dur, pas une amélioration.**
2. **La parallaxe.** POM échoue à 2 caméras au-delà de 4 personnes. Vos keyframes sont
   séquentiels le long d'un parcours, donc souvent quasi-colinéaires dans un couloir. Le gate
   de parallaxe de T1 s'applique intégralement.

**Piège.** POM est un champ moyen : `Q` reste une loi produit, l'explication mutuelle n'existe
que par le couplage du point fixe, pas dans la loi approchante. Les corrélations entre
cellules ne sont donc pas modélisées. Ça se voit quand deux cellules adjacentes se partagent
un objet et convergent toutes deux à `q ≈ 0,5` — surveiller ce mode d'échec, il ressemble à
un objet mal localisé alors que c'est une limite de l'approximation.

---

## T5 — Matching de points et triangulation sphérique (~3–5 jours, si T1 dit tangentiel)

Inchangé depuis la révision 1, mais **rétrogradé** : il ne se justifie que si T1 conclut à la
dominance tangentielle. Si le radial ou le systématique dominent, T7 est un meilleur
investissement pour le même prix.

**Hypothèse.** Deux vues d'un même objet partagent des points d'intérêt, qui donnent des
correspondances **matérielles** — invariantes au glissement du centre de bbox — et dont la
triangulation rend simultanément une position correcte et une **étendue mesurée**.

**Contexte.** Le repo n'a aucun code SfM. Les essais rapportés dans
`2026-08-11-object-search-geometry-investigation.md` (SIFT, DISK+LightGlue) ont des scripts
absents et un verdict tiède, **mais ils cherchaient à *décider* l'association**, pas à
raffiner une position sur une association connue. Le panier donne l'appariement
gratuitement : c'est un problème différent et bien plus facile.

**Contrainte.** Pas d'intrinsèques — `fov_x`/`fov_y` de `candidates.py:475-476` sont des
extensions angulaires de bbox, pas des focales. Donc : dewarper la bbox en patch perspectif
(`cv2.remap`, machinerie dans `third_party/object_search/inference/crop.py`), matcher, relever
chaque correspondance en rayon unitaire via `vendored/erp.theta_phi_to_opengl_ray`, trianguler
en géométrie sphérique avec les poses EUS du manifeste, déjà camera-to-world.

**Gate avant d'écrire quoi que ce soit (R2).** Sur cinq paniers couvrant les deux régimes de
portée : inliers médians par paire de vues (**SIFT d'abord**, ligne de base gratuite que le
document du 11/08 dit supérieure à DISK+LightGlue ici), parallaxe des paires qui matchent,
fraction de paires qui matchent. **Arrêt si** moins de 10 inliers médians, ou moins de la
moitié des paires qui matchent, ou parallaxe sous 5°. Ce dernier cas est le plus vicieux :
les vues se ressemblent parce qu'elles sont proches, et ce sont celles qui ne trianguleront
rien.

**Piège.** Coût quadratique en nombre de vues, et certaines annotations ont ~20 détections.
Mesurer le temps par panier dès le gate, pour savoir si c'est un raffinement en ligne ou un
traitement d'ingestion.

---

## T6 — Reprendre E3 avec une étendue mesurée (~1 jour, conditionné à T5)

E3 a échoué pour une raison unique et identifiée : `e_i = 0.5 · range · hypot(aw, ah)` est
dominé par la portée, donc anti-corrélé au besoin sur les deux cartes. Si T5 rend une étendue
issue d'un nuage de points, cette dépendance disparaît.

Rien à réimplémenter : `merge_score.latent_cost(object_extent_m=...)` accepte déjà un tableau
par point, `cluster_extent_m` prend la médiane, `test_merge_score.py` garantit qu'un scalaire
reproduit l'ancien comportement au bit près. Le code d'E3 est bon, son entrée était mauvaise.

**Critère de succès.** Celui d'E3, inchangé et jamais atteint : le portique passe à 1 cluster
sans que les deux chaises voisines fusionnent, sur les deux familles de paniers.

---

## Ce qui reste hors périmètre, et pourquoi

- **ODAM (arXiv:2108.10165).** Front-end GNN appris — vous avez exclu le réentraînement — et
  back-end d'ajustement de **super-quadriques sous prior d'échelle**, c'est-à-dire E4,
  qu'E0 a privé de justification. Une seule idée à garder, déjà intégrée à T7 : l'association
  est **frame-to-model**, jamais frame-to-frame. O(N·K) au lieu de O(N²), et pas de
  transitivité.
- **La sémantique dans l'association.** *Semantic Semi-Incremental Data-Association-Free
  Object SLAM* (Zhang, Hong, Leonard, arXiv:2607.23384, juillet 2026) est le successeur
  direct de la référence de T8 et ajoute labels de classe et vecteurs de features de modèles
  de fondation dans l'association. C'est exactement la brique morte chez vous : cosinus
  MetaCLIP à 0,529, et deux machines *différentes* à 0,75–0,95 de similarité. **Prendre le
  squelette de 2023, pas la sémantique de 2026.** (Note : texte intégral non consulté, seuls
  titre, auteurs et résumé — à vérifier avant de s'appuyer dessus.)
- **Les quadriques contraintes (E4).** Toujours sans objet, et T7 rend une étendue en
  sous-produit du champ.
- **Les balayages `association_sweep` sur cartes complètes.** Validation finale, pas boucle
  d'itération : le rattachement par `--near-m` dépend des positions qu'on modifie. Y revenir
  une fois une piste retenue, jamais pour arbitrer entre deux pistes.
- **Gate VLM, vote kNN, prompts négatifs.** Acquis, orthogonaux, sans interaction avec la
  géométrie. Ils ne bougent pas pendant cette série.

---

## Bibliographie : ce qui transfère, papier par papier

| référence | ce qu'on prend | ce qu'on laisse |
|---|---|---|
| **Fleuret et al., TPAMI 2008** — POM | le modèle génératif entier : explication mutuelle, occlusion naturelle, cardinalité émergente, champ moyen à images intégrales (éq. 19, 27, 30–35) | la soustraction de fond, le tracking Viterbi, la grille monoplan |
| **arXiv:2302.13264** — DAF-SLAM | la scission interne (traj + amers + associations) / externe (K), et l'estimation conjointe de la pose → T8 | le SLAM complet, l'odométrie |
| **arXiv:2607.23384** — Semantic DAF Object SLAM | rien de directement utilisable | toute la partie sémantique : c'est l'indice mort chez vous |
| **arXiv:2003.08162** — comptage de foule 3D | projeter le long du rayon plutôt qu'en un point ; l'asymétrie de la loss PCM (absent d'une vue ⇒ pas de pénalité) | le CNN entraîné de bout en bout |
| **arXiv:1811.01417** — comptage de fruits | le rejet argumenté du seuil de distance 3D, remplacé par reprojection + hongrois global exclusif ; le motif « chaînage bon marché puis réassociation globale » → T7 étape 2 | le SfM, le vote de profondeur binaire devant/derrière |
| **arXiv:2203.15691** — cartes de densité | seuil dérivé analytiquement, comptage par composante, GMM à K fixé par le comptage → T7 étape 1 | l'hypothèse d'absence d'occlusion, Σ constant |
| **arXiv:2108.10165** — ODAM | frame-to-model uniquement | le GNN appris, les super-quadriques (= E4) |

**Ce qu'aucun des sept ne résout.** Tous supposent une classe unique et homogène : Σ constant,
personnes de 175 cm, mangues. Votre vocabulaire est ouvert, donc l'ensemble des classes n'est
pas borné, et POM comme le comptage de foule réclament un prior de taille par classe. C'est
le trou du dossier, et l'atténuation proposée en T7 — une taille physique par nom de classe,
dérivée une fois hors ligne — est un prior sur un fait du monde, pas un hyperparamètre de
lieu. Nettement plus défendable que `clustering_eps_m`, qui lui est bien un paramètre de lieu.
