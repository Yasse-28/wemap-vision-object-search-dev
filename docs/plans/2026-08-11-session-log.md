# Journal de session — 2026-08-11, dans l'ordre

Compte rendu chronologique complet de la session : ce qui a été demandé, ce qui a été
fait, ce qui a été trouvé, et ce qui a été corrigé. Les quatre documents de résultats
disent *quoi*. Celui-ci dit *dans quel ordre et pourquoi*, y compris les détours.

Entrée synthétique : [`2026-08-11-session-index.md`](2026-08-11-session-index.md).
Carte unique : `bbhotel-choisy` (674 annotations, 12 prompts, 27 226 candidats ingérés).
Branche : `wip/improve-search-strategy`. Commits produits : `b33b742`, `a0a9a7c`,
`0a69315`, `1cf34f6`, `b1ed239`, et celui-ci.

---

## 1. Point de départ

Fourni : le compte rendu de la session précédente
([volet 1](2026-08-11-object-search-geometry-investigation.md)) — boost de feedback,
courbe précision-rappel, deux défauts de données, sept tentatives géométriques toutes
négatives, et une liste de six choses à faire.

Relevé d'entrée : l'item 1 de cette liste (optimiser `acceptance_threshold` par prompt,
+26 points de F1 annoncés) était marqué « demandé puis interrompu ». Deux réserves
soulevées avant de s'y remettre : l'item 3 interagit avec l'item 1 (un cluster à 2
keyframes plafonne à 0.883 et ne peut jamais passer 0.9, donc le seuil optimal absorbe
cet artefact), et un seuil réglé par prompt sur une seule carte est du fit *in-sample*.

**Demande : « des solutions globales », en commençant par un fit sur l'ensemble des
classes.**

## 2. Fit d'un seuil global sur les 12 classes

Le banc évaluait déjà tous les prompts du fichier d'annotations sans `--only-prompt`. Run
complet à `num_results` 400 (pour ne pas tronquer la queue des courbes PR), puis
post-traitement hors ligne comparant quatre règles d'acceptation **globales**.

| règle globale | macro F1 strict | LOO | macro F1 groupé | LOO |
|---|---|---|---|---|
| seuil `match_score` 0.9 (état du jour) | 0.445 | — | 0.580 | — |
| meilleur seuil `match_score` partagé | 0.598 (t=0.776) | 0.533 | 0.597 (t=0.887) | 0.552 |
| seuil `similarity_score` absolu | 0.502 | 0.496 | 0.368 | 0.138 |
| top-k global | 0.481 | 0.445 | 0.481 | 0.424 |
| seuil par prompt (plafond) | 0.712 | — | 0.724 | — |

Quatre conclusions : un seuil unique capte la majorité du gain (+15.3 des +26.7 points, et
+8.8 survivent au leave-one-prompt-out) ; le mécanisme n'est pas l'artefact des 2
keyframes (la bande `(0.776, 0.9]` contient 191 prédictions dont **156 vrais positifs**,
et seulement 15 sont des clusters à 2 keyframes) ; la vérité groupée à 2 m annule le levier
(LOO 0.552 sous la référence) ; et la normalisation par requête gagne sa place face à un
seuil absolu.

Cas limite relevé : `chaise` plafonne à F1 0.246 **quel que soit le seuil**, 31 prédictions
retournées pour 213 annotations — coupé en amont par `min_similarity`, pas par
l'acceptation.

## 3. Refonte du scoring

**Demande : rendre le scoring plus interprétable.**

Mesure de la contribution de chaque terme de
`match_score = 0.50·norm_sim + 0.15·confidence + 0.35·min(1, kf/3)`, en mAP :

| classeur | mAP strict | mAP groupé |
|---|---|---|
| score complet | 0.652 | 0.715 |
| `similarity_score` seule | **0.653** | 0.713 |
| `confidence` seule | 0.533 | 0.602 |
| `min(1, kf/3)` seule | 0.318 | 0.172 |

Quatre défauts nommés : la moitié du budget de poids ne classe rien ; les termes sont
saturés (65 % des clusters ont ≥ 3 keyframes, 53 % ≥ 5 observations) donc constants ; la
taille est comptée trois fois (`kf/3`, `min(1, n_obs/5)` dans `confidence`, et le `max` sur
N détections) ; et `norm_sim` dépend de `min_similarity`, un **paramètre de filtrage** —
le faire passer de 0.2 à 0.15 déplaçait tous les scores sans qu'aucune évidence ne change.

Sept calibrations comparées ; `ratio = sim / max_sim` gagne partout, et surtout réduit
l'écart in-sample → LOO de −6.5 à −2.1 points. C'est cette propriété qui compte : le seuil
transfère à un prompt jamais vu.

**Question posée : un seuil sur la similarité brute marche-t-il mieux ? Non.** Les optima
par prompt s'étalent sur 0.084 alors que la dynamique *intra*-prompt fait 0.025 à 0.099 —
`plante` tient entre 0.207 et 0.231, l'optimum de `defibrillateur` est 0.290. Les deux
prompts n'ont aucun point de fonctionnement commun. Imposer le seuil partagé coûte 0.191
de F1 par prompt contre 0.060 pour le ratio.

## 4. Consigne de méthode, et exclusion de vinci

**Instruction : tous les tests sur `bbhotel-choisy`. Les données de vinci ne sont pas
sûres, celles de review sont partielles.**

Elle est arrivée après que j'aie joué `vinci-st-domingue` comme validation inter-cartes.
Ce run confirmait le défaut : mAP 0.154 contre 0.652, `emergency power plant` 0 vrai
positif sur 6, `e gates` 0 sur 4, `FIDS` rappel 0.046. Le tableau de transfert inter-cartes
que j'en avais tiré a donc été écarté, y compris son résultat contraire en groupé. La
généralisation entre cartes reste **non établie**, ni pour ni contre.

Consigné en mémoire de projet (`benchmark-data-to-trust`) : une carte cassée ressemble à
une validation inter-cartes tout en ne portant aucun signal, et la rapporter comme
corroboration est pire que de n'avoir qu'une carte.

## 5. Implémentation — et un blocage

**Demande : implémenter le ratio en `score_strategy`, supprimer le weighted, sortir la
géométrie du score pour en faire du filtrage.**

Deux frictions signalées au passage :

- **le verrou multi-agent était tenu** par `claude-merge` (`state=coding`, heartbeat 312 s
  contre 1800 s de péremption). `AI_RULES.md` interdit d'écrire dans ces conditions et
  `localize.py` est exactement le fichier où une édition concurrente ferait des dégâts.
  Rien n'a été forcé ; l'ordre du travail a été inversé et la revue de littérature est
  passée avant l'implémentation. Le verrou s'est libéré ensuite, et l'arbre était devenu
  propre : l'autre agent avait commité le travail de la session précédente (`78b98a4`),
  rendant fausse la ligne « Rien n'est commité » du volet 1 ;
- **`score_strategy` + suppression du weighted** donnent un paramètre à valeur unique, que
  `AI_RULES.md` proscrit comme code mort. Arbitrage pris et signalé : remplacement franc,
  pas de commutateur. C'était la consigne la plus spécifique des deux.

Livré dans `b33b742` : `match_score = cluster_best_sim / meilleur cluster de la requête` ;
`filter_clusters_by_geometry` (`min_observations_per_cluster`, `max_cluster_spread_m`, tous
deux à off) appliqué **avant** le ranking pour que le dénominateur soit un cluster
réellement renvoyé ; `spread_m` remonté dans la réponse et dans les types du frontend ;
`normalized_similarity` supprimé. Le run bout-en-bout reproduit la prédiction hors-ligne
**exactement** (macro F1 0.632 contre 0.632 prédit), après vérification que la
reconstruction hors-ligne du score valait le `match_score` du service à 1.55e-15 sur 2148
clusters.

Trois erreurs de lint et deux de mypy traînaient depuis le commit précédent ; corrigées
pour que la porte du repo repasse au vert.

## 6. Revue de littérature

**Demande : existe-t-il des approches de la littérature qui font ce que fait l'object
search ?** Oui, et ça s'appelle **open-vocabulary 3D instance retrieval**.

OVIR-3D (CoRL 2023) est l'analogue structurel le plus proche, mais fusionne sur la
similarité de features et non sur la géométrie. ConceptGraphs (ICRA 2024) fournit la
double porte — IoU de boîtes 3D **puis** vérification par cosinus, conjonctives. HOV-SG
(RSS 2024) et KeySG (2025) font du niveau une structure d'index au lieu d'un veto tardif.
C-DOG (2025) est le partitionnement de graphe que la §4 du volet 1 laissait ouvert :
arêtes par consistance épipolaire puis clustering par δ-recouvrement, pas des composantes
connexes. FroDO formule le clustering de rayons comme problème ouvert, ce qui recadre
l'échec 3.3 comme un résultat et non une erreur de mise en œuvre.

Trois transferts retenus : la double porte, la hiérarchie étage/pièce, et le temperature
scaling comme concurrent du ratio. Aucun de ces travaux ne tourne sans reconstruction
dense — c'est là qu'on est seuls.

### Références

Liste canonique de la session. La colonne « lu » est honnête et importe pour la réutilisation :
les papiers marqués *résumé* n'ont été vus qu'à travers un résumé de recherche, pas lus.

| sujet | référence | lu |
|---|---|---|
| L'analogue structurel le plus proche : propositions 2D open-vocab fusionnées en instances 3D, sans entraînement 3D. Fusion sur similarité de features, pas sur la géométrie. | [OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data](https://arxiv.org/abs/2311.02873) (CoRL 2023) | résumé |
| **La double porte** implémentée en `a0a9a7c` : IoU de boîtes 3D (seuil 0.03) *puis* vérification par cosinus, conjonctives. | [ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://concept-graphs.github.io/) (ICRA 2024) | résumé |
| Hiérarchie étage → pièce → objet, requête décomposée et interrogée séquentiellement. Piste pour les 9 niveaux de bbhotel, où le niveau est aujourd'hui un veto tardif. | [Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation](https://hovsg.github.io/) (RSS 2024) | résumé |
| Le plus proche de notre modèle de données — scene graph hiérarchique bâti sur des **keyframes**, évalué sur de la récupération d'objets multi-étages. | [KeySG: Hierarchical Keyframe-Based 3D Scene Graphs](https://arxiv.org/pdf/2510.01049) (2025) | résumé |
| Localisation d'objets 3D **sans reconstruction** — le cadre le plus proche du nôtre. À lire en premier si on reprend le sujet. | [Memory Over Maps: 3D Object Localization Without Reconstruction](https://arxiv.org/pdf/2603.20530) (2026) | résumé automatique, vague — ne rien en citer sans l'avoir ouvert |
| Le partitionnement de graphe que la §4 du volet 1 laissait ouvert : arêtes par consistance épipolaire, puis clustering par δ-recouvrement et filtrage — **pas** des composantes connexes. Sans entraînement. | [C-DOG: Training-Free Multi-View Multi-Object Association](https://arxiv.org/pdf/2507.14095) (2025) | résumé |
| Formule l'association comme « identifier un nombre inconnu de segments de droite s'intersectant approximativement en un point » — soit exactement l'expérience 3.3 du volet 1, posée comme problème ouvert. | [FroDO: From Detections to 3D Objects](https://arxiv.org/pdf/2005.05125) (2020) | résumé |
| Soutient que le goulot de la perception open-world fine est la discrimination de CLIP lui-même. Recoupe l'axe compact/étendu. | [Is CLIP the main roadblock for fine-grained open-world perception?](https://arxiv.org/pdf/2404.03539) (2024) | résumé |
| Normaliser le score d'un échantillon par sa similarité à ses paires contrastives (negCLIPLoss). Notre `sim / max_sim` en est la version pauvre ; le **temperature scaling** y est le concurrent à tester. | [CLIPLoss and Norm-Based Data Selection Methods for Multimodal Contrastive Learning](https://arxiv.org/html/2405.19547v1) (2024) | résumé |
| Jeu de référence pour la détection sur équirectangulaire (~3 000 images, 90 000 labels). | [360-Indoor: Towards Learning Real-World Objects in 360° Indoor Equirectangular Images](https://arxiv.org/abs/1910.01712) (WACV 2020) | résumé |
| Estimation de profondeur sur panoramas sphériques. Contexte du défaut §2 du volet 1 : la profondeur explose sur les rayons montants, cohérent avec la distorsion ERP près des pôles. | [Rethinking Supervised Depth Estimation for 360° Panoramic Imagery](https://openaccess.thecvf.com/content/CVPR2022W/OmniCV/papers/He_Rethinking_Supervised_Depth_Estimation_for_360deg_Panoramic_Imagery_CVPRW_2022_paper.pdf) (CVPRW 2022) | résumé |

Réserve valable pour toute la liste : elle vient d'une recherche menée en une passe, et
aucun de ces papiers n'a été lu intégralement. Les chiffres qu'ils annoncent (seuils, mIoU)
ne sont pas vérifiés ; ce qui a été **repris et mesuré ici**, c'est la règle d'association
de ConceptGraphs, et elle l'a été sur nos données, pas sur les leurs.

## 7. La double porte, implémentée et mesurée

**Demande : tester et benchmarker la double porte de ConceptGraphs.**

`a0a9a7c` : `semantic_gate_threshold` dans `cluster_detections_leader_canopy` — une
détection rejoint le seed seulement si elle passe le rayon **et** un cosinus
cutout↔cutout. Embeddings chargés depuis pgvector uniquement quand la porte est active,
pour que le SQL du chemin par défaut reste inchangé au caractère près.

Quatre tests sur cas synthétiques **avant** toute mesure, en réponse directe au « deux bugs
de géométrie non testés » de la §6 du volet 1 : deux objets orthogonaux dans une même
boule se séparent, un objet unique reste entier, un seuil sans embeddings est inerte, un
vecteur non normalisé ne passe pas la porte par sa seule longueur.

Résultat : optimum en plateau sur 0.80–0.85, mAP 0.698 contre 0.653. Laissé à off, parce
que les deux vérités terrain donnent des verdicts opposés.

## 8. Tableau par classe, puis la première correction

**Demande : un tableau par classe.** Produit en F1 au seuil global réoptimisé de chaque
colonne. Il annonçait des gains massifs sur les fusionneurs et des pertes nettes sur les
compacts. Commité en `0a69315`.

**Ce tableau était confondu**, découvert deux étapes plus tard (§10). Recalculé en AP, « les
objets compacts perdent » tombe : à la porte 0.80, `extincteur` 0.999 → 0.996,
`cctv` 0.908 → 0.909, `ascenseur` 0.929 → **0.970**. Seules vraies pertes : `detecteur de
fumée` (−0.054) et `TV` (−0.089).

## 9. MetaCLIP2 vs SigLIP2

**Demande : tester SigLIP2 contre MetaCLIP2, et les deux ensemble ; embeddings seulement.**

Faisable parce que les vignettes des 27 226 candidats ingérés sont sur disque : détection,
cutouts, profondeur et lifting réutilisés tels quels. Les cosinus MetaCLIP2 se calculent
dans Postgres avec la formule de production, donc seul SigLIP2 était à embarquer.

Deux écarts assumés, signalés avant les chiffres : récupération exacte au lieu de HNSW, et
`min_similarity` désactivé — le 0.2 est calibré sur l'échelle MetaCLIP2. **C'est le
changement de score de l'étape 5 qui rend la comparaison possible** : un ratio est sans
échelle, le mélange pondéré ne l'était pas.

Premier verdict, sur prompts français : MetaCLIP2 gagne, SigLIP2 perd, la fusion n'aide pas.

## 10. Le confondant linguistique, et le diagnostic `lampe`

Confondant repéré avant de conclure : **les 12 prompts sont français**, MetaCLIP2
*worldwide* est multilingue par construction, et les classes où SigLIP2 s'effondrait
étaient celles à terme français distinctif. Test en anglais, deux minutes puisque les
embeddings d'images étaient en cache.

SigLIP2 gagne +0.088 de mAP en anglais, MetaCLIP2 +0.039, et la meilleure configuration
devient la fusion en anglais. Mais le tableau par classe montrait `lampe` à 0.226 pour
MetaCLIP2 en anglais contre 0.494 en français, d'où une affirmation écrite noir sur blanc :
« un modèle multilingue qui perd la moitié de sa performance entre lampe et lamp ».

**Diagnostic demandé, et affirmation réfutée** :

| bras | préds | AP | meilleur F1 | son seuil | F1@0.905 | F1@0.931 |
|---|---|---|---|---|---|---|
| « lampe » | 151 | 0.552 | 0.566 | 0.896 | 0.489 | 0.314 |
| « lamp » | 152 | **0.594** | **0.617** | 0.860 | 0.438 | 0.226 |

L'anglais est meilleur. Recouvrement des top-1000 : 78 %. Montage des 12 premières
vignettes : les mêmes suspensions jaunes des deux côtés, le français plaçant en plus une
étiquette hors-sujet et un cadre noir. Le 0.226 était `lampe` lu au seuil global du bras
anglais, très loin de son optimum propre.

**Règle adoptée : les comparaisons par classe se rapportent en AP.** Un seuil global est
choisi pour maximiser la macro et pénalise mécaniquement toute classe dont l'optimum est
ailleurs. Appliquée rétroactivement au tableau de l'étape 8.

## 11. so400m, et la fin de la piste modèles

**Demande : lancer so400m.** Variante `so400m-patch16-256` choisie pour que seule la
capacité change face au *base* déjà mesuré — même patch, même résolution, et pas
d'upscaling supplémentaire des vignettes 224 px.

| bras | vision | mAP |
|---|---|---|
| fusion MetaCLIP2 + so400m 25 %, EN | — | 0.714 |
| MetaCLIP2 huge, EN | ~630 M | 0.711 |
| SigLIP2 so400m, EN | ~400 M | 0.697 |
| SigLIP2 base, EN | ~86 M | 0.692 |
| MetaCLIP2 huge, FR *(production)* | ~630 M | 0.672 |

so400m ne gagne que **+0.005** sur base pour 4,6× les paramètres ; les trois modèles
tiennent dans 0.019 de mAP ; la fusion apporte +0.003, du bruit ; et base + so400m (0.679)
est *pire* que so400m seul — moyenner deux modèles d'une même famille ne crée rien. La
langue vaut quatre à quinze fois plus que tout ça. `chaise` (0.110–0.138) et `table`
(0.242–0.285) sont plates sur les huit bras. Commité en `1cf34f6`.

Décisions : traduire les prompts canoniques, ne pas intégrer SigLIP2, arrêter d'optimiser
la représentation.

## 12. IoU 3D : question posée, réponse négative

**Question : tester l'IoU de boîtes 3D serait-il pertinent ?**

Réponse donnée : pas sous cette forme. Nous n'avons pas d'étendue 3D — un point par
détection plus une étendue angulaire — et la fabriquer suppose d'inventer la profondeur de
la boîte, dont la taille serait proportionnelle à `d`, la profondeur que le volet 1 montre
exploser sur les rayons montants. L'IoU étant très non linéaire, une erreur de facteur 2
fait passer un recouvrement de 0.5 à 0.03 ; la distance est linéaire en l'erreur. Et
ConceptGraphs seuille l'IoU à **0.03**, soit « est-ce que ça se touche » — leur porte
géométrique est permissive, c'est la sémantique qui trie.

Test préalable proposé, une heure contre plusieurs jours : **est-ce que la porte géométrique
est encore contraignante ?** Prior annoncé : `eps` est plat.

## 13. Le balayage `eps`, et la troisième correction

**Prior faux.** De 2.0 à 0.5 m, la mAP stricte passe de 0.653 à **0.788** — +0.135, plus que
tout autre levier de la journée réuni. Mais les deux vues sont **anti-corrélées et
monotones** sur toute la plage : le groupé fait 0.713 → 0.675 dans le même mouvement.

La métrique est dégénérée le long de l'axe découper/fusionner. Elle ne classe pas deux
granularités, elle rapporte laquelle des deux vérités terrain on vient de choisir. Règle de
tri : les interventions qui changent le **classement** à granularité fixe sont mesurables
ici — le score et les modèles ont bougé dans le même sens sur les deux vues — celles qui
changent la **granularité** ne le sont pas.

Et **la porte sémantique s'avère redondante avec `eps`** : à 0.5 elle dégrade le strict
(0.788 → 0.703). Elle défaisait un sur-regroupement créé par le rayon. Troisième
rétractation. Détail dans
[le volet 4](2026-08-11-clustering-radius-and-a-degenerate-metric.md), commit `b1ed239`.

Incident à consigner : 8 des 12 runs du balayage ont **échoué silencieusement**. En zsh une
expansion non quotée ne se découpe pas en mots, donc `$G` a livré
`--semantic-gate-threshold 0.85` comme un seul argument ; et mon `grep -E "^Threshold-free"`
sur la sortie a mangé le message d'erreur en même temps. **Ne pas filtrer la sortie d'un
run long sur un motif de succès** — l'échec devient invisible.

## 14. Bilan

Livré et gardé : le score en ratio, la géométrie en filtres, la porte sémantique en opt-in,
et cinq documents.

Non recommandé, chacun pour une raison mesurée : SigLIP2 (+0.003, du bruit), la porte
sémantique (redondante avec un rayon plus petit), `eps` à 0.5 (impossible de séparer le
gain réel de la complaisance envers la vue stricte), l'IoU 3D (granularité, donc non
mesurable, plus deux raisons propres).

Recommandé : traduire les prompts canoniques (+0.039, indépendant du reste), et surtout
**corriger le regroupement des annotations en leader/canopy**, qui est passé du sixième
rang d'une liste à la dépendance bloquante de toute la ligne géométrique.

### Ce qui a coûté

- **Trois conclusions rétractées, toutes pour la même raison** : un paramètre tenu fixe sans
  justification — d'abord le seuil d'acceptation, puis `eps`. Un résultat obtenu à paramètre
  figé est un résultat sur le réglage, pas sur la méthode.
- **Deux fois, l'affirmation était plus forte que la mesure**, et les deux fois c'est un
  diagnostic ciblé qui a corrigé — pas une relecture. Les vignettes ont tranché `lampe`, le
  balayage a tranché la porte.
- **Un filtrage de sortie a masqué 8 échecs sur 12.**
- Ce qui a marché, à garder : réutiliser le code de production plutôt que le réimplémenter ;
  identifier le confondant *avant* de regarder le résultat (la langue des prompts, l'échelle
  des cosinus) ; vérifier la fidélité du chemin hors-ligne contre le service avant d'en
  tirer des conclusions (1.55e-15) ; et tester les nouveaux critères géométriques sur cas
  synthétique avant usage.
