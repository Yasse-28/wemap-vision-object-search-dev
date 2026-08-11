# Session — MetaCLIP2 vs SigLIP2, et la langue du prompt

Date : 2026-08-11, troisième volet après
[l'enquête géométrie](2026-08-11-object-search-geometry-investigation.md) et
[le score en ratio](2026-08-11-scoring-ratio-and-two-gate-association.md).
Carte : `bbhotel-choisy` uniquement (674 annotations, 12 prompts, 27 226 candidats
ingérés). Scripts dans le scratchpad de la session : `siglip_embed.py`,
`model_comparison.py`, `model_comparison_lang.py`, `lampe_diagnostic.py`,
`per_class_ap.py`, `so400m_comparison.py`.

**Conclusion en une ligne : le modèle d'embedding n'est pas le goulot, la langue du
prompt vaut plus que tout changement de modèle, et les deux classes cassées le restent
sous les trois modèles.**

---

## 1. Dispositif

Seuls les embeddings sont recalculés. Détection, extraction des cutouts, profondeur et
lifting 3D sont réutilisés tels quels : l'entrée de chaque modèle est la **vignette
stockée** (224 px) des candidats déjà en base. Tout l'aval est le code de production
importé — `load_enriched_candidates`, `localize_from_enriched_candidates` — et le
matcher et la courbe PR du banc.

Deux écarts assumés par rapport aux runs HTTP, qui rendent ces chiffres **non
comparables** à ceux du volet précédent :

- **récupération exacte et non HNSW**, identiquement pour tous les bras : similarités
  calculées sur les 27 226 candidats (MetaCLIP2 dans Postgres avec `1 − d²/2`, SigLIP2
  en numpy), puis top-`K_INTERNAL` passé à l'enrichissement ;
- **`min_similarity` désactivé** (mis au plancher du top-K). Le 0.2 est calibré sur
  l'échelle du cosinus MetaCLIP2 et filtrerait autre chose sur SigLIP2. C'est le score
  en ratio adopté le même jour qui rend la comparaison possible : étant sans échelle, il
  permet de mettre deux modèles côte à côte, ce que l'ancien score pondéré interdisait.

Fusion = moyenne des ratios au top de chaque modèle. Pas une moyenne des cosinus bruts :
les échelles diffèrent, et une moyenne brute revient à garder la plus grande échelle en
y ajoutant du bruit.

---

## 2. Résultat global (mAP, vérité stricte)

| bras | vision | dim | mAP |
|---|---|---|---|
| fusion MetaCLIP2 + so400m, 25 %, EN | — | — | **0.714** |
| MetaCLIP2 huge, EN | ~630 M | 1024 | 0.711 |
| SigLIP2 so400m, EN | ~400 M | 1152 | 0.697 |
| fusion MetaCLIP2 + so400m, 50 %, EN | — | — | 0.694 |
| SigLIP2 base, EN | ~86 M | 768 | 0.692 |
| fusion base + so400m, 50 %, EN | — | — | 0.679 |
| **MetaCLIP2 huge, FR** *(production)* | ~630 M | 1024 | **0.672** |
| SigLIP2 so400m, FR | ~400 M | 1152 | 0.621 |
| SigLIP2 base, FR | ~86 M | 768 | 0.604 |

- **La capacité ne paie pas.** so400m bat base de **+0.005** pour 4,6× les paramètres.
  Les trois modèles, de 86 M à ~630 M, tiennent dans **0.019 de mAP**.
- **La langue pèse quatre à quinze fois plus** : +0.039 pour MetaCLIP2 (0.672 → 0.711),
  **+0.076 pour so400m** (0.621 → 0.697).
- **La fusion n'apporte rien** : +0.003 sur le meilleur modèle seul, soit du bruit à 12
  prompts. Et la fusion base + so400m (0.679) est *pire* que so400m seul : moyenner deux
  modèles de la même famille ne crée rien. Le peu de complémentarité observé venait
  d'avoir deux familles, pas deux vues.
- **Une fusion à poids fixe n'est pas robuste** : `fusion 50 % FR` tombe à 0.145 d'AP sur
  `TV` parce que SigLIP2 y fait 0.104 en français, alors que MetaCLIP2 seul fait 0.843.

## 3. AP par classe

| classe | GT | MetaCLIP2 FR | MetaCLIP2 EN | base EN | so400m FR | so400m EN | fus. 25 % EN | étendue, 3 modèles EN |
|---|---|---|---|---|---|---|---|---|
| chaise | 213 | 0.138 | 0.136 | 0.127 | 0.110 | 0.122 | 0.131 | **0.014** |
| detecteur de fumée | 119 | 0.925 | 0.935 | 0.905 | 0.891 | **0.936** | 0.917 | 0.031 |
| signe de sortie | 88 | 0.816 | 0.837 | **0.883** | 0.816 | 0.842 | 0.838 | 0.046 |
| table | 77 | 0.242 | 0.242 | 0.260 | **0.270** | **0.270** | 0.285 | **0.028** |
| lampe | 55 | 0.552 | 0.594 | 0.527 | 0.632 | **0.648** | 0.546 | 0.121 |
| cctv | 41 | 0.908 | 0.933 | 0.919 | 0.916 | 0.894 | **0.942** | 0.039 |
| extincteur | 29 | **0.999** | 0.997 | 0.983 | 0.980 | 0.997 | 0.998 | 0.014 |
| plante | 22 | 0.690 | 0.741 | 0.744 | **0.826** | 0.783 | 0.734 | 0.042 |
| ascenseur | 21 | 0.929 | **0.955** | 0.815 | 0.930 | 0.927 | 0.924 | 0.140 |
| TV | 5 | 0.843 | 0.781 | 0.843 | 0.150 | **1.000** | 0.813 | 0.219 |
| poubelle | 2 | 0.393 | **0.833** | 0.750 | 0.267 | 0.393 | **0.833** | 0.440 |
| defibrillateur | 2 | 0.625 | 0.548 | 0.550 | **0.667** | 0.550 | 0.600 | 0.002 |
| **mAP** | 674 | 0.672 | 0.711 | 0.692 | 0.621 | 0.697 | **0.714** | |

La colonne d'étendue est la plus instructive : sur les huit classes à plus de 20
annotations, l'écart entre les trois modèles dépasse 0.05 dans **deux cas** (`lampe`,
`ascenseur`). Les grandes étendues restantes sont sur 2 et 5 annotations.

**`chaise` (0.110–0.138) et `table` (0.242–0.285) sont plates sur les huit bras et les
trois modèles.** Ce ne sont pas des problèmes de représentation. Rappel du volet
précédent : la double porte fait passer l'AP de `chaise` de 0.138 à 0.690 et celle de
`table` de 0.246 à 0.799.

---

## 4. Le piège méthodologique, découvert ici et rétroactif

Le premier tableau par classe de cette comparaison était en **F1 au seuil global
réoptimisé de chaque bras**, et il était faux. Il annonçait `lampe` à 0.226 pour
MetaCLIP2 en anglais contre 0.494 en français — « un modèle multilingue qui perd la
moitié de sa performance entre lampe et lamp ». Diagnostic (`lampe_diagnostic.py`) :

| bras | préds | AP | meilleur F1 | son seuil | F1@0.905 | F1@0.931 |
|---|---|---|---|---|---|---|
| MetaCLIP2 « lampe » | 151 | 0.552 | 0.566 | 0.896 | 0.489 | 0.314 |
| MetaCLIP2 « lamp » | 152 | **0.594** | **0.617** | 0.860 | 0.438 | 0.226 |

L'anglais est **meilleur**. Le 0.226 était `lampe` lu au seuil global du bras anglais
(0.931), très loin de son optimum propre (0.860) ; le recouvrement des top-1000 entre
les deux langues est de 78 %, et les vignettes des 12 premiers montrent les mêmes
suspensions jaunes des deux côtés.

**Règle à appliquer désormais : un tableau par classe se lit en AP.** Un seuil global est
choisi pour maximiser la macro et pénalise mécaniquement toute classe dont l'optimum est
ailleurs ; comparer des colonnes dont les seuils vont de 0.863 à 0.931 ne compare pas les
modèles, il compare les positions relatives de leurs seuils.

### Correction rétroactive : la porte sémantique, en AP

Le tableau par classe du volet précédent avait le même défaut. Recalculé en AP, le
verdict **se renforce** :

| classe | GT | pondéré | ratio | porte 0.80 | porte 0.85 | porte 0.90 |
|---|---|---|---|---|---|---|
| chaise | 213 | 0.140 | 0.138 | 0.221 | 0.396 | **0.690** |
| detecteur de fumée | 119 | **0.930** | 0.925 | 0.871 | 0.876 | 0.597 |
| signe de sortie | 88 | **0.871** | 0.816 | 0.848 | 0.858 | 0.853 |
| table | 77 | 0.249 | 0.246 | 0.375 | 0.711 | **0.799** |
| lampe | 55 | 0.561 | 0.552 | **0.748** | 0.691 | 0.626 |
| cctv | 41 | **0.916** | 0.908 | 0.909 | 0.857 | 0.774 |
| extincteur | 29 | **0.999** | **0.999** | 0.996 | 0.989 | 0.924 |
| plante | 22 | 0.460 | 0.461 | **0.681** | 0.626 | 0.468 |
| ascenseur | 21 | 0.943 | 0.929 | **0.970** | 0.907 | 0.580 |
| TV | 5 | **0.883** | 0.843 | 0.754 | 0.488 | 0.400 |
| **mAP** | 674 | 0.652 | 0.653 | 0.694 | **0.698** | 0.630 |

**« Les objets compacts perdent » était largement un artefact de seuil.** À la porte
0.80, `extincteur` passe de 0.999 à 0.996, `cctv` de 0.908 à 0.909, et `ascenseur`
*gagne* (0.929 → 0.970). Les seules vraies pertes sont `detecteur de fumée` (−0.054) et
`TV` (−0.089), contre des gains de +0.196 sur `lampe`, +0.220 sur `plante`, +0.129 sur
`table`. Le compromis de la porte à 0.80 est bien meilleur que ce que les F1 laissaient
croire — ce qui **ne change pas** la décision de la laisser à off, dont la raison reste
la contradiction structurelle entre les deux vérités terrain.

---

## 5. Ce qui n'est pas établi

- **La résolution d'entrée.** Les vignettes font 224 px, les cutouts viennent d'ERP en
  5760×2880. La capacité n'ayant rien donné à résolution constante, c'est le seul facteur
  plausible restant côté embeddings — mais le tester impose de ré-extraire les cutouts,
  donc ce n'est plus « ne recalculer que les embeddings ».
- **Une seule carte**, et les classes à 2 ou 5 annotations ne portent aucune conclusion.
- **Le gain de langue en usage réel.** Les annotations et les requêtes utilisateur sont en
  français ; la forme déployable est une table de traduction des prompts canoniques, pas
  une traduction à la volée, et il reste à vérifier que les requêtes libres n'y perdent
  pas.

## 6. Décisions

1. **Traduire les prompts canoniques en anglais.** +0.039 de mAP sur la configuration de
   production, aucun changement de modèle, d'index ni de schéma. Meilleur rapport
   gain/effort de la journée.
2. **Ne pas intégrer SigLIP2.** Seconde colonne d'embeddings (768 ou 1152), second index
   HNSW, second modèle à servir — non payé par +0.003.
3. **Arrêter d'optimiser le modèle d'embedding sur cette carte.** Trois modèles, deux
   familles, un facteur 7 de capacité, trois fusions : tout tient dans 0.042 de mAP.
   Le plafond est dans l'association et dans la vérité terrain, pas dans la
   représentation.
4. **Rapporter les comparaisons par classe en AP.** Vaut pour tout ce qui suit, et a
   déjà invalidé une conclusion par classe du volet précédent.
