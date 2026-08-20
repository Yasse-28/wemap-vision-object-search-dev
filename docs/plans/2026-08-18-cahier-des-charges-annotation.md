# Cahier des charges — annotation de la vérité terrain

Document de travail pour la personne qui annote. Les décisions et leurs justifications
sont dans [ADR 0009](../adr/0009-ground-truth-annotation-contract.md) ; ici il n'y a que
ce qu'il faut faire, dans quel ordre, et comment vérifier.

**À vérifier après chaque session d'annotation :**

```bash
python -m toolbox.benchmark.validate_annotations --map-path /path/to/map
```

Il dit ce qui manque, ce qui est incohérent, et quelles colonnes de mesure la carte a
gagnées. C'est le seul juge : si le script est content, le contrat est rempli.

---

## Ordre de priorité

Chaque champ débloque une mesure précise. Dans cet ordre, parce que le premier débloque
le plus et coûte le moins.

| # | Champ | Ce que ça débloque | Effort |
|---|---|---|---|
| 1 | `extent_m` | Les colonnes *classification* et *localisation* de la décomposition d'erreur, aujourd'hui refusées sur les deux cartes. Et un rayon d'appariement honnête, donc **tous** les rappels. | 1 nombre par objet |
| 2 | `object_id` | Le type *doublon*. Sans lui, deux annotations au même endroit sont indistinguables d'un clic dupliqué. | 1 identifiant par objet |
| 3 | `exhaustive_zone` | Les types *background* et *manqué*. Hors zone déclarée, un cluster sur un objet non annoté n'est pas une erreur. | 1 étiquette par zone |
| 4 | `labels.synonyms` | Supprime le faux positif « seat vs chair ». | 2–4 mots par objet |
| 5 | `labels.{depictions,visually_similar,clutter}` | Le *set ranking* d'OpenLex3D. | jugement par objet |
| 6 | Sous-ensemble double-annoté | L'accord inter-annotateur, sans lequel rien ci-dessus n'est interprétable. | 50 objets × 2 |

**Si tu ne fais qu'une chose : `extent_m`.** Les deux cartes refusent actuellement deux
colonnes sur six uniquement parce que le rayon vaut 5 m partout.

---

## 1. `extent_m` — l'emprise, pas le seuil

**La plus grande dimension horizontale de l'objet, en mètres.** Le rayon d'appariement en
est déduit (moitié de l'emprise) ; ne cherche pas à choisir un seuil, donne une taille.

- une chaise : `0.5`
- une table de restaurant : `1.2`
- un comptoir d'enregistrement : `4.0`
- un détecteur de fumée : `0.15`
- un ascenseur (la porte) : `1.5`

**Estime, ne mesure pas.** L'erreur tolérable est de l'ordre de 50 % : ce qui compte est
qu'une chaise soit sous le mètre et un comptoir au-dessus de deux. Une valeur au décimètre
près n'apporte rien de plus qu'une valeur au demi-mètre.

**Le piège.** Surestimer est plus coûteux que sous-estimer : un `extent_m` généreux
regonfle le rayon et refait tomber la carte sous la porte de séparabilité. Mesuré sur les
deux cartes, part d'annotations ayant une autre classe dans leur propre rayon :

| `extent_m` | rayon | vinci | bbhotel |
|---|---|---|---|
| 0.5 m | 0.25 m | 0.8 % ✓ | 7.3 % ✓ |
| 1.0 m | 0.50 m | 6.2 % ✓ | 34.2 % ✗ |
| 2.0 m | 1.00 m | 25.2 % ✓ | 49.6 % ✗ |
| 4.0 m | 2.00 m | 49.6 % ✗ | 84.9 % ✗ |

La limite est un tiers. Donc sur bbhotel, où tout est petit et serré, les emprises doivent
être serrées aussi — ce qui est de toute façon la vérité pour des chaises.

---

## 2. `object_id` — un objet, un identifiant

**Une chaîne stable et unique par objet physique.** Deux annotations qui partagent un
`object_id` décrivent le même objet vu deux fois ; deux annotations avec des `object_id`
différents décrivent deux objets, même à dix centimètres l'une de l'autre.

Convention suggérée : `<classe>-<compteur>`, par exemple `chaise-017`. N'importe quoi
d'unique convient, mais garde-le **stable dans le temps** : si tu réannotes une session,
un objet déjà annoté doit retrouver son identifiant.

**Une rangée de chaises est une rangée d'objets.** Chaque chaise a son propre
`object_id`. Ne regroupe pas en « rangée de chaises » : ça déplacerait le problème de
granularité de la métrique vers la vérité terrain, où il n'est plus mesurable.

**Ce que ça répare.** bbhotel contient 197 paires d'annotations strictement identiques
(même classe, même keyframe, même pixel, même profondeur) — un clic inséré deux fois.
Avec `object_id`, ce cas se lit au lieu de se deviner.

---

## 3. `exhaustive_zone` — où la vérité terrain est complète

**Une étiquette de région dans laquelle tu garantis avoir annoté *tous* les objets des
classes concernées.** Par exemple `hall-depart`, `couloir-etage-2`, `restaurant`.

C'est une garantie, pas une description. Si tu as annoté les chaises du restaurant mais
pas celles de la terrasse, la zone est `restaurant`, pas `rez-de-chaussee`.

**Pourquoi c'est nécessaire.** Un cluster posé sur un objet réel mais non annoté n'est pas
une erreur du pipeline. Sans zone déclarée, il est compté comme *background* et deux
colonnes de mesure deviennent du bruit.

**Si tu ne peux pas garantir l'exhaustivité** d'une région, laisse le champ vide plutôt
que d'inventer une zone. Le script de validation dira combien d'annotations sont hors zone
et les mesures concernées se retireront proprement.

---

## 4. Les ensembles de labels

Quatre listes par objet, du plus précis au moins précis. **Vocabulaire commun aux deux
cartes, en anglais**, avec le terme local en synonyme — deux cartes aux vocabulaires
disjoints ne sont plus comparables, et la comparaison entre cartes est le seul usage que
ces données autorisent.

### `labels.synonyms` — ça nomme l'objet

Les mots qu'on pourrait taper dans la recherche et obtenir légitimement cet objet.

- une chaise : `["chair", "seat", "chaise"]`
- un comptoir : `["check in counter", "check-in desk", "counter"]`
- un détecteur de fumée : `["smoke detector", "detecteur de fumee"]`

Le terme français existant reste, comme synonyme. Ne le supprime pas : c'est ce qui rend
les annotations déjà faites compatibles.

### `labels.depictions` — ça nomme une *image* de l'objet

Réservé aux cas où un pictogramme, une affiche ou un écran montre la chose.

- un panneau de sortie : `["exit sign pictogram", "running man symbol"]`
- une affiche d'avion dans un aéroport, sur l'annotation de l'affiche : `["airplane"]`

Si l'annotation **est elle-même** une image imprimée de sa classe (une photo d'avion, pas
un avion), mets `is_depiction` à vrai en plus.

### `labels.visually_similar` — ça nomme autre chose qui ressemble

Un objet différent qu'on confondrait de loin. Ce n'est pas un synonyme : une requête sur
ce mot ne devrait *pas* renvoyer cet objet, mais s'y tromper est une erreur douce.

- une chaise : `["stool", "armchair", "bench"]`
- un détecteur de fumée : `["ceiling speaker", "smoke alarm light"]`

### `labels.clutter` — ce que le cadrage a ramassé

Ce qui est aussi dans la boîte, sans être l'objet. Sert à mesurer la qualité des cutouts
plutôt que celle de la reconnaissance.

- une chaise sous une table : `["table", "floor"]`
- un extincteur sur un mur : `["wall", "signage"]`

**Regarde le cutout avant de remplir celui-ci.** C'est la seule des quatre listes qui
demande de voir l'image et non l'objet.

### Un mot dans deux listes

Autorisé. Il est crédité de la catégorie la plus précise, donc répéter un mot ne peut pas
baisser un score par accident.

---

## 5. Le sous-ensemble double-annoté

**50 objets par carte, annotés indépendamment par deux personnes**, sans se consulter et
sans voir le travail de l'autre. Tire-les au hasard, pas les plus faciles.

Sans ce chiffre, les six types d'erreur héritent de l'ambiguïté du présent document et on
ne pourra pas la distinguer d'un défaut du pipeline. C'est peu de travail et c'est la seule
vérification que le cahier des charges a été compris.

Ce qu'on en lira : l'accord sur `extent_m` (à 50 % près), sur `object_id` (les deux
personnes ont-elles vu le même nombre d'objets) et sur `labels.synonyms` (intersection sur
union).

---

## Format écrit

Tout va dans `ground_truth_point.extra_properties`, le blob JSON qui porte déjà l'origine
ERP du clic. Aucune migration de schéma. Les clés :

```json
{
  "object_id": "chaise-017",
  "extent_m": 0.5,
  "exhaustive_zone": "restaurant",
  "is_depiction": false,
  "labels": {
    "synonyms": ["chair", "seat", "chaise"],
    "depictions": [],
    "visually_similar": ["stool", "armchair"],
    "clutter": ["table", "floor"]
  }
}
```

Les quatre listes sont aussi acceptées à plat (`"synonyms": [...]` directement dans
`properties`), parce que l'export GeoJSON aplatit selon qui a écrit le fichier. Une chaîne
seule au lieu d'une liste est acceptée comme une liste d'un élément.

`class` garde son rôle actuel d'affichage et de regroupement. Le scoring lit
`labels.synonyms` et retombe sur `class` quand la liste est absente, donc rien ne casse
avant que les labels existent.

---

## Ce qui change dans les chiffres

À dire quand on comparera à l'historique :

- le rayon d'appariement passe d'un `accuracy` plat de 5 m à un rayon par objet. **Tous
  les rappels bougent** ; les chiffres d'avant et d'après ne sont pas comparables.
- bbhotel passe de 674 à environ 477 annotations après déduplication. Ses rappels montent,
  et c'est une correction, pas un progrès.
