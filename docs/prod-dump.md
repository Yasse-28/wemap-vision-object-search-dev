# Dump object-search (prod → local)

Récupérer les sorties de `prepare` déjà calculées en prod plutôt que de les
recalculer sur GPU. Le script vit ici depuis la passe du 2026-08-20 ; il remplace
`~/Workspace/codes/wemap/dump_object_search.sh`, qui n'était suivi par aucun dépôt.

1. **Télécharger**, en passant le dossier de la carte (ou son export pivot) :
   ```bash
   scripts/dump-object-search.sh [--thumbnails|--thumbnails-only] [--force] [--dry-run] \
       <map_path | map_id> [dest_dir]
   # ex: scripts/dump-object-search.sh /media/yacine/T7/Wemap/vps-data/maps/vinci-st-domingue/
   ```
   → `metadata.parquet` + `embeddings.npy` par video_capture depuis
   `s3://wemap-vision-storage-prod/<map_id>/<video_capture_id>/object_search/`,
   plus `trajectory.json` (étape 2).
   Destination par défaut : `<map_dir>/object-search/`, sinon
   `./object_search_dump/<map_id>/`. `dest_dir` l'écrase toujours.

   **Le fichier pivot décide quoi télécharger.** S3 conserve toutes les captures
   jamais préparées pour la carte, toutes versions de georef confondues ; seules
   celles de la version courante ont des keyframes dans le manifeste. Le script
   itère donc `videos[]` et non le listing S3, et nomme les deux écarts :
   - une capture présente dans S3 mais absente de `videos[]` → *ignorée* (avant cette
     passe elle était téléchargée puis remappée à 0 %, puis sautée à l'ingest) ;
   - une capture de `videos[]` sans sorties en prod → `prepare` n'a pas tourné :
     cette portion du lieu n'a **aucun** candidat, ce qui ressemble en tout point à
     « le modèle n'a rien trouvé ». La ligne `Coverage: n/m` la résume.

   La version vient de `map.georef_version` ; le `{version}` du nom de fichier n'est
   plus qu'un repli, et un désaccord entre les deux est signalé (un renommage ferait
   pointer les chemins S3 sur une autre version). Même règle que
   `map_manifest._map_version`.

   Avec un `map_id` numérique, ni la version ni `videos[]` ne sont connus : le script
   avertit, télécharge tout et ne peut pas récupérer `trajectory.json`.

   Le script **refuse** de dumper dans un dossier dont la racine contient déjà un
   `metadata.parquet` (la sortie de l'étape 3) : `discover_capture_dirs` s'arrête à ce
   fichier, donc les captures téléchargées à côté seraient invisibles pour le remap et
   l'ingest. Déplacer le dossier fusionné d'abord, l'étape 3 le reconstruira.

   Les sorties déjà sur disque sont **conservées** (`--force` pour les retélécharger) :
   un `metadata.parquet` neuf annule silencieusement l'étape 2. `--dry-run` affiche
   les transferts sans rien écrire.

   `--thumbnails` synchronise en plus les vignettes de cutouts (~250 Mo par capture).
   Elles sont écrites sous `<map_dir>/<thumbnail_key>`, donc servies sans réécriture
   du parquet. Sans elles, ingest et localize fonctionnent — seule la prévisualisation
   manque. `--thumbnails-only` ne récupère que les vignettes.

2. **Remapper les ids de keyframes — obligatoire, et l'oublier est silencieux :**
   ```bash
   python -m toolbox.bricks.prod_dump_remap <map_path>
   ```
   La prod écrit le vrai `VideoKeyframe.id` dans `metadata.parquet`, alors que ce
   dépôt utilise l'**index dans `geo_keyframes`**. Les deux plages se recouvrent :
   sans remap, environ un quart des candidats se rattachent à des keyframes *sans
   rapport* (positions fausses, aucune erreur nulle part) et le reste est ignoré.

   Le mapping vient de `versions/<n>/360-viewer/trajectory.json` (bucket public),
   seul artefact portant à la fois les ids prod et les positions — la jointure se
   fait sur les coordonnées WGS84. Relancer le remap est sans effet (marqueur dans
   les métadonnées du parquet). Les répertoires de capture hors version sont
   maintenant **nommés au début** du run (`report_out_of_version_captures`) au lieu
   d'apparaître comme un remap à 0 % qu'on peut confondre avec un échec de jointure.

3. **Fusionner en un seul parquet** (l'Explorer indexe par `row_index` et ne lit pas
   la disposition par capture) :
   ```bash
   python -m toolbox.bricks.prod_dump_merge <map_path>
   ```
   Écrit `object-search/{metadata.parquet,embeddings.npy,manifest.json}` à côté des
   répertoires de capture, qui sont conservés. Aucun amincissement : l'ingest amincit
   de son côté, donc l'Explorer voit aussi les propositions que l'index ne garde pas.
   Refuse de tourner si l'étape 2 n'a pas été faite.

4. **Ingérer** (disposition par défaut, aucun drapeau) :
   ```bash
   python -m toolbox.bricks.ingest_cli <map_path>
   ```

Prérequis : creds AWS `researcher` actives (`aws sts get-caller-identity`).

Contenu du dump : propositions brutes YOLO + GroundingDINO (NMS déjà appliqué),
embeddings MetaCLIP2. Le parquet prod est **déjà post-traité** (colonnes
`thumbnail_key` + `depth`), donc `prepare_postprocess` est inutile.
Pas de `object_position` (calculé à l'ingest, pas au prepare).

Les images et les profondeurs ne viennent pas de là : `retrieve-map-data`
(`retrieve_map_data.py <map_dir>`) les télécharge depuis les `image_url` / `depth_url`
du manifeste, qui ne listent que les keyframes de la version — rien à filtrer de ce
côté.
