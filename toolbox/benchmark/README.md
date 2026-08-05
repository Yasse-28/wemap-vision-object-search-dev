# Object Search HTTP Benchmark

Compares localized object-search clusters returned by the online service against
annotator ground truth in GeoJSON:

`{map_path}/benchmark/annotations.geojson`

Each annotation's query prompt is taken from `feature.properties.prompt` when
present, otherwise from `feature.properties.class`. Annotations are grouped by
their resolved prompt, and one query is issued per prompt.

## Run

Start the local standalone object-search service, then run from the repo root:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id>
```

By default it posts to:

`http://localhost:45678/<map-id>/object-search/localize`

GeoPose service example:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --api-style geopose \
  --base-url http://localhost:8000
```

Standalone online localization example:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --online \
  --candidate-count 1000 \
  --clustering-eps-m 1.5
```

Remote localization example:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --map-id <map-id> \
  --localize-url 'https://vps-api.wemap-vision-computing-1.getwemap.com/{map_id}/object-search/localize'
```

The toolbox benchmark runner spawns this script automatically; the invocations
above are for running it standalone.

## Scoring

The script accepts clusters whose selected score is strictly greater than `0.9`
by default:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --acceptance-threshold 0.9
```

The default score field is `match_score`. If that field is missing, the script
falls back to `similarity_score`, then `confidence`.

Each accepted cluster is a true positive when it can be greedily matched to one
unmatched annotation of the same class within the annotation `accuracy` radius in
meters. Unmatched accepted clusters are false positives. Unmatched annotations
are false negatives.

Passing `--group-annotation-radius-m > 0` adds a second "grouped" result block
where nearby annotations of the same class within that radius count as a single
ground-truth target.

## Outputs

By default (or under `--output-dir`):

- `results.json` / `metrics.json`: config, summary, per-class rows, per-prompt rows, matches
- `results.csv`: per-prompt metrics (plus `grouped_results.csv` when grouping is enabled)
- `raw_results.json`: raw predictions/annotations per prompt (only with `--output-dir`)
- `prompt_geojson/*.geojson`: one GeoJSON per evaluated prompt

Per-prompt GeoJSON colors:

- TP prediction clusters: green (`#22c55e`)
- FP prediction clusters: red (`#ef4444`)
- FN reference annotations: grey (`#9ca3af`)
- Matched reference annotations: blue (`#2563eb`)
