# Plan — review-feedback boost on query↔cutout similarity

Status: **not started**. Written 2026-08-06 for a later implementation session.

## Goal

Object-search recall is acceptable; precision is not. Use the `detection_review`
annotations (`annotation_service` SQLite, one DB per map) to demote candidates a
user has marked `false_positive` and promote those near `true_positive` ones.

The boost applies at the **query↔cutout similarity** step — that is the level the
annotation lives at (a cutout image with an embedding). It is off by default, so
the baseline is literally the current code path.

Scope of this plan: the feature only. Collecting more annotations (porting the
livemap-tools review UI into `toolbox/frontend`) and the benchmark campaign are
separate work.

## Constraints

- All changes in **this repo** (`wemap-vision-object-search-dev`). Nothing under
  `third_party/object_search/` — that tree is a byte-for-byte mirror.
- `toolbox/bricks/localize.py` is a port of the backend's Django code. Keep the
  ported functions recognizable; new logic goes in a separate module so a later
  re-port stays mechanical.
- **Do not commit.** Leave the work on a branch, unstaged or staged but
  uncommitted, for review.

## Prerequisites to verify first

1. **Embeddings are L2-normalised in Postgres.** The retrieval similarity is
   `sim = 1 - d²/2` over `<->` (see `services/object_search_online/app.py`
   `_hnsw_query`), which equals cosine only for unit vectors. The boost term must
   use the *same* formula or α/β have no physical meaning. If this does not hold,
   **stop and report** rather than guessing a scale.
2. `slug` ↔ `MapEntry.id` — **confirmed by the user, no need to re-verify.**

## Steps

### 1. `toolbox/bricks/feedback.py` (new)

No import from `localize` or `candidates`, so it stays testable alone.

- `ReviewFeedback(positive_ids: list[int], negative_ids: list[int])`
- `load_review_feedback(slug: str, query: str) -> ReviewFeedback | None`
  - opens `$ANNOTATION_DATA_DIR/<slug>/object-search-annotations.db` read-only
    (`sqlite3.connect("file:…?mode=ro", uri=True)`)
  - `SELECT target_id, status FROM detection_review WHERE target_type='object'`
  - **matches the query normalised** (casefold + strip) on both sides, decided
    with the user — the stored `query` is raw user input, so `FIDS` and `fids`
    must share their annotations. Read-side only; the stored rows are untouched.
  - returns `None` when the DB is missing or nothing matches, so the caller needs
    no special case.

### 2. `toolbox/bricks/candidates.py` — the computation

- `_ENRICH_SQL` gains two optional columns, correlated subqueries over the table
  itself, restricted to the same `geo_ref_id`:

  ```sql
  (SELECT MAX(1 - POWER(c.embedding <-> p.embedding, 2)/2)
     FROM object_search_candidate p
    WHERE p.id = ANY(%s) AND p.geo_ref_id = %s) AS pos_sim
  ```

  and the same for `neg_sim`. Cost is negligible (~1000 rows × a few prototypes).
- `EnrichedCandidate` gains `similarity_boosted`, plus `pos_sim` / `neg_sim` kept
  for observability.
- `load_enriched_candidates(..., feedback=None, alpha=0.0, beta=0.0)`. With no
  feedback the emitted SQL and the behaviour are **exactly** today's.
- `similarity_boosted = similarity + alpha·pos_sim − beta·neg_sim`, clipped to
  [-1, 1]; `pos_sim`/`neg_sim` default to 0 when the set is empty.
- **The sort stays on raw `similarity`.** The prefilter and the fetched set do
  not move; only the new field is added.

### 3. `toolbox/bricks/localize.py` — the routing (the delicate step)

`EnrichedCandidate.similarity` currently feeds four consumers. Route them
explicitly:

| Consumer | Field |
|---|---|
| sort in `load_enriched_candidates` | raw |
| `select_top_candidates` truncation | raw |
| leader-canopy **seed order** | raw — boosting this silently changes cluster geometry |
| `cluster_best_sim` → `rank_localization_clusters` | **boosted** |

- `LocalizationParams` gains `feedback_alpha: float = 0.0`,
  `feedback_beta: float = 0.0`. Zero ⇒ `similarity_boosted == similarity`, i.e.
  exact disablement, not approximate.

Structural ceiling worth knowing while tuning:
`match_score = 0.50·normalized_similarity + 0.15·confidence + 0.35·keyframe_score`.
Similarity is only half the score, so a penalty must push a cluster under
`min_similarity` (0.2) or break `min_keyframes_per_cluster` to actually remove a
false positive — otherwise it only reorders.

### 4. `toolbox/bricks/service.py` — the wiring

- `LocalizeRequest` gains the same two fields (default 0.0).
- In the `localize` handler, load the feedback from `map_id` + `body.text`,
  **on the text branch only**. The image branch has no usable query string —
  image searches store the *filename* in `detection_review.query`, which also
  collides across different uploads. Skip explicitly, with a comment; this is a
  known limitation, not an oversight.

### 5. Observability

- Expose `pos_sim`, `neg_sim` and the applied delta on each observation in the
  localize response.
- Log the number of positives/negatives **actually resolved** vs requested. After
  a reingest the `target_id`s match nothing (they are BIGSERIAL, wiped and
  re-inserted every run) and the boost goes silently neutral — warn when the
  counts differ.

### 6. Tests (`toolbox/tests/`)

- `feedback.py` loader: missing DB, unknown query, mixed TP/FP, case-insensitive
  match.
- The boost formula on hand-built vectors.
- **Regression: `alpha = beta = 0` reproduces the current output exactly** on a
  fixture. This is the test that protects the baseline — write it before step 3.

Order: 1 → 2 → 6 → 3 → 4 → 5.

## Edge cases

- Exact-match on `query` is fixed by the read-side normalisation above; the
  remaining key weakness (`target_id` not surviving reingest) is out of scope —
  freeze the index while using this.
- Restrict prototype ids to the request's `geo_ref_id`, or one map can borrow
  another's annotations.
- The positive set is noisy: a single cluster-level "correct" click in the review
  UI fans out to one row per cutout, so many positives were never individually
  judged. That is why the term uses `MAX` over the prototype set, not a mean.
