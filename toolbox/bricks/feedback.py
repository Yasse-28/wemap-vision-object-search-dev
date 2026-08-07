"""Review annotations, read back as similarity-boost prototypes.

The annotation service (`third_party/object_search/annotation_service/`) records
one row per `(target_type, target_id, query)` in a per-map SQLite file:

    $ANNOTATION_DATA_DIR/<slug>/object-search-annotations.db

`detection_review.target_id` is an `object_search_candidate.id`, and `status` is
`true_positive` or `false_positive`. This module turns those into two id sets, so
`candidates` can demote what a user rejected and promote what they endorsed.

**It imports nothing from `candidates` or `localize`,** so it stays testable on its
own and a later re-port of the Django layer does not have to look at it.

## Two things that will bite

1. **The stored `query` is raw user input.** `FIDS` and `fids` are the same search
   as far as anyone using the tool is concerned, so the match is done on a
   casefold+strip normalisation of *both* sides. This is read-side only — the rows
   on disk are never rewritten.
2. **`target_id` does not survive a reingest.** `object_search_candidate.id` is a
   BIGSERIAL, wiped and re-inserted by every `ingest_cli` run, so annotations
   collected before a rebuild silently point at unrelated candidates — or at
   nothing. Nothing here can detect that; the caller is expected to log resolved
   vs. requested counts and the index is expected to stay frozen while this is in
   use. See the plan's "Edge cases".
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from toolbox.logging import logger

DB_FILENAME = "object-search-annotations.db"
DEFAULT_DATA_DIR = "/data"  # matches annotation_service.db.data_dir()

_SELECT_REVIEWS = """
SELECT target_id, status, query
FROM detection_review
WHERE target_type = 'object'
"""


@dataclass(frozen=True)
class ReviewFeedback:
    """Candidate ids a user judged, for one map and one normalised query."""

    positive_ids: list[int]
    negative_ids: list[int]

    def __bool__(self) -> bool:
        return bool(self.positive_ids or self.negative_ids)


def normalize_query(query: str) -> str:
    """The form both sides of the match are compared in: casefold + strip.

    `casefold` rather than `lower` so non-ASCII queries fold the way users expect
    (German "ß" vs "ss", for one). Whitespace inside the string is left alone —
    "fire exit" and "fire  exit" are genuinely different searches.
    """
    return str(query).strip().casefold()


def annotation_db_path(slug: str) -> Path:
    """Where the annotation service keeps this map's DB.

    Reads `ANNOTATION_DATA_DIR` at call time, not at import, so a test can set it
    per-case. The `slug` is the map id — confirmed equal to `MapEntry.id`.
    """
    data_dir = Path(os.environ.get("ANNOTATION_DATA_DIR", DEFAULT_DATA_DIR))
    return data_dir / slug / DB_FILENAME


def load_review_feedback(slug: str, query: str) -> ReviewFeedback | None:
    """Reviewed candidate ids for `slug` matching `query`, or None.

    Returns `None` — not an empty `ReviewFeedback` — when there is no DB, no
    `detection_review` table, or nothing matches, so the caller has one branch to
    write instead of three. Any SQLite error is logged and swallowed: a missing or
    corrupt annotation file must never take down a search.
    """
    path = annotation_db_path(slug)
    if not path.is_file():
        logger.debug("No annotation DB for map '%s' at %s.", slug, path)
        return None

    wanted = normalize_query(query)
    if not wanted:
        return None

    positive_ids: list[int] = []
    negative_ids: list[int] = []
    try:
        # Read-only URI: the annotation service owns this file and may be writing
        # to it. Opening it read-write would also create it if it vanished, which
        # would turn "no annotations" into "an empty DB appeared".
        uri = f"file:{path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            for target_id, status, stored_query in conn.execute(_SELECT_REVIEWS):
                if normalize_query(stored_query or "") != wanted:
                    continue
                if status == "true_positive":
                    positive_ids.append(int(target_id))
                elif status == "false_positive":
                    negative_ids.append(int(target_id))
    except sqlite3.Error as exc:
        logger.warning("Could not read annotations for map '%s' (%s).", slug, exc)
        return None

    if not positive_ids and not negative_ids:
        return None

    # Deterministic order so the SQL parameter list — and therefore the query plan
    # and any test fixture — does not depend on SQLite's row order.
    feedback = ReviewFeedback(
        positive_ids=sorted(set(positive_ids)),
        negative_ids=sorted(set(negative_ids)),
    )
    logger.info(
        "Review feedback for map '%s', query '%s': %d positive, %d negative.",
        slug,
        wanted,
        len(feedback.positive_ids),
        len(feedback.negative_ids),
    )
    return feedback
