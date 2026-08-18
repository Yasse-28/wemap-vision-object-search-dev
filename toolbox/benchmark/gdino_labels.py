"""Recover the missing G-DINO labels by an argmax over the venue's own prompt.

A prepare run can leave every GroundingDINO row carrying one placeholder label — it
happened on vinci, where 813 467 of 1 063 142 rows say `gdino_venue` and nothing else.
The information is not lost, only unwritten: the venue prompt lists the exact phrases
the detector was asked for, the cutouts already carry MetaCLIP image embeddings, and
MetaCLIP text embeddings live in the same space. Encoding each phrase once and taking
the nearest one per cutout puts a name back on every box.

**This is an estimate, not a recovery.** The stored labels of a healthy map come from
GroundingDINO itself; these come from MetaCLIP's opinion of the same crop. The two
agree only as far as the two models agree, which is why `--validate` exists: run it on
a map whose labels *are* real and read the agreement before trusting the output
anywhere else.

Nothing is overwritten. The result is a sidecar next to the parquet, so the prepare
output stays exactly what prepare produced.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from prepare.prompts import gdino_classes

from toolbox.bricks import map_manifest
from toolbox.bricks.ingest_cli import EMBEDDING_DIM
from toolbox.logging import logger

#: Sidecar written next to `metadata.parquet`, never into it.
SIDECAR_NAME = "gdino_labels.parquet"
#: The placeholder a prepare run leaves when it writes no per-box class.
PLACEHOLDER_LABEL = "gdino_venue"
#: CLIP text encoders were trained on captions, not on bare nouns; the template is
#: reported alongside the agreement so the choice can be checked rather than assumed.
DEFAULT_TEMPLATE = "a photo of a {}"


@dataclass(frozen=True)
class ArgmaxLabels:
    """One label per selected row, with what the decision rested on."""

    row_index: np.ndarray
    label: np.ndarray
    score: np.ndarray
    #: Top-1 minus top-2 cosine. Near zero the argmax is a coin toss between two
    #: phrases, which is a property of the vocabulary rather than of the crop.
    margin: np.ndarray


def venue_classes(map_path: Path, venue: str | None = None) -> tuple[str, list[str]]:
    """The venue this map was prepared for, and the phrases G-DINO was given."""
    resolved = venue
    if resolved is None:
        manifest = map_manifest.load_map_manifest(map_path)
        resolved = getattr(manifest, "venue_type", None) or getattr(
            manifest, "venue", None
        )
    classes = gdino_classes(resolved)
    if not classes:
        raise SystemExit(
            f"{map_path.name}: venue {resolved!r} has no GroundingDINO vocabulary; "
            "pass --venue to force one."
        )
    return str(resolved), list(classes)


def encode_classes(
    classes: Sequence[str], *, device: str, template: str, model_id: str | None = None
) -> np.ndarray:
    """Embed every phrase once with the same text encoder the search box uses.

    Loaded on the CPU by default: the ANN service already holds a copy of this model
    on the GPU, and a few dozen phrases do not justify competing with it for memory.
    """
    from inference.embedder import METACLIP_MODEL_ID, MetaClipEmbedder

    embedder = MetaClipEmbedder(model_id or METACLIP_MODEL_ID, device)
    vectors = np.stack(
        [embedder.embed_text(template.format(name)) for name in classes]
    ).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
    return vectors


def _load_embeddings(map_path: Path, rows: int) -> np.ndarray:
    """Memory-map the raw float16 cutout embeddings, checking the length first."""
    path = map_path / "object-search" / "embeddings.npy"
    stored = path.stat().st_size // (2 * EMBEDDING_DIM)
    if stored != rows:
        raise SystemExit(
            f"{path}: {stored} embeddings for {rows} parquet rows — refusing to guess."
        )
    return np.memmap(path, dtype=np.float16, mode="r", shape=(rows, EMBEDDING_DIM))


def assign_labels(
    map_path: Path,
    classes: Sequence[str],
    class_vectors: np.ndarray,
    *,
    source: str | None = "gdino",
    chunk: int = 20000,
) -> ArgmaxLabels:
    """Nearest phrase for every selected cutout, in chunks over the memory map."""
    table = pq.read_table(
        map_path / "object-search" / "metadata.parquet",
        columns=["row_index", "detector_source"],
    )
    def column(name: str) -> np.ndarray:
        return table.column(name).to_numpy(zero_copy_only=False)

    row_index = column("row_index").astype(np.int64)
    detector = column("detector_source").astype(str)
    selected = (
        np.ones(row_index.size, dtype=bool) if source is None else detector == source
    )
    rows = np.flatnonzero(selected)
    embeddings = _load_embeddings(map_path, row_index.size)
    labels = np.empty(rows.size, dtype=object)
    scores = np.zeros(rows.size, dtype=np.float32)
    margins = np.zeros(rows.size, dtype=np.float32)
    names = np.asarray(classes, dtype=object)
    for start in range(0, rows.size, chunk):
        take = rows[start : start + chunk]
        vectors = np.asarray(embeddings[take], dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
        cosines = vectors @ class_vectors.T
        best = np.argmax(cosines, axis=1)
        top = np.take_along_axis(cosines, best[:, None], axis=1)[:, 0]
        cosines[np.arange(cosines.shape[0]), best] = -2.0
        second = cosines.max(axis=1)
        labels[start : start + take.size] = names[best]
        scores[start : start + take.size] = top
        margins[start : start + take.size] = top - second
    return ArgmaxLabels(row_index[rows], labels, scores, margins)


def write_sidecar(map_path: Path, result: ArgmaxLabels, venue: str) -> Path:
    """Write the labels beside the parquet, leaving the prepare output untouched."""
    path = map_path / "object-search" / SIDECAR_NAME
    table = pa.table(
        {
            "row_index": pa.array(result.row_index, type=pa.int64()),
            "label": pa.array(result.label.tolist(), type=pa.string()),
            "score": pa.array(result.score, type=pa.float32()),
            "margin": pa.array(result.margin, type=pa.float32()),
        },
        metadata={b"venue": venue.encode(), b"source": b"metaclip-argmax"},
    )
    pq.write_table(table, path)
    logger.info("Wrote %d labels to %s", result.row_index.size, path)
    return path


def validate(map_path: Path, result: ArgmaxLabels, top: int = 8) -> list[str]:
    """Compare the argmax against the labels this map already carries.

    Only meaningful where the stored labels are real. A map full of the placeholder
    has nothing to validate against, and this says so instead of reporting 0 %.
    """
    table = pq.read_table(
        map_path / "object-search" / "metadata.parquet", columns=["row_index", "label"]
    )
    stored = dict(
        zip(
            table.column("row_index").to_numpy(zero_copy_only=False).tolist(),
            table.column("label").to_numpy(zero_copy_only=False).astype(str).tolist(),
            strict=True,
        )
    )
    truth = np.asarray([stored.get(int(value), "") for value in result.row_index])
    usable = (truth != "") & (truth != PLACEHOLDER_LABEL)
    if not usable.any():
        return ["  aucun label réel à comparer (la carte ne porte que le placeholder)"]
    agree = result.label[usable] == truth[usable]
    lines = [
        f"  {int(usable.sum())} lignes comparables, accord top-1 : {agree.mean():.1%}",
    ]
    for low, high in ((0.0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 1.0)):
        band = usable & (result.margin >= low) & (result.margin < high)
        if band.sum() < 50:
            continue
        share = (result.label[band] == truth[band]).mean()
        lines.append(
            f"    marge [{low:.2f},{high:.2f}) n={int(band.sum()):7d}  "
            f"accord {share:6.1%}"
        )
    wrong = usable & (result.label != truth)
    if wrong.any():
        pairs = [
            f"{str(truth[i])} -> {str(result.label[i])}"
            for i in np.flatnonzero(wrong)[:2000]
        ]
        counts: dict[str, int] = {}
        for pair in pairs:
            counts[pair] = counts.get(pair, 0) + 1
        lines.append("  confusions les plus fréquentes :")
        for pair, count in sorted(counts.items(), key=lambda item: -item[1])[:top]:
            lines.append(f"    {count:5d}  {pair}")
    return lines


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the label-recovery command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--venue", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--source", default="gdino", help="'all' for every row")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Assign a label to every G-DINO cutout, and say how much to trust it."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    venue, classes = venue_classes(map_path, args.venue)
    print(f"{map_path.name} : venue {venue!r}, {len(classes)} phrases")
    print(f"  gabarit : {args.template!r}, encodeur sur {args.device}")
    vectors = encode_classes(classes, device=args.device, template=args.template)
    result = assign_labels(
        map_path,
        classes,
        vectors,
        source=None if args.source == "all" else args.source,
    )
    print(f"  {result.row_index.size} lignes étiquetées")
    values, counts = np.unique(result.label.astype(str), return_counts=True)
    order = np.argsort(counts)[::-1][:10]
    for index in order:
        print(f"    {str(values[index])[:32]:32s} {int(counts[index]):8d}")
    print(
        f"  cosinus  med {np.median(result.score):.3f}   "
        f"marge top1-top2  med {np.median(result.margin):.4f}"
    )
    if args.validate:
        print("\n  --- validation contre les labels stockés")
        for line in validate(map_path, result):
            print(line)
    if args.write:
        write_sidecar(map_path, result, venue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
