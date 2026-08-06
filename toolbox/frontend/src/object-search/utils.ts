import type { EnrichedResult, KeyframeGroup, ObjectLocalization } from "./types";

export function markerColorForScore(score: number, isSelected: boolean): string {
  if (isSelected) {
    return "#16a34a";
  }
  if (score >= 0.75) {
    return "#22c55e";
  }
  if (score >= 0.5) {
    return "#eab308";
  }
  return "#9ca3af";
}

export function localizationMatchThreshold(sensitivity: number): number {
  return Math.max(0, Math.min(1, 1 - sensitivity / 100));
}

export function localizationColorForMatch(
  loc: ObjectLocalization,
  threshold: number,
): string {
  return loc.matchScore >= threshold ? "#16a34a" : "#9ca3af";
}

export function localizationScoreSymbol(
  loc: ObjectLocalization,
  threshold: number,
): string {
  return loc.matchScore >= threshold ? "●" : "○";
}

export function buildKeyframeGroups(results: EnrichedResult[]): Record<string, KeyframeGroup> {
  const groups: Record<string, KeyframeGroup> = {};
  for (const result of results) {
    if (!result.keyframe_id) {
      continue;
    }
    const group =
      groups[result.keyframe_id] ??
      ({
        keyframeId: result.keyframe_id,
        lat: result.lat,
        lon: result.lon,
        alt: result.alt,
        level: result.level,
        bestScore: result.score,
        results: [],
      } satisfies KeyframeGroup);
    group.results.push(result);
    group.bestScore = Math.max(group.bestScore, result.score);
    if (group.lat === null && result.lat !== null) {
      group.lat = result.lat;
    }
    if (group.lon === null && result.lon !== null) {
      group.lon = result.lon;
    }
    if (group.alt === null && result.alt !== null) {
      group.alt = result.alt;
    }
    if (group.level === null && result.level !== null) {
      group.level = result.level;
    }
    groups[result.keyframe_id] = group;
  }
  for (const group of Object.values(groups)) {
    group.results.sort((a, b) => b.score - a.score || a.rank - b.rank);
  }
  return Object.fromEntries(
    Object.entries(groups).sort(
      (a, b) => b[1].bestScore - a[1].bestScore || a[0].localeCompare(b[0]),
    ),
  );
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

/**
 * URL for one search result's preview image.
 *
 * There is one source now: `/preview.png?preview_path=…` serves a file from the map
 * directory as-is, and `thumbnail_key` — which `enrichedFromCandidates` puts in
 * `preview_path` — points at the JPEG `prepare` already wrote. That thumbnail *is*
 * the cutout, and it is the only image certain to be what MetaCLIP2 embedded.
 *
 * The `/cutouts/{id}/preview.png` route this used to fall back to rendered from the
 * retired SQLite index and drew detection boxes; it is gone, and with it the `index:`
 * preview-path scheme. A result with no `thumbnail_key` simply has no preview:
 * pgvector does not carry `row_index`, so there is no way back to the parquet row a
 * re-render would need.
 */
export function cutoutPreviewUrl(mapId: string, previewPath: string | null): string {
  if (!previewPath) {
    return "";
  }
  const params = new URLSearchParams({ preview_path: previewPath });
  return `/ui/api/maps/${encodeURIComponent(mapId)}/preview.png?${params.toString()}`;
}
