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
 * URL for a cutout preview image.
 *
 * Two backend routes serve previews, and which one applies depends on where the
 * image comes from:
 *
 * - `/cutouts/{id}/preview.png` renders the cutout **from the legacy SQLite index**
 *   and draws detection boxes on it. Only available for pre-ADR-0002 maps.
 * - `/preview.png?preview_path=…` serves a file from the map directory as-is. This
 *   is what current maps use: `thumbnail_key` points at the JPEG the `prepare` job
 *   already wrote, and that thumbnail *is* the cutout crop, so there is nothing to
 *   draw boxes on.
 *
 * A `preview_path` prefixed `index:` is an index reference, not a file, so it goes
 * to the index-backed route along with the box-drawing options.
 */
export function cutoutPreviewUrl(
  mapId: string,
  cutoutId: string,
  previewPath: string | null,
  selectedObjectId: string | null,
  selectedOnly = false,
  selectedBbox: [number, number, number, number] | null = null,
): string {
  const base = `/ui/api/maps/${encodeURIComponent(mapId)}`;
  const isPlainFile = previewPath != null && previewPath !== "" && !previewPath.startsWith("index:");

  if (isPlainFile) {
    const params = new URLSearchParams({ preview_path: previewPath });
    return `${base}/preview.png?${params.toString()}`;
  }

  const params = new URLSearchParams();
  if (previewPath) {
    params.set("preview_path", previewPath);
  }
  if (selectedObjectId) {
    params.set("selected_object_id", selectedObjectId);
  }
  if (selectedOnly) {
    params.set("selected_only", "true");
  }
  if (selectedBbox) {
    params.set("selected_bbox", selectedBbox.join(","));
  }
  const query = params.toString();
  return `${base}/cutouts/${encodeURIComponent(cutoutId)}/preview.png${query ? `?${query}` : ""}`;
}
