/**
 * Which keyframes the drawn regions select, resolved in the browser.
 *
 * This is the *live* count — instant, from the marker table already loaded, on the
 * floor the Wemap SDK reports. It is not the number the export writes: the python
 * side re-derives the floor from the manifest's altitude bands (`export_roi.py`),
 * which is the only definition the exported manifest can be consistent with. The
 * save dialog therefore shows both and flags a disagreement rather than hiding it —
 * a systematic gap means the SDK's floors and the manifest's bands disagree, and
 * that is worth knowing.
 */

import { canonicalLevel, pointInPolygon } from "../annotations/geometry";
import type { KeyframeMarker } from "../index-explorer/types";
import type { RoiEntry, RoiSelection } from "./types";

const EMPTY: RoiSelection = {
  keyframeIds: [],
  total: 0,
  perLevel: [],
  countByRoi: {},
};

export function keyframesInRois(
  markers: KeyframeMarker[] | null,
  rois: RoiEntry[],
): RoiSelection {
  if (!markers?.length || rois.length === 0) {
    return EMPTY;
  }

  const byLevel = new Map<string | null, RoiEntry[]>();
  for (const roi of rois) {
    const key = canonicalLevel(roi.level);
    const bucket = byLevel.get(key);
    if (bucket) {
      bucket.push(roi);
    } else {
      byLevel.set(key, [roi]);
    }
  }

  const keyframeIds: string[] = [];
  const perLevelCounts = new Map<string | null, number>();
  const countByRoi: Record<string, number> = {};
  for (const roi of rois) {
    countByRoi[roi.id] = 0;
  }

  for (const marker of markers) {
    const level = canonicalLevel(marker.level);
    const candidates = byLevel.get(level);
    if (!candidates) {
      continue;
    }
    let inside = false;
    for (const roi of candidates) {
      if (pointInPolygon(marker.longitude, marker.latitude, roi.ring)) {
        // Every containing region gets credit, so a row's own count answers "what
        // does this region bring", while the total stays deduplicated.
        countByRoi[roi.id] += 1;
        inside = true;
      }
    }
    if (!inside) {
      continue;
    }
    keyframeIds.push(marker.id);
    perLevelCounts.set(level, (perLevelCounts.get(level) ?? 0) + 1);
  }

  const perLevel = [...perLevelCounts.entries()]
    .map(([level, count]) => ({ level, count }))
    .sort((a, b) => Number(a.level ?? Infinity) - Number(b.level ?? Infinity));

  return { keyframeIds, total: keyframeIds.length, perLevel, countByRoi };
}

/** The wire shape the export routes expect: `[lng, lat]` pairs and a numeric level. */
export function roisForRequest(
  rois: RoiEntry[],
): Array<{ ring: Array<[number, number]>; level: number | null }> {
  return rois.map((roi) => {
    const canonical = canonicalLevel(roi.level);
    const asNumber = canonical === null ? Number.NaN : Number(canonical);
    return {
      ring: roi.ring.map(
        (point) => [point.longitude, point.latitude] as [number, number],
      ),
      level: Number.isNaN(asNumber) ? null : asNumber,
    };
  });
}
