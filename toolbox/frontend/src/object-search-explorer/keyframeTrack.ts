/**
 * The capture path: consecutive keyframes joined in manifest order.
 *
 * The manifest lists keyframes in capture order, so joining each to the next draws
 * the route that was actually walked — and unlike the keyframe graph it needs no
 * `360-viewer/graph.geojson`, which most maps do not ship.
 *
 * **This depends on `/object-search-metadata/markers` preserving manifest order**,
 * which it does: `manifestKeyframeIds` maps `manifest.keyframes` straight through
 * and every step downstream is insertion-ordered. Sort that payload by id as a
 * string and `"10"` lands before `"2"`; the track becomes noise, and nothing else
 * in the panel would notice.
 *
 * Two cuts keep it honest, and both matter more than they look:
 *
 * - **Floor change.** A segment carries one level and the livemap filters overlays
 *   by floor, so a segment spanning two floors would be drawn on one of them — a
 *   line across a plan the capture never crossed.
 * - **A gap.** A capture that stops and resumes elsewhere leaves a jump between two
 *   consecutive ids. Joining across it draws a corridor nobody walked, which on a
 *   floor plan is indistinguishable from a real route.
 */

import type { LivemapSegment } from "../annotations/types";
import { metresBetween } from "../geo";
import type { KeyframeMarker } from "../index-explorer/types";

/**
 * Beyond this many metres, two consecutive keyframes are not one walk.
 *
 * Keyframes sit metres apart along a corridor; ingest's own spatial thinning is
 * 1.5 m, so anything at this scale is a break between captures rather than a stride.
 */
export const KEYFRAME_TRACK_MAX_GAP_M = 15;

export function buildKeyframeTrack(
  markers: KeyframeMarker[] | null,
  color: string,
  maxGapM: number = KEYFRAME_TRACK_MAX_GAP_M,
): LivemapSegment[] {
  if (!markers?.length) {
    return [];
  }
  const segments: LivemapSegment[] = [];
  for (let index = 1; index < markers.length; index += 1) {
    const from = markers[index - 1];
    const to = markers[index];
    if (from.level !== to.level) {
      continue;
    }
    if (metresBetween(from, to) > maxGapM) {
      continue;
    }
    segments.push({
      id: `keyframe-track-${from.id}-${to.id}`,
      fromLongitude: from.longitude,
      fromLatitude: from.latitude,
      toLongitude: to.longitude,
      toLatitude: to.latitude,
      color,
      width: 2,
      opacity: 0.7,
      level: from.level,
      // Interactive so the existing hover-reveal picks the closest keyframe: the
      // track is drawn as lines and the point under the cursor is what appears.
      interactive: true,
    });
  }
  return segments;
}
