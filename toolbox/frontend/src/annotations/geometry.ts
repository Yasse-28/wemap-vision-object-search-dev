/**
 * Shared map-space geometry: which floor a value names, and what a region contains.
 *
 * Both lived inside `ExplorerAnnotationWorkspace.tsx` while the ROI tool only
 * counted annotations. The export selects keyframes with the same region and the
 * same floor matching, and two copies of a ray cast that must agree with the
 * backend's is exactly the kind of drift this repo's gotchas are made of.
 */

import type { RoiPolygon } from "./types";

/**
 * Normalize a level value to the same canonical form the livemap uses for its
 * indoor-level feature filter (numeric levels collapse to their number string),
 * so annotation levels compare equal to the map's reported current level.
 */
export function canonicalLevel(
  value: string | null | undefined,
): string | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const asNumber = Number(value);
  return Number.isNaN(asNumber) ? String(value) : String(asNumber);
}

/**
 * Ray-casting point-in-polygon test against an open ring of vertices (the first
 * vertex is treated as reconnected to the last).
 */
export function pointInPolygon(
  longitude: number,
  latitude: number,
  ring: RoiPolygon,
): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].longitude;
    const yi = ring[i].latitude;
    const xj = ring[j].longitude;
    const yj = ring[j].latitude;
    const intersects =
      yi > latitude !== yj > latitude &&
      longitude < ((xj - xi) * (latitude - yi)) / (yj - yi) + xi;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}
