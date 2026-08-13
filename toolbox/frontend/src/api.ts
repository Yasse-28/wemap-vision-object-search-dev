/**
 * Shared client for the toolbox shell.
 *
 * Only the map list lives here. Each panel owns its own API module
 * (`object-search/api.ts`, `benchmark/api.ts`, …) — a top-level `runObjectSearch`
 * used to sit here too, but it had no callers and duplicated the object-search
 * panel's version, so it was removed with the ADR 0002 cleanup.
 */

import { fetchJson } from "./http";

export type MapSummary = {
  id: string;
  display_name: string;
  path: string;
  emmid: number | null;
  /** Georef id the map was ingested under; the partition key of the live index. */
  geo_ref_id: number;
  /** Whether the map directory and a parseable v2 manifest are present. */
  object_search_available: boolean;
  unavailable_reason: string | null;
};

export type MapsResponse = {
  maps: MapSummary[];
};

export async function fetchMaps(): Promise<MapSummary[]> {
  const data = await fetchJson("/ui/api/maps", {
    headers: { "Content-Type": "application/json" },
  });
  if (!data || typeof data !== "object" || !Array.isArray((data as MapsResponse).maps)) {
    throw new Error("Unexpected maps response.");
  }
  return (data as MapsResponse).maps.map((map) => ({
    ...map,
    geo_ref_id: map.geo_ref_id ?? 1,
    object_search_available: map.object_search_available ?? true,
    unavailable_reason: map.unavailable_reason ?? null,
  }));
}
