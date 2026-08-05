/**
 * Shared client for the toolbox shell.
 *
 * Only the map list lives here. Each panel owns its own API module
 * (`object-search/api.ts`, `benchmark/api.ts`, …) — a top-level `runObjectSearch`
 * used to sit here too, but it had no callers and duplicated the object-search
 * panel's version, so it was removed with the ADR 0002 cleanup.
 */

export type MapSummary = {
  id: string;
  display_name: string;
  path: string;
  emmid: number | null;
  /** Georef id the map was ingested under; the partition key of the live index. */
  geo_ref_id: number;
  /** LEGACY: path to a standalone `object-search.db`, if one exists. */
  object_search_index_path: string | null;
  /** Whether the map directory and its `georef.db` are present. */
  object_search_available: boolean;
  unavailable_reason: string | null;
  /**
   * Whether a legacy `object-search.db` was found. Only the index-explorer panel
   * needs it: that index is no longer produced, so the panel works for
   * pre-ADR-0002 maps only.
   */
  legacy_index_available: boolean;
  /**
   * Whether a legacy `georef.db` is present. The workbench routes that read it
   * directly — keyframe graph, depth pin, view cone, world-point projection — have no
   * v2-manifest reader yet, so they only work for v1 maps.
   */
  georef_db_available: boolean;
};

export type MapsResponse = {
  maps: MapSummary[];
};

export type SearchType = "auto" | "cutout" | "object";

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function fetchJson(path: string, options?: RequestInit): Promise<unknown> {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...options?.headers,
    },
  });
  const data = await readJson(response);
  if (!response.ok) {
    const details =
      data == null
        ? "empty response (is the object-search API running?)"
        : typeof data === "string"
          ? data
          : JSON.stringify(data);
    throw new Error(`HTTP ${response.status}: ${details}`);
  }
  return data;
}

export async function fetchMaps(): Promise<MapSummary[]> {
  const data = await fetchJson("/ui/api/maps");
  if (!data || typeof data !== "object" || !Array.isArray((data as MapsResponse).maps)) {
    throw new Error("Unexpected maps response.");
  }
  return (data as MapsResponse).maps.map((map) => ({
    ...map,
    geo_ref_id: map.geo_ref_id ?? 1,
    object_search_index_path: map.object_search_index_path ?? null,
    object_search_available: map.object_search_available ?? true,
    unavailable_reason: map.unavailable_reason ?? null,
    legacy_index_available: map.legacy_index_available ?? false,
    georef_db_available: map.georef_db_available ?? false,
  }));
}
