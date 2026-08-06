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
  /** Whether the map directory and a parseable v2 manifest are present. */
  object_search_available: boolean;
  unavailable_reason: string | null;
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
    object_search_available: map.object_search_available ?? true,
    unavailable_reason: map.unavailable_reason ?? null,
  }));
}
