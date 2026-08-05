import { readFile } from "node:fs/promises";
import path from "node:path";

export type MapEntry = {
  id: string;
  display_name: string;
  path: string;
  emmid: number | null;
  /**
   * Georef id this map's candidates were ingested under, i.e. the
   * `--geo-ref-id` passed to `toolbox.bricks.ingest_cli`. It is the partition key
   * of `object_search_candidate` and of the partial HNSW index, so a mismatch
   * returns zero hits with no error. Defaults to 1 on both sides.
   */
  geo_ref_id: number;
  /**
   * LEGACY: path to a standalone `object-search.db`. That SQLite index was
   * retired with the standalone lineage (ADR 0002) — the index now lives in
   * pgvector. Only the `/object-search-index/*` inspection routes still read it,
   * and only for maps built before the migration.
   */
  object_search_index_path: string | null;
  /** Raw per-map `objectSearch` settings, passed through to the python service. */
  object_search: Record<string, unknown> | null;
};

type RawConfig = {
  maps?: unknown;
  objectSearch?: unknown;
};

function stripJsonComments(input: string): string {
  let output = "";
  let inString = false;
  let quote = "";
  let escaped = false;

  for (let idx = 0; idx < input.length; idx += 1) {
    const current = input[idx];
    const next = input[idx + 1];

    if (inString) {
      output += current;
      if (escaped) {
        escaped = false;
      } else if (current === "\\") {
        escaped = true;
      } else if (current === quote) {
        inString = false;
        quote = "";
      }
      continue;
    }

    if (current === "\"" || current === "'") {
      inString = true;
      quote = current;
      output += current;
      continue;
    }

    if (current === "/" && next === "/") {
      while (idx < input.length && input[idx] !== "\n") {
        idx += 1;
      }
      output += "\n";
      continue;
    }

    if (current === "/" && next === "*") {
      idx += 2;
      while (idx < input.length && !(input[idx] === "*" && input[idx + 1] === "/")) {
        idx += 1;
      }
      idx += 1;
      continue;
    }

    output += current;
  }

  return output;
}

function stripTrailingCommas(input: string): string {
  return input.replace(/,\s*([}\]])/g, "$1");
}

function parseConfigText(text: string): RawConfig {
  return JSON.parse(stripTrailingCommas(stripJsonComments(text))) as RawConfig;
}

function resolveMapPath(configDir: string, mapId: string, value: unknown): string {
  if (typeof value === "string" && value.length > 0) {
    return path.resolve(configDir, value);
  }
  return path.resolve(configDir, "maps", mapId);
}

export async function loadMapEntries(configPath: string): Promise<MapEntry[]> {
  const resolvedConfigPath = path.resolve(configPath);
  const configDir = path.dirname(resolvedConfigPath);
  const data = parseConfigText(await readFile(resolvedConfigPath, "utf8"));
  if (!Array.isArray(data.maps)) {
    throw new Error("Config must contain a maps array.");
  }

  const seen = new Set<string>();
  return data.maps.map((item, index) => {
    if (!item || typeof item !== "object") {
      throw new Error(`maps[${index}] must be an object.`);
    }
    const raw = item as Record<string, unknown>;
    if (typeof raw.id !== "string" || raw.id.length === 0) {
      throw new Error(`maps[${index}] must contain a string id.`);
    }
    if (seen.has(raw.id)) {
      throw new Error(`Duplicate map id: ${raw.id}`);
    }
    seen.add(raw.id);

    const emmid =
      typeof raw.emmid === "number" && Number.isInteger(raw.emmid) ? raw.emmid : null;
    const geoRefId =
      typeof raw.geo_ref_id === "number" && Number.isInteger(raw.geo_ref_id)
        ? raw.geo_ref_id
        : 1;
    const objectSearchIndexPath =
      typeof raw.object_search_index_path === "string"
        ? raw.object_search_index_path
        : null;
    const objectSearch =
      raw.objectSearch && typeof raw.objectSearch === "object" && !Array.isArray(raw.objectSearch)
        ? (raw.objectSearch as Record<string, unknown>)
        : null;

    return {
      id: raw.id,
      display_name: raw.id,
      path: resolveMapPath(configDir, raw.id, raw.path),
      emmid,
      geo_ref_id: geoRefId,
      object_search_index_path: objectSearchIndexPath,
      object_search: objectSearch,
    };
  });
}

/** Top-level `objectSearch` defaults from the config, if present. */
export async function loadGlobalObjectSearch(
  configPath: string,
): Promise<Record<string, unknown> | null> {
  const data = parseConfigText(await readFile(path.resolve(configPath), "utf8"));
  if (data.objectSearch && typeof data.objectSearch === "object" && !Array.isArray(data.objectSearch)) {
    return data.objectSearch as Record<string, unknown>;
  }
  return null;
}
