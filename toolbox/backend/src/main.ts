import { readdir, stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";

import { loadMapEntries, type MapEntry } from "./config.js";
import {
  parseArgs,
  proxyRequest,
  repoRootFromModule,
  sendJson,
  sendText,
  serveStaticUi,
} from "./http-utils.js";
import {
  handleWorkbenchUiMapRoute,
  isWorkbenchUiMapRoute,
} from "./workbench-api.js";

const repoRoot = repoRootFromModule(import.meta.url);
const options = parseArgs(process.argv.slice(2), repoRoot);

/**
 * Routes proxied to the bricks service.
 *
 * The standalone service also served `/object-search/encode-text` and
 * `/encode-image`; neither exists in the mirrored pipeline (ADR 0002), so they are
 * gone. The `{map_id}`-prefixed shape survives because the bricks service keeps it.
 */
function isObjectSearchRoute(pathname: string): boolean {
  return /^\/[^/]+\/object-search(?:\/|$)/.test(pathname);
}

async function isFile(target: string): Promise<boolean> {
  try {
    return (await stat(target)).isFile();
  } catch {
    return false;
  }
}

async function isDirectory(target: string): Promise<boolean> {
  try {
    return (await stat(target)).isDirectory();
  } catch {
    return false;
  }
}

/** `{map_id}_{version}_{date}_{time}.json` — the v2 pose source. */
const V2_MANIFEST_PATTERN = /^.+_\d+_\d{8}_\d{6}\.json$/;

/** Whether the map directory holds a v2 manifest. */
async function hasV2Manifest(mapPath: string): Promise<boolean> {
  try {
    return (await readdir(mapPath)).some((name) => V2_MANIFEST_PATTERN.test(name));
  } catch {
    return false;
  }
}

/**
 * Map entries enriched with object-search availability for the UI map list.
 *
 * Availability used to mean "has an object-search.db". That SQLite index is gone
 * (ADR 0002) — the index lives in pgvector now, which we cannot cheaply probe from
 * here. So the check is what this process *can* see on disk: the map directory, and a
 * pose source, without which nothing downstream resolves a position.
 *
 * A pose source is a **v2 manifest** or a legacy **`georef.db`**. Checking only the
 * latter would mark every current map unavailable.
 *
 * A map that passes this check but was never ingested surfaces as an empty result
 * list rather than an error. That is the one honest gap in this signal, and it is
 * why the legacy index flag below is reported separately instead of being folded in.
 */
async function mapSummaries(configPath: string): Promise<Record<string, unknown>[]> {
  const maps = await loadMapEntries(configPath);
  return Promise.all(
    maps.map(async (map: MapEntry) => {
      let reason: string | null = null;
      const hasGeorefDb = await isFile(path.join(map.path, "georef.db"));
      if (!(await isDirectory(map.path))) {
        reason = `Map directory not found: ${map.path}`;
      } else if (!hasGeorefDb && !(await hasV2Manifest(map.path))) {
        reason =
          `No pose source in ${map.path} — expected a v2 manifest `
          + "('{map_id}_{version}_{date}_{time}.json') or a legacy georef.db. "
          + "Keyframe poses come from one of them.";
      }
      const legacyIndexPath =
        map.object_search_index_path ?? path.join(map.path, "object-search.db");
      return {
        ...map,
        object_search_available: reason == null,
        unavailable_reason: reason,
        // Drives the index-explorer panel, which still reads the retired SQLite
        // index and is therefore only usable for pre-migration maps.
        legacy_index_available: await isFile(legacyIndexPath),
        // The workbench routes that read georef.db directly (keyframe graph, depth
        // pin, view cone, world-point projection) have no v2 reader yet, so they are
        // v1-only for now. Reported so the UI can say why rather than just fail.
        georef_db_available: hasGeorefDb,
      };
    }),
  );
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
  try {
    if (request.method === "GET" && url.pathname === "/health") {
      const maps = await loadMapEntries(options.configPath);
      sendJson(response, 200, {
        status: "ok",
        service: "object-search-toolbox-backend",
        maps: maps.map((map) => map.id),
        python_api: options.pythonApiBaseUrl,
        ann_api: options.annApiBaseUrl,
        annotation_api: options.annotationBaseUrl,
      });
      return;
    }

    if (
      request.method === "GET"
      && (url.pathname === "/ui/api/maps" || url.pathname === "/workbench/api/maps")
    ) {
      sendJson(response, 200, { maps: await mapSummaries(options.configPath) });
      return;
    }

    if (isWorkbenchUiMapRoute(url.pathname)) {
      await handleWorkbenchUiMapRoute(request, response, url, options);
      return;
    }

    if (isObjectSearchRoute(url.pathname)) {
      await proxyRequest(request, response, options.pythonApiBaseUrl, url);
      return;
    }

    if (await serveStaticUi(response, options.uiDistDir, url)) {
      return;
    }

    sendText(response, 404, `No workbench route for ${request.method ?? "GET"} ${url.pathname}`);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    sendJson(response, 500, { detail });
  }
});

server.listen(options.port, options.host, () => {
  console.log(
    [
      `object-search workbench backend listening on http://${options.host}:${options.port}`,
      `config=${options.configPath}`,
      `bricks_api=${options.pythonApiBaseUrl}`,
      `ann_api=${options.annApiBaseUrl}`,
      `annotation_api=${options.annotationBaseUrl}`,
      `ui_dist=${options.uiDistDir}`,
    ].join("\n"),
  );
});
