import { stat } from "node:fs/promises";
import { createServer } from "node:http";

import { loadMapEntries, type MapEntry } from "./config.js";
import { loadMapManifest } from "./map-manifest.js";
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

async function isDirectory(target: string): Promise<boolean> {
  try {
    return (await stat(target)).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Map entries enriched with object-search availability for the UI map list.
 *
 * Availability used to mean "has an object-search.db". That SQLite index is gone
 * (ADR 0002) — the index lives in pgvector now, which we cannot cheaply probe from
 * here. So the check is what this process *can* see on disk: the map directory, and
 * a **parseable v2 manifest**, without which nothing downstream resolves a position.
 *
 * Parsing rather than name-matching is deliberate: a file whose name fits the
 * pattern but whose contents are broken used to report `available: true` and then
 * fail with a 500 deep inside a panel. The parse is cached by mtime, so the
 * repeated polling of this route costs one read per manifest revision.
 *
 * A map that passes this check but was never ingested surfaces as an empty result
 * list rather than an error — the one honest gap in this signal. The explorer's own
 * status route reports what *it* found (`metadata.parquet`, pgvector coverage), which
 * is the place to look when a map opens but has no rows.
 */
async function mapSummaries(configPath: string): Promise<Record<string, unknown>[]> {
  const maps = await loadMapEntries(configPath);
  return Promise.all(
    maps.map(async (map: MapEntry) => {
      let reason: string | null = null;
      if (!(await isDirectory(map.path))) {
        reason = `Map directory not found: ${map.path}`;
      } else {
        try {
          await loadMapManifest(map.path);
        } catch (error) {
          reason = error instanceof Error ? error.message : String(error);
        }
      }
      return {
        ...map,
        object_search_available: reason == null,
        unavailable_reason: reason,
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
        annotations: "integrated-sqlite",
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
      "annotations=integrated-sqlite",
      `ui_dist=${options.uiDistDir}`,
    ].join("\n"),
  );
});
