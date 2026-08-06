import { mkdir, rename, unlink, writeFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import path from "node:path";

import {
  benchmarkRunPayload,
  benchmarkStatusPayload,
  startBenchmarkRun,
  type BenchmarkRunParams,
} from "./benchmark-runner.js";
import { loadMapEntries, type MapEntry } from "./config.js";
import {
  readRequestBody,
  sendJson,
  sendText,
  type WorkbenchOptions,
} from "./http-utils.js";
import { keyframeGraphPayload } from "./keyframe-graph.js";
import {
  indexDepthPinPayload,
  indexKeyframeDepthPreviewPng,
  indexKeyframeEquirectPreviewPng,
  indexProjectWorldPointPayload,
  indexViewConePayload,
  keyframeMetadataPayload,
  metadataRowPayload,
  metadataRowRenderPng,
  metadataRowsPayload,
  objectSearchMetadataStatusPayload,
  previewFromPathPng,
  rowFilterParamsFromQuery,
  WorkbenchRouteError,
} from "./workbench-index.js";

/**
 * A numeric query parameter, or the fallback.
 *
 * The `searchParams.get` result must be checked for null *before* `Number()`:
 * `Number(null)` is 0, which is finite, so an absent parameter used to take the value
 * 0 rather than the fallback. That silently meant `limit=0` (→ one row after
 * clamping) and `nms_iou=0` (→ every box suppressed) on requests that passed neither.
 */
function queryNumber(url: URL, name: string, fallback: number): number {
  const raw = url.searchParams.get(name);
  if (raw === null || raw.trim() === "") {
    return fallback;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

/** Same, for an optional integer with no meaningful default. */
function queryInteger(url: URL, name: string): number | null {
  const value = queryNumber(url, name, Number.NaN);
  return Number.isInteger(value) ? value : null;
}

function queryString(url: URL, name: string): string | null {
  const value = url.searchParams.get(name);
  return value && value.length ? value : null;
}

async function requestJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const raw = await readRequestBody(request);
  if (!raw.length) {
    return {};
  }
  const parsed = JSON.parse(raw.toString("utf8")) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {};
  }
  return parsed as Record<string, unknown>;
}

function sendPng(response: ServerResponse, body: Buffer): void {
  response.writeHead(200, {
    "content-type": "image/png",
    "content-length": body.byteLength,
    "cache-control": "no-store",
  });
  response.end(body);
}

function sendImage(response: ServerResponse, body: Buffer, contentType: string): void {
  response.writeHead(200, {
    "content-type": contentType,
    "content-length": body.byteLength,
    "cache-control": "private, max-age=300",
  });
  response.end(body);
}

async function findMap(configPath: string, mapId: string): Promise<MapEntry | null> {
  const maps = await loadMapEntries(configPath);
  return maps.find((item) => item.id === mapId) ?? null;
}

async function saveAnnotations(
  map: MapEntry,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  if (body.type !== "FeatureCollection" || !Array.isArray(body.features)) {
    throw new WorkbenchRouteError(400, "Annotations must be a GeoJSON FeatureCollection.");
  }
  const annotationsDir = path.join(map.path, "annotations");
  const annotationsPath = path.join(annotationsDir, "annotations.geojson");
  const temporaryPath = path.join(
    annotationsDir,
    `.annotations.geojson.${process.pid}.${Date.now()}.tmp`,
  );
  await mkdir(annotationsDir, { recursive: true });
  let renamed = false;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(body, null, 2)}\n`, "utf8");
    await rename(temporaryPath, annotationsPath);
    renamed = true;
  } finally {
    if (!renamed) {
      await unlink(temporaryPath).catch(() => {});
    }
  }
  return {
    saved: true,
    annotation_count: body.features.length,
    path: annotationsPath,
    relative_path: "annotations/annotations.geojson",
  };
}

export function isWorkbenchUiMapRoute(pathname: string): boolean {
  return /^\/ui\/api\/maps\/[^/]+\//.test(pathname);
}

export async function handleWorkbenchUiMapRoute(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  options: WorkbenchOptions,
): Promise<boolean> {
  const match = /^\/ui\/api\/maps\/([^/]+)(?:\/(.*))?$/.exec(url.pathname);
  if (!match) {
    return false;
  }

  const mapId = decodeURIComponent(match[1]);
  const suffix = `/${match[2] ?? ""}`;
  const map = await findMap(options.configPath, mapId);
  if (!map) {
    sendJson(response, 404, { detail: `Map ${mapId} not configured` });
    return true;
  }

  const method = request.method ?? "GET";

  try {
    if (method === "GET" && suffix === "/benchmark/status") {
      sendJson(response, 200, await benchmarkStatusPayload(options, map));
      return true;
    }
    if (method === "POST" && suffix === "/benchmark/run") {
      const params = (await requestJson(request)) as BenchmarkRunParams;
      sendJson(response, 202, startBenchmarkRun(options, map, params));
      return true;
    }
    const benchmarkRunMatch = /^\/benchmark\/runs\/([^/]+)(\/raw)?$/.exec(suffix);
    if (method === "GET" && benchmarkRunMatch) {
      sendJson(
        response,
        200,
        await benchmarkRunPayload(
          map,
          decodeURIComponent(benchmarkRunMatch[1]),
          benchmarkRunMatch[2] === "/raw",
        ),
      );
      return true;
    }
    if (method === "POST" && suffix === "/annotations") {
      sendJson(response, 200, await saveAnnotations(map, await requestJson(request)));
      return true;
    }
    if (method === "GET" && /^\/keyframes\/[^/]+$/.test(suffix)) {
      sendJson(response, 200, await keyframeMetadataPayload(map, decodeURIComponent(suffix.split("/")[2])));
      return true;
    }
    if (method === "GET" && suffix === "/preview.png") {
      sendPng(response, await previewFromPathPng(map, queryString(url, "preview_path")));
      return true;
    }
    if (method === "GET" && suffix === "/object-search-metadata") {
      sendJson(
        response,
        200,
        await objectSearchMetadataStatusPayload(
          map,
          queryString(url, "selected_keyframe_id"),
          options.pythonApiBaseUrl,
        ),
      );
      return true;
    }
    if (method === "GET" && suffix === "/object-search-metadata/rows") {
      const keyframeIds = queryString(url, "keyframe_ids");
      sendJson(
        response,
        200,
        await metadataRowsPayload(map, {
          keyframeIds: keyframeIds ? keyframeIds.split(",").filter(Boolean) : null,
          offset: queryNumber(url, "offset", 0),
          limit: queryNumber(url, "limit", 2000),
          detectorSource: queryString(url, "detector_source"),
          labelQuery: queryString(url, "label"),
          withDepthOnly: url.searchParams.get("with_depth") === "true",
        }),
      );
      return true;
    }
    if (method === "GET" && suffix === "/object-search-metadata/keyframe-graph") {
      sendJson(response, 200, await keyframeGraphPayload(map));
      return true;
    }
    if (method === "GET" && /^\/object-search-metadata\/rows\/\d+$/.test(suffix)) {
      sendJson(response, 200, await metadataRowPayload(map, Number(suffix.split("/")[3])));
      return true;
    }
    if (method === "GET" && /^\/object-search-metadata\/rows\/\d+\/render\.png$/.test(suffix)) {
      sendPng(
        response,
        await metadataRowRenderPng(map, Number(suffix.split("/")[3]), {
          size: queryNumber(url, "size", 512),
          fovScale: queryNumber(url, "fov_scale", 1),
        }),
      );
      return true;
    }
    if (method === "GET" && /^\/object-search-metadata\/keyframes\/[^/]+\/equirect-preview\.png$/.test(suffix)) {
      const drawBoxes = url.searchParams.get("draw_boxes") !== "false";
      sendImage(
        response,
        await indexKeyframeEquirectPreviewPng(map, decodeURIComponent(suffix.split("/")[3]), {
          selectedRowIndex: queryInteger(url, "selected_row_index"),
          drawBoxes,
          ...rowFilterParamsFromQuery((name) => url.searchParams.get(name)),
        }),
        drawBoxes ? "image/png" : "image/jpeg",
      );
      return true;
    }
    if (method === "GET" && /^\/object-search-metadata\/keyframes\/[^/]+\/depth-preview\.png$/.test(suffix)) {
      sendPng(
        response,
        await indexKeyframeDepthPreviewPng(map, decodeURIComponent(suffix.split("/")[3])),
      );
      return true;
    }
    if (method === "POST" && /^\/object-search-metadata\/keyframes\/[^/]+\/depth-pin$/.test(suffix)) {
      sendJson(
        response,
        200,
        await indexDepthPinPayload(
          map,
          decodeURIComponent(suffix.split("/")[3]),
          await requestJson(request),
        ),
      );
      return true;
    }
    if (method === "POST" && /^\/object-search-metadata\/keyframes\/[^/]+\/view-cone$/.test(suffix)) {
      sendJson(
        response,
        200,
        await indexViewConePayload(
          map,
          decodeURIComponent(suffix.split("/")[3]),
          await requestJson(request),
        ),
      );
      return true;
    }
    if (method === "POST" && /^\/object-search-metadata\/keyframes\/[^/]+\/project-world-point$/.test(suffix)) {
      sendJson(
        response,
        200,
        await indexProjectWorldPointPayload(
          map,
          decodeURIComponent(suffix.split("/")[3]),
          await requestJson(request),
        ),
      );
      return true;
    }
  } catch (error) {
    if (error instanceof WorkbenchRouteError) {
      sendJson(response, error.status, { detail: error.message });
      return true;
    }
    throw error;
  }

  sendText(response, 404, `No workbench UI route for ${method} ${url.pathname}`);
  return true;
}
