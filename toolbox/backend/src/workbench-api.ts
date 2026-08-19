import type { IncomingMessage, ServerResponse } from "node:http";

import {
  clearWorkspaceAnnotations,
  createWorkspaceAnnotation,
  deleteDetectionGroupLabel,
  deleteDetectionReview,
  deleteWorkspaceAnnotation,
  deleteWorkspaceClass,
  importWorkspaceAnnotations,
  listWorkspaceAnnotations,
  parseWorkspaceAnnotation,
  parseWorkspaceClass,
  listAnnotations,
  listDetectionGroups,
  parseDetectionGroupLabel,
  parseReviewMutation,
  upsertDetectionGroupLabel,
  upsertDetectionReview,
  upsertWorkspaceClass,
} from "./annotation-store.js";
import {
  benchmarkRunPayload,
  benchmarkStatusPayload,
  scorePromptPayload,
  startBenchmarkRun,
  type BenchmarkRunParams,
} from "./benchmark-runner.js";
import {
  analysisLayerPayload,
  analysisRunPayload,
  analysisStatusPayload,
  startAnalysisRun,
} from "./map-analysis-runner.js";
import { loadMapEntries, type MapEntry } from "./config.js";
import {
  createDirectoryPayload,
  deleteMapPayload,
  deletionPreviewPayload,
  exportPreviewPayload,
  exportStatusPayload,
  listDirectoriesPayload,
  startExportRun,
} from "./export-roi.js";
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
  metadataKeyframesPayload,
  metadataRowsPayload,
  objectSearchMetadataMarkersPayload,
  objectSearchMetadataStatusPayload,
  previewFromPathPng,
  proposalNeighborProjectionRenderPng,
  proposalNeighborProjectionsPayload,
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


/** A nested object off a JSON body, or an empty one — the parsers validate it. */
function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function isWorkbenchUiMapRoute(pathname: string): boolean {
  // The trailing segment is optional so `DELETE /ui/api/maps/{id}` — the map itself
  // as a resource — reaches the same dispatcher as everything hanging off it.
  return /^\/ui\/api\/maps\/[^/]+(\/|$)/.test(pathname);
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
    if (method === "GET" && suffix === "/analysis/status") {
      sendJson(response, 200, await analysisStatusPayload(map));
      return true;
    }
    if (method === "POST" && suffix === "/analysis/run") {
      sendJson(response, 202, startAnalysisRun(options, map));
      return true;
    }
    // Matched before the run route below, whose `([^/]+)(\/report)?` would otherwise
    // reject the extra segments rather than fall through.
    const analysisLayerMatch = /^\/analysis\/runs\/([^/]+)\/layers\/([^/]+)$/.exec(
      suffix,
    );
    if (method === "GET" && analysisLayerMatch) {
      sendJson(
        response,
        200,
        await analysisLayerPayload(
          map,
          decodeURIComponent(analysisLayerMatch[1]),
          decodeURIComponent(analysisLayerMatch[2]),
        ),
      );
      return true;
    }
    const analysisRunMatch = /^\/analysis\/runs\/([^/]+)(\/report)?$/.exec(suffix);
    if (method === "GET" && analysisRunMatch) {
      sendJson(
        response,
        200,
        await analysisRunPayload(
          map,
          decodeURIComponent(analysisRunMatch[1]),
          analysisRunMatch[2] === "/report",
        ),
      );
      return true;
    }
    if (method === "GET" && suffix === "/export-roi/directories") {
      sendJson(
        response,
        200,
        await listDirectoriesPayload(options, map, queryString(url, "path")),
      );
      return true;
    }
    if (method === "POST" && suffix === "/export-roi/directories") {
      const body = await requestJson(request);
      sendJson(
        response,
        201,
        await createDirectoryPayload(options, map, String(body.parent ?? ""), String(body.name ?? "")),
      );
      return true;
    }
    if (method === "POST" && suffix === "/export-roi/preview") {
      sendJson(response, 200, await exportPreviewPayload(options, map, await requestJson(request)));
      return true;
    }
    if (method === "POST" && suffix === "/export-roi") {
      sendJson(response, 202, await startExportRun(options, map, await requestJson(request)));
      return true;
    }
    if (method === "GET" && suffix === "/export-roi/status") {
      sendJson(response, 200, exportStatusPayload(map.id));
      return true;
    }
    if (method === "GET" && suffix === "/deletion-preview") {
      sendJson(response, 200, await deletionPreviewPayload(options, map));
      return true;
    }
    // The map itself as a resource: the suffix is "/" because there is no trailing
    // segment (see `isWorkbenchUiMapRoute`).
    if (method === "DELETE" && suffix === "/") {
      sendJson(response, 200, await deleteMapPayload(options, map, await requestJson(request)));
      return true;
    }
    if (method === "POST" && suffix === "/benchmark/score-prompt") {
      const body = await requestJson(request);
      const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
      if (!prompt) {
        throw new WorkbenchRouteError(400, "A non-empty prompt is required.");
      }
      const params =
        body.params && typeof body.params === "object" && !Array.isArray(body.params)
          ? (body.params as BenchmarkRunParams)
          : {};
      sendJson(response, 200, await scorePromptPayload(options, map, prompt, params));
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
    if (method === "GET" && suffix === "/annotations") {
      sendJson(response, 200, listWorkspaceAnnotations(map));
      return true;
    }
    // A whole FeatureCollection in: the file-import path. It merges rather than
    // replaces, matching what dropping a file on the office always did.
    if (method === "POST" && suffix === "/annotations") {
      sendJson(response, 200, importWorkspaceAnnotations(map, await requestJson(request)));
      return true;
    }
    if (method === "DELETE" && suffix === "/annotations") {
      sendJson(response, 200, { deleted: clearWorkspaceAnnotations(map) });
      return true;
    }
    if (method === "POST" && suffix === "/annotations/annotation") {
      const body = await requestJson(request);
      let annotation;
      try {
        annotation = parseWorkspaceAnnotation(body);
      } catch (error) {
        throw new WorkbenchRouteError(400, (error as Error).message);
      }
      sendJson(response, 201, { id: createWorkspaceAnnotation(map, annotation) });
      return true;
    }
    if (method === "DELETE" && suffix === "/annotations/annotation") {
      const body = await requestJson(request);
      const id = typeof body.id === "string" ? body.id : "";
      if (!id) {
        throw new WorkbenchRouteError(400, "An annotation id is required.");
      }
      sendJson(response, 200, { deleted: deleteWorkspaceAnnotation(map, id) });
      return true;
    }
    if (method === "POST" && suffix === "/annotations/class") {
      const body = await requestJson(request);
      let next;
      try {
        next = parseWorkspaceClass(asRecord(body.class));
      } catch (error) {
        throw new WorkbenchRouteError(400, (error as Error).message);
      }
      const rawOriginal = body.original ? asRecord(body.original) : null;
      const original =
        rawOriginal && typeof rawOriginal.name === "string"
          ? {
              name: rawOriginal.name,
              annotationType:
                rawOriginal.annotationType === "polygon"
                  ? ("polygon" as const)
                  : ("point" as const),
            }
          : null;
      upsertWorkspaceClass(map, next, original);
      sendJson(response, 200, { saved: true });
      return true;
    }
    if (method === "DELETE" && suffix === "/annotations/class") {
      const body = await requestJson(request);
      const name = typeof body.name === "string" ? body.name.trim() : "";
      if (!name) {
        throw new WorkbenchRouteError(400, "A class name is required.");
      }
      const annotationType = body.annotationType === "polygon" ? "polygon" : "point";
      sendJson(response, 200, {
        deleted: deleteWorkspaceClass(map, name, annotationType),
      });
      return true;
    }
    if (method === "GET" && suffix === "/review-annotations") {
      const query = queryString(url, "query")?.trim() || null;
      sendJson(response, 200, listAnnotations(map, query));
      return true;
    }
    if (method === "POST" && suffix === "/review-annotations/detection-review") {
      const review = parseReviewMutation(await requestJson(request), true);
      sendJson(response, 201, {
        detection_review_id: upsertDetectionReview(map, review),
      });
      return true;
    }
    if (method === "DELETE" && suffix === "/review-annotations/detection-review") {
      deleteDetectionReview(
        map,
        parseReviewMutation(await requestJson(request), false),
      );
      response.writeHead(204);
      response.end();
      return true;
    }
    if (method === "GET" && suffix === "/group-annotations") {
      sendJson(response, 200, listDetectionGroups(map));
      return true;
    }
    if (method === "POST" && suffix === "/group-annotations/label") {
      sendJson(response, 201, {
        detection_group_label_id: upsertDetectionGroupLabel(
          map,
          parseDetectionGroupLabel(await requestJson(request), true),
        ),
      });
      return true;
    }
    if (method === "DELETE" && suffix === "/group-annotations/label") {
      deleteDetectionGroupLabel(
        map,
        parseDetectionGroupLabel(await requestJson(request), false),
      );
      response.writeHead(204);
      response.end();
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
          options.pythonApiBaseUrl,
        ),
      );
      return true;
    }
    if (method === "GET" && suffix === "/object-search-metadata/markers") {
      sendJson(response, 200, await objectSearchMetadataMarkersPayload(map));
      return true;
    }
    if (method === "GET" && suffix === "/object-search-metadata/keyframes") {
      const sort = queryString(url, "sort") ?? "parquet";
      if (!(["parquet", "objects-desc", "objects-asc"] as const).includes(
        sort as "parquet" | "objects-desc" | "objects-asc",
      )) {
        throw new WorkbenchRouteError(400, `Unknown keyframe sort '${sort}'.`);
      }
      sendJson(
        response,
        200,
        await metadataKeyframesPayload(
          map,
          {
            offset: queryNumber(url, "offset", 0),
            limit: queryNumber(url, "limit", 100),
            sort: sort as "parquet" | "objects-desc" | "objects-asc",
            includeEmpty: url.searchParams.get("include_empty") !== "false",
            keyframeId: queryString(url, "keyframe_id"),
          },
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
    if (
      method === "GET"
      && /^\/object-search-metadata\/rows\/\d+\/neighbor-projections$/.test(suffix)
    ) {
      sendJson(
        response,
        200,
        await proposalNeighborProjectionsPayload(
          map,
          Number(suffix.split("/")[3]),
          queryNumber(url, "count", 5),
          url.searchParams.get("diverse") !== "false",
        ),
      );
      return true;
    }
    if (
      method === "GET"
      && /^\/object-search-metadata\/rows\/\d+\/neighbor-projections\/\d+\.png$/.test(suffix)
    ) {
      sendPng(
        response,
        await proposalNeighborProjectionRenderPng(
          map,
          Number(suffix.split("/")[3]),
          suffix.split("/")[5].replace(/\.png$/, ""),
          {
            size: queryNumber(url, "size", 384),
            fovScale: queryNumber(url, "fov_scale", 2),
          },
        ),
      );
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
