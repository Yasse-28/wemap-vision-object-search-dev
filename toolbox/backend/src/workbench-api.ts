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
  cutoutDetectionsPayload,
  indexClusterOcrPayload,
  indexCutoutPayload,
  indexCutoutPreviewPng,
  indexDepthPinPayload,
  indexKeyframeDepthPreviewPng,
  indexKeyframeEquirectPreviewPng,
  indexObjectCropPng,
  indexObjectsPayload,
  indexProjectWorldPointPayload,
  indexStatusPayload,
  indexViewConePayload,
  keyframeMetadataPayload,
  legacyCutoutPreviewPng,
  previewFromPathPng,
  WorkbenchRouteError,
} from "./workbench-index.js";

function queryNumber(url: URL, name: string, fallback: number): number {
  const value = Number(url.searchParams.get(name));
  return Number.isFinite(value) ? value : fallback;
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
    if (method === "GET" && /^\/cutouts\/[^/]+\/detections$/.test(suffix)) {
      sendJson(response, 200, await cutoutDetectionsPayload(map, decodeURIComponent(suffix.split("/")[2])));
      return true;
    }
    if (method === "GET" && /^\/cutouts\/[^/]+\/preview\.png$/.test(suffix)) {
      sendPng(
        response,
        await legacyCutoutPreviewPng(
          map,
          decodeURIComponent(suffix.split("/")[2]),
          queryString(url, "selected_object_id"),
          url.searchParams.get("selected_only") === "true",
          queryString(url, "selected_bbox"),
        ),
      );
      return true;
    }
    if (method === "GET" && suffix === "/preview.png") {
      sendPng(response, await previewFromPathPng(map, queryString(url, "preview_path")));
      return true;
    }
    if (method === "GET" && suffix === "/object-search-index") {
      sendJson(response, 200, await indexStatusPayload(map, queryString(url, "selected_keyframe_id")));
      return true;
    }
    if (method === "GET" && suffix === "/object-search-index/objects") {
      sendJson(response, 200, await indexObjectsPayload(map, queryString(url, "keyframe_id")));
      return true;
    }
    if (method === "GET" && suffix === "/object-search-index/keyframe-graph") {
      sendJson(response, 200, await keyframeGraphPayload(map));
      return true;
    }
    if (method === "GET" && suffix === "/object-search-index/cluster-ocr") {
      sendJson(response, 200, await indexClusterOcrPayload(map));
      return true;
    }
    if (method === "GET" && /^\/object-search-index\/cutouts\/[^/]+$/.test(suffix)) {
      sendJson(response, 200, await indexCutoutPayload(map, decodeURIComponent(suffix.split("/")[3])));
      return true;
    }
    if (method === "GET" && /^\/object-search-index\/cutouts\/[^/]+\/preview\.png$/.test(suffix)) {
      sendPng(
        response,
        await indexCutoutPreviewPng(map, decodeURIComponent(suffix.split("/")[3]), {
          selectedObjectId: queryString(url, "selected_object_id"),
          drawBoxes: url.searchParams.get("draw_boxes") !== "false",
          nmsIou: queryNumber(url, "nms_iou", 1.0),
          minBBoxArea: queryNumber(url, "min_bbox_area", 0.0),
          maxBBoxArea: queryNumber(url, "max_bbox_area", 0.0),
          showYolo: url.searchParams.get("show_yolo") !== "false",
          showGdino: url.searchParams.get("show_gdino") !== "false",
        }),
      );
      return true;
    }
    if (method === "GET" && /^\/object-search-index\/keyframes\/[^/]+\/equirect-preview\.png$/.test(suffix)) {
      const drawBoxes = url.searchParams.get("draw_boxes") !== "false";
      sendImage(
        response,
        await indexKeyframeEquirectPreviewPng(map, decodeURIComponent(suffix.split("/")[3]), {
          selectedObjectId: queryString(url, "selected_object_id"),
          drawBoxes,
          nmsIou: queryNumber(url, "nms_iou", 1.0),
          minBBoxArea: queryNumber(url, "min_bbox_area", 0.0),
          maxBBoxArea: queryNumber(url, "max_bbox_area", 0.0),
          showYolo: url.searchParams.get("show_yolo") !== "false",
          showGdino: url.searchParams.get("show_gdino") !== "false",
        }),
        drawBoxes ? "image/png" : "image/jpeg",
      );
      return true;
    }
    if (method === "GET" && /^\/object-search-index\/keyframes\/[^/]+\/depth-preview\.png$/.test(suffix)) {
      sendPng(
        response,
        await indexKeyframeDepthPreviewPng(map, decodeURIComponent(suffix.split("/")[3])),
      );
      return true;
    }
    if (method === "POST" && /^\/object-search-index\/keyframes\/[^/]+\/depth-pin$/.test(suffix)) {
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
    if (method === "POST" && /^\/object-search-index\/keyframes\/[^/]+\/view-cone$/.test(suffix)) {
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
    if (method === "POST" && /^\/object-search-index\/keyframes\/[^/]+\/project-world-point$/.test(suffix)) {
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
    if (method === "GET" && /^\/object-search-index\/cutouts\/[^/]+\/objects\/[^/]+\/crop\.png$/.test(suffix)) {
      sendPng(
        response,
        await indexObjectCropPng(
          map,
          decodeURIComponent(suffix.split("/")[3]),
          [
            queryNumber(url, "x0", 0),
            queryNumber(url, "y0", 0),
            queryNumber(url, "x1", 1),
            queryNumber(url, "y1", 1),
          ],
        ),
      );
      return true;
    }
    if (method === "POST" && suffix === "/object-search-index/prompt-latent") {
      sendJson(response, 501, {
        detail: "Prompt latent is not ported to the TypeScript workbench backend yet.",
      });
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
