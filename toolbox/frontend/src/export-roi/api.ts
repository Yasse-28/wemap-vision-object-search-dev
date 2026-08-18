/** Client for the export/delete routes under `/ui/api/maps/{id}`. */

import { fetchJson } from "../http";
import type {
  DeletionPreview,
  DirectoryEntry,
  DirectoryListing,
  ExportJob,
  ExportPreview,
  ExportRequest,
} from "./types";

function base(mapId: string): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}`;
}

function postJson(path: string, body: unknown): Promise<unknown> {
  return fetchJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function browseDirectories(
  mapId: string,
  path: string | null,
): Promise<DirectoryListing> {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return (await fetchJson(
    `${base(mapId)}/export-roi/directories${query}`,
  )) as DirectoryListing;
}

export async function createDirectory(
  mapId: string,
  parent: string,
  name: string,
): Promise<DirectoryEntry> {
  return (await postJson(`${base(mapId)}/export-roi/directories`, {
    parent,
    name,
  })) as DirectoryEntry;
}

export async function previewExport(
  mapId: string,
  rois: ExportRequest["rois"],
): Promise<ExportPreview> {
  return (await postJson(`${base(mapId)}/export-roi/preview`, {
    rois,
  })) as ExportPreview;
}

export async function startExport(
  mapId: string,
  request: ExportRequest,
): Promise<ExportJob> {
  return (await postJson(`${base(mapId)}/export-roi`, request)) as ExportJob;
}

export async function fetchExportStatus(
  mapId: string,
): Promise<ExportJob | null> {
  const data = (await fetchJson(`${base(mapId)}/export-roi/status`)) as {
    job: ExportJob | null;
  };
  return data.job;
}

export async function fetchDeletionPreview(
  mapId: string,
): Promise<DeletionPreview> {
  return (await fetchJson(
    `${base(mapId)}/deletion-preview`,
  )) as DeletionPreview;
}

/**
 * Delete a map. `confirm_id` must equal `mapId`; the backend re-checks it, so a
 * client that skipped the dialog cannot delete with an empty body.
 */
export async function deleteMap(
  mapId: string,
): Promise<Record<string, unknown>> {
  return (await fetchJson(base(mapId), {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm_id: mapId }),
  })) as Record<string, unknown>;
}
