import type { GeoJsonFeatureCollection } from "../annotations/livemapHost";
import { fetchJson } from "../http";

import type { AnalysisPayload, AnalysisStatus } from "./types";

function mapBase(mapId: string): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}`;
}

export async function fetchAnalysisStatus(mapId: string): Promise<AnalysisStatus> {
  return (await fetchJson(`${mapBase(mapId)}/analysis/status`)) as AnalysisStatus;
}

export async function startAnalysisRun(mapId: string): Promise<{ run_id: string }> {
  return (await fetchJson(`${mapBase(mapId)}/analysis/run`, {
    method: "POST",
  })) as { run_id: string };
}

export async function fetchAnalysisRun(
  mapId: string,
  runId: string,
): Promise<AnalysisPayload> {
  return (await fetchJson(
    `${mapBase(mapId)}/analysis/runs/${encodeURIComponent(runId)}`,
  )) as AnalysisPayload;
}

export async function fetchAnalysisLayer(
  mapId: string,
  runId: string,
  name: string,
): Promise<GeoJsonFeatureCollection> {
  return (await fetchJson(
    `${mapBase(mapId)}/analysis/runs/${encodeURIComponent(runId)}/layers/`
      + encodeURIComponent(name),
  )) as GeoJsonFeatureCollection;
}

export async function fetchAnalysisReport(
  mapId: string,
  runId: string,
): Promise<string> {
  const payload = (await fetchJson(
    `${mapBase(mapId)}/analysis/runs/${encodeURIComponent(runId)}/report`,
  )) as { report?: string };
  return payload.report ?? "";
}
