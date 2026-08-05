import type {
  BenchmarkMetrics,
  BenchmarkRawResults,
  BenchmarkRunParams,
  BenchmarkStatus,
} from "./types";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function mapBase(mapId: string): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}`;
}

export async function fetchBenchmarkStatus(mapId: string): Promise<BenchmarkStatus> {
  const response = await fetch(`${mapBase(mapId)}/benchmark/status`);
  return readJson<BenchmarkStatus>(response);
}

export async function startBenchmarkRun(
  mapId: string,
  params: BenchmarkRunParams,
): Promise<{ run_id: string }> {
  const response = await fetch(`${mapBase(mapId)}/benchmark/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return readJson<{ run_id: string }>(response);
}

export async function fetchBenchmarkRun(
  mapId: string,
  runId: string,
): Promise<BenchmarkMetrics> {
  const response = await fetch(
    `${mapBase(mapId)}/benchmark/runs/${encodeURIComponent(runId)}`,
  );
  return readJson<BenchmarkMetrics>(response);
}

export async function fetchBenchmarkRunRaw(
  mapId: string,
  runId: string,
): Promise<BenchmarkRawResults> {
  const response = await fetch(
    `${mapBase(mapId)}/benchmark/runs/${encodeURIComponent(runId)}/raw`,
  );
  return readJson<BenchmarkRawResults>(response);
}
