/**
 * Map-analysis runner: `toolbox.benchmark.map_analysis` behind the Analysis panel.
 *
 * The python tool reads the **whole parquet** and needs neither postgres nor the ANN
 * service, so unlike the benchmark runner this module has no service to health-check
 * and nothing to keep alive. What it does have is a runtime measured in minutes
 * (≈14 min on bbhotel-choisy, 181 919 rows), which is why the panel never analyses on
 * open: a run is started explicitly, its sections stream in as progress, and the JSON
 * payload is cached on disk so every later visit reads the last result.
 *
 * Runs live in `{map_path}/analysis/{runId}/` as `analysis.json` (the payload the UI
 * renders), `report.txt` (the terminal report verbatim, kept because it carries the
 * per-class and per-label tables the panel summarises) and `layers/*.geojson` (the
 * measurement fields the panel draws on the livemap). Layers come from the same run as
 * the numbers on purpose: a heat map from one run beside a score card from another
 * would be impossible to tell apart from a real disagreement.
 */
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import type { MapEntry } from "./config.js";
import type { WorkbenchOptions } from "./http-utils.js";
import { runPython } from "./python-process.js";
import { WorkbenchRouteError } from "./workbench-index.js";

const RUN_ID_PATTERN = /^[0-9A-Za-z._-]+$/;
/** Layer file names, from `map_layers.LAYERS` — no path separators reach the disk. */
const LAYER_NAME_PATTERN = /^[0-9A-Za-z._-]+$/;
/** Every section the tool can produce, in report order — also the progress total. */
export const ANALYSIS_SECTIONS = [
  "s0",
  "s1",
  "s2",
  "s3",
  "s4",
  "s5",
  "s6",
  "s7",
] as const;
/**
 * Lines the python report opens each section with. Progress is read off stdout rather
 * than from a `--progress` flag the tool does not have: `report.head` prints
 * `===== S3 — …`, so the section number is the character after `===== S`.
 */
const SECTION_HEAD = /^=====\s*S(\d)/;

type AnalysisJob = {
  runId: string;
  mapId: string;
  state: "running" | "done" | "error";
  /**
   * Sections the report has ENTERED, in order. `report.head` prints at the start of
   * a section, so the last entry is the one still running — the panel says "at
   * section n of 8" rather than claiming n are done.
   */
  startedSections: string[];
  error: string | null;
  startedAt: string;
  finishedAt: string | null;
};

let activeJob: AnalysisJob | null = null;

// ---------------------------------------------------------------------------
// Paths and disk
// ---------------------------------------------------------------------------

function analysisDirectory(map: MapEntry): string {
  return path.join(map.path, "analysis");
}

function runDirectory(map: MapEntry, runId: string): string {
  if (!RUN_ID_PATTERN.test(runId)) {
    throw new WorkbenchRouteError(400, `Invalid run id '${runId}'`);
  }
  return path.join(analysisDirectory(map), runId);
}

function buildRunId(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return (
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
    + `_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`
  );
}

/**
 * Every run on disk, newest first. A directory without a readable `analysis.json` is
 * skipped rather than reported: it is either in flight or a crashed run, and in both
 * cases there is nothing for the panel to render.
 */
async function listRunIds(map: MapEntry): Promise<string[]> {
  let entries: string[];
  try {
    entries = await readdir(analysisDirectory(map));
  } catch {
    return [];
  }
  const runs: string[] = [];
  for (const entry of entries.sort().reverse()) {
    if (!RUN_ID_PATTERN.test(entry)) {
      continue;
    }
    try {
      await readFile(path.join(analysisDirectory(map), entry, "analysis.json"), "utf8");
    } catch {
      continue;
    }
    runs.push(entry);
  }
  return runs;
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

async function runAnalysisJob(
  options: WorkbenchOptions,
  map: MapEntry,
  job: AnalysisJob,
): Promise<void> {
  const directory = runDirectory(map, job.runId);
  await mkdir(directory, { recursive: true });
  const jsonPath = path.join(directory, "analysis.json");
  const layersDir = path.join(directory, "layers");

  const lines: string[] = [];
  const result = await runPython(options, {
    args: [
      "-m",
      "toolbox.benchmark.map_analysis",
      "--map-path",
      map.path,
      "--json-out",
      jsonPath,
      "--layers-dir",
      layersDir,
    ],
    onLine: (line) => {
      lines.push(line);
      const head = SECTION_HEAD.exec(line.trim());
      if (head) {
        const section = `s${head[1]}`;
        if (!job.startedSections.includes(section)) {
          job.startedSections.push(section);
        }
      }
    },
  });

  await writeFile(path.join(directory, "report.txt"), `${lines.join("\n")}\n`, "utf8");

  if (result.code !== 0) {
    const detail = (result.stderr || result.stdout).trim().split("\n").slice(-8).join("\n");
    throw new Error(`map_analysis exited with ${result.code}\n${detail}`);
  }
  // Proof the payload landed and parses, so `state: "done"` never points at a file
  // the panel will then fail to read.
  JSON.parse(await readFile(jsonPath, "utf8"));
  job.state = "done";
  job.finishedAt = new Date().toISOString();
}

export function startAnalysisRun(
  options: WorkbenchOptions,
  map: MapEntry,
): { run_id: string } {
  if (activeJob && activeJob.state === "running") {
    throw new WorkbenchRouteError(
      409,
      `An analysis run is already in progress for map '${activeJob.mapId}'`
        + ` (run ${activeJob.runId}).`,
    );
  }
  const job: AnalysisJob = {
    runId: buildRunId(),
    mapId: map.id,
    state: "running",
    startedSections: [],
    error: null,
    startedAt: new Date().toISOString(),
    finishedAt: null,
  };
  activeJob = job;

  void runAnalysisJob(options, map, job).catch((error: unknown) => {
    job.state = "error";
    job.error = error instanceof Error ? error.message : String(error);
  });

  return { run_id: job.runId };
}

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

function jobPayload(job: AnalysisJob | null): Record<string, unknown> | null {
  if (!job) {
    return null;
  }
  return {
    run_id: job.runId,
    map_id: job.mapId,
    state: job.state,
    progress: {
      started: job.startedSections.length,
      total: ANALYSIS_SECTIONS.length,
      sections: job.startedSections,
    },
    error: job.error,
    started_at: job.startedAt,
    finished_at: job.finishedAt,
  };
}

/**
 * What the panel opens on: the newest cached payload, the run list, and the in-flight
 * job if there is one. The payload is inlined rather than linked because it is ~80 KB
 * and the panel needs all of it to draw anything.
 */
export async function analysisStatusPayload(
  map: MapEntry,
): Promise<Record<string, unknown>> {
  const runs = await listRunIds(map);
  const latestId = runs[0] ?? null;
  let payload: unknown = null;
  if (latestId) {
    try {
      payload = JSON.parse(
        await readFile(path.join(runDirectory(map, latestId), "analysis.json"), "utf8"),
      );
    } catch {
      payload = null;
    }
  }
  const job = activeJob && activeJob.mapId === map.id ? activeJob : null;
  return {
    map_id: map.id,
    runs,
    latest_run_id: latestId,
    analysis: payload,
    layers: latestId ? await listLayers(map, latestId) : [],
    job: jobPayload(job),
  };
}

/**
 * The layers one run produced, by name, newest-run-first callers included.
 *
 * Read from disk rather than from a constant so the list is what the python tool
 * actually wrote: a layer whose inputs were missing writes an empty collection, and a
 * tool that gained a layer since the run should not make the panel offer a 404.
 */
async function listLayers(map: MapEntry, runId: string): Promise<string[]> {
  try {
    const entries = await readdir(path.join(runDirectory(map, runId), "layers"));
    return entries
      .filter((entry) => entry.endsWith(".geojson"))
      .map((entry) => entry.slice(0, -".geojson".length))
      .sort();
  } catch {
    return [];
  }
}

/** One run's GeoJSON layer, as text — the panel hands it straight to mapbox. */
export async function analysisLayerPayload(
  map: MapEntry,
  runId: string,
  name: string,
): Promise<unknown> {
  if (!LAYER_NAME_PATTERN.test(name)) {
    throw new WorkbenchRouteError(400, `Invalid layer name '${name}'`);
  }
  const filePath = path.join(runDirectory(map, runId), "layers", `${name}.geojson`);
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch {
    throw new WorkbenchRouteError(404, `No layer '${name}' for analysis run ${runId}`);
  }
}

/** One run's payload, or its report text when `report` is set. */
export async function analysisRunPayload(
  map: MapEntry,
  runId: string,
  report: boolean,
): Promise<unknown> {
  const fileName = report ? "report.txt" : "analysis.json";
  const filePath = path.join(runDirectory(map, runId), fileName);
  try {
    const text = await readFile(filePath, "utf8");
    return report ? { run_id: runId, report: text } : JSON.parse(text);
  } catch {
    throw new WorkbenchRouteError(404, `No ${fileName} for analysis run ${runId}`);
  }
}
