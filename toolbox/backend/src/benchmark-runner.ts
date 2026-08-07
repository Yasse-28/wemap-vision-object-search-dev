/**
 * Benchmark runner: runs the object-search HTTP benchmark for a map from the toolbox.
 *
 * Since ADR 0002 there are TWO python services, and this module treats them very
 * differently:
 *
 * - the **bricks service** (`toolbox.bricks.service`, :45678) serves
 *   `/{map_id}/object-search/localize`. It is cheap to start, so we spawn it on
 *   demand and keep it alive across runs — same as the standalone service before it.
 * - the **mirrored online service** (`services/object_search_online`, :8000) does the
 *   embedding on the GPU. We only health-check it and tell the user how to start it.
 *   Loading MetaCLIP behind a button press is not a decision to make implicitly.
 *
 * Ground truth comes from the SQLite annotation store owned by this backend.
 *
 * Results live in {map_path}/benchmark/{runId}/ (metrics.json, raw_results.json).
 */
import { spawn, type ChildProcess } from "node:child_process";
import { mkdir, readdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { buildGroundTruth } from "./annotation-store.js";
import type { MapEntry } from "./config.js";
import type { WorkbenchOptions } from "./http-utils.js";
import { WorkbenchRouteError } from "./workbench-index.js";

const RUN_ID_PATTERN = /^[0-9A-Za-z._-]+$/;
const SERVICE_READY_TIMEOUT_MS = 15 * 60 * 1000;
const SERVICE_POLL_INTERVAL_MS = 2000;
const STDERR_TAIL_LINES = 50;
const DEFAULT_REMOTE_LOCALIZE_URL_TEMPLATE =
  "https://vps-api.wemap-vision-computing-1.getwemap.com/{map_id}/object-search/localize";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PromptProgressEvent = {
  index: number;
  total: number;
  class_name: string;
  prompt: string;
  precision: number;
  recall: number;
  f1: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  elapsed_ms: number;
  error: string | null;
};

export type BenchmarkRunParams = {
  target_server?: "local" | "remote";
  remote_localize_url?: string;
  acceptance_threshold?: number;
  num_results?: number;
  min_similarity?: number;
  clustering_eps_m?: number;
  candidate_count?: number;
  group_annotation_radius_m?: number;
  score_field?: string;
};

type BenchmarkJob = {
  runId: string;
  mapId: string;
  state: "ensuring-service" | "running" | "done" | "error";
  serviceState: string | null;
  progress: { completed: number; total: number | null };
  prompts: PromptProgressEvent[];
  summary: Record<string, unknown> | null;
  error: string | null;
  startedAt: string;
  child: ChildProcess | null;
};

type ManagedService = {
  child: ChildProcess;
  pid: number;
  stderrTail: string[];
  exited: boolean;
};

let activeJob: BenchmarkJob | null = null;
let managedService: ManagedService | null = null;
let cleanupRegistered = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pythonBinaryCandidates(): string[] {
  return [
    process.env.OBJECT_SEARCH_WORKBENCH_PYTHON,
    process.env.PYTHON,
    "/home/yacine/anaconda3/envs/wemap-vision/bin/python",
    "python3",
    "python",
  ].filter((item): item is string => Boolean(item));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await stat(target);
    return true;
  } catch {
    return false;
  }
}

function appendTail(tail: string[], chunk: string): void {
  for (const line of chunk.split("\n")) {
    const trimmed = line.trimEnd();
    if (trimmed) {
      tail.push(trimmed);
    }
  }
  while (tail.length > STDERR_TAIL_LINES) {
    tail.shift();
  }
}

type HealthResponse = { status?: string; maps?: Record<string, string> };

async function fetchHealth(baseUrl: string): Promise<HealthResponse | null> {
  try {
    const response = await fetch(`${baseUrl}/health`, {
      signal: AbortSignal.timeout(2000),
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as HealthResponse;
  } catch {
    return null;
  }
}

function registerCleanup(): void {
  if (cleanupRegistered) {
    return;
  }
  cleanupRegistered = true;
  const cleanup = () => {
    if (managedService && !managedService.exited) {
      managedService.child.kill();
    }
  };
  process.on("exit", cleanup);
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.on(signal, () => {
      cleanup();
      process.exit(0);
    });
  }
}

// ---------------------------------------------------------------------------
// Python service management
// ---------------------------------------------------------------------------

/**
 * Environment for a spawned python process: the mirrored trees are not a src/
 * layout, so `prepare`/`inference`/`indexing` only import with
 * third_party/object_search on PYTHONPATH — plus the repo root for `toolbox.*`.
 */
function pythonEnv(options: WorkbenchOptions): NodeJS.ProcessEnv {
  const roots = [
    options.repoRoot,
    path.join(options.repoRoot, "third_party", "object_search"),
  ];
  const existing = process.env.PYTHONPATH;
  return {
    ...process.env,
    PYTHONPATH: existing ? [...roots, existing].join(path.delimiter) : roots.join(path.delimiter),
  };
}

/**
 * Spawn the bricks service. Unlike the standalone service it replaces, it reads the
 * toolbox config file directly, so there is no generated temp config to manage.
 */
async function spawnPythonService(
  options: WorkbenchOptions,
): Promise<ManagedService> {
  const port = new URL(options.pythonApiBaseUrl).port || "45678";
  let lastError = "";
  for (const pythonBin of pythonBinaryCandidates()) {
    const child = spawn(
      pythonBin,
      [
        "-m", "toolbox.bricks.service",
        "--config", options.configPath,
        "--host", "127.0.0.1",
        "--port", port,
        "--ann-base-url", options.annApiBaseUrl,
        "--cors",
      ],
      { cwd: options.repoRoot, env: pythonEnv(options), stdio: ["ignore", "ignore", "pipe"] },
    );
    const service: ManagedService = {
      child,
      pid: child.pid ?? -1,
      stderrTail: [],
      exited: false,
    };
    const spawned = await new Promise<boolean>((resolve) => {
      child.once("spawn", () => resolve(true));
      child.once("error", (error) => {
        lastError = String(error);
        resolve(false);
      });
    });
    if (!spawned) {
      continue;
    }
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => appendTail(service.stderrTail, chunk));
    child.on("exit", () => {
      service.exited = true;
    });
    registerCleanup();
    return service;
  }
  throw new Error(`Could not spawn the bricks service: ${lastError || "no python binary found"}`);
}

async function stopManagedService(): Promise<void> {
  if (!managedService) {
    return;
  }
  if (!managedService.exited) {
    managedService.child.kill();
  }
  managedService = null;
}

/**
 * The mirrored online service must already be running — we never start it.
 *
 * Its /health returns `{"status": "ok"}`, or 503 while the embedder or the DB pool
 * is not up. A 503 body carries the reason (warming up vs. database unreachable),
 * which is worth surfacing verbatim: "database unreachable" here almost always means
 * the pgvector container is down or DATABASE_* is misconfigured.
 */
async function assertAnnServiceReachable(options: WorkbenchOptions): Promise<void> {
  let detail = "";
  try {
    const response = await fetch(`${options.annApiBaseUrl}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (response.ok) {
      return;
    }
    detail = `HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`;
  } catch (error) {
    detail = error instanceof Error ? error.message : String(error);
  }
  throw new Error(
    `The mirrored online service at ${options.annApiBaseUrl} is not ready (${detail}).\n`
    + "Start it first — it loads MetaCLIP on the GPU, so the toolbox will not:\n"
    + "  scripts/run-online-service.sh\n"
    + "It also needs the local database up: docker compose -f infra/postgres/compose.yml up -d",
  );
}

/** Ensure the python service is reachable and the target map is ready. */
async function ensurePythonService(
  options: WorkbenchOptions,
  map: MapEntry,
  job: BenchmarkJob,
): Promise<void> {
  const deadline = Date.now() + SERVICE_READY_TIMEOUT_MS;
  let spawnedThisCall = false;

  while (Date.now() < deadline) {
    const health = await fetchHealth(options.pythonApiBaseUrl);

    if (health) {
      const mapState = health.maps?.[map.id];
      job.serviceState = mapState ?? health.status ?? null;
      if (mapState === "ready") {
        return;
      }
      if (mapState === "error") {
        throw new Error(
          `The python service failed to load map '${map.id}'`
          + (managedService ? ` (service stderr: ${managedService.stderrTail.slice(-5).join(" | ")})` : ""),
        );
      }
      if (mapState === "loading" || health.status === "starting") {
        await sleep(SERVICE_POLL_INTERVAL_MS);
        continue;
      }
      // Map unknown to the running service.
      if (managedService) {
        // Our own service predates a config change — restart it with the new config.
        await stopManagedService();
      } else {
        throw new Error(
          `A python service at ${options.pythonApiBaseUrl} is running but does not serve map `
          + `'${map.id}'. Restart it with that map configured, or stop it so the toolbox can `
          + "spawn one automatically.",
        );
      }
    }

    if (managedService?.exited) {
      const tail = managedService.stderrTail.join("\n");
      await stopManagedService();
      throw new Error(`The spawned python service exited unexpectedly:\n${tail}`);
    }
    if (!managedService) {
      if (spawnedThisCall) {
        throw new Error("The python service was spawned but never became reachable.");
      }
      job.serviceState = "starting";
      managedService = await spawnPythonService(options);
      spawnedThisCall = true;
    }
    await sleep(SERVICE_POLL_INTERVAL_MS);
  }
  throw new Error(`Timed out waiting for the python service to become ready for map '${map.id}'.`);
}

// ---------------------------------------------------------------------------
// Annotations inspection
// ---------------------------------------------------------------------------

type AnnotationsStatus = {
  exists: boolean;
  path: string;
  annotation_count: number;
  prompt_count: number;
};

async function annotationsStatus(map: MapEntry): Promise<AnnotationsStatus> {
  const annotationsPath = path.join(map.path, "benchmark", "annotations.geojson");
  const status: AnnotationsStatus = {
    exists: false,
    path: annotationsPath,
    annotation_count: 0,
    prompt_count: 0,
  };
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(annotationsPath, "utf8"));
  } catch {
    return status;
  }
  status.exists = true;
  const features = (parsed as { features?: unknown }).features;
  if (!Array.isArray(features)) {
    return status;
  }
  const prompts = new Set<string>();
  for (const feature of features) {
    if (!feature || typeof feature !== "object") {
      continue;
    }
    const properties = (feature as { properties?: Record<string, unknown> }).properties ?? {};
    const className = properties.class ?? properties.name;
    if (typeof className !== "string" || !className) {
      continue;
    }
    status.annotation_count += 1;
    const prompt = typeof properties.prompt === "string" && properties.prompt
      ? properties.prompt
      : className;
    prompts.add(prompt);
  }
  status.prompt_count = prompts.size;
  return status;
}

// ---------------------------------------------------------------------------
// Run listing / results
// ---------------------------------------------------------------------------

function runDirectory(map: MapEntry, runId: string): string {
  if (!RUN_ID_PATTERN.test(runId)) {
    throw new WorkbenchRouteError(400, `Invalid run id: ${runId}`);
  }
  return path.join(map.path, "benchmark", runId);
}

type RunListItem = {
  run_id: string;
  summary: Record<string, unknown> | null;
  has_raw: boolean;
};

async function listRuns(map: MapEntry): Promise<RunListItem[]> {
  const benchmarkDir = path.join(map.path, "benchmark");
  let entries: string[];
  try {
    entries = await readdir(benchmarkDir);
  } catch {
    return [];
  }
  const runs: RunListItem[] = [];
  for (const entry of entries.sort().reverse()) {
    if (!RUN_ID_PATTERN.test(entry)) {
      continue;
    }
    const metricsPath = path.join(benchmarkDir, entry, "metrics.json");
    let summary: Record<string, unknown> | null = null;
    try {
      const metrics = JSON.parse(await readFile(metricsPath, "utf8")) as Record<string, unknown>;
      summary = (metrics.summary as Record<string, unknown>) ?? null;
    } catch {
      continue;
    }
    runs.push({
      run_id: entry,
      summary,
      has_raw: await pathExists(path.join(benchmarkDir, entry, "raw_results.json")),
    });
  }
  return runs;
}

export async function benchmarkRunPayload(
  map: MapEntry,
  runId: string,
  raw: boolean,
): Promise<unknown> {
  const fileName = raw ? "raw_results.json" : "metrics.json";
  const filePath = path.join(runDirectory(map, runId), fileName);
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch {
    throw new WorkbenchRouteError(404, `No ${fileName} for run ${runId}`);
  }
}

// ---------------------------------------------------------------------------
// Status payload
// ---------------------------------------------------------------------------

function jobPayload(job: BenchmarkJob | null): Record<string, unknown> | null {
  if (!job) {
    return null;
  }
  return {
    run_id: job.runId,
    map_id: job.mapId,
    state: job.state,
    service_state: job.serviceState,
    progress: job.progress,
    prompts: job.prompts,
    summary: job.summary,
    error: job.error,
    started_at: job.startedAt,
  };
}

export async function benchmarkStatusPayload(
  options: WorkbenchOptions,
  map: MapEntry,
): Promise<Record<string, unknown>> {
  const health = await fetchHealth(options.pythonApiBaseUrl);
  return {
    python_service: {
      base_url: options.pythonApiBaseUrl,
      reachable: health != null,
      status: health?.status ?? null,
      map_state: health?.maps?.[map.id] ?? null,
      managed: managedService != null && !managedService.exited,
      pid: managedService && !managedService.exited ? managedService.pid : null,
    },
    annotations: await annotationsStatus(map),
    job: jobPayload(activeJob),
    runs: await listRuns(map),
  };
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

function buildRunId(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return (
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
    + `_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`
  );
}

function numberParam(params: BenchmarkRunParams, key: keyof BenchmarkRunParams, fallback: number): number {
  const value = params[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function benchmarkTarget(params: BenchmarkRunParams): "local" | "remote" {
  return params.target_server === "remote" ? "remote" : "local";
}

function remoteLocalizeUrl(params: BenchmarkRunParams): string {
  const url = typeof params.remote_localize_url === "string" && params.remote_localize_url.trim()
    ? params.remote_localize_url.trim()
    : DEFAULT_REMOTE_LOCALIZE_URL_TEMPLATE;
  return url;
}

function benchmarkScriptArgs(
  options: WorkbenchOptions,
  map: MapEntry,
  runDir: string,
  params: BenchmarkRunParams,
): string[] {
  const args = [
    path.join(options.repoRoot, "toolbox", "benchmark", "object_search_http_benchmark.py"),
    "--map-path", map.path,
    "--map-id", map.id,
    "--base-url", options.pythonApiBaseUrl,
    "--api-style", "standalone",
    "--online",
    "--output-dir", runDir,
    "--progress-json",
    "--acceptance-threshold", String(numberParam(params, "acceptance_threshold", 0.9)),
    "--num-results", String(numberParam(params, "num_results", 100)),
    "--min-similarity", String(numberParam(params, "min_similarity", 0.15)),
    "--clustering-eps-m", String(numberParam(params, "clustering_eps_m", 1.5)),
    "--candidate-count", String(numberParam(params, "candidate_count", 1000)),
    "--group-annotation-radius-m", String(numberParam(params, "group_annotation_radius_m", 0)),
  ];
  if (benchmarkTarget(params) === "remote") {
    args.push("--localize-url", remoteLocalizeUrl(params));
  }
  if (typeof params.score_field === "string" && params.score_field) {
    args.push("--score-field", params.score_field);
  }
  return args;
}

function handleProgressLine(job: BenchmarkJob, line: string): void {
  let event: Record<string, unknown>;
  try {
    event = JSON.parse(line) as Record<string, unknown>;
  } catch {
    return;
  }
  if (event.event === "start") {
    job.progress.total = typeof event.prompt_count === "number" ? event.prompt_count : null;
  } else if (event.event === "prompt") {
    job.prompts.push(event as unknown as PromptProgressEvent);
    job.progress.completed = typeof event.index === "number" ? event.index : job.progress.completed + 1;
  } else if (event.event === "done") {
    job.summary = (event.summary as Record<string, unknown>) ?? null;
  }
}

/**
 * Refresh `{map}/benchmark/annotations.geojson` from the backend's SQLite store.
 * The write is atomic because the benchmark reads this path moments later.
 * On failure, keep the existing file so a stale but useful benchmark can still run.
 */
async function regenerateGroundTruth(map: MapEntry): Promise<void> {
  const target = path.join(map.path, "benchmark", "annotations.geojson");
  try {
    const groundTruth = buildGroundTruth(map);
    const body = `${JSON.stringify(groundTruth, null, 2)}\n`;
    await mkdir(path.dirname(target), { recursive: true });
    const tmp = `${target}.tmp-${process.pid}`;
    await writeFile(tmp, body, "utf8");
    await rename(tmp, target);
    console.log(
      `[benchmark] refreshed ground truth for '${map.id}' `
      + `(${groundTruth.features.length} annotation(s)).`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(
      `[benchmark] could not refresh ground truth for '${map.id}' from the local `
      + `annotation store: ${message}; using the existing annotations.geojson.`,
    );
  }
}

async function runBenchmarkJob(
  options: WorkbenchOptions,
  map: MapEntry,
  job: BenchmarkJob,
  params: BenchmarkRunParams,
): Promise<void> {
  if (benchmarkTarget(params) === "local") {
    // Order matters: the bricks service is useless without the embedder behind it,
    // and failing here gives an actionable message instead of a wall of 503s
    // mid-benchmark.
    await assertAnnServiceReachable(options);
    await ensurePythonService(options, map, job);
  }

  await regenerateGroundTruth(map);

  job.state = "running";
  const runDir = path.join(map.path, "benchmark", job.runId);
  const scriptArgs = benchmarkScriptArgs(options, map, runDir, params);
  const stderrTail: string[] = [];

  let child: ChildProcess | null = null;
  let spawnError = "";
  for (const pythonBin of pythonBinaryCandidates()) {
    const candidate = spawn(pythonBin, scriptArgs, {
      cwd: options.repoRoot,
      env: pythonEnv(options),
      stdio: ["ignore", "pipe", "pipe"],
    });
    const spawned = await new Promise<boolean>((resolve) => {
      candidate.once("spawn", () => resolve(true));
      candidate.once("error", (error) => {
        spawnError = String(error);
        resolve(false);
      });
    });
    if (spawned) {
      child = candidate;
      break;
    }
  }
  if (!child) {
    throw new Error(`Could not spawn the benchmark script: ${spawnError || "no python binary found"}`);
  }
  const proc = child;

  job.child = proc;
  let buffered = "";
  proc.stdout?.setEncoding("utf8");
  proc.stdout?.on("data", (chunk: string) => {
    buffered += chunk;
    const lines = buffered.split("\n");
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) {
        handleProgressLine(job, line);
      }
    }
  });
  proc.stderr?.setEncoding("utf8");
  proc.stderr?.on("data", (chunk: string) => appendTail(stderrTail, chunk));

  const exitCode = await new Promise<number | null>((resolve) => {
    proc.on("close", (code) => {
      if (buffered.trim()) {
        handleProgressLine(job, buffered);
      }
      resolve(code);
    });
  });

  job.child = null;
  const metricsWritten = await pathExists(path.join(runDir, "metrics.json"));
  if (!metricsWritten) {
    throw new Error(
      `Benchmark exited with code ${exitCode} without writing metrics.json`
      + (stderrTail.length ? `:\n${stderrTail.join("\n")}` : ""),
    );
  }
  // Exit code 1 only means some prompts errored; metrics.json exists either way.
  job.state = "done";
}

export function startBenchmarkRun(
  options: WorkbenchOptions,
  map: MapEntry,
  params: BenchmarkRunParams,
): { run_id: string } {
  if (activeJob && (activeJob.state === "ensuring-service" || activeJob.state === "running")) {
    throw new WorkbenchRouteError(
      409,
      `A benchmark run is already in progress for map '${activeJob.mapId}' (run ${activeJob.runId}).`,
    );
  }

  const job: BenchmarkJob = {
    runId: buildRunId(),
    mapId: map.id,
    state: benchmarkTarget(params) === "remote" ? "running" : "ensuring-service",
    serviceState: null,
    progress: { completed: 0, total: null },
    prompts: [],
    summary: null,
    error: null,
    startedAt: new Date().toISOString(),
    child: null,
  };
  activeJob = job;

  void runBenchmarkJob(options, map, job, params).catch((error: unknown) => {
    job.state = "error";
    job.error = error instanceof Error ? error.message : String(error);
    job.child = null;
  });

  return { run_id: job.runId };
}
