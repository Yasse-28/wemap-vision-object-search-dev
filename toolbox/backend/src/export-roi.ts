/**
 * Export a drawn region as a new map, and delete a map that was exported.
 *
 * Both halves are thin: the work is in `toolbox/bricks/export_roi.py` and
 * `toolbox/bricks/delete_map.py`, because the geometry already lives beside the
 * manifest reader and because this backend has no Postgres client. What lives here
 * is the job plumbing (the benchmark runner's shape: 202 + poll), the directory
 * browser the save dialog needs, and the config splice.
 *
 * The directory browser is the one genuinely new capability, and it is confined:
 * only the parent of the source map's directory and the config's own directory are
 * reachable, so a `..` cannot walk the machine.
 */

import { mkdir, readdir } from "node:fs/promises";
import path from "node:path";

import {
  appendMapEntry,
  loadMapEntries,
  removeMapEntry,
  type MapEntry,
} from "./config.js";
import type { WorkbenchOptions } from "./http-utils.js";
import { lastJsonLine, runPython } from "./python-process.js";
import { WorkbenchRouteError } from "./workbench-index.js";

export type RoiInput = {
  /** Open ring of `[lng, lat]` vertices. */
  ring: Array<[number, number]>;
  /** Level value, or null when the map reported no floor. */
  level: number | null;
};

export type ExportJobState = "idle" | "running" | "succeeded" | "failed";

export type ExportJob = {
  jobId: string;
  mapId: string;
  state: ExportJobState;
  step: string;
  detail: string;
  startedAt: string;
  finishedAt: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
  configSnippet: string | null;
  configWritten: boolean;
};

// One export at a time. Two concurrent exports would race on `geo_ref_id`
// allocation — both would read the same maximum and pick the same next value.
let activeExport: ExportJob | null = null;

// ---------------------------------------------------------------------------
// The directory browser
// ---------------------------------------------------------------------------

export type DirectoryEntry = { name: string; path: string };

export type DirectoryListing = {
  path: string;
  parent: string | null;
  roots: string[];
  entries: DirectoryEntry[];
};

/**
 * Directories the save dialog may reach: where maps already live, and where the
 * config sits. Anything else is out of bounds — this route lists the dev
 * machine's filesystem and has no business ranging over it.
 */
function browseRoots(options: WorkbenchOptions, map: MapEntry): string[] {
  const roots = [
    path.dirname(path.resolve(map.path)),
    path.dirname(path.resolve(options.configPath)),
  ];
  return [...new Set(roots)];
}

function assertInsideRoots(target: string, roots: string[]): string {
  const resolved = path.resolve(target);
  const allowed = roots.some(
    (root) => resolved === root || resolved.startsWith(`${root}${path.sep}`),
  );
  if (!allowed) {
    throw new WorkbenchRouteError(
      403,
      `${resolved} is outside the browsable roots (${roots.join(", ")}).`,
    );
  }
  return resolved;
}

export async function listDirectoriesPayload(
  options: WorkbenchOptions,
  map: MapEntry,
  requested: string | null,
): Promise<DirectoryListing> {
  const roots = browseRoots(options, map);
  const target = assertInsideRoots(
    requested && requested.length > 0 ? requested : roots[0],
    roots,
  );

  let names: string[] = [];
  try {
    const dirents = await readdir(target, { withFileTypes: true });
    names = dirents
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
      .map((entry) => entry.name)
      .sort((a, b) => a.localeCompare(b));
  } catch (error) {
    throw new WorkbenchRouteError(
      404,
      `Cannot list ${target}: ${String(error)}`,
    );
  }

  const parent = path.dirname(target);
  return {
    path: target,
    parent: roots.some((root) => target === root) ? null : parent,
    roots,
    entries: names.map((name) => ({ name, path: path.join(target, name) })),
  };
}

export async function createDirectoryPayload(
  options: WorkbenchOptions,
  map: MapEntry,
  parent: string,
  name: string,
): Promise<DirectoryEntry> {
  const trimmed = name.trim();
  if (
    !trimmed ||
    trimmed.includes(path.sep) ||
    trimmed === "." ||
    trimmed === ".."
  ) {
    throw new WorkbenchRouteError(
      400,
      "A folder name must be a single path segment.",
    );
  }
  const roots = browseRoots(options, map);
  const target = assertInsideRoots(path.join(parent, trimmed), roots);
  await mkdir(target, { recursive: true });
  return { name: trimmed, path: target };
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function parseRois(value: unknown): RoiInput[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new WorkbenchRouteError(400, "At least one region is required.");
  }
  return value.map((item, index) => {
    const raw = item as Record<string, unknown>;
    const ring = raw.ring;
    if (!Array.isArray(ring) || ring.length < 3) {
      throw new WorkbenchRouteError(
        400,
        `rois[${index}] needs at least 3 vertices.`,
      );
    }
    return {
      ring: ring.map((point) => {
        const pair = point as [number, number];
        return [Number(pair[0]), Number(pair[1])] as [number, number];
      }),
      level:
        raw.level === null || raw.level === undefined
          ? null
          : Number(raw.level),
    };
  });
}

async function configuredMapPaths(configPath: string): Promise<string[]> {
  return (await loadMapEntries(configPath)).map((entry) => entry.path);
}

async function runExportScript(
  options: WorkbenchOptions,
  spec: Record<string, unknown>,
  dryRun: boolean,
  onLine?: (line: string) => void,
): Promise<Record<string, unknown>> {
  const args = ["-m", "toolbox.bricks.export_roi", "--spec", "-"];
  if (dryRun) {
    args.push("--dry-run");
  }
  const result = await runPython(options, {
    args,
    stdin: JSON.stringify(spec),
    onLine,
  });
  const payload = lastJsonLine(result.stdout) as Record<string, unknown> | null;
  if (result.code !== 0 || !payload) {
    throw new WorkbenchRouteError(
      422,
      lastPythonError(result.stderr) || `Export failed (exit ${result.code}).`,
    );
  }
  return payload;
}

/**
 * The last meaningful line of a python traceback.
 *
 * The whole traceback is useless in a toast and the first line is only
 * "Traceback (most recent call last)", so the exception itself is what surfaces.
 */
function lastPythonError(stderr: string): string {
  const lines = stderr
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines[lines.length - 1] ?? "";
}

export async function exportPreviewPayload(
  options: WorkbenchOptions,
  map: MapEntry,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return runExportScript(
    options,
    {
      source_map_path: map.path,
      rois: parseRois(body.rois),
      configured_map_paths: await configuredMapPaths(options.configPath),
    },
    true,
  );
}

export function exportStatusPayload(mapId: string): { job: ExportJob | null } {
  return {
    job: activeExport && activeExport.mapId === mapId ? activeExport : null,
  };
}

export async function startExportRun(
  options: WorkbenchOptions,
  map: MapEntry,
  body: Record<string, unknown>,
): Promise<ExportJob> {
  if (activeExport?.state === "running") {
    throw new WorkbenchRouteError(409, "An export is already running.");
  }

  const mapId = typeof body.map_id === "string" ? body.map_id.trim() : "";
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(mapId)) {
    throw new WorkbenchRouteError(
      400,
      "The new map id must be a plain slug (letters, digits, dot, dash, underscore).",
    );
  }
  const destination =
    typeof body.destination_path === "string" ? body.destination_path : "";
  if (!destination) {
    throw new WorkbenchRouteError(400, "A destination path is required.");
  }
  assertInsideRoots(destination, browseRoots(options, map));

  const existing = await loadMapEntries(options.configPath);
  if (existing.some((entry) => entry.id === mapId)) {
    throw new WorkbenchRouteError(
      409,
      `A map named '${mapId}' is already configured.`,
    );
  }

  const rois = parseRois(body.rois);
  const includeArtifacts = Boolean(body.include_artifacts);
  const registerInConfig = body.register_in_config !== false;

  const job: ExportJob = {
    jobId: `${Date.now()}`,
    mapId: map.id,
    state: "running",
    step: "start",
    detail: "",
    startedAt: new Date().toISOString(),
    finishedAt: null,
    error: null,
    result: null,
    configSnippet: null,
    configWritten: false,
  };
  activeExport = job;

  void (async () => {
    try {
      const result = await runExportScript(
        options,
        {
          source_map_path: map.path,
          destination_path: destination,
          map_id: mapId,
          parent_map: map.id,
          rois,
          include_artifacts: includeArtifacts,
          configured_map_paths: existing.map((entry) => entry.path),
        },
        false,
        (line) => {
          const match = /^PROGRESS\s+(\S+)\s*(.*)$/.exec(line.trim());
          if (match) {
            job.step = match[1];
            job.detail = match[2];
          }
        },
      );
      job.result = result;

      if (registerInConfig) {
        const written = await appendMapEntry(options.configPath, {
          id: mapId,
          path: String(result.path ?? destination),
          // The sub-map shows on the same livemap as its parent: same venue, same
          // indoor plan. A fresh emmid would have to be created in Wemap first.
          emmid: map.emmid,
          parent_map: map.id,
        });
        job.configWritten = written.written;
        job.configSnippet = written.snippet;
      }

      job.state = "succeeded";
    } catch (error) {
      job.state = "failed";
      job.error = error instanceof Error ? error.message : String(error);
    } finally {
      job.finishedAt = new Date().toISOString();
    }
  })();

  return job;
}

// ---------------------------------------------------------------------------
// Deletion
// ---------------------------------------------------------------------------

async function runDeleteScript(
  options: WorkbenchOptions,
  mapId: string,
  dryRun: boolean,
): Promise<Record<string, unknown>> {
  const configured = (await loadMapEntries(options.configPath)).map(
    (entry) => ({
      id: entry.id,
      path: entry.path,
      parent_map: entry.parent_map,
    }),
  );
  const args = ["-m", "toolbox.bricks.delete_map", "--spec", "-"];
  if (dryRun) {
    args.push("--dry-run");
  }
  const result = await runPython(options, {
    args,
    stdin: JSON.stringify({ map_id: mapId, configured_maps: configured }),
  });
  const payload = lastJsonLine(result.stdout) as Record<string, unknown> | null;
  if (payload && typeof payload.refused === "string") {
    throw new WorkbenchRouteError(409, payload.refused);
  }
  if (result.code !== 0 || !payload) {
    throw new WorkbenchRouteError(
      500,
      lastPythonError(result.stderr) ||
        `Deletion failed (exit ${result.code}).`,
    );
  }
  return payload;
}

export async function deletionPreviewPayload(
  options: WorkbenchOptions,
  map: MapEntry,
): Promise<Record<string, unknown>> {
  return runDeleteScript(options, map.id, true);
}

export async function deleteMapPayload(
  options: WorkbenchOptions,
  map: MapEntry,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  // The typed confirmation is re-checked here: a client that skipped the dialog
  // must not be able to delete a map with an empty request body.
  if (body.confirm_id !== map.id) {
    throw new WorkbenchRouteError(
      400,
      "Deletion requires confirm_id to match the map id exactly.",
    );
  }
  const result = await runDeleteScript(options, map.id, false);
  const removed = await removeMapEntry(options.configPath, map.id);
  return { ...result, config_entry_removed: removed };
}
