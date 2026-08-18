/**
 * Spawning the repo's python from the toolbox backend.
 *
 * `pythonBinaryCandidates` and `pythonEnv` existed in two copies before this module
 * (`benchmark-runner.ts` and `workbench-index.ts`); the export/delete routes would
 * have made a third. The interpreter search order and the PYTHONPATH are a single
 * fact about this repo, so they live in one place.
 */

import { spawn } from "node:child_process";
import path from "node:path";

import type { WorkbenchOptions } from "./http-utils.js";

/**
 * Interpreters to try, in order. The conda environment is named explicitly because
 * the system python has none of the pipeline's dependencies, and a `python3` that
 * imports nothing fails far from the cause.
 */
export function pythonBinaryCandidates(): string[] {
  return [
    process.env.OBJECT_SEARCH_WORKBENCH_PYTHON,
    process.env.PYTHON,
    "/home/yacine/anaconda3/envs/wemap-vision/bin/python",
    "python3",
    "python",
  ].filter((item): item is string => Boolean(item));
}

/**
 * Environment for a spawned python process: the mirrored trees are not a src/
 * layout, so `prepare`/`inference`/`indexing` only import with
 * third_party/object_search on PYTHONPATH — plus the repo root for `toolbox.*`.
 */
export function pythonEnv(options: WorkbenchOptions): NodeJS.ProcessEnv {
  const roots = [
    options.repoRoot,
    path.join(options.repoRoot, "third_party", "object_search"),
  ];
  const existing = process.env.PYTHONPATH;
  return {
    ...process.env,
    PYTHONPATH: existing
      ? [...roots, existing].join(path.delimiter)
      : roots.join(path.delimiter),
  };
}

export type PythonResult = {
  code: number | null;
  stdout: string;
  stderr: string;
};

export type PythonRunOptions = {
  args: string[];
  /** Written to the child's stdin and closed. Used to pass a JSON job spec. */
  stdin?: string;
  /** Called once per complete stdout line, for `PROGRESS …` reporting. */
  onLine?: (line: string) => void;
};

/**
 * Run a python module to completion, returning its streams rather than throwing.
 *
 * Both callers need the exit code *and* stdout: the scripts answer with a JSON
 * document on success and a JSON `{refused}` on a rejected request, so a non-zero
 * exit is data, not only an error.
 */
export async function runPython(
  options: WorkbenchOptions,
  run: PythonRunOptions,
): Promise<PythonResult> {
  let spawnError = "";
  for (const pythonBin of pythonBinaryCandidates()) {
    const child = spawn(pythonBin, run.args, {
      cwd: options.repoRoot,
      env: pythonEnv(options),
      stdio: ["pipe", "pipe", "pipe"],
    });
    const spawned = await new Promise<boolean>((resolve) => {
      child.once("spawn", () => resolve(true));
      child.once("error", (error) => {
        spawnError = String(error);
        resolve(false);
      });
    });
    if (!spawned) {
      continue;
    }

    let stdout = "";
    let stderr = "";
    let buffered = "";
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      stdout += chunk;
      if (!run.onLine) {
        return;
      }
      buffered += chunk;
      const lines = buffered.split("\n");
      buffered = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) {
          run.onLine(line);
        }
      }
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.stdin?.end(run.stdin ?? "");

    const code = await new Promise<number | null>((resolve) => {
      child.on("close", (value) => {
        if (buffered.trim() && run.onLine) {
          run.onLine(buffered);
        }
        resolve(value);
      });
    });
    return { code, stdout, stderr };
  }

  throw new Error(
    `Could not spawn python: ${spawnError || "no interpreter found"}`,
  );
}

/**
 * The last complete JSON object printed on stdout.
 *
 * The scripts interleave `PROGRESS …` lines with one final JSON document, so the
 * result cannot simply be `JSON.parse(stdout)`.
 */
export function lastJsonLine(stdout: string): unknown {
  const lines = stdout
    .split("\n")
    .filter((line) => line.trim().startsWith("{"));
  const last = lines[lines.length - 1];
  if (!last) {
    return null;
  }
  try {
    return JSON.parse(last);
  } catch {
    return null;
  }
}
