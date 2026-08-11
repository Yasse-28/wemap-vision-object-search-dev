import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type WorkbenchOptions = {
  configPath: string;
  /**
   * The bricks service (`toolbox.bricks.service`) — the dev-only stand-in for
   * Django's object-search API. It serves `/{map_id}/object-search/localize`, so it
   * keeps port 45678 and the path shape the standalone service used before it.
   */
  pythonApiBaseUrl: string;
  /**
   * The MIRRORED online service (`services/object_search_online`), i.e. production's
   * GPU service. Answers `/object-search/by-text|by-image` with a flat
   * `[{id, similarity}]` list — no coordinates. The bricks service calls it; the
   * toolbox only health-checks it, and never spawns it (it loads MetaCLIP on the GPU,
   * which is not something to start implicitly behind a button press).
   */
  annApiBaseUrl: string;
  host: string;
  port: number;
  uiDistDir: string;
  /** Repo root: both python services are spawned from here, with PYTHONPATH set. */
  repoRoot: string;
};

const contentTypes: Record<string, string> = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

export function repoRootFromModule(metaUrl: string): string {
  return path.resolve(path.dirname(fileURLToPath(metaUrl)), "..", "..");
}

export function sendJson(response: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
  });
  response.end(payload);
}

export function sendText(response: ServerResponse, status: number, body: string): void {
  response.writeHead(status, {
    "content-type": "text/plain; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  response.end(body);
}

export function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

export function parseArgs(argv: string[], repoRoot: string): WorkbenchOptions {
  const options: WorkbenchOptions = {
    configPath: process.env.OBJECT_SEARCH_WORKBENCH_CONFIG ?? "",
    pythonApiBaseUrl: normalizeBaseUrl(
      process.env.OBJECT_SEARCH_PYTHON_API ?? "http://127.0.0.1:45678",
    ),
    annApiBaseUrl: normalizeBaseUrl(
      process.env.OBJECT_SEARCH_ANN_URL ?? "http://127.0.0.1:45677",
    ),
    host: process.env.OBJECT_SEARCH_WORKBENCH_HOST ?? "0.0.0.0",
    port: Number(process.env.OBJECT_SEARCH_WORKBENCH_PORT ?? 45700),
    uiDistDir:
      process.env.OBJECT_SEARCH_WORKBENCH_UI_DIST ??
      path.resolve(repoRoot, "frontend", "dist"),
    // toolbox/backend → repo root is two levels up.
    repoRoot:
      process.env.OBJECT_SEARCH_REPO_ROOT ?? path.resolve(repoRoot, ".."),
  };

  for (let idx = 0; idx < argv.length; idx += 1) {
    const arg = argv[idx];
    const next = argv[idx + 1];
    if (arg === "--config" && next) {
      options.configPath = next;
      idx += 1;
    } else if (arg === "--python-api" && next) {
      options.pythonApiBaseUrl = normalizeBaseUrl(next);
      idx += 1;
    } else if (arg === "--ann-api" && next) {
      options.annApiBaseUrl = normalizeBaseUrl(next);
      idx += 1;
    } else if (arg === "--host" && next) {
      options.host = next;
      idx += 1;
    } else if (arg === "--port" && next) {
      options.port = Number(next);
      idx += 1;
    } else if (arg === "--ui-dist" && next) {
      options.uiDistDir = next;
      idx += 1;
    } else if (arg === "--repo-root" && next) {
      options.repoRoot = next;
      idx += 1;
    }
  }

  if (!options.configPath) {
    throw new Error("Missing --config or OBJECT_SEARCH_WORKBENCH_CONFIG.");
  }
  if (!Number.isInteger(options.port) || options.port <= 0) {
    throw new Error(`Invalid port: ${options.port}`);
  }

  options.configPath = path.resolve(options.configPath);
  options.uiDistDir = path.resolve(options.uiDistDir);
  options.repoRoot = path.resolve(options.repoRoot);
  return options;
}

export async function readRequestBody(request: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

export function requestHeadersForProxy(request: IncomingMessage): Headers {
  const headers = new Headers();
  for (const [name, value] of Object.entries(request.headers)) {
    if (value == null) {
      continue;
    }
    if (name.toLowerCase() === "host" || name.toLowerCase() === "content-length") {
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        headers.append(name, item);
      }
    } else {
      headers.set(name, value);
    }
  }
  return headers;
}

export async function proxyRequest(
  request: IncomingMessage,
  response: ServerResponse,
  targetBaseUrl: string,
  url: URL,
): Promise<void> {
  const target = `${targetBaseUrl}${url.pathname}${url.search}`;
  const method = request.method ?? "GET";
  const hasBody = !["GET", "HEAD"].includes(method);
  const rawBody = hasBody ? await readRequestBody(request) : undefined;
  const body =
    rawBody == null
      ? undefined
      : rawBody.buffer.slice(
          rawBody.byteOffset,
          rawBody.byteOffset + rawBody.byteLength,
        ) as ArrayBuffer;
  const proxied = await fetch(target, {
    method,
    headers: requestHeadersForProxy(request),
    body,
  });

  response.statusCode = proxied.status;
  proxied.headers.forEach((value, key) => {
    if (key.toLowerCase() !== "content-encoding") {
      response.setHeader(key, value);
    }
  });
  response.end(Buffer.from(await proxied.arrayBuffer()));
}

export async function serveStaticUi(
  response: ServerResponse,
  uiDistDir: string,
  url: URL,
): Promise<boolean> {
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === "/") {
    response.writeHead(302, { location: "/ui/" });
    response.end();
    return true;
  }
  if (!pathname.startsWith("/ui")) {
    return false;
  }

  pathname = pathname.replace(/^\/ui\/?/, "");
  const relativePath = pathname || "index.html";
  const candidate = path.resolve(uiDistDir, relativePath);
  const rootWithSeparator = `${uiDistDir}${path.sep}`;
  const filePath = candidate.startsWith(rootWithSeparator) ? candidate : path.join(uiDistDir, "index.html");

  try {
    const info = await stat(filePath);
    if (!info.isFile()) {
      return await serveIndex(response, uiDistDir);
    }
    response.writeHead(200, {
      "content-type": contentTypes[path.extname(filePath)] ?? "application/octet-stream",
      "content-length": info.size,
    });
    createReadStream(filePath).pipe(response);
    return true;
  } catch {
    return await serveIndex(response, uiDistDir);
  }
}

async function serveIndex(response: ServerResponse, uiDistDir: string): Promise<boolean> {
  const indexPath = path.join(uiDistDir, "index.html");
  try {
    const info = await stat(indexPath);
    response.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "content-length": info.size,
    });
    createReadStream(indexPath).pipe(response);
    return true;
  } catch {
    sendText(response, 404, "Object-search UI build not found. Run npm run build from object-search-toolbox.");
    return true;
  }
}
