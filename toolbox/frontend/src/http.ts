export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "ApiError";
  }
}

export async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

export async function fetchJson(
  path: string,
  options?: RequestInit,
): Promise<unknown> {
  const response = await fetch(path, {
    ...options,
    headers: {
      Accept: "application/json",
      ...options?.headers,
    },
  });
  const data = await readJson(response);
  if (!response.ok) {
    const detail =
      data == null
        ? "empty response (is the object-search API running?)"
        : typeof data === "string"
          ? data
          : JSON.stringify(data);
    throw new ApiError(response.status, detail);
  }
  return data;
}
