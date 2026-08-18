import { readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

export type MapEntry = {
  id: string;
  display_name: string;
  path: string;
  emmid: number | null;
  /**
   * LEGACY: the georef id this map's candidates were ingested under. It is the
   * partition key of `object_search_candidate` and of the partial HNSW index, and
   * it now comes from the map's v2 manifest — the same place `ingest_cli` takes
   * it — so the two cannot disagree. Still parsed for config compatibility;
   * the python service warns when it is set. Defaults to 1.
   */
  geo_ref_id: number;
  /** Raw per-map `objectSearch` settings, passed through to the python service. */
  object_search: Record<string, unknown> | null;
  /**
   * Id of the map this one was exported from, or null for a map received from
   * production. It is what makes a map a *sub-map*: the UI nests it under its parent,
   * and it is the gate on deletion — a map without one was not produced here and this
   * repo cannot rebuild its pixels.
   */
  parent_map: string | null;
};

type RawConfig = {
  maps?: unknown;
  objectSearch?: unknown;
};

function stripJsonComments(input: string): string {
  let output = "";
  let inString = false;
  let quote = "";
  let escaped = false;

  for (let idx = 0; idx < input.length; idx += 1) {
    const current = input[idx];
    const next = input[idx + 1];

    if (inString) {
      output += current;
      if (escaped) {
        escaped = false;
      } else if (current === "\\") {
        escaped = true;
      } else if (current === quote) {
        inString = false;
        quote = "";
      }
      continue;
    }

    if (current === "\"" || current === "'") {
      inString = true;
      quote = current;
      output += current;
      continue;
    }

    if (current === "/" && next === "/") {
      while (idx < input.length && input[idx] !== "\n") {
        idx += 1;
      }
      output += "\n";
      continue;
    }

    if (current === "/" && next === "*") {
      idx += 2;
      while (idx < input.length && !(input[idx] === "*" && input[idx + 1] === "/")) {
        idx += 1;
      }
      idx += 1;
      continue;
    }

    output += current;
  }

  return output;
}

function stripTrailingCommas(input: string): string {
  return input.replace(/,\s*([}\]])/g, "$1");
}

function parseConfigText(text: string): RawConfig {
  return JSON.parse(stripTrailingCommas(stripJsonComments(text))) as RawConfig;
}

function resolveMapPath(configDir: string, mapId: string, value: unknown): string {
  if (typeof value === "string" && value.length > 0) {
    return path.resolve(configDir, value);
  }
  return path.resolve(configDir, "maps", mapId);
}

export async function loadMapEntries(configPath: string): Promise<MapEntry[]> {
  const resolvedConfigPath = path.resolve(configPath);
  const configDir = path.dirname(resolvedConfigPath);
  const data = parseConfigText(await readFile(resolvedConfigPath, "utf8"));
  if (!Array.isArray(data.maps)) {
    throw new Error("Config must contain a maps array.");
  }

  const seen = new Set<string>();
  return data.maps.map((item, index) => {
    if (!item || typeof item !== "object") {
      throw new Error(`maps[${index}] must be an object.`);
    }
    const raw = item as Record<string, unknown>;
    if (typeof raw.id !== "string" || raw.id.length === 0) {
      throw new Error(`maps[${index}] must contain a string id.`);
    }
    if (seen.has(raw.id)) {
      throw new Error(`Duplicate map id: ${raw.id}`);
    }
    seen.add(raw.id);

    const emmid =
      typeof raw.emmid === "number" && Number.isInteger(raw.emmid) ? raw.emmid : null;
    const geoRefId =
      typeof raw.geo_ref_id === "number" && Number.isInteger(raw.geo_ref_id)
        ? raw.geo_ref_id
        : 1;
    const objectSearch =
      raw.objectSearch && typeof raw.objectSearch === "object" && !Array.isArray(raw.objectSearch)
        ? (raw.objectSearch as Record<string, unknown>)
        : null;

    return {
      id: raw.id,
      display_name: raw.id,
      path: resolveMapPath(configDir, raw.id, raw.path),
      emmid,
      geo_ref_id: geoRefId,
      object_search: objectSearch,
      parent_map:
        typeof raw.parent_map === "string" && raw.parent_map.length > 0
          ? raw.parent_map
          : null,
    };
  });
}

/** Top-level `objectSearch` defaults from the config, if present. */
export async function loadGlobalObjectSearch(
  configPath: string,
): Promise<Record<string, unknown> | null> {
  const data = parseConfigText(await readFile(path.resolve(configPath), "utf8"));
  if (data.objectSearch && typeof data.objectSearch === "object" && !Array.isArray(data.objectSearch)) {
    return data.objectSearch as Record<string, unknown>;
  }
  return null;
}

/** One `maps[]` entry as the export writes it. */
export type NewMapEntry = {
  id: string;
  path: string;
  emmid: number | null;
  parent_map: string;
};

/**
 * The `maps` array's bounds in the raw config text.
 *
 * Found by scanning rather than by parsing, because the config is JSON5 with
 * comments and the reader strips them: re-serialising a parsed config would
 * silently discard every comment the user wrote. Scanning keeps the file
 * byte-identical outside the splice.
 */
function findMapsArray(text: string): { open: number; close: number } | null {
  const keyMatch = /(["']?)maps\1\s*:\s*\[/.exec(text);
  if (!keyMatch) {
    return null;
  }
  const open = keyMatch.index + keyMatch[0].length - 1;

  let depth = 0;
  let inString = false;
  let quote = "";
  let escaped = false;
  for (let idx = open; idx < text.length; idx += 1) {
    const current = text[idx];
    const next = text[idx + 1];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (current === "\\") {
        escaped = true;
      } else if (current === quote) {
        inString = false;
      }
      continue;
    }
    if (current === '"' || current === "'") {
      inString = true;
      quote = current;
      continue;
    }
    if (current === "/" && next === "/") {
      while (idx < text.length && text[idx] !== "\n") {
        idx += 1;
      }
      continue;
    }
    if (current === "/" && next === "*") {
      idx += 2;
      while (
        idx < text.length &&
        !(text[idx] === "*" && text[idx + 1] === "/")
      ) {
        idx += 1;
      }
      idx += 1;
      continue;
    }
    if (current === "[" || current === "{") {
      depth += 1;
    } else if (current === "]" || current === "}") {
      depth -= 1;
      if (depth === 0) {
        return { open, close: idx };
      }
    }
  }
  return null;
}

/** Indentation of the last line before `index`, so a splice matches its neighbours. */
function indentBefore(text: string, index: number): string {
  const lineStart = text.lastIndexOf("\n", index - 1) + 1;
  const line = text.slice(lineStart, index);
  return /^\s*/.exec(line)?.[0] ?? "  ";
}

async function writeConfigAtomically(
  configPath: string,
  text: string,
): Promise<void> {
  await writeFile(
    `${configPath}.bak`,
    await readFile(configPath, "utf8"),
    "utf8",
  );
  const temporary = `${configPath}.tmp-${process.pid}`;
  await writeFile(temporary, text, "utf8");
  await rename(temporary, configPath);
}

/**
 * Add one map to the config, preserving comments and formatting.
 *
 * Returns the JSON5 block that was inserted. When the `maps` array cannot be
 * located unambiguously nothing is written and `written` is false — the caller
 * shows the block for the user to paste rather than guessing at the file.
 */
export async function appendMapEntry(
  configPath: string,
  entry: NewMapEntry,
): Promise<{ written: boolean; snippet: string }> {
  const resolvedConfigPath = path.resolve(configPath);
  const configDir = path.dirname(resolvedConfigPath);
  const relative = path.relative(configDir, entry.path);
  // A path under the config directory stays relative, like the hand-written
  // entries; anything else is absolute, because `..` chains are worse than a
  // long path.
  const storedPath =
    relative && !relative.startsWith("..") && !path.isAbsolute(relative)
      ? relative
      : entry.path;

  // Keys are quoted: `parseConfigText` only strips comments and trailing commas
  // before handing the text to `JSON.parse`, which rejects a bare key. The config
  // is JSON-with-comments, not full JSON5, whatever the extension suggests.
  const fields = [
    `"id": ${JSON.stringify(entry.id)}`,
    `"path": ${JSON.stringify(storedPath)}`,
    ...(entry.emmid === null ? [] : [`"emmid": ${entry.emmid}`]),
    `"parent_map": ${JSON.stringify(entry.parent_map)}`,
  ];

  const text = await readFile(resolvedConfigPath, "utf8");
  const bounds = findMapsArray(text);
  const indent = bounds ? indentBefore(text, bounds.close) : "  ";
  const snippet = `${indent}  { ${fields.join(", ")} },`;

  if (!bounds) {
    return { written: false, snippet: snippet.trim() };
  }

  // The insertion point is right after the last real element, not just before the
  // closing bracket. Real configs park whole commented-out entries between the two,
  // and sniffing the text before `]` for a trailing `}` reads the `}` inside a
  // comment: it then "adds" the separating comma inside that comment, where it does
  // nothing, and the array is left with two elements and no comma between them.
  const body = text.slice(bounds.open + 1, bounds.close);
  const elements = splitTopLevelObjects(body);
  const { commaOffset, snippetOffset } = insertionPointAfterLastElement(
    text,
    bounds,
    elements,
  );

  // Two offsets, because they are not always the same place: the separating comma
  // belongs immediately after the previous element's `}`, while the new entry goes
  // after any comment trailing on that line. Writing the comma at the later offset
  // would put it inside that comment, where it separates nothing.
  const commaAt = commaOffset ?? snippetOffset;
  const spliced =
    text.slice(0, commaAt) +
    (commaOffset === null ? "" : ",") +
    text.slice(commaAt, snippetOffset) +
    `\n${snippet}` +
    text.slice(snippetOffset);

  await writeConfigAtomically(resolvedConfigPath, spliced);
  return { written: true, snippet: snippet.trim() };
}

/**
 * Where a new element goes, and where its separating comma goes.
 *
 * `commaOffset` is null when a comma already follows the last element. It is
 * reported separately from `snippetOffset` because a comment trailing on the
 * element's own line pushes the new entry to the next line, while the comma must
 * stay in front of that comment — appended after it, it would be commented out and
 * separate nothing. An empty array needs neither offset distinguished.
 */
function insertionPointAfterLastElement(
  text: string,
  bounds: { open: number; close: number },
  elements: Array<{ start: number; end: number }>,
): { commaOffset: number | null; snippetOffset: number } {
  const last = elements[elements.length - 1];
  if (!last) {
    return { commaOffset: null, snippetOffset: bounds.open + 1 };
  }

  const elementEnd = bounds.open + 1 + last.end;
  let offset = elementEnd;
  while (offset < bounds.close && /[ \t]/.test(text[offset])) {
    offset += 1;
  }

  let commaOffset: number | null = elementEnd;
  if (text[offset] === ",") {
    offset += 1;
    commaOffset = null;
  }

  // A `//` comment trailing on this line describes the previous entry, so the new
  // one starts after it rather than in front of it.
  const lineEnd = text.indexOf("\n", offset);
  const rest = text.slice(offset, lineEnd === -1 ? bounds.close : lineEnd);
  const snippetOffset = /^[ \t]*\/\//.test(rest)
    ? lineEnd === -1
      ? bounds.close
      : lineEnd
    : offset;

  return { commaOffset, snippetOffset };
}

/**
 * Remove one map from the config by id, preserving comments and formatting.
 *
 * The entry is located by scanning the objects inside `maps` for a matching `id`,
 * again without re-serialising. Returns false when no entry matched.
 */
export async function removeMapEntry(
  configPath: string,
  mapId: string,
): Promise<boolean> {
  const resolvedConfigPath = path.resolve(configPath);
  const text = await readFile(resolvedConfigPath, "utf8");
  const bounds = findMapsArray(text);
  if (!bounds) {
    return false;
  }

  const body = text.slice(bounds.open + 1, bounds.close);
  const objects = splitTopLevelObjects(body);
  const target = objects.find((item) => {
    const idMatch = /(["']?)id\1\s*:\s*(["'])(.*?)\2/.exec(
      body.slice(item.start, item.end),
    );
    return idMatch?.[3] === mapId;
  });
  if (!target) {
    return false;
  }

  // Swallow the separating comma and the blank line the entry occupied, so the
  // array does not accumulate holes across deletions.
  let from = bounds.open + 1 + target.start;
  let to = bounds.open + 1 + target.end;
  while (to < text.length && /[\s,]/.test(text[to])) {
    if (text[to] === "\n") {
      to += 1;
      break;
    }
    to += 1;
  }
  while (from > 0 && /[ \t]/.test(text[from - 1])) {
    from -= 1;
  }

  await writeConfigAtomically(
    resolvedConfigPath,
    text.slice(0, from) + text.slice(to),
  );
  return true;
}

/** Offsets of each top-level `{…}` inside an array body, comments and strings aside. */
function splitTopLevelObjects(
  body: string,
): Array<{ start: number; end: number }> {
  const spans: Array<{ start: number; end: number }> = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let quote = "";
  let escaped = false;

  for (let idx = 0; idx < body.length; idx += 1) {
    const current = body[idx];
    const next = body[idx + 1];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (current === "\\") {
        escaped = true;
      } else if (current === quote) {
        inString = false;
      }
      continue;
    }
    if (current === '"' || current === "'") {
      inString = true;
      quote = current;
      continue;
    }
    if (current === "/" && next === "/") {
      while (idx < body.length && body[idx] !== "\n") {
        idx += 1;
      }
      continue;
    }
    if (current === "/" && next === "*") {
      idx += 2;
      while (
        idx < body.length &&
        !(body[idx] === "*" && body[idx + 1] === "/")
      ) {
        idx += 1;
      }
      idx += 1;
      continue;
    }
    if (current === "{") {
      if (depth === 0) {
        start = idx;
      }
      depth += 1;
    } else if (current === "}") {
      depth -= 1;
      if (depth === 0 && start >= 0) {
        spans.push({ start, end: idx + 1 });
        start = -1;
      }
    }
  }
  return spans;
}
