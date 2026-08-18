/**
 * The two halves of the export that are not python: the config splice and the
 * directory browser's confinement.
 *
 * The config is edited textually rather than re-serialised, because the reader
 * strips comments and a round trip would delete every one of them. That makes the
 * splice worth pinning: it is string surgery on a file the user hand-maintains.
 */

import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { appendMapEntry, loadMapEntries, removeMapEntry } from "./config.js";
import {
  createDirectoryPayload,
  listDirectoriesPayload,
} from "./export-roi.js";
import type { WorkbenchOptions } from "./http-utils.js";
import type { MapEntry } from "./config.js";

// Quoted keys and comments: `parseConfigText` strips the comments, then hands the
// rest to `JSON.parse`, which would reject a bare key. This is the real shape.
const CONFIG_WITH_COMMENTS = `{
  // The maps this toolbox knows about.
  "maps": [
    // The production map, fetched with ../retrieve-map-data.
    { "id": "vinci", "path": "maps/vinci", "emmid": 12345 }, // live
  ],
  "objectSearch": { "minSimilarity": 0.15 }
}
`;

async function scratchConfig(text = CONFIG_WITH_COMMENTS): Promise<string> {
  const dir = await mkdtemp(path.join(tmpdir(), "export-roi-"));
  const configPath = path.join(dir, "config.json5");
  await writeFile(configPath, text, "utf8");
  return configPath;
}

function optionsFor(configPath: string): WorkbenchOptions {
  return {
    configPath,
    pythonApiBaseUrl: "http://127.0.0.1:45678",
    annApiBaseUrl: "http://127.0.0.1:45677",
    host: "127.0.0.1",
    port: 45700,
    uiDistDir: "/nonexistent",
    repoRoot: "/nonexistent",
  };
}

function mapAt(configPath: string, id = "vinci"): MapEntry {
  return {
    id,
    display_name: id,
    path: path.join(path.dirname(configPath), "maps", id),
    emmid: 12345,
    geo_ref_id: 1,
    object_search: null,
    parent_map: null,
  };
}

test("appending a map keeps every comment in the config", async () => {
  const configPath = await scratchConfig();

  const result = await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: path.join(path.dirname(configPath), "maps", "vinci-roi"),
    emmid: 12345,
    parent_map: "vinci",
  });

  assert.equal(result.written, true);
  const text = await readFile(configPath, "utf8");
  assert.ok(text.includes("// The maps this toolbox knows about."));
  assert.ok(
    text.includes("// The production map, fetched with ../retrieve-map-data."),
  );
  assert.ok(text.includes("// live"));

  const entries = await loadMapEntries(configPath);
  assert.deepEqual(
    entries.map((entry) => entry.id),
    ["vinci", "vinci-roi"],
  );
  const exported = entries[1];
  assert.equal(exported.parent_map, "vinci");
  assert.equal(exported.emmid, 12345);
  // A path under the config directory is stored relative, like the hand-written ones.
  assert.ok(text.includes('"path": "maps/vinci-roi"'));
});

/**
 * The shape that actually broke: the last real entry has **no** trailing comma, and
 * two hundred lines of commented-out entries sit between it and the `]`. Sniffing
 * the text before the bracket finds the `}` of a comment, "adds" the comma inside
 * that comment where it does nothing, and leaves two elements with none between
 * them — `Expected ',' or ']' after array element`.
 */
const CONFIG_WITH_COMMENTED_OUT_ENTRIES = `{
  "maps": [
    {
      "id": "vinci",
      "path": "maps/vinci",
      "emmid": 32996
    }

    // {
    //     "id": "retired-1",
    //     "path": "maps/retired/part-1",
    //     "emmid": 23257
    // },
    // {
    //     "id": "retired-2",
    //     "path": "maps/retired/part-2",
    //     "emmid": 23257
    // }
  ]
}
`;

test("appending after commented-out entries keeps the config parseable", async () => {
  const configPath = await scratchConfig(CONFIG_WITH_COMMENTED_OUT_ENTRIES);

  const result = await appendMapEntry(configPath, {
    id: "vinci-zone-1",
    path: path.join(path.dirname(configPath), "maps", "vinci-zone-1"),
    emmid: 32996,
    parent_map: "vinci",
  });

  assert.equal(result.written, true);
  assert.deepEqual(
    (await loadMapEntries(configPath)).map((entry) => entry.id),
    ["vinci", "vinci-zone-1"],
  );
  const text = await readFile(configPath, "utf8");
  // The commented-out entries are untouched — no comma smuggled into a comment.
  assert.ok(text.includes('//     "id": "retired-2",'));
  assert.ok(!text.includes('// },\n    { "id": "vinci-zone-1"'));
});

test("a trailing line comment stays with the entry it describes", async () => {
  const configPath = await scratchConfig();

  await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: path.join(path.dirname(configPath), "maps", "vinci-roi"),
    emmid: null,
    parent_map: "vinci",
  });

  const lines = (await readFile(configPath, "utf8")).split("\n");
  const commentLine = lines.findIndex((line) => line.includes("// live"));
  assert.ok(lines[commentLine].includes('"id": "vinci"'));
  assert.ok(lines[commentLine + 1].includes('"id": "vinci-roi"'));
});

test("a comma is never written into a trailing comment", async () => {
  // The dangerous shape: no separating comma, and a comment on the element's own
  // line. The comma has to land before the comment; appended after it, it is
  // commented out and the array is left with two elements and nothing between them.
  const configPath = await scratchConfig(
    '{\n  "maps": [\n    { "id": "vinci", "path": "maps/vinci" } // the live one\n  ]\n}\n',
  );

  await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: "/tmp/vinci-roi",
    emmid: null,
    parent_map: "vinci",
  });

  assert.deepEqual(
    (await loadMapEntries(configPath)).map((entry) => entry.id),
    ["vinci", "vinci-roi"],
  );
  const text = await readFile(configPath, "utf8");
  assert.ok(text.includes('"maps/vinci" }, // the live one'));
});

test("appending to an empty maps array works", async () => {
  const configPath = await scratchConfig('{\n  "maps": [\n  ]\n}\n');

  const result = await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: "/tmp/vinci-roi",
    emmid: null,
    parent_map: "vinci",
  });

  assert.equal(result.written, true);
  assert.deepEqual(
    (await loadMapEntries(configPath)).map((entry) => entry.id),
    ["vinci-roi"],
  );
});

test("a backup is written before the config is touched", async () => {
  const configPath = await scratchConfig();

  await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: path.join(path.dirname(configPath), "maps", "vinci-roi"),
    emmid: null,
    parent_map: "vinci",
  });

  assert.equal(
    await readFile(`${configPath}.bak`, "utf8"),
    CONFIG_WITH_COMMENTS,
  );
});

test("removing a map leaves the others and their comments intact", async () => {
  const configPath = await scratchConfig();
  await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: path.join(path.dirname(configPath), "maps", "vinci-roi"),
    emmid: null,
    parent_map: "vinci",
  });

  assert.equal(await removeMapEntry(configPath, "vinci-roi"), true);

  const text = await readFile(configPath, "utf8");
  assert.ok(!text.includes("vinci-roi"));
  assert.ok(
    text.includes("// The production map, fetched with ../retrieve-map-data."),
  );
  assert.deepEqual(
    (await loadMapEntries(configPath)).map((entry) => entry.id),
    ["vinci"],
  );
});

test("removing an absent map changes nothing", async () => {
  const configPath = await scratchConfig();

  assert.equal(await removeMapEntry(configPath, "not-there"), false);
  assert.equal(await readFile(configPath, "utf8"), CONFIG_WITH_COMMENTS);
});

test("a config whose maps array cannot be located is not rewritten", async () => {
  const configPath = await scratchConfig('{ "objectSearch": {} }\n');

  const result = await appendMapEntry(configPath, {
    id: "vinci-roi",
    path: "/tmp/vinci-roi",
    emmid: null,
    parent_map: "vinci",
  });

  assert.equal(result.written, false);
  assert.ok(result.snippet.includes("vinci-roi"));
  assert.equal(await readFile(configPath, "utf8"), '{ "objectSearch": {} }\n');
});

test("the directory browser refuses to leave its roots", async () => {
  const configPath = await scratchConfig();
  const options = optionsFor(configPath);
  const map = mapAt(configPath);

  await assert.rejects(
    () => listDirectoriesPayload(options, map, "/etc"),
    /outside the browsable roots/,
  );
  await assert.rejects(
    () =>
      listDirectoriesPayload(
        options,
        map,
        path.join(path.dirname(configPath), "..", ".."),
      ),
    /outside the browsable roots/,
  );
});

test("a new folder must be one path segment inside the roots", async () => {
  const configPath = await scratchConfig();
  const options = optionsFor(configPath);
  const map = mapAt(configPath);
  const root = path.dirname(configPath);

  await assert.rejects(
    () => createDirectoryPayload(options, map, root, "../escape"),
    /single path segment/,
  );

  const created = await createDirectoryPayload(options, map, root, "vinci-roi");
  assert.equal(created.path, path.join(root, "vinci-roi"));

  const listing = await listDirectoriesPayload(options, map, root);
  assert.ok(listing.entries.some((entry) => entry.name === "vinci-roi"));
  // The listing root has no parent to walk up to.
  assert.equal(listing.parent, null);
});
