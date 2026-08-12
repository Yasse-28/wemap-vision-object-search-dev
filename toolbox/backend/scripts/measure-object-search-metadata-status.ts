/** Measure legacy and columnar metadata status JSON without starting a server. */

import { loadMapEntries } from "../src/config.js";
import {
  legacyObjectSearchMetadataStatusPayloadForComparison,
  objectSearchMetadataStatusPayload,
} from "../src/workbench-index.js";

function formatBytes(bytes: number): string {
  return `${bytes.toLocaleString("en-US")} bytes (${(bytes / 1024).toFixed(1)} KiB)`;
}

async function main(): Promise<void> {
  const [configPath, mapId = "bbhotel-choisy"] = process.argv.slice(2);
  if (!configPath) {
    throw new Error(
      "Usage: npm run measure:metadata-status -- <config> [map-id]",
    );
  }
  const maps = await loadMapEntries(configPath);
  const map = maps.find((entry) => entry.id === mapId);
  if (!map) {
    throw new Error(`Map '${mapId}' is not present in ${configPath}.`);
  }

  const [before, after] = await Promise.all([
    legacyObjectSearchMetadataStatusPayloadForComparison(map),
    objectSearchMetadataStatusPayload(map),
  ]);
  const beforeBytes = Buffer.byteLength(JSON.stringify(before));
  const afterBytes = Buffer.byteLength(JSON.stringify(after));
  const reduction = beforeBytes === 0 ? 0 : (1 - afterBytes / beforeBytes) * 100;

  console.log(`Map: ${map.id}`);
  console.log(`Before (object arrays): ${formatBytes(beforeBytes)}`);
  console.log(`After (columnar arrays): ${formatBytes(afterBytes)}`);
  console.log(
    `Reduction: ${reduction.toFixed(1)}% (${(beforeBytes / afterBytes).toFixed(2)}x smaller)`,
  );
}

await main();
