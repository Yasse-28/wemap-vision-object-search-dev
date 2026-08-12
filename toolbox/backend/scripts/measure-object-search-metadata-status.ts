/** Measure the split metadata payloads without starting a server. */

import { loadMapEntries } from "../src/config.js";
import {
  metadataKeyframesPayload,
  objectSearchMetadataMarkersPayload,
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

  const [status, markers, keyframePage] = await Promise.all([
    objectSearchMetadataStatusPayload(map),
    objectSearchMetadataMarkersPayload(map),
    metadataKeyframesPayload(map, {
      offset: 0,
      limit: 1,
      sort: "parquet",
      includeEmpty: true,
      keyframeId: null,
    }),
  ]);
  const statusBytes = Buffer.byteLength(JSON.stringify(status));
  const markerBytes = Buffer.byteLength(JSON.stringify(markers));
  const keyframePageBytes = Buffer.byteLength(JSON.stringify(keyframePage));

  console.log(`Map: ${map.id}`);
  console.log(`Status: ${formatBytes(statusBytes)}`);
  console.log(`Markers: ${formatBytes(markerBytes)}`);
  console.log(`Keyframe page (limit 1): ${formatBytes(keyframePageBytes)}`);
}

await main();
