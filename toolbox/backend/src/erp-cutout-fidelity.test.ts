/**
 * The test that carries the whole geometry chain.
 *
 * The thumbnails on disk **are** the reference render: `prepare` wrote them from the
 * same ERP with the same projection, and they are what MetaCLIP2 embedded. So for a
 * sample of rows this re-renders from the map's ERP with the production code path
 * (`loadMetadata` → `gnomonicFfmpegFilter` → ffmpeg), applies `paddingMask`, and
 * cross-correlates against the JPEG.
 *
 * That single pass/fail covers float16 decode, radians→degrees, the rotation order,
 * axis signs, image resolution and the mask semantics at once. Nothing else in this
 * migration has that leverage — a sign error anywhere shows up as a correlation near
 * zero rather than as a plausible-looking preview of the wrong wall.
 *
 * It needs a real prepared map, so it is opt-in:
 *
 *     OBJECT_SEARCH_TEST_MAP=/path/to/map npm test -w backend
 *
 * ## Thresholds, and why they are what they are
 *
 * Two of them, because one number cannot separate "noisy" from "wrong". Measured on
 * `bbhotel-choisy` (5760x2880 ERP, 224px thumbnails, 40 rows spread over 181919):
 *
 * - correct geometry: 39 rows in 0.93–0.997, one at 0.87 (a small low-texture crop
 *   upsampled ~1.5x, whose correlation still peaks exactly at zero offset);
 * - a 0.25° yaw or pitch error: ~0.63–0.70;
 * - a 0.5° error: ~0.46–0.51;
 * - a genuinely degenerate render: ~0.04.
 *
 * So a **per-row floor of 0.8** sits in the empty band between resampling noise and
 * the smallest misalignment worth catching, and a **median of 0.93** stops a general
 * degradation from hiding behind that floor. Tuning either one to make a red test
 * pass defeats the purpose — investigate the row instead, starting with whether its
 * correlation peaks off-centre.
 *
 * Optional: `OBJECT_SEARCH_TEST_MAP_SAMPLES` (default 12),
 * `OBJECT_SEARCH_TEST_MAP_MIN_NCC` (default 0.8),
 * `OBJECT_SEARCH_TEST_MAP_MIN_MEDIAN_NCC` (default 0.93).
 */

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";

import {
  assertEquirect2to1,
  gnomonicFfmpegFilter,
  isRenderableGnomonic,
  paddingMask,
} from "./erp-geometry.js";
import { loadMapManifest } from "./map-manifest.js";
import { requireMetadata, type MetadataRow } from "./object-search-metadata.js";

const execFileAsync = promisify(execFile);

const MAP_PATH = process.env.OBJECT_SEARCH_TEST_MAP ?? "";
const SAMPLES = Number(process.env.OBJECT_SEARCH_TEST_MAP_SAMPLES ?? 12);
const MIN_NCC = Number(process.env.OBJECT_SEARCH_TEST_MAP_MIN_NCC ?? 0.8);
const MIN_MEDIAN_NCC = Number(process.env.OBJECT_SEARCH_TEST_MAP_MIN_MEDIAN_NCC ?? 0.93);

async function imageSize(imagePath: string): Promise<{ width: number; height: number }> {
  const { stdout } = await execFileAsync("identify", ["-format", "%w %h", imagePath]);
  const [width, height] = stdout.trim().split(/\s+/).map(Number);
  return { width, height };
}

/** Raw 8-bit grayscale bytes, so the two images can be compared pixel by pixel. */
async function grayFromJpeg(imagePath: string): Promise<Uint8Array> {
  const { stdout } = await execFileAsync(
    "convert",
    [imagePath, "-colorspace", "Gray", "-depth", "8", "gray:-"],
    { encoding: "buffer", maxBuffer: 1 << 26 },
  );
  return new Uint8Array(stdout as unknown as Buffer);
}

async function grayFromErp(
  erpPath: string,
  row: MetadataRow,
  size: number,
): Promise<Uint8Array> {
  const filter = gnomonicFfmpegFilter(
    {
      thetaCenter: row.thetaCenter,
      phiCenter: row.phiCenter,
      angularWidth: row.angularWidth,
      angularHeight: row.angularHeight,
    },
    { size, squareCanvas: true },
  );
  const { stdout } = await execFileAsync(
    "ffmpeg",
    [
      "-hide_banner", "-loglevel", "error",
      "-i", erpPath,
      "-vf", `${filter},format=gray`,
      "-frames:v", "1",
      "-f", "rawvideo",
      "-pix_fmt", "gray",
      "-",
    ],
    { encoding: "buffer", maxBuffer: 1 << 26 },
  );
  return new Uint8Array(stdout as unknown as Buffer);
}

/** Normalised cross-correlation over the mask's kept pixels. */
function maskedNcc(a: Uint8Array, b: Uint8Array, mask: Uint8Array): number {
  let count = 0;
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i]) {
      count += 1;
      sumA += a[i];
      sumB += b[i];
    }
  }
  if (count < 16) {
    return Number.NaN;
  }
  const meanA = sumA / count;
  const meanB = sumB / count;
  let numerator = 0;
  let varA = 0;
  let varB = 0;
  for (let i = 0; i < mask.length; i += 1) {
    if (!mask[i]) {
      continue;
    }
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    numerator += da * db;
    varA += da * da;
    varB += db * db;
  }
  return varA > 0 && varB > 0 ? numerator / Math.sqrt(varA * varB) : Number.NaN;
}

test(
  "re-rendered cutouts match the stored thumbnails",
  { skip: MAP_PATH ? false : "set OBJECT_SEARCH_TEST_MAP to a prepared map directory" },
  async (t) => {
    const map = {
      id: "fidelity",
      display_name: "fidelity",
      path: path.resolve(MAP_PATH),
      emmid: null,
      geo_ref_id: 1,
      object_search: null,
    };
    const metadata = await requireMetadata(map);
    const manifest = await loadMapManifest(map.path);
    assert.ok(metadata.postprocessed, "map must be post-processed (needs thumbnail_key)");

    // Deterministic spread over the file rather than a random sample: a flaky
    // geometry test is worse than no geometry test.
    const picked: MetadataRow[] = [];
    let skippedDegenerate = 0;
    for (let i = 0; i < SAMPLES; i += 1) {
      const row = metadata.rows[Math.floor(((i + 0.5) * metadata.rows.length) / SAMPLES)];
      if (!row?.thumbnailKey) {
        continue;
      }
      // A proposal spanning >= 180 degrees has no rectilinear view, so the thumbnail
      // on disk is not a valid reference either — skipping it is not leniency, it is
      // the absence of anything to compare with. Counted so it is never silent.
      if (
        !isRenderableGnomonic({
          thetaCenter: row.thetaCenter,
          phiCenter: row.phiCenter,
          angularWidth: row.angularWidth,
          angularHeight: row.angularHeight,
        })
      ) {
        skippedDegenerate += 1;
        continue;
      }
      picked.push(row);
    }
    assert.ok(picked.length > 0, "no rows with a stored thumbnail");
    if (skippedDegenerate) {
      t.diagnostic(`skipped ${skippedDegenerate} row(s) spanning >= 180 degrees`);
    }

    const scores: number[] = [];
    for (const row of picked) {
      const imageFilename = manifest.keyframeById.get(row.videoKeyframeId)?.imageFilename;
      assert.ok(imageFilename, `keyframe ${row.videoKeyframeId} is not in the manifest`);
      const erpPath = path.join(map.path, "images", imageFilename);
      const erpSize = await imageSize(erpPath);
      assertEquirect2to1(erpSize.width, erpSize.height);

      const thumbnailPath = path.join(map.path, row.thumbnailKey!);
      const thumbnailSize = await imageSize(thumbnailPath);
      const [stored, rendered] = await Promise.all([
        grayFromJpeg(thumbnailPath),
        grayFromErp(erpPath, row, thumbnailSize.width),
      ]);
      assert.equal(rendered.length, stored.length, `size mismatch for row ${row.rowIndex}`);

      const mask = paddingMask(
        {
          thetaCenter: row.thetaCenter,
          phiCenter: row.phiCenter,
          angularWidth: row.angularWidth,
          angularHeight: row.angularHeight,
        },
        thumbnailSize.width,
        thumbnailSize.height,
      );
      const score = maskedNcc(rendered, stored, mask);
      scores.push(score);
      t.diagnostic(
        `row ${row.rowIndex} keyframe ${row.videoKeyframeId} ncc ${score.toFixed(4)}`,
      );
      assert.ok(
        score >= MIN_NCC,
        `row ${row.rowIndex}: ncc ${score.toFixed(4)} < ${MIN_NCC}. The re-rendered view `
        + "does not match the stored thumbnail — suspect a sign flip, a degree/radian "
        + "confusion, or the wrong source image.",
      );
    }
    const sorted = [...scores].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    t.diagnostic(
      `worst ncc ${sorted[0].toFixed(4)}, median ${median.toFixed(4)} over ${scores.length} rows`,
    );
    assert.ok(
      median >= MIN_MEDIAN_NCC,
      `median ncc ${median.toFixed(4)} < ${MIN_MEDIAN_NCC}: the whole sample degraded, `
      + "which is a geometry change rather than one awkward crop.",
    );
  },
);
