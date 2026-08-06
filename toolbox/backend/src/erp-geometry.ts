/**
 * ERP ⇄ gnomonic geometry for the v2 object-search metadata.
 *
 * `metadata.parquet` stores one proposal per row as a view centre plus an angular
 * extent (`theta_center`, `phi_center`, `angular_width`, `angular_height`, all
 * radians). This module is the only place that turns those four numbers back into
 * something drawable: an axis-aligned ERP rectangle, an ERP direction for a click
 * inside a re-rendered cutout, and the ffmpeg `v360` arguments that reproduce the
 * cutout itself.
 *
 * It replaces the cubemap algebra (`cubemapFace`, `faceYawPitch`,
 * `cubemapPixelToEquirect`) that the retired SQLite index needed. There is no
 * cubemap in v2 — a parquet row *is* both the cutout and the detection.
 *
 * ## Conventions (copied from the pipeline, not re-derived)
 *
 * `prepare/convention.py::erp_pixel_centers_to_spherical`:
 *
 *     theta = (cx / W) · 2π − π        phi = π/2 − (cy / H) · π
 *
 * and `prepare/proposal_cutouts.py::create_proposal_cutouts`:
 *
 *     angular_width  = w_pix / W · 2π   angular_height = h_pix / H · π
 *
 * so the inverse is exact in ratio units, which is what {@link erpBboxRatios}
 * returns. Note that `angular_width` is an **ERP arc**: it overstates the true
 * angular width by 1/cos φ (12% at φ=37°). That is the upstream's choice, baked
 * into the embeddings — reproduce it, do not "fix" it.
 *
 * ## Precision
 *
 * The stored angles are float16. Near ±π, θ has an ulp of ~0.002 rad — about 2.5 px
 * on an 8192-wide ERP. Reconstructed boxes are good to a few pixels, not exact.
 * Anything user-facing should say "reconstructed".
 */

/** The four canonical columns of one `metadata.parquet` row, in radians. */
export type ErpViewAngles = {
  thetaCenter: number;
  phiCenter: number;
  angularWidth: number;
  angularHeight: number;
};

/** An axis-aligned ERP rectangle in ratio units (0..1 of width / height). */
export type ErpRect = {
  u0: number;
  v0: number;
  u1: number;
  v1: number;
};

const TWO_PI = 2 * Math.PI;

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * The ERP box a proposal was cut from, in ratio units.
 *
 * `u` is returned **unwrapped**: a box straddling the seam yields `u0 < 0` or
 * `u1 > 1`. That is deliberate — a photosphere consumer maps `u` to a yaw, where
 * 1.02 and 0.02 are the same direction, and wrapping there would draw the box the
 * long way round the sphere. Consumers that need pixels inside the image (the
 * ImageMagick overlay, an SVG viewBox) must call {@link erpRectsWrapped}.
 *
 * `v` is clamped: a box centred near a pole would otherwise reach past the top or
 * bottom row, which no ERP raster has.
 */
export function erpBboxRatios(angles: ErpViewAngles): ErpRect {
  const halfWidth = angles.angularWidth / 2;
  const halfHeight = angles.angularHeight / 2;
  return {
    u0: (angles.thetaCenter - halfWidth + Math.PI) / TWO_PI,
    u1: (angles.thetaCenter + halfWidth + Math.PI) / TWO_PI,
    v0: clamp01((Math.PI / 2 - angles.phiCenter - halfHeight) / Math.PI),
    v1: clamp01((Math.PI / 2 - angles.phiCenter + halfHeight) / Math.PI),
  };
}

/**
 * {@link erpBboxRatios} wrapped into `[0, 1]`, split in two when it crosses the seam.
 *
 * Splitting should be rare: the detectors run on an unwrapped raster, so a proposal
 * cannot legitimately straddle u=0. Float16 rounding can still push θ a hair past
 * ±π, which is exactly the case that would otherwise draw a full-width box.
 */
export function erpRectsWrapped(angles: ErpViewAngles): ErpRect[] {
  const rect = erpBboxRatios(angles);
  const width = Math.min(1, rect.u1 - rect.u0);
  if (width <= 0) {
    return [];
  }
  const start = ((rect.u0 % 1) + 1) % 1;
  const end = start + width;
  if (end <= 1) {
    return [{ ...rect, u0: start, u1: end }];
  }
  return [
    { ...rect, u0: start, u1: 1 },
    { ...rect, u0: 0, u1: end - 1 },
  ];
}

/**
 * Angular footprint in square degrees.
 *
 * The v2 replacement for the cubemap era's ERP-pixel `bbox_area`: there is no fixed
 * raster any more, so an area filter has to be expressed on the sphere. Same caveat
 * as `angular_width` — this is arc-based, so it overstates near the poles.
 */
export function angularAreaDeg2(angles: ErpViewAngles): number {
  const scale = 180 / Math.PI;
  return Math.max(0, angles.angularWidth * scale) * Math.max(0, angles.angularHeight * scale);
}

/**
 * Where a point inside a re-rendered cutout lands on the ERP, in ratio units.
 *
 * The exact replacement for `cubemapPixelToEquirect`, and the same rotation as
 * `inference/crop.py::crop_rectilinear_from_erp` — which is itself
 * `gnomonic_projection_batch` with one sample instead of a grid. `yRatio` is
 * negated because the top of the rendered grid is `+φ`.
 *
 * Pass pixel-centre ratios (`(i + 0.5) / size`): the image being clicked was
 * rendered by ffmpeg, whose output pixels are area samples, not grid corners.
 */
export function cutoutRatioToErpUv(
  angles: ErpViewAngles,
  xRatio: number,
  yRatio: number,
): [number, number] {
  const x0 = (2 * clamp01(xRatio) - 1) * Math.tan(angles.angularWidth / 2);
  const y0 = -(2 * clamp01(yRatio) - 1) * Math.tan(angles.angularHeight / 2);
  const norm = Math.hypot(x0, y0, 1);
  const x = x0 / norm;
  const y = y0 / norm;
  const z = 1 / norm;

  const sinT = Math.sin(angles.thetaCenter);
  const cosT = Math.cos(angles.thetaCenter);
  const sinP = Math.sin(angles.phiCenter);
  const cosP = Math.cos(angles.phiCenter);

  const worldX = cosP * sinT * z + cosT * x - sinP * sinT * y;
  const worldY = sinP * z + cosP * y;
  const worldZ = cosP * cosT * z - sinT * x - sinP * cosT * y;

  const theta = Math.atan2(worldX, worldZ);
  const phi = Math.asin(Math.max(-1, Math.min(1, worldY)));
  return [
    (((theta + Math.PI) / TWO_PI) % 1 + 1) % 1,
    clamp01((Math.PI / 2 - phi) / Math.PI),
  ];
}

/**
 * Refuse a source image that is not a 2:1 equirectangular raster.
 *
 * On W = 2H the pipeline's aspect rescale (`fov_y = fov_x / ratio` for wide boxes)
 * is a no-op, so the stored `angular_*` values *are* the FOV ffmpeg must render.
 * Off 2:1 that stops being true and every cutout preview silently shows a slightly
 * wrong region — the failure this guard exists to make loud.
 */
export function assertEquirect2to1(width: number, height: number): void {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error(`Invalid image size ${width}x${height}.`);
  }
  if (width !== 2 * height) {
    throw new Error(
      `Source image is ${width}x${height}, not 2:1 equirectangular. The stored `
      + "angular_width/angular_height are only the rendered FOV on a 2:1 ERP, so a "
      + "cutout rendered from this image would show the wrong region.",
    );
  }
}

/**
 * `build_padding_mask` (`prepare/proposal_cutouts.py:72-86`), on a `size`-square grid.
 *
 * Returns a row-major mask of the pixels the stored thumbnail kept. Note what it does
 * for a wide box: it keeps the central `1/ratio` band of an already-stretched render,
 * i.e. it **crops** rather than padding. Reproducing it is the only way to compare a
 * re-render with the JPEG on disk.
 */
export function paddingMask(angles: ErpViewAngles, width: number, height: number): Uint8Array {
  const ratio = angles.angularWidth / Math.max(angles.angularHeight, 1e-6);
  const mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const baseY = height > 1 ? -1 + (2 * y) / (height - 1) : 0;
    for (let x = 0; x < width; x += 1) {
      const baseX = width > 1 ? -1 + (2 * x) / (width - 1) : 0;
      const keep =
        ratio > 1
          ? Math.abs(baseY) <= 1 / (ratio + 1e-6)
          : Math.abs(baseX) <= ratio;
      mask[y * width + x] = keep ? 1 : 0;
    }
  }
  return mask;
}

/** Degrees, clamped just inside ffmpeg's `v360` limits (it rejects fov >= 180). */
function fovDegrees(radians: number, scale: number): number {
  const degrees = radians * 180 / Math.PI * scale;
  return Math.max(0.05, Math.min(MAX_GNOMONIC_FOV_DEG, degrees));
}

const MAX_GNOMONIC_FOV_DEG = 179;

/**
 * Whether a rectilinear view of this proposal exists at all.
 *
 * A gnomonic projection is undefined at 180°: `tan(fov/2)` diverges, and past it goes
 * negative. Some GroundingDINO proposals span nearly the whole ERP — a `handrail` at
 * `angular_width = 357°` is real, observed data — and for those the *stored thumbnail
 * is itself degenerate*, because `create_proposal_cutouts` computed `tan(3.12) < 0`.
 *
 * So this is not a rendering limitation to clamp around: there is nothing correct to
 * render, and clamping to 179° would quietly show a completely different region. The
 * render route answers 422, and the fidelity test skips these rows because its
 * reference is meaningless for them.
 */
export function isRenderableGnomonic(angles: ErpViewAngles): boolean {
  const limit = MAX_GNOMONIC_FOV_DEG * Math.PI / 180;
  return (
    angles.angularWidth > 0
    && angles.angularHeight > 0
    && angles.angularWidth < limit
    && angles.angularHeight < limit
  );
}

/**
 * The `v360` filter that reproduces (or widens) one proposal's cutout.
 *
 * `baseCutoutPng` already rendered cubemap faces this way; only the source of the
 * angles changes. ffmpeg's default rotation order (`ypr`) matches the pipeline's
 * `R_y(θ)·R_x(φ)`, verified against the real binary with both yaw and pitch non-zero
 * — do **not** set `rorder`, and do not negate either axis.
 *
 * `fovScale > 1` widens the view around the same centre: a context view, which is
 * what an explorer actually wants and what the stored 336px thumbnail cannot give.
 */
export function gnomonicFfmpegFilter(
  angles: ErpViewAngles,
  options: { size: number; fovScale?: number; squareCanvas?: boolean },
): string {
  const scale = options.fovScale && options.fovScale > 0 ? options.fovScale : 1;
  const hFov = fovDegrees(angles.angularWidth, scale);
  const vFov = fovDegrees(angles.angularHeight, scale);
  // Default: keep the proposal's aspect ratio, so the view is undistorted. That is
  // *not* what is on disk — `create_proposal_cutouts` renders the same FOVs onto a
  // square grid and then applies `build_padding_mask`. `squareCanvas` reproduces that
  // stretched pre-mask render, which is what the fidelity test compares against.
  const ratio = options.squareCanvas ? 1 : hFov / vFov;
  const width = ratio >= 1 ? options.size : Math.max(1, Math.round(options.size * ratio));
  const height = ratio >= 1 ? Math.max(1, Math.round(options.size / ratio)) : options.size;
  const yaw = angles.thetaCenter * 180 / Math.PI;
  const pitch = angles.phiCenter * 180 / Math.PI;
  return (
    "v360=input=equirect:output=flat"
    + `:yaw=${yaw}:pitch=${pitch}`
    + `:h_fov=${hFov}:v_fov=${vFov}`
    + `:w=${width}:h=${height}`
  );
}
