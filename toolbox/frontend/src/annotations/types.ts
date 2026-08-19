export type AnnotationType = "point" | "polygon";

export type AnnotationGeometryType = "Point" | "Polygon";

export type AnnotationClass = {
  name: string;
  color: string;
  prompt?: string | null;
  annotationType: AnnotationType;
};

export type AnnotationLabels = {
  synonyms: string[];
  depictions: string[];
  visuallySimilar: string[];
  clutter: string[];
};

export type AnnotationGroundTruth = {
  objectId: string | null;
  extentM: number | null;
  exhaustiveZone: string | null;
  isDepiction: boolean;
  labels: AnnotationLabels;
};

export type AnnotationSource = {
  sourceSchemaVersion?: number;
  keyframeId: string;
  /** Stable identity from `VideoKeyframe.uuid`, inferred from the image asset today. */
  videoKeyframeUuid?: string | null;
  videoKeyframeId?: number | null;
  videoCaptureUuid?: string | null;
  videoCaptureId?: number | null;
  videoCaptureIndex?: number | null;
  frameNumber?: number | null;
  frameTimeS?: number | null;
  mapUuid?: string | null;
  geoRefId?: number | null;
  geoRefVersion?: number | null;
  geoKeyframeId?: number | null;
  imageFilename?: string | null;
  imageStorageKey?: string | null;
  imageSha256?: string | null;
  depthFilename?: string | null;
  depthStorageKey?: string | null;
  resolutionStatus?: "resolved" | "orphaned" | "legacy-unverified" | "ambiguous";
  projection: "erp" | "cutout";
  /**
   * `metadata.parquet` row the click came from, when `projection === "cutout"`.
   *
   * Was `cutoutId`, a standalone-index cutout id. Files written before the v2
   * migration carry `source_cutout_id` instead and are still read — see
   * `geojson.ts`, which keeps the old key on the way in but no longer writes it.
   */
  rowIndex: number | null;
  xRatio: number;
  yRatio: number;
  erpU: number;
  erpV: number;
  depthM: number;
};

export type AnnotationFeature = {
  id: string;
  className: string;
  prompt: string | null;
  classColor: string;
  annotationType: AnnotationType;
  geometryType: AnnotationGeometryType;
  coordinates: number[] | number[][][];
  altitude: number | null;
  level: string | null;
  accuracyM: number;
  source: AnnotationSource | null;
  groundTruth: AnnotationGroundTruth;
};

export type LngLat = {
  latitude: number;
  longitude: number;
};

// ROI region for annotation counting: an ordered ring of polygon vertices
// (not closed — the first vertex is implicitly reconnected to the last).
export type RoiPolygon = LngLat[];

export type FocusTarget = {
  latitude: number;
  longitude: number;
  level?: string | null;
};

export type MapClickEvent = {
  latitude: number;
  longitude: number;
  altitude: number | null;
  level: string | null;
  draftVertices: LngLat[];
  draftClosed: boolean;
  draftTimestamp: number;
};

export type LivemapMarker = {
  id: string;
  latitude: number;
  longitude: number;
  color: string;
  level: string | null;
  radius?: number;
  scaleWithZoom?: boolean;
  strokeColor?: string;
  strokeWidth?: number;
  opacity?: number;
  label?: string;
  textColor?: string;
  textSize?: number;
};

export type LivemapPolygon = {
  id: string;
  coordinates: number[][];
  color: string;
  level: string | null;
  opacity?: number;
  lineOpacity?: number;
  lineWidth?: number;
};

export type AnnotationMode = "point" | "polygon" | "inspect";

export type LivemapSegment = {
  id: string;
  fromLatitude: number;
  fromLongitude: number;
  toLatitude: number;
  toLongitude: number;
  color?: string;
  width?: number;
  opacity?: number;
  level?: string | null;
  interactive?: boolean;
};

export function isHexColor(value: string): boolean {
  return /^#[0-9A-Fa-f]{6}$/.test(value);
}
