export type AnnotationType = "point" | "polygon";

export type AnnotationGeometryType = "Point" | "Polygon";

export type AnnotationClass = {
  name: string;
  color: string;
  prompt?: string | null;
  annotationType: AnnotationType;
};

export type AnnotationSource = {
  keyframeId: string;
  projection: "erp" | "cutout";
  cutoutId: string | null;
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
