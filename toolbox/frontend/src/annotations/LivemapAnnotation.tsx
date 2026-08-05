import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import {
  acquireLivemapHost,
  createLivemapState,
  getLivemapCallbacks,
  type GeoJsonFeature,
  type LivemapCone,
  type LivemapHost,
  type LivemapHostCallbacks,
  type LivemapState,
  type MapboxFeature,
  type MapboxMap,
  type MapboxPointerEvent,
  type WemapPrivateInterface,
  releaseLivemapHost,
  resetConsumerState,
} from "./livemapHost";
import {
  AnnotationMode,
  FocusTarget,
  LivemapMarker,
  LivemapPolygon,
  LivemapSegment,
  LngLat,
  MapClickEvent,
  RoiPolygon,
} from "./types";

export type { LivemapCone };


const POLYGON_SOURCE_ID = "annotation-polygons";
const POINT_SOURCE_ID = "annotation-points";
const DRAFT_SOURCE_ID = "annotation-draft";
const POINT_CIRCLE_LAYER_ID = "annotation-points-circle";
const POINT_LABEL_LAYER_ID = "annotation-points-label";
const POLYGON_FILL_LAYER_ID = "annotation-polygons-fill";
const POLYGON_LINE_LAYER_ID = "annotation-polygons-line";
const DRAFT_FILL_LAYER_ID = "annotation-draft-fill";
const DRAFT_LINE_LAYER_ID = "annotation-draft-line";
const DRAFT_POINT_LAYER_ID = "annotation-draft-points";
const SEGMENT_SOURCE_ID = "annotation-segments";
const SEGMENT_LINE_LAYER_ID = "annotation-segments-line";
const SEGMENT_HIT_LAYER_ID = "annotation-segments-hit";
const CONE_SOURCE_ID = "annotation-cone";
const CONE_LAYER_ID = "annotation-cone-layer";
const CONE_ICON_ID = "annotation-cone-icon";
const CONE_ICON_SIZE = 128;
const ROI_SOURCE_ID = "annotation-roi";
const ROI_FILL_LAYER_ID = "annotation-roi-fill";
const ROI_LINE_LAYER_ID = "annotation-roi-line";
const ROI_POINT_LAYER_ID = "annotation-roi-points";
const ROI_COLOR = "#2563eb";
// Screen-pixel radius around the first vertex within which a click closes the
// polygon (instead of adding another vertex).
const ROI_CLOSE_PX = 14;

function ensureConeIcon(map: MapboxMap, fovDeg: number, color: string): void {
  if (map.hasImage(CONE_ICON_ID)) return;
  const canvas = document.createElement("canvas");
  canvas.width = CONE_ICON_SIZE;
  canvas.height = CONE_ICON_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const cx = CONE_ICON_SIZE / 2;
  const cy = CONE_ICON_SIZE / 2;
  const r = CONE_ICON_SIZE / 2 - 4;
  const halfFov = (fovDeg * Math.PI) / 180 / 2;
  // Parse hex color for fill/stroke with alpha
  ctx.fillStyle = color + "59"; // ~35% opacity
  ctx.strokeStyle = color + "e6"; // ~90% opacity
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, -Math.PI / 2 - halfFov, -Math.PI / 2 + halfFov);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  map.addImage(
    CONE_ICON_ID,
    ctx.getImageData(0, 0, CONE_ICON_SIZE, CONE_ICON_SIZE),
    { pixelRatio: 2 },
  );
}

function distanceMeters(a: LngLat, b: LngLat): number {
  const latScale = 111_320.0;
  const lonScale =
    111_320.0 * Math.cos((((a.latitude + b.latitude) / 2.0) * Math.PI) / 180);
  const dx = (b.longitude - a.longitude) * lonScale;
  const dy = (b.latitude - a.latitude) * latScale;
  return Math.hypot(dx, dy);
}

function normalizeLevel(value: unknown): string | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return normalizeLevel(
      obj.level ??
        obj.id ??
        obj.identifier ??
        obj.name ??
        obj.label ??
        obj.value,
    );
  }
  return String(value);
}

function deriveLevelFromFloor(floor: unknown): string | null {
  if (!floor || typeof floor !== "object") {
    return null;
  }
  const obj = floor as Record<string, unknown>;
  return normalizeLevel(obj.id ?? obj.level ?? obj.identifier ?? obj.name);
}

function deriveAltitudeFromFloor(floor: unknown): number | null {
  if (!floor || typeof floor !== "object") {
    return null;
  }
  const obj = floor as Record<string, unknown>;
  const min = obj.minaltitude;
  const max = obj.maxaltitude;
  if (typeof min === "number" && typeof max === "number") {
    return min;
  }
  if (typeof min === "number") return min;
  if (typeof max === "number") return max;
  for (const key of [
    "altitude",
    "height",
    "floorHeight",
    "floorHeightFromGround",
    "heightFromGround",
  ]) {
    const value = obj[key];
    if (typeof value === "number") {
      return value;
    }
  }
  return null;
}

function getLevelProperties(level: string | null | undefined): Record<string, string> {
  if (level === undefined || level === null || level === "") {
    return {};
  }
  const asNumber = Number(level);
  if (!Number.isNaN(asNumber)) {
    return { level: String(asNumber) };
  }
  return { level: String(level) };
}

function tupleCoordinates(coordinates: number[][]): Array<[number, number]> {
  return coordinates.map((coordinate) => [coordinate[0] ?? 0, coordinate[1] ?? 0]);
}

function nearestMarker(
  position: LngLat & { level: string | null },
  markers: LivemapMarker[],
): LivemapMarker | null {
  let nearest: LivemapMarker | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const marker of markers) {
    if (position.level !== null && marker.level !== position.level) {
      continue;
    }
    const distance = distanceMeters(position, marker);
    if (distance < nearestDistance) {
      nearest = marker;
      nearestDistance = distance;
    }
  }
  return nearest;
}

export type LivemapAnnotationProps = {
  emmid: number;
  markers: LivemapMarker[];
  segments?: LivemapSegment[];
  polygons: LivemapPolygon[];
  draftVertices: LngLat[];
  draftClosed: boolean;
  pendingPoint: LngLat | null;
  activeColor: string | null;
  mode: AnnotationMode;
  focusTarget: FocusTarget | null;
  onMapClick: (event: MapClickEvent) => void;
  onMarkerClick?: (markerId: string) => void;
  onMarkerContextMenu?: (
    markerId: string,
    position: { x: number; y: number },
  ) => void;
  onSegmentHover?: (position: LngLat & { level: string | null }) => void;
  onSegmentClick?: (position: LngLat & { level: string | null }) => void;
  onSegmentLeave?: () => void;
  segmentHoverMarkers?: LivemapMarker[];
  onSegmentMarkerClick?: (markerId: string) => void;
  markerPopover?: {
    latitude: number;
    longitude: number;
    content: ReactNode;
  } | null;
  onStatus?: (status: string) => void;
  onLevelChange?: (level: string | null) => void;
  height?: number | string;
  cone?: LivemapCone | null;
  coneColor?: string;
  roiActive?: boolean;
  roiPolygon?: RoiPolygon | null;
  onRoiChange?: (polygon: RoiPolygon | null) => void;
};

function LivemapAnnotation(props: LivemapAnnotationProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapElRef = useRef<HTMLDivElement | null>(null);
  const statusElRef = useRef<HTMLDivElement | null>(null);
  const hostRef = useRef<LivemapHost | null>(null);
  // Points at the host state once acquired. Handlers registered on the shared
  // livemap read it through this ref, so it must never be reset on unmount.
  const stateRef = useRef<LivemapState>(createLivemapState());
  // Shared with every component instance mounting the same livemap: handlers
  // registered once on the map dispatch through it to the mounted panel.
  const callbacks = useMemo(
    () => getLivemapCallbacks(props.emmid),
    [props.emmid],
  );
  const callbacksRef = useRef<LivemapHostCallbacks>(callbacks);
  callbacksRef.current = callbacks;
  const unmountedRef = useRef(false);
  const propsRef = useRef(props);
  propsRef.current = props;
  const markerPopoverRef = useRef(props.markerPopover);
  const [markerPopoverPosition, setMarkerPopoverPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);

  // Keep the shared slot in sync with this component's props on every render.
  useEffect(() => {
    callbacks.onMapClick = props.onMapClick;
    callbacks.onMarkerClick = props.onMarkerClick;
    callbacks.onMarkerContextMenu = props.onMarkerContextMenu;
    callbacks.onSegmentHover = props.onSegmentHover;
    callbacks.onSegmentClick = props.onSegmentClick;
    callbacks.onSegmentLeave = props.onSegmentLeave;
    callbacks.onSegmentMarkerClick = props.onSegmentMarkerClick;
    callbacks.onRoiChange = props.onRoiChange;
    callbacks.onStatus = props.onStatus;
    callbacks.onLevelChange = props.onLevelChange;
    callbacks.onReady = handleLivemapReady;
    callbacks.onFloorChanged = handleFloorChanged;
    callbacks.onIndoorLevelChanged = handleIndoorLevelChanged;
    callbacks.onSdkMapClick = handleSdkMapClick;
    callbacks.onCameraMove = syncMarkerPopoverPosition;
  });

  function syncMarkerPopoverPosition() {
    const markerPopover = markerPopoverRef.current;
    const map = stateRef.current.mapboxMap;
    if (!markerPopover || !map || typeof map.project !== "function") {
      setMarkerPopoverPosition(null);
      return;
    }
    const point = map.project([
      markerPopover.longitude,
      markerPopover.latitude,
    ]);
    setMarkerPopoverPosition({
      left: Number(point.x),
      top: Number(point.y),
    });
  }

  function setStatus(message: string) {
    if (statusElRef.current) {
      statusElRef.current.textContent = message;
    }
    callbacksRef.current.onStatus?.(message);
  }

  async function initPrivateFloorStore() {
    const state = stateRef.current;
    if (
      state.mapboxMap ||
      typeof window.wemap === "undefined" ||
      !window.wemap.v1 ||
      typeof window.wemap.v1.getPrivateInterface !== "function"
    ) {
      return;
    }
    const getPrivateInterface = window.wemap.v1.getPrivateInterface;
    try {
      const privateInterface = await new Promise<WemapPrivateInterface>((resolve) => {
        getPrivateInterface(resolve);
      });
      const facade =
        privateInterface && typeof privateInterface.getFacade === "function"
          ? privateInterface.getFacade()
          : null;
      state.facade = facade;
      state.floorStore =
        facade && typeof facade.getStore === "function"
          ? facade.getStore("FloorStore")
          : null;
      state.mapboxMap = facade?.getApplication?.().map?._driver?._map ?? null;
      const host = hostRef.current;
      if (state.mapboxMap && host && !host.cameraListenersRegistered) {
        // Registered once per livemap: dispatch through the shared slot so the
        // panel currently mounted receives the event, not the one that mounted
        // the map first.
        state.mapboxMap.on("move", () => callbacksRef.current.onCameraMove?.());
        state.mapboxMap.on("resize", () => callbacksRef.current.onCameraMove?.());
        host.cameraListenersRegistered = true;
      }
      if (state.mapboxMap) {
        syncMarkerPopoverPosition();
      }
    } catch (error) {
      console.error("Failed to initialize private floor store", error);
    }
  }

  function ensureOverlayLayers() {
    const state = stateRef.current;
    if (!state.mapboxMap || state.overlayLayersReady) {
      return;
    }

    const map = state.mapboxMap;

    if (!map.getSource(POLYGON_SOURCE_ID)) {
      map.addSource(POLYGON_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getSource(POINT_SOURCE_ID)) {
      map.addSource(POINT_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getSource(DRAFT_SOURCE_ID)) {
      map.addSource(DRAFT_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getSource(SEGMENT_SOURCE_ID)) {
      map.addSource(SEGMENT_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getLayer(POINT_CIRCLE_LAYER_ID)) {
      map.addLayer({
        id: POINT_CIRCLE_LAYER_ID,
        source: POINT_SOURCE_ID,
        type: "circle",
        paint: {
          "circle-color": ["get", "color"],
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            10,
            [
              "*",
              ["coalesce", ["get", "radius"], 6],
              ["case", ["==", ["get", "scale_with_zoom"], true], 0.1, 1],
            ],
            15,
            [
              "*",
              ["coalesce", ["get", "radius"], 6],
              ["case", ["==", ["get", "scale_with_zoom"], true], 0.35, 1],
            ],
            18,
            [
              "*",
              ["coalesce", ["get", "radius"], 6],
              ["case", ["==", ["get", "scale_with_zoom"], true], 0.65, 1],
            ],
            21,
            ["coalesce", ["get", "radius"], 6],
          ],
          "circle-opacity": ["coalesce", ["get", "opacity"], 1],
          "circle-stroke-color": [
            "coalesce",
            ["get", "stroke_color"],
            "#ffffff",
          ],
          "circle-stroke-width": ["coalesce", ["get", "stroke_width"], 1.5],
        },
      });
    }

    if (!map.getLayer(POINT_LABEL_LAYER_ID)) {
      map.addLayer({
        id: POINT_LABEL_LAYER_ID,
        source: POINT_SOURCE_ID,
        type: "symbol",
        layout: {
          "text-field": ["coalesce", ["get", "label"], ""],
          "text-size": ["coalesce", ["get", "text_size"], 12],
          "text-allow-overlap": true,
          "text-ignore-placement": true,
        },
        paint: {
          "text-color": ["coalesce", ["get", "text_color"], "#111827"],
          "text-halo-color": "#ffffff",
          "text-halo-width": 1,
        },
      });
    }

    if (!map.getLayer(POLYGON_FILL_LAYER_ID)) {
      map.addLayer({
        id: POLYGON_FILL_LAYER_ID,
        source: POLYGON_SOURCE_ID,
        type: "fill",
        paint: {
          "fill-color": ["get", "color"],
          "fill-opacity": ["coalesce", ["get", "opacity"], 0.2],
        },
      });
    }

    if (!map.getLayer(POLYGON_LINE_LAYER_ID)) {
      map.addLayer({
        id: POLYGON_LINE_LAYER_ID,
        source: POLYGON_SOURCE_ID,
        type: "line",
        paint: {
          "line-color": ["get", "color"],
          "line-width": ["coalesce", ["get", "line_width"], 2],
          "line-opacity": ["coalesce", ["get", "line_opacity"], 1],
        },
      });
    }

    if (!map.getLayer(DRAFT_LINE_LAYER_ID)) {
      map.addLayer({
        id: DRAFT_LINE_LAYER_ID,
        source: DRAFT_SOURCE_ID,
        type: "line",
        filter: ["==", ["geometry-type"], "LineString"],
        paint: {
          "line-color": ["coalesce", ["get", "color"], "#ff4d4f"],
          "line-width": 2,
          "line-dasharray": [2, 1],
        },
      });
    }

    if (!map.getLayer(DRAFT_FILL_LAYER_ID)) {
      map.addLayer({
        id: DRAFT_FILL_LAYER_ID,
        source: DRAFT_SOURCE_ID,
        type: "fill",
        filter: ["==", ["geometry-type"], "Polygon"],
        paint: {
          "fill-color": ["coalesce", ["get", "color"], "#ff4d4f"],
          "fill-opacity": 0.2,
        },
      });
    }

    if (!map.getLayer(DRAFT_POINT_LAYER_ID)) {
      map.addLayer({
        id: DRAFT_POINT_LAYER_ID,
        source: DRAFT_SOURCE_ID,
        type: "circle",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-color": ["coalesce", ["get", "color"], "#ff4d4f"],
          "circle-radius": 4,
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1.5,
        },
      });
    }

    if (!map.getLayer(SEGMENT_HIT_LAYER_ID)) {
      map.addLayer({
        id: SEGMENT_HIT_LAYER_ID,
        source: SEGMENT_SOURCE_ID,
        type: "line",
        filter: ["!=", ["get", "interactive"], false],
        paint: {
          "line-color": "#000000",
          "line-width": 22,
          "line-opacity": 0.001,
        },
      });
    }

    if (!map.getLayer(SEGMENT_LINE_LAYER_ID)) {
      map.addLayer({
        id: SEGMENT_LINE_LAYER_ID,
        source: SEGMENT_SOURCE_ID,
        type: "line",
        paint: {
          "line-color": ["coalesce", ["get", "color"], "#2563eb"],
          "line-width": ["coalesce", ["get", "width"], 2],
          "line-opacity": ["coalesce", ["get", "opacity"], 0.9],
        },
      });
    }

    if (!map.getSource(CONE_SOURCE_ID)) {
      map.addSource(CONE_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getLayer(CONE_LAYER_ID)) {
      map.addLayer({
        id: CONE_LAYER_ID,
        type: "symbol",
        source: CONE_SOURCE_ID,
        layout: {
          "icon-image": CONE_ICON_ID,
          "icon-rotate": ["get", "bearing"],
          "icon-rotation-alignment": "map",
          "icon-pitch-alignment": "map",
          "icon-size": [
            "interpolate", ["linear"], ["zoom"],
            10, 0.15,
            15, 0.4,
            18, 0.7,
            21, 1.0,
          ],
          "icon-allow-overlap": true,
          "icon-ignore-placement": true,
        },
      });
    }

    if (!map.getSource(ROI_SOURCE_ID)) {
      map.addSource(ROI_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getLayer(ROI_FILL_LAYER_ID)) {
      map.addLayer({
        id: ROI_FILL_LAYER_ID,
        source: ROI_SOURCE_ID,
        type: "fill",
        filter: ["==", ["geometry-type"], "Polygon"],
        paint: {
          "fill-color": ROI_COLOR,
          "fill-opacity": 0.12,
        },
      });
    }

    if (!map.getLayer(ROI_LINE_LAYER_ID)) {
      map.addLayer({
        id: ROI_LINE_LAYER_ID,
        source: ROI_SOURCE_ID,
        type: "line",
        filter: ["!=", ["geometry-type"], "Point"],
        paint: {
          "line-color": ROI_COLOR,
          "line-width": 2,
          "line-dasharray": [2, 1],
        },
      });
    }

    if (!map.getLayer(ROI_POINT_LAYER_ID)) {
      map.addLayer({
        id: ROI_POINT_LAYER_ID,
        source: ROI_SOURCE_ID,
        type: "circle",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-color": ROI_COLOR,
          // The first vertex is drawn larger as the click-to-close target.
          "circle-radius": ["case", ["==", ["get", "index"], 0], 6, 4],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1.5,
        },
      });
    }

    if (!state.roiHandlersRegistered) {
      map.on("click", (event: MapboxPointerEvent) => {
        if (!state.roiActive) return;
        const clicked: LngLat = {
          latitude: event.lngLat.lat,
          longitude: event.lngLat.lng,
        };
        // A click after the polygon is closed starts a fresh one.
        if (state.roiClosed) {
          state.roiVertices = [clicked];
          state.roiClosed = false;
          state.roiCursor = null;
          syncRoi();
          callbacksRef.current.onRoiChange?.(null);
          return;
        }
        // Click near the first vertex closes the ring (needs >= 3 vertices).
        if (state.roiVertices.length >= 3) {
          const first = state.roiVertices[0];
          const firstPoint = map.project([first.longitude, first.latitude]);
          const dpx = Math.hypot(
            event.point.x - firstPoint.x,
            event.point.y - firstPoint.y,
          );
          if (dpx <= ROI_CLOSE_PX) {
            state.roiClosed = true;
            state.roiCursor = null;
            syncRoi();
            callbacksRef.current.onRoiChange?.([...state.roiVertices]);
            return;
          }
        }
        state.roiVertices = [...state.roiVertices, clicked];
        syncRoi();
      });
      map.on("mousemove", (event: MapboxPointerEvent) => {
        if (!state.roiActive || state.roiClosed || state.roiVertices.length === 0) {
          return;
        }
        state.roiCursor = {
          latitude: event.lngLat.lat,
          longitude: event.lngLat.lng,
        };
        syncRoi();
      });
      state.roiHandlersRegistered = true;
    }

    if (!state.markerClickHandlerRegistered) {
      const emitFeatureClick = (feature: MapboxFeature | undefined) => {
        const properties = feature?.properties ?? {};
        const markerId = properties.id;
        if (!markerId) {
          return;
        }
        callbacksRef.current.onMarkerClick?.(String(markerId));
      };
      map.on("click", POINT_CIRCLE_LAYER_ID, (event: MapboxPointerEvent) => {
        const feature = event?.features?.[0];
        emitFeatureClick(feature);
      });
      map.on("click", POINT_LABEL_LAYER_ID, (event: MapboxPointerEvent) => {
        const feature = event?.features?.[0];
        emitFeatureClick(feature);
      });
      map.on("click", SEGMENT_HIT_LAYER_ID, (event: MapboxPointerEvent) => {
        const properties = event?.features?.[0]?.properties ?? {};
        const position = {
          latitude: event.lngLat.lat,
          longitude: event.lngLat.lng,
          level:
            typeof properties.level === "string"
              ? properties.level
              : null,
        };
        const marker = nearestMarker(position, stateRef.current.segmentHoverMarkers);
        if (marker) {
          callbacksRef.current.onSegmentMarkerClick?.(marker.id);
          return;
        }
        callbacksRef.current.onSegmentClick?.(position);
      });
      const emitFeatureContextMenu = (event: MapboxPointerEvent) => {
        const markerId = event?.features?.[0]?.properties?.id;
        if (!markerId) {
          return;
        }
        event.preventDefault?.();
        event.originalEvent?.preventDefault?.();
        const original = event.originalEvent;
        callbacksRef.current.onMarkerContextMenu?.(String(markerId), {
          x: original?.clientX ?? event.point?.x ?? 0,
          y: original?.clientY ?? event.point?.y ?? 0,
        });
      };
      map.on("contextmenu", POINT_CIRCLE_LAYER_ID, emitFeatureContextMenu);
      map.on("contextmenu", POINT_LABEL_LAYER_ID, emitFeatureContextMenu);
      map.on("mouseenter", POINT_CIRCLE_LAYER_ID, () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", POINT_CIRCLE_LAYER_ID, () => {
        map.getCanvas().style.cursor = "";
      });
      map.on("mouseenter", POINT_LABEL_LAYER_ID, () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", POINT_LABEL_LAYER_ID, () => {
        map.getCanvas().style.cursor = "";
      });
      map.on("mousemove", SEGMENT_HIT_LAYER_ID, (event: MapboxPointerEvent) => {
        const properties = event?.features?.[0]?.properties ?? {};
        const position = {
          latitude: event.lngLat.lat,
          longitude: event.lngLat.lng,
          level:
            typeof properties.level === "string"
              ? properties.level
              : null,
        };
        const state = stateRef.current;
        const marker = nearestMarker(position, state.segmentHoverMarkers);
        if (state.segmentHoverMarker?.id !== marker?.id) {
          state.segmentHoverMarker = marker;
          syncPoints();
        }
        map.getCanvas().style.cursor = "pointer";
        callbacksRef.current.onSegmentHover?.(position);
      });
      map.on("mouseleave", SEGMENT_HIT_LAYER_ID, () => {
        const state = stateRef.current;
        if (state.segmentHoverMarker) {
          state.segmentHoverMarker = null;
          syncPoints();
        }
        map.getCanvas().style.cursor = "";
        callbacksRef.current.onSegmentLeave?.();
      });
      state.markerClickHandlerRegistered = true;
    }

    if (
      state.livemap &&
      typeof state.livemap.addExternalIndoorLayers === "function"
    ) {
      state.livemap.addExternalIndoorLayers([
        POINT_CIRCLE_LAYER_ID,
        POINT_LABEL_LAYER_ID,
        POLYGON_FILL_LAYER_ID,
        POLYGON_LINE_LAYER_ID,
        DRAFT_FILL_LAYER_ID,
        DRAFT_LINE_LAYER_ID,
        DRAFT_POINT_LAYER_ID,
        SEGMENT_HIT_LAYER_ID,
        SEGMENT_LINE_LAYER_ID,
        CONE_LAYER_ID,
        ROI_FILL_LAYER_ID,
        ROI_LINE_LAYER_ID,
        ROI_POINT_LAYER_ID,
      ]);
    }

    state.overlayLayersReady = true;
  }

  function syncPoints() {
    const state = stateRef.current;
    const map = state.mapboxMap;
    if (!map) return;
    const pointSource = map.getSource(POINT_SOURCE_ID);
    if (!pointSource) return;
    const renderedMarkers: LivemapMarker[] = [...state.markers];
    if (
      state.segmentHoverMarker &&
      !renderedMarkers.some((marker) => marker.id === state.segmentHoverMarker?.id)
    ) {
      renderedMarkers.push(state.segmentHoverMarker);
    }
    if (state.pendingPoint && state.mode === "point") {
      renderedMarkers.push({
        id: "preview-click",
        latitude: state.pendingPoint.latitude,
        longitude: state.pendingPoint.longitude,
        color: state.activeColor ?? "#ff4d4f",
        level: state.currentLevel,
      });
    }
    pointSource.setData({
      type: "FeatureCollection",
      features: renderedMarkers.map((point) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [point.longitude, point.latitude] },
        properties: {
          id: point.id,
          color: point.color,
          radius: point.radius,
          scale_with_zoom: point.scaleWithZoom ?? false,
          stroke_color: point.strokeColor,
          stroke_width: point.strokeWidth,
          opacity: point.opacity,
          label: point.label,
          text_color: point.textColor,
          text_size: point.textSize,
          ...getLevelProperties(point.level),
        },
      })),
    });
  }

  function syncSegments() {
    const state = stateRef.current;
    const map = state.mapboxMap;
    if (!map) return;
    const segmentSource = map.getSource(SEGMENT_SOURCE_ID);
    if (!segmentSource) return;
    segmentSource.setData({
      type: "FeatureCollection",
      features: state.segments.map((segment) => ({
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: [
            [segment.fromLongitude, segment.fromLatitude],
            [segment.toLongitude, segment.toLatitude],
          ],
        },
        properties: {
          id: segment.id,
          color: segment.color ?? "#2563eb",
          width: segment.width ?? 2,
          opacity: segment.opacity ?? 0.9,
          interactive: segment.interactive ?? true,
          ...getLevelProperties(segment.level),
        },
      })),
    });
  }

  function syncPolygons() {
    const state = stateRef.current;
    const map = state.mapboxMap;
    if (!map) return;
    const polygonSource = map.getSource(POLYGON_SOURCE_ID);
    if (!polygonSource) return;
    polygonSource.setData({
      type: "FeatureCollection",
      features: state.polygons.map((polygon) => ({
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [tupleCoordinates(polygon.coordinates)] },
        properties: {
          id: polygon.id,
          color: polygon.color,
          opacity: polygon.opacity,
          line_opacity: polygon.lineOpacity,
          line_width: polygon.lineWidth,
          ...getLevelProperties(polygon.level),
        },
      })),
    });
  }

  function syncCone() {
    const state = stateRef.current;
    const map = state.mapboxMap;
    if (!map) return;
    const src = map.getSource(CONE_SOURCE_ID);
    if (!src) return;
    const cone = state.cone;
    if (!cone) {
      src.setData({ type: "FeatureCollection", features: [] });
      return;
    }
    ensureConeIcon(map, cone.fovDeg ?? 50, props.coneColor ?? "#16a34a");
    src.setData({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "Point", coordinates: [cone.longitude, cone.latitude] },
          properties: {
            bearing: cone.bearingDeg,
            ...getLevelProperties(cone.level ?? null),
          },
        },
      ],
    });
  }

  function syncRoi() {
    const state = stateRef.current;
    const map = state.mapboxMap;
    if (!map) return;
    const src = map.getSource(ROI_SOURCE_ID);
    if (!src) return;
    const verts = state.roiVertices;
    const features: GeoJsonFeature[] = verts.map((vertex, index) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [vertex.longitude, vertex.latitude] },
      properties: { index },
    }));
    if (state.roiClosed && verts.length >= 3) {
      const ring = verts
        .map((v): [number, number] => [v.longitude, v.latitude])
        .concat([[verts[0].longitude, verts[0].latitude]]);
      features.push({
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [ring] },
        properties: {},
      });
    } else if (verts.length >= 1) {
      // Open path while drawing, with a rubber-band segment to the cursor.
      const line = verts.map((v): [number, number] => [v.longitude, v.latitude]);
      if (state.roiCursor) {
        line.push([state.roiCursor.longitude, state.roiCursor.latitude]);
      }
      if (line.length >= 2) {
        features.push({
          type: "Feature",
          geometry: { type: "LineString", coordinates: line },
          properties: {},
        });
      }
    }
    src.setData({ type: "FeatureCollection", features });
  }

  function applyRoiInteraction() {
    const state = stateRef.current;
    const map = state.mapboxMap;
    if (!map) return;
    if (state.roiActive) {
      // Panning stays enabled between clicks; only the cursor hints drawing.
      if (map.getCanvas()) map.getCanvas().style.cursor = "crosshair";
    } else {
      state.roiVertices = [];
      state.roiClosed = false;
      state.roiCursor = null;
      if (map.getCanvas()) map.getCanvas().style.cursor = "";
      syncRoi();
    }
  }

  function syncOverlays() {
    const state = stateRef.current;
    if (!state.mapboxMap) {
      return;
    }
    ensureOverlayLayers();
    const map = state.mapboxMap;

    const draftSource = map.getSource(DRAFT_SOURCE_ID);
    if (!draftSource) {
      return;
    }

    syncPoints();
    syncSegments();
    syncPolygons();
    syncRoi();

    const draft = state.draftVertices;
    const draftFeatures: GeoJsonFeature[] = draft.map((vertex, index) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [vertex.longitude, vertex.latitude],
      },
      properties: {
        index,
        color: state.activeColor,
        ...getLevelProperties(state.currentLevel),
      },
    }));
    if (state.draftClosed && draft.length >= 3) {
      const ring = draft
        .map((v): [number, number] => [v.longitude, v.latitude])
        .concat([[draft[0].longitude, draft[0].latitude]]);
      draftFeatures.push({
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [ring] },
        properties: {
          color: state.activeColor,
          ...getLevelProperties(state.currentLevel),
        },
      });
      draftFeatures.push({
        type: "Feature",
        geometry: { type: "LineString", coordinates: ring },
        properties: {
          color: state.activeColor,
          ...getLevelProperties(state.currentLevel),
        },
      });
    } else if (draft.length >= 2) {
      draftFeatures.push({
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: draft.map((v): [number, number] => [v.longitude, v.latitude]),
        },
        properties: {
          color: state.activeColor,
          ...getLevelProperties(state.currentLevel),
        },
      });
    }

    draftSource.setData({
      type: "FeatureCollection",
      features: draftFeatures,
    });
  }

  async function focusOnTarget(target: FocusTarget | null) {
    const state = stateRef.current;
    if (!target || !state.livemap) {
      return;
    }
    try {
      if (target.level !== undefined && target.level !== null && target.level !== "") {
        const levelNumber = Number(target.level);
        if (
          !Number.isNaN(levelNumber) &&
          typeof state.livemap.setIndoorLevel === "function"
        ) {
          state.livemap.setIndoorLevel(levelNumber);
        }
      }
      if (state.mapboxMap) {
        state.mapboxMap.easeTo({
          center: [target.longitude, target.latitude],
          zoom: 21,
          duration: 800,
        });
      } else if (typeof state.livemap.centerTo === "function") {
        await state.livemap.centerTo({
          center: {
            latitude: target.latitude,
            longitude: target.longitude,
          },
          zoom: 21,
        });
      }
    } catch (error) {
      console.error("Failed to focus annotation target", error, target);
    }
  }

  async function refreshCurrentLevel() {
    const state = stateRef.current;
    if (!state.livemap) {
      return;
    }
    try {
      await initPrivateFloorStore();
      const floor =
        typeof state.livemap.getCurrentFloor === "function"
          ? await state.livemap.getCurrentFloor()
          : null;
      const indoorLevel =
        typeof state.livemap.getIndoorLevel === "function"
          ? await state.livemap.getIndoorLevel()
          : null;
      const storeFloor =
        state.floorStore && typeof state.floorStore.getCurrentFloor === "function"
          ? state.floorStore.getCurrentFloor()
          : null;
      const storeIndoorLevel =
        state.floorStore &&
        typeof state.floorStore.getCurrentIndoorLevel === "function"
          ? state.floorStore.getCurrentIndoorLevel()
          : null;
      state.currentLevel =
        normalizeLevel(storeIndoorLevel) ??
        deriveLevelFromFloor(floor) ??
        normalizeLevel(indoorLevel);
      state.currentAltitude =
        deriveAltitudeFromFloor(storeFloor) ??
        deriveAltitudeFromFloor(floor) ??
        deriveAltitudeFromFloor(indoorLevel);
      callbacksRef.current.onLevelChange?.(state.currentLevel);
      setStatus(
        state.currentLevel === null
          ? "Click on the map to capture a point"
          : `Current level: ${state.currentLevel}`,
      );
    } catch (error) {
      console.error("Failed to query livemap level", error);
      setStatus("Livemap loaded");
    }
  }

  async function emitClickEvent(data: { latitude: number; longitude: number }) {
    const state = stateRef.current;
    if (state.mode === "inspect") {
      return;
    }
    if (state.mode === "point") {
      // Optimistically show preview at click location until parent commits.
      state.pendingPoint = { latitude: data.latitude, longitude: data.longitude };
      syncOverlays();
    } else {
      state.draftTimestamp = Date.now();
      const clicked: LngLat = {
        latitude: data.latitude,
        longitude: data.longitude,
      };
      if (
        state.draftVertices.length >= 3 &&
        distanceMeters(state.draftVertices[0], clicked) <= 3.0
      ) {
        state.draftClosed = true;
      } else {
        state.draftClosed = false;
        state.draftVertices = [...state.draftVertices, clicked];
      }
      syncOverlays();
    }
    await refreshCurrentLevel();
    callbacksRef.current.onMapClick?.({
      latitude: data.latitude,
      longitude: data.longitude,
      altitude: state.currentAltitude,
      level: state.currentLevel,
      draftVertices: [...state.draftVertices],
      draftClosed: state.draftClosed,
      draftTimestamp: state.draftTimestamp,
    });
  }

  // Post-"ready" initialisation. Also run on a cache hit, where the SDK event
  // has already fired for a previous mount and will not fire again.
  async function initializeMapView() {
    const state = stateRef.current;
    await initPrivateFloorStore();
    if (unmountedRef.current) {
      return;
    }
    await refreshCurrentLevel();
    ensureOverlayLayers();
    applyRoiInteraction();
    syncOverlays();
    syncCone();
    if (state.mapboxMap && typeof state.mapboxMap.resize === "function") {
      state.mapboxMap.resize();
    }
    // Retry sync after a short delay to ensure mapbox layers are fully ready
    setTimeout(() => {
      if (!unmountedRef.current) {
        syncOverlays();
      }
    }, 500);
  }

  async function handleLivemapReady() {
    await initializeMapView();
  }

  async function handleFloorChanged(data: Record<string, unknown>) {
    const state = stateRef.current;
    state.currentLevel = deriveLevelFromFloor(data && data.floor);
    state.currentAltitude =
      deriveAltitudeFromFloor(data && data.floor) ?? state.currentAltitude;
    await refreshCurrentLevel();
  }

  async function handleIndoorLevelChanged(data: Record<string, unknown>) {
    const state = stateRef.current;
    const source =
      data &&
      (data.level ??
        data.indoorLevel ??
        data.floor ??
        data.currentIndoorLevel);
    state.currentLevel = normalizeLevel(source) ?? state.currentLevel;
    state.currentAltitude =
      deriveAltitudeFromFloor(source) ?? state.currentAltitude;
    await refreshCurrentLevel();
  }

  function handleSdkMapClick(data: { latitude: number; longitude: number }) {
    void emitClickEvent(data);
  }

  /**
   * Copies the current props into the freshly acquired host state.
   *
   * The per-prop effects run before the host promise resolves, so they write to
   * the placeholder state; without this the panel would render an empty map
   * until one of its props happens to change.
   */
  function adoptProps(state: LivemapState) {
    const current = propsRef.current;
    state.mode = current.mode;
    state.activeColor = current.activeColor;
    state.markers = current.markers;
    state.segmentHoverMarkers = current.segmentHoverMarkers ?? [];
    state.segments = current.segments ?? [];
    state.polygons = current.polygons;
    state.pendingPoint = current.pendingPoint;
    state.draftVertices = current.draftVertices;
    state.draftClosed = current.draftClosed;
    state.cone = current.cone ?? null;
    state.roiActive = current.roiActive ?? false;
    const roiPolygon = current.roiPolygon ?? null;
    state.roiVertices = roiPolygon ?? [];
    state.roiClosed = roiPolygon !== null;
    state.roiCursor = null;
  }

  // Mount / unmount: attach the cached livemap, or create it on first use.
  // The livemap is never destroyed here — it is parked off-screen so switching
  // tabs does not reload the map (see livemapHost.ts).
  useEffect(() => {
    unmountedRef.current = false;
    let cancelled = false;
    setStatus("Loading livemap\u2026");
    acquireLivemapHost(props.emmid)
      .then((host) => {
        hostRef.current = host;
        stateRef.current = host.state;
        if (cancelled || unmountedRef.current) {
          releaseLivemapHost(host);
          return;
        }
        adoptProps(host.state);
        mapElRef.current?.appendChild(host.el);
        host.state.mapboxMap?.resize();
        if (host.ready) {
          void initializeMapView();
        }
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setStatus(`Failed to load Wemap SDK: ${error.message}`);
      });
    return () => {
      cancelled = true;
      unmountedRef.current = true;
      const host = hostRef.current;
      if (!host) {
        return;
      }
      // Hand the map over with empty overlays; the next panel resyncs its own.
      resetConsumerState(host.state);
      syncOverlays();
      syncCone();
      releaseLivemapHost(host);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.emmid]);

  // Mode / activeColor changes
  useEffect(() => {
    const state = stateRef.current;
    state.mode = props.mode;
    state.activeColor = props.activeColor;
    if (props.mode === "polygon") {
      state.pendingPoint = null;
    }
    syncOverlays();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.mode, props.activeColor]);

  // Points — only update point source when markers/pendingPoint change
  useEffect(() => {
    const state = stateRef.current;
    state.markers = props.markers;
    state.pendingPoint = props.pendingPoint;
    syncPoints();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.markers, props.pendingPoint]);

  useEffect(() => {
    const state = stateRef.current;
    state.segmentHoverMarkers = props.segmentHoverMarkers ?? [];
    if (state.segmentHoverMarker) {
      state.segmentHoverMarker =
        state.segmentHoverMarkers.find(
          (marker) => marker.id === state.segmentHoverMarker?.id,
        ) ?? null;
      syncPoints();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.segmentHoverMarkers]);

  // Segments — only update segment source when segments change
  useEffect(() => {
    const state = stateRef.current;
    state.segments = props.segments ?? [];
    syncSegments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.segments]);

  // Polygons — only update polygon source when polygons change
  useEffect(() => {
    const state = stateRef.current;
    state.polygons = props.polygons;
    syncPolygons();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.polygons]);

  // Cone — update cone source directly when cone prop changes
  useEffect(() => {
    const state = stateRef.current;
    state.cone = props.cone ?? null;
    syncCone();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.cone]);

  // ROI active — set the drawing cursor; clears the in-progress ring when off
  useEffect(() => {
    const state = stateRef.current;
    state.roiActive = props.roiActive ?? false;
    applyRoiInteraction();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.roiActive]);

  // ROI polygon — reflect the controlled ring (external clear / set)
  useEffect(() => {
    const state = stateRef.current;
    const polygon = props.roiPolygon ?? null;
    if (polygon === null) {
      // Ignore the null echo of our own restart while a new ring is in progress.
      if (state.roiVertices.length > 0 && !state.roiClosed) {
        return;
      }
      state.roiVertices = [];
      state.roiClosed = false;
      state.roiCursor = null;
    } else {
      state.roiVertices = polygon;
      state.roiClosed = true;
      state.roiCursor = null;
    }
    syncRoi();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.roiPolygon]);

  // Draft — updates both point and draft sources (pendingPoint interacts with mode)
  useEffect(() => {
    const state = stateRef.current;
    state.draftVertices = props.draftVertices;
    state.draftClosed = props.draftClosed;
    syncOverlays();
    // Retry sync if mapboxMap wasn't ready on first attempt
    if (!state.mapboxMap && !unmountedRef.current) {
      const retryTimer = setTimeout(() => {
        if (!unmountedRef.current) {
          syncOverlays();
        }
      }, 800);
      return () => clearTimeout(retryTimer);
    }
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.draftVertices, props.draftClosed]);

  // Focus changes
  useEffect(() => {
    if (props.focusTarget) {
      void focusOnTarget(props.focusTarget);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.focusTarget]);

  useEffect(() => {
    markerPopoverRef.current = props.markerPopover;
    syncMarkerPopoverPosition();
    if (props.markerPopover && !stateRef.current.mapboxMap) {
      const retryTimer = window.setTimeout(syncMarkerPopoverPosition, 500);
      return () => window.clearTimeout(retryTimer);
    }
    return undefined;
  }, [props.markerPopover]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      const state = stateRef.current;
      if (state.mapboxMap && typeof state.mapboxMap.resize === "function") {
        state.mapboxMap.resize();
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const heightStyle =
    typeof props.height === "number"
      ? `${props.height}px`
      : (props.height ?? "100%");

  return (
    <div
      ref={containerRef}
      className="livemap-container"
      style={{ position: "relative", width: "100%", height: heightStyle }}
    >
      <div ref={statusElRef} className="livemap-status">
        Loading livemap…
      </div>
      <div
        ref={mapElRef}
        className="livemap-canvas"
        style={{ width: "100%", height: "100%" }}
      />
      {props.markerPopover && markerPopoverPosition ? (
        <div
          className="livemap-marker-popover"
          style={{
            left: markerPopoverPosition.left,
            top: markerPopoverPosition.top,
          }}
          onClick={(event) => event.stopPropagation()}
        >
          {props.markerPopover.content}
        </div>
      ) : null}
    </div>
  );
}

export default LivemapAnnotation;
