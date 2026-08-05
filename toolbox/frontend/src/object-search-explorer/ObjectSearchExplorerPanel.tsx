import {
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type CSSProperties,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { MapSummary } from "../api";
import CollapsibleSection from "../annotations/CollapsibleSection";
import {
  ExplorerAnnotationControls,
  ExplorerAnnotationList,
  useExplorerAnnotationWorkspace,
} from "../annotations/ExplorerAnnotationWorkspace";
import LivemapAnnotation, { type LivemapCone } from "../annotations/LivemapAnnotation";
import type { LivemapMarker, LivemapPolygon, LivemapSegment } from "../annotations/types";
import DismissibleAlert from "../DismissibleAlert";
import {
  fetchIndexCutout,
  fetchIndexObjects,
  fetchIndexStatus,
  fetchKeyframeGraph,
  indexCutoutPreviewUrl,
  indexKeyframeDepthPreviewUrl,
  indexKeyframeEquirectPreviewUrl,
  projectWorldPoint,
  resolveDepthPin,
} from "../index-explorer/api";
import type {
  CutoutIndexPayload,
  DepthPinResponse,
  IndexObjectRecord,
  IndexStatusResponse,
  IndexSummary,
  KeyframeGraphResponse,
} from "../index-explorer/types";
import BboxPostProcessControls from "./BboxPostProcessControls";
import { bboxArea, postProcessDetections } from "./bboxPostProcess";
import { useBboxPostProcessParams } from "./useBboxPostProcessParams";
import { useEquirectFrameHeight } from "./useEquirectFrameHeight";
import { useLivemapFrameHeight } from "./useLivemapFrameHeight";

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
};

type CutoutRow = {
  keyframeId: string;
  cutoutId: string;
  objectCount: number;
};

type SortMode = "keyframe" | "objects-desc" | "objects-asc";
type PanoramaViewMode = "image" | "depth";
type DepthImagePin = {
  requestId: string;
  source: "erp" | "cutout";
  keyframeId: string;
  cutoutId: string | null;
  xRatio: number;
  yRatio: number;
  erpXRatio: number | null;
  erpYRatio: number | null;
  status: "resolving" | "resolved" | "error";
  depthM: number | null;
  latitude: number | null;
  longitude: number | null;
  altitude: number | null;
  level: string | null;
  pointWorldWds: [number, number, number] | null;
  message: string | null;
};

type PhotosphereNavigationCandidate = {
  id: string;
  localX: number;
  localZ: number;
};

const PAGE_SIZE_OPTIONS = [6, 12, 24, 48];
const COLUMN_OPTIONS = [1, 2, 3, 4, 5];
const KEYFRAME_MARKER_COLOR = "#9ca3af";
const KEYFRAME_MARKER_SELECTED_COLOR = "#16a34a";
const KEYFRAME_MARKER_RADIUS = 6;
const KEYFRAME_MARKER_SELECTED_RADIUS = 8;
const CUBEMAP_FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"] as const;
const EquirectPhotoSphereViewer = lazy(() => import("./EquirectPhotoSphereViewer"));
const DEPTH_PIN_MIN_DEPTH_M = 0.25;
const VIEW_CONE_HALF_ANGLE_DEG = 25;
const VIEW_CONE_COLOR = KEYFRAME_MARKER_SELECTED_COLOR;
const VISUAL_SPLIT_MIN_PERCENT = 25;
const VISUAL_SPLIT_MAX_PERCENT = 75;
const KEYFRAME_GRAPH_STORAGE_KEY = "object-search-gui.showKeyframeGraph";
const KEYFRAME_GRAPH_COLOR = "#64748b";
const KEYFRAME_LINK_COLOR = "#9333ea";
const PHOTOSPHERE_NAVIGATION_MAX_DISTANCE_M = 40;

/** Initial compass bearing (degrees, 0 = north, clockwise) from one lng/lat to another. */
function bearingDegrees(
  from: { latitude: number; longitude: number },
  to: { latitude: number; longitude: number },
): number {
  const toRad = Math.PI / 180;
  const lat1 = from.latitude * toRad;
  const lat2 = to.latitude * toRad;
  const dLon = (to.longitude - from.longitude) * toRad;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}
const EARTH_RADIUS_M = 6378137;

function readStoredBoolean(key: string, fallback: boolean): boolean {
  try {
    const stored = localStorage.getItem(key);
    return stored === null ? fallback : stored !== "false";
  } catch {
    return fallback;
  }
}

function projectKeyframeToLocalFloor(
  source: {
    latitude: number;
    longitude: number;
    heading_deg: number;
  },
  destination: {
    latitude: number;
    longitude: number;
  },
): Pick<PhotosphereNavigationCandidate, "localX" | "localZ"> & {
  distanceM: number;
} {
  const latitudeRad = source.latitude * Math.PI / 180;
  const east =
    (destination.longitude - source.longitude) *
    Math.PI / 180 *
    EARTH_RADIUS_M *
    Math.cos(latitudeRad);
  const north =
    (destination.latitude - source.latitude) *
    Math.PI / 180 *
    EARTH_RADIUS_M;
  const distance = Math.hypot(east, north);
  const bearing = Math.atan2(east, north);
  const localYaw = bearing - source.heading_deg * Math.PI / 180;
  return {
    localX: distance * Math.sin(localYaw),
    localZ: -distance * Math.cos(localYaw),
    distanceM: distance,
  };
}

function buildCutoutRows(summary: IndexSummary): CutoutRow[] {
  return summary.keyframe_ids.flatMap((keyframeId) =>
    (summary.cutout_ids_by_keyframe[keyframeId] ?? []).map((cutoutId) => ({
      keyframeId,
      cutoutId,
      objectCount: summary.object_count_by_cutout[cutoutId] ?? 0,
    })),
  );
}

function manifestNumber(manifest: Record<string, unknown> | null, key: string, fallback: number): number {
  const value = Number(manifest?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

function cubemapFace(cutoutId: string, manifest: Record<string, unknown> | null): string {
  const stride = Math.max(1, manifestNumber(manifest, "id_stride", 1024));
  const localIndex = Number(cutoutId) % stride;
  return CUBEMAP_FACE_ORDER[localIndex] ?? "front";
}

function cubemapPixelToEquirect(
  u: number,
  v: number,
  faceName: string,
  faceSize: number,
  fovDeg: number,
): [number, number] {
  const f = 0.5 * faceSize / Math.tan(0.5 * fovDeg * Math.PI / 180);
  const c = (faceSize - 1) / 2;
  let x = (u - c) / f;
  let y = (v - c) / f;
  let z = 1;
  const norm = Math.hypot(x, y, z);
  x /= norm;
  y /= norm;
  z /= norm;
  let px = x;
  let py = y;
  let pz = z;
  switch (faceName) {
    case "back":
      px = -x;
      pz = -z;
      break;
    case "left":
      px = -z;
      pz = x;
      break;
    case "right":
      px = z;
      pz = -x;
      break;
    case "top":
      py = -z;
      pz = y;
      break;
    case "bottom":
      py = z;
      pz = -y;
      break;
  }
  const lon = Math.atan2(px, pz);
  const lat = Math.asin(Math.max(-1, Math.min(1, py)));
  return [((lon / (2 * Math.PI) + 0.5) % 1 + 1) % 1, lat / Math.PI + 0.5];
}

function bboxPolygonRatios(
  item: IndexObjectRecord,
  manifest: Record<string, unknown> | null,
): Array<[number, number]> {
  const faceSize = Math.max(1, manifestNumber(manifest, "cubemap_face_size", 512));
  const fov = Math.max(1, manifestNumber(manifest, "cubemap_fov_deg", 90));
  const faceName = cubemapFace(item.cutout_id, manifest);
  const corners: Array<[number, number]> = [
    [item.bbox[0], item.bbox[1]],
    [item.bbox[2], item.bbox[1]],
    [item.bbox[2], item.bbox[3]],
    [item.bbox[0], item.bbox[3]],
  ];
  return corners.map(([x, y]) => cubemapPixelToEquirect(x, y, faceName, faceSize, fov));
}


function ObjectSearchExplorerPanel(props: Props) {
  const [status, setStatus] = useState<IndexStatusResponse | null>(null);
  const [keyframeGraph, setKeyframeGraph] = useState<KeyframeGraphResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const [keyframeFilter, setKeyframeFilter] = useState("all");
  const [includeEmpty, setIncludeEmpty] = useState(true);
  const [showPhotosphereBoxes, setShowPhotosphereBoxes] = useState(false);
  const [panoramaViewMode, setPanoramaViewMode] =
    useState<PanoramaViewMode>("image");
  const [showKeyframeGraph, setShowKeyframeGraph] = useState(() =>
    readStoredBoolean(KEYFRAME_GRAPH_STORAGE_KEY, true),
  );
  const [sortMode, setSortMode] = useState<SortMode>("keyframe");
  const [pageSize, setPageSize] = useState(12);
  const [columns, setColumns] = useState(3);
  const [page, setPage] = useState(1);
  const [visualSplitPercent, setVisualSplitPercent] = useState(45);
  const [isResizingVisualSplit, setIsResizingVisualSplit] = useState(false);

  const [selectedCutoutId, setSelectedCutoutId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [cutoutPayload, setCutoutPayload] = useState<CutoutIndexPayload | null>(null);
  const [isLoadingCutout, setIsLoadingCutout] = useState(false);
  const [equirectPreviewError, setEquirectPreviewError] = useState<string | null>(null);
  const [equirectPreviewSrc, setEquirectPreviewSrc] = useState<string | null>(null);
  const [equirectPreviewKeyframeId, setEquirectPreviewKeyframeId] = useState<string | null>(null);
  const [photosphereYawRad, setPhotosphereYawRad] = useState<number | null>(null);
  const [photosphereViewKeyframeId, setPhotosphereViewKeyframeId] = useState<string | null>(null);
  const [depthPin, setDepthPin] = useState<DepthImagePin | null>(null);
  const [depthPinPopoverOpen, setDepthPinPopoverOpen] = useState(false);
  const [annotationMenu, setAnnotationMenu] = useState<{
    annotationId: string;
    x: number;
    y: number;
  } | null>(null);
  const [photosphereClassMenuOpen, setPhotosphereClassMenuOpen] = useState(false);
  const [keyframeLink, setKeyframeLink] = useState<{
    annotationId: string;
    keyframeId: string;
  } | null>(null);
  const [photosphereOrient, setPhotosphereOrient] = useState<{
    yawRad: number;
    token: number;
  } | null>(null);
  const preservedCompassBearingDegRef = useRef<number | null>(null);
  const preservedTextureYRatioRef = useRef(0.5);
  const visualSplitRef = useRef<HTMLDivElement | null>(null);
  const detectionRowRefs = useRef(new Map<string, HTMLTableRowElement>());
  const shouldFocusSelectedDetectionRef = useRef(false);
  const { params: bboxPostProcess, setParams: setBboxPostProcess, resetParams: resetBboxPostProcess } =
    useBboxPostProcessParams();
  const [indexObjects, setIndexObjects] = useState<IndexObjectRecord[]>([]);
  const annotation = useExplorerAnnotationWorkspace(props.mapId);

  const summary = status?.summary ?? null;
  const manifest = summary?.manifest ?? null;
  const polygonForPhotosphereDetection = useCallback(
    (item: IndexObjectRecord) => bboxPolygonRatios(item, manifest),
    [manifest],
  );

  useEffect(() => {
    if (!props.isMapKnown) {
      setIsLoading(false);
      return;
    }
    setDepthPin(null);
    setDepthPinPopoverOpen(false);
    setPhotosphereYawRad(null);
    setPhotosphereViewKeyframeId(null);
    setAnnotationMenu(null);
    setPhotosphereClassMenuOpen(false);
    setKeyframeLink(null);
    setPhotosphereOrient(null);
    preservedCompassBearingDegRef.current = null;
    preservedTextureYRatioRef.current = 0.5;
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    fetchIndexStatus(props.mapId, null)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setStatus(payload);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown]);

  useEffect(() => {
    if (!annotationMenu) {
      return;
    }
    const close = () => setAnnotationMenu(null);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setAnnotationMenu(null);
      }
    };
    window.addEventListener("click", close);
    window.addEventListener("resize", close);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("resize", close);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [annotationMenu]);

  useEffect(() => {
    if (!photosphereClassMenuOpen) {
      return;
    }
    const close = () => setPhotosphereClassMenuOpen(false);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPhotosphereClassMenuOpen(false);
      }
    };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", close);
    };
  }, [photosphereClassMenuOpen]);

  useEffect(() => {
    try {
      localStorage.setItem(KEYFRAME_GRAPH_STORAGE_KEY, String(showKeyframeGraph));
    } catch {
      /* ignore */
    }
  }, [showKeyframeGraph]);

  useEffect(() => {
    if (!props.isMapKnown) {
      setKeyframeGraph(null);
      return;
    }
    setKeyframeGraph(null);
    let cancelled = false;
    fetchKeyframeGraph(props.mapId)
      .then((payload) => {
        if (!cancelled) {
          setKeyframeGraph(payload);
        }
      })
      .catch((graphError: Error) => {
        if (!cancelled) {
          setKeyframeGraph({
            available: false,
            path: "",
            edges: [],
            skipped_feature_count: 0,
            error: graphError.message,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown]);

  const boxCountsByCutout = useMemo(() => {
    const grouped = new Map<string, IndexObjectRecord[]>();
    for (const item of indexObjects) {
      const list = grouped.get(item.cutout_id) ?? [];
      list.push(item);
      grouped.set(item.cutout_id, list);
    }
    const counts = new Map<string, { raw: number; filtered: number }>();
    for (const [cutoutId, detections] of grouped) {
      const filtered = postProcessDetections(detections, bboxPostProcess);
      counts.set(cutoutId, { raw: detections.length, filtered: filtered.length });
    }
    return counts;
  }, [indexObjects, bboxPostProcess]);

  const filteredObjectsByCutout = useMemo(() => {
    const grouped = new Map<string, IndexObjectRecord[]>();
    for (const item of indexObjects) {
      const list = grouped.get(item.cutout_id) ?? [];
      list.push(item);
      grouped.set(item.cutout_id, list);
    }
    const out = new Map<string, IndexObjectRecord[]>();
    for (const [cutoutId, detections] of grouped) {
      out.set(cutoutId, postProcessDetections(detections, bboxPostProcess));
    }
    return out;
  }, [indexObjects, bboxPostProcess]);

  const allRows = useMemo(() => {
    return summary ? buildCutoutRows(summary) : [];
  }, [summary]);

  const visibleRows = useMemo(() => {
    const rows = allRows
      .filter((row) => keyframeFilter === "all" || row.keyframeId === keyframeFilter)
      .filter((row) => includeEmpty || row.objectCount > 0);
    const sortedRows = [...rows];
    if (sortMode === "objects-desc") {
      sortedRows.sort(
        (a, b) =>
          b.objectCount - a.objectCount ||
          a.keyframeId.localeCompare(b.keyframeId) ||
          a.cutoutId.localeCompare(b.cutoutId),
      );
    } else if (sortMode === "objects-asc") {
      sortedRows.sort(
        (a, b) =>
          a.objectCount - b.objectCount ||
          a.keyframeId.localeCompare(b.keyframeId) ||
          a.cutoutId.localeCompare(b.cutoutId),
      );
    }
    return sortedRows;
  }, [allRows, includeEmpty, keyframeFilter, sortMode]);

  const emmid = props.map?.emmid ?? null;
  const { height: livemapFrameHeight, isDragging: isResizingLivemap, startResize: startLivemapResize } =
    useLivemapFrameHeight();
  const { height: equirectFrameHeight, isDragging: isResizingEquirect, startResize: startEquirectResize } =
    useEquirectFrameHeight();

  const selectedKeyframeForMap = keyframeFilter !== "all" ? keyframeFilter : null;
  const totalPages = Math.max(1, Math.ceil(visibleRows.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageRows = visibleRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  const pageKeyframeIds = [...new Set(pageRows.map((row) => row.keyframeId))];
  const pageKeyframeIdsKey = pageKeyframeIds.join("|");
  const selectedRow =
    allRows.find((row) => row.cutoutId === selectedCutoutId) ?? null;
  const selectedRowOnPage = pageRows.find((row) => row.cutoutId === selectedCutoutId) ?? null;
  const activeRow = selectedRowOnPage ?? pageRows[0] ?? null;

  const activeKeyframeId =
    keyframeFilter !== "all" ? keyframeFilter : activeRow?.keyframeId ?? null;
  const activeKeyframeMarker =
    status?.markers?.find((marker) => marker.id === activeKeyframeId) ?? null;

  const livemapMarkers: LivemapMarker[] = useMemo(() => {
    const markers: LivemapMarker[] = [];
    const activeMarker = activeKeyframeId
      ? status?.markers?.find((marker) => marker.id === activeKeyframeId)
      : null;
    if (activeMarker) {
      markers.push({
        id: activeMarker.id,
        latitude: activeMarker.latitude,
        longitude: activeMarker.longitude,
        level: activeMarker.level,
        color: KEYFRAME_MARKER_SELECTED_COLOR,
        radius: KEYFRAME_MARKER_SELECTED_RADIUS,
        scaleWithZoom: true,
      });
    }
    if (
      depthPin?.status === "resolved" &&
      depthPin.latitude !== null &&
      depthPin.longitude !== null
    ) {
      markers.push({
        id: "__depth_pin",
        latitude: depthPin.latitude,
        longitude: depthPin.longitude,
        level: depthPin.level,
        color: "#e11d48",
        radius: 9,
      });
    }
    markers.push(...annotation.markers);
    return markers;
  }, [
    activeKeyframeId,
    status?.markers,
    depthPin,
    annotation.markers,
  ]);

  const graphHoverMarkers: LivemapMarker[] = useMemo(() => {
    if (!showKeyframeGraph || !status?.markers || !summary?.keyframe_ids.length) {
      return [];
    }
    const indexedKeyframes = new Set(summary.keyframe_ids);
    return status.markers
      .filter((marker) => indexedKeyframes.has(marker.id))
      .map((marker) => {
        const isSelected = marker.id === selectedKeyframeForMap;
        return {
          id: marker.id,
          latitude: marker.latitude,
          longitude: marker.longitude,
          level: marker.level,
          color: isSelected ? KEYFRAME_MARKER_SELECTED_COLOR : KEYFRAME_MARKER_COLOR,
          radius: isSelected ? KEYFRAME_MARKER_SELECTED_RADIUS : KEYFRAME_MARKER_RADIUS,
          scaleWithZoom: true,
        };
      });
  }, [
    showKeyframeGraph,
    status?.markers,
    summary?.keyframe_ids,
    selectedKeyframeForMap,
  ]);

  useEffect(() => {
    setDepthPinPopoverOpen(false);
  }, [depthPin?.requestId]);

  const showLivemapResizer =
    emmid !== null && !!summary?.keyframe_ids.length && (status?.markers?.length ?? 0) > 0;
  const initialPhotosphereView = useMemo(() => {
    const headingDeg = activeKeyframeMarker?.heading_deg;
    const bearingDeg = preservedCompassBearingDegRef.current;
    const yawRad =
      headingDeg !== null &&
      headingDeg !== undefined &&
      bearingDeg !== null
        ? Math.atan2(
            Math.sin((bearingDeg - headingDeg) * Math.PI / 180),
            Math.cos((bearingDeg - headingDeg) * Math.PI / 180),
          )
        : 0;
    return {
      yawRad,
      textureYRatio: preservedTextureYRatioRef.current,
    };
  }, [activeKeyframeId, activeKeyframeMarker?.heading_deg]);
  const photosphereNavigationCandidates = useMemo<
    PhotosphereNavigationCandidate[]
  >(() => {
    if (
      !activeKeyframeId ||
      !activeKeyframeMarker ||
      activeKeyframeMarker.heading_deg === null ||
      !status?.markers
    ) {
      return [];
    }
    const indexedKeyframes = new Set(summary?.keyframe_ids ?? []);
    return status.markers.flatMap((marker) => {
      if (
        marker.id === activeKeyframeId ||
        !indexedKeyframes.has(marker.id) ||
        marker.level !== activeKeyframeMarker.level
      ) {
        return [];
      }
      const localPosition = projectKeyframeToLocalFloor(
        {
          latitude: activeKeyframeMarker.latitude,
          longitude: activeKeyframeMarker.longitude,
          heading_deg: activeKeyframeMarker.heading_deg as number,
        },
        marker,
      );
      if (localPosition.distanceM > PHOTOSPHERE_NAVIGATION_MAX_DISTANCE_M) {
        return [];
      }
      return [{
        id: marker.id,
        localX: localPosition.localX,
        localZ: localPosition.localZ,
      }];
    });
  }, [
    activeKeyframeId,
    activeKeyframeMarker,
    status?.markers,
    summary?.keyframe_ids,
  ]);

  const cone = useMemo<LivemapCone | null>(() => {
    if (
      !activeKeyframeMarker ||
      photosphereViewKeyframeId !== activeKeyframeId ||
      photosphereYawRad === null ||
      activeKeyframeMarker.heading_deg === null ||
      activeKeyframeMarker.heading_deg === undefined
    ) {
      return null;
    }
    const bearingDeg =
      (((activeKeyframeMarker.heading_deg + (photosphereYawRad * 180) / Math.PI) % 360) + 360) % 360;
    return {
      latitude: activeKeyframeMarker.latitude,
      longitude: activeKeyframeMarker.longitude,
      bearingDeg,
      fovDeg: VIEW_CONE_HALF_ANGLE_DEG * 2,
      level: activeKeyframeMarker.level,
    };
  }, [activeKeyframeMarker, photosphereViewKeyframeId, activeKeyframeId, photosphereYawRad]);

  const livemapPolygons: LivemapPolygon[] = useMemo(
    () => annotation.polygons,
    [annotation.polygons],
  );
  const keyframeGraphSegments: LivemapSegment[] = useMemo(() => {
    if (!showKeyframeGraph || !keyframeGraph?.available) {
      return [];
    }
    const indexedKeyframes = new Set(summary?.keyframe_ids ?? []);
    return keyframeGraph.edges.flatMap((edge) => {
      if (
        edge.keyframe_id_1 &&
        edge.keyframe_id_2 &&
        (!indexedKeyframes.has(edge.keyframe_id_1) || !indexedKeyframes.has(edge.keyframe_id_2))
      ) {
        return [];
      }
      const levels = edge.levels.length ? edge.levels : [null];
      return levels.map((level, levelIndex) => ({
        id: `${edge.id}-${levelIndex}`,
        fromLongitude: edge.from[0],
        fromLatitude: edge.from[1],
        toLongitude: edge.to[0],
        toLatitude: edge.to[1],
        color: KEYFRAME_GRAPH_COLOR,
        width: 2,
        opacity: 0.6,
        level,
        interactive: true,
      }));
    });
  }, [keyframeGraph, showKeyframeGraph, summary?.keyframe_ids]);
  const keyframeLinkSegments: LivemapSegment[] = useMemo(() => {
    if (!keyframeLink) {
      return [];
    }
    const ann = annotation.annotations.find(
      (item) =>
        item.id === keyframeLink.annotationId && item.annotationType === "point",
    );
    const keyframeMarker = status?.markers?.find(
      (marker) => marker.id === keyframeLink.keyframeId,
    );
    if (!ann || !keyframeMarker) {
      return [];
    }
    const coordinates = ann.coordinates as number[];
    return [
      {
        id: `__kf_link_${keyframeLink.annotationId}`,
        fromLongitude: coordinates[0],
        fromLatitude: coordinates[1],
        toLongitude: keyframeMarker.longitude,
        toLatitude: keyframeMarker.latitude,
        color: KEYFRAME_LINK_COLOR,
        width: 2,
        opacity: 0.9,
        level: ann.level,
        interactive: false,
      },
    ];
  }, [keyframeLink, annotation.annotations, status?.markers]);
  const livemapSegments: LivemapSegment[] = useMemo(
    () => [...keyframeGraphSegments, ...keyframeLinkSegments],
    [keyframeGraphSegments, keyframeLinkSegments],
  );

  const selectGraphKeyframe = useCallback(
    (keyframeId: string) => {
      if (!summary?.keyframe_ids.includes(keyframeId)) {
        return;
      }
      setKeyframeFilter(keyframeId);
      setSelectedCutoutId(null);
      setSelectedObjectId(null);
      setDepthPin(null);
      setDepthPinPopoverOpen(false);
    },
    [summary?.keyframe_ids],
  );

  const handlePhotosphereViewChange = useCallback(
    (keyframeId: string, yawRad: number, textureYRatio: number) => {
      if (keyframeId !== activeKeyframeId) {
        return;
      }
      setPhotosphereViewKeyframeId(keyframeId);
      setPhotosphereYawRad(yawRad);
      preservedTextureYRatioRef.current = textureYRatio;
      const headingDeg = status?.markers?.find((marker) => marker.id === keyframeId)?.heading_deg;
      if (headingDeg !== null && headingDeg !== undefined) {
        preservedCompassBearingDegRef.current =
          ((headingDeg + yawRad * 180 / Math.PI) % 360 + 360) % 360;
      }
    },
    [activeKeyframeId, status?.markers],
  );

  useEffect(() => {
    if (
      !activeKeyframeId ||
      !depthPin?.pointWorldWds ||
      depthPin.status === "resolving" ||
      depthPin.keyframeId === activeKeyframeId
    ) {
      return;
    }
    const requestId = depthPin.requestId;
    let cancelled = false;
    projectWorldPoint(props.mapId, activeKeyframeId, depthPin.pointWorldWds)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setDepthPin((current) => {
          if (!current || current.requestId !== requestId) {
            return current;
          }
          return {
            ...current,
            source: "erp",
            keyframeId: activeKeyframeId,
            cutoutId: null,
            xRatio: payload.erp_u,
            yRatio: payload.erp_v,
            erpXRatio: payload.erp_u,
            erpYRatio: payload.erp_v,
            depthM: payload.distance_m,
            status: "resolved",
            message: `Projected point ${payload.distance_m.toFixed(2)} m away`,
          };
        });
      })
      .catch((err: Error) => {
        if (cancelled) {
          return;
        }
        setDepthPin((current) => {
          if (!current || current.requestId !== requestId) {
            return current;
          }
          return {
            ...current,
            source: "erp",
            keyframeId: activeKeyframeId,
            cutoutId: null,
            erpXRatio: null,
            erpYRatio: null,
            status: "error",
            message: `Could not project selected point: ${err.message}`,
          };
        });
      });
    return () => {
      cancelled = true;
    };
  }, [
    activeKeyframeId,
    depthPin?.keyframeId,
    depthPin?.pointWorldWds,
    depthPin?.requestId,
    depthPin?.status,
    props.mapId,
  ]);

  useEffect(() => {
    if (!props.isMapKnown || !status?.available || !pageKeyframeIds.length) {
      setIndexObjects([]);
      return;
    }
    let cancelled = false;
    Promise.all(pageKeyframeIds.map((keyframeId) => fetchIndexObjects(props.mapId, keyframeId)))
      .then((groups) => {
        if (!cancelled) {
          setIndexObjects(groups.flat());
        }
      })
      .catch(() => {
        if (!cancelled) {
          setIndexObjects([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown, status?.available, pageKeyframeIdsKey]);

  const activeKeyframeDetections = useMemo(() => {
    if (!activeKeyframeId) {
      return [];
    }
    const cutoutIds = summary?.cutout_ids_by_keyframe[activeKeyframeId] ?? [];
    return cutoutIds.flatMap((cutoutId) => filteredObjectsByCutout.get(cutoutId) ?? []);
  }, [activeKeyframeId, filteredObjectsByCutout, summary?.cutout_ids_by_keyframe]);
  const activeKeyframeRawDetectionCount = useMemo(() => {
    if (!activeKeyframeId || !summary) {
      return 0;
    }
    return (summary.cutout_ids_by_keyframe[activeKeyframeId] ?? []).reduce(
      (count, cutoutId) => count + (summary.object_count_by_cutout[cutoutId] ?? 0),
      0,
    );
  }, [activeKeyframeId, summary]);
  const rawDetections = cutoutPayload?.detections ?? [];
  const filteredDetections = useMemo(
    () => postProcessDetections(rawDetections, bboxPostProcess),
    [rawDetections, bboxPostProcess],
  );
  const bboxAreaSliderMax = useMemo(() => {
    if (!rawDetections.length) {
      return 100_000;
    }
    const maxArea = Math.max(...rawDetections.map((item) => bboxArea(item.bbox)));
    return Math.max(1_000, Math.ceil(maxArea / 500) * 500);
  }, [rawDetections]);

  const keyframeEquirectPreviewUrl = activeKeyframeId
    ? indexKeyframeEquirectPreviewUrl(
        props.mapId,
        activeKeyframeId,
        null,
        undefined,
        { drawBoxes: false },
      )
    : null;
  const keyframeDepthPreviewUrl = activeKeyframeId
    ? indexKeyframeDepthPreviewUrl(props.mapId, activeKeyframeId)
    : null;
  const keyframePreviewUrl =
    panoramaViewMode === "depth"
      ? keyframeDepthPreviewUrl
      : keyframeEquirectPreviewUrl;
  const adjacentKeyframePreviewUrls = useMemo(() => {
    if (!activeKeyframeId || !summary) {
      return [];
    }
    const index = summary.keyframe_ids.indexOf(activeKeyframeId);
    if (index < 0) {
      return [];
    }
    return [summary.keyframe_ids[index - 1], summary.keyframe_ids[index + 1]]
      .filter((keyframeId): keyframeId is string => Boolean(keyframeId))
      .map((keyframeId) =>
        indexKeyframeEquirectPreviewUrl(
          props.mapId,
          keyframeId,
          null,
          undefined,
          { drawBoxes: false },
        ),
      );
  }, [activeKeyframeId, props.mapId, summary]);

  useEffect(() => {
    const controller = new AbortController();
    const preload = async () => {
      for (const url of adjacentKeyframePreviewUrls) {
        if (controller.signal.aborted) {
          return;
        }
        try {
          const response = await fetch(url, { signal: controller.signal });
          if (response.ok) {
            await response.blob();
          }
        } catch {
          // Preloading is opportunistic.
        }
      }
    };
    const idleWindow = window as Window & {
      requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (idleWindow.requestIdleCallback) {
      const handle = idleWindow.requestIdleCallback(() => void preload(), { timeout: 1000 });
      return () => {
        controller.abort();
        idleWindow.cancelIdleCallback?.(handle);
      };
    }
    const handle = window.setTimeout(() => void preload(), 200);
    return () => {
      controller.abort();
      window.clearTimeout(handle);
    };
  }, [adjacentKeyframePreviewUrls]);

  useEffect(() => {
    if (!isResizingVisualSplit) {
      return;
    }
    document.body.classList.add("explorer-visual-split-dragging");
    const handlePointerMove = (event: PointerEvent) => {
      const rect = visualSplitRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) {
        return;
      }
      const next = ((event.clientX - rect.left) / rect.width) * 100;
      setVisualSplitPercent(
        Math.max(VISUAL_SPLIT_MIN_PERCENT, Math.min(VISUAL_SPLIT_MAX_PERCENT, next)),
      );
    };
    const stopResize = () => {
      setIsResizingVisualSplit(false);
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize);
    window.addEventListener("pointercancel", stopResize);
    return () => {
      document.body.classList.remove("explorer-visual-split-dragging");
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    };
  }, [isResizingVisualSplit]);

  const startVisualSplitResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setIsResizingVisualSplit(true);
  };

  useEffect(() => {
    if (!keyframePreviewUrl) {
      setEquirectPreviewError(null);
      setEquirectPreviewSrc(null);
      setEquirectPreviewKeyframeId(null);
      setPhotosphereYawRad(null);
      return;
    }
    setPhotosphereYawRad(null);

    const controller = new AbortController();
    let objectUrl: string | null = null;
    const url = keyframePreviewUrl;

    async function loadEquirectPreview() {
      setEquirectPreviewError(null);
      try {
        const response = await fetch(url, {
          signal: controller.signal,
        });
        const contentType = response.headers.get("content-type") ?? "";
        if (!response.ok) {
          let detail = `${response.status} ${response.statusText}`;
          try {
            const payload = (await response.json()) as { detail?: unknown };
            if (typeof payload.detail === "string") {
              detail = payload.detail;
            }
          } catch {
            // ignore non-JSON error bodies
          }
          if (!controller.signal.aborted) {
            setEquirectPreviewError(detail);
          }
          return;
        }
        if (contentType.includes("text/html")) {
          if (!controller.signal.aborted) {
            setEquirectPreviewError(
              "The preview API returned the UI page instead of an image. Restart the object-search toolbox backend.",
            );
          }
          return;
        }
        if (!contentType.startsWith("image/")) {
          if (!controller.signal.aborted) {
            setEquirectPreviewError(
              `Unexpected preview response type: ${contentType || "unknown"}.`,
            );
          }
          return;
        }
        const blob = await response.blob();
        if (controller.signal.aborted) {
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        setEquirectPreviewSrc(objectUrl);
        setEquirectPreviewKeyframeId(activeKeyframeId);
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setEquirectPreviewError(`Could not load ${panoramaViewMode} preview: ${message}`);
      }
    }

    void loadEquirectPreview();

    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [activeKeyframeId, keyframePreviewUrl, panoramaViewMode]);

  useEffect(() => {
    setSelectedObjectId((current) => {
      if (current && filteredDetections.some((item) => item.id === current)) {
        return current;
      }
      return filteredDetections[0]?.id ?? null;
    });
  }, [filteredDetections, selectedCutoutId]);

  useEffect(() => {
    if (!shouldFocusSelectedDetectionRef.current || !selectedObjectId) {
      return;
    }
    shouldFocusSelectedDetectionRef.current = false;
    detectionRowRefs.current.get(selectedObjectId)?.focus();
  }, [selectedObjectId]);

  const livemapFocus = useMemo(() => {
    if (annotation.focusTarget) {
      return annotation.focusTarget;
    }
    if (!activeKeyframeId) {
      return null;
    }
    const marker = status?.markers?.find((item) => item.id === activeKeyframeId);
    if (!marker) {
      return null;
    }
    return {
      latitude: marker.latitude,
      longitude: marker.longitude,
      level: marker.level,
    };
  }, [activeKeyframeId, status?.markers, annotation.focusTarget]);

  useEffect(() => {
    setPage(1);
  }, [includeEmpty, keyframeFilter, pageSize, sortMode]);

  useEffect(() => {
    if (currentPage !== page) {
      setPage(currentPage);
    }
  }, [currentPage, page]);

  useEffect(() => {
    if (!visibleRows.length) {
      setSelectedCutoutId(null);
      return;
    }
    setSelectedCutoutId((current) =>
      current && visibleRows.some((row) => row.cutoutId === current)
        ? current
        : visibleRows[0].cutoutId,
    );
  }, [visibleRows]);

  useEffect(() => {
    if (!selectedCutoutId || !status?.available) {
      setCutoutPayload(null);
      setSelectedObjectId(null);
      return;
    }
    let cancelled = false;
    setIsLoadingCutout(true);
    fetchIndexCutout(props.mapId, selectedCutoutId)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setCutoutPayload(payload);
        setSelectedObjectId(null);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setCutoutPayload(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingCutout(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, selectedCutoutId, status?.available]);

  if (!props.isMapKnown) {
    return (
      <section className="object-search-explorer-panel page-section">
        <p className="error-box">Map not found in the current config.</p>
      </section>
    );
  }

  if (isLoading) {
    return (
      <section className="object-search-explorer-panel page-section">
        <p className="muted">Loading object-search explorer...</p>
      </section>
    );
  }

  if (!status?.available || !summary) {
    const checkedPaths = status?.checked_paths ?? [];
    return (
      <section className="object-search-explorer-panel page-section">
        {error ? (
          <DismissibleAlert variant="error" onDismiss={() => setError(null)}>
            {error}
          </DismissibleAlert>
        ) : (
          <p className="info-box">
            No object-search index database (object-search.db) found for this map.
          </p>
        )}
        {status?.map_path || props.map?.path ? (
          <p className="path-text">Map path: {status?.map_path ?? props.map?.path}</p>
        ) : null}
        {checkedPaths.length ? (
          <ul className="checked-paths-list">
            {checkedPaths.map((path) => (
              <li key={path}>
                <code>{path}</code>
              </li>
            ))}
          </ul>
        ) : null}
      </section>
    );
  }

  const totalObjects = Object.values(summary.object_count_by_cutout).reduce(
    (sum, count) => sum + count,
    0,
  );
  const selectedDetection =
    filteredDetections.find((item) => item.id === selectedObjectId) ?? null;
  const activeKeyframeIndex = activeKeyframeId
    ? summary.keyframe_ids.indexOf(activeKeyframeId)
    : -1;
  const hasPreviousKeyframe = activeKeyframeIndex > 0;
  const hasNextKeyframe =
    activeKeyframeIndex >= 0 && activeKeyframeIndex < summary.keyframe_ids.length - 1;

  const selectKeyframe = (keyframeId: string) => {
    if (!summary.keyframe_ids.includes(keyframeId) || keyframeId === activeKeyframeId) {
      return;
    }
    setKeyframeFilter(keyframeId);
    setSelectedCutoutId(null);
    setSelectedObjectId(null);
    setDepthPin(null);
    setDepthPinPopoverOpen(false);
  };

  const handleMarkerContextMenu = (
    markerId: string,
    position: { x: number; y: number },
  ) => {
    const prefix = "__annotation_";
    if (!markerId.startsWith(prefix)) {
      setAnnotationMenu(null);
      return;
    }
    const annotationId = markerId.slice(prefix.length);
    const exists = annotation.annotations.some(
      (item) => item.id === annotationId && item.annotationType === "point",
    );
    if (!exists) {
      setAnnotationMenu(null);
      return;
    }
    annotation.selectAnnotationMarker(markerId);
    setAnnotationMenu({ annotationId, x: position.x, y: position.y });
  };

  const showKeyframeForAnnotation = (annotationId: string) => {
    const ann = annotation.annotations.find((item) => item.id === annotationId);
    const keyframeId = ann?.source?.keyframeId ?? null;
    setAnnotationMenu(null);
    if (!ann || !keyframeId) {
      return;
    }
    // Orient the panorama toward the localization: convert the keyframe→annotation
    // compass bearing into a viewer yaw relative to the keyframe heading.
    const keyframeMarker = status?.markers?.find(
      (marker) => marker.id === keyframeId,
    );
    const coordinates = ann.coordinates as number[];
    const headingDeg = keyframeMarker?.heading_deg;
    if (
      keyframeMarker &&
      coordinates.length >= 2 &&
      headingDeg !== null &&
      headingDeg !== undefined
    ) {
      const bearing = bearingDegrees(
        { latitude: keyframeMarker.latitude, longitude: keyframeMarker.longitude },
        { latitude: coordinates[1], longitude: coordinates[0] },
      );
      const yawRad = Math.atan2(
        Math.sin(((bearing - headingDeg) * Math.PI) / 180),
        Math.cos(((bearing - headingDeg) * Math.PI) / 180),
      );
      // Pre-set the preserved bearing so the initial yaw on keyframe switch is
      // already correct, and bump the token so an already-open keyframe re-orients.
      preservedCompassBearingDegRef.current = bearing;
      setPhotosphereOrient((prev) => ({ yawRad, token: (prev?.token ?? 0) + 1 }));
    }
    selectKeyframe(keyframeId);
    setKeyframeLink({ annotationId, keyframeId });
  };

  const hideKeyframeLink = () => {
    setAnnotationMenu(null);
    setKeyframeLink(null);
  };

  const selectAdjacentKeyframe = (direction: -1 | 1) => {
    if (activeKeyframeIndex < 0) {
      return;
    }
    const nextIndex = Math.min(
      summary.keyframe_ids.length - 1,
      Math.max(0, activeKeyframeIndex + direction),
    );
    const nextKeyframeId = summary.keyframe_ids[nextIndex];
    if (!nextKeyframeId || nextKeyframeId === activeKeyframeId) {
      return;
    }
    selectKeyframe(nextKeyframeId);
  };

  const selectAdjacentDetection = (direction: -1 | 1) => {
    if (!filteredDetections.length) {
      return;
    }
    const currentIndex = selectedObjectId
      ? filteredDetections.findIndex((item) => item.id === selectedObjectId)
      : -1;
    const nextIndex =
      currentIndex === -1
        ? direction > 0
          ? 0
          : filteredDetections.length - 1
        : Math.min(
            filteredDetections.length - 1,
            Math.max(0, currentIndex + direction),
          );
    const nextDetection = filteredDetections[nextIndex];
    if (nextDetection.id !== selectedObjectId) {
      shouldFocusSelectedDetectionRef.current = true;
    }
    setSelectedObjectId(nextDetection.id);
  };

  const handleDetectionRowKeyDown = (
    event: ReactKeyboardEvent<HTMLTableRowElement>,
  ) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      selectAdjacentDetection(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      selectAdjacentDetection(-1);
    }
  };

  const updateResolvedDepthPin = (
    requestId: string,
    payload: DepthPinResponse,
  ) => {
    setDepthPin((current) => {
      if (!current || current.requestId !== requestId) {
        return current;
      }
      return {
        ...current,
        status: "resolved",
        erpXRatio: payload.erp_u,
        erpYRatio: payload.erp_v,
        depthM: payload.depth_m,
        latitude: payload.latitude,
        longitude: payload.longitude,
        altitude: payload.altitude,
        level: payload.level,
        pointWorldWds: payload.point_world_wds,
        message: `Depth ${payload.depth_m.toFixed(2)} m`,
      };
    });
  };

  const updateRejectedDepthPin = (requestId: string, message: string) => {
    setDepthPin((current) => {
      if (!current || current.requestId !== requestId) {
        return current;
      }
      return {
        ...current,
        status: "error",
        message,
      };
    });
  };

  const readClickRatio = (event: ReactMouseEvent<HTMLElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      xRatio: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      yRatio: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  };

  const placeDepthPin = (
    source: "erp" | "cutout",
    keyframeId: string,
    xRatio: number,
    yRatio: number,
    cutoutId: string | null = null,
  ) => {
    const requestId = `${Date.now()}-${Math.random()}`;
    if (annotation.enabled) {
      annotation.beginDraft(requestId, {
        keyframeId,
        projection: source,
        cutoutId,
        xRatio,
        yRatio,
      });
    }
    setDepthPin({
      requestId,
      source,
      keyframeId,
      cutoutId,
      xRatio,
      yRatio,
      erpXRatio: source === "erp" ? xRatio : null,
      erpYRatio: source === "erp" ? yRatio : null,
      status: "resolving",
      depthM: null,
      latitude: null,
      longitude: null,
      altitude: null,
      level: null,
      pointWorldWds: null,
      message: "Resolving depth...",
    });
    void resolveDepthPin(props.mapId, keyframeId, {
      projection: source,
      x_ratio: xRatio,
      y_ratio: yRatio,
      cutout_id: cutoutId ?? undefined,
      sample_radius: 3,
      min_depth_m: DEPTH_PIN_MIN_DEPTH_M,
    })
      .then((payload) => {
        updateResolvedDepthPin(requestId, payload);
        if (annotation.enabled) {
          annotation.resolveDraft(requestId, payload);
        }
      })
      .catch((err: Error) => {
        updateRejectedDepthPin(requestId, err.message);
        if (annotation.enabled) {
          annotation.rejectDraft(requestId, err.message);
        }
      });
  };

  const placePhotosphereDepthPin = (xRatio: number, yRatio: number) => {
    if (!activeKeyframeId) {
      return;
    }
    placeDepthPin("erp", activeKeyframeId, xRatio, yRatio);
  };

  const renderKeyframeStepper = (className = "keyframe-stepper") => (
    <div className={className} aria-label="Switch keyframe">
      <button
        type="button"
        aria-label="Previous keyframe"
        title="Previous keyframe"
        disabled={!hasPreviousKeyframe}
        onClick={() => selectAdjacentKeyframe(-1)}
      >
        &lt;
      </button>
      <button
        type="button"
        aria-label="Next keyframe"
        title="Next keyframe"
        disabled={!hasNextKeyframe}
        onClick={() => selectAdjacentKeyframe(1)}
      >
        &gt;
      </button>
    </div>
  );

  return (
    <section className="object-search-explorer-panel">
      <div className="object-search-explorer-header">
        <div>
          <p className="eyebrow">Object Search Explorer</p>
          <h2>Cutouts and detected boxes</h2>
        </div>
        <p className="path-text">{summary.index_path}</p>
      </div>

      {error ? (
        <DismissibleAlert variant="error" onDismiss={() => setError(null)}>
          {error}
        </DismissibleAlert>
      ) : null}
      {annotation.errorMessage ? (
        <DismissibleAlert
          variant="error"
          onDismiss={() => annotation.setErrorMessage(null)}
        >
          {annotation.errorMessage}
        </DismissibleAlert>
      ) : null}
      {annotation.infoMessage ? (
        <DismissibleAlert
          variant="info"
          onDismiss={() => annotation.setInfoMessage(null)}
        >
          {annotation.infoMessage}
        </DismissibleAlert>
      ) : null}

      <div className="metrics-row">
        <span>Keyframes {summary.keyframe_ids.length}</span>
        <span>Cutouts {allRows.length}</span>
        <span>Objects {totalObjects}</span>
        <span>Visible {visibleRows.length}</span>
      </div>

      <ExplorerAnnotationControls workspace={annotation} />

      <div
        ref={visualSplitRef}
        className="object-search-visual-split"
        style={{ "--visual-map-width": `${visualSplitPercent}%` } as CSSProperties}
      >
        <div className="object-search-visual-pane">
          <section className="object-search-explorer-livemap">
            <div className="object-search-explorer-livemap-header">
              <h3>Keyframe map</h3>
              <div className="object-search-explorer-livemap-toggles">
                {keyframeGraph?.available ? (
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={showKeyframeGraph}
                      onChange={(event) => setShowKeyframeGraph(event.target.checked)}
                    />
                    Show graph ({keyframeGraph.edges.length} edges)
                  </label>
                ) : null}
                <div className="object-search-livemap-roi-toolbar">
                  <span>ROI</span>
                  <button
                    type="button"
                    className={`secondary-button${
                      annotation.roiActive ? " is-active" : ""
                    }`}
                    aria-pressed={annotation.roiActive}
                    onClick={annotation.toggleRoi}
                  >
                    {annotation.roiActive ? "Drawing..." : "Draw"}
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={!annotation.roiActive && !annotation.roiPolygon}
                    onClick={annotation.clearRoi}
                  >
                    Clear
                  </button>
                  <span className="object-search-livemap-roi-summary">
                    {annotation.roiCounts
                      ? `Total ${annotation.roiCounts.total}`
                      : `Level ${annotation.currentLevel === null ? "-" : annotation.currentLevel}`}
                  </span>
                </div>
              </div>
            </div>
            {keyframeGraph?.error ? (
              <p className="map-caption">Keyframe graph unavailable: {keyframeGraph.error}</p>
            ) : null}
            {emmid === null ? (
              <p className="info-box">
                Map preview unavailable: no <code>emmid</code> configured for this map.
              </p>
            ) : !summary.keyframe_ids.length ? (
              <p className="muted">The index does not contain any keyframes.</p>
            ) : !(status?.markers?.length ?? 0) ? (
              <p className="muted">
                No keyframe coordinates could be resolved from local map metadata.
              </p>
            ) : (
              <>
                <div
                  className="object-search-explorer-livemap-frame"
                  style={{ height: livemapFrameHeight }}
                >
                  <LivemapAnnotation
                    emmid={emmid}
                    markers={livemapMarkers}
                    segments={livemapSegments}
                    polygons={livemapPolygons}
                    draftVertices={[]}
                    draftClosed={false}
                    pendingPoint={null}
                    activeColor={null}
                    mode="inspect"
                    focusTarget={livemapFocus}
                    onMapClick={() => {}}
                    cone={cone}
                    coneColor={VIEW_CONE_COLOR}
                    roiActive={annotation.roiActive}
                    roiPolygon={annotation.roiPolygon}
                    onRoiChange={annotation.setRoiPolygon}
                    onLevelChange={annotation.setCurrentLevel}
                    onMarkerContextMenu={handleMarkerContextMenu}
                    segmentHoverMarkers={graphHoverMarkers}
                    onSegmentMarkerClick={selectGraphKeyframe}
                    markerPopover={
                      depthPinPopoverOpen &&
                      depthPin?.status === "resolved" &&
                      depthPin.latitude !== null &&
                      depthPin.longitude !== null
                        ? {
                            latitude: depthPin.latitude,
                            longitude: depthPin.longitude,
                            content: (
                              <div className="object-search-depth-popover">
                                <button
                                  type="button"
                                  className="object-search-depth-popover-close"
                                  aria-label="Close"
                                  onClick={() => setDepthPinPopoverOpen(false)}
                                >
                                  ×
                                </button>
                                <strong>Depth point</strong>
                                {depthPin.depthM !== null ? (
                                  <span>Depth: {depthPin.depthM.toFixed(2)} m</span>
                                ) : null}
                                {depthPin.level !== null ? (
                                  <span>Floor: {depthPin.level}</span>
                                ) : null}
                                <span>
                                  {depthPin.latitude.toFixed(7)},{" "}
                                  {depthPin.longitude.toFixed(7)}
                                </span>
                                {depthPin.altitude !== null ? (
                                  <span>Altitude: {depthPin.altitude.toFixed(2)} m</span>
                                ) : null}
                                <span>Keyframe: {depthPin.keyframeId}</span>
                                <button
                                  type="button"
                                  className="danger-button"
                                  onClick={() => {
                                    setDepthPin(null);
                                    setDepthPinPopoverOpen(false);
                                    annotation.discardDraft();
                                  }}
                                >
                                  Remove
                                </button>
                              </div>
                            ),
                          }
                        : null
                    }
                    onMarkerClick={(markerId) => {
                      if (markerId === "__depth_pin") {
                        setDepthPinPopoverOpen(true);
                        return;
                      }
                      setDepthPinPopoverOpen(false);
                      if (annotation.selectAnnotationMarker(markerId)) {
                        return;
                      }
                      selectKeyframe(markerId);
                    }}
                  />
                </div>
                <p className="map-caption">
                  {status?.markers?.length ?? 0} / {summary.keyframe_ids.length} keyframes have
                  resolvable map coordinates.
                  {showKeyframeGraph
                    ? " Hover the graph to reveal the closest keyframe; click the graph or revealed point to filter cutouts."
                    : " Enable the graph to reveal nearby keyframes on hover."}
                </p>
              </>
            )}
          </section>

          {showLivemapResizer ? (
            <div
              className={`object-search-explorer-livemap-resizer${
                isResizingLivemap ? " is-dragging" : ""
              }`}
              role="separator"
              aria-orientation="horizontal"
              aria-label="Resize keyframe map"
              aria-valuenow={livemapFrameHeight}
              aria-valuemin={200}
              aria-valuemax={720}
              title="Drag to resize keyframe map"
              onPointerDown={startLivemapResize}
            />
          ) : null}
        </div>

        <div
          className={`object-search-visual-splitter${
            isResizingVisualSplit ? " is-dragging" : ""
          }`}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize map and photosphere columns"
          aria-valuenow={Math.round(visualSplitPercent)}
          aria-valuemin={VISUAL_SPLIT_MIN_PERCENT}
          aria-valuemax={VISUAL_SPLIT_MAX_PERCENT}
          title="Drag to resize map and photosphere"
          onPointerDown={startVisualSplitResize}
        />

        <div className="object-search-visual-pane">
          {keyframePreviewUrl ? (
            <section className="object-search-keyframe-equirect">
              <div className="object-search-keyframe-equirect-header">
                <h3>Image</h3>
                <span className="muted">
                  Keyframe {activeKeyframeId}
                </span>
                <label className="object-search-panorama-view-selector">
                  View
                  <select
                    value={panoramaViewMode}
                    onChange={(event) =>
                      setPanoramaViewMode(event.target.value as PanoramaViewMode)
                    }
                  >
                    <option value="image">Image</option>
                    <option value="depth">Depth</option>
                  </select>
                </label>
                <span className="muted">
                  Detections {activeKeyframeRawDetectionCount} | Filtered bboxes{" "}
                  {activeKeyframeDetections.length}
                </span>
                <label className="inline-check">
                  <input
                    type="checkbox"
                    checked={showPhotosphereBoxes}
                    disabled={panoramaViewMode !== "image"}
                    onChange={(event) => setShowPhotosphereBoxes(event.target.checked)}
                  />
                  Show boxes
                </label>
                {renderKeyframeStepper()}
              </div>
              <div
                className="object-search-keyframe-equirect-container"
                style={{ minHeight: equirectFrameHeight }}
              >
                {equirectPreviewError ? (
                  <p className="error-box">{equirectPreviewError}</p>
                ) : equirectPreviewSrc ? (
                  <Suspense fallback={<p className="muted">Loading photosphere viewer...</p>}>
                    <EquirectPhotoSphereViewer
                      keyframeId={equirectPreviewKeyframeId ?? activeKeyframeId}
                      imageSrc={equirectPreviewSrc}
                      height={equirectFrameHeight}
                      initialYawRad={initialPhotosphereView.yawRad}
                      initialTextureYRatio={initialPhotosphereView.textureYRatio}
                      orientYawRad={photosphereOrient?.yawRad}
                      orientToken={photosphereOrient?.token}
                      detections={
                        panoramaViewMode === "image" &&
                        showPhotosphereBoxes &&
                        equirectPreviewKeyframeId === activeKeyframeId
                          ? activeKeyframeDetections
                          : []
                      }
                      selectedObjectId={selectedObjectId}
                      depthPin={
                        equirectPreviewKeyframeId === activeKeyframeId &&
                        depthPin?.keyframeId === activeKeyframeId &&
                        depthPin.erpXRatio !== null &&
                        depthPin.erpYRatio !== null
                          ? {
                              xRatio: depthPin.erpXRatio,
                              yRatio: depthPin.erpYRatio,
                              status: depthPin.status,
                            }
                          : null
                      }
                      polygonForDetection={polygonForPhotosphereDetection}
                      onDepthPin={placePhotosphereDepthPin}
                      allowDepthPinOnMarker={annotation.enabled}
                      navigationCandidates={photosphereNavigationCandidates}
                      onNavigate={selectKeyframe}
                      onViewChange={handlePhotosphereViewChange}
                    />
                  </Suspense>
                ) : (
                  <p className="muted">Loading {panoramaViewMode} preview...</p>
                )}
              </div>
              <p className="map-caption">
                {panoramaViewMode === "depth"
                  ? "Depth preview is colorized from near red to far blue; invalid depth pixels are black."
                  : showPhotosphereBoxes
                  ? "Post-processed bounding boxes from all cutouts in this keyframe, projected onto the photosphere."
                  : "Bounding boxes are hidden for smoother photosphere navigation."}
              </p>
              {depthPin?.keyframeId === activeKeyframeId ? (
                <p className={`object-search-depth-pin-caption is-${depthPin.status}`}>
                  {depthPin.message}
                  {depthPin.status === "resolved" && depthPin.latitude !== null && depthPin.longitude !== null
                    ? ` | ${depthPin.latitude.toFixed(7)}, ${depthPin.longitude.toFixed(7)}`
                    : ""}
                </p>
              ) : null}
              {annotation.enabled ? (
                <div className="object-search-photosphere-annotation-actions">
                  <div
                    className="object-search-photosphere-class-picker"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <button
                      type="button"
                      className="object-search-photosphere-annotation-class"
                      aria-haspopup="listbox"
                      aria-expanded={photosphereClassMenuOpen}
                      onClick={() => setPhotosphereClassMenuOpen((open) => !open)}
                    >
                      {annotation.activeClass?.name ?? "No class selected"}
                    </button>
                    {photosphereClassMenuOpen ? (
                      <ul
                        className="object-search-photosphere-class-menu"
                        role="listbox"
                        aria-label="Annotation classes"
                      >
                        {annotation.pointClasses.length ? (
                          annotation.pointClasses.map((item) => (
                            <li key={`${item.name}::${item.annotationType}`}>
                              <button
                                type="button"
                                className={`object-search-photosphere-class-option${
                                  annotation.activeClassKey === `${item.name}::${item.annotationType}`
                                    ? " is-active"
                                    : ""
                                }`}
                                role="option"
                                aria-selected={
                                  annotation.activeClassKey === `${item.name}::${item.annotationType}`
                                }
                                onClick={() => {
                                  annotation.setActiveClassKey(`${item.name}::${item.annotationType}`);
                                  setPhotosphereClassMenuOpen(false);
                                }}
                              >
                                <span
                                  className="color-swatch"
                                  style={{ background: item.color }}
                                />
                                <span>{item.name}</span>
                              </button>
                            </li>
                          ))
                        ) : (
                          <li className="object-search-photosphere-class-empty">
                            No point classes
                          </li>
                        )}
                      </ul>
                    ) : null}
                  </div>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={
                      annotation.draft?.status !== "resolved" ||
                      !annotation.activeClass
                    }
                    onClick={annotation.saveDraft}
                  >
                    Save annotation
                  </button>
                </div>
              ) : null}
            </section>
          ) : null}

          {keyframePreviewUrl && equirectPreviewSrc ? (
            <div
              className={`object-search-explorer-equirect-resizer${
                isResizingEquirect ? " is-dragging" : ""
              }`}
              role="separator"
              aria-orientation="horizontal"
              aria-label="Resize image preview"
              aria-valuenow={equirectFrameHeight}
              aria-valuemin={120}
              aria-valuemax={600}
              title="Drag to resize image preview"
              onPointerDown={startEquirectResize}
            />
          ) : null}
        </div>
      </div>

      <ExplorerAnnotationList workspace={annotation} />

      <CollapsibleSection
        title="Cutout/keyframe explorer"
        summary={`${visibleRows.length} visible | ${allRows.length} cutouts`}
        sectionClassName="object-search-explorer-browser"
        defaultOpen
      >
        <div className="object-search-explorer-layout">
          <div className="object-search-explorer-main">
            <div className="keyframe-sticky-bar">
              <div className="keyframe-filter-control">
                <div className="keyframe-filter-header">
                  <span>Keyframe</span>
                  {renderKeyframeStepper("keyframe-stepper keyframe-stepper--compact")}
                </div>
                <select
                  value={keyframeFilter}
                  onChange={(event) => setKeyframeFilter(event.target.value)}
                >
                  <option value="all">All keyframes</option>
                  {summary.keyframe_ids.map((keyframeId) => (
                    <option key={keyframeId} value={keyframeId}>
                      {keyframeId} |{" "}
                      {(summary.cutout_ids_by_keyframe[keyframeId] ?? []).length} cutouts
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="object-search-explorer-controls">
              <label>
                Sort
                <select
                  value={sortMode}
                  onChange={(event) => setSortMode(event.target.value as SortMode)}
                >
                  <option value="keyframe">Keyframe order</option>
                  <option value="objects-desc">Most boxes first</option>
                  <option value="objects-asc">Fewest boxes first</option>
                </select>
              </label>

            <label>
              Cutouts per page
              <select
                value={String(pageSize)}
                onChange={(event) => setPageSize(Number(event.target.value))}
              >
                {PAGE_SIZE_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Columns
              <select
                value={String(columns)}
                onChange={(event) => setColumns(Number(event.target.value))}
              >
                {COLUMN_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <div className="checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={includeEmpty}
                  onChange={(event) => setIncludeEmpty(event.target.checked)}
                />
                Include empty cutouts
              </label>
            </div>
          </div>

          <div className="object-search-explorer-pagination">
            <button
              type="button"
              onClick={() => setPage((value) => Math.max(1, value - 1))}
              disabled={currentPage <= 1}
            >
              Previous
            </button>
            <span>
              Page {currentPage} / {totalPages}
            </span>
            <button
              type="button"
              onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
              disabled={currentPage >= totalPages}
            >
              Next
            </button>
          </div>

          {!pageRows.length ? (
            <p className="muted">No cutouts match the current filters.</p>
          ) : (
            <div
              className="object-search-cutout-grid"
              style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
            >
              {pageRows.map((row) => {
                const boxCounts = boxCountsByCutout.get(row.cutoutId);
                const overlayDetections = filteredObjectsByCutout.get(row.cutoutId) ?? [];
                const faceSize = Math.max(1, manifestNumber(manifest, "cubemap_face_size", 512));
                const boxCountLabel = boxCounts
                  ? `${boxCounts.filtered} / ${boxCounts.raw} boxes`
                  : `${row.objectCount} boxes`;
                return (
                  <button
                    className={`object-search-cutout-card${
                      row.cutoutId === selectedCutoutId ? " is-selected" : ""
                    }`}
                    type="button"
                    key={row.cutoutId}
                    onClick={() => {
                      setSelectedCutoutId(row.cutoutId);
                      setSelectedObjectId(null);
                    }}
                  >
                    <span className="object-search-cutout-image-wrap">
                      <img
                        src={indexCutoutPreviewUrl(
                          props.mapId,
                          row.cutoutId,
                          null,
                          { drawBoxes: false },
                        )}
                        alt={`Cutout ${row.cutoutId} with filtered detected boxes`}
                        title="Click to place a depth pin"
                        loading="lazy"
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedCutoutId(row.cutoutId);
                          setSelectedObjectId(null);
                          const { xRatio, yRatio } = readClickRatio(event);
                          placeDepthPin("cutout", row.keyframeId, xRatio, yRatio, row.cutoutId);
                        }}
                      />
                      {overlayDetections.length ? (
                        <svg
                          className="object-search-bbox-overlay"
                          viewBox={`0 0 ${faceSize} ${faceSize}`}
                          aria-hidden="true"
                        >
                          {overlayDetections.map((item) => {
                            const [x0, y0, x1, y1] = item.bbox;
                            return (
                              <g key={item.id}>
                                <rect
                                  x={x0}
                                  y={y0}
                                  width={Math.max(0, x1 - x0)}
                                  height={Math.max(0, y1 - y0)}
                                  className={item.id === selectedObjectId ? "is-selected" : ""}
                                  vectorEffect="non-scaling-stroke"
                                />
                                {item.label ? (
                                  <text x={x0 + 3} y={Math.max(12, y0 - 4)}>
                                    {item.label}
                                  </text>
                                ) : null}
                              </g>
                            );
                          })}
                        </svg>
                      ) : null}
                      {depthPin?.source === "cutout" && depthPin.cutoutId === row.cutoutId ? (
                        <span
                          className={`object-search-depth-pin is-${depthPin.status}`}
                          style={{
                            left: `${depthPin.xRatio * 100}%`,
                            top: `${depthPin.yRatio * 100}%`,
                          }}
                        />
                      ) : null}
                    </span>
                    <span className="object-search-cutout-card-meta">
                      <strong>{row.cutoutId}</strong>
                      <small>
                        keyframe {row.keyframeId} | {boxCountLabel}
                      </small>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <aside className="object-search-explorer-inspector">
          <h3>Selected cutout</h3>
          {selectedRow ? (
            <>
              <div className="metrics-row">
                <span>Keyframe {selectedRow.keyframeId}</span>
                <span>Cutout {selectedRow.cutoutId}</span>
                <span>
                  {filteredDetections.length} / {rawDetections.length} boxes
                </span>
              </div>

              <BboxPostProcessControls
                params={bboxPostProcess}
                areaSliderMax={bboxAreaSliderMax}
                rawCount={rawDetections.length}
                filteredCount={filteredDetections.length}
                onChange={setBboxPostProcess}
                onReset={resetBboxPostProcess}
              />

              {isLoadingCutout ? <p className="muted">Loading bbox metadata...</p> : null}

              {rawDetections.length && !filteredDetections.length ? (
                <p className="info-box">All boxes were removed by the current post-processing filters.</p>
              ) : null}

              {filteredDetections.length ? (
                <>
                  <label>
                    Detected box
                    <select
                      value={selectedObjectId ?? ""}
                      onChange={(event) => setSelectedObjectId(event.target.value)}
                    >
                      {filteredDetections.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.id} | bbox {item.bbox.join(", ")}
                        </option>
                      ))}
                    </select>
                  </label>

                  {selectedDetection ? (
                    <CollapsibleSection title="Selected object JSON" defaultOpen={false}>
                      <pre className="debug-json">
                        {JSON.stringify(
                          {
                            ...selectedDetection,
                            bbox_area: Math.round(bboxArea(selectedDetection.bbox)),
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </CollapsibleSection>
                  ) : null}

                  <CollapsibleSection
                    title="Filtered detected boxes"
                    summary={`${filteredDetections.length} boxes`}
                    defaultOpen
                  >
                    <div className="table-wrap object-search-box-table">
                      <table>
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Label</th>
                            <th>Source</th>
                            <th>Bbox</th>
                            <th>Area</th>
                            <th>OCR</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredDetections.map((item) => (
                            <tr
                              key={item.id}
                              ref={(node) => {
                                if (node) {
                                  detectionRowRefs.current.set(item.id, node);
                                } else {
                                  detectionRowRefs.current.delete(item.id);
                                }
                              }}
                              className={item.id === selectedObjectId ? "is-selected" : ""}
                              tabIndex={0}
                              aria-selected={item.id === selectedObjectId}
                              onClick={(event) => {
                                setSelectedObjectId(item.id);
                                event.currentTarget.focus();
                              }}
                              onKeyDown={handleDetectionRowKeyDown}
                            >
                              <td>{item.id}</td>
                              <td>{item.label || "-"}</td>
                              <td>{item.detection_source || "-"}</td>
                              <td>{item.bbox.join(", ")}</td>
                              <td>{Math.round(bboxArea(item.bbox))}</td>
                              <td>{item.ocr_key || item.ocr_text || "-"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>

                  <CollapsibleSection title="Preview debug" defaultOpen={false}>
                    <pre className="debug-json">
                      {JSON.stringify(cutoutPayload?.preview_debug ?? {}, null, 2)}
                    </pre>
                  </CollapsibleSection>
                </>
              ) : !isLoadingCutout ? (
                <p className="muted">No detected boxes are stored for this cutout.</p>
              ) : null}
            </>
          ) : (
            <p className="muted">Select a cutout to inspect its bounding boxes.</p>
          )}
        </aside>
      </div>
      </CollapsibleSection>
      {annotationMenu
        ? (() => {
            const menuAnnotation = annotation.annotations.find(
              (item) => item.id === annotationMenu.annotationId,
            );
            const keyframeId = menuAnnotation?.source?.keyframeId ?? null;
            const isLinked =
              keyframeLink?.annotationId === annotationMenu.annotationId;
            return (
              <ul
                className="annotation-context-menu"
                style={{ left: annotationMenu.x, top: annotationMenu.y }}
                onClick={(event) => event.stopPropagation()}
              >
                <li>
                  <button
                    type="button"
                    className="annotation-context-menu-item"
                    disabled={!keyframeId}
                    onClick={() =>
                      showKeyframeForAnnotation(annotationMenu.annotationId)
                    }
                  >
                    {keyframeId ? "Show keyframe" : "No reference keyframe"}
                  </button>
                </li>
                {isLinked ? (
                  <li>
                    <button
                      type="button"
                      className="annotation-context-menu-item"
                      onClick={hideKeyframeLink}
                    >
                      Hide keyframe link
                    </button>
                  </li>
                ) : null}
                <li>
                  <button
                    type="button"
                    className="annotation-context-menu-item is-danger"
                    onClick={() => {
                      annotation.deleteAnnotation(annotationMenu.annotationId);
                      setAnnotationMenu(null);
                      if (
                        keyframeLink?.annotationId === annotationMenu.annotationId
                      ) {
                        setKeyframeLink(null);
                      }
                    }}
                  >
                    Delete
                  </button>
                </li>
              </ul>
            );
          })()
        : null}
    </section>
  );
}

export default ObjectSearchExplorerPanel;
