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
  fetchKeyframeGraph,
  fetchMetadataRow,
  fetchMetadataRows,
  fetchMetadataStatus,
  indexKeyframeDepthPreviewUrl,
  indexKeyframeEquirectPreviewUrl,
  projectWorldPoint,
  resolveDepthPin,
  rowRenderUrl,
  rowThumbnailUrl,
} from "../index-explorer/api";
import type {
  DepthPinResponse,
  KeyframeGraphResponse,
  KeyframeSummary,
  MetadataRowDetail,
  MetadataRowRecord,
  MetadataStatusResponse,
  MetadataSummary,
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

/** How a keyframe stands with respect to the live pgvector index. */
type KeyframeIndexState = "indexed" | "pruned" | "not-prepared" | "unknown";

type SortMode = "keyframe" | "objects-desc" | "objects-asc";
type PanoramaViewMode = "image" | "depth";
type DepthImagePin = {
  requestId: string;
  source: "erp" | "cutout";
  keyframeId: string;
  rowIndex: number | null;
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

// Keyframes per page, not proposals: one ERP carries 70-270 proposals, so the old
// [6, 12, 24, 48] would have put well over a thousand thumbnails in one grid.
const PAGE_SIZE_OPTIONS = [1, 2, 4, 8];
const COLUMN_OPTIONS = [1, 2, 3, 4, 5];
const KEYFRAME_MARKER_COLOR = "#9ca3af";
const KEYFRAME_MARKER_SELECTED_COLOR = "#16a34a";
const KEYFRAME_MARKER_RADIUS = 6;
const KEYFRAME_MARKER_SELECTED_RADIUS = 8;
/** Keyframes with rows in the parquet but pruned from pgvector at ingest time. */
const KEYFRAME_MARKER_PRUNED_COLOR = "#f97316";
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

/**
 * The four corners of a row's reconstructed ERP box, in texture ratios.
 *
 * The box arrives ready-made in `erp_bbox_ratios`, so there is no geometry here any
 * more — the 45 lines of cubemap algebra this replaces lived in two copies (here and
 * in the backend) and had to agree. `u` is intentionally left unwrapped: the viewer
 * reads it as a yaw, where 1.02 and 0.02 are the same direction.
 */
function bboxPolygonRatios(row: MetadataRowRecord): Array<[number, number]> {
  const [u0, v0, u1, v1] = row.erp_bbox_ratios;
  return [
    [u0, v0],
    [u1, v0],
    [u1, v1],
    [u0, v1],
  ];
}

/**
 * Tri-state per keyframe, because "has a pose", "has proposals" and "is in the live
 * index" are three different things and conflating them is what this panel exists to
 * prevent. `unknown` means Postgres did not answer — never an error.
 */
function keyframeIndexState(summary: KeyframeSummary | undefined): KeyframeIndexState {
  if (!summary) {
    return "not-prepared";
  }
  if (summary.ingested === null) {
    return "unknown";
  }
  return summary.ingested > 0 ? "indexed" : "pruned";
}


function ObjectSearchExplorerPanel(props: Props) {
  const [status, setStatus] = useState<MetadataStatusResponse | null>(null);
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
  const [pageSize, setPageSize] = useState(1);
  const [columns, setColumns] = useState(3);
  const [page, setPage] = useState(1);
  const [visualSplitPercent, setVisualSplitPercent] = useState(45);
  const [isResizingVisualSplit, setIsResizingVisualSplit] = useState(false);

  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);
  const [rowDetail, setRowDetail] = useState<MetadataRowDetail | null>(null);
  const [isLoadingRow, setIsLoadingRow] = useState(false);
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
  const detectionRowRefs = useRef(new Map<number, HTMLTableRowElement>());
  const shouldFocusSelectedDetectionRef = useRef(false);
  const { params: bboxPostProcess, setParams: setBboxPostProcess, resetParams: resetBboxPostProcess } =
    useBboxPostProcessParams();
  const [metadataRows, setMetadataRows] = useState<MetadataRowRecord[]>([]);
  const annotation = useExplorerAnnotationWorkspace(props.mapId);

  const summary: MetadataSummary | null = status?.summary ?? null;
  const polygonForPhotosphereDetection = useCallback(
    (row: MetadataRowRecord) => bboxPolygonRatios(row),
    [],
  );
  /**
   * `indexOf` over a 400-element array, in the middle of a render, is how this used
   * to answer "is this keyframe indexed?" — several times per frame.
   */
  const keyframeSummaryById = useMemo(() => {
    const byId = new Map<string, KeyframeSummary>();
    for (const item of summary?.keyframes ?? []) {
      byId.set(item.id, item);
    }
    return byId;
  }, [summary?.keyframes]);
  const preparedKeyframeIds = useMemo(
    () => (summary?.keyframes ?? []).map((item) => item.id),
    [summary?.keyframes],
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
    fetchMetadataStatus(props.mapId, null)
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

  /**
   * The page's rows after post-processing, grouped by keyframe.
   *
   * One row is one proposal *and* one detection in v2, so the old three-level
   * `keyframe → cutout → objects` walk collapses to a single grouping.
   */
  const filteredRowsByKeyframe = useMemo(() => {
    const grouped = new Map<string, MetadataRowRecord[]>();
    for (const row of metadataRows) {
      const list = grouped.get(row.keyframe_id) ?? [];
      list.push(row);
      grouped.set(row.keyframe_id, list);
    }
    const out = new Map<string, MetadataRowRecord[]>();
    for (const [keyframeId, rows] of grouped) {
      out.set(keyframeId, postProcessDetections(rows, bboxPostProcess));
    }
    return out;
  }, [metadataRows, bboxPostProcess]);

  /** Keyframes that have proposals, in parquet order — the browsable unit. */
  const visibleKeyframes = useMemo(() => {
    const keyframes = (summary?.keyframes ?? [])
      .filter((item) => keyframeFilter === "all" || item.id === keyframeFilter)
      .filter((item) => includeEmpty || item.row_count > 0);
    const sorted = [...keyframes];
    if (sortMode === "objects-desc") {
      sorted.sort((a, b) => b.row_count - a.row_count || a.id.localeCompare(b.id));
    } else if (sortMode === "objects-asc") {
      sorted.sort((a, b) => a.row_count - b.row_count || a.id.localeCompare(b.id));
    }
    return sorted;
  }, [summary?.keyframes, includeEmpty, keyframeFilter, sortMode]);

  const emmid = props.map?.emmid ?? null;
  const { height: livemapFrameHeight, isDragging: isResizingLivemap, startResize: startLivemapResize } =
    useLivemapFrameHeight();
  const { height: equirectFrameHeight, isDragging: isResizingEquirect, startResize: startEquirectResize } =
    useEquirectFrameHeight();

  const firstNavigableKeyframeId = useMemo(() => {
    let best: string | null = null;
    for (const marker of status?.markers ?? []) {
      if (best === null || Number(marker.id) < Number(best)) {
        best = marker.id;
      }
    }
    return best;
  }, [status?.markers]);

  const selectedKeyframeForMap = keyframeFilter !== "all" ? keyframeFilter : null;
  // Pagination is keyframe-major: one ERP and its proposals is how the panel is
  // actually used, and it keeps the row fetch to one request per page.
  const totalPages = Math.max(1, Math.ceil(visibleKeyframes.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageKeyframes = visibleKeyframes.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize,
  );
  const pageKeyframeIds = pageKeyframes.map((item) => item.id);
  const pageKeyframeIdsKey = pageKeyframeIds.join("|");
  const pageRows = pageKeyframeIds.flatMap(
    (keyframeId) => filteredRowsByKeyframe.get(keyframeId) ?? [],
  );
  const pageRowIndexKey = pageRows.map((row) => row.row_index).join("|");
  const selectedPageRow =
    pageRows.find((row) => row.row_index === selectedRowIndex) ?? null;
  const activeRow = selectedPageRow ?? pageRows[0] ?? null;

  // Falls back to the first keyframe with a pose, so the panorama, the depth preview
  // and the depth pin still work on a map that was never prepared — that is the whole
  // point of not gating this panel on the metadata.
  const activeKeyframeId =
    keyframeFilter !== "all"
      ? keyframeFilter
      : activeRow?.keyframe_id ?? pageKeyframeIds[0] ?? firstNavigableKeyframeId;
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
    if (!showKeyframeGraph || !status?.markers?.length) {
      return [];
    }
    // Keyframes with no proposals at all are dropped; those that have proposals but
    // were pruned at ingest are shown in a third colour rather than passed off as
    // indexed, which is the distinction the pgvector coverage exists to expose.
    return status.markers.flatMap((marker) => {
      const keyframe = keyframeSummaryById.get(marker.id);
      if (!keyframe) {
        return [];
      }
      const isSelected = marker.id === selectedKeyframeForMap;
      const state = keyframeIndexState(keyframe);
      return [{
        id: marker.id,
        latitude: marker.latitude,
        longitude: marker.longitude,
        level: marker.level,
        color: isSelected
          ? KEYFRAME_MARKER_SELECTED_COLOR
          : state === "pruned"
            ? KEYFRAME_MARKER_PRUNED_COLOR
            : KEYFRAME_MARKER_COLOR,
        radius: isSelected ? KEYFRAME_MARKER_SELECTED_RADIUS : KEYFRAME_MARKER_RADIUS,
        scaleWithZoom: true,
      }];
    });
  }, [
    showKeyframeGraph,
    status?.markers,
    keyframeSummaryById,
    selectedKeyframeForMap,
  ]);

  useEffect(() => {
    setDepthPinPopoverOpen(false);
  }, [depthPin?.requestId]);

  const showLivemapResizer =
    emmid !== null && (status?.markers?.length ?? 0) > 0;
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
    return status.markers.flatMap((marker) => {
      // Any keyframe with a pose is navigable: walking the panorama does not need
      // the map to have been prepared, let alone ingested.
      if (
        marker.id === activeKeyframeId ||
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
    return keyframeGraph.edges.flatMap((edge) => {
      if (
        preparedKeyframeIds.length &&
        edge.keyframe_id_1 &&
        edge.keyframe_id_2 &&
        (!keyframeSummaryById.has(edge.keyframe_id_1) || !keyframeSummaryById.has(edge.keyframe_id_2))
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
  }, [keyframeGraph, showKeyframeGraph, keyframeSummaryById, preparedKeyframeIds]);
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
      if (!status?.markers?.some((marker) => marker.id === keyframeId)) {
        return;
      }
      setKeyframeFilter(keyframeId);
      setSelectedRowIndex(null);
      setDepthPin(null);
      setDepthPinPopoverOpen(false);
    },
    [status?.markers],
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
            rowIndex: null,
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
            rowIndex: null,
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

  // One request for the whole page, where the cubemap explorer issued one per
  // keyframe per page. The backend holds the parsed parquet, so this is a slice.
  useEffect(() => {
    if (!props.isMapKnown || !status?.available || !pageKeyframeIds.length) {
      setMetadataRows([]);
      return;
    }
    let cancelled = false;
    fetchMetadataRows(props.mapId, { keyframeIds: pageKeyframeIds })
      .then((payload) => {
        if (!cancelled) {
          setMetadataRows(payload.rows);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setMetadataRows([]);
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown, status?.available, pageKeyframeIdsKey]);

  const activeKeyframeDetections = useMemo(
    () => (activeKeyframeId ? filteredRowsByKeyframe.get(activeKeyframeId) ?? [] : []),
    [activeKeyframeId, filteredRowsByKeyframe],
  );
  const activeKeyframeRawDetectionCount = activeKeyframeId
    ? keyframeSummaryById.get(activeKeyframeId)?.row_count ?? 0
    : 0;
  const rawDetections = useMemo(
    () => (activeKeyframeId ? metadataRows.filter((row) => row.keyframe_id === activeKeyframeId) : []),
    [activeKeyframeId, metadataRows],
  );
  const filteredDetections = activeKeyframeDetections;
  /** Square degrees now, not ERP pixels — see bboxPostProcess. */
  const bboxAreaSliderMax = useMemo(() => {
    if (!rawDetections.length) {
      return 400;
    }
    const maxArea = Math.max(...rawDetections.map((item) => bboxArea(item)));
    return Math.max(10, Math.ceil(maxArea / 10) * 10);
  }, [rawDetections]);
  /**
   * Keyframes that can be stepped through in the panorama: every keyframe with a
   * pose, whether or not it was prepared or ingested.
   */
  const navigableKeyframeIds = useMemo(
    () =>
      (status?.markers ?? [])
        .map((marker) => marker.id)
        .sort((a, b) => Number(a) - Number(b)),
    [status?.markers],
  );

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
    if (!activeKeyframeId) {
      return [];
    }
    const index = navigableKeyframeIds.indexOf(activeKeyframeId);
    if (index < 0) {
      return [];
    }
    return [navigableKeyframeIds[index - 1], navigableKeyframeIds[index + 1]]
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
  }, [activeKeyframeId, props.mapId, navigableKeyframeIds]);

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
    if (!shouldFocusSelectedDetectionRef.current || selectedRowIndex === null) {
      return;
    }
    shouldFocusSelectedDetectionRef.current = false;
    detectionRowRefs.current.get(selectedRowIndex)?.focus();
  }, [selectedRowIndex]);

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

  // Keep the selection on the page: post-processing can remove the selected row.
  useEffect(() => {
    if (!pageRows.length) {
      setSelectedRowIndex(null);
      return;
    }
    setSelectedRowIndex((current) =>
      current !== null && pageRows.some((row) => row.row_index === current)
        ? current
        : pageRows[0].row_index,
    );
  }, [pageRowIndexKey]);

  useEffect(() => {
    if (selectedRowIndex === null || !status?.available) {
      setRowDetail(null);
      return;
    }
    let cancelled = false;
    setIsLoadingRow(true);
    fetchMetadataRow(props.mapId, selectedRowIndex)
      .then((payload) => {
        if (!cancelled) {
          setRowDetail(payload);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setRowDetail(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingRow(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, selectedRowIndex, status?.available]);

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

  /**
   * The metadata warning is scoped to the row table below, not to the panel.
   *
   * Everything visual here — photosphere, depth preview, depth pin, view cone,
   * keyframe graph, livemap markers, annotations — runs off the v2 manifest. Gating
   * the whole panel on an index is what made a perfectly usable v2 map render as a
   * single "no object-search.db found" box.
   */
  const metadataNotice = !status?.available
    ? {
        variant: "info" as const,
        text:
          status?.error
          ?? "No object-search metadata (object-search/metadata.parquet) for this map.",
        checkedPaths: status?.checked_paths ?? [],
      }
    : !status.postprocessed
      ? {
          variant: "warning" as const,
          text:
            "This metadata.parquet has not been post-processed: no thumbnail_key and no "
            + "depth column, so proposal previews and 3D positions are unavailable. Run "
            + "`python -m toolbox.bricks.prepare_postprocess <map_path>`.",
          checkedPaths: [] as string[],
        }
      : null;

  const totalRows = summary?.row_count ?? 0;
  const selectedDetection =
    pageRows.find((row) => row.row_index === selectedRowIndex) ?? rowDetail?.row ?? null;
  const activeKeyframeIndex = activeKeyframeId
    ? navigableKeyframeIds.indexOf(activeKeyframeId)
    : -1;
  const hasPreviousKeyframe = activeKeyframeIndex > 0;
  const hasNextKeyframe =
    activeKeyframeIndex >= 0 && activeKeyframeIndex < navigableKeyframeIds.length - 1;
  const activeKeyframeSummary = activeKeyframeId
    ? keyframeSummaryById.get(activeKeyframeId)
    : undefined;

  const selectKeyframe = (keyframeId: string) => {
    if (!navigableKeyframeIds.includes(keyframeId) || keyframeId === activeKeyframeId) {
      return;
    }
    setKeyframeFilter(keyframeId);
    setSelectedRowIndex(null);
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
      navigableKeyframeIds.length - 1,
      Math.max(0, activeKeyframeIndex + direction),
    );
    const nextKeyframeId = navigableKeyframeIds[nextIndex];
    if (!nextKeyframeId || nextKeyframeId === activeKeyframeId) {
      return;
    }
    selectKeyframe(nextKeyframeId);
  };

  const selectAdjacentDetection = (direction: -1 | 1) => {
    if (!filteredDetections.length) {
      return;
    }
    const currentIndex =
      selectedRowIndex !== null
        ? filteredDetections.findIndex((item) => item.row_index === selectedRowIndex)
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
    if (nextDetection.row_index !== selectedRowIndex) {
      shouldFocusSelectedDetectionRef.current = true;
    }
    setSelectedRowIndex(nextDetection.row_index);
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
    rowIndex: number | null = null,
  ) => {
    const requestId = `${Date.now()}-${Math.random()}`;
    if (annotation.enabled) {
      annotation.beginDraft(requestId, {
        keyframeId,
        projection: source,
        rowIndex,
        xRatio,
        yRatio,
      });
    }
    setDepthPin({
      requestId,
      source,
      keyframeId,
      rowIndex,
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
      row_index: rowIndex ?? undefined,
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
          <h2>Proposals and reconstructed boxes</h2>
        </div>
        <p className="path-text">{summary?.metadata_path ?? status?.map_path ?? ""}</p>
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

      {metadataNotice ? (
        <div className={metadataNotice.variant === "info" ? "info-box" : "warning-box"}>
          <p>{metadataNotice.text}</p>
          {metadataNotice.checkedPaths.length ? (
            <ul className="checked-paths-list">
              {metadataNotice.checkedPaths.map((checkedPath) => (
                <li key={checkedPath}>
                  <code>{checkedPath}</code>
                </li>
              ))}
            </ul>
          ) : null}
          <p className="map-caption">
            The panorama, depth preview, depth pin, view cone and annotations below do not
            need it — they read the map manifest.
          </p>
        </div>
      ) : null}

      {/*
        "Keyframes with a pose" and "keyframes with proposals" are two different
        numbers, and a third — how many made it into pgvector — differs again.
        Reporting them separately is the whole point of this row.
      */}
      <div className="metrics-row">
        <span>Keyframes with a pose {status?.manifest_keyframe_count ?? 0}</span>
        <span>Keyframes with proposals {summary?.keyframes.length ?? 0}</span>
        <span>Proposals {totalRows}</span>
        <span>With depth {summary?.with_depth_count ?? 0}</span>
        {summary?.coverage ? (
          <span>
            Ingested {summary.coverage.ingested_total}
            {summary.coverage.no_position_total
              ? ` (${summary.coverage.no_position_total} without a 3D position)`
              : ""}
          </span>
        ) : summary ? (
          <span title="The bricks service did not answer; keyframes show as unknown.">
            Ingested unknown
          </span>
        ) : null}
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
                  {status?.markers?.length ?? 0} keyframes have a resolvable pose;{" "}
                  {summary?.keyframes.length ?? 0} of them carry proposals.
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
                  Proposals {activeKeyframeRawDetectionCount} | Kept{" "}
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
                      selectedRowIndex={selectedRowIndex}
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
                  ? "Proposal boxes reconstructed from the stored float16 angles — accurate to a few ERP pixels, not exact."
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
        title="Proposal explorer"
        summary={`${pageRows.length} shown | ${visibleKeyframes.length} keyframes`}
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
                  {/*
                    Every keyframe with a pose, not only the prepared ones: this select
                    also drives which panorama is shown, and a keyframe with no
                    proposals is still worth looking at.
                  */}
                  {navigableKeyframeIds.map((keyframeId) => {
                    const keyframe = keyframeSummaryById.get(keyframeId);
                    return (
                      <option key={keyframeId} value={keyframeId}>
                        {keyframeId} |{" "}
                        {keyframe ? `${keyframe.row_count} proposals` : "not prepared"}
                        {keyframe?.ingested === 0 ? " | pruned" : ""}
                      </option>
                    );
                  })}
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
                  <option value="objects-desc">Most proposals first</option>
                  <option value="objects-asc">Fewest proposals first</option>
                </select>
              </label>

            <label>
              Keyframes per page
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
                Include keyframes with no proposals
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
            <p className="muted">
              {summary
                ? "No proposals match the current filters."
                : "No proposals: this map has no object-search metadata."}
            </p>
          ) : (
            <div
              className="object-search-cutout-grid"
              style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
            >
              {pageRows.map((row) => {
                // The stored thumbnail is the default: it is the only image certain to
                // be what MetaCLIP2 embedded. A re-render is the fallback for maps
                // prepared without crops.
                const previewUrl = row.thumbnail_key
                  ? rowThumbnailUrl(props.mapId, row.thumbnail_key)
                  : rowRenderUrl(props.mapId, row.row_index, { size: 336 });
                return (
                  <button
                    className={`object-search-cutout-card${
                      row.row_index === selectedRowIndex ? " is-selected" : ""
                    }`}
                    type="button"
                    key={row.row_index}
                    onClick={() => setSelectedRowIndex(row.row_index)}
                  >
                    <span className="object-search-cutout-image-wrap">
                      <img
                        src={previewUrl}
                        alt={`Proposal ${row.row_index}${row.label ? ` (${row.label})` : ""}`}
                        title="Click to place a depth pin"
                        loading="lazy"
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedRowIndex(row.row_index);
                          const { xRatio, yRatio } = readClickRatio(event);
                          placeDepthPin(
                            "cutout",
                            row.keyframe_id,
                            xRatio,
                            yRatio,
                            row.row_index,
                          );
                        }}
                      />
                      {depthPin?.source === "cutout" && depthPin.rowIndex === row.row_index ? (
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
                      <strong>
                        {row.row_index} {row.label ? `| ${row.label}` : ""}
                      </strong>
                      <small>
                        keyframe {row.keyframe_id} | {row.detector_source || "?"}
                        {row.detection_score !== null
                          ? ` ${row.detection_score.toFixed(2)}`
                          : ""}
                        {" | "}
                        {row.depth !== null ? `${row.depth.toFixed(1)} m` : "no depth"}
                      </small>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <aside className="object-search-explorer-inspector">
          <h3>Selected proposal</h3>
          {selectedDetection ? (
            <>
              <div className="metrics-row">
                <span>Keyframe {selectedDetection.keyframe_id}</span>
                <span>Row {selectedDetection.row_index}</span>
                <span>
                  {filteredDetections.length} / {rawDetections.length} proposals kept
                </span>
                {activeKeyframeSummary ? (
                  <span
                    title={
                      activeKeyframeSummary.ingested === null
                        ? "The bricks service did not answer."
                        : `${activeKeyframeSummary.ingested} rows in pgvector`
                    }
                  >
                    {keyframeIndexState(activeKeyframeSummary)}
                  </span>
                ) : null}
              </div>

              {/*
                The stored thumbnail (left) is what MetaCLIP2 saw, masked to a square by
                build_padding_mask. The re-render (right) is the proposal's true angular
                extent at a widened FOV — the same four numbers, two different views.
              */}
              <div className="object-search-proposal-previews">
                {selectedDetection.thumbnail_key ? (
                  <figure>
                    <img
                      src={rowThumbnailUrl(props.mapId, selectedDetection.thumbnail_key)}
                      alt={`Stored thumbnail for proposal ${selectedDetection.row_index}`}
                      loading="lazy"
                    />
                    <figcaption>Stored thumbnail (embedded)</figcaption>
                  </figure>
                ) : null}
                <figure>
                  <img
                    src={rowRenderUrl(props.mapId, selectedDetection.row_index, {
                      size: 512,
                      fovScale: 2,
                    })}
                    alt={`Context view for proposal ${selectedDetection.row_index}`}
                    loading="lazy"
                    onClick={(event) => {
                      const { xRatio, yRatio } = readClickRatio(event);
                      placeDepthPin(
                        "cutout",
                        selectedDetection.keyframe_id,
                        xRatio,
                        yRatio,
                        selectedDetection.row_index,
                      );
                    }}
                  />
                  <figcaption>Context re-render, 2x FOV (reconstructed)</figcaption>
                </figure>
              </div>

              <BboxPostProcessControls
                params={bboxPostProcess}
                areaSliderMax={bboxAreaSliderMax}
                rawCount={rawDetections.length}
                filteredCount={filteredDetections.length}
                onChange={setBboxPostProcess}
                onReset={resetBboxPostProcess}
              />

              {isLoadingRow ? <p className="muted">Loading row metadata...</p> : null}

              {rawDetections.length && !filteredDetections.length ? (
                <p className="info-box">
                  All proposals were removed by the current post-processing filters.
                </p>
              ) : null}

              <CollapsibleSection title="Selected row JSON" defaultOpen={false}>
                <pre className="debug-json">
                  {JSON.stringify(selectedDetection, null, 2)}
                </pre>
              </CollapsibleSection>

              <CollapsibleSection
                title="Proposals in this keyframe"
                summary={`${filteredDetections.length} kept`}
                defaultOpen
              >
                <div className="table-wrap object-search-box-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Row</th>
                        <th>Label</th>
                        <th>Source</th>
                        <th>Score</th>
                        <th>Area (deg²)</th>
                        <th>Depth</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredDetections.map((item) => (
                        <tr
                          key={item.row_index}
                          ref={(node) => {
                            if (node) {
                              detectionRowRefs.current.set(item.row_index, node);
                            } else {
                              detectionRowRefs.current.delete(item.row_index);
                            }
                          }}
                          className={item.row_index === selectedRowIndex ? "is-selected" : ""}
                          tabIndex={0}
                          aria-selected={item.row_index === selectedRowIndex}
                          onClick={(event) => {
                            setSelectedRowIndex(item.row_index);
                            event.currentTarget.focus();
                          }}
                          onKeyDown={handleDetectionRowKeyDown}
                        >
                          <td>{item.row_index}</td>
                          <td>{item.label || "-"}</td>
                          <td>{item.detector_source || "-"}</td>
                          <td>
                            {item.detection_score === null
                              ? "-"
                              : item.detection_score.toFixed(3)}
                          </td>
                          <td>{bboxArea(item).toFixed(2)}</td>
                          <td>{item.depth === null ? "-" : `${item.depth.toFixed(2)} m`}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CollapsibleSection>

              <CollapsibleSection title="Preview debug" defaultOpen={false}>
                <pre className="debug-json">
                  {JSON.stringify(rowDetail?.preview_debug ?? {}, null, 2)}
                </pre>
              </CollapsibleSection>
            </>
          ) : (
            <p className="muted">Select a proposal to inspect it.</p>
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
