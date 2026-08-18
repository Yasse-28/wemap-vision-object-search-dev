import {
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type CSSProperties,
  type ReactElement,
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
import ExportDialog from "../export-roi/ExportDialog";
import ExportRoiPanel from "../export-roi/ExportRoiPanel";
import { keyframesInRois, roisForRequest } from "../export-roi/selection";
import { useExportSession } from "../export-roi/session";
import "../export-roi/export-roi.css";
import { bearingDegrees, projectKeyframeToLocalFloor } from "../geo";
import {
  fetchKeyframeGraph,
  fetchMetadataKeyframes,
  fetchMetadataMarkers,
  fetchMetadataRows,
  fetchMetadataStatus,
  fetchProposalNeighborProjections,
  indexKeyframeDepthPreviewUrl,
  indexKeyframeEquirectPreviewUrl,
  projectWorldPoint,
  proposalNeighborProjectionRenderUrl,
  resolveDepthPin,
  rowRenderUrl,
  rowThumbnailUrl,
} from "../index-explorer/api";
import type {
  DepthPinResponse,
  KeyframeGraphResponse,
  KeyframeMarker,
  KeyframeSummary,
  MetadataKeyframesResponse,
  MetadataRowRecord,
  MetadataStatusResponse,
  MetadataSummary,
  ProposalNeighborProjection,
  ProposalNeighborProjectionsResponse,
} from "../index-explorer/types";
import {
  KEYFRAME_GRAPH_COLOR,
  KEYFRAME_TRACK_COLOR,
  KEYFRAME_LINK_COLOR,
  KEYFRAME_MARKER_COLOR,
  KEYFRAME_MARKER_PRUNED_COLOR,
  KEYFRAME_MARKER_RADIUS,
  KEYFRAME_MARKER_SELECTED_COLOR,
  KEYFRAME_MARKER_SELECTED_RADIUS,
  KEYFRAME_MARKER_UNKNOWN_COLOR,
  EXPORT_ROI_COLOR,
  VIEW_CONE_COLOR,
  VIEW_CONE_HALF_ANGLE_DEG,
} from "../theme";
import BboxPostProcessControls from "./BboxPostProcessControls";
import {
  bboxArea,
  bboxPolygonRatios,
  DEFAULT_BBOX_POST_PROCESS,
  detectorSourceFamily,
  postProcessDetections,
} from "./bboxPostProcess";
import { useBboxPostProcessParams } from "./useBboxPostProcessParams";
import { buildKeyframeTrack } from "./keyframeTrack";
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
type ProposalDisplayState = "kept" | "discarded";
type DisplayedProposal = {
  row: MetadataRowRecord;
  state: ProposalDisplayState;
};
type ImageCursorRatio = {
  xRatio: number;
  yRatio: number;
};
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

// One ERP and all its proposals form one page. Mixing keyframes made the grid hard
// to scan and could put well over a thousand thumbnails in the same workspace.
const KEYFRAMES_PER_PAGE = 1;
const COLUMN_OPTIONS = [3, 4, 5, 6];
const EquirectPhotoSphereViewer = lazy(() => import("./EquirectPhotoSphereViewer"));
const DEPTH_PIN_MIN_DEPTH_M = 0.25;
const VISUAL_SPLIT_MIN_PERCENT = 25;
const VISUAL_SPLIT_MAX_PERCENT = 75;
const KEYFRAME_GRAPH_STORAGE_KEY = "object-search-gui.showKeyframeGraph";
const KEYFRAME_TRACK_STORAGE_KEY = "object-search-gui.showKeyframeTrack";
const PHOTOSPHERE_NAVIGATION_MAX_DISTANCE_M = 40;

function readStoredBoolean(key: string, fallback: boolean): boolean {
  try {
    const stored = localStorage.getItem(key);
    return stored === null ? fallback : stored !== "false";
  } catch {
    return fallback;
  }
}

function formatProposalShare(count: number, total: number): string {
  if (total <= 0) {
    return "0%";
  }
  const percentage = (count / total) * 100;
  return `${percentage.toFixed(1).replace(/\.0$/, "")}%`;
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

function confidenceBand(score: number): { label: string; tone: "high" | "medium" | "low" } {
  if (score >= 75) {
    return { label: "High", tone: "high" };
  }
  if (score >= 45) {
    return { label: "Medium", tone: "medium" };
  }
  return { label: "Low", tone: "low" };
}

function ProjectionConfidence(props: {
  projection: ProposalNeighborProjection;
}): ReactElement {
  const featureScore = typeof props.projection.feature_match_score === "number"
    ? props.projection.feature_match_score
    : null;
  const featureStatus = props.projection.feature_match_status ?? "unavailable";
  const edgeNccScore = typeof props.projection.edge_ncc_score === "number"
    ? props.projection.edge_ncc_score
    : null;
  const edgeNccStatus = props.projection.edge_ncc_status ?? "unavailable";
  const geometry = confidenceBand(props.projection.geometric_confidence);
  const visibility = props.projection.occlusion_confidence === null
    ? null
    : confidenceBand(props.projection.occlusion_confidence);
  const features = featureScore === null
    ? null
    : confidenceBand(featureScore);
  const appearance = edgeNccScore === null ? null : confidenceBand(edgeNccScore);
  const geometryTitle = [
    `Source distance ${props.projection.distance_score}%`,
    `view angle ${props.projection.view_angle_score}% (${props.projection.viewpoint_angle_deg.toFixed(1)}°)`,
    `apparent size ${props.projection.apparent_size_score}%`,
  ].join(" · ");
  const visibilityTitle = props.projection.observed_depth_m === null
    ? "No target depth sample is available."
    : `Expected ${props.projection.distance_to_proposal_m.toFixed(2)} m · observed ${props.projection.observed_depth_m.toFixed(2)} m`;
  const featureTitle = featureScore === null
    ? "SIFT matching is unavailable."
    : `${props.projection.feature_inliers} geometric inliers from ${props.projection.feature_matches} ratio-test matches`;
  const appearanceTitle = edgeNccStatus === "uninformative"
    ? "The proposal region does not contain enough edge structure for NCC."
    : edgeNccScore === null
      ? "Edge NCC comparison is unavailable."
      : "Gaussian multi-scale gradient correlation over the proposal region; low scores can still result from viewpoint changes.";

  return (
    <span className="object-search-projection-confidence">
      <span
        className={`object-search-confidence-score is-${geometry.tone}`}
        title={geometryTitle}
      >
        <span className="object-search-confidence-heading">
          <span>Geometry</span>
          <strong>{geometry.label} · {props.projection.geometric_confidence}%</strong>
        </span>
        <span className="object-search-confidence-meter" aria-hidden="true">
          <span style={{ width: `${props.projection.geometric_confidence}%` }} />
        </span>
      </span>
      <span
        className={`object-search-confidence-score is-${visibility?.tone ?? "unknown"}`}
        title={visibilityTitle}
      >
        <span className="object-search-confidence-heading">
          <span>Visibility</span>
          <strong>
            {visibility
              ? `${
                  props.projection.occlusion_status === "occluded"
                    ? "Blocked"
                    : props.projection.occlusion_status === "depth_mismatch"
                      ? "Mismatch"
                      : visibility.label
                } · ${props.projection.occlusion_confidence}%`
              : "No depth"}
          </strong>
        </span>
        <span className="object-search-confidence-meter" aria-hidden="true">
          {props.projection.occlusion_confidence !== null ? (
            <span style={{ width: `${props.projection.occlusion_confidence}%` }} />
          ) : null}
        </span>
      </span>
      <span
        className={`object-search-confidence-score is-${
          featureStatus === "no_features"
            ? "unknown"
            : features?.tone ?? "unknown"
        }`}
        title={featureTitle}
      >
        <span className="object-search-confidence-heading">
          <span>Features</span>
          <strong>
            {featureStatus === "unavailable"
              ? "Unavailable"
              : featureStatus === "no_features"
                ? "No features"
                : `${featureStatus === "weak" ? "Weak" : features?.label} · ${featureScore}%`}
          </strong>
        </span>
        <span className="object-search-confidence-meter" aria-hidden="true">
          {featureScore !== null ? (
            <span style={{ width: `${featureScore}%` }} />
          ) : null}
        </span>
      </span>
      <span
        className={`object-search-confidence-score is-${appearance?.tone ?? "unknown"}`}
        title={appearanceTitle}
      >
        <span className="object-search-confidence-heading">
          <span>Edge NCC</span>
          <strong>
            {edgeNccStatus === "uninformative"
              ? "Uninformative"
              : appearance
                ? `${appearance.label} · ${edgeNccScore}%`
                : "Unavailable"}
          </strong>
        </span>
        <span className="object-search-confidence-meter" aria-hidden="true">
          {edgeNccScore !== null ? (
            <span style={{ width: `${edgeNccScore}%` }} />
          ) : null}
        </span>
      </span>
    </span>
  );
}


function ObjectSearchExplorerPanel(props: Props) {
  const [status, setStatus] = useState<MetadataStatusResponse | null>(null);
  const [markers, setMarkers] = useState<KeyframeMarker[] | null>(null);
  const [keyframePage, setKeyframePage] =
    useState<MetadataKeyframesResponse | null>(null);
  const [keyframeGraph, setKeyframeGraph] = useState<KeyframeGraphResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const [keyframeFilter, setKeyframeFilter] = useState("all");
  const [includeEmpty, setIncludeEmpty] = useState(true);
  const [showPhotosphereBoxes, setShowPhotosphereBoxes] = useState(false);
  const [panoramaViewMode, setPanoramaViewMode] =
    useState<PanoramaViewMode>("image");
  // On by default: without it the livemap shows an empty floor plan on any map with
  // no 360-viewer graph, which is most of them.
  const [showKeyframeTrack, setShowKeyframeTrack] = useState(() =>
    readStoredBoolean(KEYFRAME_TRACK_STORAGE_KEY, true),
  );
  const [showKeyframeGraph, setShowKeyframeGraph] = useState(() =>
    readStoredBoolean(KEYFRAME_GRAPH_STORAGE_KEY, true),
  );
  const [sortMode, setSortMode] = useState<SortMode>("keyframe");
  const [columns, setColumns] = useState(5);
  const [showKeptProposals, setShowKeptProposals] = useState(true);
  const [showDiscardedProposals, setShowDiscardedProposals] = useState(false);
  const [page, setPage] = useState(1);
  const [visualSplitPercent, setVisualSplitPercent] = useState(45);
  const [isResizingVisualSplit, setIsResizingVisualSplit] = useState(false);

  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);
  const [neighborProjectionCount, setNeighborProjectionCount] = useState(5);
  const [ignoreNeighborMinDistance, setIgnoreNeighborMinDistance] = useState(false);
  const [neighborProjections, setNeighborProjections] =
    useState<ProposalNeighborProjectionsResponse | null>(null);
  const [neighborProjectionLoading, setNeighborProjectionLoading] = useState(false);
  const [neighborProjectionError, setNeighborProjectionError] = useState<string | null>(null);
  const [equirectPreviewError, setEquirectPreviewError] = useState<string | null>(null);
  const [equirectPreviewSrc, setEquirectPreviewSrc] = useState<string | null>(null);
  const [equirectPreviewKeyframeId, setEquirectPreviewKeyframeId] = useState<string | null>(null);
  const [photosphereYawRad, setPhotosphereYawRad] = useState<number | null>(null);
  const [photosphereViewKeyframeId, setPhotosphereViewKeyframeId] = useState<string | null>(null);
  const [depthPin, setDepthPin] = useState<DepthImagePin | null>(null);
  const [depthPinPopoverOpen, setDepthPinPopoverOpen] = useState(false);
  const [selectedPreviewCursor, setSelectedPreviewCursor] =
    useState<ImageCursorRatio | null>(null);
  const [isDepthModifierPressed, setIsDepthModifierPressed] = useState(false);
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
  const markerRequestRef = useRef<{
    mapId: string;
    promise: Promise<KeyframeMarker[]>;
  } | null>(null);
  const visualSplitRef = useRef<HTMLDivElement | null>(null);
  const inspectorRef = useRef<HTMLElement | null>(null);
  const neighborProjectionRequestRef = useRef(0);
  const { params: bboxPostProcess, setParams: setBboxPostProcess } =
    useBboxPostProcessParams();
  const [metadataRows, setMetadataRows] = useState<MetadataRowRecord[]>([]);
  const annotation = useExplorerAnnotationWorkspace(props.mapId);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsDepthModifierPressed(true);
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsDepthModifierPressed(false);
      }
    };
    const clearDepthModifier = () => setIsDepthModifierPressed(false);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", clearDepthModifier);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", clearDepthModifier);
    };
  }, []);

  useEffect(() => {
    setSelectedPreviewCursor(null);
    neighborProjectionRequestRef.current += 1;
    setNeighborProjections(null);
    setNeighborProjectionError(null);
    setNeighborProjectionLoading(false);
  }, [selectedRowIndex]);

  const summary: MetadataSummary | null = status?.summary ?? null;
  const keyframeSummaries = keyframePage?.keyframes ?? [];
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
    for (const item of keyframeSummaries) {
      byId.set(item.id, item);
    }
    return byId;
  }, [keyframeSummaries]);

  useEffect(() => {
    if (!props.isMapKnown) {
      setStatus(null);
      setMarkers(null);
      setKeyframePage(null);
      markerRequestRef.current = null;
      setIsLoading(false);
      return;
    }
    setDepthPin(null);
    setMarkers(null);
    setKeyframePage(null);
    markerRequestRef.current = null;
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
    fetchMetadataStatus(props.mapId)
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
    try {
      localStorage.setItem(KEYFRAME_TRACK_STORAGE_KEY, String(showKeyframeTrack));
    } catch {
      /* ignore */
    }
  }, [showKeyframeTrack]);

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

  const keptRowIndexes = useMemo(
    () =>
      new Set(
        [...filteredRowsByKeyframe.values()]
          .flat()
          .map((row) => row.row_index),
      ),
    [filteredRowsByKeyframe],
  );

  const emmid = props.map?.emmid ?? null;

  // The ROI tool draws one polygon; what it selects is the choice here. Counting
  // annotations was its only job until the export needed the same gesture over
  // keyframes, so the drawing stays single and the target is switched.
  const [roiTarget, setRoiTarget] = useState<"annotations" | "keyframes">("annotations");
  const exportSession = useExportSession(props.mapId);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const exportSelection = useMemo(
    () => keyframesInRois(markers, exportSession.rois),
    [markers, exportSession.rois],
  );
  const { height: livemapFrameHeight, isDragging: isResizingLivemap, startResize: startLivemapResize } =
    useLivemapFrameHeight();
  const { height: equirectFrameHeight, isDragging: isResizingEquirect, startResize: startEquirectResize } =
    useEquirectFrameHeight();

  const firstNavigableKeyframeId = status?.first_keyframe_id ?? null;

  const selectedKeyframeForMap = keyframeFilter !== "all" ? keyframeFilter : null;
  // Pagination is keyframe-major: one ERP and its proposals is how the panel is
  // actually used, and it keeps the row fetch to one request per page.
  const totalKeyframes = keyframePage?.total ?? 0;
  const totalPages = Math.max(
    1,
    Math.ceil(totalKeyframes / KEYFRAMES_PER_PAGE),
  );
  const currentPage = Math.min(page, totalPages);
  const pageKeyframes = keyframeSummaries;
  const pageKeyframeIds = pageKeyframes.map((item) => item.id);
  const pageKeyframeIdsKey = pageKeyframeIds.join("|");

  useEffect(() => {
    if (!props.isMapKnown || !status?.available) {
      setKeyframePage(null);
      return;
    }
    let cancelled = false;
    fetchMetadataKeyframes(props.mapId, {
      offset: (currentPage - 1) * KEYFRAMES_PER_PAGE,
      limit: KEYFRAMES_PER_PAGE,
      sort: sortMode === "keyframe" ? "parquet" : sortMode,
      includeEmpty,
      keyframeId: keyframeFilter === "all" ? null : keyframeFilter,
    })
      .then((payload) => {
        if (!cancelled) setKeyframePage(payload);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setKeyframePage(null);
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    props.mapId,
    props.isMapKnown,
    status?.available,
    currentPage,
    sortMode,
    includeEmpty,
    keyframeFilter,
  ]);
  const displayedProposals: DisplayedProposal[] = metadataRows.flatMap((row) => {
    const state: ProposalDisplayState = keptRowIndexes.has(row.row_index)
      ? "kept"
      : "discarded";
    const isVisible =
      (state === "kept" && showKeptProposals) ||
      (state === "discarded" && showDiscardedProposals);
    return isVisible ? [{ row, state }] : [];
  });
  const pageRows = displayedProposals.map((proposal) => proposal.row);
  const pageKeptCount = keptRowIndexes.size;
  const pageDiscardedCount = Math.max(0, metadataRows.length - pageKeptCount);
  const detectorBreakdown = useMemo(() => {
    let yoloCount = 0;
    let gdinoCount = 0;
    for (const [source, count] of Object.entries(
      summary?.detector_source_counts ?? {},
    )) {
      const family = detectorSourceFamily(source);
      if (family === "yolo") {
        yoloCount += count;
      } else if (family === "gdino") {
        gdinoCount += count;
      }
    }
    const total = summary?.row_count ?? 0;
    return {
      total,
      yoloCount,
      gdinoCount,
      yoloShare: formatProposalShare(yoloCount, total),
      gdinoShare: formatProposalShare(gdinoCount, total),
    };
  }, [summary]);
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
    markers?.find((marker) => marker.id === activeKeyframeId) ?? null;

  useEffect(() => {
    // The export's ROI count reads this table too: without it the panel would report
    // zero selected keyframes rather than "not loaded", which is the worse failure.
    const needsMarkerTable =
      showKeyframeGraph ||
      showKeyframeTrack ||
      roiTarget === "keyframes" ||
      activeKeyframeId !== null ||
      keyframeLink !== null;
    if (!props.isMapKnown || !needsMarkerTable || markers !== null) {
      return;
    }
    // A single active-keyframe lookup intentionally loads and caches the full table;
    // the same table also serves navigation cones and the optional full overlay.
    let cancelled = false;
    const request =
      markerRequestRef.current?.mapId === props.mapId
        ? markerRequestRef.current.promise
        : fetchMetadataMarkers(props.mapId);
    markerRequestRef.current = { mapId: props.mapId, promise: request };
    request
      .then((payload) => {
        if (!cancelled) setMarkers(payload);
      })
      .catch((err: Error) => {
        if (markerRequestRef.current?.promise === request) {
          markerRequestRef.current = null;
        }
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [
    props.mapId,
    props.isMapKnown,
    showKeyframeGraph,
    showKeyframeTrack,
    roiTarget,
    activeKeyframeId,
    keyframeLink,
    markers,
  ]);

  const livemapMarkers: LivemapMarker[] = useMemo(() => {
    const output: LivemapMarker[] = [];
    const activeMarker = activeKeyframeId
      ? markers?.find((marker) => marker.id === activeKeyframeId)
      : null;
    if (activeMarker) {
      output.push({
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
      output.push({
        id: "__depth_pin",
        latitude: depthPin.latitude,
        longitude: depthPin.longitude,
        level: depthPin.level,
        color: "#e11d48",
        radius: 9,
      });
    }
    output.push(...annotation.markers);
    return output;
  }, [
    activeKeyframeId,
    markers,
    depthPin,
    annotation.markers,
  ]);

  // The markers the hover-reveal picks from. Nothing renders them all: an overlay is
  // shown as lines, and the point under the cursor is the one that appears.
  const hoverRevealMarkers: LivemapMarker[] = useMemo(() => {
    if ((!showKeyframeGraph && !showKeyframeTrack) || !markers?.length) {
      return [];
    }
    // Only the current server page has summaries. Every other pose remains visible
    // in a distinct unknown colour instead of being passed off as indexed or pruned.
    return markers.map((marker) => {
      const keyframe = keyframeSummaryById.get(marker.id);
      const isSelected = marker.id === selectedKeyframeForMap;
      const state = keyframe ? keyframeIndexState(keyframe) : "unknown";
      return {
        id: marker.id,
        latitude: marker.latitude,
        longitude: marker.longitude,
        level: marker.level,
        color: isSelected
          ? KEYFRAME_MARKER_SELECTED_COLOR
          : state === "pruned"
            ? KEYFRAME_MARKER_PRUNED_COLOR
            : state === "unknown"
              ? KEYFRAME_MARKER_UNKNOWN_COLOR
              : KEYFRAME_MARKER_COLOR,
        radius: isSelected ? KEYFRAME_MARKER_SELECTED_RADIUS : KEYFRAME_MARKER_RADIUS,
        scaleWithZoom: true,
      };
    });
  }, [
    showKeyframeGraph,
    showKeyframeTrack,
    markers,
    keyframeSummaryById,
    selectedKeyframeForMap,
  ]);

  useEffect(() => {
    setDepthPinPopoverOpen(false);
  }, [depthPin?.requestId]);

  const showLivemapResizer =
    emmid !== null && (markers?.length ?? 0) > 0;
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
      !markers
    ) {
      return [];
    }
    return markers.flatMap((marker) => {
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
    markers,
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

  const livemapPolygons: LivemapPolygon[] = useMemo(() => {
    // Saved regions ride the existing polygon overlay: `LivemapPolygon` carries a
    // level and Mapbox already filters on it, so per-floor display costs nothing.
    const sessionPolygons: LivemapPolygon[] = exportSession.rois.map((roi) => ({
      id: `export-roi-${roi.id}`,
      coordinates: roi.ring.map((point) => [point.longitude, point.latitude]),
      color: EXPORT_ROI_COLOR,
      level: roi.level,
      opacity: 0.16,
      lineOpacity: 0.9,
      lineWidth: 2,
    }));
    return [...annotation.polygons, ...sessionPolygons];
  }, [annotation.polygons, exportSession.rois]);
  const keyframeTrackSegments: LivemapSegment[] = useMemo(
    () => (showKeyframeTrack ? buildKeyframeTrack(markers, KEYFRAME_TRACK_COLOR) : []),
    [showKeyframeTrack, markers],
  );

  const keyframeGraphSegments: LivemapSegment[] = useMemo(() => {
    if (!showKeyframeGraph || !keyframeGraph?.available) {
      return [];
    }
    return keyframeGraph.edges.flatMap((edge) => {
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
  }, [keyframeGraph, showKeyframeGraph]);
  const keyframeLinkSegments: LivemapSegment[] = useMemo(() => {
    if (!keyframeLink) {
      return [];
    }
    const ann = annotation.annotations.find(
      (item) =>
        item.id === keyframeLink.annotationId && item.annotationType === "point",
    );
    const keyframeMarker = markers?.find(
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
  }, [keyframeLink, annotation.annotations, markers]);
  const livemapSegments: LivemapSegment[] = useMemo(
    () => [...keyframeTrackSegments, ...keyframeGraphSegments, ...keyframeLinkSegments],
    [keyframeTrackSegments, keyframeGraphSegments, keyframeLinkSegments],
  );

  const selectGraphKeyframe = useCallback(
    (keyframeId: string) => {
      if (!markers?.some((marker) => marker.id === keyframeId)) {
        return;
      }
      setKeyframeFilter(keyframeId);
      setSelectedRowIndex(null);
      setDepthPin(null);
      setDepthPinPopoverOpen(false);
    },
    [markers],
  );

  const handlePhotosphereViewChange = useCallback(
    (keyframeId: string, yawRad: number, textureYRatio: number) => {
      if (keyframeId !== activeKeyframeId) {
        return;
      }
      setPhotosphereViewKeyframeId(keyframeId);
      setPhotosphereYawRad(yawRad);
      preservedTextureYRatioRef.current = textureYRatio;
      const headingDeg = markers?.find((marker) => marker.id === keyframeId)?.heading_deg;
      if (headingDeg !== null && headingDeg !== undefined) {
        preservedCompassBearingDegRef.current =
          ((headingDeg + yawRad * 180 / Math.PI) % 360 + 360) % 360;
      }
    },
    [activeKeyframeId, markers],
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
      (markers ?? [])
        .map((marker) => marker.id)
        .sort((a, b) => Number(a) - Number(b)),
    [markers],
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

  const livemapFocus = useMemo(() => {
    if (annotation.focusTarget) {
      return annotation.focusTarget;
    }
    if (!activeKeyframeId) {
      return null;
    }
    const marker = markers?.find((item) => item.id === activeKeyframeId);
    if (!marker) {
      return null;
    }
    return {
      latitude: marker.latitude,
      longitude: marker.longitude,
      level: marker.level,
    };
  }, [activeKeyframeId, markers, annotation.focusTarget]);

  useEffect(() => {
    setPage(1);
  }, [includeEmpty, keyframeFilter, sortMode]);

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
    pageRows.find((row) => row.row_index === selectedRowIndex) ?? null;
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
    const keyframeMarker = markers?.find(
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

  const selectProposalAndRevealPreview = (rowIndex: number): void => {
    setSelectedRowIndex(rowIndex);
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    inspectorRef.current?.scrollTo({
      top: 0,
      behavior: prefersReducedMotion ? "auto" : "smooth",
    });
  };

  const loadNeighborProjections = async (): Promise<void> => {
    if (!selectedDetection || selectedDetection.depth === null) {
      return;
    }
    const requestId = neighborProjectionRequestRef.current + 1;
    neighborProjectionRequestRef.current = requestId;
    setNeighborProjectionLoading(true);
    setNeighborProjectionError(null);
    try {
      const payload = await fetchProposalNeighborProjections(
        props.mapId,
        selectedDetection.row_index,
        neighborProjectionCount,
        !ignoreNeighborMinDistance,
      );
      if (neighborProjectionRequestRef.current === requestId) {
        setNeighborProjections(payload);
      }
    } catch (projectionError) {
      if (neighborProjectionRequestRef.current === requestId) {
        setNeighborProjections(null);
        setNeighborProjectionError(
          projectionError instanceof Error
            ? projectionError.message
            : String(projectionError),
        );
      }
    } finally {
      if (neighborProjectionRequestRef.current === requestId) {
        setNeighborProjectionLoading(false);
      }
    }
  };

  const openNeighborProjection = (projection: ProposalNeighborProjection): void => {
    setPhotosphereOrient((previous) => ({
      yawRad: projection.theta_center,
      token: (previous?.token ?? 0) + 1,
    }));
    selectKeyframe(projection.keyframe_id);
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    document.getElementById("explorer-spatial")?.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "start",
    });
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
          <h2>Inspect proposals in spatial context</h2>
          <p className="object-search-explorer-intro">
            Choose a keyframe, compare its map and panorama, then inspect or annotate
            the detections that matter.
          </p>
        </div>
        <details className="object-search-explorer-source">
          <summary>Data source</summary>
          <p className="path-text">{summary?.metadata_path ?? status?.map_path ?? ""}</p>
        </details>
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
      <dl className="explorer-health-strip" aria-label="Map processing coverage">
        <div title="Keyframes with a valid pose in the map manifest.">
          <dt>Posed</dt>
          <dd>{status?.manifest_keyframe_count ?? 0}</dd>
        </div>
        <div title="Keyframes matching the current Explorer filters. This is not an ML similarity score.">
          <dt>Matching</dt>
          <dd>{totalKeyframes}</dd>
        </div>
        <div
          className="explorer-health-proposals"
          title="Object-detection proposals stored in the map metadata."
        >
          <dt>Proposals</dt>
          <dd>{totalRows}</dd>
          <dd
            className="explorer-health-proposal-breakdown"
            aria-label="Detector share of proposals across this map"
          >
            <span
              title={`${detectorBreakdown.yoloCount} of ${detectorBreakdown.total} proposals across this map.`}
            >
              YOLO-W <strong>{detectorBreakdown.yoloShare}</strong>
            </span>
            <span
              title={`${detectorBreakdown.gdinoCount} of ${detectorBreakdown.total} proposals across this map.`}
            >
              G-DINO <strong>{detectorBreakdown.gdinoShare}</strong>
            </span>
          </dd>
        </div>
        <div title="Proposals with a resolved depth value available for 3D positioning.">
          <dt>With depth</dt>
          <dd>{summary?.with_depth_count ?? 0}</dd>
        </div>
        {summary?.coverage ? (
          <div
            title={
              summary.coverage.no_position_total
                ? `Proposals stored in the search index. ${summary.coverage.no_position_total} have no 3D position.`
                : "Proposals stored in the search index."
            }
          >
            <dt>Ingested</dt>
            <dd>{summary.coverage.ingested_total}</dd>
          </div>
        ) : summary ? (
          <div title="The bricks service did not answer; keyframes show as unknown.">
            <dt>Ingested</dt>
            <dd>Unknown</dd>
          </div>
        ) : null}
      </dl>

      <div className="explorer-context-rail" aria-label="Explorer workflow">
        <div className="explorer-context-keyframe">
          <span className="explorer-context-label">Current keyframe</span>
          <select
            aria-label="Current keyframe"
            value={keyframeFilter}
            onChange={(event) => setKeyframeFilter(event.target.value)}
          >
            <option value="all">Automatic (page selection)</option>
            {navigableKeyframeIds.map((keyframeId) => {
              const keyframe = keyframeSummaryById.get(keyframeId);
              return (
                <option key={keyframeId} value={keyframeId}>
                  {keyframeId}
                  {keyframe ? ` · ${keyframe.row_count} proposals` : ""}
                  {keyframe?.ingested === 0 ? " · pruned" : ""}
                </option>
              );
            })}
          </select>
          {renderKeyframeStepper("keyframe-stepper keyframe-stepper--compact")}
        </div>
        <nav
          className="explorer-workflow-links"
          aria-label="Jump to Explorer section"
        >
          <a href="#explorer-spatial">1 · Locate</a>
          <a href="#explorer-proposals">2 · Inspect</a>
          <a href="#explorer-annotation">3 · Annotate</a>
        </nav>
      </div>

      {roiTarget === "keyframes" ? (
        <ExportRoiPanel
          session={exportSession}
          selection={exportSelection}
          currentLevel={annotation.currentLevel}
          onExport={() => {
            exportSession.stop();
            setExportDialogOpen(true);
          }}
        />
      ) : null}

      {exportDialogOpen ? (
        <ExportDialog
          mapId={props.mapId}
          rois={roisForRequest(exportSession.rois)}
          liveTotal={exportSelection.total}
          onClose={() => setExportDialogOpen(false)}
          onExported={() => exportSession.clear()}
        />
      ) : null}

      <div
        id="explorer-spatial"
        ref={visualSplitRef}
        className="object-search-visual-split"
        style={{ "--visual-map-width": `${visualSplitPercent}%` } as CSSProperties}
      >
        <div className="object-search-visual-pane">
          <section className="object-search-explorer-livemap">
            <div className="object-search-explorer-livemap-header">
              <div>
                <h3>Spatial view</h3>
                <span className="object-search-section-kicker">
                  Locate the active capture
                </span>
              </div>
              <details className="explorer-map-options">
                <summary>Map tools</summary>
                <div className="object-search-explorer-livemap-toggles">
                  {keyframeGraph?.available ? (
                    <label className="inline-check">
                      <input
                        type="checkbox"
                        checked={showKeyframeGraph}
                        onChange={(event) =>
                          setShowKeyframeGraph(event.target.checked)
                        }
                      />
                      Show graph ({keyframeGraph.edges.length} edges)
                    </label>
                  ) : null}
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={showKeyframeTrack}
                      onChange={(event) =>
                        setShowKeyframeTrack(event.target.checked)
                      }
                    />
                    Show track ({keyframeTrackSegments.length} segments)
                  </label>
                  <div className="object-search-livemap-roi-toolbar">
                    <span>ROI</span>
                    <select
                      value={roiTarget}
                      aria-label="What the drawn region selects"
                      onChange={(event) => {
                        annotation.clearRoi();
                        setRoiTarget(
                          event.target.value as "annotations" | "keyframes",
                        );
                      }}
                    >
                      <option value="annotations">Count annotations</option>
                      <option value="keyframes">Select keyframes</option>
                    </select>
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
                      {roiTarget === "keyframes"
                        ? `${exportSelection.total} keyframes`
                        : annotation.roiCounts
                          ? `Total ${annotation.roiCounts.total}`
                          : `Level ${
                              annotation.currentLevel === null
                                ? "-"
                                : annotation.currentLevel
                            }`}
                    </span>
                  </div>
                </div>
              </details>
            </div>
            {keyframeGraph?.error ? (
              <p className="map-caption">Keyframe graph unavailable: {keyframeGraph.error}</p>
            ) : null}
            {emmid === null ? (
              <p className="info-box">
                Map preview unavailable: no <code>emmid</code> configured for this map.
              </p>
            ) : status === null || status.marker_count === 0 ? (
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
                    onRoiChange={(polygon) => {
                      annotation.setRoiPolygon(polygon);
                      // The level is frozen at close time, not read at export time:
                      // the whole point of a session is drawing on one floor, then
                      // another.
                      if (roiTarget === "keyframes" && polygon && exportSession.active) {
                        exportSession.addRoi(polygon, annotation.currentLevel);
                      }
                    }}
                    onLevelChange={annotation.setCurrentLevel}
                    onMarkerContextMenu={handleMarkerContextMenu}
                    segmentHoverMarkers={hoverRevealMarkers}
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
                  {status?.marker_count ?? 0} keyframes have a resolvable pose;{" "}
                  {totalKeyframes} match the proposal filters.
                  {showKeyframeGraph || showKeyframeTrack
                    ? " Hover an overlay to reveal the closest keyframe; click it or the revealed point to filter cutouts."
                    : " Enable the track — or the graph, where one exists — to reveal keyframes on hover."}
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
                <div className="object-search-keyframe-heading">
                  <h3>Panorama</h3>
                  <span className="object-search-section-kicker">
                    Keyframe {activeKeyframeId} · {activeKeyframeDetections.length} of{" "}
                    {activeKeyframeRawDetectionCount} proposals kept
                  </span>
                </div>
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

      <div id="explorer-proposals" className="explorer-proposals-anchor">
        <CollapsibleSection
          title="Proposal explorer"
          summary={`${displayedProposals.length} shown · 1 keyframe`}
          sectionClassName="object-search-explorer-browser"
          defaultOpen
        >
          <div className="object-search-proposal-toolbar">
            <fieldset className="object-search-proposal-visibility">
              <legend>Show proposals</legend>
              <label>
                <input
                  type="checkbox"
                  checked={showKeptProposals}
                  onChange={(event) => setShowKeptProposals(event.target.checked)}
                />
                Kept <strong>{pageKeptCount}</strong>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showDiscardedProposals}
                  onChange={(event) =>
                    setShowDiscardedProposals(event.target.checked)
                  }
                />
                Discarded <strong>{pageDiscardedCount}</strong>
              </label>
            </fieldset>

            <fieldset className="object-search-proposal-visibility">
              <legend>Detectors</legend>
              <label>
                <input
                  type="checkbox"
                  checked={bboxPostProcess.showYolo}
                  onChange={(event) =>
                    setBboxPostProcess({
                      ...bboxPostProcess,
                      showYolo: event.target.checked,
                    })
                  }
                />
                YOLO-W
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={bboxPostProcess.showGdino}
                  onChange={(event) =>
                    setBboxPostProcess({
                      ...bboxPostProcess,
                      showGdino: event.target.checked,
                    })
                  }
                />
                G-DINO
              </label>
            </fieldset>

            <BboxPostProcessControls
              params={bboxPostProcess}
              areaSliderMax={bboxAreaSliderMax}
              rawCount={metadataRows.length}
              filteredCount={pageKeptCount}
              onChange={setBboxPostProcess}
              onReset={() => {
                const { showYolo, showGdino } = bboxPostProcess;
                setBboxPostProcess({
                  ...DEFAULT_BBOX_POST_PROCESS,
                  showYolo,
                  showGdino,
                });
              }}
            />
          </div>

          <div className="object-search-explorer-layout">
            <div className="object-search-explorer-main">
              <details className="explorer-display-options">
                <summary>
                  Display and paging
                  <span>
                    1 keyframe · {columns} columns
                  </span>
                </summary>
                <div className="object-search-explorer-controls">
                  <label>
                    Sort
                    <select
                      value={sortMode}
                      onChange={(event) =>
                        setSortMode(event.target.value as SortMode)
                      }
                    >
                      <option value="keyframe">Keyframe order</option>
                      <option value="objects-desc">Most proposals first</option>
                      <option value="objects-asc">Fewest proposals first</option>
                    </select>
                  </label>

                  <label>
                    Columns
                    <select
                      value={String(columns)}
                      onChange={(event) =>
                        setColumns(Number(event.target.value))
                      }
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
                        onChange={(event) =>
                          setIncludeEmpty(event.target.checked)
                        }
                      />
                      Include keyframes with no proposals
                    </label>
                  </div>
                </div>
              </details>

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

          {!showKeptProposals && !showDiscardedProposals ? (
            <p className="muted">Select Kept or Discarded to show proposals.</p>
          ) : !pageRows.length ? (
            <p className="muted">
              {summary
                ? "No proposals match the current filters."
                : "No proposals: this map has no object-search metadata."}
            </p>
          ) : (
            <section className="object-search-keyframe-proposals">
              <header>
                <div>
                  <p className="eyebrow">Current page</p>
                  <h4>Keyframe {pageKeyframeIds[0] ?? "-"}</h4>
                </div>
                <span>
                  {pageKeptCount} kept · {pageDiscardedCount} discarded
                </span>
              </header>
              <div
                className="object-search-cutout-grid"
                style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
              >
              {displayedProposals.map(({ row, state }) => {
                // The stored thumbnail is the default: it is the only image certain to
                // be what MetaCLIP2 embedded. A re-render is the fallback for maps
                // prepared without crops.
                const previewUrl = row.thumbnail_key
                  ? rowThumbnailUrl(props.mapId, row.thumbnail_key)
                  : rowRenderUrl(props.mapId, row.row_index, { size: 336 });
                const label = row.label || "Unlabelled proposal";
                const technicalSummary = [
                  `Row ${row.row_index}`,
                  row.detector_source || "unknown source",
                  row.detection_score === null
                    ? "unknown score"
                    : `score ${row.detection_score.toFixed(3)}`,
                  row.depth === null ? "no depth" : `${row.depth.toFixed(2)} m depth`,
                ].join(" · ");
                return (
                  <button
                    className={`object-search-cutout-card${
                      row.row_index === selectedRowIndex ? " is-selected" : ""
                    }${state === "discarded" ? " is-discarded" : ""}`}
                    type="button"
                    key={row.row_index}
                    aria-label={`${label}, ${state}. ${technicalSummary}`}
                    title={technicalSummary}
                    onClick={() => selectProposalAndRevealPreview(row.row_index)}
                  >
                    <span className="object-search-cutout-image-wrap">
                      <span className="object-search-proposal-row-badge">
                        #{row.row_index}
                      </span>
                      {state === "discarded" ? (
                        <span className="object-search-proposal-state-badge">
                          Discarded
                        </span>
                      ) : null}
                      <img
                        src={previewUrl}
                        alt={`Proposal ${row.row_index} (${label})`}
                        loading="lazy"
                      />
                    </span>
                    <span className="object-search-cutout-card-label" title={label}>
                      {label}
                    </span>
                  </button>
                );
              })}
              </div>
            </section>
          )}
        </div>

        <aside ref={inspectorRef} className="object-search-explorer-inspector">
          <h3>Selected proposal</h3>
          {selectedDetection ? (
            <>
              <div className="object-search-selected-heading">
                <strong>{selectedDetection.label || "Unlabelled proposal"}</strong>
                <span>Row {selectedDetection.row_index}</span>
              </div>
              <div className="object-search-selected-facts">
                <span>Keyframe {selectedDetection.keyframe_id}</span>
                <span>{selectedDetection.detector_source || "Unknown source"}</span>
                <span>
                  Score {selectedDetection.detection_score?.toFixed(3) ?? "unknown"}
                </span>
                <span>
                  {selectedDetection.depth === null
                    ? "No depth"
                    : `${selectedDetection.depth.toFixed(2)} m depth`}
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
                The grid already shows what MetaCLIP2 saw. The inspector leads with the
                useful delta: the proposal's true angular extent at a widened FOV.
              */}
              <figure className="object-search-context-preview">
                <span className="object-search-depth-pin-stage">
                  <img
                    src={rowRenderUrl(props.mapId, selectedDetection.row_index, {
                      size: 512,
                      fovScale: 2,
                    })}
                    alt={`Context view for proposal ${selectedDetection.row_index}`}
                    title="Ctrl+click to place a depth pin"
                    loading="lazy"
                    onMouseMove={(event) => {
                      setSelectedPreviewCursor(readClickRatio(event));
                      setIsDepthModifierPressed(event.ctrlKey);
                    }}
                    onMouseLeave={() => setSelectedPreviewCursor(null)}
                    onClick={(event) => {
                      if (!event.ctrlKey) {
                        return;
                      }
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
                  {isDepthModifierPressed && selectedPreviewCursor ? (
                    <span
                      className="object-search-depth-projection-cursor"
                      aria-hidden="true"
                      style={{
                        left: `${selectedPreviewCursor.xRatio * 100}%`,
                        top: `${selectedPreviewCursor.yRatio * 100}%`,
                      }}
                    />
                  ) : null}
                  {depthPin?.source === "cutout" &&
                  depthPin.rowIndex === selectedDetection.row_index ? (
                    <span
                      className={`object-search-depth-pin is-${depthPin.status}`}
                      style={{
                        left: `${depthPin.xRatio * 100}%`,
                        top: `${depthPin.yRatio * 100}%`,
                      }}
                    />
                  ) : null}
                </span>
                <figcaption>Context re-render · 2× FOV · Ctrl+click to place a depth pin</figcaption>
              </figure>

              <CollapsibleSection
                title="Project into nearby keyframes"
                summary={
                  selectedDetection.depth === null
                    ? "Depth required"
                    : neighborProjections
                      ? `${neighborProjections.projections.length} views`
                      : `${neighborProjectionCount} ${
                          ignoreNeighborMinDistance ? "nearest" : "diverse"
                        } views`
                }
                defaultOpen={false}
              >
                <div className="object-search-neighbor-projection-controls">
                  <label>
                    Neighbours
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={neighborProjectionCount}
                      onChange={(event) =>
                        setNeighborProjectionCount(
                          Math.max(1, Math.min(12, Number(event.target.value) || 1)),
                        )
                      }
                    />
                  </label>
                  <label
                    className="object-search-neighbor-distance-toggle"
                    title="Return the closest keyframes even when their camera positions are less than 0.5 m apart."
                  >
                    <input
                      type="checkbox"
                      checked={ignoreNeighborMinDistance}
                      onChange={(event) => {
                        setIgnoreNeighborMinDistance(event.target.checked);
                        setNeighborProjections(null);
                      }}
                    />
                    <span>
                      Closest only
                      <small>No 0.5 m spacing</small>
                    </span>
                  </label>
                  <button
                    type="button"
                    className="primary-button"
                    disabled={
                      selectedDetection.depth === null || neighborProjectionLoading
                    }
                    onClick={() => void loadNeighborProjections()}
                  >
                    {neighborProjectionLoading ? "Projecting..." : "Project"}
                  </button>
                </div>

                {selectedDetection.depth === null ? (
                  <p className="muted">
                    This proposal has no depth, so it cannot be lifted into 3D.
                  </p>
                ) : null}
                {neighborProjectionError ? (
                  <p className="warning-box" role="alert">
                    {neighborProjectionError}
                  </p>
                ) : null}
                {neighborProjections ? (
                  <>
                    <p className="map-caption">{neighborProjections.note}</p>
                    {neighborProjections.projections.length ? (
                      <div className="object-search-neighbor-projection-grid">
                        {neighborProjections.projections.map((projection) => (
                          <button
                            key={projection.keyframe_id}
                            type="button"
                            className="object-search-neighbor-projection-card"
                            aria-label={`Open keyframe ${projection.keyframe_id} in the panorama`}
                            onClick={() => openNeighborProjection(projection)}
                          >
                            <span className="object-search-neighbor-projection-stage">
                              <img
                                src={proposalNeighborProjectionRenderUrl(
                                  props.mapId,
                                  selectedDetection.row_index,
                                  projection.keyframe_id,
                                  { size: 384, fovScale: 2 },
                                )}
                                alt={`Proposal ${selectedDetection.row_index} projected into keyframe ${projection.keyframe_id}`}
                                loading="lazy"
                              />
                              <span
                                className="object-search-neighbor-projection-box"
                                aria-hidden="true"
                              />
                              <span
                                className="object-search-depth-projection-cursor"
                                aria-hidden="true"
                                style={{ left: "50%", top: "50%" }}
                              />
                            </span>
                            <span className="object-search-neighbor-projection-caption">
                              <strong>Keyframe {projection.keyframe_id}</strong>
                              <span>
                                {projection.distance_from_source_m.toFixed(1)} m from
                                source · {projection.distance_to_proposal_m.toFixed(1)} m
                                to proposal
                              </span>
                              <ProjectionConfidence projection={projection} />
                              <span className="object-search-neighbor-projection-action">
                                Open in panorama →
                              </span>
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No neighbouring keyframe is available.</p>
                    )}
                  </>
                ) : null}
              </CollapsibleSection>

              {selectedDetection.thumbnail_key ? (
                <CollapsibleSection
                  title="Compare embedded crop"
                  summary="What MetaCLIP2 saw"
                  defaultOpen={false}
                >
                  <figure className="object-search-embedded-preview">
                    <img
                      src={rowThumbnailUrl(props.mapId, selectedDetection.thumbnail_key)}
                      alt={`Embedded crop for proposal ${selectedDetection.row_index}`}
                      loading="lazy"
                    />
                    <figcaption>Stored thumbnail used for embedding</figcaption>
                  </figure>
                </CollapsibleSection>
              ) : null}

              {rawDetections.length && !filteredDetections.length ? (
                <p className="info-box">
                  All proposals were removed by the current post-processing filters.
                </p>
              ) : null}

            </>
          ) : (
            <p className="muted">Select a proposal to inspect it.</p>
          )}
        </aside>
        </div>
        </CollapsibleSection>
      </div>
      <div id="explorer-annotation" className="explorer-annotation-anchor">
        <ExplorerAnnotationControls workspace={annotation} />
      </div>
      <ExplorerAnnotationList workspace={annotation} />
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
