import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { MapSummary } from "../api";
import DismissibleAlert from "../DismissibleAlert";
import {
  ExplorerAnnotationControls,
  ExplorerAnnotationList,
  annotationClassKey,
  canSaveAnnotation,
  useExplorerAnnotationWorkspace,
} from "../annotations/ExplorerAnnotationWorkspace";
import LivemapAnnotation, { type LivemapCone } from "../annotations/LivemapAnnotation";
import CollapsibleSection from "../annotations/CollapsibleSection";
import type { AnnotationFeature, LivemapMarker, LivemapSegment } from "../annotations/types";
import {
  fetchMetadataMarkers,
  fetchMetadataRows,
  fetchProposalNeighborProjections,
  indexKeyframeEquirectPreviewUrl,
  proposalNeighborProjectionRenderUrl,
  resolveDepthPin,
  rowThumbnailUrl,
} from "../index-explorer/api";
import type {
  KeyframeMarker,
  MetadataRowRecord,
  ProposalNeighborProjection,
  ProposalNeighborProjectionsResponse,
} from "../index-explorer/types";
import { bboxPolygonRatios } from "../object-search-explorer/bboxPostProcess";
import type { EditableBox } from "../object-search-explorer/EquirectPhotoSphereViewer";
import { buildKeyframeTrack } from "../object-search-explorer/keyframeTrack";
import {
  EQUIRECT_FRAME_MAX,
  EQUIRECT_FRAME_MIN,
  useEquirectFrameHeight,
} from "../object-search-explorer/useEquirectFrameHeight";
import {
  KEYFRAME_MARKER_COLOR,
  KEYFRAME_MARKER_RADIUS,
  KEYFRAME_MARKER_SELECTED_COLOR,
  KEYFRAME_MARKER_SELECTED_RADIUS,
  KEYFRAME_TRACK_COLOR,
  VIEW_CONE_COLOR,
} from "../theme";
import AngularRibbon from "./AngularRibbon";
import { classLabelDefaults } from "./classLabels";
import {
  estimatedExtentM,
  extentFromAngularWidth,
  ratioDistance,
  regionCentroid,
  type RegionVertex,
} from "./regionGeometry";
import {
  EMPTY_HANDOFF,
  readHandoff,
  withKeyframe,
  withSelection,
  withSkipped,
  withValidated,
  writeHandoff,
  type AnnotationHandoff,
} from "./handoff";

const EquirectPhotoSphereViewer = lazy(
  () => import("../object-search-explorer/EquirectPhotoSphereViewer"),
);

/** Tall enough to judge an object, short enough to leave the form on screen. */
const VIEWER_DEFAULT_HEIGHT = 320;
/** Farther than this between pointer down and up is a look-around, not a click. */
const CLICK_SLOP_PX = 4;
/**
 * How close to the first vertex a click has to land to close the outline, in texture
 * ratios. Matches the livemap-tools annotator, whose gesture this mirrors.
 */
const POLYGON_CLOSE_DISTANCE_RATIO = 0.0025;

/** Half the cone the capsule's livemap draws, matching the Explorer's view cone. */
const CONE_FOV_DEG = 60;
/** Rows per keyframe are in the tens; one request covers the turn. */
const ROWS_PER_KEYFRAME = 500;
const DEPTH_PIN_MIN_DEPTH_M = 0.35;
/** How many nearby keyframes to offer once an object is annotated. */
const NEIGHBOR_SUGGESTION_COUNT = 6;

/** An object already described on one keyframe, offered on another via reprojection. */
type SuggestionSource = { annotation: AnnotationFeature; rowIndex: number };

/**
 * A suggestion the annotator is positioning on `targetKeyframeId`. `box` is what the
 * viewer renders and lets the annotator drag; `orientYawRad` only fires once, when
 * the panorama first lands on that keyframe, so it does not fight a look-around.
 */
type ActiveSuggestion = {
  source: SuggestionSource;
  originKeyframeId: string;
  targetKeyframeId: string;
  orientYawRad: number;
  box: EditableBox;
};

function boxCenterRatio(box: EditableBox): { xRatio: number; yRatio: number } {
  return { xRatio: (box.u0 + box.u1) / 2, yRatio: (box.v0 + box.v1) / 2 };
}

/** A reprojected point + angular size, converted to the box the viewer overlays. */
function editableBoxFromProjection(projection: ProposalNeighborProjection): EditableBox {
  const halfWidthRatio = projection.angular_width / (2 * Math.PI) / 2;
  const halfHeightRatio = projection.angular_height / Math.PI / 2;
  return {
    u0: projection.erp_u - halfWidthRatio,
    u1: projection.erp_u + halfWidthRatio,
    v0: projection.erp_v - halfHeightRatio,
    v1: projection.erp_v + halfHeightRatio,
  };
}

type Mode = "object" | "zones" | "review";

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "object", label: "Object", hint: "Describe the object under the cursor" },
  { id: "zones", label: "Zones", hint: "Exhaustive coverage, at map scale" },
  { id: "review", label: "Review", hint: "What is already saved" },
];

function degrees(radians: number): number {
  return Math.round((radians * 180) / Math.PI);
}

/**
 * The Annotation tab: describe and validate.
 *
 * The Explorer answers "which detection is this?"; this panel answers "what is the
 * object, and is the record right?". They no longer compete for the screen, but the
 * visual context travels: the capsule at the top carries the proposal, the whole
 * 360° turn it sits on, and the livemap, so an annotator can confirm the object
 * without going back — and "Back to the proposal" is one click when they must.
 */
function AnnotationPanel(props: {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
  onBackToProposal: () => void;
}) {
  const workspace = useExplorerAnnotationWorkspace(props.mapId);
  const [handoff, setHandoff] = useState<AnnotationHandoff>(EMPTY_HANDOFF);
  const [rows, setRows] = useState<MetadataRowRecord[]>([]);
  const [rowsError, setRowsError] = useState<string | null>(null);
  const [isLoadingRows, setIsLoadingRows] = useState(false);
  const [markers, setMarkers] = useState<KeyframeMarker[]>([]);
  const [mode, setMode] = useState<Mode>("object");
  const [isMapOpen, setIsMapOpen] = useState(true);
  /** Where the panorama is pointed, and the last point clicked in it. */
  const [orient, setOrient] = useState<{ yawRad: number; token: number } | null>(null);
  const [erpPin, setErpPin] = useState<{
    xRatio: number;
    yRatio: number;
    status: "resolving" | "resolved" | "error";
  } | null>(null);
  const [hoveredRowIndex, setHoveredRowIndex] = useState<number | null>(null);
  /** The outline being traced over an object no detector proposed. */
  const [regionVertices, setRegionVertices] = useState<RegionVertex[]>([]);
  const [isDrawingRegion, setIsDrawingRegion] = useState(false);
  // The object just annotated, and where reprojection says it should also appear.
  const [suggestionSource, setSuggestionSource] = useState<SuggestionSource | null>(null);
  const [neighborSuggestions, setNeighborSuggestions] =
    useState<ProposalNeighborProjectionsResponse | null>(null);
  const [neighborSuggestionsError, setNeighborSuggestionsError] = useState<string | null>(null);
  const [neighborSuggestionsLoading, setNeighborSuggestionsLoading] = useState(false);
  // The suggestion the annotator is currently positioning, if any.
  const [activeSuggestion, setActiveSuggestion] = useState<ActiveSuggestion | null>(null);
  // In suggestion review, the viewer fills the overlay instead of the resized
  // frame height, so its pixel height is tracked separately from a live measurement.
  const suggestionViewerRef = useRef<HTMLDivElement | null>(null);
  const [suggestionViewerHeight, setSuggestionViewerHeight] = useState<number | null>(null);
  const {
    height: viewerHeight,
    isDragging: isResizingViewer,
    startResize: startViewerResize,
  } = useEquirectFrameHeight({
    storageKey: "object-search-gui.annotationViewerHeight",
    defaultHeight: VIEWER_DEFAULT_HEIGHT,
  });
  // The pointer gesture that tells a box click apart from a look-around.
  const hoveredRowRef = useRef<number | null>(null);
  const pointerDownRef = useRef<{ x: number; y: number } | null>(null);

  // The draft is started for a row exactly once, so re-renders cannot wipe a form
  // the annotator is in the middle of filling.
  const startedForRowRef = useRef<number | null>(null);
  const advanceOnSaveRef = useRef(false);
  const savedCountRef = useRef(0);
  const neighborSuggestionRequestRef = useRef(0);
  // Which suggestion's geometry has already been resolved, so re-renders that leave
  // it unchanged do not re-issue the depth-pin request.
  const startedSuggestionRef = useRef<ActiveSuggestion | null>(null);
  // Set right before a save, so the length-increase effect below can tell an actual
  // save apart from the store's initial bulk load of existing annotations — only the
  // former should offer reprojection.
  const suggestAfterSaveRef = useRef(false);

  useEffect(() => {
    setHandoff(readHandoff(props.mapId));
    startedForRowRef.current = null;
    setSuggestionSource(null);
    setNeighborSuggestions(null);
    setNeighborSuggestionsError(null);
    setActiveSuggestion(null);
  }, [props.mapId]);

  const updateHandoff = useCallback(
    (next: AnnotationHandoff) => {
      setHandoff(next);
      writeHandoff(props.mapId, next);
    },
    [props.mapId],
  );

  // Annotation is what this tab is for; there is no mode to switch on.
  useEffect(() => {
    workspace.setEnabled(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.mapId]);

  useEffect(() => {
    let isMounted = true;
    fetchMetadataMarkers(props.mapId)
      .then((items) => {
        if (isMounted) {
          setMarkers(items);
        }
      })
      .catch(() => {
        // The capsule's livemap is context, not the record: a missing marker list
        // costs the cone, not the annotation.
      });
    return () => {
      isMounted = false;
    };
  }, [props.mapId]);

  const keyframeId = handoff.keyframeId;

  useEffect(() => {
    if (!keyframeId) {
      setRows([]);
      return;
    }
    let isMounted = true;
    setIsLoadingRows(true);
    fetchMetadataRows(props.mapId, {
      keyframeIds: [keyframeId],
      limit: ROWS_PER_KEYFRAME,
    })
      .then((payload) => {
        if (isMounted) {
          setRows(payload.rows);
          setRowsError(null);
        }
      })
      .catch((error: Error) => {
        if (isMounted) {
          setRows([]);
          setRowsError(error.message);
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingRows(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [props.mapId, keyframeId]);

  const selectedRow = useMemo(
    () => rows.find((row) => row.row_index === handoff.rowIndex) ?? null,
    [rows, handoff.rowIndex],
  );

  // Rows on this keyframe that already carry a saved ground-truth annotation, so a
  // click on one re-opens what is already known instead of a blank draft.
  const annotationByRowIndex = useMemo(() => {
    const map = new Map<number, (typeof workspace.annotations)[number]>();
    if (!keyframeId) {
      return map;
    }
    for (const annotation of workspace.annotations) {
      if (annotation.source?.keyframeId === keyframeId && annotation.source.rowIndex !== null) {
        map.set(annotation.source.rowIndex, annotation);
      }
    }
    return map;
  }, [workspace.annotations, keyframeId]);

  const annotatedRowIndices = useMemo(
    () => new Set(annotationByRowIndex.keys()),
    [annotationByRowIndex],
  );

  const keyframeMarker = useMemo(
    () => markers.find((marker) => marker.id === keyframeId) ?? null,
    [markers, keyframeId],
  );

  /**
   * Resolve one image point into the draft the form saves.
   *
   * The depth pin is always resolved here, never carried across the tab switch: the
   * Explorer sends the click, this panel asks the current manifest what it means, so
   * a re-ingest between the two cannot leave a stale position in the ground truth.
   */
  const startDraft = useCallback(
    (request: {
      projection: "erp" | "cutout";
      xRatio: number;
      yRatio: number;
      rowIndex: number | null;
      region?: RegionVertex[] | null;
    }) => {
      if (!keyframeId) {
        return;
      }
      const requestId = `${Date.now()}-${Math.random()}`;
      workspace.beginDraft(requestId, {
        keyframeId,
        projection: request.projection,
        rowIndex: request.rowIndex,
        xRatio: request.xRatio,
        yRatio: request.yRatio,
        // `saveDraft` spreads the draft source into the saved feature, so these ride
        // to the store without the write path needing to know about them.
        ...(request.region?.length
          ? { erpRegion: request.region, missedDetection: true }
          : {}),
      });
      if (request.projection === "erp") {
        setErpPin({ xRatio: request.xRatio, yRatio: request.yRatio, status: "resolving" });
      }
      void resolveDepthPin(props.mapId, keyframeId, {
        projection: request.projection,
        x_ratio: request.xRatio,
        y_ratio: request.yRatio,
        row_index: request.rowIndex ?? undefined,
        sample_radius: 3,
        min_depth_m: DEPTH_PIN_MIN_DEPTH_M,
      })
        .then((payload) => {
          workspace.resolveDraft(requestId, payload);
          if (request.projection === "erp") {
            setErpPin((current) => (current ? { ...current, status: "resolved" } : current));
          }
        })
        .catch((error: Error) => {
          workspace.rejectDraft(requestId, error.message);
          if (request.projection === "erp") {
            setErpPin((current) => (current ? { ...current, status: "error" } : current));
          }
        });
    },
    [props.mapId, keyframeId, workspace],
  );

  const startDraftForRow = useCallback(
    (row: MetadataRowRecord | null) => {
      const pin = handoff.pin;
      if (pin) {
        startDraft({
          projection: pin.projection,
          xRatio: pin.xRatio,
          yRatio: pin.yRatio,
          rowIndex: pin.projection === "cutout" ? row?.row_index ?? null : null,
        });
        return;
      }
      if (!row) {
        return;
      }
      // No click came over: the proposal's own centre is the point.
      startDraft({ projection: "cutout", xRatio: 0.5, yRatio: 0.5, rowIndex: row.row_index });
    },
    [handoff.pin, startDraft],
  );

  // Arriving on a proposal, or moving to the next one, resolves its point.
  useEffect(() => {
    if (!keyframeId || isLoadingRows) {
      return;
    }
    if (handoff.rowIndex === null && !handoff.pin) {
      return;
    }
    if (startedForRowRef.current === handoff.rowIndex) {
      return;
    }
    if (handoff.rowIndex !== null && !selectedRow) {
      return;
    }
    startedForRowRef.current = handoff.rowIndex;
    // Already annotated: reopen the saved record instead of starting a blank draft.
    const existing =
      handoff.rowIndex !== null ? annotationByRowIndex.get(handoff.rowIndex) : undefined;
    if (existing) {
      workspace.editAnnotation(existing);
      return;
    }
    startDraftForRow(selectedRow);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    keyframeId,
    isLoadingRows,
    handoff.rowIndex,
    handoff.pin,
    selectedRow,
    startDraftForRow,
    annotationByRowIndex,
  ]);

  // Point the panorama at whatever is being annotated. `theta_center` is the yaw the
  // viewer takes, which is how the Explorer opens a neighbour projection too.
  useEffect(() => {
    if (!selectedRow) {
      return;
    }
    setOrient((previous) => ({
      yawRad: selectedRow.theta_center,
      token: (previous?.token ?? 0) + 1,
    }));
    setErpPin(null);
  }, [selectedRow]);

  useEffect(() => {
    setOrient(null);
    setErpPin(null);
  }, [keyframeId]);

  // Once the panorama lands on the suggestion's keyframe, aim at it — but only the
  // once: this must not fight a look-around the annotator starts afterwards.
  useEffect(() => {
    if (!activeSuggestion || activeSuggestion.targetKeyframeId !== keyframeId) {
      return;
    }
    setOrient((previous) => ({
      yawRad: activeSuggestion.orientYawRad,
      token: (previous?.token ?? 0) + 1,
    }));
  }, [activeSuggestion, keyframeId]);

  // A manual row pick abandons whatever suggestion was being positioned.
  useEffect(() => {
    if (handoff.rowIndex !== null) {
      setActiveSuggestion(null);
    }
  }, [handoff.rowIndex]);

  /** What reprojection says about where else `annotation`'s object should appear. */
  const loadNeighborSuggestions = useCallback(
    async (annotation: AnnotationFeature) => {
      const rowIndex = annotation.source?.rowIndex;
      if (rowIndex == null) {
        return;
      }
      const requestId = neighborSuggestionRequestRef.current + 1;
      neighborSuggestionRequestRef.current = requestId;
      setSuggestionSource({ annotation, rowIndex });
      setActiveSuggestion(null);
      setNeighborSuggestions(null);
      setNeighborSuggestionsLoading(true);
      setNeighborSuggestionsError(null);
      try {
        const payload = await fetchProposalNeighborProjections(
          props.mapId,
          rowIndex,
          NEIGHBOR_SUGGESTION_COUNT,
        );
        if (neighborSuggestionRequestRef.current === requestId) {
          setNeighborSuggestions(payload);
        }
      } catch (error) {
        if (neighborSuggestionRequestRef.current === requestId) {
          setNeighborSuggestionsError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (neighborSuggestionRequestRef.current === requestId) {
          setNeighborSuggestionsLoading(false);
        }
      }
    },
    [props.mapId],
  );

  // Resolves the suggestion box's centre into a depth-anchored point: once when it
  // is opened, and again every time a drag commits a new position.
  useEffect(() => {
    if (!activeSuggestion || activeSuggestion.targetKeyframeId !== keyframeId) {
      return;
    }
    if (startedSuggestionRef.current === activeSuggestion) {
      return;
    }
    startedSuggestionRef.current = activeSuggestion;
    const center = boxCenterRatio(activeSuggestion.box);
    startDraft({ projection: "erp", xRatio: center.xRatio, yRatio: center.yRatio, rowIndex: null });
  }, [activeSuggestion, keyframeId, startDraft]);

  // `saveDraft` reports success by clearing the draft and appending the stored row;
  // that is the only signal the queue may advance on, and the only moment a fresh
  // annotation exists to reproject suggestions from.
  useEffect(() => {
    const count = workspace.annotations.length;
    if (count > savedCountRef.current) {
      const saved = workspace.annotations[count - 1];
      if (advanceOnSaveRef.current) {
        advanceOnSaveRef.current = false;
        updateHandoff(withValidated(handoff));
      }
      if (suggestAfterSaveRef.current) {
        suggestAfterSaveRef.current = false;
        setActiveSuggestion(null);
        if (saved) {
          void loadNeighborSuggestions(saved);
        }
      }
    }
    savedCountRef.current = count;
  }, [workspace.annotations.length, handoff, updateHandoff, loadNeighborSuggestions]);

  const cone = useMemo<LivemapCone | null>(() => {
    if (!keyframeMarker || keyframeMarker.heading_deg == null) {
      return null;
    }
    const thetaDeg = selectedRow ? degrees(selectedRow.theta_center) : 0;
    return {
      latitude: keyframeMarker.latitude,
      longitude: keyframeMarker.longitude,
      bearingDeg: (((keyframeMarker.heading_deg + thetaDeg) % 360) + 360) % 360,
      fovDeg: CONE_FOV_DEG,
      level: keyframeMarker.level,
    };
  }, [keyframeMarker, selectedRow]);

  /**
   * The capture path, drawn as lines. Nothing renders one marker per keyframe: a map
   * has thousands of poses, so the track is a line and the keyframe under the cursor
   * is the one that appears — the mechanism the Explorer already uses.
   */
  const trackSegments = useMemo<LivemapSegment[]>(
    () => buildKeyframeTrack(markers, KEYFRAME_TRACK_COLOR),
    [markers],
  );

  const hoverRevealMarkers = useMemo<LivemapMarker[]>(
    () =>
      markers.map((marker) => ({
        id: marker.id,
        latitude: marker.latitude,
        longitude: marker.longitude,
        level: marker.level,
        // No keyframe summaries are loaded here, so no indexed/pruned state is
        // implied: every pose reads the same until it is the one being annotated.
        color: marker.id === keyframeId ? KEYFRAME_MARKER_SELECTED_COLOR : KEYFRAME_MARKER_COLOR,
        radius: marker.id === keyframeId ? KEYFRAME_MARKER_SELECTED_RADIUS : KEYFRAME_MARKER_RADIUS,
        scaleWithZoom: true,
      })),
    [markers, keyframeId],
  );

  const livemapMarkers = useMemo<LivemapMarker[]>(() => {
    const output: LivemapMarker[] = [];
    if (keyframeMarker) {
      output.push({
        id: keyframeMarker.id,
        latitude: keyframeMarker.latitude,
        longitude: keyframeMarker.longitude,
        level: keyframeMarker.level,
        color: KEYFRAME_MARKER_SELECTED_COLOR,
        radius: KEYFRAME_MARKER_SELECTED_RADIUS,
        scaleWithZoom: true,
      });
    }
    output.push(...workspace.markers);
    return output;
  }, [keyframeMarker, workspace.markers]);

  const selectKeyframe = useCallback(
    (nextKeyframeId: string) => {
      if (!markers.some((marker) => marker.id === nextKeyframeId)) {
        return;
      }
      startedForRowRef.current = null;
      updateHandoff(withKeyframe(readHandoff(props.mapId), nextKeyframeId));
    },
    [markers, props.mapId, updateHandoff],
  );

  const openNeighborSuggestion = useCallback(
    (projection: ProposalNeighborProjection) => {
      if (!suggestionSource || !keyframeId) {
        return;
      }
      // The class does not change with the keyframe, so the labels-refill effect
      // below must see it as unchanged too, or it would overwrite the prefilled
      // synonyms with that class's generic defaults.
      lastClassKeyRef.current = annotationClassKey(suggestionSource.annotation);
      workspace.beginSuggestedDraft(suggestionSource.annotation);
      setActiveSuggestion({
        source: suggestionSource,
        originKeyframeId: keyframeId,
        targetKeyframeId: projection.keyframe_id,
        orientYawRad: projection.theta_center,
        box: editableBoxFromProjection(projection),
      });
      selectKeyframe(projection.keyframe_id);
    },
    [suggestionSource, keyframeId, selectKeyframe, workspace],
  );

  const closeSuggestionReview = useCallback(() => {
    if (!activeSuggestion) {
      return;
    }
    if (workspace.editingAnnotation) {
      workspace.cancelAnnotationEdit();
    } else {
      workspace.discardDraft();
    }
    const { originKeyframeId } = activeSuggestion;
    setActiveSuggestion(null);
    selectKeyframe(originKeyframeId);
  }, [activeSuggestion, workspace, selectKeyframe]);

  const isReviewingSuggestion = Boolean(
    activeSuggestion && activeSuggestion.targetKeyframeId === keyframeId,
  );

  useEffect(() => {
    if (!isReviewingSuggestion) {
      setSuggestionViewerHeight(null);
      return;
    }
    const node = suggestionViewerRef.current;
    if (!node) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const height = entries[0]?.contentRect.height;
      if (height) {
        setSuggestionViewerHeight(height);
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [isReviewingSuggestion]);

  const cancelRegion = useCallback(() => {
    setIsDrawingRegion(false);
    setRegionVertices([]);
  }, []);

  const closeRegionWith = useCallback((vertices: RegionVertex[]) => {
    const centroid = regionCentroid(vertices);
    if (!centroid || vertices.length < 3) {
      return;
    }
    setIsDrawingRegion(false);
    // The point saved is the region's centre, resolved through depth like any other;
    // the outline rides along in the source so the size is not lost.
    startDraft({
      projection: "erp",
      xRatio: centroid[0],
      yRatio: centroid[1],
      rowIndex: null,
      region: vertices,
    });
    // Selecting nothing: a missed detection belongs to no proposal, and leaving the
    // previous row selected would attach the form to the wrong thing.
    startedForRowRef.current = null;
    updateHandoff({ ...handoff, rowIndex: null, pin: null });
    setRegionVertices([]);
  }, [startDraft, handoff, updateHandoff]);

  const closeRegion = useCallback(
    () => closeRegionWith(regionVertices),
    [closeRegionWith, regionVertices],
  );

  /**
   * A vertex, unless it lands on the first one — then the outline closes, which is
   * the gesture the livemap-tools annotator uses.
   */
  const addRegionPoint = useCallback(
    (uRatio: number, vRatio: number) => {
      const point: RegionVertex = [uRatio, vRatio];
      // Closing is a side effect, so it is decided here rather than inside a state
      // updater, which React is free to run more than once.
      if (
        regionVertices.length >= 3 &&
        ratioDistance(regionVertices[0], point) <= POLYGON_CLOSE_DISTANCE_RATIO
      ) {
        closeRegionWith(regionVertices);
        return;
      }
      setRegionVertices((current) => [...current, point]);
    },
    [regionVertices, closeRegionWith],
  );

  // A region belongs to the keyframe it was traced on.
  useEffect(() => {
    setIsDrawingRegion(false);
    setRegionVertices([]);
  }, [keyframeId]);

  /**
   * What the proposal's own box says the object measures, at the depth just resolved.
   * Offered, never applied on its own: the box is the detector's, and the depth is a
   * sample at the centre that may well belong to the wall behind.
   */
  const extentSuggestion = useMemo(() => {
    const depthM = workspace.draft?.response?.depth_m ?? null;
    const region = workspace.draft?.source?.erpRegion ?? null;
    if (workspace.draft?.status !== "resolved" || depthM === null) {
      return null;
    }
    const valueM = region?.length
      ? estimatedExtentM(region, depthM)
      : selectedRow
        ? extentFromAngularWidth(selectedRow.angular_width, selectedRow.phi_center, depthM)
        : null;
    if (valueM === null || valueM <= 0) {
      return null;
    }
    const widthDeg = region?.length
      ? null
      : Math.round((selectedRow!.angular_width * 180) / Math.PI * 10) / 10;
    return {
      valueM,
      explanation: region?.length
        ? `From the outline you drew, at ${depthM.toFixed(2)} m. Check it against the object.`
        : `From the proposal's ${widthDeg}° width at ${depthM.toFixed(2)} m. Check it against the object.`,
      onApply: () => workspace.setExtentInput(String(valueM)),
    };
  }, [workspace.draft, selectedRow, workspace.setExtentInput]);

  /**
   * Synonyms and visually-similar terms describe the class, not the point, so they are
   * carried over from what this class already holds: forced when the class changes
   * (whatever was in the fields belonged to another class) and refilled once a save
   * has cleared them. A field the annotator is in the middle of editing is left alone,
   * and editing it is how a correction becomes the class's new answer.
   */
  const lastClassKeyRef = useRef<string | null>(null);
  useEffect(() => {
    const defaults = classLabelDefaults(
      workspace.annotations,
      workspace.activeClass?.name ?? null,
    );
    const classChanged = lastClassKeyRef.current !== workspace.activeClassKey;
    lastClassKeyRef.current = workspace.activeClassKey;
    if (!defaults) {
      return;
    }
    if (defaults.synonyms.length && (classChanged || !workspace.synonymsInput.trim())) {
      workspace.setSynonymsInput(defaults.synonyms.join(", "));
    }
    if (
      defaults.visuallySimilar.length &&
      (classChanged || !workspace.visuallySimilarInput.trim())
    ) {
      workspace.setVisuallySimilarInput(defaults.visuallySimilar.join(", "));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace.activeClassKey, workspace.annotations, workspace.draft?.requestId]);

  /**
   * The poses either side of this one in the capture. `/markers` is in manifest
   * order, which is the order the capture was walked, so stepping through it is
   * walking the route rather than jumping around the map — the same assumption the
   * Explorer's "Show track" rests on.
   */
  const keyframeNeighbours = useMemo(() => {
    const index = markers.findIndex((marker) => marker.id === keyframeId);
    if (index < 0) {
      return { previous: null as string | null, next: null as string | null };
    }
    return {
      previous: index > 0 ? markers[index - 1].id : null,
      next: index + 1 < markers.length ? markers[index + 1].id : null,
    };
  }, [markers, keyframeId]);

  const remaining = handoff.queue.filter((row) => !handoff.done.includes(row)).length;
  const canSave = canSaveAnnotation(workspace);

  // Rendered either inline below the viewer, or — while reviewing a suggestion —
  // inside the large-view overlay, so the annotator can still edit and save
  // without the overlay hiding them.
  const objectControlsNode =
    mode === "object" ? (
      <ExplorerAnnotationControls
        workspace={workspace}
        sections={["mode", "classes", "object", "metadata"]}
        bare
        hideSaveActions
        extentSuggestion={extentSuggestion}
      />
    ) : null;

  const actionBarNode =
    mode === "object" && keyframeId ? (
      <div className="annotation-action-bar">
        <button
          type="button"
          className="primary-button is-annotation"
          disabled={!canSave}
          onClick={() => {
            advanceOnSaveRef.current = handoff.queue.length > 1;
            suggestAfterSaveRef.current = true;
            if (workspace.editingAnnotation) {
              workspace.saveAnnotationEdit();
            } else {
              workspace.saveDraft();
            }
          }}
        >
          {workspace.editingAnnotation
            ? "Update annotation"
            : handoff.queue.length > 1
              ? "Save and next"
              : "Save annotation"}
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={handoff.queue.length < 2}
          title="Move to the next proposal in the queue without saving this one"
          onClick={() => updateHandoff(withSkipped(handoff))}
        >
          Skip
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={!workspace.draft && !workspace.editingAnnotation}
          title="Drop the resolved point and start this proposal again"
          onClick={() => {
            if (activeSuggestion && activeSuggestion.targetKeyframeId === keyframeId) {
              closeSuggestionReview();
              return;
            }
            setActiveSuggestion(null);
            if (workspace.editingAnnotation) {
              workspace.cancelAnnotationEdit();
            } else {
              workspace.discardDraft();
            }
          }}
        >
          Discard
        </button>
        <span className="annotation-action-hint">
          {activeSuggestion && activeSuggestion.targetKeyframeId === keyframeId
            ? "Positioning a suggested copy of this object — drag the corners, then save."
            : `${remaining} left in this queue · the point is re-resolved on the newest manifest`}
        </span>
      </div>
    ) : null;

  if (!props.isMapKnown) {
    return (
      <section className="annotation-panel-tab page-section">
        <p className="error-box">Map not found in the current config.</p>
      </section>
    );
  }

  return (
    <section className="annotation-panel-tab">
      <div className="annotation-tab-header">
        <div>
          <p className="eyebrow is-annotation">Annotation</p>
          <h2>Describe and validate</h2>
        </div>
        <div className="annotation-tab-header-actions">
          <div className="annotation-mode-switch" role="tablist" aria-label="Annotation mode">
            {MODES.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={mode === item.id}
                title={item.hint}
                className={`annotation-mode-tab${mode === item.id ? " is-active" : ""}`}
                onClick={() => setMode(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <details className="annotation-io-menu">
            <summary aria-label="Import and export">⋯</summary>
            <div className="annotation-io-menu-body">
              <ExplorerAnnotationControls workspace={workspace} sections={["io"]} bare />
            </div>
          </details>
        </div>
      </div>

      {workspace.errorMessage ? (
        <DismissibleAlert variant="error" onDismiss={() => workspace.setErrorMessage(null)}>
          {workspace.errorMessage}
        </DismissibleAlert>
      ) : null}
      {workspace.infoMessage ? (
        <DismissibleAlert variant="info" onDismiss={() => workspace.setInfoMessage(null)}>
          {workspace.infoMessage}
        </DismissibleAlert>
      ) : null}
      {rowsError ? <p className="error-box">{rowsError}</p> : null}

      <div className={`annotation-capsule${isMapOpen ? "" : " is-map-collapsed"}`}>
        <div
          className={`annotation-capsule-main${
            isReviewingSuggestion ? " is-suggestion-review" : ""
          }`}
        >
          {isReviewingSuggestion ? (
            <button
              type="button"
              className="annotation-suggestion-close"
              aria-label="Close suggestion review"
              title="Close this suggestion (discards the positioned copy)"
              onClick={closeSuggestionReview}
            >
              ×
            </button>
          ) : null}
          <div className="annotation-capsule-bar">
            <span className="explorer-context-label">Keyframe</span>
            <strong className="annotation-capsule-id">{keyframeId ?? "none"}</strong>
            <span
              className="keyframe-stepper keyframe-stepper--compact"
              aria-label="Step through the capture"
            >
              <button
                type="button"
                aria-label="Previous keyframe"
                title={
                  keyframeNeighbours.previous
                    ? `Previous keyframe: ${keyframeNeighbours.previous}`
                    : "No earlier keyframe in the capture"
                }
                disabled={!keyframeNeighbours.previous}
                onClick={() =>
                  keyframeNeighbours.previous && selectKeyframe(keyframeNeighbours.previous)
                }
              >
                &lt;
              </button>
              <button
                type="button"
                aria-label="Next keyframe"
                title={
                  keyframeNeighbours.next
                    ? `Next keyframe: ${keyframeNeighbours.next}`
                    : "No later keyframe in the capture"
                }
                disabled={!keyframeNeighbours.next}
                onClick={() =>
                  keyframeNeighbours.next && selectKeyframe(keyframeNeighbours.next)
                }
              >
                &gt;
              </button>
            </span>
            <span className="annotation-capsule-divider" aria-hidden="true" />
            <span className="explorer-context-label">Proposal</span>
            <strong className="annotation-capsule-id">
              {handoff.rowIndex ?? (handoff.pin ? "panorama point" : "none")}
            </strong>
            {selectedRow ? (
              <>
                <span className="annotation-capsule-angles">
                  θ {degrees(selectedRow.theta_center)}° · φ {degrees(selectedRow.phi_center)}°
                  {selectedRow.depth != null
                    ? ` · ${selectedRow.depth.toFixed(1)} m`
                    : " · no depth"}
                </span>
                {/*
                  What the detector thought this was. It is a prediction, not ground
                  truth — that is the whole reason someone is annotating it — so it is
                  labelled as one and applied only on request, to the prompt, which is
                  per point, never to the synonyms, which belong to the class.
                */}
                {selectedRow.label ? (
                  <button
                    type="button"
                    className="annotation-predicted-label"
                    title={`Predicted by ${selectedRow.detector_source || "an unknown detector"}${
                      selectedRow.detection_score != null
                        ? ` at ${selectedRow.detection_score.toFixed(3)}`
                        : ""
                    } — click to use it as the prompt`}
                    onClick={() => workspace.setPromptInput(selectedRow.label ?? "")}
                  >
                    <span className="annotation-predicted-label-kicker">predicted</span>
                    {selectedRow.label}
                    {selectedRow.detection_score != null ? (
                      <span className="annotation-predicted-label-score">
                        {selectedRow.detection_score.toFixed(2)}
                      </span>
                    ) : null}
                  </button>
                ) : (
                  <span className="annotation-capsule-angles">unlabelled proposal</span>
                )}
              </>
            ) : null}
            <span className="annotation-capsule-queue">
              Queue {handoff.done.length} / {handoff.queue.length || 1}
            </span>
            {isDrawingRegion ? (
              <span className="annotation-region-controls">
                <span className="annotation-region-hint">
                  {regionVertices.length} point{regionVertices.length === 1 ? "" : "s"}
                  {regionVertices.length < 3
                    ? " · 3 needed"
                    : " · click the first point to close"}
                </span>
                <button
                  type="button"
                  className="primary-button is-annotation"
                  disabled={regionVertices.length < 3}
                  onClick={closeRegion}
                >
                  Close the outline
                </button>
                <button type="button" className="secondary-button" onClick={cancelRegion}>
                  Cancel
                </button>
              </span>
            ) : (
              <button
                type="button"
                className="secondary-button"
                disabled={!keyframeId}
                title="Trace an object no detector proposed, then describe it like any other"
                onClick={() => {
                  setRegionVertices([]);
                  setIsDrawingRegion(true);
                }}
              >
                Missed detection
              </button>
            )}
            <button
              type="button"
              className="secondary-button annotation-return-button"
              onClick={props.onBackToProposal}
            >
              ↩ Back to the proposal
            </button>
          </div>

          {keyframeId ? (
            <>
              <div
                ref={suggestionViewerRef}
                className={`annotation-capsule-viewer${
                  hoveredRowIndex !== null ? " is-over-detection" : ""
                }${isDrawingRegion ? " is-drawing-region" : ""}${
                  isReviewingSuggestion ? " is-suggestion-review-viewer" : ""
                }`}
                style={isReviewingSuggestion ? undefined : { height: viewerHeight }}
                onPointerDown={(event) => {
                  pointerDownRef.current = { x: event.clientX, y: event.clientY };
                }}
                onPointerUp={(event) => {
                  const start = pointerDownRef.current;
                  pointerDownRef.current = null;
                  // Ctrl+click is the depth pin, and a drag is the viewer looking
                  // around: neither may change the selection under the annotator.
                  if (!start || event.ctrlKey || event.metaKey || isDrawingRegion) {
                    return;
                  }
                  const moved = Math.hypot(event.clientX - start.x, event.clientY - start.y);
                  if (moved > CLICK_SLOP_PX) {
                    return;
                  }
                  const rowIndex = hoveredRowRef.current;
                  if (rowIndex !== null && rowIndex !== handoff.rowIndex) {
                    updateHandoff(withSelection(handoff, rowIndex));
                  }
                }}
              >
                <Suspense fallback={<p className="muted">Loading photosphere viewer...</p>}>
                  <EquirectPhotoSphereViewer
                    keyframeId={keyframeId}
                    imageSrc={indexKeyframeEquirectPreviewUrl(props.mapId, keyframeId, null, undefined, {
                      drawBoxes: false,
                    })}
                    height={
                      isReviewingSuggestion && suggestionViewerHeight
                        ? suggestionViewerHeight
                        : viewerHeight
                    }
                    initialYawRad={selectedRow?.theta_center ?? 0}
                    initialTextureYRatio={0.5}
                    orientYawRad={orient?.yawRad}
                    orientToken={orient?.token}
                    detections={rows}
                    selectedRowIndex={handoff.rowIndex}
                    annotatedRowIndices={annotatedRowIndices}
                    editableBox={
                      activeSuggestion && activeSuggestion.targetKeyframeId === keyframeId
                        ? activeSuggestion.box
                        : null
                    }
                    onEditableBoxChange={(box) =>
                      setActiveSuggestion((current) => (current ? { ...current, box } : current))
                    }
                    depthPin={erpPin}
                    polygonForDetection={bboxPolygonRatios}
                    onDepthPin={(xRatio, yRatio) =>
                      startDraft({ projection: "erp", xRatio, yRatio, rowIndex: null })
                    }
                    allowDepthPinOnMarker
                    onHoverDetection={(hovered) => {
                      hoveredRowRef.current = hovered?.rowIndex ?? null;
                      setHoveredRowIndex(hovered?.rowIndex ?? null);
                    }}
                    regionDrawActive={isDrawingRegion}
                    draftRegion={regionVertices}
                    onRegionPoint={addRegionPoint}
                  />
                </Suspense>
              </div>
              {isReviewingSuggestion ? null : (
                <div
                  className={`object-search-explorer-equirect-resizer${
                    isResizingViewer ? " is-dragging" : ""
                  }`}
                  role="separator"
                  aria-orientation="horizontal"
                  aria-label="Resize the panorama"
                  aria-valuenow={viewerHeight}
                  aria-valuemin={EQUIRECT_FRAME_MIN}
                  aria-valuemax={EQUIRECT_FRAME_MAX}
                  title="Drag to resize the panorama"
                  onPointerDown={startViewerResize}
                />
              )}
              <AngularRibbon
                rows={rows}
                selectedRowIndex={handoff.rowIndex}
                doneRowIndices={handoff.done}
                queuedRowIndices={handoff.queue}
                onSelect={(rowIndex) => updateHandoff(withSelection(handoff, rowIndex))}
              />
              {isReviewingSuggestion ? (
                <div className="annotation-suggestion-review-controls">
                  {objectControlsNode}
                  {actionBarNode}
                </div>
              ) : null}
            </>
          ) : (
            <p className="annotation-capsule-empty">
              Pick a keyframe on the map — hover the capture path and click the pose you
              want — or send a proposal over from the Explorer with “Annotate this
              proposal”.
            </p>
          )}
        </div>

        <div className="annotation-capsule-side">
          <div className="annotation-capsule-side-header">
            <span className="explorer-context-label">Livemap</span>
            <span className="muted">{workspace.currentLevel ?? "all levels"}</span>
            <button
              type="button"
              className="annotation-map-toggle"
              aria-expanded={isMapOpen}
              title={isMapOpen ? "Collapse the map" : "Expand the map"}
              onClick={() => setIsMapOpen((open) => !open)}
            >
              {isMapOpen ? "▾" : "▸"}
            </button>
          </div>
          {isMapOpen ? (
            <>
              <div className="annotation-capsule-livemap">
                {props.map?.emmid != null ? (
                  <LivemapAnnotation
                    emmid={props.map.emmid}
                    markers={livemapMarkers}
                    segments={trackSegments}
                    polygons={workspace.polygons}
                    draftVertices={[]}
                    draftClosed={false}
                    pendingPoint={null}
                    activeColor={workspace.activeClass?.color ?? null}
                    mode={mode === "zones" ? "polygon" : "inspect"}
                    focusTarget={workspace.focusTarget}
                    onMapClick={() => {}}
                    onMarkerClick={workspace.selectAnnotationMarker}
                    onLevelChange={workspace.setCurrentLevel}
                    cone={cone}
                    coneColor={VIEW_CONE_COLOR}
                    segmentHoverMarkers={hoverRevealMarkers}
                    onSegmentMarkerClick={selectKeyframe}
                    roiActive={workspace.roiActive}
                    roiPolygon={workspace.roiPolygon}
                    onRoiChange={workspace.setRoiPolygon}
                    height="100%"
                  />
                ) : (
                  <p className="muted">This map has no emmid, so the livemap cannot open.</p>
                )}
              </div>
              <p className="annotation-capsule-map-hint">
                Hover the capture path to reveal a keyframe, click it to annotate that
                turn.
              </p>
            </>
          ) : null}
          {selectedRow?.thumbnail_key ? (
            <img
              className="annotation-capsule-cutout"
              src={rowThumbnailUrl(props.mapId, selectedRow.thumbnail_key)}
              alt={`Proposal ${selectedRow.row_index}`}
            />
          ) : null}
        </div>
      </div>

      <div className="annotation-mode-body">
        {mode === "object" && suggestionSource ? (
          <CollapsibleSection
            title="Suggested on nearby keyframes"
            summary={
              neighborSuggestionsLoading
                ? "Projecting..."
                : neighborSuggestions
                  ? `${neighborSuggestions.projections.length} views`
                  : "—"
            }
            defaultOpen
          >
            <p className="map-caption">
              Same object ({suggestionSource.annotation.groundTruth.objectId ?? "unnamed"}),
              reprojected from keyframe {suggestionSource.annotation.source?.keyframeId}. The
              box lands close but rarely exact — drag its corners to fit before saving.
            </p>
            {neighborSuggestionsError ? (
              <p className="warning-box" role="alert">
                {neighborSuggestionsError}
              </p>
            ) : null}
            {neighborSuggestions ? (
              neighborSuggestions.projections.length ? (
                <div className="annotation-suggestion-strip">
                  {neighborSuggestions.projections.map((projection) => (
                    <button
                      key={projection.keyframe_id}
                      type="button"
                      className={`annotation-suggestion-thumb${
                        activeSuggestion?.targetKeyframeId === projection.keyframe_id
                          ? " is-active"
                          : ""
                      }`}
                      aria-label={`Suggest this object on keyframe ${projection.keyframe_id}`}
                      title={`Keyframe ${projection.keyframe_id} · ${projection.distance_from_source_m.toFixed(
                        1,
                      )} m from source · ${(projection.geometric_confidence * 100).toFixed(0)}% confidence`}
                      onClick={() => openNeighborSuggestion(projection)}
                    >
                      <span className="annotation-suggestion-thumb-stage">
                        <img
                          src={proposalNeighborProjectionRenderUrl(
                            props.mapId,
                            suggestionSource.rowIndex,
                            projection.keyframe_id,
                            { size: 160, fovScale: 2 },
                          )}
                          alt={`Object projected into keyframe ${projection.keyframe_id}`}
                          loading="lazy"
                        />
                        <span className="object-search-neighbor-projection-box" aria-hidden="true" />
                      </span>
                      <span className="annotation-suggestion-thumb-caption">
                        Keyframe {projection.keyframe_id}
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="muted">{neighborSuggestions.note}</p>
              )
            ) : null}
          </CollapsibleSection>
        ) : null}
        {isReviewingSuggestion ? null : objectControlsNode}
        {mode === "zones" ? (
          <ExplorerAnnotationControls workspace={workspace} sections={["roi"]} bare />
        ) : null}
        {mode === "review" ? <ExplorerAnnotationList workspace={workspace} defaultOpen /> : null}
      </div>

      {isReviewingSuggestion ? null : actionBarNode}
    </section>
  );
}

export default AnnotationPanel;
