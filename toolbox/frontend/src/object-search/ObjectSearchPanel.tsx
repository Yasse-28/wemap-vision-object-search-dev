import type { CSSProperties, FormEvent, PointerEvent as ReactPointerEvent } from "react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { NavigationCandidate } from "../object-search-explorer/EquirectPhotoSphereViewer";
import { ReviewButtons } from "../object-search-review/ReviewControls";
import {
  type ObjectSearchReviews,
  useObjectSearchReviews,
} from "../object-search-review/useObjectSearchReviews";

import type { MapSummary } from "../api";
import LivemapAnnotation, { type LivemapCone } from "../annotations/LivemapAnnotation";
import type { FocusTarget, LivemapMarker, LivemapSegment } from "../annotations/types";
import {
  fetchKeyframeGraph,
  fetchMetadataStatus,
  indexKeyframeEquirectPreviewUrl,
} from "../index-explorer/api";
import type { KeyframeGraphResponse, MetadataStatusResponse } from "../index-explorer/types";
import { runObjectSearch } from "./api";
import type {
  EnrichedResult,
  KeyframeGroup,
  ObjectLocalization,
  ObjectObservation,
  ObjectSearchRunResult,
  OnlineLocalizeOverrides,
  QueryInputMode,
} from "./types";
import { DEFAULT_ONLINE_OVERRIDES, ObjectSearchMode } from "./types";
import {
  buildKeyframeGroups,
  cutoutPreviewUrl,
  formatNumber,
  localizationColorForMatch,
  localizationMatchThreshold,
  localizationScoreSymbol,
  markerColorForScore,
} from "./utils";
import "./object-search-vision.css";

const EquirectPhotoSphereViewer = lazy(
  () => import("../object-search-explorer/EquirectPhotoSphereViewer"),
);

// ── Constants ────────────────────────────────────────────────────────────────

const KEYFRAME_MARKER_COLOR = "#9ca3af";
const KEYFRAME_MARKER_SELECTED_COLOR = "#16a34a";
const KEYFRAME_GRAPH_COLOR = "#64748b";
const VIEW_CONE_HALF_ANGLE_DEG = 25;
const VIEW_CONE_COLOR = KEYFRAME_MARKER_SELECTED_COLOR;
const COL_MIN = 25;
const COL_MAX = 75;
const ROW_MIN = 20;
const ROW_MAX = 80;
const EARTH_RADIUS_M = 6378137;
const EMPTY_LOCALIZATIONS: ObjectLocalization[] = [];

function projectKeyframeToLocalFloor(
  source: { latitude: number; longitude: number; heading_deg: number },
  destination: { latitude: number; longitude: number },
): Pick<NavigationCandidate, "localX" | "localZ"> {
  const latitudeRad = (source.latitude * Math.PI) / 180;
  const east =
    ((destination.longitude - source.longitude) * Math.PI) / 180 *
    EARTH_RADIUS_M *
    Math.cos(latitudeRad);
  const north =
    ((destination.latitude - source.latitude) * Math.PI) / 180 * EARTH_RADIUS_M;
  const distance = Math.hypot(east, north);
  const bearing = Math.atan2(east, north);
  const localYaw = bearing - (source.heading_deg * Math.PI) / 180;
  return { localX: distance * Math.sin(localYaw), localZ: -distance * Math.cos(localYaw) };
}

// ── Types ────────────────────────────────────────────────────────────────────

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
  reviewMode?: boolean;
};

type BboxDraft = { left: number; top: number; width: number; height: number };

// ── BboxDrawOverlay ──────────────────────────────────────────────────────────

function BboxDrawOverlay(props: {
  photosphereWrapRef: React.RefObject<HTMLDivElement | null>;
  onCapture: (file: File) => void;
  onCancel: () => void;
  onError: (msg: string) => void;
}) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<BboxDraft | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const startRef = useRef<{ x: number; y: number } | null>(null);

  const hasDraft = draft && draft.width > 8 && draft.height > 8;

  function capture() {
    if (!draft || !props.photosphereWrapRef.current) {
      props.onError("Draw a bounding box before using it as an image query.");
      return;
    }
    const wrap = props.photosphereWrapRef.current;
    const canvas = wrap.querySelector<HTMLCanvasElement>(".object-search-keyframe-photosphere-canvas");
    if (!canvas) {
      props.onError("Viewer canvas not found — is the panorama loaded?");
      return;
    }

    const wrapRect = wrap.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    const scaleX = canvasRect.width > 0 ? canvas.width / canvasRect.width : 1;
    const scaleY = canvasRect.height > 0 ? canvas.height / canvasRect.height : 1;

    // draft coords are relative to the overlay (same origin as wrap)
    const x = (draft.left - (canvasRect.left - wrapRect.left)) * scaleX;
    const y = (draft.top - (canvasRect.top - wrapRect.top)) * scaleY;
    const w = draft.width * scaleX;
    const h = draft.height * scaleY;

    if (w < 4 || h < 4) {
      props.onError("The selected bounding box is too small to capture.");
      return;
    }

    const tmp = document.createElement("canvas");
    tmp.width = Math.round(w);
    tmp.height = Math.round(h);
    const ctx = tmp.getContext("2d");
    if (!ctx) {
      props.onError("Could not create the bounding-box image.");
      return;
    }
    try {
      ctx.drawImage(canvas, x, y, w, h, 0, 0, tmp.width, tmp.height);
    } catch (err) {
      props.onError(`Failed to capture bbox: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    tmp.toBlob((blob) => {
      if (!blob) {
        props.onError("Failed to encode bbox crop (toBlob returned null).");
        return;
      }
      props.onCapture(new File([blob], "bbox-crop.png", { type: "image/png" }));
    }, "image/png");
  }

  return (
    <div
      ref={overlayRef}
      className="os-bbox-overlay"
      onPointerDown={(e) => {
        const rect = overlayRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        startRef.current = { x, y };
        setDraft(null);
        setIsDragging(true);
        overlayRef.current!.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!isDragging || !startRef.current || !overlayRef.current) return;
        const rect = overlayRef.current.getBoundingClientRect();
        const ex = e.clientX - rect.left;
        const ey = e.clientY - rect.top;
        setDraft({
          left: Math.min(startRef.current.x, ex),
          top: Math.min(startRef.current.y, ey),
          width: Math.abs(ex - startRef.current.x),
          height: Math.abs(ey - startRef.current.y),
        });
      }}
      onPointerUp={() => setIsDragging(false)}
    >
      {draft ? (
        <div
          className="os-bbox-rect"
          style={{ left: draft.left, top: draft.top, width: draft.width, height: draft.height }}
        />
      ) : null}
      <div
        className="os-bbox-actions"
        onPointerDown={(event) => event.stopPropagation()}
        onPointerMove={(event) => event.stopPropagation()}
        onPointerUp={(event) => event.stopPropagation()}
      >
        {hasDraft ? (
          <button type="button" className="object-search-button os-pane-btn" onClick={capture}>
            Use as query
          </button>
        ) : null}
        <button
          type="button"
          className="object-search-secondary-button os-pane-btn"
          onClick={props.onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

function ObjectSearchPanel(props: Props) {
  // ── Existing search state ──────────────────────────────────────────────────
  const [queryInputMode, setQueryInputMode] = useState<QueryInputMode>("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [numResults, setNumResults] = useState(100);
  const [maxObservationsPerCluster, setMaxObservationsPerCluster] = useState(1000);
  const [localizeSensitivity, setLocalizeSensitivity] = useState(10);
  const [searchMode, setSearchMode] = useState<ObjectSearchMode>("localize-online");
  const [searchType, setSearchType] = useState<"auto" | "cutout" | "object">("auto");
  const [onlineOverrides, setOnlineOverrides] =
    useState<OnlineLocalizeOverrides>(DEFAULT_ONLINE_OVERRIDES);
  const [showOnlineOverrides, setShowOnlineOverrides] = useState(false);
  const [useRemoteServer, setUseRemoteServer] = useState(false);
  const [remoteServerUrl, setRemoteServerUrl] = useState(
    "https://vps-api.wemap-vision-computing-1.getwemap.com/{map_id}/object-search/localize",
  );
  const [result, setResult] = useState<ObjectSearchRunResult | null>(null);
  const [resultQuery, setResultQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const [selectedKeyframeId, setSelectedKeyframeId] = useState<string | null>(null);
  const [selectedCutoutId, setSelectedCutoutId] = useState<string | null>(null);
  const [selectedLocIdx, setSelectedLocIdx] = useState(0);
  const [selectedObsIdx, setSelectedObsIdx] = useState(0);
  const [clusterFocus, setClusterFocus] = useState<
    (FocusTarget & { requestId: number }) | null
  >(null);

  const imageInputRef = useRef<HTMLInputElement>(null);

  // ── New: keyframe / map state ──────────────────────────────────────────────
  const [indexStatus, setIndexStatus] = useState<MetadataStatusResponse | null>(null);
  const [keyframeGraph, setKeyframeGraph] = useState<KeyframeGraphResponse | null>(null);
  const [showKeyframeMarkers, setShowKeyframeMarkers] = useState(true);
  const [showKeyframeGraph, setShowKeyframeGraph] = useState(true);
  const [showDebugRays, setShowDebugRays] = useState(false);

  // ── New: panorama state ────────────────────────────────────────────────────
  const [activeKeyframeId, setActiveKeyframeId] = useState<string | null>(null);
  const [photosphereYawRad, setPhotosphereYawRad] = useState<number | null>(null);
  const [photosphereViewKeyframeId, setPhotosphereViewKeyframeId] = useState<string | null>(null);
  const [bboxMode, setBboxMode] = useState(false);
  const [panoramaExpanded, setPanoramaExpanded] = useState(false);
  const [panoramaViewRequestId, setPanoramaViewRequestId] = useState(0);
  const preservedCompassBearingDegRef = useRef<number | null>(null);
  const preservedTextureYRatioRef = useRef(0.5);

  // ── New: layout split state ────────────────────────────────────────────────
  const [colWidthPct, setColWidthPct] = useState(55);
  const [rowHeightPct, setRowHeightPct] = useState(45);
  const panelRef = useRef<HTMLElement>(null);
  const photosphereWrapRef = useRef<HTMLDivElement>(null);

  // ── New: floating search controls (draggable + collapsible) ────────────────
  const [controlsPos, setControlsPos] = useState<{ x: number; y: number } | null>(null);
  const [controlsSize, setControlsSize] = useState<{ w: number; h: number } | null>(null);
  const [controlsCollapsed, setControlsCollapsed] = useState(false);
  const controlsRef = useRef<HTMLDivElement>(null);

  const isLocalizeMode = searchMode !== "text";
  const emmid = props.map?.emmid ?? null;

  // ── Fetch keyframe data ────────────────────────────────────────────────────
  useEffect(() => {
    if (!props.isMapKnown) return;
    let cancelled = false;
    // Only `markers` is used here — every keyframe with a pose, which is now the
    // case whether or not the map has object-search metadata.
    fetchMetadataStatus(props.mapId, null)
      .then((s) => {
        if (!cancelled) setIndexStatus(s);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown]);

  useEffect(() => {
    if (!props.isMapKnown) return;
    let cancelled = false;
    fetchKeyframeGraph(props.mapId)
      .then((g) => {
        if (!cancelled) setKeyframeGraph(g);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown]);

  // ── Existing: detections for selected cutout ───────────────────────────────
  const keyframeGroups = useMemo(() => {
    if (!result || result.mode !== "text") return {} as Record<string, KeyframeGroup>;
    return buildKeyframeGroups(result.enriched);
  }, [result]);

  const selectedGroup = selectedKeyframeId ? keyframeGroups[selectedKeyframeId] : null;
  const selectedCutout: EnrichedResult | null = useMemo(() => {
    if (!selectedGroup) return null;
    const cutoutId =
      selectedCutoutId ?? selectedGroup.results[0]?.cutout_id ?? selectedGroup.results[0]?.id;
    return (
      selectedGroup.results.find((item) => (item.cutout_id ?? item.id) === cutoutId) ??
      selectedGroup.results[0] ??
      null
    );
  }, [selectedGroup, selectedCutoutId]);

  const allLocalizations =
    result && result.mode !== "text" ? result.localizations : EMPTY_LOCALIZATIONS;
  const reviews = useObjectSearchReviews({
    enabled: props.reviewMode === true,
    mapId: props.mapId,
    query: result?.mode === "localize-online" ? resultQuery : "",
    localizations: allLocalizations,
  });
  const localizationDisplayThreshold = localizationMatchThreshold(localizeSensitivity);
  const localizations = useMemo(
    () =>
      allLocalizations.filter(
        (loc) => loc.matchScore >= localizationDisplayThreshold,
      ),
    [allLocalizations, localizationDisplayThreshold],
  );
  const keyframeCoords = result && result.mode !== "text" ? result.keyframes : {};
  const keyframeMapPositions = useMemo(() => {
    const positions = { ...keyframeCoords };
    for (const marker of indexStatus?.markers ?? []) {
      if (positions[marker.id]) {
        continue;
      }
      positions[marker.id] = {
        lat: marker.latitude,
        lon: marker.longitude,
        alt: null,
        level: marker.level,
      };
    }
    return positions;
  }, [indexStatus?.markers, keyframeCoords]);
  const selectedLoc = localizations[selectedLocIdx] ?? null;
  const selectedObservation = selectedLoc?.observations[selectedObsIdx] ?? null;
  const selectedObservationCamera =
    selectedObservation ? keyframeMapPositions[selectedObservation.keyframeId] : null;

  useEffect(() => {
    if (selectedLocIdx !== 0 && selectedLocIdx >= localizations.length) setSelectedLocIdx(0);
  }, [localizations.length, selectedLocIdx]);

  useEffect(() => {
    if (!selectedLoc) {
      setSelectedObsIdx(0);
      return;
    }
    if (selectedObsIdx >= selectedLoc.observations.length) setSelectedObsIdx(0);
  }, [selectedLoc, selectedObsIdx]);

  // ── Derived: keyframe marker for active KF ─────────────────────────────────
  const activeKeyframeMarker =
    indexStatus?.markers?.find((m) => m.id === activeKeyframeId) ?? null;
  const initialPhotosphereView = useMemo(() => {
    const headingDeg = activeKeyframeMarker?.heading_deg;
    const bearingDeg = preservedCompassBearingDegRef.current;
    const yawRad =
      headingDeg !== null &&
      headingDeg !== undefined &&
      bearingDeg !== null
        ? Math.atan2(
            Math.sin(((bearingDeg - headingDeg) * Math.PI) / 180),
            Math.cos(((bearingDeg - headingDeg) * Math.PI) / 180),
          )
        : 0;
    return {
      yawRad,
      textureYRatio: preservedTextureYRatioRef.current,
    };
  }, [activeKeyframeId, activeKeyframeMarker?.heading_deg]);

  const photosphereImageSrc = activeKeyframeId
    ? indexKeyframeEquirectPreviewUrl(props.mapId, activeKeyframeId, null, undefined, {
        drawBoxes: false,
      })
    : null;

  // ── Markers ────────────────────────────────────────────────────────────────
  const textMarkers: LivemapMarker[] = useMemo(() => {
    if (!selectedGroup || result?.mode !== "text") return [];
    return Object.entries(keyframeGroups).flatMap(([keyframeId, group]) => {
      if (group.lat == null || group.lon == null) return [];
      return [
        {
          id: keyframeId,
          latitude: group.lat,
          longitude: group.lon,
          level: group.level,
          color: markerColorForScore(group.bestScore, keyframeId === selectedKeyframeId),
        },
      ];
    });
  }, [keyframeGroups, selectedGroup, selectedKeyframeId, result]);

  const localizeMarkers: LivemapMarker[] = useMemo(() => {
    const markers: LivemapMarker[] = [];
    for (let idx = 0; idx < localizations.length; idx++) {
      const loc = localizations[idx];
      markers.push({
        id: `obj_${idx}`,
        latitude: loc.lat,
        longitude: loc.lng,
        level: loc.level,
        color: localizationColorForMatch(loc, localizationDisplayThreshold),
        radius: idx === selectedLocIdx ? 8 : 6,
      });
    }
    if (selectedObservation?.coordinates) {
      markers.push({
        id: `selected_obs_${selectedLocIdx}_${selectedObsIdx}`,
        latitude: selectedObservation.coordinates[0],
        longitude: selectedObservation.coordinates[1],
        level: selectedObservationCamera?.level ?? null,
        color: "#f97316",
        radius: 7,
        strokeColor: "#ffffff",
        strokeWidth: 2,
      });
    }
    if (showDebugRays && selectedLoc) {
      for (let obsIdx = 0; obsIdx < selectedLoc.observations.length; obsIdx++) {
        const obs = selectedLoc.observations[obsIdx];
        if (!obs.coordinates) continue;
        markers.push({
          id: `obs_${selectedLocIdx}_${obsIdx}`,
          latitude: obs.coordinates[0],
          longitude: obs.coordinates[1],
          level: null,
          color: "#f97316",
          radius: 4,
        });
      }
    }
    return markers;
  }, [
    localizations,
    selectedLocIdx,
    selectedObsIdx,
    showDebugRays,
    selectedLoc,
    selectedObservation,
    selectedObservationCamera,
    localizationDisplayThreshold,
  ]);

  const resultMarkers = result?.mode === "text" ? textMarkers : localizeMarkers;

  const allMapMarkers: LivemapMarker[] = useMemo(() => {
    const markers: LivemapMarker[] = [];
    if (showKeyframeMarkers && indexStatus?.markers) {
      for (const m of indexStatus.markers) {
        const isActive = m.id === activeKeyframeId;
        markers.push({
          id: `kf_${m.id}`,
          latitude: m.latitude,
          longitude: m.longitude,
          level: m.level,
          color: isActive ? KEYFRAME_MARKER_SELECTED_COLOR : KEYFRAME_MARKER_COLOR,
          radius: isActive ? 8 : 6,
          scaleWithZoom: true,
        });
      }
    }
    for (const m of resultMarkers) {
      markers.push(m);
    }
    return markers;
  }, [indexStatus?.markers, showKeyframeMarkers, activeKeyframeId, resultMarkers]);

  // ── Segments ───────────────────────────────────────────────────────────────
  const localizeSegments: LivemapSegment[] = useMemo(() => {
    const segments: LivemapSegment[] = [];
    if (selectedObservation?.coordinates && selectedObservationCamera) {
      segments.push({
        id: `selected_ray_${selectedLocIdx}_${selectedObsIdx}`,
        fromLatitude: selectedObservationCamera.lat,
        fromLongitude: selectedObservationCamera.lon,
        toLatitude: selectedObservation.coordinates[0],
        toLongitude: selectedObservation.coordinates[1],
        color: "#f97316",
        width: 1,
        opacity: 0.95,
        level: selectedObservationCamera.level,
      });
    }
    if (showDebugRays && selectedLoc) {
      for (let obsIdx = 0; obsIdx < selectedLoc.observations.length; obsIdx++) {
        const obs = selectedLoc.observations[obsIdx];
        if (!obs.coordinates) continue;
        const cam = keyframeMapPositions[obs.keyframeId];
        if (!cam) continue;
        segments.push({
          id: `ray_${selectedLocIdx}_${obsIdx}`,
          fromLatitude: cam.lat,
          fromLongitude: cam.lon,
          toLatitude: obs.coordinates[0],
          toLongitude: obs.coordinates[1],
          color: "#f97316",
          width: obsIdx === selectedObsIdx ? 1 : 2,
          opacity: obsIdx === selectedObsIdx ? 0.95 : 0.45,
          level: cam.level,
        });
      }
    }
    return segments;
  }, [
    showDebugRays,
    selectedLoc,
    keyframeMapPositions,
    selectedLocIdx,
    selectedObsIdx,
    selectedObservation,
    selectedObservationCamera,
  ]);

  const allMapSegments: LivemapSegment[] = useMemo(() => {
    const segments: LivemapSegment[] = [];
    if (showKeyframeGraph && keyframeGraph?.available && keyframeGraph.edges) {
      for (const edge of keyframeGraph.edges) {
        const levels = edge.levels.length ? edge.levels : [null];
        for (let li = 0; li < levels.length; li++) {
          segments.push({
            id: `graph_${edge.id}-${li}`,
            fromLongitude: edge.from[0],
            fromLatitude: edge.from[1],
            toLongitude: edge.to[0],
            toLatitude: edge.to[1],
            color: KEYFRAME_GRAPH_COLOR,
            width: 1,
            opacity: 0.5,
            level: levels[li],
          });
        }
      }
    }
    for (const s of localizeSegments) {
      segments.push(s);
    }
    return segments;
  }, [showKeyframeGraph, keyframeGraph, localizeSegments]);

  // ── Cone ───────────────────────────────────────────────────────────────────
  const livemapCone = useMemo<LivemapCone | null>(() => {
    if (
      !activeKeyframeMarker ||
      photosphereViewKeyframeId !== activeKeyframeId ||
      photosphereYawRad === null ||
      activeKeyframeMarker.heading_deg == null
    )
      return null;
    const bearingDeg =
      (((activeKeyframeMarker.heading_deg + (photosphereYawRad * 180) / Math.PI) % 360) + 360) %
      360;
    return {
      latitude: activeKeyframeMarker.latitude,
      longitude: activeKeyframeMarker.longitude,
      bearingDeg,
      fovDeg: VIEW_CONE_HALF_ANGLE_DEG * 2,
      level: activeKeyframeMarker.level,
    };
  }, [activeKeyframeMarker, photosphereViewKeyframeId, activeKeyframeId, photosphereYawRad]);

  // ── Keyframe navigation ────────────────────────────────────────────────────
  const keyframeIds = useMemo(
    () => indexStatus?.markers?.map((m) => m.id) ?? [],
    [indexStatus],
  );

  const activeKeyframeIndex = activeKeyframeId ? keyframeIds.indexOf(activeKeyframeId) : -1;
  const hasPreviousKeyframe = activeKeyframeIndex > 0;
  const hasNextKeyframe = activeKeyframeIndex >= 0 && activeKeyframeIndex < keyframeIds.length - 1;

  const photosphereNavigationCandidates = useMemo<NavigationCandidate[]>(() => {
    if (!activeKeyframeId || !activeKeyframeMarker || activeKeyframeMarker.heading_deg == null || !indexStatus?.markers) {
      return [];
    }
    return indexStatus.markers.flatMap((marker) => {
      if (marker.id === activeKeyframeId || marker.level !== activeKeyframeMarker.level) return [];
      return [{
        id: marker.id,
        ...projectKeyframeToLocalFloor(
          {
            latitude: activeKeyframeMarker.latitude,
            longitude: activeKeyframeMarker.longitude,
            heading_deg: activeKeyframeMarker.heading_deg as number,
          },
          marker,
        ),
      }];
    });
  }, [activeKeyframeId, activeKeyframeMarker, indexStatus?.markers]);

  const navigateToKeyframe = useCallback(
    (keyframeId: string, compassBearingDeg?: number) => {
      if (compassBearingDeg !== undefined && Number.isFinite(compassBearingDeg)) {
        preservedCompassBearingDegRef.current =
          ((compassBearingDeg % 360) + 360) % 360;
        setPanoramaViewRequestId((requestId) => requestId + 1);
      }
      setClusterFocus(null);
      setActiveKeyframeId(keyframeId);
      setPhotosphereYawRad(null);
      setPhotosphereViewKeyframeId(null);
      setBboxMode(false);
    },
    [],
  );

  const selectAdjacentKeyframe = useCallback((direction: -1 | 1) => {
    if (activeKeyframeIndex < 0) return;
    const nextId = keyframeIds[Math.max(0, Math.min(keyframeIds.length - 1, activeKeyframeIndex + direction))];
    if (nextId && nextId !== activeKeyframeId) navigateToKeyframe(nextId);
  }, [activeKeyframeIndex, keyframeIds, activeKeyframeId, navigateToKeyframe]);

  // ── Map focus ──────────────────────────────────────────────────────────────
  const mapFocus = useMemo(() => {
    if (clusterFocus) {
      return clusterFocus;
    }
    if (activeKeyframeMarker) {
      return {
        latitude: activeKeyframeMarker.latitude,
        longitude: activeKeyframeMarker.longitude,
        level: activeKeyframeMarker.level,
      };
    }
    if (result?.mode === "text" && selectedGroup?.lat != null && selectedGroup.lon != null) {
      return { latitude: selectedGroup.lat, longitude: selectedGroup.lon, level: selectedGroup.level };
    }
    if (selectedLoc) {
      return { latitude: selectedLoc.lat, longitude: selectedLoc.lng, level: selectedLoc.level };
    }
    return null;
  }, [clusterFocus, activeKeyframeMarker, result, selectedGroup, selectedLoc]);

  const previewUrl = selectedCutout?.preview_path
    ? cutoutPreviewUrl(props.mapId, selectedCutout.preview_path)
    : null;

  // ── Submit ─────────────────────────────────────────────────────────────────
  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (!props.isMapKnown || !props.map) {
      setError("Map not found in the current config.");
      return;
    }
    if (queryInputMode === "text" && !text.trim()) {
      setError("Enter a query.");
      return;
    }
    if (queryInputMode === "image" && !imageFile) {
      setError("Upload a query image.");
      return;
    }
    setIsLoading(true);
    try {
      const next = await runObjectSearch({
        mapId: props.mapId,
        mapPath: props.map.path,
        searchMode,
        queryInputMode,
        text: text.trim(),
        imageFile,
        numResults,
        searchType,
        maxObservationsPerCluster,
        onlineOverrides,
        remoteUrl:
          useRemoteServer && searchMode === "localize-online"
            ? remoteServerUrl.replace("{map_id}", encodeURIComponent(props.mapId))
            : null,
      });
      setResult(next);
      setResultQuery(imageFile?.name ?? text.trim());
      setSelectedKeyframeId(null);
      setSelectedCutoutId(null);
      setSelectedLocIdx(0);
      setSelectedObsIdx(0);
      setClusterFocus(null);
      if (next.mode === "text" && next.enriched.length) {
        const groups = buildKeyframeGroups(next.enriched);
        const firstId = Object.keys(groups)[0] ?? null;
        setSelectedKeyframeId(firstId);
      }
    } catch (requestError) {
      setResult(null);
      setResultQuery("");
      setError(requestError instanceof Error ? requestError.message : String(requestError));
    } finally {
      setIsLoading(false);
    }
  }

  const focusLocalization = useCallback(
    (locIndex: number) => {
      const localization = localizations[locIndex];
      if (!localization) {
        return;
      }
      setSelectedLocIdx(locIndex);
      setSelectedObsIdx(0);
      setClusterFocus((current) => ({
        latitude: localization.lat,
        longitude: localization.lng,
        level: localization.level,
        requestId: (current?.requestId ?? 0) + 1,
      }));
    },
    [localizations],
  );

  const selectLocalizationObservation = useCallback(
    (locIndex: number, obsIndex: number) => {
      const observation = localizations[locIndex]?.observations[obsIndex];
      if (!observation) {
        return;
      }
      setSelectedLocIdx(locIndex);
      setSelectedObsIdx(obsIndex);
      navigateToKeyframe(
        observation.keyframeId,
        observation.heading === null ? undefined : observation.heading,
      );
    },
    [localizations, navigateToKeyframe],
  );

  // ── Marker click ───────────────────────────────────────────────────────────
  const handleMarkerClick = useCallback(
    (markerId: string) => {
      if (markerId.startsWith("kf_")) {
        const kfId = markerId.slice(3);
        const selectedObservationMatches =
          selectedObservation?.keyframeId === kfId &&
          selectedObservation.heading !== null;
        const matchingObservationIndex = selectedObservationMatches
          ? selectedObsIdx
          : selectedLoc?.observations.findIndex(
              (observation) =>
                observation.keyframeId === kfId && observation.heading !== null,
            );
        const selectedHeading =
          matchingObservationIndex !== undefined && matchingObservationIndex >= 0
            ? selectedLoc?.observations[matchingObservationIndex]?.heading
            : undefined;
        if (matchingObservationIndex !== undefined && matchingObservationIndex >= 0) {
          setSelectedObsIdx(matchingObservationIndex);
        }
        navigateToKeyframe(
          kfId,
          selectedHeading === null || selectedHeading === undefined
            ? undefined
            : selectedHeading,
        );
        return;
      }
      if (!result) return;
      if (markerId.startsWith("obj_")) {
        const idx = Number(markerId.split("_")[1]);
        if (Number.isFinite(idx)) {
          focusLocalization(idx);
        }
        return;
      }
      if (result.mode === "text") {
        setSelectedKeyframeId(markerId);
        setSelectedCutoutId(null);
      }
    },
    [
      focusLocalization,
      navigateToKeyframe,
      result,
      selectedLoc,
      selectedObservation,
    ],
  );

  // ── View change ────────────────────────────────────────────────────────────
  const handleViewChange = useCallback(
    (kfId: string, yawRad: number, textureYRatio: number) => {
      if (kfId !== activeKeyframeId) return;
      setPhotosphereViewKeyframeId(kfId);
      setPhotosphereYawRad(yawRad);
      preservedTextureYRatioRef.current = textureYRatio;
      const headingDeg = indexStatus?.markers?.find((marker) => marker.id === kfId)?.heading_deg;
      if (headingDeg !== null && headingDeg !== undefined) {
        preservedCompassBearingDegRef.current =
          ((headingDeg + (yawRad * 180) / Math.PI) % 360 + 360) % 360;
      }
    },
    [activeKeyframeId, indexStatus?.markers],
  );

  // ── Layout resize ──────────────────────────────────────────────────────────
  const startHResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const panel = panelRef.current;
    if (!panel) return;
    const startY = event.clientY;
    const panelH = panel.getBoundingClientRect().height;
    const startPct = rowHeightPct;
    document.body.classList.add("os-resizing-h");
    const onMove = (e: PointerEvent) => {
      const next = Math.max(ROW_MIN, Math.min(ROW_MAX, startPct + ((e.clientY - startY) / panelH) * 100));
      setRowHeightPct(next);
    };
    const onUp = () => {
      document.body.classList.remove("os-resizing-h");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const startVResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const panel = panelRef.current;
    if (!panel) return;
    const startX = event.clientX;
    const panelW = panel.getBoundingClientRect().width;
    const startPct = colWidthPct;
    document.body.classList.add("os-resizing-v");
    const onMove = (e: PointerEvent) => {
      const next = Math.max(COL_MIN, Math.min(COL_MAX, startPct + ((e.clientX - startX) / panelW) * 100));
      setColWidthPct(next);
    };
    const onUp = () => {
      document.body.classList.remove("os-resizing-v");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const startControlsDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest("button")) return;
    event.preventDefault();
    const card = controlsRef.current;
    const parent = card?.parentElement as HTMLElement | null;
    if (!card || !parent) return;
    const cardRect = card.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;
    const originX = cardRect.left - parentRect.left;
    const originY = cardRect.top - parentRect.top;
    document.body.classList.add("os-dragging");
    const onMove = (e: PointerEvent) => {
      const maxX = Math.max(0, parentRect.width - cardRect.width);
      const maxY = Math.max(0, parentRect.height - cardRect.height);
      setControlsPos({
        x: Math.max(0, Math.min(maxX, originX + (e.clientX - startX))),
        y: Math.max(0, Math.min(maxY, originY + (e.clientY - startY))),
      });
    };
    const onUp = () => {
      document.body.classList.remove("os-dragging");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const startControlsResize =
    (dir: string) => (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();
      const card = controlsRef.current;
      const parent = card?.parentElement as HTMLElement | null;
      if (!card || !parent) return;
      const cardRect = card.getBoundingClientRect();
      const parentRect = parent.getBoundingClientRect();
      const origin = {
        left: cardRect.left - parentRect.left,
        top: cardRect.top - parentRect.top,
        width: cardRect.width,
        height: cardRect.height,
      };
      const startX = event.clientX;
      const startY = event.clientY;
      const MIN_W = 240;
      const MIN_H = 80;
      const handle = event.currentTarget;
      handle.setPointerCapture(event.pointerId);
      document.body.classList.add("os-dragging");
      const onMove = (e: PointerEvent) => {
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        let { left, top, width, height } = origin;
        if (dir.includes("e")) width = origin.width + dx;
        if (dir.includes("s")) height = origin.height + dy;
        if (dir.includes("w")) {
          width = origin.width - dx;
          left = origin.left + dx;
        }
        if (dir.includes("n")) {
          height = origin.height - dy;
          top = origin.top + dy;
        }
        if (width < MIN_W) {
          if (dir.includes("w")) left -= MIN_W - width;
          width = MIN_W;
        }
        if (height < MIN_H) {
          if (dir.includes("n")) top -= MIN_H - height;
          height = MIN_H;
        }
        left = Math.max(0, Math.min(left, parentRect.width - width));
        top = Math.max(0, Math.min(top, parentRect.height - height));
        width = Math.min(width, parentRect.width - left);
        height = Math.min(height, parentRect.height - top);
        setControlsPos({ x: left, y: top });
        setControlsSize({ w: width, h: height });
      };
      const onUp = (e: PointerEvent) => {
        document.body.classList.remove("os-dragging");
        handle.releasePointerCapture(e.pointerId);
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
      };
      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
    };

  // ── Render ─────────────────────────────────────────────────────────────────
  const hasPanorama = !!activeKeyframeId;

  return (
    <section
      ref={panelRef}
      className={`object-search-panel os-panel${
        panoramaExpanded ? " os-panel--panorama-expanded" : ""
      }`}
      style={
        {
          "--os-col-w": `${colWidthPct}%`,
          "--os-row-h": `${rowHeightPct}%`,
        } as CSSProperties
      }
    >
      {/* ── Left column: panorama (top) + map (bottom) ── */}
      <div className="os-cell-left">
      {/* ── Panorama ── */}
      <div className="os-cell-panorama">
        {hasPanorama && photosphereImageSrc ? (
          <div className="os-pane">
            <div className="os-pane-header">
              <span className="os-pane-title">Panorama</span>
              <span className="os-pane-subtitle">{activeKeyframeId}</span>
              <div className="keyframe-stepper os-kf-stepper">
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
              <div className="os-pane-header-actions">
                <button
                  type="button"
                  className="object-search-secondary-button os-pane-btn"
                  onClick={() => setPanoramaExpanded((expanded) => !expanded)}
                  aria-label={panoramaExpanded ? "Collapse panorama" : "Expand panorama"}
                  title={panoramaExpanded ? "Collapse panorama" : "Expand panorama"}
                >
                  {panoramaExpanded ? "Collapse" : "Expand"}
                </button>
                <button
                  type="button"
                  className="object-search-secondary-button os-pane-btn"
                  onClick={() => {
                    setPanoramaExpanded(false);
                    setActiveKeyframeId(null);
                    setBboxMode(false);
                    setPhotosphereYawRad(null);
                    setPhotosphereViewKeyframeId(null);
                  }}
                  aria-label="Close panorama"
                >
                  ×
                </button>
              </div>
            </div>
            <div ref={photosphereWrapRef} className="os-photosphere-wrap">
              <Suspense
                fallback={<div className="os-photosphere-loading">Loading viewer…</div>}
              >
                <EquirectPhotoSphereViewer
                  key={`${activeKeyframeId}-${panoramaViewRequestId}`}
                  keyframeId={activeKeyframeId}
                  imageSrc={photosphereImageSrc}
                  height={1000}
                  initialYawRad={initialPhotosphereView.yawRad}
                  initialTextureYRatio={initialPhotosphereView.textureYRatio}
                  detections={[]}
                  selectedRowIndex={null}
                  depthPin={null}
                  polygonForDetection={() => []}
                  onDepthPin={() => {}}
                  onViewChange={handleViewChange}
                  navigationCandidates={photosphereNavigationCandidates}
                  onNavigate={navigateToKeyframe}
                />
              </Suspense>
              {bboxMode ? (
                <BboxDrawOverlay
                  photosphereWrapRef={photosphereWrapRef}
                  onCapture={(file) => {
                    setImageFile(file);
                    setQueryInputMode("image");
                    setSearchMode("localize-online");
                    setBboxMode(false);
                  }}
                  onCancel={() => setBboxMode(false)}
                  onError={(msg) => { setError(msg); setBboxMode(false); }}
                />
              ) : null}
            </div>
          </div>
        ) : (
          <div className="os-panorama-placeholder">
            <p>Click a keyframe on the map to view its panorama</p>
          </div>
        )}

        {/* ── Floating search controls (draggable, collapsible) ── */}
        <div
          ref={controlsRef}
          className={`os-floating-controls${controlsPos ? " is-dragged" : ""}${controlsCollapsed ? " is-collapsed" : ""}`}
          style={{
            ...(controlsPos ? { left: controlsPos.x, top: controlsPos.y } : {}),
            ...(controlsSize ? { width: controlsSize.w } : {}),
            ...(controlsSize && !controlsCollapsed ? { height: controlsSize.h } : {}),
          }}
        >
          <div className="os-floating-controls-header" onPointerDown={startControlsDrag}>
            <span className="os-floating-controls-title">
              {props.reviewMode ? "Annotation" : "Object Search"}
            </span>
            {result ? (
              <span className="os-pane-subtitle">
                {result.mode === "text"
                  ? `${result.enriched.length} results`
                  : `${localizations.length} clusters`}
              </span>
            ) : null}
            <button
              type="button"
              className="os-floating-controls-collapse"
              onClick={() => setControlsCollapsed((v) => !v)}
              aria-label={controlsCollapsed ? "Expand controls" : "Collapse controls"}
              title={controlsCollapsed ? "Expand" : "Collapse"}
            >
              {controlsCollapsed ? "▾" : "▴"}
            </button>
          </div>
          <div className="os-floating-controls-body">
          {error ? (
            <div className="object-search-error-banner" role="alert" style={{ margin: "8px 12px 0" }}>
              <span>{error}</span>
              <button type="button" onClick={() => setError(null)} aria-label="Dismiss">
                ×
              </button>
            </div>
          ) : null}
          <form className="os-search-form" onSubmit={onSubmit}>
            {/* Query input row */}
            <div className="object-search-toolbar-row">
              {queryInputMode === "text" ? (
                <input
                  className="object-search-input"
                  type="text"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Search objects…"
                  disabled={!props.isMapKnown || isLoading}
                />
              ) : imageFile ? (
                <div className="object-search-selected-image-chip">
                  <span>{imageFile.name}</span>
                  <button
                    type="button"
                    className="object-search-chip-clear-button"
                    onClick={() => setImageFile(null)}
                    disabled={isLoading}
                    aria-label="Clear image"
                  >
                    ×
                  </button>
                </div>
              ) : (
                <span className="object-search-status-row">Select an image query below</span>
              )}
              <input
                className="object-search-num-results"
                type="number"
                min={1}
                max={1000}
                value={numResults}
                onChange={(e) => setNumResults(Number(e.target.value))}
                disabled={!props.isMapKnown || isLoading}
                title="Number of results"
              />
              <button
                className="object-search-button"
                type="submit"
                disabled={!props.isMapKnown || isLoading}
              >
                {isLoading ? "Searching…" : "Search"}
              </button>
            </div>

            {/* Input mode toggle */}
            <div className="object-search-toolbar-row">
              <div className="object-search-mode-toggle" role="group" aria-label="Query input">
                <button
                  type="button"
                  className={`object-search-secondary-button${queryInputMode === "text" ? " active-query" : ""}`}
                  onClick={() => setQueryInputMode("text")}
                >
                  Text
                </button>
                <button
                  type="button"
                  className={`object-search-secondary-button${queryInputMode === "image" ? " active-query" : ""}`}
                  onClick={() => setQueryInputMode("image")}
                >
                  Image
                </button>
                <button
                  type="button"
                  className={`object-search-secondary-button${bboxMode ? " active-query" : ""}`}
                  onClick={() => {
                    if (!activeKeyframeId) return;
                    setQueryInputMode("image");
                    setBboxMode((m) => !m);
                  }}
                  disabled={!activeKeyframeId}
                  title={activeKeyframeId ? "Draw a bbox on the panorama" : "Select a keyframe first"}
                >
                  Draw bbox
                </button>
              </div>
              {queryInputMode === "image" ? (
                <>
                  <input
                    ref={imageInputRef}
                    className="object-search-image-input"
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
                    disabled={!props.isMapKnown || isLoading}
                  />
                  <button
                    type="button"
                    className={`object-search-secondary-button${imageFile ? " active-query" : ""}`}
                    onClick={() => imageInputRef.current?.click()}
                    disabled={!props.isMapKnown || isLoading}
                  >
                    {imageFile ? "Change image" : "Pick image"}
                  </button>
                </>
              ) : null}
            </div>

            {/* Search mode */}
            <div className="object-search-toolbar-row object-search-toolbar-controls">
              <div className="object-search-mode-toggle" role="group" aria-label="Search mode">
                <button
                  type="button"
                  className={`object-search-secondary-button${searchMode === "text" ? " active-query" : ""}`}
                  onClick={() => setSearchMode("text")}
                >
                  Text search
                </button>
                <button
                  type="button"
                  className={`object-search-secondary-button${searchMode === "localize-online" ? " active-query" : ""}`}
                  onClick={() => setSearchMode("localize-online")}
                >
                  Localize
                </button>
              </div>
              {searchMode === "text" ? (
                <label className="object-search-inline-field">
                  Search type
                  <select
                    className="object-search-inline-select"
                    value={searchType}
                    onChange={(e) => setSearchType(e.target.value as "auto" | "cutout" | "object")}
                    disabled={!props.isMapKnown || isLoading}
                  >
                    <option value="auto">Auto</option>
                    <option value="cutout">Cutout</option>
                    <option value="object">Object</option>
                  </select>
                </label>
              ) : null}
            </div>

            {/* Localize sliders */}
            {isLocalizeMode ? (
              <div className="object-search-toolbar-row object-search-toolbar-controls">
                <div className="object-search-slider-control">
                  <div className="object-search-slider-header">
                    <label htmlFor="os-sensitivity">Sensitivity</label>
                    <span>{localizeSensitivity}%</span>
                  </div>
                  <input
                    id="os-sensitivity"
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={localizeSensitivity}
                    onChange={(e) => setLocalizeSensitivity(Number(e.target.value))}
                    disabled={!props.isMapKnown || isLoading}
                  />
                  <div className="object-search-slider-help">
                    Shows localizations with match_score &gt;= 1 - sensitivity / 100.
                  </div>
                </div>
                <div className="object-search-slider-control">
                  <div className="object-search-slider-header">
                    <label htmlFor="os-max-obs">Max observations / cluster</label>
                    <span>{maxObservationsPerCluster}</span>
                  </div>
                  <input
                    id="os-max-obs"
                    type="range"
                    min={1}
                    max={2000}
                    step={1}
                    value={Math.min(maxObservationsPerCluster, 2000)}
                    onChange={(e) => setMaxObservationsPerCluster(Number(e.target.value))}
                    disabled={!props.isMapKnown || isLoading}
                  />
                </div>
                {searchMode === "localize-online" ? (
                  <div className="object-search-slider-control">
                    <div className="object-search-slider-header">
                      <label htmlFor="os-merge-radius">Merge radius</label>
                      <span>{onlineOverrides.merge_radius.toFixed(1)} m</span>
                    </div>
                    <input
                      id="os-merge-radius"
                      type="range"
                      min={0}
                      max={7}
                      step={0.1}
                      value={onlineOverrides.merge_radius}
                      onChange={(e) =>
                        setOnlineOverrides({ ...onlineOverrides, merge_radius: Number(e.target.value) })
                      }
                      disabled={!props.isMapKnown || isLoading}
                    />
                  </div>
                ) : null}
              </div>
            ) : null}

            {searchMode === "localize-online" ? (
              <div className="object-search-toolbar-row checkbox-row--compact os-remote-row">
                <label>
                  <input
                    type="checkbox"
                    checked={useRemoteServer}
                    onChange={(e) => setUseRemoteServer(e.target.checked)}
                  />
                  Remote server
                </label>
                {useRemoteServer ? (
                  <input
                    className="object-search-input os-remote-url-input"
                    type="url"
                    value={remoteServerUrl}
                    onChange={(e) => setRemoteServerUrl(e.target.value)}
                    placeholder="https://…/{map_id}/object-search/localize"
                    spellCheck={false}
                  />
                ) : null}
              </div>
            ) : null}

            {searchMode === "localize-online" ? (
              <CollapsibleOnlineOverrides
                open={showOnlineOverrides}
                onToggle={() => setShowOnlineOverrides((v) => !v)}
                overrides={onlineOverrides}
                onChange={setOnlineOverrides}
              />
            ) : null}

            {isLocalizeMode && result ? (
              <div className="object-search-toolbar-row checkbox-row--compact">
                <label>
                  <input
                    type="checkbox"
                    checked={showDebugRays}
                    onChange={(e) => setShowDebugRays(e.target.checked)}
                  />
                  Show depth rays
                </label>
              </div>
            ) : null}
          </form>
          </div>
          {!controlsCollapsed ? (
            <>
              <div className="os-rz os-rz-e" onPointerDown={startControlsResize("e")} />
              <div className="os-rz os-rz-w" onPointerDown={startControlsResize("w")} />
              <div className="os-rz os-rz-s" onPointerDown={startControlsResize("s")} />
              <div className="os-rz os-rz-n" onPointerDown={startControlsResize("n")} />
              <div className="os-rz os-rz-se" onPointerDown={startControlsResize("se")} />
              <div className="os-rz os-rz-sw" onPointerDown={startControlsResize("sw")} />
              <div className="os-rz os-rz-ne" onPointerDown={startControlsResize("ne")} />
              <div className="os-rz os-rz-nw" onPointerDown={startControlsResize("nw")} />
            </>
          ) : null}
        </div>
      </div>

      {/* ── Horizontal divider (panorama ↔ map) ── */}
      <div
        className="os-h-divider"
        onPointerDown={startHResize}
        role="separator"
        aria-orientation="horizontal"
      />

      {/* ── Bottom-left: Livemap ── */}
      <div className="os-cell-map">
        <div className="os-pane">
          <div className="os-pane-header">
            <span className="os-pane-title">Map</span>
            <div className="checkbox-row--compact" style={{ marginLeft: 8 }}>
              <label>
                <input
                  type="checkbox"
                  checked={showKeyframeMarkers}
                  onChange={(e) => setShowKeyframeMarkers(e.target.checked)}
                />
                Keyframes
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showKeyframeGraph}
                  onChange={(e) => setShowKeyframeGraph(e.target.checked)}
                />
                Graph
              </label>
            </div>
          </div>
          <div className="os-map-wrap">
            {emmid === null ? (
              <div className="object-search-map-placeholder">
                <p className="info-box">Map preview unavailable: no emmid configured.</p>
              </div>
            ) : (
              <LivemapAnnotation
                emmid={emmid}
                markers={allMapMarkers}
                segments={allMapSegments}
                polygons={[]}
                draftVertices={[]}
                draftClosed={false}
                pendingPoint={null}
                activeColor={null}
                mode="inspect"
                focusTarget={mapFocus}
                cone={livemapCone}
                coneColor={VIEW_CONE_COLOR}
                onMapClick={() => {}}
                onMarkerClick={handleMarkerClick}
                height="100%"
              />
            )}
          </div>
        </div>
      </div>
      </div>

      {/* ── Vertical divider (left column ↔ results) ── */}
      <div
        className="os-v-divider"
        onPointerDown={startVResize}
        role="separator"
        aria-orientation="vertical"
      />

      {/* ── Bottom-right: Results ── */}
      <div className="os-cell-results">
        <div className="os-pane os-results-pane">
          <div className="os-pane-header">
            <span className="os-pane-title">Results</span>
          </div>
          <div className="os-results-scroll">
            {props.reviewMode && result?.mode === "localize-online" ? (
              <div className="object-search-review-toolbar">
                <span>
                  {reviews.isLoading
                    ? "Loading reviews…"
                    : `${reviews.reviewedCount} reviewed · ${reviews.truePositiveCount} correct · ${reviews.falsePositiveCount} incorrect`}
                </span>
                <span className="object-search-review-history-actions">
                  <button
                    type="button"
                    className="object-search-secondary-button"
                    disabled={!reviews.canUndo}
                    onClick={reviews.undo}
                    title="Undo review (Ctrl/Cmd+Z)"
                  >
                    Undo
                  </button>
                  <button
                    type="button"
                    className="object-search-secondary-button"
                    disabled={!reviews.canRedo}
                    onClick={reviews.redo}
                    title="Redo review (Ctrl/Cmd+Shift+Z)"
                  >
                    Redo
                  </button>
                </span>
              </div>
            ) : null}
            {props.reviewMode && reviews.error ? (
              <div className="object-search-error-banner" role="alert">
                <span>Annotations: {reviews.error}</span>
              </div>
            ) : null}
            {!result ? (
              <p className="os-results-empty">Run a search to see results here.</p>
            ) : result.mode === "text" ? (
              <div className="os-results-dock-inner">
                <section className="object-search-dock-overview">
                  <h3>Overview</h3>
                  <OverviewSection result={result} keyframeGroups={keyframeGroups} />
                </section>
                <section className="object-search-dock-inspector">
                  <h3>Inspector</h3>
                  <TextInspector
                    group={selectedGroup}
                    selectedCutout={selectedCutout}
                    selectedCutoutId={selectedCutoutId}
                    onSelectCutout={setSelectedCutoutId}
                    previewUrl={previewUrl}
                  />
                </section>
              </div>
            ) : (
              <div className="os-results-dock-inner os-results-dock-inner--localize">
                <section className="object-search-cluster-sidebar">
                  <LocalizeInspector
                    mapId={props.mapId}
                    localizations={localizations}
                    matchThreshold={localizationDisplayThreshold}
                    selectedIdx={selectedLocIdx}
                    onSelectIdx={focusLocalization}
                    selectedObsIdx={selectedObsIdx}
                    onSelectObservation={selectLocalizationObservation}
                    reviews={props.reviewMode ? reviews : null}
                  />
                </section>
                <section className="object-search-localize-detail">
                  <div className="object-search-localize-overview">
                    <h3>Overview</h3>
                    <OverviewSection
                      result={result}
                      keyframeGroups={keyframeGroups}
                      localizations={localizations}
                    />
                  </div>
                  <SelectedObservationPanel
                    mapId={props.mapId}
                    localization={selectedLoc}
                    localizationIndex={selectedLocIdx}
                    observation={selectedObservation}
                    observationIndex={selectedObsIdx}
                  />
                </section>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function CollapsibleOnlineOverrides(props: {
  open: boolean;
  onToggle: () => void;
  overrides: OnlineLocalizeOverrides;
  onChange: (value: OnlineLocalizeOverrides) => void;
}) {
  return (
    <div className="object-search-toolbar-row object-search-localize-mode-row">
      <button
        type="button"
        className={`object-search-secondary-button${props.open ? " active" : ""}`}
        onClick={props.onToggle}
        aria-expanded={props.open}
      >
        {props.open ? "Hide overrides" : "Online overrides"}
      </button>
      {props.open ? (
        <div className="object-search-online-overrides">
          <label className="object-search-online-checkbox">
            <input
              type="checkbox"
              checked={props.overrides.use_stored_positions}
              onChange={(event) =>
                props.onChange({
                  ...props.overrides,
                  use_stored_positions: event.target.checked,
                })
              }
            />
            <span>Use stored 3D positions</span>
          </label>
          <label className="object-search-online-checkbox">
            <input
              type="checkbox"
              checked={props.overrides.robust_centroid}
              onChange={(event) =>
                props.onChange({ ...props.overrides, robust_centroid: event.target.checked })
              }
            />
            <span>Robust centroid</span>
          </label>
          <label className="object-search-online-checkbox">
            <input
              type="checkbox"
              checked={props.overrides.include_debug}
              onChange={(event) =>
                props.onChange({ ...props.overrides, include_debug: event.target.checked })
              }
            />
            <span>Include debug keyframes (slower)</span>
          </label>
          <label className="object-search-online-input">
            <span>embedding_similarity_threshold</span>
            <input
              type="number"
              step={0.01}
              value={props.overrides.embedding_similarity_threshold}
              onChange={(event) =>
                props.onChange({
                  ...props.overrides,
                  embedding_similarity_threshold: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="object-search-online-input">
            <span>min_keyframes_per_cluster</span>
            <input
              type="number"
              value={props.overrides.min_keyframes_per_cluster}
              onChange={(event) =>
                props.onChange({
                  ...props.overrides,
                  min_keyframes_per_cluster: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="object-search-online-input">
            <span>feedback_alpha</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={props.overrides.feedback_alpha}
              onChange={(event) =>
                props.onChange({
                  ...props.overrides,
                  feedback_alpha: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="object-search-online-input">
            <span>feedback_beta</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={props.overrides.feedback_beta}
              onChange={(event) =>
                props.onChange({
                  ...props.overrides,
                  feedback_beta: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="object-search-online-input">
            <span>candidate_count</span>
            <input
              type="number"
              value={props.overrides.candidate_count}
              onChange={(event) =>
                props.onChange({
                  ...props.overrides,
                  candidate_count: Number(event.target.value),
                })
              }
            />
          </label>
        </div>
      ) : null}
    </div>
  );
}

function OverviewSection(props: {
  result: ObjectSearchRunResult | null;
  keyframeGroups: Record<string, KeyframeGroup>;
  localizations?: ObjectLocalization[];
}) {
  const endpointLabel = props.result?.debug.url ?? null;
  const elapsedLabel =
    props.result?.debug.elapsedMs == null
      ? null
      : `${Math.round(props.result.debug.elapsedMs)} ms`;
  return (
    <div className="object-search-overview-body">
      {!props.result ? (
        <p className="muted">No results yet. Run a query.</p>
      ) : (
        <>
          <div className="object-search-endpoint-row">
            {endpointLabel ? <span title={endpointLabel}>{endpointLabel}</span> : null}
            {elapsedLabel ? <strong>{elapsedLabel}</strong> : null}
          </div>
          {props.result.mode === "text" ? (
            <>
              <div className="metrics-row">
                <span>Keyframes: {Object.keys(props.keyframeGroups).length}</span>
                <span>Results: {props.result.enriched.length}</span>
              </div>
              <details>
                <summary>Ranked results</summary>
                <ul className="ranked-list">
                  {props.result.enriched.map((item) => (
                    <li key={`${item.rank}-${item.id}`}>
                      <strong>
                        #{item.rank} {item.id}
                      </strong>{" "}
                      score {formatNumber(item.score)}
                      {item.keyframe_id ? ` · keyframe ${item.keyframe_id}` : ""}
                      {item.resolution_error ? (
                        <small className="block-muted">{item.resolution_error}</small>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </details>
            </>
          ) : (
            <>
              <div className="metrics-row">
                <span>Clusters: {(props.localizations ?? props.result.localizations).length}</span>
                <span>
                  Detections:{" "}
                  {(props.localizations ?? props.result.localizations).reduce(
                    (sum, loc) => sum + loc.observationCount,
                    0,
                  )}
                </span>
              </div>
              <details>
                <summary>Ranked localizations</summary>
                <ul className="ranked-list">
                  {(props.localizations ?? props.result.localizations).map((loc, index) => (
                    <li key={`loc-${index}`}>
                      <strong>
                        #{index + 1} match {formatNumber(loc.matchScore)}
                      </strong>{" "}
                      sim {formatNumber(loc.similarityScore)}
                      <small className="block-muted">
                        {formatNumber(loc.lat)}, {formatNumber(loc.lng)} ·{" "}
                        {loc.observationCount} detections
                      </small>
                    </li>
                  ))}
                </ul>
              </details>
            </>
          )}
        </>
      )}
    </div>
  );
}

function TextInspector(props: {
  group: KeyframeGroup | null;
  selectedCutout: EnrichedResult | null;
  selectedCutoutId: string | null;
  onSelectCutout: (id: string) => void;
  previewUrl: string | null;
}) {
  if (!props.group) {
    return <p className="muted">Select a keyframe on the map.</p>;
  }
  const cutoutOptions = props.group.results.map((item) => item.cutout_id ?? item.id);
  return (
    <>
      <p>
        <strong>Keyframe {props.group.keyframeId}</strong>
        <br />
        <span className="muted">
          {props.group.results.length} cutouts · best {formatNumber(props.group.bestScore)}
        </span>
      </p>
      <label>
        Cutout
        <select
          value={props.selectedCutoutId ?? cutoutOptions[0] ?? ""}
          onChange={(event) => props.onSelectCutout(event.target.value)}
        >
          {cutoutOptions.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </label>
      {props.previewUrl ? (
        <img className="inspector-preview" src={props.previewUrl} alt="Cutout preview" />
      ) : (
        <p className="muted">Preview unavailable for this cutout.</p>
      )}
    </>
  );
}

function LocalizeInspector(props: {
  mapId: string;
  localizations: ObjectLocalization[];
  matchThreshold: number;
  selectedIdx: number;
  onSelectIdx: (idx: number) => void;
  selectedObsIdx: number;
  onSelectObservation: (locIdx: number, obsIdx: number) => void;
  reviews: ObjectSearchReviews | null;
}) {
  const [collapsedClusters, setCollapsedClusters] = useState<Set<number>>(() => new Set());

  useEffect(() => {
    setCollapsedClusters(new Set());
  }, [props.localizations]);

  const allClustersCollapsed =
    props.localizations.length > 0 &&
    collapsedClusters.size === props.localizations.length;

  if (!props.localizations.length) {
    return (
      <>
        <div className="object-search-cluster-sidebar-header">
          <h3>Clusters</h3>
        </div>
        <p className="muted">No localizations.</p>
      </>
    );
  }
  return (
    <>
      <div className="object-search-cluster-sidebar-header">
        <h3>Clusters</h3>
        <button
          type="button"
          className="object-search-secondary-button object-search-collapse-all-button"
          disabled={allClustersCollapsed}
          onClick={() =>
            setCollapsedClusters(
              new Set(props.localizations.map((_, locIndex) => locIndex)),
            )
          }
        >
          Collapse all
        </button>
      </div>
      <div className="object-search-cluster-list">
        {props.localizations.map((loc, locIndex) => {
          const isSelectedCluster = locIndex === props.selectedIdx;
          const isCollapsed = collapsedClusters.has(locIndex);
          const clusterReviewStatus = props.reviews?.clusterStatus(loc) ?? null;
          return (
            <article
              className={`object-search-cluster-group${
                isSelectedCluster ? " is-selected" : ""
              }${clusterReviewStatus ? ` is-review-${clusterReviewStatus}` : ""}`}
              key={`cluster-${locIndex}`}
            >
              <div className="object-search-cluster-header">
                <button
                  type="button"
                  className="object-search-cluster-collapse-button"
                  aria-label={`${isCollapsed ? "Expand" : "Collapse"} cluster ${locIndex + 1}`}
                  aria-expanded={!isCollapsed}
                  onClick={() => {
                    setCollapsedClusters((current) => {
                      const next = new Set(current);
                      if (next.has(locIndex)) {
                        next.delete(locIndex);
                      } else {
                        next.add(locIndex);
                      }
                      return next;
                    });
                  }}
                >
                  {isCollapsed ? "+" : "-"}
                </button>
                <button
                  type="button"
                  className="object-search-cluster-select-button"
                  onClick={() => {
                    props.onSelectIdx(locIndex);
                  }}
                >
                  <span className="object-search-cluster-label">
                    {localizationScoreSymbol(loc, props.matchThreshold)} Cluster {locIndex + 1}
                  </span>
                  <span className="object-search-cluster-score">
                    Match {formatNumber(loc.matchScore)}
                  </span>
                  <span className="object-search-cluster-count">
                    {loc.observations.length || loc.observationCount} detections
                  </span>
                </button>
                {props.reviews ? (
                  <ReviewButtons
                    label={`Cluster ${locIndex + 1}`}
                    status={clusterReviewStatus}
                    onChange={(status) => props.reviews?.setClusterStatus(loc, status)}
                  />
                ) : null}
              </div>
              <div className="object-search-cluster-meta">
                <span>sim {formatNumber(loc.similarityScore)}</span>
                <span>conf {formatNumber(loc.confidence)}</span>
                {loc.level !== null ? <span>level {loc.level}</span> : null}
                <span>
                  {formatNumber(loc.lat)}, {formatNumber(loc.lng)}
                </span>
              </div>
              {isCollapsed ? null : loc.observations.length ? (
                <div className="object-search-cluster-detections">
                  {loc.observations.map((obs, obsIndex) => {
                    const isSelectedDetection =
                      isSelectedCluster && obsIndex === props.selectedObsIdx;
                    const reviewStatus = props.reviews?.observationStatus(obs) ?? null;
                    return (
                      <div
                        key={`${locIndex}-${obs.objectIdx}-${obsIndex}`}
                        className={`object-search-detection-card${
                          isSelectedDetection ? " is-selected" : ""
                        }${reviewStatus ? ` is-review-${reviewStatus}` : ""}`}
                      >
                        <button
                          type="button"
                          className="object-search-detection-select-button"
                          onClick={() => props.onSelectObservation(locIndex, obsIndex)}
                        >
                          <span className="object-search-detection-image-wrap">
                            <img
                              src={cutoutPreviewUrl(props.mapId, obs.thumbnail)}
                              alt={`Detection ${obsIndex + 1} in cluster ${locIndex + 1}`}
                              loading="lazy"
                            />
                            <span className="object-search-detection-index">
                              {obsIndex + 1}
                            </span>
                            <span className="object-search-detection-score">
                              {formatNumber(obs.similarityScore)}
                            </span>
                          </span>
                          <span className="object-search-detection-meta">
                            <strong>Keyframe {obs.keyframeId}</strong>
                            <small>cutout {obs.cutoutId}</small>
                          </span>
                        </button>
                        {props.reviews ? (
                          <ReviewButtons
                            label={`Detection ${obsIndex + 1} in cluster ${locIndex + 1}`}
                            status={reviewStatus}
                            onChange={(status) =>
                              props.reviews?.setObservationStatus(obs, status)
                            }
                          />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="muted">No detections returned for this cluster.</p>
              )}
            </article>
          );
        })}
      </div>
    </>
  );
}

function SelectedObservationPanel(props: {
  mapId: string;
  localization: ObjectLocalization | null;
  localizationIndex: number;
  observation: ObjectObservation | null;
  observationIndex: number;
}) {
  if (!props.localization) {
    return (
      <section className="object-search-selected-observation">
        <h3>Selected cutout</h3>
        <p className="muted">Select a cluster to inspect its detections.</p>
      </section>
    );
  }
  if (!props.observation) {
    return (
      <section className="object-search-selected-observation">
        <h3>Selected cutout</h3>
        <p className="muted">No detection is available for this cluster.</p>
      </section>
    );
  }

  const previewUrl = cutoutPreviewUrl(props.mapId, props.observation.thumbnail);

  return (
    <section className="object-search-selected-observation">
      <div className="object-search-selected-observation-header">
        <h3>Selected cutout</h3>
        <span>
          Cluster {props.localizationIndex + 1} · Detection {props.observationIndex + 1}
        </span>
      </div>
      <div className="object-search-selected-observation-body">
        <img
          className="object-search-selected-observation-image"
          src={previewUrl}
          alt={`Selected cutout ${props.observation.cutoutId}`}
        />
        <div className="object-search-selected-observation-meta">
          <div>
            <strong>Keyframe</strong>
            <span>{props.observation.keyframeId}</span>
          </div>
          <div>
            <strong>Cutout</strong>
            <span>{props.observation.cutoutId}</span>
          </div>
          <div>
            <strong>Similarity</strong>
            <span>{formatNumber(props.observation.similarityScore)}</span>
          </div>
          <div>
            <strong>Bbox</strong>
            <span>{props.observation.bbox.map((value) => Math.round(value)).join(", ")}</span>
          </div>
          {props.observation.heading !== null ? (
            <div>
              <strong>Heading</strong>
              <span>{formatNumber(props.observation.heading)} deg</span>
            </div>
          ) : null}
          {props.observation.quaternion !== null ? (
            <div>
              <strong>Quaternion</strong>
              <span>{props.observation.quaternion.map(formatNumber).join(", ")}</span>
            </div>
          ) : null}
          {props.observation.coordinates ? (
            <div>
              <strong>Coordinates</strong>
              <span>
                {formatNumber(props.observation.coordinates[0])},{" "}
                {formatNumber(props.observation.coordinates[1])}
              </span>
            </div>
          ) : null}
          <div>
            <strong>Cluster position</strong>
            <span>
              {formatNumber(props.localization.lat)}, {formatNumber(props.localization.lng)}
            </span>
          </div>
          {props.localization.level !== null ? (
            <div>
              <strong>Cluster level</strong>
              <span>{props.localization.level}</span>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default ObjectSearchPanel;
