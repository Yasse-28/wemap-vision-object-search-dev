import type { CSSProperties, FormEvent, PointerEvent as ReactPointerEvent } from "react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { NavigationCandidate } from "../object-search-explorer/EquirectPhotoSphereViewer";
import { ReviewSummaryBar } from "../object-search-review/ReviewSummaryBar";
import { useObjectSearchReviews } from "../object-search-review/useObjectSearchReviews";

import type { MapSummary } from "../api";
import LivemapAnnotation, { type LivemapCone } from "../annotations/LivemapAnnotation";
import type { FocusTarget, LivemapMarker, LivemapSegment } from "../annotations/types";
import {
  BenchmarkApiError,
  fetchBenchmarkRun,
  fetchBenchmarkStatus,
  scorePrompt,
} from "../benchmark/api";
import type { ByPromptRow, PromptScore } from "../benchmark/types";
import { DEFAULT_MATCH_ACCURACY_M } from "../benchmark/types";
import { projectKeyframeToLocalFloor } from "../geo";
import {
  fetchKeyframeGraph,
  fetchMetadataMarkers,
  indexKeyframeEquirectPreviewUrl,
} from "../index-explorer/api";
import type {
  KeyframeGraphResponse,
  KeyframeMarker,
} from "../index-explorer/types";
import {
  KEYFRAME_GRAPH_COLOR,
  KEYFRAME_MARKER_COLOR,
  KEYFRAME_MARKER_RADIUS,
  KEYFRAME_MARKER_SELECTED_COLOR,
  KEYFRAME_MARKER_SELECTED_RADIUS,
  VIEW_CONE_COLOR,
  VIEW_CONE_HALF_ANGLE_DEG,
} from "../theme";
import BboxDrawOverlay from "./BboxDrawOverlay";
import CollapsibleOnlineOverrides from "./CollapsibleOnlineOverrides";
import LocalizeInspector from "./LocalizeInspector";
import OverviewSection from "./OverviewSection";
import SelectedObservationPanel from "./SelectedObservationPanel";
import TextInspector from "./TextInspector";
import { runObjectSearch } from "./api";
import type {
  EnrichedResult,
  KeyframeGroup,
  ObjectLocalization,
  ObjectSearchRunResult,
  OnlineLocalizeOverrides,
  QueryInputMode,
} from "./types";
import { DEFAULT_ONLINE_OVERRIDES, ObjectSearchMode } from "./types";
import { normalizeBenchmarkPrompt } from "./promptScoreFormat";
import {
  buildKeyframeGroups,
  cutoutPreviewUrl,
  localizationColorForMatch,
  localizationMatchThreshold,
  markerColorForScore,
} from "./utils";
import "./object-search-vision.css";

const EquirectPhotoSphereViewer = lazy(
  () => import("../object-search-explorer/EquirectPhotoSphereViewer"),
);

// ── Constants ────────────────────────────────────────────────────────────────

const COL_MIN = 25;
const COL_MAX = 75;
const ROW_MIN = 20;
const ROW_MAX = 80;
const EMPTY_LOCALIZATIONS: ObjectLocalization[] = [];

// ── Types ────────────────────────────────────────────────────────────────────

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
};

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
  const [promptScore, setPromptScore] = useState<PromptScore | null>(null);
  const [isPromptScoring, setIsPromptScoring] = useState(false);
  const [promptScoreMissing, setPromptScoreMissing] = useState(false);
  const [promptScoreError, setPromptScoreError] = useState<string | null>(null);
  const [isReviewOpen, setIsReviewOpen] = useState(false);
  const [promptBaseline, setPromptBaseline] = useState<{
    runId: string;
    row: ByPromptRow;
    config: Record<string, unknown>;
  } | null>(null);

  const [selectedKeyframeId, setSelectedKeyframeId] = useState<string | null>(null);
  const [selectedCutoutId, setSelectedCutoutId] = useState<string | null>(null);
  const [selectedLocIdx, setSelectedLocIdx] = useState(0);
  const [selectedObsIdx, setSelectedObsIdx] = useState(0);
  const [clusterFocus, setClusterFocus] = useState<
    (FocusTarget & { requestId: number }) | null
  >(null);

  const imageInputRef = useRef<HTMLInputElement>(null);

  // ── New: keyframe / map state ──────────────────────────────────────────────
  const [keyframeMarkers, setKeyframeMarkers] = useState<KeyframeMarker[] | null>(null);
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
  const markerRequestRef = useRef<{
    mapId: string;
    promise: Promise<KeyframeMarker[]>;
  } | null>(null);

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
    setKeyframeMarkers(null);
    markerRequestRef.current = null;
  }, [props.mapId]);

  useEffect(() => {
    const needsMarkerTable =
      showKeyframeMarkers || activeKeyframeId !== null || result !== null;
    if (!props.isMapKnown || !needsMarkerTable || keyframeMarkers !== null) return;
    // A single coordinate or heading lookup deliberately fills the same cached table
    // used by the overlay, avoiding a second single-marker API shape.
    let cancelled = false;
    const request =
      markerRequestRef.current?.mapId === props.mapId
        ? markerRequestRef.current.promise
        : fetchMetadataMarkers(props.mapId);
    markerRequestRef.current = { mapId: props.mapId, promise: request };
    request
      .then((payload) => {
        if (!cancelled) setKeyframeMarkers(payload);
      })
      .catch(() => {
        if (markerRequestRef.current?.promise === request) {
          markerRequestRef.current = null;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    props.mapId,
    props.isMapKnown,
    showKeyframeMarkers,
    activeKeyframeId,
    result,
    keyframeMarkers,
  ]);

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
  const localizationDisplayThreshold = localizationMatchThreshold(localizeSensitivity);
  const localizations = useMemo(
    () =>
      allLocalizations.filter(
        (loc) => loc.matchScore >= localizationDisplayThreshold,
      ),
    [allLocalizations, localizationDisplayThreshold],
  );
  // The *displayed* localizations, not `allLocalizations`: this feeds the hook's
  // `inResults` flag, whose whole job is to say whether an annotation is still
  // reachable on screen. A cluster the sensitivity slider filters out is not.
  const reviews = useObjectSearchReviews({
    enabled: true,
    mapId: props.mapId,
    query: result?.mode === "localize-online" ? resultQuery : "",
    localizations,
  });

  // Keyed on every parameter the score is measured with, not just the query: tuning
  // them between two scores is the whole workflow, and left keyed on the query alone
  // the stale numbers would sit there looking current.
  useEffect(() => {
    setPromptScore(null);
    setPromptScoreMissing(false);
    setPromptScoreError(null);
  }, [
    props.mapId,
    resultQuery,
    onlineOverrides.feedback_alpha,
    onlineOverrides.feedback_beta,
    onlineOverrides.feedback_normalization,
    onlineOverrides.min_similarity,
    onlineOverrides.merge_radius,
    onlineOverrides.candidate_count,
    onlineOverrides.min_keyframes_per_cluster,
    localizeSensitivity,
    numResults,
    maxObservationsPerCluster,
  ]);

  useEffect(() => {
    let cancelled = false;
    setPromptBaseline(null);
    if (!isReviewOpen || result?.mode !== "localize-online" || !resultQuery) {
      return () => {
        cancelled = true;
      };
    }
    const normalizedQuery = normalizeBenchmarkPrompt(resultQuery);
    void fetchBenchmarkStatus(props.mapId)
      .then(async (status) => {
        const newestRun = status.runs[0];
        if (!newestRun) return null;
        const metrics = await fetchBenchmarkRun(props.mapId, newestRun.run_id);
        const row = metrics.by_prompt.find(
          (candidate) =>
            !candidate.error
            && normalizeBenchmarkPrompt(candidate.prompt) === normalizedQuery,
        );
        return row
          ? { runId: newestRun.run_id, row, config: metrics.config }
          : null;
      })
      .then((baseline) => {
        if (!cancelled) setPromptBaseline(baseline);
      })
      .catch(() => {
        // A baseline is supplementary; scoring the current prompt remains available.
      });
    return () => {
      cancelled = true;
    };
  }, [isReviewOpen, props.mapId, result?.mode, resultQuery]);

  async function handleScorePrompt(): Promise<void> {
    if (!resultQuery || isPromptScoring) return;
    setIsPromptScoring(true);
    setPromptScore(null);
    setPromptScoreMissing(false);
    setPromptScoreError(null);
    try {
      const score = await scorePrompt(props.mapId, resultQuery, {
        num_results: numResults,
        clustering_eps_m: onlineOverrides.merge_radius,
        candidate_count: onlineOverrides.candidate_count,
        feedback_alpha: onlineOverrides.feedback_alpha,
        feedback_beta: onlineOverrides.feedback_beta,
        feedback_normalization: onlineOverrides.feedback_normalization,
        min_keyframes_per_cluster: onlineOverrides.min_keyframes_per_cluster,
        max_observations_per_cluster: maxObservationsPerCluster,
        // The script defaults to 0.15, the service to 0.2. Left unsent, the metrics
        // are computed on a different candidate set from the list on screen — the
        // floor decides which clusters exist, and therefore which one is the best.
        min_similarity: onlineOverrides.min_similarity,
        // Scoring parameters, not localization ones — they decide what the metrics
        // count. The group radius tracks the clustering radius (the pipeline cannot
        // emit two clusters closer than that, so a finer ground truth caps recall),
        // and the match radius must stay under half the spacing between neighbouring
        // annotations of one class. Same values as the Benchmark tab's defaults.
        group_annotation_radius_m: onlineOverrides.merge_radius,
        default_accuracy: DEFAULT_MATCH_ACCURACY_M,
        // Score the clusters the user is actually looking at. The benchmark keeps
        // predictions scoring *strictly above* its acceptance threshold, while the list
        // is filtered on `>= localizationDisplayThreshold`; the epsilon makes the two
        // sets identical instead of off by the clusters sitting exactly on the bar.
        // Without this the score ignored the Sensitivity slider entirely and always
        // measured the script's own default of 0.9.
        acceptance_threshold: Math.max(0, localizationDisplayThreshold - 1e-9),
      });
      setPromptScore(score);
    } catch (scoreError) {
      if (scoreError instanceof BenchmarkApiError && scoreError.status === 404) {
        setPromptScoreMissing(true);
      } else {
        setPromptScoreError(
          scoreError instanceof Error ? scoreError.message : String(scoreError),
        );
      }
    } finally {
      setIsPromptScoring(false);
    }
  }
  const keyframeCoords = result && result.mode !== "text" ? result.keyframes : {};
  const keyframeMapPositions = useMemo(() => {
    const positions = { ...keyframeCoords };
    for (const marker of keyframeMarkers ?? []) {
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
  }, [keyframeMarkers, keyframeCoords]);
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
    keyframeMarkers?.find((m) => m.id === activeKeyframeId) ?? null;
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
    if (showKeyframeMarkers && keyframeMarkers) {
      for (const m of keyframeMarkers) {
        const isActive = m.id === activeKeyframeId;
        markers.push({
          id: `kf_${m.id}`,
          latitude: m.latitude,
          longitude: m.longitude,
          level: m.level,
          color: isActive ? KEYFRAME_MARKER_SELECTED_COLOR : KEYFRAME_MARKER_COLOR,
          radius: isActive ? KEYFRAME_MARKER_SELECTED_RADIUS : KEYFRAME_MARKER_RADIUS,
          scaleWithZoom: true,
        });
      }
    }
    for (const m of resultMarkers) {
      markers.push(m);
    }
    return markers;
  }, [keyframeMarkers, showKeyframeMarkers, activeKeyframeId, resultMarkers]);

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
    () => keyframeMarkers?.map((m) => m.id) ?? [],
    [keyframeMarkers],
  );

  const activeKeyframeIndex = activeKeyframeId ? keyframeIds.indexOf(activeKeyframeId) : -1;
  const hasPreviousKeyframe = activeKeyframeIndex > 0;
  const hasNextKeyframe = activeKeyframeIndex >= 0 && activeKeyframeIndex < keyframeIds.length - 1;

  const photosphereNavigationCandidates = useMemo<NavigationCandidate[]>(() => {
    if (!activeKeyframeId || !activeKeyframeMarker || activeKeyframeMarker.heading_deg == null || !keyframeMarkers) {
      return [];
    }
    return keyframeMarkers.flatMap((marker) => {
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
  }, [activeKeyframeId, activeKeyframeMarker, keyframeMarkers]);

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
      const headingDeg = keyframeMarkers?.find((marker) => marker.id === kfId)?.heading_deg;
      if (headingDeg !== null && headingDeg !== undefined) {
        preservedCompassBearingDegRef.current =
          ((headingDeg + (yawRad * 180) / Math.PI) % 360 + 360) % 360;
      }
    },
    [activeKeyframeId, keyframeMarkers],
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
              Object Search
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
                    reviews={reviews}
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
            {result?.mode === "localize-online"
            || reviews.annotations.length > 0 ? (
              <ReviewSummaryBar
                reviews={reviews}
                promptScore={promptScore}
                promptBaseline={promptBaseline}
                promptScoreError={promptScoreError}
                promptScoreMissing={promptScoreMissing}
                isPromptScoring={isPromptScoring}
                resultQuery={resultQuery}
                onScorePrompt={handleScorePrompt}
                open={isReviewOpen}
                onToggle={() => setIsReviewOpen((current) => !current)}
              />
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}

export default ObjectSearchPanel;
