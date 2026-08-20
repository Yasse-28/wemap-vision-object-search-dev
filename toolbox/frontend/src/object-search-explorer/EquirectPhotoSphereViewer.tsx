import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

import type { MetadataRowRecord } from "../index-explorer/types";

const FOV_DEFAULT = 60;
/**
 * How far the panorama zooms, as a vertical field of view in degrees.
 *
 * The floor is what makes a small distant object judgeable — annotating a sign at
 * 15 m needs more than a 30° view. The ceiling stops at 130°: a perspective camera
 * stretches the edges badly past that, and beyond ~150° the projection inverts.
 */
const FOV_MIN = 12;
const FOV_MAX = 130;
/** One press of the zoom buttons, as a ratio — about six presses end to end. */
const ZOOM_BUTTON_FACTOR = 1.45;
const SPHERE_RADIUS = 500;
const OVERLAY_RADIUS = 495;
/** Screen-space width of a drawn outline, in pixels. */
const REGION_LINE_WIDTH_PX = 3;
const REGION_DRAFT_COLOR = 0x2563eb;
const CLICK_DRAG_THRESHOLD_PX = 5;

const EDGE_STEPS = 8;
const FLOOR_CAMERA_HEIGHT_M = 1.9;
const FLOOR_CURSOR_RADIUS_M = 0.12;
const FLOOR_SNAP_INNER_RADIUS_M = 0.14;
const FLOOR_SNAP_OUTER_RADIUS_M = 0.24;

export type NavigationCandidate = {
  id: string;
  localX: number;
  localZ: number;
};

type DepthPinMarker = {
  xRatio: number;
  yRatio: number;
  status: "resolving" | "resolved" | "error";
};

type CursorPoint = {
  x: number;
  y: number;
};

/** A hovered detection box, with the cursor position that hit it (container-relative). */
export type HoveredDetection = {
  rowIndex: number;
  x: number;
  y: number;
};

/** A box in texture-ratio space the annotator can drag and resize, e.g. a reprojected suggestion. */
export type EditableBox = {
  u0: number;
  v0: number;
  u1: number;
  v1: number;
};

type EditableBoxCorner = "nw" | "ne" | "se" | "sw";

type BoxDragState =
  | { mode: "corner"; corner: EditableBoxCorner; startBox: EditableBox }
  | { mode: "move"; startBox: EditableBox; startRatio: { xRatio: number; yRatio: number } };

type Props = {
  keyframeId: string;
  imageSrc: string;
  height: number;
  initialYawRad: number;
  initialTextureYRatio: number;
  // Imperative re-orientation: when `orientToken` changes, the camera yaw snaps
  // to `orientYawRad` (used to face a localization without changing keyframe).
  orientYawRad?: number;
  orientToken?: number;
  detections: MetadataRowRecord[];
  selectedRowIndex: number | null;
  /** Rows that already carry a saved ground-truth annotation, styled apart from the rest. */
  annotatedRowIndices?: ReadonlySet<number>;
  /**
   * Saved annotations with no detector row behind them for this keyframe — a
   * missed detection traced by hand, or a reprojected suggestion once confirmed
   * — drawn from the outline stored on the annotation, styled the same as an
   * already-annotated proposal.
   */
  annotationOutlines?: Array<{ id: string; region: Array<[number, number]> }>;
  /** A reprojected suggestion the annotator can drag and resize before saving it. */
  editableBox?: EditableBox | null;
  onEditableBoxChange?: (box: EditableBox) => void;
  depthPin: DepthPinMarker | null;
  polygonForDetection: (item: MetadataRowRecord) => Array<[number, number]>;
  onDepthPin: (xRatio: number, yRatio: number) => void;
  allowDepthPinOnMarker?: boolean;
  navigationCandidates?: NavigationCandidate[];
  onNavigate?: (keyframeId: string) => void;
  /**
   * Fires as the cursor enters and leaves a detection box. Hit-testing happens in
   * texture-ratio space rather than by raycasting the overlay: the boxes are line
   * loops, so a raycast only ever hits their border.
   */
  onHoverDetection?: (hovered: HoveredDetection | null) => void;
  onViewChange?: (keyframeId: string, yawRad: number, textureYRatio: number) => void;
  /**
   * Region drawing, for an object no detector proposed. While it is on, a plain
   * click reports a vertex in texture-ratio space instead of navigating a floor;
   * Ctrl+click still places a depth pin and dragging still looks around.
   */
  regionDrawActive?: boolean;
  onRegionPoint?: (uRatio: number, vRatio: number) => void;
  /** The vertices drawn so far, in texture ratios, painted over the image. */
  draftRegion?: Array<[number, number]> | null;
};

type Runtime = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  material: THREE.MeshBasicMaterial;
  geometry: THREE.SphereGeometry;
  overlayGroup: THREE.Group;
  /** The rubber-band segment while a region is being traced, rebuilt on move. */
  liveDraftGroup: THREE.Group;
  /** The reprojected suggestion box and its corner handles, rebuilt on every edit. */
  editableBoxGroup: THREE.Group;
  /** One shared Vector2: `LineMaterial` reads it, so a resize is a single `set`. */
  resolution: THREE.Vector2;
  raycaster: THREE.Raycaster;
  pointer: THREE.Vector2;
  theta: number;
  phi: number;
  fov: number;
  dirty: boolean;
  texture: THREE.Texture | null;
  imageWidth: number;
  imageHeight: number;
  animationFrame: number;
  viewUpdateRaf: number | null;
  pendingView: { yaw: number; textureYRatio: number } | null;
  lastReportedYaw: number | null;
  lastReportedTextureYRatio: number | null;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampRatio(value: number): number {
  return clamp(value, 0, 1);
}

function normalizeYaw(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function textureRatioToDirection(
  xRatio: number,
  yRatio: number,
  radius = OVERLAY_RADIUS,
): THREE.Vector3 {
  const yaw = (clampRatio(xRatio) - 0.5) * 2 * Math.PI;
  const pitch = (0.5 - clampRatio(yRatio)) * Math.PI;
  const cosPitch = Math.cos(pitch);
  return new THREE.Vector3(
    Math.sin(yaw) * cosPitch * radius,
    Math.sin(pitch) * radius,
    -Math.cos(yaw) * cosPitch * radius,
  );
}

function directionToTextureRatio(direction: THREE.Vector3): {
  xRatio: number;
  yRatio: number;
} {
  const normalized = direction.clone().normalize();
  const yaw = Math.atan2(normalized.x, -normalized.z);
  const pitch = Math.asin(clamp(normalized.y, -1, 1));
  return {
    xRatio: ((0.5 + yaw / (2 * Math.PI)) % 1 + 1) % 1,
    yRatio: clampRatio(0.5 - pitch / Math.PI),
  };
}

function interpolatedEdgePoints(
  from: [number, number],
  to: [number, number],
): THREE.Vector3[] {
  const fromYaw = (from[0] - 0.5) * 2 * Math.PI;
  const toYaw = (to[0] - 0.5) * 2 * Math.PI;
  const yawDelta = normalizeYaw(toYaw - fromYaw);
  return Array.from({ length: EDGE_STEPS + 1 }, (_, index) => {
    const ratio = index / EDGE_STEPS;
    const yaw = fromYaw + yawDelta * ratio;
    const pitch =
      (0.5 - (from[1] + (to[1] - from[1]) * ratio)) * Math.PI;
    const cosPitch = Math.cos(pitch);
    return new THREE.Vector3(
      Math.sin(yaw) * cosPitch * OVERLAY_RADIUS,
      Math.sin(pitch) * OVERLAY_RADIUS,
      -Math.cos(yaw) * cosPitch * OVERLAY_RADIUS,
    );
  });
}

/**
 * A polyline of real, constant pixel width.
 *
 * `LineBasicMaterial.linewidth` is ignored by nearly every WebGL implementation, so
 * an outline drawn with it is one pixel wide however thick it asks to be — unusable
 * over a photograph. `Line2` carries its own screen-space width instead, which is
 * also why it needs the viewport resolution and has to be told about a resize.
 */
function makeThickLine(
  points: THREE.Vector3[],
  color: number,
  resolution: THREE.Vector2,
  closed: boolean,
): Line2 {
  const ring = closed && points.length ? [...points, points[0]] : points;
  const geometry = new LineGeometry();
  geometry.setPositions(ring.flatMap((point) => [point.x, point.y, point.z]));
  const material = new LineMaterial({
    color,
    linewidth: REGION_LINE_WIDTH_PX,
    worldUnits: false,
    depthTest: false,
    transparent: true,
    opacity: 0.95,
    resolution,
  });
  return new Line2(geometry, material);
}

/** The world path along a region's edges; `closed` adds the edge back to the start. */
function regionWorldPoints(
  ratios: Array<[number, number]>,
  closed: boolean,
): THREE.Vector3[] {
  const points: THREE.Vector3[] = [];
  const edges = closed ? ratios.length : ratios.length - 1;
  for (let index = 0; index < edges; index += 1) {
    const segment = interpolatedEdgePoints(ratios[index], ratios[(index + 1) % ratios.length]);
    points.push(...(index === 0 ? segment : segment.slice(1)));
  }
  return points;
}

function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (child instanceof THREE.Line || child instanceof THREE.Mesh) {
      child.geometry.dispose();
      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      for (const material of materials) {
        material.dispose();
      }
    }
  });
}

/** Orange: a suggestion awaiting the annotator's confirmation, not yet a record. */
const EDITABLE_BOX_COLOR = 0xf97316;
const EDITABLE_BOX_HANDLE_RADIUS = 8;
const EDITABLE_BOX_CORNERS: readonly EditableBoxCorner[] = ["nw", "ne", "se", "sw"];

/** Rebuilds the suggestion outline and its four drag handles from scratch. */
function rebuildEditableBoxOverlay(runtime: Runtime, box: EditableBox | null): void {
  const group = runtime.editableBoxGroup;
  for (const child of group.children.slice()) {
    group.remove(child);
    disposeObject(child);
  }
  if (!box) {
    return;
  }
  const corners: Array<[number, number]> = [
    [box.u0, box.v0],
    [box.u1, box.v0],
    [box.u1, box.v1],
    [box.u0, box.v1],
  ];
  const points: THREE.Vector3[] = [];
  for (let index = 0; index < 4; index += 1) {
    points.push(
      ...interpolatedEdgePoints(corners[index], corners[(index + 1) % 4]).slice(
        index === 0 ? 0 : 1,
      ),
    );
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: EDITABLE_BOX_COLOR,
    depthTest: false,
    transparent: true,
    opacity: 1,
  });
  const line = new THREE.LineLoop(geometry, material);
  line.userData.markerType = "editable-box";
  line.renderOrder = 25;
  group.add(line);

  EDITABLE_BOX_CORNERS.forEach((corner, index) => {
    const [u, v] = corners[index];
    const handle = new THREE.Mesh(
      new THREE.SphereGeometry(EDITABLE_BOX_HANDLE_RADIUS, 12, 10),
      new THREE.MeshBasicMaterial({ color: EDITABLE_BOX_COLOR, depthTest: false }),
    );
    handle.position.copy(textureRatioToDirection(u, v, OVERLAY_RADIUS - 2));
    handle.userData.markerType = "editable-box-handle";
    handle.userData.corner = corner;
    handle.renderOrder = 26;
    group.add(handle);
  });
}

export default function EquirectPhotoSphereViewer(props: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runtimeRef = useRef<Runtime | null>(null);
  const keyframeIdRef = useRef(props.keyframeId);
  const onDepthPinRef = useRef(props.onDepthPin);
  const onViewChangeRef = useRef(props.onViewChange);
  const allowDepthPinOnMarkerRef = useRef(props.allowDepthPinOnMarker ?? false);
  const navigationCandidatesRef = useRef(props.navigationCandidates ?? []);
  const onNavigateRef = useRef(props.onNavigate);
  const onHoverDetectionRef = useRef(props.onHoverDetection);
  // Boxes in texture-ratio space, smallest first, so the innermost box wins a hit.
  const hoverTargetsRef = useRef<
    Array<{ rowIndex: number; u0: number; v0: number; u1: number; v1: number }>
  >([]);
  const hoveredRowIndexRef = useRef<number | null>(null);
  /** The drawn box per row, so hovering restyles two lines instead of rebuilding. */
  const bboxLinesRef = useRef(new Map<number, THREE.LineLoop>());
  const selectedRowIndexRef = useRef(props.selectedRowIndex);
  const annotatedRowIndicesRef = useRef(props.annotatedRowIndices);
  /**
   * The suggestion box's live value while it is being dragged. Only synced from
   * `props.editableBox` in a dedicated effect (not on every render) so an unrelated
   * re-render mid-drag cannot snap it back to the pre-drag prop.
   */
  const editableBoxRef = useRef<EditableBox | null>(props.editableBox ?? null);
  const onEditableBoxChangeRef = useRef(props.onEditableBoxChange);
  const regionDrawActiveRef = useRef(props.regionDrawActive ?? false);
  const onRegionPointRef = useRef(props.onRegionPoint);
  const draftRegionRef = useRef<Array<[number, number]>>(props.draftRegion ?? []);
  const initialYawRadRef = useRef(props.initialYawRad);
  const initialTextureYRatioRef = useRef(props.initialTextureYRatio);
  const textureKeyframeIdRef = useRef<string | null>(null);
  const [loadStatus, setLoadStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [isDepthProjectionActive, setIsDepthProjectionActive] = useState(false);
  const [depthProjectionCursor, setDepthProjectionCursor] =
    useState<CursorPoint | null>(null);

  keyframeIdRef.current = props.keyframeId;
  onDepthPinRef.current = props.onDepthPin;
  onViewChangeRef.current = props.onViewChange;
  allowDepthPinOnMarkerRef.current = props.allowDepthPinOnMarker ?? false;
  navigationCandidatesRef.current = props.navigationCandidates ?? [];
  onNavigateRef.current = props.onNavigate;
  onHoverDetectionRef.current = props.onHoverDetection;
  selectedRowIndexRef.current = props.selectedRowIndex;
  annotatedRowIndicesRef.current = props.annotatedRowIndices;
  onEditableBoxChangeRef.current = props.onEditableBoxChange;
  regionDrawActiveRef.current = props.regionDrawActive ?? false;
  onRegionPointRef.current = props.onRegionPoint;
  draftRegionRef.current = props.draftRegion ?? [];
  initialYawRadRef.current = props.initialYawRad;
  initialTextureYRatioRef.current = props.initialTextureYRatio;

  /**
   * How a box reads: teal normally, violet when a ground-truth annotation is already
   * saved for it, red when it is the one being worked on, and blue — the annotation
   * accent — while the cursor is over it. `linewidth` is ignored by most WebGL
   * implementations, so the highlight has to be carried by colour and opacity rather
   * than by a thicker stroke.
   */
  const styleBboxLine = (
    line: THREE.LineLoop,
    selected: boolean,
    hovered: boolean,
    annotated: boolean,
  ) => {
    const material = line.material as THREE.LineBasicMaterial;
    material.color.setHex(
      hovered ? 0x2563eb : selected ? 0xff6b6b : annotated ? 0x8b5cf6 : 0x11b5ae,
    );
    material.opacity = hovered || selected ? 1 : annotated ? 0.95 : 0.9;
    material.needsUpdate = true;
    line.renderOrder = hovered ? 30 : selected ? 20 : annotated ? 15 : 10;
  };

  const reportView = (runtime: Runtime, immediate = false) => {
    runtime.pendingView = {
      yaw: runtime.theta,
      textureYRatio: clampRatio(0.5 - runtime.phi / Math.PI),
    };
    const flush = () => {
      runtime.viewUpdateRaf = null;
      const pending = runtime.pendingView;
      if (!pending) {
        return;
      }
      runtime.pendingView = null;
      runtime.lastReportedYaw = pending.yaw;
      runtime.lastReportedTextureYRatio = pending.textureYRatio;
      onViewChangeRef.current?.(
        keyframeIdRef.current,
        pending.yaw,
        pending.textureYRatio,
      );
    };
    if (immediate) {
      if (runtime.viewUpdateRaf !== null) {
        cancelAnimationFrame(runtime.viewUpdateRaf);
        runtime.viewUpdateRaf = null;
      }
      flush();
    } else if (runtime.viewUpdateRaf === null) {
      runtime.viewUpdateRaf = requestAnimationFrame(flush);
    }
  };

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV_DEFAULT, 16 / 9, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    const geometry = new THREE.SphereGeometry(SPHERE_RADIUS, 60, 40);
    const material = new THREE.MeshBasicMaterial({
      side: THREE.BackSide,
      color: 0x111827,
    });
    scene.add(new THREE.Mesh(geometry, material));
    const overlayGroup = new THREE.Group();
    scene.add(overlayGroup);
    const liveDraftGroup = new THREE.Group();
    scene.add(liveDraftGroup);
    const editableBoxGroup = new THREE.Group();
    scene.add(editableBoxGroup);
    const floorCursor = new THREE.Mesh(
      new THREE.CircleGeometry(FLOOR_CURSOR_RADIUS_M, 32),
      new THREE.MeshBasicMaterial({
        color: 0x4dc3ff,
        transparent: true,
        opacity: 0.6,
        side: THREE.DoubleSide,
        depthTest: false,
      }),
    );
    floorCursor.rotation.x = -Math.PI / 2;
    floorCursor.position.y = -FLOOR_CAMERA_HEIGHT_M;
    floorCursor.renderOrder = 40;
    floorCursor.visible = false;
    scene.add(floorCursor);
    const floorSnap = new THREE.Mesh(
      new THREE.RingGeometry(
        FLOOR_SNAP_INNER_RADIUS_M,
        FLOOR_SNAP_OUTER_RADIUS_M,
        32,
      ),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide,
        depthTest: false,
      }),
    );
    floorSnap.rotation.x = -Math.PI / 2;
    floorSnap.position.y = -FLOOR_CAMERA_HEIGHT_M + 0.005;
    floorSnap.renderOrder = 41;
    floorSnap.visible = false;
    scene.add(floorSnap);
    const floorPlane = new THREE.Plane(
      new THREE.Vector3(0, 1, 0),
      FLOOR_CAMERA_HEIGHT_M,
    );
    const floorHit = new THREE.Vector3();

    const runtime: Runtime = {
      renderer,
      scene,
      camera,
      material,
      geometry,
      overlayGroup,
      liveDraftGroup,
      editableBoxGroup,
      resolution: new THREE.Vector2(1, 1),
      raycaster: new THREE.Raycaster(),
      pointer: new THREE.Vector2(),
      theta: normalizeYaw(initialYawRadRef.current),
      phi: clamp(
        (0.5 - initialTextureYRatioRef.current) * Math.PI,
        -Math.PI / 2 + 0.01,
        Math.PI / 2 - 0.01,
      ),
      fov: FOV_DEFAULT,
      dirty: true,
      texture: null,
      imageWidth: 0,
      imageHeight: 0,
      animationFrame: 0,
      viewUpdateRaf: null,
      pendingView: null,
      lastReportedYaw: null,
      lastReportedTextureYRatio: null,
    };
    runtime.raycaster.params.Line = { threshold: 3 };
    runtimeRef.current = runtime;
    rebuildEditableBoxOverlay(runtime, editableBoxRef.current);

    const resize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      renderer.setSize(width, height, false);
      runtime.resolution.set(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      runtime.dirty = true;
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let isDragging = false;
    let isTouchActive = false;
    /** Set while a suggestion box handle or body is being dragged; blocks look-around. */
    let boxDrag: BoxDragState | null = null;
    let previousX = 0;
    let previousY = 0;
    let downX = 0;
    let downY = 0;

    const updateDepthProjectionCursor = (event: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
        setDepthProjectionCursor({ x, y });
      } else {
        setDepthProjectionCursor(null);
      }
      setIsDepthProjectionActive(event.ctrlKey);
    };

    const clearDepthProjectionCursor = () => {
      setIsDepthProjectionActive(false);
      setDepthProjectionCursor(null);
    };

    const hideFloorNavigation = () => {
      if (!floorCursor.visible && !floorSnap.visible) {
        return;
      }
      floorCursor.visible = false;
      floorSnap.visible = false;
      runtime.dirty = true;
    };

    const updateFloorSnap = (
      clientX: number,
      clientY: number,
    ): string | null => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        hideFloorNavigation();
        return null;
      }
      runtime.pointer.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      runtime.raycaster.setFromCamera(runtime.pointer, camera);
      if (!runtime.raycaster.ray.intersectPlane(floorPlane, floorHit)) {
        hideFloorNavigation();
        return null;
      }
      floorCursor.position.set(
        floorHit.x,
        -FLOOR_CAMERA_HEIGHT_M,
        floorHit.z,
      );
      floorCursor.visible = true;

      let nearest: NavigationCandidate | null = null;
      let nearestDistanceSquared = Number.POSITIVE_INFINITY;
      for (const candidate of navigationCandidatesRef.current) {
        const dx = candidate.localX - floorHit.x;
        const dz = candidate.localZ - floorHit.z;
        const distanceSquared = dx * dx + dz * dz;
        if (distanceSquared < nearestDistanceSquared) {
          nearest = candidate;
          nearestDistanceSquared = distanceSquared;
        }
      }
      if (nearest) {
        floorSnap.position.set(
          nearest.localX,
          -FLOOR_CAMERA_HEIGHT_M + 0.005,
          nearest.localZ,
        );
        floorSnap.visible = true;
      } else {
        floorSnap.visible = false;
      }
      runtime.dirty = true;
      return nearest?.id ?? null;
    };

    const updateView = (x: number, y: number) => {
      const dx = x - previousX;
      const dy = y - previousY;
      const sensitivity = runtime.fov / FOV_DEFAULT;
      runtime.theta = normalizeYaw(
        runtime.theta -
          (dx / Math.max(1, canvas.clientWidth)) * Math.PI * sensitivity,
      );
      runtime.phi = clamp(
        runtime.phi +
          (dy / Math.max(1, canvas.clientHeight)) *
            (Math.PI / 2) *
            sensitivity,
        -Math.PI / 2 + 0.01,
        Math.PI / 2 - 0.01,
      );
      previousX = x;
      previousY = y;
      runtime.dirty = true;
      reportView(runtime);
    };

    const clickTextureRatio = (
      clientX: number,
      clientY: number,
    ): { xRatio: number; yRatio: number } | null => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return null;
      }
      runtime.pointer.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      runtime.raycaster.setFromCamera(runtime.pointer, camera);
      const markerHits = runtime.raycaster.intersectObjects(
        overlayGroup.children,
        true,
      );
      const markerType = markerHits[0]?.object.userData.markerType;
      if (
        markerType &&
        (!allowDepthPinOnMarkerRef.current || markerType === "depth-pin")
      ) {
        return null;
      }
      return directionToTextureRatio(runtime.raycaster.ray.direction);
    };

    /** Where the cursor points on the sphere, ignoring whatever marker sits under it. */
    const rayTextureRatio = (
      clientX: number,
      clientY: number,
    ): { xRatio: number; yRatio: number } | null => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return null;
      }
      runtime.pointer.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      runtime.raycaster.setFromCamera(runtime.pointer, camera);
      return directionToTextureRatio(runtime.raycaster.ray.direction);
    };

    const isInsideEditableBox = (xRatio: number, yRatio: number): boolean => {
      const box = editableBoxRef.current;
      if (!box) {
        return false;
      }
      const v0 = Math.min(box.v0, box.v1);
      const v1 = Math.max(box.v0, box.v1);
      if (yRatio < v0 || yRatio > v1) {
        return false;
      }
      const u0 = Math.min(box.u0, box.u1);
      const u1 = Math.max(box.u0, box.u1);
      return [xRatio - 1, xRatio, xRatio + 1].some((u) => u >= u0 && u <= u1);
    };

    const hitEditableBoxHandle = (
      clientX: number,
      clientY: number,
    ): EditableBoxCorner | null => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height || !runtime.editableBoxGroup.children.length) {
        return null;
      }
      runtime.pointer.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      runtime.raycaster.setFromCamera(runtime.pointer, camera);
      const hits = runtime.raycaster.intersectObjects(runtime.editableBoxGroup.children, false);
      const hit = hits.find((item) => item.object.userData.markerType === "editable-box-handle");
      return (hit?.object.userData.corner as EditableBoxCorner | undefined) ?? null;
    };

    /** Applies the cursor's current ray to whatever the drag grabbed, and redraws. */
    const updateBoxDrag = (clientX: number, clientY: number) => {
      if (!boxDrag) {
        return;
      }
      const ratio = rayTextureRatio(clientX, clientY);
      if (!ratio) {
        return;
      }
      let next: EditableBox;
      if (boxDrag.mode === "corner") {
        next = { ...boxDrag.startBox };
        if (boxDrag.corner === "nw") {
          next.u0 = ratio.xRatio;
          next.v0 = ratio.yRatio;
        } else if (boxDrag.corner === "ne") {
          next.u1 = ratio.xRatio;
          next.v0 = ratio.yRatio;
        } else if (boxDrag.corner === "se") {
          next.u1 = ratio.xRatio;
          next.v1 = ratio.yRatio;
        } else {
          next.u0 = ratio.xRatio;
          next.v1 = ratio.yRatio;
        }
      } else {
        const du = ratio.xRatio - boxDrag.startRatio.xRatio;
        const dv = ratio.yRatio - boxDrag.startRatio.yRatio;
        next = {
          u0: boxDrag.startBox.u0 + du,
          u1: boxDrag.startBox.u1 + du,
          v0: clamp(boxDrag.startBox.v0 + dv, 0, 1),
          v1: clamp(boxDrag.startBox.v1 + dv, 0, 1),
        };
      }
      editableBoxRef.current = next;
      rebuildEditableBoxOverlay(runtime, next);
      runtime.dirty = true;
    };

    const reportHover = (hovered: HoveredDetection | null) => {
      const nextRowIndex = hovered?.rowIndex ?? null;
      if (nextRowIndex === null && hoveredRowIndexRef.current === null) {
        return;
      }
      const previousRowIndex = hoveredRowIndexRef.current;
      hoveredRowIndexRef.current = nextRowIndex;
      if (previousRowIndex !== nextRowIndex) {
        const selectedRowIndex = selectedRowIndexRef.current;
        for (const rowIndex of [previousRowIndex, nextRowIndex]) {
          if (rowIndex === null) {
            continue;
          }
          const line = bboxLinesRef.current.get(rowIndex);
          if (line) {
            styleBboxLine(
              line,
              rowIndex === selectedRowIndex,
              rowIndex === nextRowIndex,
              annotatedRowIndicesRef.current?.has(rowIndex) ?? false,
            );
          }
        }
        runtime.dirty = true;
      }
      onHoverDetectionRef.current?.(hovered);
    };

    const updateHoveredDetection = (clientX: number, clientY: number) => {
      if (!onHoverDetectionRef.current || !hoverTargetsRef.current.length) {
        reportHover(null);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        reportHover(null);
        return;
      }
      runtime.pointer.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      runtime.raycaster.setFromCamera(runtime.pointer, camera);
      const { xRatio, yRatio } = directionToTextureRatio(
        runtime.raycaster.ray.direction,
      );
      for (const target of hoverTargetsRef.current) {
        if (yRatio < target.v0 || yRatio > target.v1) {
          continue;
        }
        // The stored `u` is unwrapped, so a box near the seam has to be met halfway.
        const inU = [xRatio - 1, xRatio, xRatio + 1].some(
          (u) => u >= target.u0 && u <= target.u1,
        );
        if (inU) {
          const container = containerRef.current;
          const containerRect = container?.getBoundingClientRect() ?? rect;
          reportHover({
            rowIndex: target.rowIndex,
            x: clientX - containerRect.left,
            y: clientY - containerRect.top,
          });
          return;
        }
      }
      reportHover(null);
    };

    const onMouseDown = (event: MouseEvent) => {
      if (isTouchActive) {
        return;
      }
      if (!regionDrawActiveRef.current && !event.ctrlKey && !event.metaKey) {
        const handle = hitEditableBoxHandle(event.clientX, event.clientY);
        const box = editableBoxRef.current;
        if (handle && box) {
          boxDrag = { mode: "corner", corner: handle, startBox: { ...box } };
          downX = previousX = event.clientX;
          downY = previousY = event.clientY;
          return;
        }
        if (box) {
          const ratio = rayTextureRatio(event.clientX, event.clientY);
          if (ratio && isInsideEditableBox(ratio.xRatio, ratio.yRatio)) {
            boxDrag = { mode: "move", startBox: { ...box }, startRatio: ratio };
            downX = previousX = event.clientX;
            downY = previousY = event.clientY;
            return;
          }
        }
      }
      isDragging = true;
      downX = previousX = event.clientX;
      downY = previousY = event.clientY;
    };
    const clearLiveDraft = () => {
      if (!runtime.liveDraftGroup.children.length) {
        return;
      }
      for (const child of runtime.liveDraftGroup.children.slice()) {
        runtime.liveDraftGroup.remove(child);
        disposeObject(child);
      }
      runtime.dirty = true;
    };

    /**
     * The edge from the last placed vertex to the cursor. Without it the outline only
     * appears one edge behind the pointer, and there is no way to judge where an edge
     * will land before committing it.
     */
    const updateLiveDraft = (clientX: number, clientY: number) => {
      const vertices = draftRegionRef.current;
      if (!regionDrawActiveRef.current || !vertices.length) {
        clearLiveDraft();
        return;
      }
      const ratio = clickTextureRatio(clientX, clientY);
      if (!ratio) {
        clearLiveDraft();
        return;
      }
      clearLiveDraft();
      const cursor: [number, number] = [ratio.xRatio, ratio.yRatio];
      const last = vertices[vertices.length - 1];
      const points = interpolatedEdgePoints(last, cursor);
      if (vertices.length >= 3) {
        // Show the closing edge too, so "click the first point to finish" reads as a
        // shape rather than as a line that happens to end near where it started.
        points.push(...interpolatedEdgePoints(cursor, vertices[0]).slice(1));
      }
      const line = makeThickLine(points, REGION_DRAFT_COLOR, runtime.resolution, false);
      line.material.opacity = 0.6;
      line.userData.markerType = "region-draft";
      line.renderOrder = 39;
      runtime.liveDraftGroup.add(line);
      runtime.dirty = true;
    };

    const onMouseMove = (event: MouseEvent) => {
      updateDepthProjectionCursor(event);
      updateLiveDraft(event.clientX, event.clientY);
      if (boxDrag) {
        updateBoxDrag(event.clientX, event.clientY);
        return;
      }
      if (isDragging) {
        reportHover(null);
        updateView(event.clientX, event.clientY);
        return;
      }
      updateHoveredDetection(event.clientX, event.clientY);
      if (event.ctrlKey) {
        hideFloorNavigation();
      } else {
        updateFloorSnap(event.clientX, event.clientY);
      }
    };
    const onMouseUp = (event: MouseEvent) => {
      if (boxDrag) {
        boxDrag = null;
        const box = editableBoxRef.current;
        if (box) {
          onEditableBoxChangeRef.current?.(box);
        }
        return;
      }
      if (!isDragging) {
        return;
      }
      isDragging = false;
      reportView(runtime, true);
      if (
        Math.hypot(event.clientX - downX, event.clientY - downY) <
        CLICK_DRAG_THRESHOLD_PX
      ) {
        if (event.ctrlKey) {
          const ratio = clickTextureRatio(event.clientX, event.clientY);
          if (ratio) {
            onDepthPinRef.current(ratio.xRatio, ratio.yRatio);
          }
        } else if (regionDrawActiveRef.current) {
          const ratio = clickTextureRatio(event.clientX, event.clientY);
          if (ratio) {
            onRegionPointRef.current?.(ratio.xRatio, ratio.yRatio);
          }
        } else {
          const destinationId = updateFloorSnap(event.clientX, event.clientY);
          if (destinationId) {
            onNavigateRef.current?.(destinationId);
          }
        }
      }
    };
    const onMouseLeave = () => {
      if (!isDragging) {
        hideFloorNavigation();
      }
      reportHover(null);
      setDepthProjectionCursor(null);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsDepthProjectionActive(true);
        hideFloorNavigation();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        clearDepthProjectionCursor();
      }
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      // Proportional, not additive: over a range this wide a fixed step in degrees is
      // a nudge at 130° and a jump across half the range at 12°.
      runtime.fov = clamp(
        runtime.fov * Math.exp(event.deltaY * 0.0015),
        FOV_MIN,
        FOV_MAX,
      );
      runtime.dirty = true;
    };
    const onTouchStart = (event: TouchEvent) => {
      if (event.touches.length !== 1) {
        return;
      }
      isTouchActive = true;
      const touch = event.touches[0];
      downX = previousX = touch.clientX;
      downY = previousY = touch.clientY;
    };
    const onTouchMove = (event: TouchEvent) => {
      if (!isTouchActive || event.touches.length !== 1) {
        return;
      }
      event.preventDefault();
      const touch = event.touches[0];
      updateView(touch.clientX, touch.clientY);
    };
    const onTouchEnd = (event: TouchEvent) => {
      if (!isTouchActive) {
        return;
      }
      isTouchActive = false;
      reportView(runtime, true);
      const touch = event.changedTouches[0];
      if (
        touch &&
        Math.hypot(touch.clientX - downX, touch.clientY - downY) <
          CLICK_DRAG_THRESHOLD_PX
      ) {
        const ratio = clickTextureRatio(touch.clientX, touch.clientY);
        if (ratio) {
          if (regionDrawActiveRef.current) {
            onRegionPointRef.current?.(ratio.xRatio, ratio.yRatio);
          } else {
            onDepthPinRef.current(ratio.xRatio, ratio.yRatio);
          }
        }
      }
    };

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd, { passive: true });
    canvas.addEventListener("touchcancel", onTouchEnd, { passive: true });
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", clearDepthProjectionCursor);

    const animate = () => {
      runtime.animationFrame = requestAnimationFrame(animate);
      if (!runtime.dirty) {
        return;
      }
      runtime.dirty = false;
      camera.lookAt(
        Math.cos(runtime.phi) * Math.sin(runtime.theta),
        Math.sin(runtime.phi),
        -Math.cos(runtime.phi) * Math.cos(runtime.theta),
      );
      camera.fov = runtime.fov;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(runtime.animationFrame);
      if (runtime.viewUpdateRaf !== null) {
        cancelAnimationFrame(runtime.viewUpdateRaf);
      }
      resizeObserver.disconnect();
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("touchcancel", onTouchEnd);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", clearDepthProjectionCursor);
      for (const child of overlayGroup.children.slice()) {
        overlayGroup.remove(child);
        disposeObject(child);
      }
      runtime.texture?.dispose();
      floorCursor.geometry.dispose();
      (floorCursor.material as THREE.Material).dispose();
      floorSnap.geometry.dispose();
      (floorSnap.material as THREE.Material).dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      runtimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    let cancelled = false;
    const shouldResetView = textureKeyframeIdRef.current !== props.keyframeId;
    textureKeyframeIdRef.current = props.keyframeId;
    setLoadStatus("loading");
    if (shouldResetView) {
      runtime.theta = normalizeYaw(props.initialYawRad);
      runtime.phi = clamp(
        (0.5 - props.initialTextureYRatio) * Math.PI,
        -Math.PI / 2 + 0.01,
        Math.PI / 2 - 0.01,
      );
    }
    // Clear old texture immediately so the sphere goes dark rather than
    // showing the previous keyframe's image at the new initial orientation.
    if (runtime.texture) {
      runtime.texture.dispose();
      runtime.texture = null;
    }
    runtime.material.map = null;
    runtime.material.color.set(0x111827);
    runtime.material.needsUpdate = true;
    runtime.dirty = true;

    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "";
    loader.load(
      props.imageSrc,
      (texture) => {
        if (cancelled) {
          texture.dispose();
          return;
        }
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.wrapS = THREE.RepeatWrapping;
        texture.repeat.x = -1;
        texture.offset.x = 0.25;
        runtime.texture = texture;
        runtime.material.map = texture;
        runtime.material.color.set(0xffffff);
        runtime.material.needsUpdate = true;
        const image = texture.image as { width?: number; height?: number };
        runtime.imageWidth = image.width ?? 0;
        runtime.imageHeight = image.height ?? 0;
        runtime.dirty = true;
        setLoadStatus("ready");
        reportView(runtime, true);
      },
      undefined,
      () => {
        if (!cancelled) {
          setLoadStatus("error");
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [props.imageSrc]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime || props.orientToken === undefined) {
      return;
    }
    runtime.theta = normalizeYaw(props.orientYawRad ?? 0);
    runtime.dirty = true;
    reportView(runtime, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.orientToken]);

  // Only reacts to the prop identity changing — never to an unrelated re-render — so
  // it cannot stomp on a box mid-drag, which updates the same ref imperatively.
  useEffect(() => {
    editableBoxRef.current = props.editableBox ?? null;
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    rebuildEditableBoxOverlay(runtime, editableBoxRef.current);
    runtime.dirty = true;
  }, [props.editableBox]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    for (const child of runtime.overlayGroup.children.slice()) {
      runtime.overlayGroup.remove(child);
      disposeObject(child);
    }
    bboxLinesRef.current.clear();

    // A rubber band left behind after the outline is closed or cancelled would point
    // at a vertex that no longer exists; it is only rebuilt while drawing.
    if (!props.regionDrawActive || !props.draftRegion?.length) {
      for (const child of runtime.liveDraftGroup.children.slice()) {
        runtime.liveDraftGroup.remove(child);
        disposeObject(child);
      }
    }

    const hoverTargets: Array<{
      rowIndex: number;
      u0: number;
      v0: number;
      u1: number;
      v1: number;
    }> = [];

    for (const item of props.detections) {
      const ratios = props.polygonForDetection(item);
      if (ratios.length < 2) {
        continue;
      }
      hoverTargets.push({
        rowIndex: item.row_index,
        u0: Math.min(...ratios.map(([u]) => u)),
        u1: Math.max(...ratios.map(([u]) => u)),
        v0: Math.min(...ratios.map(([, v]) => v)),
        v1: Math.max(...ratios.map(([, v]) => v)),
      });
      const points: THREE.Vector3[] = [];
      for (let index = 0; index < ratios.length; index += 1) {
        points.push(
          ...interpolatedEdgePoints(
            ratios[index],
            ratios[(index + 1) % ratios.length],
          ).slice(index === 0 ? 0 : 1),
        );
      }
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const selected = item.row_index === props.selectedRowIndex;
      const material = new THREE.LineBasicMaterial({
        linewidth: selected ? 3 : 2,
        depthTest: false,
        transparent: true,
      });
      const line = new THREE.LineLoop(geometry, material);
      line.userData.markerType = "bbox";
      // A rebuild must not drop the highlight the cursor is currently sitting on.
      styleBboxLine(
        line,
        selected,
        item.row_index === hoveredRowIndexRef.current,
        annotatedRowIndicesRef.current?.has(item.row_index) ?? false,
      );
      bboxLinesRef.current.set(item.row_index, line);
      runtime.overlayGroup.add(line);
    }

    for (const outline of props.annotationOutlines ?? []) {
      if (outline.region.length < 2) {
        continue;
      }
      const points: THREE.Vector3[] = [];
      for (let index = 0; index < outline.region.length; index += 1) {
        points.push(
          ...interpolatedEdgePoints(
            outline.region[index],
            outline.region[(index + 1) % outline.region.length],
          ).slice(index === 0 ? 0 : 1),
        );
      }
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        linewidth: 2,
        depthTest: false,
        transparent: true,
      });
      const line = new THREE.LineLoop(geometry, material);
      line.userData.markerType = "annotation-outline";
      styleBboxLine(line, false, false, true);
      runtime.overlayGroup.add(line);
    }

    // Smallest first: a box fully inside another must win the hit, or it could never
    // be hovered at all.
    hoverTargets.sort(
      (a, b) => (a.u1 - a.u0) * (a.v1 - a.v0) - (b.u1 - b.u0) * (b.v1 - b.v0),
    );
    hoverTargetsRef.current = hoverTargets;

    const draftRegion = props.draftRegion ?? [];
    if (draftRegion.length) {
      if (draftRegion.length >= 2) {
        const closed = draftRegion.length >= 3;
        const line = makeThickLine(
          regionWorldPoints(draftRegion, closed),
          REGION_DRAFT_COLOR,
          runtime.resolution,
          false,
        );
        line.userData.markerType = "region-draft";
        line.renderOrder = 40;
        runtime.overlayGroup.add(line);
      }

      // Every clicked point shows from the very first one — a vertex with no edge yet
      // is still the only feedback that the click landed — and each carries a white
      // ring so it reads against a bright ceiling as well as a dark floor.
      draftRegion.forEach((vertex, index) => {
        const halo = new THREE.Mesh(
          new THREE.SphereGeometry(4.6, 14, 12),
          new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }),
        );
        halo.position.copy(textureRatioToDirection(vertex[0], vertex[1], OVERLAY_RADIUS - 1));
        halo.userData.markerType = "region-draft";
        halo.renderOrder = 41;
        runtime.overlayGroup.add(halo);

        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(3.2, 14, 12),
          new THREE.MeshBasicMaterial({
            // The first vertex closes the outline, so it is named apart.
            color: index === 0 ? 0x0b3aa8 : REGION_DRAFT_COLOR,
            depthTest: false,
          }),
        );
        dot.position.copy(textureRatioToDirection(vertex[0], vertex[1], OVERLAY_RADIUS - 2));
        dot.userData.markerType = "region-draft";
        dot.renderOrder = 42;
        runtime.overlayGroup.add(dot);
      });
    }

    if (props.depthPin) {
      const color =
        props.depthPin.status === "resolving"
          ? 0xf59e0b
          : props.depthPin.status === "error"
            ? 0x6b7280
            : 0xe11d48;
      const geometry = new THREE.SphereGeometry(3.5, 16, 12);
      const material = new THREE.MeshBasicMaterial({
        color,
        depthTest: false,
      });
      const marker = new THREE.Mesh(geometry, material);
      marker.position.copy(
        textureRatioToDirection(
          props.depthPin.xRatio,
          props.depthPin.yRatio,
          OVERLAY_RADIUS - 2,
        ),
      );
      marker.userData.markerType = "depth-pin";
      marker.renderOrder = 30;
      runtime.overlayGroup.add(marker);
    }
    runtime.dirty = true;
  }, [
    props.depthPin,
    props.detections,
    props.draftRegion,
    props.regionDrawActive,
    props.polygonForDetection,
    props.selectedRowIndex,
    props.annotatedRowIndices,
    props.annotationOutlines,
  ]);

  /** One button press, as a ratio of the current field of view. */
  const zoom = (factor: number) => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    runtime.fov = clamp(runtime.fov * factor, FOV_MIN, FOV_MAX);
    runtime.dirty = true;
  };

  return (
    <div
      ref={containerRef}
      className="object-search-keyframe-photosphere"
      style={{ height: props.height }}
      aria-label="Panoramic keyframe viewer"
    >
      <canvas
        ref={canvasRef}
        className="object-search-keyframe-photosphere-canvas"
        title={
          props.navigationCandidates?.length
            ? "Click the floor to move to the nearest keyframe. Ctrl+click to place a depth pin."
            : undefined
        }
      />
      {isDepthProjectionActive && depthProjectionCursor ? (
        <div
          className="object-search-depth-projection-cursor"
          style={{
            left: depthProjectionCursor.x,
            top: depthProjectionCursor.y,
          }}
        >
          <span aria-hidden="true" />
        </div>
      ) : null}
      <div className="object-search-keyframe-photosphere-toolbar">
        <button type="button" title="Zoom in" aria-label="Zoom in" onClick={() => zoom(1 / ZOOM_BUTTON_FACTOR)}>
          +
        </button>
        <button type="button" title="Zoom out" aria-label="Zoom out" onClick={() => zoom(ZOOM_BUTTON_FACTOR)}>
          -
        </button>
        <button
          type="button"
          title="Fullscreen"
          aria-label="Fullscreen"
          onClick={() => void containerRef.current?.requestFullscreen()}
        >
          []
        </button>
      </div>
      {loadStatus === "loading" ? (
        <div className="object-search-keyframe-photosphere-status">Loading panorama...</div>
      ) : null}
      {loadStatus === "error" ? (
        <div className="object-search-keyframe-photosphere-status is-error">
          Failed to load panorama
        </div>
      ) : null}
    </div>
  );
}
