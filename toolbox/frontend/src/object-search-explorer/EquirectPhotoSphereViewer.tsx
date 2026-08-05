import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

import type { IndexObjectRecord } from "../index-explorer/types";

const FOV_DEFAULT = 60;
const FOV_MIN = 30;
const FOV_MAX = 100;
const SPHERE_RADIUS = 500;
const OVERLAY_RADIUS = 495;
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
  detections: IndexObjectRecord[];
  selectedObjectId: string | null;
  depthPin: DepthPinMarker | null;
  polygonForDetection: (item: IndexObjectRecord) => Array<[number, number]>;
  onDepthPin: (xRatio: number, yRatio: number) => void;
  allowDepthPinOnMarker?: boolean;
  navigationCandidates?: NavigationCandidate[];
  onNavigate?: (keyframeId: string) => void;
  onViewChange?: (keyframeId: string, yawRad: number, textureYRatio: number) => void;
};

type Runtime = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  material: THREE.MeshBasicMaterial;
  geometry: THREE.SphereGeometry;
  overlayGroup: THREE.Group;
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
  initialYawRadRef.current = props.initialYawRad;
  initialTextureYRatioRef.current = props.initialTextureYRatio;

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

    const resize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      runtime.dirty = true;
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let isDragging = false;
    let isTouchActive = false;
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

    const onMouseDown = (event: MouseEvent) => {
      if (isTouchActive) {
        return;
      }
      isDragging = true;
      downX = previousX = event.clientX;
      downY = previousY = event.clientY;
    };
    const onMouseMove = (event: MouseEvent) => {
      updateDepthProjectionCursor(event);
      if (isDragging) {
        updateView(event.clientX, event.clientY);
      } else if (event.ctrlKey) {
        hideFloorNavigation();
      } else {
        updateFloorSnap(event.clientX, event.clientY);
      }
    };
    const onMouseUp = (event: MouseEvent) => {
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
      runtime.fov = clamp(runtime.fov + event.deltaY * 0.08, FOV_MIN, FOV_MAX);
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
          onDepthPinRef.current(ratio.xRatio, ratio.yRatio);
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

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    for (const child of runtime.overlayGroup.children.slice()) {
      runtime.overlayGroup.remove(child);
      disposeObject(child);
    }

    for (const item of props.detections) {
      const ratios = props.polygonForDetection(item);
      if (ratios.length < 2) {
        continue;
      }
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
      const selected = item.id === props.selectedObjectId;
      const material = new THREE.LineBasicMaterial({
        color: selected ? 0xff6b6b : 0x11b5ae,
        linewidth: selected ? 3 : 2,
        depthTest: false,
        transparent: true,
        opacity: selected ? 1 : 0.9,
      });
      const line = new THREE.LineLoop(geometry, material);
      line.userData.markerType = "bbox";
      line.renderOrder = selected ? 20 : 10;
      runtime.overlayGroup.add(line);
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
    props.polygonForDetection,
    props.selectedObjectId,
  ]);

  const zoom = (delta: number) => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    runtime.fov = clamp(runtime.fov + delta, FOV_MIN, FOV_MAX);
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
        <button type="button" title="Zoom in" aria-label="Zoom in" onClick={() => zoom(-10)}>
          +
        </button>
        <button type="button" title="Zoom out" aria-label="Zoom out" onClick={() => zoom(10)}>
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
