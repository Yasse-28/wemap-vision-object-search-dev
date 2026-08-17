import {
  AnnotationMode,
  LivemapMarker,
  LivemapPolygon,
  LivemapSegment,
  LngLat,
  MapClickEvent,
  RoiPolygon,
} from "./types";

/**
 * Cache for the third-party Wemap livemap.
 *
 * The SDK exposes a single global (`window.wemap.v1.getPrivateInterface`), so
 * destroying and re-creating a livemap on every tab switch leaves the page with
 * a stale facade pointing at a dead Mapbox map — the symptom being a blank map
 * until a full page reload. Instead, the livemap is created once per `emmid`
 * inside a detached container that survives unmounts: the mounted component
 * only re-parents that container and re-synchronises its overlays.
 */

export type LivemapCone = {
  latitude: number;
  longitude: number;
  bearingDeg: number;
  fovDeg?: number;
  level?: string | null;
};

declare global {
  interface Window {
    wemap?: WemapSdk;
  }
}

export type GeoJsonGeometry =
  | { type: "Point"; coordinates: [number, number] }
  | { type: "LineString"; coordinates: Array<[number, number]> }
  | { type: "Polygon"; coordinates: Array<Array<[number, number]>> };

export type GeoJsonFeature = {
  type: "Feature";
  geometry: GeoJsonGeometry;
  properties: Record<string, unknown>;
};

export type GeoJsonFeatureCollection = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

export type MapboxSource = {
  setData(data: GeoJsonFeatureCollection): void;
};

export type MapboxPoint = {
  x: number;
  y: number;
};

export type MapboxFeature = {
  properties?: Record<string, unknown>;
};

export type MapboxPointerEvent = {
  features?: MapboxFeature[];
  lngLat: {
    lat: number;
    lng: number;
  };
  point: MapboxPoint;
  originalEvent?: {
    clientX?: number;
    clientY?: number;
    preventDefault?: () => void;
  };
  preventDefault?: () => void;
};

export type MapboxMap = {
  addImage(id: string, image: ImageData, options?: { pixelRatio?: number }): void;
  addLayer(layer: Record<string, unknown>): void;
  addSource(id: string, source: Record<string, unknown>): void;
  easeTo(options: { center: [number, number]; zoom: number; duration: number }): void;
  getCanvas(): HTMLCanvasElement;
  getLayer(id: string): unknown;
  getSource(id: string): MapboxSource | undefined;
  hasImage(id: string): boolean;
  on(event: string, listener: (event: MapboxPointerEvent) => void): void;
  on(event: string, layerId: string, listener: (event: MapboxPointerEvent) => void): void;
  project(coordinates: [number, number]): MapboxPoint;
  resize(): void;
};

export type WemapLivemap = {
  addEventListener(event: "ready", listener: () => void | Promise<void>): void;
  addEventListener(
    event: "floorChanged" | "indoorLevelChanged",
    listener: (data: Record<string, unknown>) => void | Promise<void>,
  ): void;
  addEventListener(
    event: "mapClick",
    listener: (data: { latitude: number; longitude: number }) => void,
  ): void;
  addExternalIndoorLayers?(layerIds: string[]): void;
  centerTo?(options: {
    center: {
      latitude: number;
      longitude: number;
    };
    zoom: number;
  }): Promise<void>;
  destroy?(): void;
  disablePositioningSystem?(): void;
  getCurrentFloor?(): Promise<unknown>;
  getIndoorLevel?(): Promise<unknown>;
  setIndoorLevel?(level: number): void;
};

export type WemapFloorStore = {
  getCurrentFloor?(): unknown;
  getCurrentIndoorLevel?(): unknown;
};

export type WemapFacade = {
  getApplication?(): {
    map?: {
      _driver?: {
        _map?: MapboxMap;
      };
    };
  };
  getStore?(name: "FloorStore"): WemapFloorStore | null;
};

export type WemapPrivateInterface = {
  getFacade?(): WemapFacade | null;
};

export type WemapSdk = {
  v1?: {
    createLivemap?(
      container: HTMLElement,
      options: Record<string, unknown>,
      useShadowDom: boolean,
    ): WemapLivemap;
    getPrivateInterface?(callback: (privateInterface: WemapPrivateInterface) => void): void;
  };
};

const SDK_URL = "https://livemap.getwemap.com/js/sdk.min.js";
const SDK_READY_TIMEOUT_MS = 15_000;
const SDK_READY_POLL_MS = 100;

let sdkLoader: Promise<void> | null = null;

function isWemapSdkReady(): boolean {
  return (
    typeof window.wemap !== "undefined" &&
    !!window.wemap?.v1?.createLivemap
  );
}

function waitForWemapSdkReady(): Promise<void> {
  if (isWemapSdkReady()) {
    return Promise.resolve();
  }
  return new Promise<void>((resolve, reject) => {
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      if (isWemapSdkReady()) {
        window.clearInterval(timer);
        resolve();
        return;
      }
      if (Date.now() - startedAt > SDK_READY_TIMEOUT_MS) {
        window.clearInterval(timer);
        reject(new Error("Wemap SDK loaded but did not become ready."));
      }
    }, SDK_READY_POLL_MS);
  });
}

export function loadWemapSdk(): Promise<void> {
  if (isWemapSdkReady()) {
    return Promise.resolve();
  }
  if (sdkLoader) {
    return sdkLoader;
  }
  sdkLoader = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${SDK_URL}"]`,
    );
    if (existing) {
      if (existing.dataset.loaded === "true") {
        void waitForWemapSdkReady().then(resolve, reject);
        return;
      }
      existing.addEventListener("load", () => {
        existing.dataset.loaded = "true";
        void waitForWemapSdkReady().then(resolve, reject);
      });
      existing.addEventListener("error", () =>
        reject(new Error("Failed to load Wemap SDK.")),
      );
      void waitForWemapSdkReady().then(resolve, reject);
      return;
    }
    const script = document.createElement("script");
    script.src = SDK_URL;
    script.async = true;
    script.onload = () => {
      script.dataset.loaded = "true";
      void waitForWemapSdkReady().then(resolve, reject);
    };
    script.onerror = () => reject(new Error("Failed to load Wemap SDK."));
    document.head.appendChild(script);
  });
  sdkLoader = sdkLoader.catch((error) => {
    sdkLoader = null;
    throw error;
  });
  return sdkLoader;
}

/**
 * Livemap state that outlives the mounted component.
 *
 * The SDK/Mapbox handlers registered once per host close over this object, so
 * it is always mutated in place and never replaced.
 */
export type LivemapState = {
  livemap: WemapLivemap | null;
  facade: WemapFacade | null;
  floorStore: WemapFloorStore | null;
  mapboxMap: MapboxMap | null;
  overlayLayersReady: boolean;
  currentLevel: string | null;
  currentAltitude: number | null;
  activeColor: string | null;
  mode: AnnotationMode;
  draftVertices: LngLat[];
  draftClosed: boolean;
  draftTimestamp: number;
  markers: LivemapMarker[];
  segmentHoverMarkers: LivemapMarker[];
  segmentHoverMarker: LivemapMarker | null;
  segments: LivemapSegment[];
  polygons: LivemapPolygon[];
  pendingPoint: LngLat | null;
  markerClickHandlerRegistered: boolean;
  roiHandlersRegistered: boolean;
  cone: LivemapCone | null;
  roiActive: boolean;
  roiVertices: LngLat[];
  roiClosed: boolean;
  roiCursor: LngLat | null;
};

export function createLivemapState(): LivemapState {
  return {
    livemap: null,
    facade: null,
    floorStore: null,
    mapboxMap: null,
    overlayLayersReady: false,
    currentLevel: null,
    currentAltitude: null,
    ...createConsumerState(),
    markerClickHandlerRegistered: false,
    roiHandlersRegistered: false,
  };
}

/** State owned by the mounted panel, wiped when it hands the map over. */
function createConsumerState() {
  return {
    activeColor: null,
    mode: "point" as AnnotationMode,
    draftVertices: [] as LngLat[],
    draftClosed: false,
    draftTimestamp: 0,
    markers: [] as LivemapMarker[],
    segmentHoverMarkers: [] as LivemapMarker[],
    segmentHoverMarker: null as LivemapMarker | null,
    segments: [] as LivemapSegment[],
    polygons: [] as LivemapPolygon[],
    pendingPoint: null as LngLat | null,
    cone: null as LivemapCone | null,
    roiActive: false,
    roiVertices: [] as LngLat[],
    roiClosed: false,
    roiCursor: null as LngLat | null,
  };
}

/** Clears the overlays of the outgoing panel, keeping map/camera/level state. */
export function resetConsumerState(state: LivemapState): void {
  Object.assign(state, createConsumerState());
}

/**
 * Callbacks of the currently mounted component.
 *
 * Handlers are registered once per livemap instance, so they must not close
 * over a component's props: they read this shared, mutable slot instead, which
 * the mounted component overwrites on every render.
 */
export type LivemapHostCallbacks = {
  // Props of the mounted component.
  onMapClick?: (event: MapClickEvent) => void;
  onMarkerClick?: (markerId: string) => void;
  /** Marker under the cursor, or null when it leaves. */
  onMarkerHover?: (markerId: string | null) => void;
  onMarkerContextMenu?: (
    markerId: string,
    position: { x: number; y: number },
  ) => void;
  onSegmentHover?: (position: LngLat & { level: string | null }) => void;
  onSegmentClick?: (position: LngLat & { level: string | null }) => void;
  onSegmentLeave?: () => void;
  onSegmentMarkerClick?: (markerId: string) => void;
  onRoiChange?: (polygon: RoiPolygon | null) => void;
  onStatus?: (status: string) => void;
  onLevelChange?: (level: string | null) => void;
  // Host events forwarded to the mounted component.
  onReady?: () => void | Promise<void>;
  onFloorChanged?: (data: Record<string, unknown>) => void | Promise<void>;
  onIndoorLevelChanged?: (data: Record<string, unknown>) => void | Promise<void>;
  onSdkMapClick?: (data: { latitude: number; longitude: number }) => void;
  onCameraMove?: () => void;
};

export type LivemapHost = {
  emmid: number;
  /** Detached container carrying the livemap DOM; re-parented on mount. */
  el: HTMLDivElement;
  state: LivemapState;
  callbacks: LivemapHostCallbacks;
  /** The SDK "ready" event already fired; it will not fire again. */
  ready: boolean;
  attached: boolean;
  cameraListenersRegistered: boolean;
};

const callbacksByEmmid = new Map<number, LivemapHostCallbacks>();

/**
 * Stable callbacks slot for an `emmid`, shared by every component instance that
 * mounts the same livemap.
 */
export function getLivemapCallbacks(emmid: number): LivemapHostCallbacks {
  let callbacks = callbacksByEmmid.get(emmid);
  if (!callbacks) {
    callbacks = {};
    callbacksByEmmid.set(emmid, callbacks);
  }
  return callbacks;
}

let parkingElement: HTMLDivElement | null = null;

/**
 * Off-screen holder for detached livemaps. Deliberately sized (not
 * `display:none`) so Mapbox keeps a non-zero canvas while parked.
 */
function getParkingElement(): HTMLDivElement {
  if (parkingElement && parkingElement.isConnected) {
    return parkingElement;
  }
  const element = document.createElement("div");
  element.setAttribute("data-livemap-parking", "");
  element.style.cssText =
    "position:absolute;left:-10000px;top:0;width:800px;height:600px;overflow:hidden;pointer-events:none;";
  document.body.appendChild(element);
  parkingElement = element;
  return element;
}

let currentHost: LivemapHost | null = null;
let pendingHost: { emmid: number; promise: Promise<LivemapHost> } | null = null;

function registerSdkListeners(host: LivemapHost): void {
  const livemap = host.state.livemap;
  if (!livemap) {
    return;
  }
  livemap.addEventListener("ready", () => {
    host.ready = true;
    return host.callbacks.onReady?.();
  });
  livemap.addEventListener("floorChanged", (data) =>
    host.callbacks.onFloorChanged?.(data),
  );
  livemap.addEventListener("indoorLevelChanged", (data) =>
    host.callbacks.onIndoorLevelChanged?.(data),
  );
  livemap.addEventListener("mapClick", (data) => {
    host.callbacks.onSdkMapClick?.(data);
  });
}

function destroyHost(host: LivemapHost): void {
  try {
    host.state.livemap?.destroy?.();
  } catch (error) {
    console.warn("Error destroying livemap", error);
  }
  host.el.remove();
  callbacksByEmmid.delete(host.emmid);
  if (currentHost === host) {
    currentHost = null;
  }
}

async function createHost(emmid: number): Promise<LivemapHost> {
  await loadWemapSdk();
  const createLivemap = window.wemap?.v1?.createLivemap;
  if (!createLivemap) {
    throw new Error("Wemap SDK is not available.");
  }
  // Only one livemap is kept alive; switching map means the old one is useless.
  if (currentHost) {
    destroyHost(currentHost);
  }
  const el = document.createElement("div");
  el.className = "livemap-host";
  el.style.width = "100%";
  el.style.height = "100%";
  getParkingElement().appendChild(el);

  const host: LivemapHost = {
    emmid,
    el,
    state: createLivemapState(),
    callbacks: getLivemapCallbacks(emmid),
    ready: false,
    attached: false,
    cameraListenersRegistered: false,
  };
  host.state.livemap = createLivemap(
    el,
    {
      emmid,
      aroundme: false,
      maxzoom: 24,
      enabledcontrols: true,
      deeplinkingenabled: false,
      searchcontrol: false,
      enablesidebar: false,
      forcelittlescreenview: true,
      disable_analytics: true,
      arnavigationdata: {
        providers: {
          usePositionSmoother: false,
        },
      },
    },
    false,
  );
  host.state.livemap.disablePositioningSystem?.();
  registerSdkListeners(host);
  currentHost = host;
  return host;
}

/**
 * Returns the cached livemap for `emmid`, creating it on first use.
 *
 * Concurrent calls (React StrictMode double-mount, tab switch) share the same
 * in-flight creation so a single livemap is ever instantiated.
 */
export function acquireLivemapHost(emmid: number): Promise<LivemapHost> {
  if (currentHost && currentHost.emmid === emmid) {
    currentHost.attached = true;
    return Promise.resolve(currentHost);
  }
  if (pendingHost && pendingHost.emmid === emmid) {
    return pendingHost.promise;
  }
  const promise = createHost(emmid).then(
    (host) => {
      if (pendingHost?.promise === promise) {
        pendingHost = null;
      }
      host.attached = true;
      return host;
    },
    (error: unknown) => {
      if (pendingHost?.promise === promise) {
        pendingHost = null;
      }
      throw error;
    },
  );
  pendingHost = { emmid, promise };
  return promise;
}

/** Parks the livemap off-screen for the next panel; never destroys it. */
export function releaseLivemapHost(host: LivemapHost): void {
  host.attached = false;
  getParkingElement().appendChild(host.el);
}
