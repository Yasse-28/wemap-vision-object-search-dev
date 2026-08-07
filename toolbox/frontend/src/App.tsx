import { useEffect, useMemo, useState } from "react";

import { fetchMaps, MapSummary } from "./api";
import BenchmarkPanel from "./benchmark/BenchmarkPanel";
import FeaturePanel from "./FeaturePanel";
import ObjectSearchExplorerPanel from "./object-search-explorer/ObjectSearchExplorerPanel";
import ObjectSearchPanel from "./object-search/ObjectSearchPanel";

type Feature =
  | "object-search"
  | "object-search-explorer"
  | "annotation"
  | "benchmark";

function getPathMapId(): string | null {
  return parseMapRoute().mapId;
}

function parseMapRoute(): { mapId: string | null; feature: Feature } {
  const prefix = "/ui/maps/";
  if (!window.location.pathname.startsWith(prefix)) {
    return { mapId: null, feature: "object-search" };
  }
  const segments = window.location.pathname
    .slice(prefix.length)
    .split("/")
    .filter(Boolean);
  const mapId = decodeURIComponent(segments[0] ?? "");
  if (!mapId) {
    return { mapId: null, feature: "object-search" };
  }
  let feature: Feature = "object-search";
  if (segments[1] === "annotation") {
    feature = "annotation";
  } else if (segments[1] === "object-search-explorer") {
    feature = "object-search-explorer";
  } else if (segments[1] === "benchmark") {
    feature = "benchmark";
  }
  return { mapId, feature };
}

function mapExplorerPath(mapId: string, feature: Feature): string {
  const base = `/ui/maps/${encodeURIComponent(mapId)}`;
  if (feature === "object-search-explorer") {
    return `${base}/object-search-explorer`;
  }
  if (feature === "annotation") {
    return `${base}/annotation`;
  }
  if (feature === "benchmark") {
    return `${base}/benchmark`;
  }
  return base;
}

function App() {
  const [maps, setMaps] = useState<MapSummary[]>([]);
  const [selectedMapId, setSelectedMapId] = useState<string | null>(getPathMapId());
  const [mapsError, setMapsError] = useState<string | null>(null);
  const [isLoadingMaps, setIsLoadingMaps] = useState(true);

  useEffect(() => {
    let isMounted = true;
    fetchMaps()
      .then((items) => {
        if (!isMounted) {
          return;
        }
        setMaps(items);
        setMapsError(null);
      })
      .catch((error: Error) => {
        if (!isMounted) {
          return;
        }
        setMapsError(error.message);
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingMaps(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    const onPopState = () => {
      setSelectedMapId(getPathMapId());
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const selectedMap = useMemo(
    () => maps.find((item) => item.id === selectedMapId) ?? null,
    [maps, selectedMapId],
  );

  function openHome() {
    window.history.pushState(null, "", "/ui");
    setSelectedMapId(null);
  }

  function openMap(mapId: string) {
    window.history.pushState(null, "", `/ui/maps/${encodeURIComponent(mapId)}`);
    setSelectedMapId(mapId);
  }

  return (
    <main className="app-shell">
      {!selectedMapId ? (
        <HomePage
          maps={maps}
          isLoading={isLoadingMaps}
          error={mapsError}
          onOpenMap={openMap}
        />
      ) : (
        <MapExplorer
          map={selectedMap}
          mapId={selectedMapId}
          mapsLoaded={!isLoadingMaps}
          onBack={openHome}
        />
      )}
    </main>
  );
}

function HomePage(props: {
  maps: MapSummary[];
  isLoading: boolean;
  error: string | null;
  onOpenMap: (mapId: string) => void;
}) {
  return (
    <section className="page-section">
      <div className="section-heading">
        <p className="eyebrow">Maps</p>
        <h1>Select a map</h1>
      </div>

      {props.isLoading ? <p className="muted">Loading maps...</p> : null}
      {props.error ? <p className="error-box">{props.error}</p> : null}
      {!props.isLoading && !props.error && props.maps.length === 0 ? (
        <p className="muted">No maps are configured.</p>
      ) : null}

      <div className="map-list">
        {props.maps.map((map) =>
          map.object_search_available ? (
            <button
              className="map-row"
              type="button"
              key={map.id}
              onClick={() => props.onOpenMap(map.id)}
            >
              <span>
                <strong>{map.id}</strong>
                <small>{map.path}</small>
              </span>
              <span aria-hidden="true">Open</span>
            </button>
          ) : (
            <div className="map-row is-unavailable" key={map.id} aria-disabled="true">
              <span>
                <strong>{map.id}</strong>
                <small>{map.path}</small>
              </span>
              <span
                className="map-row-info"
                title={map.unavailable_reason ?? "This map is not usable."}
                role="img"
                aria-label={map.unavailable_reason ?? "This map is not usable."}
              >
                ⓘ
              </span>
            </div>
          ),
        )}
      </div>
    </section>
  );
}

function MapExplorer(props: {
  map: MapSummary | null;
  mapId: string;
  mapsLoaded: boolean;
  onBack: () => void;
}) {
  const displayId = props.map?.id ?? props.mapId;
  const [activeFeature, setActiveFeature] = useState<Feature>(
    () => parseMapRoute().feature,
  );

  useEffect(() => {
    const onPopState = () => {
      setActiveFeature(parseMapRoute().feature);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  function selectFeature(feature: Feature) {
    window.history.pushState(null, "", mapExplorerPath(props.mapId, feature));
    setActiveFeature(feature);
  }

  return (
    <section className="explorer-grid">
      <FeaturePanel onBack={props.onBack}>
        <div>
          <p className="eyebrow">Map</p>
          <h1>{displayId}</h1>
          {props.map?.path ? <p className="path-text">{props.map.path}</p> : null}
          {!props.map && props.mapsLoaded ? (
            <p className="error-box">Map not found in the current config.</p>
          ) : null}
        </div>

        <nav className="feature-list" aria-label="Map features">
          <button
            className={`feature-button${
              activeFeature === "object-search" ? " is-active" : ""
            }`}
            type="button"
            onClick={() => selectFeature("object-search")}
          >
            Object Search
          </button>
          <button
            className={`feature-button${
              activeFeature === "object-search-explorer" ? " is-active" : ""
            }`}
            type="button"
            onClick={() => selectFeature("object-search-explorer")}
          >
            Object Search Explorer
          </button>
          <button
            className={`feature-button${
              activeFeature === "annotation" ? " is-active" : ""
            }`}
            type="button"
            onClick={() => selectFeature("annotation")}
          >
            Annotation
          </button>
          <button
            className={`feature-button${
              activeFeature === "benchmark" ? " is-active" : ""
            }`}
            type="button"
            onClick={() => selectFeature("benchmark")}
          >
            Benchmark
          </button>
        </nav>
      </FeaturePanel>

      {activeFeature === "object-search" ? (
        <ObjectSearchPanel
          map={props.map}
          mapId={props.mapId}
          isMapKnown={Boolean(props.map)}
        />
      ) : activeFeature === "object-search-explorer" ? (
        <ObjectSearchExplorerPanel
          map={props.map}
          mapId={props.mapId}
          isMapKnown={Boolean(props.map)}
        />
      ) : activeFeature === "annotation" ? (
        <ObjectSearchPanel
          map={props.map}
          mapId={props.mapId}
          isMapKnown={Boolean(props.map)}
          reviewMode
        />
      ) : activeFeature === "benchmark" ? (
        <BenchmarkPanel
          map={props.map}
          mapId={props.mapId}
          isMapKnown={Boolean(props.map)}
        />
      ) : null}
    </section>
  );
}

export default App;
