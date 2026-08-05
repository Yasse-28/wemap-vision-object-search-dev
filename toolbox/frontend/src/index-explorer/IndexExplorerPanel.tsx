import { FormEvent, useEffect, useMemo, useState } from "react";

import type { MapSummary } from "../api";
import CollapsibleSection from "../annotations/CollapsibleSection";
import LivemapAnnotation from "../annotations/LivemapAnnotation";
import type { LivemapMarker } from "../annotations/types";
import DismissibleAlert from "../DismissibleAlert";
import {
  computePromptLatent,
  fetchClusterOcr,
  fetchIndexCutout,
  fetchIndexObjects,
  fetchIndexStatus,
  indexCutoutPreviewUrl,
  indexObjectCropUrl,
} from "./api";
import LatentScatter from "./LatentScatter";
import type {
  ClusterOcrRecord,
  CutoutIndexPayload,
  ExplorerView,
  IndexObjectRecord,
  IndexStatusResponse,
  IndexSummary,
  PromptLatentResponse,
} from "./types";
import { objectMatchesOcrFilters, parseOcrSourceFilter } from "./utils";

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
};

const OCR_SOURCE_OPTIONS = [
  "all",
  "0 - none",
  "1 - lightweight PPOCR",
  "2 - PaddleOCR-VL",
];

function IndexExplorerPanel(props: Props) {
  const emmid = props.map?.emmid ?? null;
  const [view, setView] = useState<ExplorerView>("browse");
  const [status, setStatus] = useState<IndexStatusResponse | null>(null);
  const [allObjects, setAllObjects] = useState<IndexObjectRecord[]>([]);
  const [clusterOcr, setClusterOcr] = useState<ClusterOcrRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const [selectedKeyframeId, setSelectedKeyframeId] = useState<string | null>(null);
  const [selectedCutoutId, setSelectedCutoutId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [cutoutPayload, setCutoutPayload] = useState<CutoutIndexPayload | null>(null);

  const [ocrQuery, setOcrQuery] = useState("");
  const [ocrSource, setOcrSource] = useState("all");
  const [ocrRequireKey, setOcrRequireKey] = useState(false);
  const [ocrCandidateOnly, setOcrCandidateOnly] = useState(false);
  const [ocrAssignedOnly, setOcrAssignedOnly] = useState(false);
  const [ocrSelectedId, setOcrSelectedId] = useState<string | null>(null);

  const [promptText, setPromptText] = useState("");
  const [latentThreshold, setLatentThreshold] = useState(0.25);
  const [latentTopK, setLatentTopK] = useState(10);
  const [latentProjection, setLatentProjection] = useState<"UMAP" | "PCA">("UMAP");
  const [latentResult, setLatentResult] = useState<PromptLatentResponse | null>(null);
  const [latentLoading, setLatentLoading] = useState(false);
  const [latentQuery, setLatentQuery] = useState<string | null>(null);

  const summary: IndexSummary | null = status?.summary ?? null;

  useEffect(() => {
    if (!props.isMapKnown) {
      setIsLoading(false);
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    Promise.all([
      fetchIndexStatus(props.mapId, selectedKeyframeId),
      fetchIndexObjects(props.mapId).catch(() => [] as IndexObjectRecord[]),
      fetchClusterOcr(props.mapId).catch(() => [] as ClusterOcrRecord[]),
    ])
      .then(([indexStatus, objects, clusters]) => {
        if (cancelled) {
          return;
        }
        setStatus(indexStatus);
        setAllObjects(objects);
        setClusterOcr(clusters);
        if (indexStatus.summary?.default_prompts.length && !promptText) {
          setPromptText(indexStatus.summary.default_prompts.join("\n"));
        }
        if (indexStatus.summary?.keyframe_ids.length) {
          const first = indexStatus.summary.keyframe_ids[0];
          setSelectedKeyframeId((current) =>
            current && indexStatus.summary!.keyframe_ids.includes(current)
              ? current
              : first,
          );
        }
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
    if (!props.isMapKnown || !status?.available) {
      return;
    }
    let cancelled = false;
    fetchIndexStatus(props.mapId, selectedKeyframeId)
      .then((payload) => {
        if (!cancelled) {
          setStatus(payload);
        }
      })
      .catch(() => {
        // keep previous markers
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown, selectedKeyframeId, status?.available]);

  const cutoutIds = useMemo(() => {
    if (!summary || !selectedKeyframeId) {
      return [] as string[];
    }
    return summary.cutout_ids_by_keyframe[selectedKeyframeId] ?? [];
  }, [summary, selectedKeyframeId]);

  useEffect(() => {
    if (!cutoutIds.length) {
      setSelectedCutoutId(null);
      return;
    }
    setSelectedCutoutId((current) =>
      current && cutoutIds.includes(current) ? current : cutoutIds[0],
    );
  }, [cutoutIds]);

  useEffect(() => {
    if (!selectedCutoutId || !status?.available) {
      setCutoutPayload(null);
      setSelectedObjectId(null);
      return;
    }
    let cancelled = false;
    fetchIndexCutout(props.mapId, selectedCutoutId)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setCutoutPayload(payload);
        const ids = payload.detections.map((item) => item.id);
        setSelectedObjectId((current) =>
          current && ids.includes(current) ? current : ids[0] ?? null,
        );
      })
      .catch(() => {
        if (!cancelled) {
          setCutoutPayload(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, selectedCutoutId, status?.available]);

  const markers: LivemapMarker[] = useMemo(() => {
    if (!status?.markers) {
      return [];
    }
    return status.markers.map((marker) => ({
      id: marker.id,
      latitude: marker.latitude,
      longitude: marker.longitude,
      level: marker.level,
      color: marker.color,
      radius: marker.radius,
    }));
  }, [status?.markers]);

  const focusTarget = useMemo(() => {
    const marker = markers.find((item) => item.id === selectedKeyframeId);
    if (!marker) {
      return null;
    }
    return {
      latitude: marker.latitude,
      longitude: marker.longitude,
      level: marker.level,
    };
  }, [markers, selectedKeyframeId]);

  const filteredOcrObjects = useMemo(() => {
    const sourceFilter = parseOcrSourceFilter(ocrSource);
    return allObjects.filter((item) =>
      objectMatchesOcrFilters(item, {
        query: ocrQuery.trim(),
        sourceFilter,
        requireKey: ocrRequireKey,
        candidateOnly: ocrCandidateOnly,
        assignedOnly: ocrAssignedOnly,
      }),
    );
  }, [allObjects, ocrQuery, ocrSource, ocrRequireKey, ocrCandidateOnly, ocrAssignedOnly]);

  useEffect(() => {
    if (!filteredOcrObjects.length) {
      setOcrSelectedId(null);
      return;
    }
    setOcrSelectedId((current) =>
      current && filteredOcrObjects.some((item) => item.id === current)
        ? current
        : filteredOcrObjects[0].id,
    );
  }, [filteredOcrObjects]);

  const selectedOcrObject = filteredOcrObjects.find((item) => item.id === ocrSelectedId) ?? null;
  const totalCutouts = summary
    ? Object.values(summary.cutout_ids_by_keyframe).reduce((sum, ids) => sum + ids.length, 0)
    : 0;
  const totalObjects = summary
    ? Object.values(summary.object_count_by_cutout).reduce((sum, count) => sum + count, 0)
    : 0;

  async function runPromptLatent(event: FormEvent) {
    event.preventDefault();
    setLatentLoading(true);
    setError(null);
    try {
      const payload = await computePromptLatent(props.mapId, {
        prompts_text: promptText,
        threshold: latentThreshold,
        top_k: latentTopK,
        projection: latentProjection,
        selected_query: latentQuery,
        scale_points_by_bbox_area: true,
      });
      setLatentResult(payload);
      if (payload.selected_query) {
        setLatentQuery(payload.selected_query);
      } else if (payload.query_labels?.length) {
        setLatentQuery(payload.query_labels[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLatentLoading(false);
    }
  }

  function syncOcrToBrowse() {
    if (!selectedOcrObject) {
      return;
    }
    setSelectedKeyframeId(selectedOcrObject.keyframe_id);
    setSelectedCutoutId(selectedOcrObject.cutout_id);
    setSelectedObjectId(selectedOcrObject.id);
    setView("browse");
  }

  if (!props.isMapKnown) {
    return (
      <section className="index-explorer-panel page-section">
        <p className="error-box">Map not found in the current config.</p>
      </section>
    );
  }

  if (isLoading) {
    return (
      <section className="index-explorer-panel page-section">
        <p className="muted">Loading object-search index…</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="index-explorer-panel page-section">
        <DismissibleAlert variant="error" onDismiss={() => setError(null)}>
          {error}
        </DismissibleAlert>
        <p className="muted">
          The OS Data Explorer needs <code>object-search toolbox backend</code>{" "}
          with <code>--config</code> pointing at your VPS config. Vite proxies API
          calls to port 45700 by default, or to <code>VITE_API_PROXY_TARGET</code>{" "}
          when set.
        </p>
        {props.map?.path ? (
          <p className="path-text">
            Map path from <code>/ui/api/maps</code>: {props.map.path}
          </p>
        ) : null}
      </section>
    );
  }

  if (!status?.available || !summary) {
    const checkedPaths = status?.checked_paths ?? [];
    return (
      <section className="index-explorer-panel page-section">
        <p className="info-box">
          No object-search index database (object-search.db) found for this map.
        </p>
        {status?.map_path || props.map?.path ? (
          <p className="path-text">
            Map path: {status?.map_path ?? props.map?.path}
          </p>
        ) : null}
        {status?.object_search_index_path ? (
          <p className="path-text">
            Configured index override: {status.object_search_index_path}
          </p>
        ) : null}
        {checkedPaths.length ? (
          <>
            <p className="muted">Paths checked by the API:</p>
            <ul className="checked-paths-list">
              {checkedPaths.map((path) => (
                <li key={path}>
                  <code>{path}</code>
                </li>
              ))}
            </ul>
          </>
        ) : null}
        <p className="muted">
          If the file exists on disk but appears above as missing, the API process
          likely uses a different config file or cannot read that directory (mount /
          permissions).
        </p>
      </section>
    );
  }

  const previewUrl =
    selectedCutoutId !== null
      ? indexCutoutPreviewUrl(props.mapId, selectedCutoutId, selectedObjectId)
      : null;

  return (
    <section className="index-explorer-panel">
      <div className="index-explorer-toolbar">
        <div className="segmented-control" role="group" aria-label="Explorer view">
          <button
            type="button"
            className={view === "browse" ? "is-active" : ""}
            onClick={() => setView("browse")}
          >
            Browse
          </button>
          <button
            type="button"
            className={view === "prompt-latent" ? "is-active" : ""}
            onClick={() => setView("prompt-latent")}
          >
            Prompt Latent Space
          </button>
        </div>
        <p className="path-text">{summary.index_path}</p>
      </div>

      {error ? (
        <DismissibleAlert variant="error" onDismiss={() => setError(null)}>
          {error}
        </DismissibleAlert>
      ) : null}

      {view === "prompt-latent" ? (
        <div className="index-latent-layout">
          <form className="search-form index-latent-form" onSubmit={runPromptLatent}>
            <label>
              Reference prompts
              <textarea
                value={promptText}
                onChange={(event) => setPromptText(event.target.value)}
                rows={6}
              />
            </label>
            <div className="form-row">
              <label>
                Activation threshold
                <input
                  type="number"
                  min={-1}
                  max={1}
                  step={0.01}
                  value={latentThreshold}
                  onChange={(event) => setLatentThreshold(Number(event.target.value))}
                />
              </label>
              <label>
                Top-k rows
                <input
                  type="number"
                  min={3}
                  max={50}
                  value={latentTopK}
                  onChange={(event) => setLatentTopK(Number(event.target.value))}
                />
              </label>
              <label>
                Projection
                <select
                  value={latentProjection}
                  onChange={(event) =>
                    setLatentProjection(event.target.value as "UMAP" | "PCA")
                  }
                >
                  <option value="UMAP">UMAP</option>
                  <option value="PCA">PCA</option>
                </select>
              </label>
            </div>
            <button type="submit" disabled={latentLoading}>
              {latentLoading ? "Computing…" : "Compute latent view"}
            </button>
          </form>

          {latentResult?.warnings?.map((warning) => (
            <p key={warning} className="info-box">
              {warning}
            </p>
          ))}

          {latentResult?.available && latentResult.points ? (
            <>
              <div className="form-row">
                <label>
                  Query
                  <select
                    value={latentQuery ?? latentResult.selected_query ?? ""}
                    onChange={async (event) => {
                      const query = event.target.value;
                      setLatentQuery(query);
                      setLatentLoading(true);
                      try {
                        const payload = await computePromptLatent(props.mapId, {
                          prompts_text: promptText,
                          threshold: latentThreshold,
                          top_k: latentTopK,
                          projection: latentProjection,
                          selected_query: query,
                          scale_points_by_bbox_area: true,
                        });
                        setLatentResult(payload);
                      } catch (err) {
                        setError(err instanceof Error ? err.message : String(err));
                      } finally {
                        setLatentLoading(false);
                      }
                    }}
                  >
                    {(latentResult.query_labels ?? []).map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
                {latentResult.embedding_backend ? (
                  <p className="muted">Embeddings: {latentResult.embedding_backend}</p>
                ) : null}
              </div>
              <LatentScatter points={latentResult.points} />
              {latentResult.topk?.length ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Object</th>
                        <th>Cutout</th>
                        <th>Similarity</th>
                        <th>Bbox area</th>
                      </tr>
                    </thead>
                    <tbody>
                      {latentResult.topk.map((row) => (
                        <tr key={String(row.object_id)}>
                          <td>{String(row.object_id)}</td>
                          <td>{String(row.cutout_id)}</td>
                          <td>{String(row.similarity)}</td>
                          <td>{String(row.bbox_area)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </>
          ) : latentResult?.message ? (
            <p className="muted">{latentResult.message}</p>
          ) : null}
        </div>
      ) : (
        <>
          <div className="index-explorer-grid">
            <div className="index-explorer-column">
              <h3>Object-search index</h3>
              <div className="metrics-row">
                <span>Keyframes {summary.keyframe_ids.length}</span>
                <span>Cutouts {totalCutouts}</span>
                <span>Objects {totalObjects}</span>
              </div>
              <div className="metrics-row">
                <span>OCR texts {summary.ocr_text_count}</span>
                <span>OCR keys {summary.ocr_key_count}</span>
                <span>OCR clusters {summary.cluster_ocr_count}</span>
              </div>
              <div className="metrics-row">
                <span>OCR candidates {summary.ocr_candidate_count}</span>
                <span>OCR assigned {summary.ocr_assigned_count}</span>
              </div>
              {Object.keys(summary.ocr_source_counts).length ? (
                <p className="map-caption">
                  OCR sources:{" "}
                  {Object.entries(summary.ocr_source_counts)
                    .map(([source, count]) => `${source}: ${count}`)
                    .join(" | ")}
                </p>
              ) : null}

              <CollapsibleSection title="Manifest" defaultOpen={false}>
                <pre className="debug-json">{JSON.stringify(summary.manifest, null, 2)}</pre>
              </CollapsibleSection>

              {!summary.keyframe_ids.length ? (
                <p className="muted">The index does not contain any cutouts.</p>
              ) : (
                <>
                  <label>
                    Keyframe
                    <select
                      value={selectedKeyframeId ?? ""}
                      onChange={(event) => {
                        setSelectedKeyframeId(event.target.value);
                        setSelectedCutoutId(null);
                        setSelectedObjectId(null);
                      }}
                    >
                      {summary.keyframe_ids.map((keyframeId) => (
                        <option key={keyframeId} value={keyframeId}>
                          {keyframeId} |{" "}
                          {(summary.cutout_ids_by_keyframe[keyframeId] ?? []).length} cutouts
                        </option>
                      ))}
                    </select>
                  </label>

                  <CollapsibleSection
                    title="Cutouts in keyframe"
                    summary={`${cutoutIds.length} cutouts`}
                  >
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Cutout</th>
                            <th>Objects</th>
                          </tr>
                        </thead>
                        <tbody>
                          {cutoutIds.map((cutoutId) => (
                            <tr key={cutoutId}>
                              <td>{cutoutId}</td>
                              <td>{summary.object_count_by_cutout[cutoutId] ?? 0}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>
                </>
              )}
            </div>

            <div className="index-explorer-column index-explorer-map-column">
              <h3>Livemap</h3>
              {emmid === null ? (
                <p className="info-box">Map preview unavailable: no EMMID configured for this map.</p>
              ) : !summary.keyframe_ids.length ? (
                <p className="muted">The index does not contain any keyframes.</p>
              ) : !markers.length ? (
                <p className="muted">
                  No keyframe coordinates could be resolved from local map metadata.
                </p>
              ) : (
                <>
                  <LivemapAnnotation
                    emmid={emmid}
                    markers={markers}
                    segments={[]}
                    polygons={[]}
                    draftVertices={[]}
                    draftClosed={false}
                    pendingPoint={null}
                    activeColor={null}
                    mode="inspect"
                    focusTarget={focusTarget}
                    onMapClick={() => {}}
                    onMarkerClick={(markerId) => {
                      setSelectedKeyframeId(markerId);
                      setSelectedCutoutId(null);
                      setSelectedObjectId(null);
                    }}
                  />
                  <p className="map-caption">
                    {markers.length} / {summary.keyframe_ids.length} keyframes have resolvable map
                    coordinates.
                  </p>
                </>
              )}
            </div>

            <div className="index-explorer-column">
              <h3>Cutout</h3>
              {!cutoutIds.length || !selectedCutoutId ? (
                <p className="muted">Select a keyframe to inspect its cutouts.</p>
              ) : (
                <>
                  <label>
                    Cutout
                    <select
                      value={selectedCutoutId}
                      onChange={(event) => {
                        setSelectedCutoutId(event.target.value);
                        setSelectedObjectId(null);
                      }}
                    >
                      {cutoutIds.map((cutoutId) => (
                        <option key={cutoutId} value={cutoutId}>
                          {cutoutId} | {summary.object_count_by_cutout[cutoutId] ?? 0} objects
                        </option>
                      ))}
                    </select>
                  </label>
                  {previewUrl ? (
                    <img
                      className="inspector-preview"
                      src={previewUrl}
                      alt={`Cutout ${selectedCutoutId}`}
                    />
                  ) : (
                    <p className="muted">Preview unavailable for this cutout.</p>
                  )}
                  <p className="map-caption">
                    keyframe {selectedKeyframeId} |{" "}
                    {summary.object_count_by_cutout[selectedCutoutId] ?? 0} objects
                  </p>
                  <CollapsibleSection title="Preview debug" defaultOpen={false}>
                    <pre className="debug-json">
                      {JSON.stringify(cutoutPayload?.preview_debug ?? {}, null, 2)}
                    </pre>
                  </CollapsibleSection>
                </>
              )}
            </div>

            <div className="index-explorer-column">
              <h3>Objects</h3>
              {!cutoutPayload?.detections.length ? (
                <p className="muted">No objects stored for this cutout in the index.</p>
              ) : (
                <>
                  <label>
                    Object
                    <select
                      value={selectedObjectId ?? ""}
                      onChange={(event) => setSelectedObjectId(event.target.value)}
                    >
                      {cutoutPayload.detections.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.id} | bbox {item.bbox.join(", ")}
                        </option>
                      ))}
                    </select>
                  </label>
                  <pre className="debug-json">
                    {JSON.stringify(
                      cutoutPayload.detections.find((item) => item.id === selectedObjectId) ??
                        null,
                      null,
                      2,
                    )}
                  </pre>
                  <CollapsibleSection title="All object boxes" defaultOpen>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Bbox</th>
                            <th>OCR key</th>
                          </tr>
                        </thead>
                        <tbody>
                          {cutoutPayload.detections.map((item) => (
                            <tr key={item.id}>
                              <td>{item.id}</td>
                              <td>{item.bbox.join(", ")}</td>
                              <td>{item.ocr_key || item.ocr_text || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>
                </>
              )}
            </div>
          </div>

          <section className="index-ocr-section">
            <h3>OCR Browser</h3>
            {!allObjects.length ? (
              <p className="muted">No objects stored in this index.</p>
            ) : (
              <>
                <div className="index-ocr-filters">
                  <label>
                    Search OCR text, tokens, or key
                    <input
                      type="search"
                      value={ocrQuery}
                      onChange={(event) => setOcrQuery(event.target.value)}
                    />
                  </label>
                  <label>
                    OCR source
                    <select value={ocrSource} onChange={(event) => setOcrSource(event.target.value)}>
                      {OCR_SOURCE_OPTIONS.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="checkbox-row">
                    <label>
                      <input
                        type="checkbox"
                        checked={ocrRequireKey}
                        onChange={(event) => setOcrRequireKey(event.target.checked)}
                      />
                      Has OCR key
                    </label>
                    <label>
                      <input
                        type="checkbox"
                        checked={ocrCandidateOnly}
                        onChange={(event) => setOcrCandidateOnly(event.target.checked)}
                      />
                      Candidate only
                    </label>
                    <label>
                      <input
                        type="checkbox"
                        checked={ocrAssignedOnly}
                        onChange={(event) => setOcrAssignedOnly(event.target.checked)}
                      />
                      Assigned only
                    </label>
                  </div>
                </div>
                <p className="map-caption">
                  {filteredOcrObjects.length} / {allObjects.length} objects match
                </p>
                {filteredOcrObjects.length ? (
                  <div className="index-ocr-layout">
                    <div className="index-ocr-results">
                      <div className="table-wrap index-ocr-table">
                        <table>
                          <thead>
                            <tr>
                              <th>ID</th>
                              <th>Keyframe</th>
                              <th>Cutout</th>
                              <th>OCR key</th>
                              <th>OCR text</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredOcrObjects.map((item) => (
                              <tr
                                key={item.id}
                                className={item.id === ocrSelectedId ? "is-selected" : ""}
                                onClick={() => setOcrSelectedId(item.id)}
                              >
                                <td>{item.id}</td>
                                <td>{item.keyframe_id}</td>
                                <td>{item.cutout_id}</td>
                                <td>{item.ocr_key || "—"}</td>
                                <td>{item.ocr_text || "—"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <label>
                        Inspect OCR object
                        <select
                          value={ocrSelectedId ?? ""}
                          onChange={(event) => setOcrSelectedId(event.target.value)}
                        >
                          {filteredOcrObjects.map((item) => (
                            <option key={item.id} value={item.id}>
                              {item.id} | {item.ocr_key || item.ocr_text || "no OCR"}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className="index-ocr-preview">
                      {selectedOcrObject ? (
                        <>
                          <h4>Selected OCR object</h4>
                          <img
                            className="inspector-preview"
                            src={indexObjectCropUrl(
                              props.mapId,
                              selectedOcrObject.cutout_id,
                              selectedOcrObject.id,
                              selectedOcrObject.bbox,
                            )}
                            alt={`Crop ${selectedOcrObject.id}`}
                          />
                          <img
                            className="inspector-preview"
                            src={indexCutoutPreviewUrl(
                              props.mapId,
                              selectedOcrObject.cutout_id,
                              selectedOcrObject.id,
                            )}
                            alt={`Cutout ${selectedOcrObject.cutout_id}`}
                          />
                          <pre className="debug-json">
                            {JSON.stringify(selectedOcrObject, null, 2)}
                          </pre>
                          <button
                            type="button"
                            onClick={syncOcrToBrowse}
                            disabled={
                              selectedOcrObject.keyframe_id === selectedKeyframeId &&
                              selectedOcrObject.cutout_id === selectedCutoutId &&
                              selectedOcrObject.id === selectedObjectId
                            }
                          >
                            Show in top preview
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <p className="muted">No OCR objects match the current filters.</p>
                )}
                {clusterOcr.length ? (
                  <CollapsibleSection title="Cluster OCR summaries" defaultOpen={false}>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Cluster</th>
                            <th>Key</th>
                            <th>Text</th>
                            <th>Observations</th>
                          </tr>
                        </thead>
                        <tbody>
                          {clusterOcr.map((row) => (
                            <tr key={row.cluster_id}>
                              <td>{row.cluster_id}</td>
                              <td>{row.ocr_key || "—"}</td>
                              <td>{row.ocr_text || "—"}</td>
                              <td>{row.ocr_observation_count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>
                ) : null}
              </>
            )}
          </section>
        </>
      )}
    </section>
  );
}

export default IndexExplorerPanel;
