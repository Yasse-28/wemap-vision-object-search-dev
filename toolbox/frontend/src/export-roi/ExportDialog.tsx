/**
 * The save dialog: pick a folder, name the map, export.
 *
 * The folder browser lists directories through the backend, confined to the map
 * root and the config directory — the browser has no directory picker that yields
 * a server-side path, and the export writes on the machine the backend runs on.
 *
 * Two counts appear here on purpose. The panel's is instant, from the marker table,
 * on the floor the Wemap SDK reports. The one below it comes from the python
 * selection, which reads the floor from the manifest's altitude bands — the only
 * definition the exported manifest can be consistent with. A gap between them is
 * shown rather than smoothed over.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError } from "../http";
import {
  browseDirectories,
  createDirectory,
  fetchExportStatus,
  previewExport,
  startExport,
} from "./api";
import type {
  DirectoryListing,
  ExportJob,
  ExportPreview,
  ExportRequest,
} from "./types";

const POLL_INTERVAL_MS = 700;

function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    return error.detail;
  }
  return error instanceof Error ? error.message : String(error);
}

export default function ExportDialog(props: {
  mapId: string;
  rois: ExportRequest["rois"];
  /** The panel's client-side count, shown beside the authoritative one. */
  liveTotal: number;
  onClose: () => void;
  onExported: () => void;
}) {
  const [listing, setListing] = useState<DirectoryListing | null>(null);
  const [newMapId, setNewMapId] = useState(`${props.mapId}-roi`);
  const [newFolderName, setNewFolderName] = useState("");
  const [includeArtifacts, setIncludeArtifacts] = useState(false);
  const [registerInConfig, setRegisterInConfig] = useState(true);
  const [preview, setPreview] = useState<ExportPreview | null>(null);
  const [job, setJob] = useState<ExportJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const navigate = useCallback(
    async (path: string | null) => {
      try {
        setListing(await browseDirectories(props.mapId, path));
        setError(null);
      } catch (caught) {
        setError(errorText(caught));
      }
    },
    [props.mapId],
  );

  useEffect(() => {
    void navigate(null);
  }, [navigate]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const result = await previewExport(props.mapId, props.rois);
        if (!cancelled) {
          setPreview(result);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(errorText(caught));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.rois]);

  // Held in a ref so the poll below depends on the job state alone. Depending on
  // `props` would tear the interval down and rebuild it on every render.
  const onExported = useRef(props.onExported);
  onExported.current = props.onExported;

  // Poll while the export runs; the route answers 202 and the work happens in a
  // spawned python process, exactly as the benchmark runner does.
  const mapId = props.mapId;
  useEffect(() => {
    if (job?.state !== "running") {
      return;
    }
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const next = await fetchExportStatus(mapId);
          if (next) {
            setJob(next);
            if (next.state === "succeeded") {
              onExported.current();
            }
          }
        } catch (caught) {
          setError(errorText(caught));
        }
      })();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [job?.state, mapId]);

  const destination = useMemo(
    () => (listing ? `${listing.path}/${newMapId}` : ""),
    [listing, newMapId],
  );

  const countMismatch =
    preview !== null && preview.total !== props.liveTotal
      ? preview.total - props.liveTotal
      : 0;

  const onCreateFolder = async () => {
    if (!listing || !newFolderName.trim()) {
      return;
    }
    try {
      const created = await createDirectory(
        props.mapId,
        listing.path,
        newFolderName,
      );
      setNewFolderName("");
      await navigate(created.path);
    } catch (caught) {
      setError(errorText(caught));
    }
  };

  const onExport = async () => {
    if (!listing) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setJob(
        await startExport(props.mapId, {
          map_id: newMapId,
          destination_path: destination,
          rois: props.rois,
          include_artifacts: includeArtifacts,
          register_in_config: registerInConfig,
        }),
      );
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  };

  const done = job?.state === "succeeded";
  const failed = job?.state === "failed";

  return (
    <div className="export-dialog-backdrop" role="dialog" aria-modal="true">
      <div className="export-dialog">
        <header className="export-dialog-header">
          <h3>Export as a new map</h3>
          <button
            type="button"
            className="icon-button"
            aria-label="Close"
            onClick={props.onClose}
          >
            ×
          </button>
        </header>

        {error ? <p className="error-box">{error}</p> : null}

        {done ? (
          <div className="export-dialog-done">
            <p className="info-box">
              Exported {String(job?.result?.keyframe_count ?? "")} keyframes to{" "}
              <code>{String(job?.result?.path ?? destination)}</code>.
            </p>
            {job?.configWritten ? (
              <p className="muted">The map was added to the toolbox config.</p>
            ) : job?.configSnippet ? (
              <>
                <p className="muted">
                  The config could not be edited automatically. Paste this into
                  its <code>maps</code> array:
                </p>
                <pre className="export-dialog-snippet">{job.configSnippet}</pre>
              </>
            ) : null}
            <button
              type="button"
              className="primary-button"
              onClick={props.onClose}
            >
              Close
            </button>
          </div>
        ) : (
          <>
            <section className="export-dialog-browser">
              <div className="export-dialog-breadcrumb">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!listing?.parent}
                  onClick={() => void navigate(listing?.parent ?? null)}
                >
                  ↑ Up
                </button>
                <code>{listing?.path ?? "…"}</code>
              </div>
              <ul className="export-dialog-dirs">
                {listing?.entries.length === 0 ? (
                  <li className="muted">No sub-folder here.</li>
                ) : (
                  listing?.entries.map((entry) => (
                    <li key={entry.path}>
                      <button
                        type="button"
                        onClick={() => void navigate(entry.path)}
                      >
                        📁 {entry.name}
                      </button>
                    </li>
                  ))
                )}
              </ul>
              <div className="export-dialog-newfolder">
                <input
                  type="text"
                  value={newFolderName}
                  placeholder="New folder name"
                  onChange={(event) => setNewFolderName(event.target.value)}
                />
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!newFolderName.trim()}
                  onClick={() => void onCreateFolder()}
                >
                  Create
                </button>
              </div>
            </section>

            <section className="export-dialog-fields">
              <label>
                <span>Map id</span>
                <input
                  type="text"
                  value={newMapId}
                  onChange={(event) => setNewMapId(event.target.value)}
                />
              </label>
              <p className="muted">
                Destination: <code>{destination || "…"}</code>
              </p>

              <p className="export-dialog-count">
                Keyframes to export:{" "}
                <strong>{preview ? preview.total : "…"}</strong>
                {countMismatch !== 0 ? (
                  <span className="export-dialog-mismatch">
                    {" "}
                    ({countMismatch > 0 ? "+" : ""}
                    {countMismatch} vs the map's {props.liveTotal} — the
                    manifest's altitude bands and the map's floors disagree)
                  </span>
                ) : null}
              </p>
              <p className="muted">
                New <code>geo_ref_id</code>:{" "}
                {preview ? preview.geo_ref_id : "…"}
                {preview && !preview.geo_ref_id_allocated_against_database
                  ? " (database unreachable; allocated from the configured manifests alone)"
                  : ""}
              </p>

              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={includeArtifacts}
                  onChange={(event) =>
                    setIncludeArtifacts(event.target.checked)
                  }
                />
                Include the object-search artifacts (skips re-running prepare)
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={registerInConfig}
                  onChange={(event) =>
                    setRegisterInConfig(event.target.checked)
                  }
                />
                Add the map to the toolbox config
              </label>
            </section>

            {job?.state === "running" ? (
              <p className="muted">
                Exporting… {job.step} {job.detail}
              </p>
            ) : null}
            {failed ? <p className="error-box">{job?.error}</p> : null}

            <footer className="export-dialog-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={props.onClose}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary-button"
                disabled={
                  busy ||
                  job?.state === "running" ||
                  !listing ||
                  !newMapId.trim() ||
                  preview?.total === 0
                }
                onClick={() => void onExport()}
              >
                Export
              </button>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}
