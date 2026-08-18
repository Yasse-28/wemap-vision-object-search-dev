/**
 * Confirmation for the one destructive action in the toolbox.
 *
 * It lists what will go before it asks, and the button stays inert until the map's
 * id is retyped exactly. The backend re-checks that id, so this is the ergonomic
 * half of the gate, not the whole of it.
 */

import { useEffect, useState } from "react";

import { ApiError } from "../http";
import { deleteMap, fetchDeletionPreview } from "./api";
import type { DeletionPreview } from "./types";

function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    return error.detail;
  }
  return error instanceof Error ? error.message : String(error);
}

function countLabel(value: number | null): string {
  return value === null ? "unknown (database unreachable)" : String(value);
}

export default function DeleteMapDialog(props: {
  mapId: string;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [preview, setPreview] = useState<DeletionPreview | null>(null);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const result = await fetchDeletionPreview(props.mapId);
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
  }, [props.mapId]);

  const onConfirm = async () => {
    setBusy(true);
    setError(null);
    try {
      await deleteMap(props.mapId);
      props.onDeleted();
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="export-dialog-backdrop" role="dialog" aria-modal="true">
      <div className="export-dialog">
        <header className="export-dialog-header">
          <h3>Delete {props.mapId}</h3>
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

        {preview ? (
          <>
            <p className="muted">This cannot be undone. It removes:</p>
            <ul className="delete-map-list">
              <li>
                {preview.removes_directory ? (
                  <>
                    the directory <code>{preview.path}</code> (
                    {countLabel(preview.keyframe_count)} keyframes, with their
                    images and depths)
                  </>
                ) : (
                  <>
                    <em>not</em> the directory <code>{preview.path}</code> — it
                    carries no export provenance, so it is left alone
                  </>
                )}
              </li>
              <li>
                {countLabel(preview.candidate_rows)} candidate rows and{" "}
                {countLabel(preview.geokeyframe_rows)} keyframe rows under{" "}
                <code>geo_ref_id {preview.geo_ref_id ?? "?"}</code>
              </li>
              {preview.index_name ? (
                <li>
                  the partial HNSW index <code>{preview.index_name}</code>
                </li>
              ) : null}
              <li>its entry in the toolbox config</li>
            </ul>

            {preview.warnings.map((warning) => (
              <p key={warning} className="info-box">
                {warning}
              </p>
            ))}

            <label className="delete-map-confirm">
              <span>
                Type <code>{props.mapId}</code> to confirm
              </span>
              <input
                type="text"
                value={typed}
                autoComplete="off"
                onChange={(event) => setTyped(event.target.value)}
              />
            </label>
          </>
        ) : error ? null : (
          <p className="muted">Checking what would be deleted…</p>
        )}

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
            className="primary-button is-danger"
            disabled={busy || !preview || typed !== props.mapId}
            onClick={() => void onConfirm()}
          >
            Delete permanently
          </button>
        </footer>
      </div>
    </div>
  );
}
