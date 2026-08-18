import { useRef, useState } from "react";
import type * as React from "react";

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

export default BboxDrawOverlay;

