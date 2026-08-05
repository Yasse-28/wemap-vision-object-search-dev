import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";

const STORAGE_KEY = "object-search-gui.explorerLivemapHeight";
export const LIVEMAP_FRAME_DEFAULT_HEIGHT = 360;
const LIVEMAP_FRAME_MIN_HEIGHT = 200;
const LIVEMAP_FRAME_MAX_HEIGHT = 720;

function clampHeight(value: number): number {
  return Math.min(
    LIVEMAP_FRAME_MAX_HEIGHT,
    Math.max(LIVEMAP_FRAME_MIN_HEIGHT, Math.round(value)),
  );
}

function readStoredHeight(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) {
      return LIVEMAP_FRAME_DEFAULT_HEIGHT;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      return LIVEMAP_FRAME_DEFAULT_HEIGHT;
    }
    return clampHeight(value);
  } catch {
    return LIVEMAP_FRAME_DEFAULT_HEIGHT;
  }
}

export function useLivemapFrameHeight() {
  const [height, setHeight] = useState(readStoredHeight);
  const [isDragging, setIsDragging] = useState(false);
  const heightRef = useRef(height);
  heightRef.current = height;

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(height));
    } catch {
      /* ignore */
    }
  }, [height]);

  const startResize = useCallback((event: PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const handle = event.currentTarget;
    const pointerId = event.pointerId;
    const startY = event.clientY;
    const startHeight = heightRef.current;
    setIsDragging(true);
    handle.setPointerCapture(pointerId);

    function finishResize() {
      setIsDragging(false);
      document.body.classList.remove("explorer-livemap-dragging");
      if (handle.hasPointerCapture(pointerId)) {
        handle.releasePointerCapture(pointerId);
      }
      handle.removeEventListener("pointermove", onPointerMove);
      handle.removeEventListener("pointerup", onPointerUp);
      handle.removeEventListener("pointercancel", onPointerUp);
    }

    function onPointerMove(moveEvent: globalThis.PointerEvent) {
      if (moveEvent.pointerId !== pointerId) {
        return;
      }
      setHeight(clampHeight(startHeight + (moveEvent.clientY - startY)));
    }

    function onPointerUp(upEvent: globalThis.PointerEvent) {
      if (upEvent.pointerId !== pointerId) {
        return;
      }
      finishResize();
    }

    document.body.classList.add("explorer-livemap-dragging");
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);
  }, []);

  return { height, isDragging, startResize };
}
