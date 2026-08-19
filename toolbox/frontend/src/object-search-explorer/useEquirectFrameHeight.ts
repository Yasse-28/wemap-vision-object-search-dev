import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";

const STORAGE_KEY = "object-search-gui.explorerEquirectHeight";
export const EQUIRECT_FRAME_DEFAULT_HEIGHT = 280;
export const EQUIRECT_FRAME_MIN = 120;
export const EQUIRECT_FRAME_MAX = 600;
const EQUIRECT_FRAME_MIN_HEIGHT = 120;
const EQUIRECT_FRAME_MAX_HEIGHT = 600;

function clampHeight(value: number): number {
  return Math.min(
    EQUIRECT_FRAME_MAX_HEIGHT,
    Math.max(EQUIRECT_FRAME_MIN_HEIGHT, Math.round(value)),
  );
}

function readStoredHeight(storageKey: string, fallback: number): number {
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw === null) {
      return fallback;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      return fallback;
    }
    return clampHeight(value);
  } catch {
    return fallback;
  }
}

/**
 * Two panes use this — the Explorer's panorama and the Annotation capsule's — and
 * they are resized independently, so the caller names its own stored key. The
 * defaults are the Explorer's, which is what every existing caller gets.
 */
export function useEquirectFrameHeight(options?: {
  storageKey?: string;
  defaultHeight?: number;
}) {
  const storageKey = options?.storageKey ?? STORAGE_KEY;
  const fallback = clampHeight(options?.defaultHeight ?? EQUIRECT_FRAME_DEFAULT_HEIGHT);
  const [height, setHeight] = useState(() => readStoredHeight(storageKey, fallback));
  const [isDragging, setIsDragging] = useState(false);
  const heightRef = useRef(height);
  heightRef.current = height;

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, String(height));
    } catch {
      /* ignore */
    }
  }, [storageKey, height]);

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
      document.body.classList.remove("explorer-equirect-dragging");
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

    document.body.classList.add("explorer-equirect-dragging");
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);
  }, []);

  return { height, isDragging, startResize };
}
