import { ReactNode, useCallback, useEffect, useRef, useState } from "react";

const STORAGE_WIDTH_KEY = "object-search-gui.featurePanelWidth";
const STORAGE_COLLAPSED_KEY = "object-search-gui.featurePanelCollapsed";
const DEFAULT_WIDTH = 280;
const MIN_WIDTH = 200;
const MAX_WIDTH = 480;

type Props = {
  onBack: () => void;
  children: ReactNode;
};

function readStoredWidth(): number {
  try {
    const raw = localStorage.getItem(STORAGE_WIDTH_KEY);
    if (raw === null) {
      return DEFAULT_WIDTH;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      return DEFAULT_WIDTH;
    }
    return clampWidth(value);
  } catch {
    return DEFAULT_WIDTH;
  }
}

function readStoredCollapsed(): boolean {
  try {
    return localStorage.getItem(STORAGE_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function clampWidth(value: number): number {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(value)));
}

function FeaturePanel(props: Props) {
  const [width, setWidth] = useState(readStoredWidth);
  const [collapsed, setCollapsed] = useState(readStoredCollapsed);
  const [isDragging, setIsDragging] = useState(false);
  const widthRef = useRef(width);
  widthRef.current = width;

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_WIDTH_KEY, String(width));
    } catch {
      /* ignore */
    }
  }, [width]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  const startResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (collapsed) {
      return;
    }
    event.preventDefault();
    const handle = event.currentTarget;
    const pointerId = event.pointerId;
    const startX = event.clientX;
    const startWidth = widthRef.current;
    setIsDragging(true);
    handle.setPointerCapture(pointerId);

    function finishResize() {
      setIsDragging(false);
      document.body.classList.remove("feature-panel-dragging");
      if (handle.hasPointerCapture(pointerId)) {
        handle.releasePointerCapture(pointerId);
      }
      handle.removeEventListener("pointermove", onPointerMove);
      handle.removeEventListener("pointerup", onPointerUp);
      handle.removeEventListener("pointercancel", onPointerUp);
    }

    function onPointerMove(moveEvent: PointerEvent) {
      if (moveEvent.pointerId !== pointerId) {
        return;
      }
      setWidth(clampWidth(startWidth + moveEvent.clientX - startX));
    }

    function onPointerUp(upEvent: PointerEvent) {
      if (upEvent.pointerId !== pointerId) {
        return;
      }
      finishResize();
    }

    document.body.classList.add("feature-panel-dragging");
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);
  }, [collapsed]);

  const panelWidth = collapsed ? 0 : width;

  return (
    <>
      <aside
        className={`feature-panel${collapsed ? " is-collapsed" : ""}${
          isDragging ? " is-resizing" : ""
        }`}
        style={{ width: panelWidth }}
        aria-hidden={collapsed}
      >
        <div className="feature-panel-toolbar">
          <button className="back-button" type="button" onClick={props.onBack}>
            Back to maps
          </button>
          <button
            type="button"
            className="feature-panel-toggle"
            onClick={() => setCollapsed(true)}
            aria-label="Hide sidebar"
            title="Hide sidebar"
          >
            ‹
          </button>
        </div>
        {props.children}
        <div
          className={`feature-panel-resizer${isDragging ? " is-dragging" : ""}`}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          aria-valuenow={width}
          aria-valuemin={MIN_WIDTH}
          aria-valuemax={MAX_WIDTH}
          title="Drag to resize sidebar"
          onPointerDown={startResize}
        />
      </aside>
      {collapsed ? (
        <button
          type="button"
          className="feature-panel-expand"
          onClick={() => setCollapsed(false)}
          aria-label="Show sidebar"
          title="Show sidebar"
        >
          ›
        </button>
      ) : null}
    </>
  );
}

export default FeaturePanel;
