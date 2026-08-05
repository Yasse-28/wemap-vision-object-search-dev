import { useCallback, useEffect, useRef, useState } from "react";

const SIZE_STORAGE_KEY = "object-search-gui.promptBoxSize";
const POSITION_STORAGE_KEY = "object-search-gui.promptBoxPosition";

const DEFAULT_WIDTH = 760;
const DEFAULT_TOP = 20;
const MIN_WIDTH = 420;
const MIN_HEIGHT = 120;
const MAX_HEIGHT_RATIO = 0.75;
const EDGE_GAP = 12;

type PromptBoxSize = {
  width: number;
  height: number | null;
};

type PromptBoxPosition = {
  x: number;
  y: number;
};

type ResizeEdge = "bottom" | "right" | "left" | "corner";

function maxWidth(): number {
  return Math.max(MIN_WIDTH, window.innerWidth - 48);
}

function maxHeight(): number {
  return Math.max(MIN_HEIGHT, Math.round(window.innerHeight * MAX_HEIGHT_RATIO));
}

function clampWidth(value: number): number {
  return Math.min(maxWidth(), Math.max(MIN_WIDTH, Math.round(value)));
}

function clampHeight(value: number): number {
  return Math.min(maxHeight(), Math.max(MIN_HEIGHT, Math.round(value)));
}

function centeredX(width: number): number {
  return Math.round((window.innerWidth - width) / 2);
}

function clampPosition(
  position: PromptBoxPosition,
  width = DEFAULT_WIDTH,
  height = MIN_HEIGHT,
): PromptBoxPosition {
  const maxX = Math.max(EDGE_GAP, window.innerWidth - width - EDGE_GAP);
  const maxY = Math.max(EDGE_GAP, window.innerHeight - height - EDGE_GAP);
  return {
    x: Math.min(maxX, Math.max(EDGE_GAP, Math.round(position.x))),
    y: Math.min(maxY, Math.max(EDGE_GAP, Math.round(position.y))),
  };
}

function readStoredSize(): PromptBoxSize {
  try {
    const raw = localStorage.getItem(SIZE_STORAGE_KEY);
    if (!raw) {
      return { width: DEFAULT_WIDTH, height: null };
    }
    const parsed = JSON.parse(raw) as PromptBoxSize;
    return {
      width: clampWidth(parsed.width ?? DEFAULT_WIDTH),
      height:
        parsed.height === null || parsed.height === undefined
          ? null
          : clampHeight(parsed.height),
    };
  } catch {
    return { width: DEFAULT_WIDTH, height: null };
  }
}

function readStoredPosition(size: PromptBoxSize): PromptBoxPosition {
  try {
    const raw = localStorage.getItem(POSITION_STORAGE_KEY);
    if (!raw) {
      return clampPosition({ x: centeredX(size.width), y: DEFAULT_TOP }, size.width);
    }
    const parsed = JSON.parse(raw) as PromptBoxPosition;
    return clampPosition(
      {
        x: parsed.x ?? centeredX(size.width),
        y: parsed.y ?? DEFAULT_TOP,
      },
      size.width,
    );
  } catch {
    return clampPosition({ x: centeredX(size.width), y: DEFAULT_TOP }, size.width);
  }
}

export function usePromptBoxResize() {
  const [size, setSize] = useState<PromptBoxSize>(readStoredSize);
  const [position, setPosition] = useState<PromptBoxPosition>(() =>
    readStoredPosition(size),
  );
  const [isResizing, setIsResizing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [activeEdge, setActiveEdge] = useState<ResizeEdge | null>(null);
  const sizeRef = useRef(size);
  const positionRef = useRef(position);
  sizeRef.current = size;
  positionRef.current = position;

  useEffect(() => {
    try {
      localStorage.setItem(SIZE_STORAGE_KEY, JSON.stringify(size));
    } catch {
      /* ignore */
    }
  }, [size]);

  useEffect(() => {
    try {
      localStorage.setItem(POSITION_STORAGE_KEY, JSON.stringify(position));
    } catch {
      /* ignore */
    }
  }, [position]);

  useEffect(() => {
    function onWindowResize() {
      setSize((current) => ({
        width: clampWidth(current.width),
        height:
          current.height === null ? null : clampHeight(current.height),
      }));
      setPosition((current) =>
        clampPosition(current, sizeRef.current.width, sizeRef.current.height ?? MIN_HEIGHT),
      );
    }
    window.addEventListener("resize", onWindowResize);
    return () => window.removeEventListener("resize", onWindowResize);
  }, []);

  const startResize = useCallback((edge: ResizeEdge, event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const handle = event.currentTarget;
    const pointerId = event.pointerId;
    const startX = event.clientX;
    const startY = event.clientY;
    const startWidth = sizeRef.current.width;
    const box = handle.closest(".object-search-prompt-box") as HTMLElement | null;
    const startHeight =
      sizeRef.current.height ?? box?.getBoundingClientRect().height ?? MIN_HEIGHT;

    setIsResizing(true);
    setActiveEdge(edge);
    handle.setPointerCapture(pointerId);
    document.body.classList.add("prompt-box-resizing");

    function finishResize() {
      setIsResizing(false);
      setActiveEdge(null);
      document.body.classList.remove("prompt-box-resizing");
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
      const deltaX = moveEvent.clientX - startX;
      const deltaY = moveEvent.clientY - startY;
      let nextWidth = startWidth;
      let nextHeight = startHeight;

      if (edge === "right" || edge === "corner") {
        nextWidth = startWidth + deltaX;
      }
      if (edge === "left") {
        nextWidth = startWidth - deltaX;
      }
      if (edge === "bottom" || edge === "corner") {
        nextHeight = startHeight + deltaY;
      }

      const clampedSize = {
        width: clampWidth(nextWidth),
        height: clampHeight(nextHeight),
      };
      setSize(clampedSize);
      setPosition((current) =>
        clampPosition(current, clampedSize.width, clampedSize.height),
      );
    }

    function onPointerUp(upEvent: PointerEvent) {
      if (upEvent.pointerId !== pointerId) {
        return;
      }
      finishResize();
    }

    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);
  }, []);

  const startDrag = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const handle = event.currentTarget;
    const pointerId = event.pointerId;
    const box = handle.closest(".object-search-prompt-box") as HTMLElement | null;
    const rect = box?.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;
    const startPosition = positionRef.current;
    const boxWidth = rect?.width ?? sizeRef.current.width;
    const boxHeight = rect?.height ?? sizeRef.current.height ?? MIN_HEIGHT;

    setIsDragging(true);
    handle.setPointerCapture(pointerId);
    document.body.classList.add("prompt-box-dragging");

    function finishDrag() {
      setIsDragging(false);
      document.body.classList.remove("prompt-box-dragging");
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
      setPosition(
        clampPosition(
          {
            x: startPosition.x + moveEvent.clientX - startX,
            y: startPosition.y + moveEvent.clientY - startY,
          },
          boxWidth,
          boxHeight,
        ),
      );
    }

    function onPointerUp(upEvent: PointerEvent) {
      if (upEvent.pointerId !== pointerId) {
        return;
      }
      finishDrag();
    }

    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);
  }, []);

  const boxStyle = {
    width: size.width,
    height: size.height ?? undefined,
    minHeight: size.height === null ? undefined : MIN_HEIGHT,
  } as const;

  const overlayStyle = {
    left: position.x,
    top: position.y,
    transform: "none",
  } as const;

  return {
    size,
    position,
    boxStyle,
    overlayStyle,
    isResizing,
    isDragging,
    activeEdge,
    startResize,
    startDrag,
  };
}
