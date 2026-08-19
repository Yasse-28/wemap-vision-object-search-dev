/**
 * What travels from the Explorer to the Annotation tab.
 *
 * `App.tsx` mounts one panel at a time, so the two tabs share no React state and a
 * reload lands on whichever tab the URL names. The handoff therefore lives in
 * `localStorage`, per map — the same reason `matching/basket.ts` and
 * `export-roi/session.ts` do. It stays small on purpose: an id, a queue of row
 * indices, and the click that produced a depth pin. Everything else (the row, its
 * angles, the panorama) is re-fetched from `/object-search-metadata/*`, which is one
 * request and cannot go stale the way a copy would.
 */

const STORAGE_PREFIX = "object-search-gui.annotationHandoff";

/** The click the Explorer resolved into a depth pin, replayed by the tab. */
export type HandoffPin = {
  projection: "erp" | "cutout";
  xRatio: number;
  yRatio: number;
};

export type AnnotationHandoff = {
  keyframeId: string | null;
  /** The proposal being annotated; null when the point came from the panorama. */
  rowIndex: number | null;
  /** Row indices queued from the Explorer, in the order they were sent. */
  queue: number[];
  /** Rows already validated in this queue, so the tab can show what is left. */
  done: number[];
  pin: HandoffPin | null;
};

export const EMPTY_HANDOFF: AnnotationHandoff = {
  keyframeId: null,
  rowIndex: null,
  queue: [],
  done: [],
  pin: null,
};

function storageKey(mapId: string): string {
  return `${STORAGE_PREFIX}.${mapId}`;
}

function isFiniteRatio(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function parsePin(value: unknown): HandoffPin | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as Record<string, unknown>;
  if (candidate.projection !== "erp" && candidate.projection !== "cutout") {
    return null;
  }
  if (!isFiniteRatio(candidate.xRatio) || !isFiniteRatio(candidate.yRatio)) {
    return null;
  }
  return {
    projection: candidate.projection,
    xRatio: candidate.xRatio,
    yRatio: candidate.yRatio,
  };
}

function parseRowList(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<number>();
  for (const item of value) {
    if (typeof item === "number" && Number.isInteger(item) && item >= 0) {
      seen.add(item);
    }
  }
  return [...seen];
}

/** Tolerates anything: a hand-edited entry must not break the tab on arrival. */
export function parseHandoff(raw: string | null): AnnotationHandoff {
  if (!raw) {
    return EMPTY_HANDOFF;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return EMPTY_HANDOFF;
  }
  if (!parsed || typeof parsed !== "object") {
    return EMPTY_HANDOFF;
  }
  const candidate = parsed as Record<string, unknown>;
  const queue = parseRowList(candidate.queue);
  const done = parseRowList(candidate.done).filter((row) => queue.includes(row));
  const rowIndex =
    typeof candidate.rowIndex === "number" && Number.isInteger(candidate.rowIndex)
      ? candidate.rowIndex
      : null;
  return {
    keyframeId: typeof candidate.keyframeId === "string" ? candidate.keyframeId : null,
    rowIndex,
    queue,
    done,
    pin: parsePin(candidate.pin),
  };
}

/**
 * Send one proposal over. Re-sending a queued row re-selects it rather than
 * duplicating it — the Explorer's button is a "go here", not an "add one more".
 */
export function withProposal(
  current: AnnotationHandoff,
  proposal: { keyframeId: string; rowIndex: number; pin?: HandoffPin | null },
): AnnotationHandoff {
  const sameKeyframe = current.keyframeId === proposal.keyframeId;
  const queue = sameKeyframe ? current.queue : [];
  const done = sameKeyframe ? current.done : [];
  return {
    keyframeId: proposal.keyframeId,
    rowIndex: proposal.rowIndex,
    queue: queue.includes(proposal.rowIndex) ? queue : [...queue, proposal.rowIndex],
    done: done.filter((row) => row !== proposal.rowIndex),
    pin: proposal.pin ?? null,
  };
}

/** A panorama point carries no proposal: the queue is left alone. */
export function withPanoramaPoint(
  current: AnnotationHandoff,
  point: { keyframeId: string; pin: HandoffPin },
): AnnotationHandoff {
  const sameKeyframe = current.keyframeId === point.keyframeId;
  return {
    keyframeId: point.keyframeId,
    rowIndex: null,
    queue: sameKeyframe ? current.queue : [],
    done: sameKeyframe ? current.done : [],
    pin: point.pin,
  };
}

/**
 * Move to another keyframe from the map. The queue belonged to the old turn, so it
 * goes with it; nothing is auto-selected, because which proposal matters is a
 * question the annotator answers on the ribbon.
 */
export function withKeyframe(
  current: AnnotationHandoff,
  keyframeId: string,
): AnnotationHandoff {
  if (current.keyframeId === keyframeId) {
    return current;
  }
  return { keyframeId, rowIndex: null, queue: [], done: [], pin: null };
}

export function withSelection(
  current: AnnotationHandoff,
  rowIndex: number,
): AnnotationHandoff {
  return { ...current, rowIndex, pin: null };
}

/** Mark the current row validated and move to the first row still to do. */
export function withValidated(current: AnnotationHandoff): AnnotationHandoff {
  if (current.rowIndex === null) {
    return current;
  }
  const done = current.done.includes(current.rowIndex)
    ? current.done
    : [...current.done, current.rowIndex];
  const next = current.queue.find((row) => !done.includes(row)) ?? null;
  return { ...current, done, rowIndex: next ?? current.rowIndex, pin: null };
}

/** Skip without validating: the row stays in the queue, we just look past it. */
export function withSkipped(current: AnnotationHandoff): AnnotationHandoff {
  if (current.rowIndex === null) {
    return current;
  }
  const position = current.queue.indexOf(current.rowIndex);
  if (position < 0) {
    return current;
  }
  const after = current.queue.slice(position + 1);
  const next =
    after.find((row) => !current.done.includes(row)) ??
    current.queue.find((row) => row !== current.rowIndex && !current.done.includes(row));
  return next == null ? current : { ...current, rowIndex: next, pin: null };
}

export function readHandoff(mapId: string): AnnotationHandoff {
  try {
    return parseHandoff(localStorage.getItem(storageKey(mapId)));
  } catch {
    return EMPTY_HANDOFF;
  }
}

export function writeHandoff(mapId: string, handoff: AnnotationHandoff): void {
  try {
    localStorage.setItem(storageKey(mapId), JSON.stringify(handoff));
  } catch {
    /* A full or disabled store costs the queue, not the annotation. */
  }
}

export function clearHandoff(mapId: string): void {
  try {
    localStorage.removeItem(storageKey(mapId));
  } catch {
    /* ignore */
  }
}
