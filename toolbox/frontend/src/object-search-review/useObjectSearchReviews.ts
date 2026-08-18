import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { ObjectLocalization, ObjectObservation } from "../object-search/types";
import {
  fetchDetectionReviews,
  type ReviewStatus,
  setDetectionReview,
} from "./api";

type ReviewChange = {
  targetId: number;
  previousStatus: ReviewStatus | null;
  status: ReviewStatus | null;
};

type ReviewAction = {
  changes: ReviewChange[];
};

type ReviewRecord = { status: ReviewStatus; seq: number };

export type ReviewAnnotation = {
  targetId: number;
  status: ReviewStatus;
  inResults: boolean;
};

const MAX_HISTORY_LENGTH = 100;

function isEditableElement(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      target.tagName === "INPUT" ||
      target.tagName === "TEXTAREA" ||
      target.tagName === "SELECT")
  );
}

function uniqueTargetIds(observations: ObjectObservation[]): number[] {
  return [...new Set(observations.map((observation) => observation.objectIdx))];
}

export type ObjectSearchReviews = {
  reviews: ReadonlyMap<number, ReviewStatus>;
  annotations: ReviewAnnotation[];
  error: string | null;
  isLoading: boolean;
  canUndo: boolean;
  canRedo: boolean;
  reviewedCount: number;
  truePositiveCount: number;
  falsePositiveCount: number;
  observationStatus: (observation: ObjectObservation) => ReviewStatus | null;
  clusterStatus: (localization: ObjectLocalization) => ReviewStatus | null;
  setObservationStatus: (
    observation: ObjectObservation,
    status: ReviewStatus,
  ) => void;
  setClusterStatus: (localization: ObjectLocalization, status: ReviewStatus) => void;
  clearAnnotation: (targetId: number) => void;
  undo: () => void;
  redo: () => void;
};

export function useObjectSearchReviews(options: {
  enabled: boolean;
  mapId: string;
  query: string;
  localizations: ObjectLocalization[];
}): ObjectSearchReviews {
  const [records, setRecords] = useState<Map<number, ReviewRecord>>(new Map());
  const [undoStack, setUndoStack] = useState<ReviewAction[]>([]);
  const [redoStack, setRedoStack] = useState<ReviewAction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextSeq = useRef(0);

  const targetIds = useMemo(
    () =>
      new Set(
        options.localizations.flatMap((localization) =>
          localization.observations.map((observation) => observation.objectIdx),
        ),
      ),
    [options.localizations],
  );

  // Reloads on map/query only — deliberately NOT on the current results, so an
  // annotation that empties the result list can still be undone from the history.
  useEffect(() => {
    setUndoStack([]);
    setRedoStack([]);
    setError(null);
    nextSeq.current = 0;
    // Dropped synchronously, before the fetch: the records still in state belong
    // to the *previous* query, and every mutation below writes with the new one.
    // Left in place they stay clickable for the length of the round-trip, and a
    // delete would target a (query, target_id) pair that does not exist — the row
    // vanishes from the UI while surviving in the store.
    setRecords(new Map());
    if (!options.enabled || !options.query) {
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    fetchDetectionReviews(options.mapId, options.query)
      .then((items) => {
        if (cancelled) {
          return;
        }
        nextSeq.current = items.length;
        setRecords(
          new Map(
            items.map((item, index) => [
              item.targetId,
              { status: item.status, seq: index + 1 },
            ]),
          ),
        );
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setRecords(new Map());
          setError(cause instanceof Error ? cause.message : String(cause));
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
  }, [options.enabled, options.mapId, options.query]);

  const reviews = useMemo(
    () =>
      new Map(
        [...records].map(([targetId, record]) => [targetId, record.status]),
      ),
    [records],
  );

  const annotations = useMemo(
    () =>
      [...records]
        .map(([targetId, record]) => ({
          targetId,
          status: record.status,
          inResults: targetIds.has(targetId),
          seq: record.seq,
        }))
        .sort((left, right) => right.seq - left.seq)
        .map(({ targetId, status, inResults }) => ({ targetId, status, inResults })),
    [records, targetIds],
  );

  const persistChanges = useCallback(
    (changes: ReviewChange[], usePreviousStatus: boolean): void => {
      if (!options.query) {
        return;
      }
      void Promise.all(
        changes.map((change) =>
          setDetectionReview(
            options.mapId,
            options.query,
            change.targetId,
            usePreviousStatus ? change.previousStatus : change.status,
          ),
        ),
      ).catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : String(cause));
      });
    },
    [options.mapId, options.query],
  );

  const applyChanges = useCallback(
    (changes: ReviewChange[], usePreviousStatus: boolean): void => {
      const next = new Map(records);
      for (const change of changes) {
        const status = usePreviousStatus ? change.previousStatus : change.status;
        if (status === null) {
          next.delete(change.targetId);
        } else {
          next.set(change.targetId, {
            status,
            seq: (nextSeq.current += 1),
          });
        }
      }
      setRecords(next);
      persistChanges(changes, usePreviousStatus);
    },
    [persistChanges, records],
  );

  const commit = useCallback(
    (changes: ReviewChange[]): void => {
      if (!changes.length || !options.query) {
        return;
      }
      const action = { changes };
      applyChanges(changes, false);
      setUndoStack((current) => [...current.slice(-(MAX_HISTORY_LENGTH - 1)), action]);
      setRedoStack([]);
    },
    [applyChanges, options.query],
  );

  const setTargetsStatus = useCallback(
    (targetIdsToChange: number[], requestedStatus: ReviewStatus): void => {
      const currentStatuses = targetIdsToChange.map(
        (targetId) => reviews.get(targetId) ?? null,
      );
      const status = currentStatuses.every((current) => current === requestedStatus)
        ? null
        : requestedStatus;
      commit(
        targetIdsToChange.map((targetId, index) => ({
          targetId,
          previousStatus: currentStatuses[index],
          status,
        })),
      );
    },
    [commit, reviews],
  );

  const undo = useCallback((): void => {
    const action = undoStack.at(-1);
    if (!action) {
      return;
    }
    applyChanges(action.changes, true);
    setUndoStack(undoStack.slice(0, -1));
    setRedoStack((current) => [
      ...current.slice(-(MAX_HISTORY_LENGTH - 1)),
      action,
    ]);
  }, [applyChanges, undoStack]);

  const redo = useCallback((): void => {
    const action = redoStack.at(-1);
    if (!action) {
      return;
    }
    applyChanges(action.changes, false);
    setRedoStack(redoStack.slice(0, -1));
    setUndoStack((current) => [
      ...current.slice(-(MAX_HISTORY_LENGTH - 1)),
      action,
    ]);
  }, [applyChanges, redoStack]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (isEditableElement(event.target)) {
        return;
      }
      const key = event.key.toLowerCase();
      const hasModifier = event.ctrlKey || event.metaKey;
      const isUndo = hasModifier && key === "z" && !event.shiftKey;
      const isRedo =
        (hasModifier && key === "z" && event.shiftKey) ||
        (hasModifier && key === "y");
      if (isUndo && undoStack.length) {
        event.preventDefault();
        undo();
      } else if (isRedo && redoStack.length) {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [redo, redoStack.length, undo, undoStack.length]);

  const counts = useMemo(() => {
    // Counts cover every annotation for the query, including hidden results.
    let truePositiveCount = 0;
    let falsePositiveCount = 0;
    for (const status of reviews.values()) {
      if (status === "true_positive") {
        truePositiveCount += 1;
      } else {
        falsePositiveCount += 1;
      }
    }
    return { truePositiveCount, falsePositiveCount };
  }, [reviews]);

  return {
    reviews,
    annotations,
    error,
    isLoading,
    canUndo: undoStack.length > 0,
    canRedo: redoStack.length > 0,
    reviewedCount: reviews.size,
    ...counts,
    observationStatus: (observation) => reviews.get(observation.objectIdx) ?? null,
    clusterStatus: (localization) => {
      const ids = uniqueTargetIds(localization.observations);
      if (!ids.length) {
        return null;
      }
      const first = reviews.get(ids[0]);
      return first !== undefined && ids.every((targetId) => reviews.get(targetId) === first)
        ? first
        : null;
    },
    setObservationStatus: (observation, status) =>
      setTargetsStatus([observation.objectIdx], status),
    setClusterStatus: (localization, status) =>
      setTargetsStatus(uniqueTargetIds(localization.observations), status),
    clearAnnotation: (targetId) => {
      const previousStatus = reviews.get(targetId);
      if (previousStatus !== undefined) {
        commit([{ targetId, previousStatus, status: null }]);
      }
    },
    undo,
    redo,
  };
}
