/**
 * The selection session: the regions drawn so far, across floors.
 *
 * Persisted in `localStorage` per map, for the same reason the matching basket is
 * (`matching/basket.ts`): `App.tsx` mounts one panel at a time, so navigating away
 * unmounts the Explorer, and `sessionStorage` is per browser tab — a session that
 * only lived in component state would evaporate on the first tab switch, in the
 * middle of the very gesture it exists to support.
 */

import { useCallback, useEffect, useState } from "react";

import type { RoiPolygon } from "../annotations/types";
import type { ExportSessionState, RoiEntry } from "./types";

const STORAGE_PREFIX = "object-search-gui.exportRoiSession.v1";

const EMPTY: ExportSessionState = { active: false, rois: [] };

function storageKey(mapId: string): string {
  return `${STORAGE_PREFIX}.${mapId}`;
}

function readStored(mapId: string): ExportSessionState {
  try {
    const raw = localStorage.getItem(storageKey(mapId));
    if (!raw) {
      return EMPTY;
    }
    const parsed = JSON.parse(raw) as Partial<ExportSessionState>;
    return {
      active: Boolean(parsed.active),
      rois: Array.isArray(parsed.rois) ? (parsed.rois as RoiEntry[]) : [],
    };
  } catch {
    return EMPTY;
  }
}

export type ExportSession = {
  active: boolean;
  rois: RoiEntry[];
  /** Ids the list has ticked, for the grouped delete. */
  selectedIds: Set<string>;
  start: () => void;
  stop: () => void;
  addRoi: (ring: RoiPolygon, level: string | null) => void;
  removeRoi: (id: string) => void;
  removeSelected: () => void;
  toggleSelected: (id: string) => void;
  setAllSelected: (selected: boolean) => void;
  clear: () => void;
};

export function useExportSession(mapId: string): ExportSession {
  const [state, setState] = useState<ExportSessionState>(() =>
    readStored(mapId),
  );
  // Selection is deliberately not persisted: it is a gesture in progress, and a
  // reload that restored ticked rows next to a delete button would be a trap.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    setState(readStored(mapId));
    setSelectedIds(new Set());
    const onStorage = (event: StorageEvent) => {
      if (event.key === storageKey(mapId)) {
        setState(readStored(mapId));
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [mapId]);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey(mapId), JSON.stringify(state));
    } catch {
      // A full or disabled localStorage costs the persistence, not the feature.
    }
  }, [mapId, state]);

  const start = useCallback(() => {
    setState((current) => ({ ...current, active: true }));
  }, []);

  const stop = useCallback(() => {
    setState((current) => ({ ...current, active: false }));
  }, []);

  const addRoi = useCallback((ring: RoiPolygon, level: string | null) => {
    if (ring.length < 3) {
      return;
    }
    setState((current) => ({
      ...current,
      rois: [
        ...current.rois,
        {
          id: `roi-${Date.now()}-${current.rois.length}`,
          ring,
          level,
          createdAt: Date.now(),
        },
      ],
    }));
  }, []);

  const removeRoi = useCallback((id: string) => {
    setState((current) => ({
      ...current,
      rois: current.rois.filter((roi) => roi.id !== id),
    }));
    setSelectedIds((current) => {
      if (!current.has(id)) {
        return current;
      }
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  }, []);

  const removeSelected = useCallback(() => {
    setSelectedIds((selected) => {
      setState((current) => ({
        ...current,
        rois: current.rois.filter((roi) => !selected.has(roi.id)),
      }));
      return new Set();
    });
  }, []);

  const toggleSelected = useCallback((id: string) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const setAllSelected = useCallback((selected: boolean) => {
    setState((current) => {
      setSelectedIds(
        selected ? new Set(current.rois.map((roi) => roi.id)) : new Set(),
      );
      return current;
    });
  }, []);

  const clear = useCallback(() => {
    setState({ active: false, rois: [] });
    setSelectedIds(new Set());
  }, []);

  return {
    active: state.active,
    rois: state.rois,
    selectedIds,
    start,
    stop,
    addRoi,
    removeRoi,
    removeSelected,
    toggleSelected,
    setAllSelected,
    clear,
  };
}
