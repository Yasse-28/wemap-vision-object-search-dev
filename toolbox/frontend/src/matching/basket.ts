/**
 * The matching basket: the set of detections an inspection is run over.
 *
 * The item is a **direction in a keyframe**, not a `row_index`. That is not a detail:
 * the panorama fills the basket from parquet rows, the results list fills it from
 * cluster observations, and pgvector carries no `row_index` — `(keyframe, theta, phi)`
 * is the only key both sides can produce. The parquet row is resolved lazily, when
 * something actually needs it.
 *
 * `rays` is a list because the basket has to survive SAM2: a detection contributes one
 * ray today (its box centre) and a mask's worth of them later. Anything consuming an
 * item must loop, even while the loop has one iteration.
 */

import { useCallback, useEffect, useState } from "react";

export type BasketRay = {
  /** Radians, `prepare/convention.py` — the same pair the parquet and pgvector store. */
  thetaCenter: number;
  phiCenter: number;
};

export type BasketItem = {
  /** `keyframe:theta:phi`, rounded — two sources naming one detection must collide. */
  key: string;
  keyframeId: string;
  rays: BasketRay[];
  /** Where it came from: a panorama box, a search cluster, or an annotated group. */
  source: "panorama" | "cluster" | "group";
  /** Rank of the cluster it came from, 1-based; null when it came from elsewhere. */
  clusterRank: number | null;
  /** Name of the annotated group it was loaded from, when that is the origin. */
  groupName?: string | null;
  /**
   * Where the cluster it came from was localized, captured on add.
   *
   * Stored rather than looked up later because the matching tab has no search
   * results of its own: by the time you judge the basket, the query that produced
   * the cluster may be long gone.
   */
  clusterPosition?: { lat: number; lng: number; alt: number; level: string | null };
  label: string | null;
  thumbnail: string | null;
  similarity: number | null;
};

const STORAGE_PREFIX = "object-search-gui.matchingBasket.v1";

/**
 * 4 decimals is ~0.006° — below the float16 rounding the angles already carry, so two
 * spellings of the same detection collide, and two real neighbours do not.
 */
export function basketKey(keyframeId: string, ray: BasketRay): string {
  return `${keyframeId}:${ray.thetaCenter.toFixed(4)}:${ray.phiCenter.toFixed(4)}`;
}

function storageKey(mapId: string): string {
  return `${STORAGE_PREFIX}.${mapId}`;
}

function readStored(mapId: string): BasketItem[] {
  try {
    const raw = localStorage.getItem(storageKey(mapId));
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as BasketItem[]) : [];
  } catch {
    return [];
  }
}

export type MatchingBasket = {
  items: BasketItem[];
  has: (key: string) => boolean;
  add: (items: BasketItem[]) => void;
  remove: (key: string) => void;
  clear: () => void;
  /** Fill in previews for items added without one (an annotated group carries none). */
  setThumbnails: (thumbnails: Map<string, string | null>) => void;
};

/**
 * Per map, and in `localStorage` rather than `sessionStorage`: filling the basket in
 * Object Search and judging it under Matching is a two-step gesture that people do
 * across browser tabs, and `sessionStorage` is per tab — the basket looked empty on
 * arrival. Scoped by map id because a keyframe id means something else elsewhere.
 *
 * `storage` events keep two open tabs on the same basket; without it, the tab that
 * did not do the write keeps showing a stale one.
 */
export function useMatchingBasket(mapId: string): MatchingBasket {
  const [items, setItems] = useState<BasketItem[]>(() => readStored(mapId));

  useEffect(() => {
    setItems(readStored(mapId));
    const onStorage = (event: StorageEvent) => {
      if (event.key === storageKey(mapId)) {
        setItems(readStored(mapId));
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [mapId]);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey(mapId), JSON.stringify(items));
    } catch {
      // A full or disabled localStorage costs the persistence, not the feature.
    }
  }, [mapId, items]);

  const add = useCallback((incoming: BasketItem[]) => {
    setItems((current) => {
      const known = new Set(current.map((item) => item.key));
      const fresh = incoming.filter((item) => !known.has(item.key));
      return fresh.length ? [...current, ...fresh] : current;
    });
  }, []);

  const remove = useCallback((key: string) => {
    setItems((current) => current.filter((item) => item.key !== key));
  }, []);

  const clear = useCallback(() => setItems([]), []);

  const setThumbnails = useCallback((thumbnails: Map<string, string | null>) => {
    setItems((current) => {
      let changed = false;
      const next = current.map((item) => {
        const thumbnail = thumbnails.get(item.key);
        if (!thumbnail || item.thumbnail === thumbnail) {
          return item;
        }
        changed = true;
        return { ...item, thumbnail };
      });
      // Returning `current` unchanged matters: this runs from an effect that reacts
      // to `items`, and a new array every time would loop.
      return changed ? next : current;
    });
  }, []);

  const has = useCallback(
    (key: string) => items.some((item) => item.key === key),
    [items],
  );

  return { items, has, add, remove, clear, setThumbnails };
}
