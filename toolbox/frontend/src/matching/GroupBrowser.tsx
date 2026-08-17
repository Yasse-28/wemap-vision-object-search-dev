/**
 * Pick which members of an annotated group go into the basket.
 *
 * "Load the whole group" is the common case but not the only one: half the point of
 * the tool is asking whether *part* of a group holds up on its own, which needs the
 * members visible and individually selectable. Previews come from the matching
 * endpoint — an annotation stores a direction, not a crop.
 */

import { useEffect, useMemo, useState } from "react";

import { cutoutPreviewUrl } from "../object-search/utils";
import { runMatching } from "./api";
import type { GroupAnnotationTarget } from "./groupAnnotations";
import { groupAnnotationKey } from "./groupAnnotations";

export type GroupBrowserSelection = {
  target: GroupAnnotationTarget;
  thumbnail: string | null;
};

export function GroupBrowser(props: {
  mapId: string;
  groupName: string;
  members: GroupAnnotationTarget[];
  onClose: () => void;
  onAdd: (selection: GroupBrowserSelection[]) => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(props.members.map((member) => groupAnnotationKey(member))),
  );
  const [thumbnails, setThumbnails] = useState<Map<string, string | null>>(
    () => new Map(),
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") props.onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [props]);

  useEffect(() => {
    let cancelled = false;
    runMatching(
      props.mapId,
      props.members.map((member) => ({
        keyframeId: member.keyframeId,
        thetaCenter: member.thetaCenter,
        phiCenter: member.phiCenter,
      })),
      // Only the thumbnails matter here, so the geometry settings are irrelevant.
      { inlierThresholdDeg: 8 },
    )
      .then((payload) => {
        if (cancelled) return;
        const next = new Map<string, string | null>();
        payload.items.forEach((item, index) => {
          const member = props.members[index];
          if (member) next.set(groupAnnotationKey(member), item.thumbnail);
        });
        setThumbnails(next);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.members]);

  const selection = useMemo(
    () =>
      props.members
        .filter((member) => selected.has(groupAnnotationKey(member)))
        .map((member) => ({
          target: member,
          thumbnail: thumbnails.get(groupAnnotationKey(member)) ?? null,
        })),
    [props.members, selected, thumbnails],
  );

  const toggle = (key: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <div
      className="group-browser"
      role="dialog"
      aria-modal="true"
      aria-label={`Members of ${props.groupName}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) props.onClose();
      }}
    >
      <div className="group-browser-panel">
        <header>
          <h3>
            {props.groupName} · {props.members.length} detections
          </h3>
          <button
            type="button"
            className="object-search-secondary-button"
            onClick={() =>
              setSelected(new Set(props.members.map((m) => groupAnnotationKey(m))))
            }
          >
            All
          </button>
          <button
            type="button"
            className="object-search-secondary-button"
            onClick={() => setSelected(new Set())}
          >
            None
          </button>
          <button
            type="button"
            className="object-search-primary-button"
            disabled={!selection.length}
            onClick={() => props.onAdd(selection)}
          >
            Add {selection.length} to basket
          </button>
          <button
            type="button"
            className="object-search-secondary-button"
            aria-label="Close"
            onClick={props.onClose}
          >
            ×
          </button>
        </header>

        {error ? <p className="error-box">{error}</p> : null}

        <div className="group-browser-grid">
          {props.members.map((member, index) => {
            const key = groupAnnotationKey(member);
            const thumbnail = thumbnails.get(key) ?? null;
            const preview = cutoutPreviewUrl(props.mapId, thumbnail);
            const isSelected = selected.has(key);
            return (
              <button
                type="button"
                key={key}
                className={`group-browser-card${isSelected ? " is-selected" : ""}`}
                aria-pressed={isSelected}
                onClick={() => toggle(key)}
              >
                {preview ? (
                  <img src={preview} alt="" loading="lazy" />
                ) : (
                  <span className="group-browser-card-empty" aria-hidden />
                )}
                <span>
                  #{index + 1} · kf {member.keyframeId}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default GroupBrowser;
