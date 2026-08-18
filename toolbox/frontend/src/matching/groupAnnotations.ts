/**
 * Client for the reference partition — which detections are one physical object.
 *
 * The unit is a detection, not a cluster: an annotation says "this detection belongs
 * to group X", so two clusters merge by receiving one name and a cluster splits by
 * having its members named differently. Nothing here edits a cluster; clusters are
 * the prediction being judged.
 */

import { useCallback, useEffect, useState } from "react";

/** null is a value, not an absence: "annotated as not an object". */
export type GroupName = string | null;

export type GroupAnnotationTarget = {
  keyframeId: string;
  thetaCenter: number;
  phiCenter: number;
  rowIndex?: number | null;
};

type GroupMemberResponse = {
  keyframe_id?: unknown;
  theta_center?: unknown;
  phi_center?: unknown;
  group_name?: unknown;
};

type GroupsResponse = {
  groups?: Array<{ group_name?: unknown; members?: unknown }>;
  not_an_object?: unknown;
};

/**
 * 6 decimals, because that is what the backend rounds to before writing: the angles
 * are the primary key, so both sides must spell them identically or an update writes
 * a second row instead of moving the detection.
 */
export function groupAnnotationKey(target: GroupAnnotationTarget): string {
  return `${target.keyframeId}:${target.thetaCenter.toFixed(6)}:${target.phiCenter.toFixed(6)}`;
}

function groupsUrl(mapId: string, suffix = ""): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}/group-annotations${suffix}`;
}

async function errorDetail(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `HTTP ${response.status}`;
  }
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    return typeof parsed.detail === "string" ? parsed.detail : text;
  } catch {
    return text;
  }
}

function memberKey(raw: GroupMemberResponse): string | null {
  const keyframeId = typeof raw.keyframe_id === "string" ? raw.keyframe_id : null;
  const theta = Number(raw.theta_center);
  const phi = Number(raw.phi_center);
  if (!keyframeId || !Number.isFinite(theta) || !Number.isFinite(phi)) {
    return null;
  }
  return groupAnnotationKey({ keyframeId, thetaCenter: theta, phiCenter: phi });
}

export type GroupAnnotations = {
  /** Detection key → group name, or null for "not an object". Absent = unannotated. */
  labels: Map<string, GroupName>;
  /** Every group name in use on this map, for the picker. */
  names: string[];
  /**
   * Group name → its members, so an annotated group can be loaded straight into the
   * matching basket. Without it, "check whether the group I just annotated holds up"
   * means finding every member a second time by hand.
   */
  members: Map<string, GroupAnnotationTarget[]>;
  error: string | null;
  labelFor: (target: GroupAnnotationTarget) => GroupName | undefined;
  setLabel: (targets: GroupAnnotationTarget[], groupName: GroupName) => Promise<void>;
  clearLabel: (targets: GroupAnnotationTarget[]) => Promise<void>;
  /** Drop every label of a group; resolves to how many were removed. */
  deleteGroup: (groupName: string) => Promise<number>;
};

export function useGroupAnnotations(mapId: string): GroupAnnotations {
  const [labels, setLabels] = useState<Map<string, GroupName>>(() => new Map());
  const [names, setNames] = useState<string[]>([]);
  const [members, setMembers] = useState<Map<string, GroupAnnotationTarget[]>>(
    () => new Map(),
  );
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const response = await fetch(groupsUrl(mapId), {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(await errorDetail(response));
      }
      const payload = (await response.json()) as GroupsResponse;
      const next = new Map<string, GroupName>();
      const groupNames: string[] = [];
      const byGroup = new Map<string, GroupAnnotationTarget[]>();
      for (const group of payload.groups ?? []) {
        const name = typeof group.group_name === "string" ? group.group_name : null;
        if (name === null) continue;
        groupNames.push(name);
        const targets: GroupAnnotationTarget[] = [];
        for (const member of Array.isArray(group.members) ? group.members : []) {
          const raw = member as GroupMemberResponse;
          const key = memberKey(raw);
          if (!key) continue;
          next.set(key, name);
          targets.push({
            keyframeId: String(raw.keyframe_id),
            thetaCenter: Number(raw.theta_center),
            phiCenter: Number(raw.phi_center),
          });
        }
        byGroup.set(name, targets);
      }
      setMembers(byGroup);
      for (const member of Array.isArray(payload.not_an_object)
        ? payload.not_an_object
        : []) {
        const key = memberKey(member as GroupMemberResponse);
        if (key) next.set(key, null);
      }
      setLabels(next);
      setNames(groupNames.sort((a, b) => a.localeCompare(b)));
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [mapId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const mutate = useCallback(
    async (
      targets: GroupAnnotationTarget[],
      request: (target: GroupAnnotationTarget) => Promise<Response>,
    ) => {
      try {
        // Sequential on purpose: a cluster is tens of detections, and one SQLite
        // writer serialises them anyway. Parallel requests would only add lock churn.
        for (const target of targets) {
          const response = await request(target);
          if (!response.ok) {
            throw new Error(await errorDetail(response));
          }
        }
        await reload();
      } catch (err) {
        setError((err as Error).message);
      }
    },
    [reload],
  );

  const setLabel = useCallback(
    (targets: GroupAnnotationTarget[], groupName: GroupName) =>
      mutate(targets, (target) =>
        fetch(groupsUrl(mapId, "/label"), {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            keyframe_id: target.keyframeId,
            theta_center: target.thetaCenter,
            phi_center: target.phiCenter,
            row_index: target.rowIndex ?? null,
            group_name: groupName,
          }),
        }),
      ),
    [mapId, mutate],
  );

  const clearLabel = useCallback(
    (targets: GroupAnnotationTarget[]) =>
      mutate(targets, (target) =>
        fetch(groupsUrl(mapId, "/label"), {
          method: "DELETE",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            keyframe_id: target.keyframeId,
            theta_center: target.thetaCenter,
            phi_center: target.phiCenter,
          }),
        }),
      ),
    [mapId, mutate],
  );

  const deleteGroup = useCallback(
    async (groupName: string) => {
      try {
        const response = await fetch(groupsUrl(mapId, "/group"), {
          method: "DELETE",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ group_name: groupName }),
        });
        if (!response.ok) {
          throw new Error(await errorDetail(response));
        }
        const payload = (await response.json()) as { deleted?: unknown };
        await reload();
        return Number(payload.deleted) || 0;
      } catch (err) {
        setError((err as Error).message);
        return 0;
      }
    },
    [mapId, reload],
  );

  const labelFor = useCallback(
    (target: GroupAnnotationTarget) => labels.get(groupAnnotationKey(target)),
    [labels],
  );

  return { labels, names, members, error, labelFor, setLabel, clearLabel, deleteGroup };
}
