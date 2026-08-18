/**
 * The map list as a tree: sub-maps nest under the map they were exported from.
 *
 * Two cases the flat list never had to handle, and both must stay visible rather
 * than disappear: a sub-map whose parent is no longer in the config (deleting the
 * parent's entry by hand, say) is shown at the root, and a `parent_map` cycle —
 * which nothing creates today but a hand-edited config can — is broken instead of
 * recursed into.
 */

import type { MapSummary } from "../api";

export type MapTreeNode = {
  map: MapSummary;
  children: MapTreeNode[];
  depth: number;
};

export function buildMapTree(maps: MapSummary[]): MapTreeNode[] {
  const known = new Set(maps.map((map) => map.id));
  const childrenByParent = new Map<string, MapSummary[]>();
  const roots: MapSummary[] = [];

  for (const map of maps) {
    const parent = map.parent_map;
    if (parent && known.has(parent) && parent !== map.id) {
      const bucket = childrenByParent.get(parent);
      if (bucket) {
        bucket.push(map);
      } else {
        childrenByParent.set(parent, [map]);
      }
    } else {
      roots.push(map);
    }
  }

  const visited = new Set<string>();
  const build = (map: MapSummary, depth: number): MapTreeNode => {
    visited.add(map.id);
    const children = (childrenByParent.get(map.id) ?? [])
      .filter((child) => !visited.has(child.id))
      .map((child) => build(child, depth + 1));
    return { map, children, depth };
  };

  const tree = roots.map((map) => build(map, 0));

  // A cycle leaves its members unreachable from any root. They are real configured
  // maps, so they are surfaced flat rather than swallowed.
  const orphans = maps.filter((map) => !visited.has(map.id));
  return [...tree, ...orphans.map((map) => ({ map, children: [], depth: 0 }))];
}
