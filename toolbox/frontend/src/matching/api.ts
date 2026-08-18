/** Client for the bricks matching endpoint (cosine matrix + triangulation). */

export type MatchingItemResult = {
  keyframe_id: string;
  theta_center: number;
  phi_center: number;
  resolved: boolean;
  candidate_id: number | null;
  thumbnail: string | null;
  has_embedding: boolean;
  has_stored_position: boolean;
  /** `[lat, lng, alt]`, null when the detection did not resolve. */
  keyframe_wgs84: [number, number, number] | null;
  /** End of a fixed-length ray from the keyframe, so a failed match still draws. */
  ray_end_wgs84: [number, number, number] | null;
  /** The depth-based position the pipeline stored, when it has one. */
  stored_wgs84: [number, number, number] | null;
};

/** Sequential RANSAC, or the two order-free linkage methods to compare it against. */
export type PartitionMethod =
  | "sequential"
  | "jlinkage"
  | "tlinkage"
  | "gasp"
  | "gasp1v2";

export type TriangulationHypothesis = {
  /** Positions in the request's item list. */
  items: number[];
  lat: number;
  lng: number;
  alt: number;
  level: number | null;
  mean_residual_deg: number | null;
  mean_residual_m: number | null;
  max_parallax_deg: number | null;
};

export type TriangulationResponse = {
  available: boolean;
  reason?: string;
  /** Positions in the request's item list, not indices into the ray list. */
  inlier_items?: number[];
  ray_index?: number[];
  residuals_deg?: number[];
  residuals_m?: number[];
  mean_inlier_residual_deg?: number | null;
  mean_inlier_residual_m?: number | null;
  /** Detections dropped because their depth-map point sits beyond the cap. */
  beyond_max_depth_items?: number[];
  /** Same-keyframe pairs the angular rule declared to be different objects. */
  cannot_link_pairs?: Array<[number, number]>;
  max_parallax_deg?: number | null;
  lat?: number;
  lng?: number;
  alt?: number;
  /** Floor the point lands on, resolved like a cluster's; null when unresolved. */
  level?: number | null;
  /** The set read as a partition: one entry per object found, largest first. */
  hypotheses?: TriangulationHypothesis[];
  unassigned_items?: number[];
  partition_method?: PartitionMethod;
};

export type MatchingResponse = {
  items: MatchingItemResult[];
  /** N×N cosine, null where an embedding is missing. */
  similarity: Array<Array<number | null>>;
  triangulation: TriangulationResponse;
};

export async function runMatching(
  mapId: string,
  items: Array<{ keyframeId: string; thetaCenter: number; phiCenter: number }>,
  options: {
    inlierThresholdDeg: number;
    partitionMethod?: PartitionMethod;
    /** null = no cap. Detections deeper than this are left out of the geometry. */
    maxDepthM?: number | null;
    /** Pull towards the depth-map point; 0 = pure ray geometry. */
    depthWeight?: number;
    /** Two angularly disjoint boxes of one panorama are two objects (GASP only). */
    cannotLinkSameKeyframe?: boolean;
    /** Weight of the co-visibility conflict cost; 0 disables it (GASP 1v2 only). */
    covisibilityWeight?: number;
  },
): Promise<MatchingResponse> {
  const response = await fetch(
    `/${encodeURIComponent(mapId)}/object-search/matching`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        items: items.map((item) => ({
          keyframe_id: item.keyframeId,
          theta_center: item.thetaCenter,
          phi_center: item.phiCenter,
        })),
        inlier_threshold_deg: options.inlierThresholdDeg,
        partition_method: options.partitionMethod ?? "sequential",
        max_depth_m: options.maxDepthM ?? null,
        depth_weight: options.depthWeight ?? 0,
        cannot_link_same_keyframe: options.cannotLinkSameKeyframe ?? false,
        covisibility_weight: options.covisibilityWeight ?? 0,
      }),
    },
  );
  if (!response.ok) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      throw new Error(
        typeof parsed.detail === "string" ? parsed.detail : `HTTP ${response.status}`,
      );
    } catch (err) {
      throw err instanceof Error && err.message !== "Unexpected end of JSON input"
        ? err
        : new Error(`HTTP ${response.status}`);
    }
  }
  return (await response.json()) as MatchingResponse;
}
