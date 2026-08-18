export const KEYFRAME_MARKER_COLOR = "#9ca3af";
export const KEYFRAME_MARKER_SELECTED_COLOR = "#16a34a";
export const KEYFRAME_MARKER_RADIUS = 6;
export const KEYFRAME_MARKER_SELECTED_RADIUS = 8;
/** Keyframes with rows in the parquet but pruned from pgvector at ingest time. */
export const KEYFRAME_MARKER_PRUNED_COLOR = "#f97316";
/** No summary is available for this marker on the server-paged current page. */
export const KEYFRAME_MARKER_UNKNOWN_COLOR = "#8b5cf6";
export const KEYFRAME_GRAPH_COLOR = "#64748b";
export const KEYFRAME_LINK_COLOR = "#9333ea";
export const VIEW_CONE_HALF_ANGLE_DEG = 25;
export const VIEW_CONE_COLOR = KEYFRAME_MARKER_SELECTED_COLOR;
/** Saved export regions. Distinct from the in-progress ROI's blue, and from every
    marker state, so a region is never read as a keyframe class. */
export const EXPORT_ROI_COLOR = "#0f766e";
/** The capture path joining consecutive keyframes, when no 360-viewer graph exists. */
export const KEYFRAME_TRACK_COLOR = "#0ea5e9";
