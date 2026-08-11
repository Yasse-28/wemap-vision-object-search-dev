/**
 * One-line rendering of a benchmark run's `config` block.
 *
 * A run list where every entry looks alike is exactly how a boosted run ends up
 * compared against another boosted run and read as "the boost does nothing". Both
 * the Benchmark tab (per stored run) and the Annotation tab (per prompt score and
 * its baseline) need the same rendering, so it lives here rather than in either.
 */

/** The parameters that actually change the numbers, in the order they read best. */
const CONFIG_SUMMARY_KEYS: Array<[string, string]> = [
  ["acceptance_threshold", "acceptance"],
  ["min_similarity", "min_sim"],
  ["clustering_eps_m", "eps"],
  ["candidate_count", "candidates"],
  ["min_keyframes_per_cluster", "min_kf"],
  // Both decide what the metrics count, not what the pipeline returns — two runs that
  // differ here are not comparable however identical the rest is.
  ["group_annotation_radius_m", "group_r"],
  ["default_accuracy", "match_r"],
  ["feedback_alpha", "α"],
  ["feedback_beta", "β"],
  ["feedback_normalization", "norm"],
];

/**
 * Trims float noise without hiding a real value. The Annotation tab derives its
 * acceptance threshold as `slider - 1e-9`, so verbatim it reads "0.899999999" and the
 * reader wonders what the epsilon is for.
 */
function formatValue(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? String(Number(value.toFixed(4)))
    : String(value);
}

export function configSummary(config: Record<string, unknown> | undefined): string {
  if (!config) {
    return "unknown parameters";
  }
  const parts = CONFIG_SUMMARY_KEYS.flatMap(([key, label]) => {
    const value = config[key];
    return value == null ? [] : [`${label} ${formatValue(value)}`];
  });
  return parts.length ? parts.join(" · ") : "unknown parameters";
}
