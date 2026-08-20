import { useCallback, useEffect, useMemo, useState } from "react";

import type { MapSummary } from "../api";
import AnalysisMap from "./AnalysisMap";
import { fetchAnalysisStatus, startAnalysisRun } from "./api";
import KeyframeThresholdChart from "./KeyframeThresholdChart";
import MetricHint from "./MetricHint";
import { explain, verdictFor, type Verdict } from "./metricCatalogue";
import {
  buildIndicators,
  integer,
  metres,
  number,
  ONLINE_MIN_KEYFRAMES,
  percent,
  type Indicator,
} from "./indicators";
import type { AnalysisStatus, Percentiles } from "./types";

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
};

const ACTIVE_POLL_MS = 2000;

// ---------------------------------------------------------------------------
// Small pieces
// ---------------------------------------------------------------------------

function ScoreTile(props: { indicator: Indicator }) {
  const metric = explain(props.indicator.id);
  const verdict: Verdict = verdictFor(props.indicator.id, props.indicator.value);
  const body = (
    <span className={`analysis-tile is-${verdict}`}>
      <span className="analysis-tile-label">
        {metric?.label ?? props.indicator.id}
        {metric ? <span className="analysis-tile-mark">?</span> : null}
      </span>
      <span className="analysis-tile-value">{props.indicator.display}</span>
      <span className="analysis-tile-detail">{props.indicator.detail}</span>
    </span>
  );
  return metric ? <MetricHint metric={metric}>{body}</MetricHint> : body;
}

function MetricLabel(props: { id: string; fallback: string }) {
  const metric = explain(props.id);
  if (!metric) {
    return <span>{props.fallback}</span>;
  }
  return (
    <MetricHint metric={metric}>
      <span className="analysis-row-label">
        {props.fallback}
        <span className="analysis-tile-mark">?</span>
      </span>
    </MetricHint>
  );
}

function Row(props: { label: string; value: string; metricId?: string }) {
  return (
    <div className="analysis-row">
      {props.metricId ? (
        <MetricLabel id={props.metricId} fallback={props.label} />
      ) : (
        <span className="analysis-row-label">{props.label}</span>
      )}
      <span className="analysis-row-value">{props.value}</span>
    </div>
  );
}

function PercentileTable(props: {
  rows: [string, Percentiles | undefined][];
  format?: (value: number) => string;
}) {
  const format = props.format ?? ((value: number) => number(value));
  return (
    <table className="analysis-table">
      <thead>
        <tr>
          <th>quantity</th>
          <th>n</th>
          <th>p10</th>
          <th>median</th>
          <th>p90</th>
          <th>max</th>
        </tr>
      </thead>
      <tbody>
        {props.rows.map(([label, values]) =>
          values ? (
            <tr key={label}>
              <td>{label}</td>
              <td>{integer(values.n)}</td>
              <td>{format(values.p10)}</td>
              <td>{format(values.median)}</td>
              <td>{format(values.p90)}</td>
              <td>{format(values.max)}</td>
            </tr>
          ) : null,
        )}
      </tbody>
    </table>
  );
}

function Section(props: {
  id: string;
  title: string;
  subtitle: string;
  /** Missing data is a state of its own: say so instead of drawing empty tables. */
  inert?: string | null;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <section className="analysis-section">
      <button
        className="analysis-section-head"
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span className="analysis-section-id">{props.id}</span>
        <span className="analysis-section-title">{props.title}</span>
        <span className="analysis-section-subtitle">{props.subtitle}</span>
        <span className="analysis-section-chevron">{open ? "−" : "+"}</span>
      </button>
      {open ? (
        <div className="analysis-section-body">
          {props.inert ? <p className="analysis-inert">{props.inert}</p> : props.children}
        </div>
      ) : null}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

function AnalysisPanel(props: Props) {
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchAnalysisStatus(props.mapId));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [props.mapId]);

  useEffect(() => {
    if (!props.isMapKnown) {
      return;
    }
    void refresh();
  }, [props.isMapKnown, refresh]);

  const running = status?.job?.state === "running";
  useEffect(() => {
    if (!running) {
      return;
    }
    const timer = window.setInterval(() => void refresh(), ACTIVE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [running, refresh]);

  const onRun = useCallback(async () => {
    setStarting(true);
    try {
      await startAnalysisRun(props.mapId);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setStarting(false);
    }
  }, [props.mapId, refresh]);

  const analysis = status?.analysis ?? null;
  const indicators = useMemo(
    () => (analysis ? buildIndicators(analysis) : []),
    [analysis],
  );

  if (!props.isMapKnown) {
    return (
      <section className="analysis-panel page-section">
        <p className="error-box">Map not found in the current config.</p>
      </section>
    );
  }

  const job = status?.job ?? null;
  const s0 = analysis?.s0;
  const s1 = analysis?.s1;
  const s2 = analysis?.s2;
  const s3 = analysis?.s3;
  const s4 = analysis?.s4;
  const s5 = analysis?.s5;
  const s6 = analysis?.s6;
  const s7 = analysis?.s7;

  return (
    <section className="analysis-panel page-section">
      <div className="analysis-toolbar">
        <div>
          <p className="eyebrow">Analysis</p>
          <h2 className="section-heading">What this prepared map holds</h2>
          <p className="muted">
            The whole parquet, not a query&apos;s results — no database and no ANN
            service. Hover any label for what it measures and what a bad value means.
          </p>
        </div>
        <div className="analysis-toolbar-actions">
          <button
            className="primary-button"
            type="button"
            onClick={() => void onRun()}
            disabled={running || starting}
          >
            {running ? "Analysing…" : "Run analysis"}
          </button>
          {status?.latest_run_id ? (
            <p className="muted">last run {status.latest_run_id}</p>
          ) : (
            <p className="muted">never analysed</p>
          )}
        </div>
      </div>

      {error ? <p className="error-box">{error}</p> : null}
      {job?.state === "error" ? (
        <pre className="error-box analysis-trace">{job.error}</pre>
      ) : null}
      {running ? (
        <p className="analysis-progress">
          {`at section ${job?.progress.started ?? 0} of ${job?.progress.total ?? 8}`}
          {" — reads the whole parquet, expect several minutes"}
        </p>
      ) : null}

      {!analysis ? (
        <p className="muted">
          No cached analysis for this map yet. A run reads every parquet row and takes
          minutes, so it is never started implicitly.
        </p>
      ) : (
        <>
          {s0 && s0.missing.length > 0 ? (
            <MetricHint metric={explain("missing_pieces")!}>
              <p className="analysis-missing">
                <strong>Missing from this index:</strong> {s0.missing.join(" · ")} — the
                sections that read it report emptiness, not a measured zero.
              </p>
            </MetricHint>
          ) : null}

          <div className="analysis-tiles">
            {indicators.map((indicator) => (
              <ScoreTile key={indicator.id} indicator={indicator} />
            ))}
          </div>

          <section className="analysis-section is-open">
            <div className="analysis-section-head is-static">
              <span className="analysis-section-id">map</span>
              <span className="analysis-section-title">Where the measurements are</span>
              <span className="analysis-section-subtitle">
                one field at a time, on the building
              </span>
            </div>
            <div className="analysis-section-body">
              <AnalysisMap
                mapId={props.mapId}
                emmid={props.map?.emmid ?? null}
                runId={status?.latest_run_id ?? null}
                available={status?.layers ?? []}
              />
            </div>
          </section>

          <Section
            id="s0"
            title="Inventory"
            subtitle="what the map holds, and what it is missing"
          >
            <Row label="Keyframes in manifest" value={integer(s0?.keyframes_manifest)} />
            <Row
              label="Keyframes with detections"
              value={integer(s0?.keyframes_with_detections)}
            />
            <Row label="Detections" value={integer(s0?.detections)} />
            <Row
              label="Placed in 3D"
              value={`${integer(s0?.placed)} (${percent(
                s0 && s0.detections ? s0.placed / s0.detections : undefined,
              )})`}
              metricId="placed_share"
            />
            <Row label="Ground-truth annotations" value={integer(s0?.ground_truth)} />
            <Row label="Cutout reviews" value={integer(s0?.reviews)} />
            <Row label="Hand group labels" value={integer(s0?.group_labels)} />
            <Row
              label="Embeddings"
              value={
                s0?.embeddings
                  ? `${integer(s0.embeddings[0])} × ${integer(s0.embeddings[1])}`
                  : "none"
              }
            />
          </Section>

          <Section
            id="s1"
            title="Detections"
            subtitle="distributions, split by detector"
            inert={s1 ? null : "No detections in this index."}
          >
            <PercentileTable
              rows={[
                ["range (m)", s1?.range_m],
                ["angular diagonal (deg)", s1?.angular_diagonal_deg],
                ["phi (deg)", s1?.phi_deg],
                ["proposals per panorama", s1?.per_keyframe],
              ]}
            />
            <h4 className="analysis-subheading">By detector</h4>
            <table className="analysis-table">
              <thead>
                <tr>
                  <th>detector</th>
                  <th>rows</th>
                  <th>median range</th>
                  <th>
                    <MetricLabel id="detector_score_median" fallback="median score" />
                  </th>
                  <th>p90 score</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(s1?.by_source ?? {}).map(([name, summary]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{integer(summary.n)}</td>
                    <td>{metres(summary.range_m.median)}</td>
                    <td>{number(summary.score.median, 3)}</td>
                    <td>{number(summary.score.p90, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>

          <Section
            id="s2"
            title="Labels against ground truth"
            subtitle="always read next to the base rate"
            inert={s2 ? null : "No labels or no annotations to compare them with."}
          >
            <Row
              label={`Base rate at ${number(s2?.attachment_radius_m, 1)} m`}
              value={percent(s2?.base_rate["2.0"] ?? s2?.base_rate["2"])}
              metricId="base_rate_2m"
            />
            <Row
              label="Label ↔ class mutual information"
              value={number(s2?.mutual_information_label_gt_class, 3)}
              metricId="mutual_information"
            />
            <Row
              label="Neighbour purity — same class"
              value={percent(s2?.embedding_purity.same_class)}
              metricId="purity_same_class"
            />
            <Row
              label="Neighbour purity — same object"
              value={percent(s2?.embedding_purity.same_object)}
              metricId="purity_same_object"
            />
            <h4 className="analysis-subheading">Base rate by radius</h4>
            <div className="analysis-inline-rows">
              {Object.entries(s2?.base_rate ?? {}).map(([radius, share]) => (
                <span key={radius} className="analysis-chip">
                  {`${radius} m: ${percent(share)}`}
                </span>
              ))}
            </div>
          </Section>

          <Section
            id="s3"
            title="Ground truth against detections"
            subtitle="the depth-free measurement first"
            inert={s3 ? null : "This map carries no annotations."}
          >
            <Row
              label="Covered in its own panorama"
              value={percent(s3?.covered_in_own_keyframe)}
              metricId="covered_in_own_keyframe"
            />
            <Row
              label="Source panorama indexed"
              value={percent(s3?.source_keyframe_indexed)}
              metricId="source_keyframe_indexed"
            />
            <Row
              label="Never looked at"
              value={integer(s3?.unseen.never_looked)}
              metricId="unseen_never_looked"
            />
            <Row
              label="Looked at and missed"
              value={integer(s3?.unseen.looked_and_missed)}
            />
            <Row
              label="Seen on a degenerate baseline"
              value={integer(s3?.degenerate_baseline)}
            />

            {s3?.keyframe_threshold && s3.keyframe_threshold.length > 0 ? (
              <>
                <h4 className="analysis-subheading">
                  <MetricLabel
                    id="keyframe_threshold_cost"
                    fallback="What min_keyframes_per_cluster costs"
                  />
                </h4>
                <KeyframeThresholdChart
                  points={s3.keyframe_threshold}
                  defaultThreshold={ONLINE_MIN_KEYFRAMES}
                />
                <p className="analysis-legend">
                  <span className="analysis-legend-key" /> annotations retained
                  <span className="analysis-legend-key is-secondary" /> of those, covered
                  depth-free
                </p>
              </>
            ) : null}

            <h4 className="analysis-subheading">Observability per annotation</h4>
            <PercentileTable
              rows={[
                ["detections", s3?.observability.detections],
                ["parallax achieved (deg)", s3?.observability.achieved_parallax_deg],
                ["parallax available (deg)", s3?.observability.available_parallax_deg],
                ["nearest keyframe (m)", s3?.observability.nearest_keyframe_m],
                ["trajectory anisotropy", s3?.observability.trajectory_anisotropy],
                ["azimuth sectors achieved", s3?.observability.achieved_coverage],
                ["azimuth sectors available", s3?.observability.available_coverage],
                ["useful pair share", s3?.observability.useful_pair_share],
              ]}
            />

            <h4 className="analysis-subheading">Depth-free coverage by class</h4>
            <div className="analysis-inline-rows">
              {Object.entries(s3?.covered_by_class ?? {})
                .sort((left, right) => left[1] - right[1])
                .map(([name, share]) => (
                  <span
                    key={name}
                    className={`analysis-chip is-${verdictFor(
                      "covered_in_own_keyframe",
                      share,
                    )}`}
                  >
                    {`${name}: ${percent(share)}`}
                  </span>
                ))}
            </div>
          </Section>

          <Section
            id="s4"
            title="Pairwise cues"
            subtitle="conditional AUC is the one that decides"
            inert={s4 ? null : "Not enough annotated pairs to draw the populations."}
          >
            <p className="muted">
              {`${integer(s4?.sampled_pairs)} pairs sampled — ${integer(
                s4?.same_object,
              )} same object, ${integer(s4?.different_object)} different, ${integer(
                s4?.contaminated,
              )} contaminated.`}
            </p>
            <table className="analysis-table">
              <thead>
                <tr>
                  <th>cue</th>
                  <th>
                    <MetricLabel id="cue_raw_auc" fallback="raw AUC" />
                  </th>
                  <th>
                    <MetricLabel id="best_conditional_auc" fallback="conditional AUC" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(s4?.auc ?? {})
                  .sort((left, right) => right[1].conditional - left[1].conditional)
                  .map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td className="is-muted">{number(value.raw, 3)}</td>
                      <td
                        className={`is-${verdictFor("best_conditional_auc", value.conditional)}`}
                      >
                        {number(value.conditional, 3)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </Section>

          <Section
            id="s5"
            title="Duplicates and counting"
            subtitle="how much of the index is the same thing twice"
            inert={s5 ? null : "No overlapping boxes sampled."}
          >
            <Row label="Keyframes sampled" value={integer(s5?.sampled_keyframes)} />
            <Row label="Overlapping in-view pairs" value={integer(s5?.overlapping_pairs)} />
            <Row
              label="Cross-detector share"
              value={percent(s5?.cross_source_share)}
              metricId="cross_source_share"
            />
            <PercentileTable rows={[["overlap cosine", s5?.overlap_cosine]]} />
          </Section>

          <Section
            id="s6"
            title="Hubness"
            subtitle="no ground truth at all"
            inert={s6 ? null : "This index carries no embeddings."}
          >
            <table className="analysis-table">
              <thead>
                <tr>
                  <th>quantity</th>
                  <th>raw</th>
                  <th>recentred</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <MetricLabel id="hubness_skewness" fallback="skewness" />
                  </td>
                  <td className={`is-${verdictFor("hubness_skewness", s6?.raw.skewness ?? Number.NaN)}`}>
                    {number(s6?.raw.skewness)}
                  </td>
                  <td>{number(s6?.centred.skewness)}</td>
                </tr>
                <tr>
                  <td>hub share</td>
                  <td>{percent(s6?.raw.hub_share, 2)}</td>
                  <td>{percent(s6?.centred.hub_share, 2)}</td>
                </tr>
                <tr>
                  <td>
                    <MetricLabel id="hub_mass" fallback="hub mass" />
                  </td>
                  <td className={`is-${verdictFor("hub_mass", s6?.raw.hub_mass ?? Number.NaN)}`}>
                    {percent(s6?.raw.hub_mass, 2)}
                  </td>
                  <td>{percent(s6?.centred.hub_mass, 2)}</td>
                </tr>
                <tr>
                  <td>antihub share</td>
                  <td>{percent(s6?.raw.antihub_share, 2)}</td>
                  <td>{percent(s6?.centred.antihub_share, 2)}</td>
                </tr>
              </tbody>
            </table>
            <p className="muted">
              {`${integer(s6?.sampled)} rows sampled at k = ${integer(s6?.k)}.`}
            </p>
          </Section>

          <Section
            id="s7"
            title="Label sets"
            subtitle="detector labels against the set of acceptable ones"
            inert={
              !s7 || s7.coverage === 0
                ? "No annotation carries a label set, so this section cannot be computed"
                  + " — that is a missing input, not a score of zero."
                : null
            }
          >
            <Row
              label="Annotations with a label set"
              value={percent(s7?.coverage)}
              metricId="label_set_coverage"
            />
            <Row label="Annotations" value={integer(s7?.annotations)} />
          </Section>
        </>
      )}
    </section>
  );
}

export default AnalysisPanel;
