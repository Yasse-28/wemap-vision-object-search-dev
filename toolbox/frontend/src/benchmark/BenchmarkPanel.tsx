import { useCallback, useEffect, useRef, useState } from "react";

import type { MapSummary } from "../api";
import {
  fetchBenchmarkRun,
  fetchBenchmarkStatus,
  startBenchmarkRun,
} from "./api";
import type {
  BenchmarkMetrics,
  BenchmarkRunParams,
  BenchmarkStatus,
} from "./types";

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
};

const DEFAULT_PARAMS: BenchmarkRunParams = {
  target_server: "local",
  remote_localize_url:
    "https://vps-api.wemap-vision-computing-1.getwemap.com/{map_id}/object-search/localize",
  acceptance_threshold: 0.9,
  num_results: 100,
  min_similarity: 0.15,
  clustering_eps_m: 1.5,
  candidate_count: 1000,
  group_annotation_radius_m: 0,
};

const ACTIVE_POLL_MS = 1000;
const IDLE_POLL_MS = 10000;

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMs(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

function serviceBadge(status: BenchmarkStatus | null): {
  label: string;
  tone: "ok" | "warn" | "down";
} {
  const service = status?.python_service;
  if (!service) {
    return { label: "unknown", tone: "warn" };
  }
  if (!service.reachable) {
    return { label: "down — will auto-start on run", tone: "down" };
  }
  if (service.map_state === "ready") {
    return { label: "ready", tone: "ok" };
  }
  if (service.map_state === "loading" || service.status === "starting") {
    return { label: "loading", tone: "warn" };
  }
  if (service.map_state == null) {
    return { label: "up (map not loaded)", tone: "warn" };
  }
  return { label: service.map_state, tone: "warn" };
}

function BenchmarkPanel(props: Props) {
  const [status, setStatus] = useState<BenchmarkStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [params, setParams] = useState<BenchmarkRunParams>(DEFAULT_PARAMS);
  const [runError, setRunError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<BenchmarkMetrics | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const previousJobState = useRef<string | null>(null);

  const jobActive =
    status?.job != null
    && (status.job.state === "ensuring-service" || status.job.state === "running");

  const refreshStatus = useCallback(async () => {
    try {
      const payload = await fetchBenchmarkStatus(props.mapId);
      setStatus(payload);
      setStatusError(null);
      return payload;
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
      return null;
    }
  }, [props.mapId]);

  // Poll status: fast while a job is active, slow when idle.
  useEffect(() => {
    if (!props.isMapKnown) {
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      const payload = await refreshStatus();
      if (cancelled) {
        return;
      }
      const active =
        payload?.job != null
        && (payload.job.state === "ensuring-service" || payload.job.state === "running");
      timer = window.setTimeout(tick, active ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    };
    void tick();

    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [props.isMapKnown, refreshStatus]);

  // Auto-select the newest run when a job finishes or on first load.
  useEffect(() => {
    if (!status) {
      return;
    }
    const jobState = status.job?.state ?? null;
    const jobJustFinished =
      previousJobState.current != null
      && previousJobState.current !== "done"
      && jobState === "done"
      && status.job?.map_id === props.mapId;
    previousJobState.current = jobState;

    if (jobJustFinished && status.job) {
      setSelectedRunId(status.job.run_id);
      return;
    }
    if (selectedRunId == null && status.runs.length) {
      setSelectedRunId(status.runs[0].run_id);
    }
  }, [status, props.mapId, selectedRunId]);

  // Load metrics for the selected run.
  useEffect(() => {
    if (!selectedRunId || !props.isMapKnown) {
      setMetrics(null);
      return;
    }
    let cancelled = false;
    setMetricsError(null);
    fetchBenchmarkRun(props.mapId, selectedRunId)
      .then((payload) => {
        if (!cancelled) {
          setMetrics(payload);
        }
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setMetrics(null);
          setMetricsError(error.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.isMapKnown, selectedRunId]);

  async function onRun() {
    setIsStarting(true);
    setRunError(null);
    try {
      await startBenchmarkRun(props.mapId, params);
      await refreshStatus();
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsStarting(false);
    }
  }

  function paramInput(
    label: string,
    key: keyof BenchmarkRunParams,
    step: number,
    min: number,
    disabled = false,
  ) {
    return (
      <label className="benchmark-param">
        <span>{label}</span>
        <input
          type="number"
          step={step}
          min={min}
          disabled={disabled}
          value={params[key]}
          onChange={(event) => {
            const value = Number(event.target.value);
            setParams((current) => ({ ...current, [key]: value }));
          }}
        />
      </label>
    );
  }

  if (!props.isMapKnown) {
    return (
      <section className="benchmark-panel page-section">
        <p className="error-box">Map not found in the current config.</p>
      </section>
    );
  }

  const badge = serviceBadge(status);
  const annotations = status?.annotations ?? null;
  const job = status?.job ?? null;
  const groupedMetricsEnabled = params.group_annotation_radius_m > 0;
  const remoteTargetSelected = params.target_server === "remote";
  const canRun =
    Boolean(annotations?.exists)
    && Boolean(annotations && annotations.prompt_count > 0)
    && !jobActive
    && !isStarting;

  return (
    <section className="benchmark-panel page-section">
      <div className="section-heading">
        <p className="eyebrow">Benchmark</p>
        <h1>Object Search Benchmark</h1>
      </div>

      {statusError ? <p className="error-box">{statusError}</p> : null}

      <div className="benchmark-status-bar">
        <div className="benchmark-status-item">
          <span className="benchmark-status-label">Local Python service</span>
          <span className={`benchmark-badge is-${badge.tone}`}>{badge.label}</span>
          {status?.python_service.managed && status.python_service.pid ? (
            <span className="muted">managed, pid {status.python_service.pid}</span>
          ) : null}
        </div>
        <div className="benchmark-status-item">
          <span className="benchmark-status-label">Annotations</span>
          {annotations?.exists ? (
            <span>
              {annotations.annotation_count} annotations, {annotations.prompt_count} prompts
            </span>
          ) : (
            <span className="benchmark-badge is-down">
              no benchmark/annotations.geojson found
            </span>
          )}
        </div>
      </div>

      <div className="benchmark-run-form">
        <label className="benchmark-param">
          <span>Target server</span>
          <select
            value={params.target_server}
            onChange={(event) => {
              const value = event.target.value === "remote" ? "remote" : "local";
              setParams((current) => ({ ...current, target_server: value }));
            }}
          >
            <option value="local">Local managed service</option>
            <option value="remote">Remote server</option>
          </select>
        </label>
        {remoteTargetSelected ? (
          <label className="benchmark-param benchmark-url-param">
            <span>Remote localize URL</span>
            <input
              type="url"
              value={params.remote_localize_url}
              onChange={(event) => {
                setParams((current) => ({
                  ...current,
                  remote_localize_url: event.target.value,
                }));
              }}
            />
          </label>
        ) : null}
        {paramInput("Acceptance threshold", "acceptance_threshold", 0.05, 0)}
        {paramInput("Num results", "num_results", 10, 1)}
        {paramInput("Min similarity", "min_similarity", 0.05, -1)}
        {paramInput("Clustering eps (m)", "clustering_eps_m", 0.5, 0.1)}
        {paramInput("Candidate count", "candidate_count", 100, 1)}
        <label className="benchmark-param benchmark-checkbox">
          <span>Group close annotations</span>
          <input
            type="checkbox"
            checked={groupedMetricsEnabled}
            onChange={(event) => {
              setParams((current) => ({
                ...current,
                group_annotation_radius_m: event.target.checked
                  ? (current.group_annotation_radius_m > 0 ? current.group_annotation_radius_m : 0.25)
                  : 0,
              }));
            }}
          />
        </label>
        {paramInput(
          "Group radius (m)",
          "group_annotation_radius_m",
          0.05,
          0,
          !groupedMetricsEnabled,
        )}
        <button
          className="benchmark-run-button"
          type="button"
          disabled={!canRun}
          onClick={() => void onRun()}
        >
          {jobActive ? "Running…" : "Run benchmark"}
        </button>
      </div>
      {runError ? <p className="error-box">{runError}</p> : null}

      {job && job.state !== "done" ? (
        <div className="benchmark-progress">
          {job.state === "ensuring-service" ? (
            <p>
              Starting python service…
              {job.service_state ? ` (${job.service_state})` : null}
            </p>
          ) : null}
          {job.state === "error" ? (
            <p className="error-box">Run {job.run_id} failed: {job.error}</p>
          ) : null}
          {job.state === "running" ? (
            <>
              <p>
                Run {job.run_id}: {job.progress.completed}
                {job.progress.total != null ? ` / ${job.progress.total}` : null} prompts
              </p>
              <progress
                value={job.progress.completed}
                max={job.progress.total ?? undefined}
              />
            </>
          ) : null}
          {job.prompts.length ? (
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Prompt</th>
                  <th>P</th>
                  <th>R</th>
                  <th>F1</th>
                  <th>TP</th>
                  <th>FP</th>
                  <th>FN</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {job.prompts.map((item) => (
                  <tr key={item.index} className={item.error ? "is-error" : undefined}>
                    <td>{item.class_name}</td>
                    <td>{item.prompt}</td>
                    <td>{formatPercent(item.precision)}</td>
                    <td>{formatPercent(item.recall)}</td>
                    <td>{formatPercent(item.f1)}</td>
                    <td>{item.true_positives}</td>
                    <td>{item.false_positives}</td>
                    <td>{item.false_negatives}</td>
                    <td>{formatMs(item.elapsed_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      ) : null}

      <div className="benchmark-results">
        <div className="benchmark-results-header">
          <h2>Results</h2>
          {status?.runs.length ? (
            <select
              value={selectedRunId ?? ""}
              onChange={(event) => setSelectedRunId(event.target.value || null)}
            >
              {status.runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id}
                  {run.summary ? ` — F1 ${formatPercent(run.summary.f1)}` : ""}
                </option>
              ))}
            </select>
          ) : (
            <span className="muted">No past runs for this map.</span>
          )}
        </div>

        {metricsError ? <p className="error-box">{metricsError}</p> : null}

        {metrics ? (
          <>
            <h3>Strict metrics</h3>
            <div className="benchmark-summary-cards">
              <SummaryCard label="Precision" value={formatPercent(metrics.summary.precision)} />
              <SummaryCard label="Recall" value={formatPercent(metrics.summary.recall)} />
              <SummaryCard label="F1" value={formatPercent(metrics.summary.f1)} />
              <SummaryCard label="TP" value={String(metrics.summary.true_positives)} />
              <SummaryCard label="FP" value={String(metrics.summary.false_positives)} />
              <SummaryCard label="FN" value={String(metrics.summary.false_negatives)} />
              <SummaryCard label="Prompts" value={String(metrics.summary.prompt_count)} />
              <SummaryCard label="Errors" value={String(metrics.summary.error_count)} />
            </div>

            <h3>By class</h3>
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>P</th>
                  <th>R</th>
                  <th>F1</th>
                  <th>TP</th>
                  <th>FP</th>
                  <th>FN</th>
                  <th>Prompts</th>
                  <th>Errors</th>
                </tr>
              </thead>
              <tbody>
                {metrics.by_class.map((row) => (
                  <tr key={row.class_name}>
                    <td>{row.class_name}</td>
                    <td>{formatPercent(row.precision)}</td>
                    <td>{formatPercent(row.recall)}</td>
                    <td>{formatPercent(row.f1)}</td>
                    <td>{row.true_positives}</td>
                    <td>{row.false_positives}</td>
                    <td>{row.false_negatives}</td>
                    <td>{row.prompt_count}</td>
                    <td>{row.errors}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h3>By prompt</h3>
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Prompt</th>
                  <th>P</th>
                  <th>R</th>
                  <th>F1</th>
                  <th>TP</th>
                  <th>FP</th>
                  <th>FN</th>
                  <th>Accepted</th>
                  <th>Rejected</th>
                  <th>GT</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {metrics.by_prompt.map((row) => (
                  <tr
                    key={`${row.class_name}-${row.prompt}`}
                    className={row.error ? "is-error" : undefined}
                  >
                    <td>{row.class_name}</td>
                    <td>{row.error ? `${row.prompt} (${row.error})` : row.prompt}</td>
                    <td>{formatPercent(row.precision)}</td>
                    <td>{formatPercent(row.recall)}</td>
                    <td>{formatPercent(row.f1)}</td>
                    <td>{row.true_positives}</td>
                    <td>{row.false_positives}</td>
                    <td>{row.false_negatives}</td>
                    <td>{row.accepted_predictions}</td>
                    <td>{row.rejected_predictions}</td>
                    <td>{row.ground_truth}</td>
                    <td>{formatMs(row.elapsed_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {metrics.grouped ? (
              <>
                <h3>
                  Grouped metrics ({metrics.grouped.group_annotation_radius_m} m)
                </h3>
                <div className="benchmark-summary-cards">
                  <SummaryCard
                    label="Precision"
                    value={formatPercent(metrics.grouped.summary.precision)}
                  />
                  <SummaryCard
                    label="Recall"
                    value={formatPercent(metrics.grouped.summary.recall)}
                  />
                  <SummaryCard
                    label="F1"
                    value={formatPercent(metrics.grouped.summary.f1)}
                  />
                  <SummaryCard
                    label="TP"
                    value={String(metrics.grouped.summary.true_positives)}
                  />
                  <SummaryCard
                    label="FP"
                    value={String(metrics.grouped.summary.false_positives)}
                  />
                  <SummaryCard
                    label="FN"
                    value={String(metrics.grouped.summary.false_negatives)}
                  />
                  <SummaryCard
                    label="Prompts"
                    value={String(metrics.grouped.summary.prompt_count)}
                  />
                  <SummaryCard
                    label="Errors"
                    value={String(metrics.grouped.summary.error_count)}
                  />
                </div>

                <h3>Grouped by class</h3>
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>P</th>
                      <th>R</th>
                      <th>F1</th>
                      <th>TP</th>
                      <th>FP</th>
                      <th>FN</th>
                      <th>Prompts</th>
                      <th>Errors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.grouped.by_class.map((row) => (
                      <tr key={`grouped-${row.class_name}`}>
                        <td>{row.class_name}</td>
                        <td>{formatPercent(row.precision)}</td>
                        <td>{formatPercent(row.recall)}</td>
                        <td>{formatPercent(row.f1)}</td>
                        <td>{row.true_positives}</td>
                        <td>{row.false_positives}</td>
                        <td>{row.false_negatives}</td>
                        <td>{row.prompt_count}</td>
                        <td>{row.errors}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h3>Grouped by prompt</h3>
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Prompt</th>
                      <th>P</th>
                      <th>R</th>
                      <th>F1</th>
                      <th>TP</th>
                      <th>FP</th>
                      <th>FN</th>
                      <th>Accepted</th>
                      <th>Rejected</th>
                      <th>GT groups</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.grouped.by_prompt.map((row) => (
                      <tr
                        key={`grouped-${row.class_name}-${row.prompt}`}
                        className={row.error ? "is-error" : undefined}
                      >
                        <td>{row.class_name}</td>
                        <td>{row.error ? `${row.prompt} (${row.error})` : row.prompt}</td>
                        <td>{formatPercent(row.precision)}</td>
                        <td>{formatPercent(row.recall)}</td>
                        <td>{formatPercent(row.f1)}</td>
                        <td>{row.true_positives}</td>
                        <td>{row.false_positives}</td>
                        <td>{row.false_negatives}</td>
                        <td>{row.accepted_predictions}</td>
                        <td>{row.rejected_predictions}</td>
                        <td>{row.ground_truth}</td>
                        <td>{formatMs(row.elapsed_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : null}
          </>
        ) : null}
      </div>
    </section>
  );
}

function SummaryCard(props: { label: string; value: string }) {
  return (
    <div className="benchmark-summary-card">
      <span className="benchmark-summary-value">{props.value}</span>
      <span className="benchmark-summary-label">{props.label}</span>
    </div>
  );
}

export default BenchmarkPanel;
