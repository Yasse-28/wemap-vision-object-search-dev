import type { FormEvent } from "react";
import { useEffect, useRef, useState } from "react";

import CollapsibleOnlineOverrides from "./CollapsibleOnlineOverrides";
import type {
  ObjectSearchMode,
  OnlineLocalizeOverrides,
  QueryInputMode,
} from "./types";

const COLLAPSED_STORAGE_KEY = "object-search-gui.searchCardCollapsed";
const ADVANCED_OPEN_STORAGE_KEY = "object-search-gui.advancedOptionsOpen";

function readStoredBoolean(key: string): boolean {
  try {
    return localStorage.getItem(key) === "true";
  } catch {
    return false;
  }
}

type Props = {
  isMapKnown: boolean;
  isLoading: boolean;
  error: string | null;
  resultSummary: string | null;
  queryInputMode: QueryInputMode;
  text: string;
  imageFile: File | null;
  numResults: number;
  searchMode: ObjectSearchMode;
  activeKeyframeId: string | null;
  bboxMode: boolean;
  localizeSensitivity: number;
  maxObservationsPerCluster: number;
  onlineOverrides: OnlineLocalizeOverrides;
  useRemoteServer: boolean;
  remoteServerUrl: string;
  showDebugRays: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onDismissError: () => void;
  onQueryInputModeChange: (mode: QueryInputMode) => void;
  onTextChange: (value: string) => void;
  onImageFileChange: (file: File | null) => void;
  onNumResultsChange: (value: number) => void;
  onSearchModeChange: (mode: ObjectSearchMode) => void;
  onBboxModeChange: (enabled: boolean) => void;
  onLocalizeSensitivityChange: (value: number) => void;
  onMaxObservationsPerClusterChange: (value: number) => void;
  onOnlineOverridesChange: (value: OnlineLocalizeOverrides) => void;
  onUseRemoteServerChange: (enabled: boolean) => void;
  onRemoteServerUrlChange: (value: string) => void;
  onShowDebugRaysChange: (enabled: boolean) => void;
};

function SearchControlsCard(props: Props) {
  const [collapsed, setCollapsed] = useState(() =>
    readStoredBoolean(COLLAPSED_STORAGE_KEY),
  );
  const [advancedOpen, setAdvancedOpen] = useState(() =>
    readStoredBoolean(ADVANCED_OPEN_STORAGE_KEY),
  );
  const imageInputRef = useRef<HTMLInputElement>(null);
  const isLocalizeMode = props.searchMode !== "text";

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSED_STORAGE_KEY, String(collapsed));
    } catch {
      // The UI remains usable when storage is unavailable.
    }
  }, [collapsed]);

  useEffect(() => {
    try {
      localStorage.setItem(ADVANCED_OPEN_STORAGE_KEY, String(advancedOpen));
    } catch {
      // The UI remains usable when storage is unavailable.
    }
  }, [advancedOpen]);

  return (
    <section className={`os-search-card${collapsed ? " is-collapsed" : ""}`}>
      <header className="os-search-card-header">
        <span className="os-search-card-title">Object Search</span>
        {props.resultSummary ? (
          <span className="os-pane-subtitle">{props.resultSummary}</span>
        ) : null}
        {isLocalizeMode ? (
          <span className="os-search-card-summary">
            sens {props.localizeSensitivity}% · max {props.maxObservationsPerCluster} · {" "}
            {props.onlineOverrides.merge_radius.toFixed(1)} m
          </span>
        ) : null}
        <button
          type="button"
          className="os-search-card-collapse"
          onClick={() => setCollapsed((current) => !current)}
          aria-label={collapsed ? "Expand search controls" : "Collapse search controls"}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? "▾" : "▴"}
        </button>
      </header>

      {collapsed ? null : (
        <div className="os-search-card-body">
          {props.error ? (
            <div className="object-search-error-banner" role="alert">
              <span>{props.error}</span>
              <button type="button" onClick={props.onDismissError} aria-label="Dismiss">
                ×
              </button>
            </div>
          ) : null}

          <form className="os-search-form" onSubmit={props.onSubmit}>
            <div className="object-search-toolbar-row os-search-query-row">
              {props.queryInputMode === "text" ? (
                <input
                  className="object-search-input"
                  type="text"
                  value={props.text}
                  onChange={(event) => props.onTextChange(event.target.value)}
                  placeholder="Search objects…"
                  disabled={!props.isMapKnown || props.isLoading}
                />
              ) : props.imageFile ? (
                <div className="object-search-selected-image-chip">
                  <span>{props.imageFile.name}</span>
                  <button
                    type="button"
                    className="object-search-chip-clear-button"
                    onClick={() => props.onImageFileChange(null)}
                    disabled={props.isLoading}
                    aria-label="Clear image"
                  >
                    ×
                  </button>
                </div>
              ) : (
                <span className="object-search-status-row">Select an image query below</span>
              )}
              <input
                className="object-search-num-results"
                type="number"
                min={1}
                max={1000}
                value={props.numResults}
                onChange={(event) => props.onNumResultsChange(Number(event.target.value))}
                disabled={!props.isMapKnown || props.isLoading}
                title="Number of results"
              />
              <button
                className="object-search-button"
                type="submit"
                disabled={!props.isMapKnown || props.isLoading}
              >
                {props.isLoading ? "Searching…" : "Search"}
              </button>
            </div>

            <div className="object-search-toolbar-row os-search-modes-row">
              <div className="object-search-mode-toggle" role="group" aria-label="Query input">
                <button
                  type="button"
                  className={`object-search-secondary-button${
                    props.queryInputMode === "text" ? " active-query" : ""
                  }`}
                  onClick={() => props.onQueryInputModeChange("text")}
                >
                  Text
                </button>
                <button
                  type="button"
                  className={`object-search-secondary-button${
                    props.queryInputMode === "image" ? " active-query" : ""
                  }`}
                  onClick={() => props.onQueryInputModeChange("image")}
                >
                  Image
                </button>
                <button
                  type="button"
                  className={`object-search-secondary-button${
                    props.bboxMode ? " active-query" : ""
                  }`}
                  onClick={() => {
                    if (!props.activeKeyframeId) return;
                    props.onQueryInputModeChange("image");
                    props.onBboxModeChange(!props.bboxMode);
                  }}
                  disabled={!props.activeKeyframeId}
                  title={
                    props.activeKeyframeId
                      ? "Draw a bbox on the panorama"
                      : "Select a keyframe first"
                  }
                >
                  Draw bbox
                </button>
              </div>

              {props.queryInputMode === "image" ? (
                <>
                  <input
                    ref={imageInputRef}
                    className="object-search-image-input"
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={(event) =>
                      props.onImageFileChange(event.target.files?.[0] ?? null)
                    }
                    disabled={!props.isMapKnown || props.isLoading}
                  />
                  <button
                    type="button"
                    className={`object-search-secondary-button${
                      props.imageFile ? " active-query" : ""
                    }`}
                    onClick={() => imageInputRef.current?.click()}
                    disabled={!props.isMapKnown || props.isLoading}
                  >
                    {props.imageFile ? "Change image" : "Pick image"}
                  </button>
                </>
              ) : null}

              <div className="object-search-mode-toggle" role="group" aria-label="Search mode">
                <button
                  type="button"
                  className={`object-search-secondary-button${
                    props.searchMode === "text" ? " active-query" : ""
                  }`}
                  onClick={() => props.onSearchModeChange("text")}
                >
                  Text search
                </button>
                <button
                  type="button"
                  className={`object-search-secondary-button${
                    props.searchMode === "localize-online" ? " active-query" : ""
                  }`}
                  onClick={() => props.onSearchModeChange("localize-online")}
                >
                  Localize
                </button>
              </div>

              <button
                type="button"
                className={`object-search-secondary-button os-advanced-toggle${
                  advancedOpen ? " active" : ""
                }`}
                onClick={() => setAdvancedOpen((current) => !current)}
                aria-expanded={advancedOpen}
              >
                Options avancées {advancedOpen ? "▴" : "▾"}
              </button>
            </div>

            {advancedOpen ? (
              <div className="os-search-advanced">
                {isLocalizeMode ? (
                  <div className="object-search-toolbar-row object-search-toolbar-controls">
                    <div className="object-search-slider-control">
                      <div className="object-search-slider-header">
                        <label htmlFor="os-sensitivity">Sensibilité</label>
                        <span>{props.localizeSensitivity}%</span>
                      </div>
                      <input
                        id="os-sensitivity"
                        type="range"
                        min={0}
                        max={100}
                        step={1}
                        value={props.localizeSensitivity}
                        onChange={(event) =>
                          props.onLocalizeSensitivityChange(Number(event.target.value))
                        }
                        disabled={!props.isMapKnown || props.isLoading}
                      />
                      <div className="object-search-slider-help">
                        Shows localizations with match_score &gt;= 1 - sensitivity / 100.
                      </div>
                    </div>
                    <div className="object-search-slider-control">
                      <div className="object-search-slider-header">
                        <label htmlFor="os-max-obs">Max observations par cluster</label>
                        <span>{props.maxObservationsPerCluster}</span>
                      </div>
                      <input
                        id="os-max-obs"
                        type="range"
                        min={1}
                        max={2000}
                        step={1}
                        value={Math.min(props.maxObservationsPerCluster, 2000)}
                        onChange={(event) =>
                          props.onMaxObservationsPerClusterChange(Number(event.target.value))
                        }
                        disabled={!props.isMapKnown || props.isLoading}
                      />
                    </div>
                    {props.searchMode === "localize-online" ? (
                      <div className="object-search-slider-control">
                        <div className="object-search-slider-header">
                          <label htmlFor="os-merge-radius">Rayon de fusion</label>
                          <span>{props.onlineOverrides.merge_radius.toFixed(1)} m</span>
                        </div>
                        <input
                          id="os-merge-radius"
                          type="range"
                          min={0}
                          max={7}
                          step={0.1}
                          value={props.onlineOverrides.merge_radius}
                          onChange={(event) =>
                            props.onOnlineOverridesChange({
                              ...props.onlineOverrides,
                              merge_radius: Number(event.target.value),
                            })
                          }
                          disabled={!props.isMapKnown || props.isLoading}
                        />
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {props.searchMode === "localize-online" ? (
                  <>
                    <div className="object-search-toolbar-row checkbox-row--compact os-remote-row">
                      <label
                        title="Le service de production miroir n’expose pas /localize ; utilisez une instance bricks distante."
                      >
                        <input
                          type="checkbox"
                          checked={props.useRemoteServer}
                          onChange={(event) =>
                            props.onUseRemoteServerChange(event.target.checked)
                          }
                        />
                        Instance bricks distante
                      </label>
                      {props.useRemoteServer ? (
                        <input
                          className="object-search-input os-remote-url-input"
                          type="url"
                          value={props.remoteServerUrl}
                          onChange={(event) =>
                            props.onRemoteServerUrlChange(event.target.value)
                          }
                          placeholder="https://…/{map_id}/object-search/localize"
                          spellCheck={false}
                          title="URL d’une instance bricks exposant /localize"
                        />
                      ) : null}
                    </div>
                    <CollapsibleOnlineOverrides
                      overrides={props.onlineOverrides}
                      onChange={props.onOnlineOverridesChange}
                    />
                  </>
                ) : null}

                {isLocalizeMode ? (
                  <div className="object-search-toolbar-row checkbox-row--compact">
                    <label>
                      <input
                        type="checkbox"
                        checked={props.showDebugRays}
                        onChange={(event) =>
                          props.onShowDebugRaysChange(event.target.checked)
                        }
                      />
                      Show depth rays
                    </label>
                  </div>
                ) : null}
              </div>
            ) : null}
          </form>
        </div>
      )}
    </section>
  );
}

export default SearchControlsCard;
