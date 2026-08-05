import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type MouseEvent as ReactMouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { DepthPinResponse } from "../index-explorer/types";
import CollapsibleSection from "./CollapsibleSection";
import {
  newAnnotationId,
  parseAnnotationGeoJSON,
  saveGeoJSONAs,
  saveGeoJSONToMap,
} from "./geojson";
import visibleIcon from "../assets/visible.png";
import type {
  AnnotationClass,
  AnnotationFeature,
  AnnotationSource,
  FocusTarget,
  LivemapMarker,
  LivemapPolygon,
  RoiPolygon,
} from "./types";
import { isHexColor } from "./types";

const DEFAULT_ACCURACY_M = 5;
const DEFAULT_NEW_CLASS_COLOR = "#ff6b6b";

export type AnnotationDraft = {
  requestId: string;
  status: "resolving" | "resolved" | "error";
  source: Omit<AnnotationSource, "erpU" | "erpV" | "depthM">;
  response: DepthPinResponse | null;
  message: string;
};

export type ExplorerAnnotationWorkspace = ReturnType<typeof useExplorerAnnotationWorkspace>;

export function useExplorerAnnotationWorkspace(mapId: string) {
  const [enabled, setEnabled] = useState(false);
  const [classes, setClasses] = useState<AnnotationClass[]>([]);
  const [annotations, setAnnotations] = useState<AnnotationFeature[]>([]);
  const [hiddenClassKeys, setHiddenClassKeys] = useState<Set<string>>(
    () => new Set(),
  );
  const [activeClassKey, setActiveClassKey] = useState<string | null>(null);
  const [promptInput, setPromptInput] = useState("");
  const [altitudeInput, setAltitudeInput] = useState("");
  const [levelInput, setLevelInput] = useState("");
  const [accuracyInput, setAccuracyInput] = useState(DEFAULT_ACCURACY_M);
  const [draft, setDraft] = useState<AnnotationDraft | null>(null);
  const [focusTarget, setFocusTarget] = useState<FocusTarget | null>(null);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(
    null,
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [roiActive, setRoiActive] = useState(false);
  const [roiPolygon, setRoiPolygon] = useState<RoiPolygon | null>(null);
  const [currentLevel, setCurrentLevel] = useState<string | null>(null);

  useEffect(() => {
    setEnabled(false);
    setClasses([]);
    setAnnotations([]);
    setHiddenClassKeys(new Set());
    setActiveClassKey(null);
    setPromptInput("");
    setAltitudeInput("");
    setLevelInput("");
    setAccuracyInput(DEFAULT_ACCURACY_M);
    setDraft(null);
    setFocusTarget(null);
    setSelectedAnnotationId(null);
    setErrorMessage(null);
    setInfoMessage(null);
    setRoiActive(false);
    setRoiPolygon(null);
    setCurrentLevel(null);
  }, [mapId]);

  const pointClasses = useMemo(
    () => classes.filter((item) => item.annotationType === "point"),
    [classes],
  );
  const activeClass = useMemo(
    () => pointClasses.find((item) => classKey(item) === activeClassKey) ?? null,
    [activeClassKey, pointClasses],
  );
  const allClassesVisible = useMemo(
    () =>
      classes.length > 0 &&
      classes.every((item) => !hiddenClassKeys.has(classKey(item))),
    [classes, hiddenClassKeys],
  );
  const markers: LivemapMarker[] = useMemo(
    () =>
      annotations
        .filter(
          (item) =>
            item.annotationType === "point" &&
            !hiddenClassKeys.has(annotationClassKey(item)),
        )
        .map((item) => {
          const coordinates = item.coordinates as number[];
          const isSelected = item.id === selectedAnnotationId;
          return {
            id: `__annotation_${item.id}`,
            latitude: coordinates[1],
            longitude: coordinates[0],
            color: item.classColor,
            level: item.level,
            radius: isSelected ? 11 : 8,
            strokeColor: isSelected ? "#111827" : "#ffffff",
            strokeWidth: isSelected ? 4 : 2,
            label: item.className,
          };
        }),
    [annotations, hiddenClassKeys, selectedAnnotationId],
  );
  const polygons: LivemapPolygon[] = useMemo(
    () =>
      annotations
        .filter(
          (item) =>
            item.annotationType === "polygon" &&
            !hiddenClassKeys.has(annotationClassKey(item)),
        )
        .map((item) => ({
          id: `__annotation_${item.id}`,
          coordinates: (item.coordinates as number[][][])[0] ?? [],
          color: item.classColor,
          level: item.level,
        })),
    [annotations, hiddenClassKeys],
  );

  const roiCounts = useMemo(
    () =>
      countAnnotationsInPolygon(
        annotations,
        roiPolygon,
        hiddenClassKeys,
        currentLevel,
      ),
    [annotations, roiPolygon, hiddenClassKeys, currentLevel],
  );

  useEffect(() => {
    if (
      selectedAnnotationId &&
      !annotations.some((item) => item.id === selectedAnnotationId)
    ) {
      setSelectedAnnotationId(null);
    }
  }, [annotations, selectedAnnotationId]);

  function toggleRoi() {
    // Toggling the tool either way resets the ROI so each session draws fresh.
    setRoiPolygon(null);
    setRoiActive((current) => !current);
  }

  function clearRoi() {
    setRoiActive(false);
    setRoiPolygon(null);
  }

  function toggleClassVisibility(item: AnnotationClass) {
    const key = classKey(item);
    setHiddenClassKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  function toggleAllClassVisibility() {
    setHiddenClassKeys(
      allClassesVisible
        ? new Set(classes.map(classKey))
        : new Set(),
    );
  }

  function beginDraft(
    requestId: string,
    source: Omit<AnnotationSource, "erpU" | "erpV" | "depthM">,
  ) {
    setFocusTarget(null);
    setDraft({
      requestId,
      status: "resolving",
      source,
      response: null,
      message: "Resolving image point...",
    });
    setErrorMessage(null);
    setInfoMessage(null);
  }

  function resolveDraft(requestId: string, response: DepthPinResponse) {
    setDraft((current) => {
      if (!current || current.requestId !== requestId) {
        return current;
      }
      setAltitudeInput(String(response.altitude));
      setLevelInput(response.level ?? "");
      return {
        ...current,
        status: "resolved",
        response,
        message: `Depth ${response.depth_m.toFixed(2)} m`,
      };
    });
  }

  function rejectDraft(requestId: string, message: string) {
    setDraft((current) => {
      if (!current || current.requestId !== requestId) {
        return current;
      }
      setErrorMessage(message);
      return { ...current, status: "error", message };
    });
  }

  function saveDraft() {
    if (!activeClass) {
      setErrorMessage("Select or create a point class before saving.");
      return;
    }
    if (!draft?.response || draft.status !== "resolved") {
      setErrorMessage("Wait for the image point to resolve before saving.");
      return;
    }
    const response = draft.response;
    const altitude = parseOptionalNumber(altitudeInput);
    const feature: AnnotationFeature = {
      id: newAnnotationId(),
      className: activeClass.name,
      prompt: promptInput.trim() || activeClass.prompt || null,
      classColor: activeClass.color,
      annotationType: "point",
      geometryType: "Point",
      coordinates: [response.longitude, response.latitude],
      altitude: altitude ?? response.altitude,
      level: levelInput.trim() || response.level,
      accuracyM: clampAccuracy(accuracyInput),
      source: {
        ...draft.source,
        erpU: response.erp_u,
        erpV: response.erp_v,
        depthM: response.depth_m,
      },
    };
    setAnnotations((current) => [...current, feature]);
    setDraft(null);
    setInfoMessage(`Added ${activeClass.name} annotation.`);
    setErrorMessage(null);
  }

  function discardDraft() {
    setDraft(null);
  }

  function deleteAnnotation(id: string) {
    setAnnotations((current) => current.filter((item) => item.id !== id));
    setSelectedAnnotationId((current) => (current === id ? null : current));
  }

  function focusAnnotation(annotation: AnnotationFeature) {
    setSelectedAnnotationId(annotation.id);
    const target = focusTargetFor(annotation);
    if (target) {
      setFocusTarget({ ...target });
    }
  }

  function selectAnnotationMarker(markerId: string): boolean {
    const prefix = "__annotation_";
    if (!markerId.startsWith(prefix)) {
      return false;
    }
    const annotationId = markerId.slice(prefix.length);
    const annotation = annotations.find(
      (item) => item.id === annotationId && item.annotationType === "point",
    );
    if (!annotation) {
      return false;
    }
    setSelectedAnnotationId(annotation.id);
    return true;
  }

  return {
    mapId,
    enabled,
    setEnabled,
    classes,
    setClasses,
    annotations,
    setAnnotations,
    hiddenClassKeys,
    setHiddenClassKeys,
    allClassesVisible,
    toggleClassVisibility,
    toggleAllClassVisibility,
    activeClassKey,
    setActiveClassKey,
    activeClass,
    pointClasses,
    promptInput,
    setPromptInput,
    altitudeInput,
    setAltitudeInput,
    levelInput,
    setLevelInput,
    accuracyInput,
    setAccuracyInput,
    draft,
    beginDraft,
    resolveDraft,
    rejectDraft,
    saveDraft,
    discardDraft,
    deleteAnnotation,
    focusAnnotation,
    selectAnnotationMarker,
    focusTarget,
    selectedAnnotationId,
    markers,
    polygons,
    roiActive,
    setRoiActive,
    roiPolygon,
    setRoiPolygon,
    toggleRoi,
    clearRoi,
    roiCounts,
    currentLevel,
    setCurrentLevel,
    errorMessage,
    setErrorMessage,
    infoMessage,
    setInfoMessage,
  };
}

export function ExplorerAnnotationControls(props: {
  workspace: ExplorerAnnotationWorkspace;
}) {
  const { workspace } = props;
  const [newClassName, setNewClassName] = useState("");
  const [newClassColor, setNewClassColor] = useState(DEFAULT_NEW_CLASS_COLOR);
  const [isImportDragOver, setIsImportDragOver] = useState(false);
  const [classMenu, setClassMenu] = useState<{
    classKey: string;
    x: number;
    y: number;
  } | null>(null);
  const [editingClass, setEditingClass] = useState<{
    original: AnnotationClass;
    x: number;
    y: number;
  } | null>(null);
  const [editClassName, setEditClassName] = useState("");
  const [editClassColor, setEditClassColor] = useState(DEFAULT_NEW_CLASS_COLOR);
  const [editClassPrompt, setEditClassPrompt] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!classMenu && !editingClass) {
      return;
    }
    const close = () => {
      setClassMenu(null);
      setEditingClass(null);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close();
      }
    };
    window.addEventListener("click", close);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("blur", close);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("blur", close);
    };
  }, [classMenu, editingClass]);

  function addClass(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = newClassName.trim();
    if (!name) {
      workspace.setErrorMessage("Class name is required.");
      return;
    }
    if (!isHexColor(newClassColor)) {
      workspace.setErrorMessage("Color must be a #RRGGBB hex value.");
      return;
    }
    if (workspace.pointClasses.some((item) => item.name === name)) {
      workspace.setErrorMessage(`A point class named "${name}" already exists.`);
      return;
    }
    const next: AnnotationClass = {
      name,
      color: newClassColor,
      prompt: null,
      annotationType: "point",
    };
    workspace.setClasses((current) => [...current, next]);
    workspace.setActiveClassKey(classKey(next));
    workspace.setErrorMessage(null);
    setNewClassName("");
  }

  function deleteClass(item: AnnotationClass) {
    const matching = workspace.annotations.filter(
      (annotation) =>
        annotation.className === item.name &&
        annotation.annotationType === item.annotationType,
    );
    const message = matching.length
      ? `Delete class "${item.name}" and its ${matching.length} annotations?`
      : `Delete class "${item.name}"?`;
    if (!window.confirm(message)) {
      return;
    }
    workspace.setClasses((current) =>
      current.filter((candidate) => classKey(candidate) !== classKey(item)),
    );
    workspace.setAnnotations((current) =>
      current.filter(
        (annotation) =>
          !(
            annotation.className === item.name &&
            annotation.annotationType === item.annotationType
          ),
      ),
    );
    if (workspace.hiddenClassKeys.has(classKey(item))) {
      workspace.toggleClassVisibility(item);
    }
    if (workspace.activeClassKey === classKey(item)) {
      workspace.setActiveClassKey(null);
      workspace.discardDraft();
    }
  }

  function openClassMenu(
    event: ReactMouseEvent<HTMLElement>,
    item: AnnotationClass,
  ) {
    event.preventDefault();
    event.stopPropagation();
    setEditingClass(null);
    setClassMenu({
      classKey: classKey(item),
      x: event.clientX,
      y: event.clientY,
    });
  }

  function openClassEditor(item: AnnotationClass, position: { x: number; y: number }) {
    setClassMenu(null);
    setEditingClass({ original: item, x: position.x, y: position.y });
    setEditClassName(item.name);
    setEditClassColor(item.color);
    setEditClassPrompt(item.prompt ?? promptForClass(item) ?? "");
  }

  function saveClassEdit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!editingClass) {
      return;
    }
    const original = editingClass.original;
    const nextName = editClassName.trim();
    const nextColor = editClassColor;
    if (!nextName) {
      workspace.setErrorMessage("Class name is required.");
      return;
    }
    if (!isHexColor(nextColor)) {
      workspace.setErrorMessage("Color must be a #RRGGBB hex value.");
      return;
    }
    const originalKey = classKey(original);
    const nextPrompt = editClassPrompt.trim() || null;
    const nextClass: AnnotationClass = {
      name: nextName,
      color: nextColor,
      prompt: nextPrompt,
      annotationType: original.annotationType,
    };
    const nextKey = classKey(nextClass);
    if (
      nextKey !== originalKey &&
      workspace.classes.some((item) => classKey(item) === nextKey)
    ) {
      workspace.setErrorMessage(`A ${original.annotationType} class named "${nextName}" already exists.`);
      return;
    }
    workspace.setClasses((current) =>
      current.map((item) => (classKey(item) === originalKey ? nextClass : item)),
    );
    workspace.setAnnotations((current) =>
      current.map((annotation) =>
        annotation.className === original.name &&
        annotation.annotationType === original.annotationType
          ? {
              ...annotation,
              className: nextName,
              classColor: nextColor,
              prompt: nextPrompt,
            }
          : annotation,
      ),
    );
    if (workspace.hiddenClassKeys.has(originalKey)) {
      workspace.setHiddenClassKeys((current) => {
        const next = new Set(current);
        next.delete(originalKey);
        next.add(nextKey);
        return next;
      });
    }
    if (workspace.activeClassKey === originalKey) {
      workspace.setActiveClassKey(nextKey);
    }
    workspace.setInfoMessage(`Updated ${original.name} class.`);
    workspace.setErrorMessage(null);
    setEditingClass(null);
  }

  function promptForClass(item: AnnotationClass): string | null {
    return (
      workspace.annotations.find(
        (annotation) =>
          annotation.className === item.name &&
          annotation.annotationType === item.annotationType &&
          annotation.prompt,
      )?.prompt ?? null
    );
  }

  function importFile(file: File) {
    if (!isGeoJsonFile(file)) {
      workspace.setErrorMessage("Choose a .geojson or .json file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const result = parseAnnotationGeoJSON(
          String(reader.result ?? ""),
          workspace.classes,
          workspace.annotations,
        );
        workspace.setClasses(result.classes);
        workspace.setAnnotations(result.annotations);
        workspace.setInfoMessage(
          `Loaded ${result.importedAnnotations} annotations (${result.createdClasses} new classes).`,
        );
        workspace.setErrorMessage(null);
      } catch (error) {
        workspace.setErrorMessage(error instanceof Error ? error.message : String(error));
      }
    };
    reader.onerror = () => workspace.setErrorMessage("Failed to read file.");
    reader.readAsText(file);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) {
      importFile(file);
    }
  }

  function handleDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setIsImportDragOver(false);
    const files = event.dataTransfer.files;
    if (files.length !== 1) {
      workspace.setErrorMessage("Drop one GeoJSON file at a time.");
      return;
    }
    importFile(files[0]);
  }

  async function handleSaveToMap() {
    try {
      const result = await saveGeoJSONToMap(
        workspace.mapId,
        workspace.annotations,
      );
      workspace.setInfoMessage(
        `Saved ${result.annotation_count} annotations to ${result.relative_path}.`,
      );
      workspace.setErrorMessage(null);
    } catch (error) {
      workspace.setErrorMessage(
        `Could not save annotations: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }

  async function handleSaveAs() {
    try {
      const result = await saveGeoJSONAs(
        workspace.mapId,
        workspace.annotations,
      );
      if (result === "cancelled") {
        return;
      }
      workspace.setInfoMessage(
        result === "saved"
          ? `Saved ${workspace.annotations.length} annotations.`
          : `Downloaded ${workspace.annotations.length} annotations.`,
      );
      workspace.setErrorMessage(null);
    } catch (error) {
      workspace.setErrorMessage(
        `Could not save GeoJSON: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }

  const menuClass = classMenu
    ? workspace.classes.find((item) => classKey(item) === classMenu.classKey) ?? null
    : null;

  return (
    <CollapsibleSection
      title="Annotation office"
      summary={`${workspace.annotations.length} saved`}
      sectionClassName="explorer-annotation-toolbar"
      defaultOpen
    >
      <div className="explorer-annotation-mode-row">
        <label className="inline-check">
          <input
            type="checkbox"
            checked={workspace.enabled}
            onChange={(event) => {
              workspace.setEnabled(event.target.checked);
              if (!event.target.checked) {
                workspace.discardDraft();
              }
            }}
          />
          Annotation mode
        </label>
        <label className="explorer-annotation-active-class">
          Class
          <select
            value={workspace.activeClassKey ?? ""}
            onChange={(event) =>
              workspace.setActiveClassKey(event.target.value || null)
            }
          >
            <option value="">No class selected</option>
            {workspace.pointClasses.map((item) => (
              <option key={classKey(item)} value={classKey(item)}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <label className="explorer-annotation-prompt">
          Prompt
          <input
            value={workspace.promptInput}
            onChange={(event) => workspace.setPromptInput(event.target.value)}
            placeholder={workspace.activeClass?.prompt || "Optional"}
          />
        </label>
      </div>

      <div className="explorer-annotation-primary-grid">
        <section>
          <div className="explorer-class-heading">
            <h4>Classes</h4>
            <button
              type="button"
              className={`annotation-visibility-button${
                workspace.allClassesVisible ? "" : " is-hidden"
              }`}
              disabled={!workspace.classes.length}
              aria-label={
                workspace.allClassesVisible
                  ? "Hide all annotation classes"
                  : "Show all annotation classes"
              }
              title={
                workspace.allClassesVisible
                  ? "Hide all annotation classes"
                  : "Show all annotation classes"
              }
              onClick={workspace.toggleAllClassVisibility}
            >
              <img src={visibleIcon} alt="" />
            </button>
          </div>
          {workspace.classes.length ? (
            <ul className="class-list">
              {workspace.classes.map((item) => {
                const isVisible = !workspace.hiddenClassKeys.has(classKey(item));
                return (
                <li
                  key={classKey(item)}
                  className={`class-row${
                    workspace.activeClassKey === classKey(item) ? " is-active" : ""
                  }${isVisible ? "" : " is-hidden"}`}
                  onContextMenu={(event) => openClassMenu(event, item)}
                >
                  <button
                    type="button"
                    className={`class-select${
                      item.annotationType === "point" ? "" : " is-readonly"
                    }`}
                    disabled={item.annotationType !== "point"}
                    onClick={() =>
                      item.annotationType === "point"
                        ? workspace.setActiveClassKey(classKey(item))
                        : undefined
                    }
                  >
                    <span
                      className="color-swatch"
                      style={{ background: item.color }}
                    />
                    <span className="class-name">{item.name}</span>
                    <span className="class-type">{item.annotationType}</span>
                  </button>
                  <button
                    type="button"
                    className={`annotation-visibility-button${
                      isVisible ? "" : " is-hidden"
                    }`}
                    aria-label={`${isVisible ? "Hide" : "Show"} class ${item.name}`}
                    title={`${isVisible ? "Hide" : "Show"} class ${item.name}`}
                    onClick={() => workspace.toggleClassVisibility(item)}
                  >
                    <img src={visibleIcon} alt="" />
                  </button>
                  <button
                    type="button"
                    className="class-delete"
                    aria-label={`Delete class ${item.name}`}
                    onClick={() => deleteClass(item)}
                  >
                    ×
                  </button>
                </li>
                );
              })}
            </ul>
          ) : (
            <p className="muted">Create a point class or load annotations.</p>
          )}
          <form className="explorer-new-class-form" onSubmit={addClass}>
            <input
              value={newClassName}
              onChange={(event) => setNewClassName(event.target.value)}
              placeholder="New class"
              aria-label="New annotation class"
            />
            <input
              type="color"
              value={newClassColor}
              onChange={(event) => setNewClassColor(event.target.value)}
              aria-label="New class color"
            />
            <button className="secondary-button" type="submit">
              Add
            </button>
          </form>
          {menuClass && classMenu ? (
            <ul
              className="annotation-context-menu"
              style={{ left: classMenu.x, top: classMenu.y }}
              onClick={(event) => event.stopPropagation()}
              onContextMenu={(event) => event.preventDefault()}
            >
              <li>
                <button
                  type="button"
                  className="annotation-context-menu-item"
                  onClick={() => openClassEditor(menuClass, classMenu)}
                >
                  Edit
                </button>
              </li>
            </ul>
          ) : null}
          {editingClass ? (
            <form
              className="annotation-class-edit-panel"
              style={{ left: editingClass.x, top: editingClass.y }}
              onSubmit={saveClassEdit}
              onClick={(event) => event.stopPropagation()}
              onContextMenu={(event) => event.preventDefault()}
            >
              <label>
                Class name
                <input
                  value={editClassName}
                  onChange={(event) => setEditClassName(event.target.value)}
                  autoFocus
                />
              </label>
              <label>
                Color
                <input
                  type="color"
                  value={editClassColor}
                  onChange={(event) => setEditClassColor(event.target.value)}
                />
              </label>
              <label>
                Prompt (optional)
                <input
                  value={editClassPrompt}
                  onChange={(event) => setEditClassPrompt(event.target.value)}
                  placeholder="Defaults to no prompt"
                />
              </label>
              <div className="button-row">
                <button className="primary-button" type="submit">
                  Save
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => setEditingClass(null)}
                >
                  Cancel
                </button>
              </div>
            </form>
          ) : null}
        </section>

        <section>
          <h4>Pending point</h4>
          {workspace.draft ? (
            <>
              <p className={`object-search-depth-pin-caption is-${workspace.draft.status}`}>
                {workspace.draft.message}
                {workspace.draft.response
                  ? ` | ${workspace.draft.response.latitude.toFixed(7)}, ${workspace.draft.response.longitude.toFixed(7)}`
                  : ""}
              </p>
              <div className="button-row">
                <button
                  className="primary-button"
                  type="button"
                  disabled={
                    workspace.draft.status !== "resolved" || !workspace.activeClass
                  }
                  onClick={workspace.saveDraft}
                >
                  Save annotation
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={workspace.discardDraft}
                >
                  Discard
                </button>
              </div>
            </>
          ) : (
            <p className="muted">No pending image point.</p>
          )}
        </section>
      </div>

      <div className="explorer-annotation-secondary-grid">
        <CollapsibleSection
          title="Metadata"
          sectionClassName="collapsible-subsection annotation-compact-panel"
          defaultOpen={false}
        >
          <div className="form-row">
            <label>
              Altitude (m)
              <input
                type="number"
                step="0.1"
                value={workspace.altitudeInput}
                onChange={(event) => workspace.setAltitudeInput(event.target.value)}
                placeholder="resolved"
              />
            </label>
            <label>
              Level
              <input
                value={workspace.levelInput}
                onChange={(event) => workspace.setLevelInput(event.target.value)}
                placeholder="resolved"
              />
            </label>
          </div>
          <label>
            Accuracy (m): {workspace.accuracyInput.toFixed(1)}
            <input
              type="range"
              min={0}
              max={100}
              step={0.5}
              value={workspace.accuracyInput}
              onChange={(event) => workspace.setAccuracyInput(Number(event.target.value))}
            />
          </label>
        </CollapsibleSection>

        <CollapsibleSection
          title="ROI count"
          sectionClassName="collapsible-subsection annotation-compact-panel"
          defaultOpen={false}
        >
          <div className="explorer-annotation-roi">
            <p className="muted">
              Counts visible point annotations inside a drawn polygon.
              {` On level ${
                workspace.currentLevel === null ? "—" : workspace.currentLevel
              }.`}
            </p>
            {workspace.roiCounts ? (
              workspace.roiCounts.total ? (
                <>
                  <ul className="roi-count-list">
                    {workspace.roiCounts.perClass.map((entry) => (
                      <li key={entry.name} className="roi-count-row">
                        <span
                          className="color-swatch"
                          style={{ background: entry.color }}
                        />
                        <span className="roi-count-name">{entry.name}</span>
                        <span className="roi-count-value">{entry.count}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="roi-count-total">
                    Total: <strong>{workspace.roiCounts.total}</strong>
                  </p>
                </>
              ) : (
                <p className="muted">No annotations in the selected region.</p>
              )
            ) : (
              <p className="muted">Draw an ROI on the map to count annotations.</p>
            )}
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Load / Save"
          sectionClassName="collapsible-subsection annotation-compact-panel"
          defaultOpen={false}
        >
          <div
            className={`explorer-annotation-import${
              isImportDragOver ? " is-drag-over" : ""
            }`}
            onDragEnter={(event) => {
              event.preventDefault();
              setIsImportDragOver(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsImportDragOver(false)}
            onDrop={handleDrop}
          >
            <div className="button-row">
              <button
                className="secondary-button"
                type="button"
                onClick={() => fileInputRef.current?.click()}
              >
                Load GeoJSON
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={!workspace.annotations.length}
                onClick={() => void handleSaveToMap()}
              >
                Save
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={!workspace.annotations.length}
                onClick={() => void handleSaveAs()}
              >
                Save As…
              </button>
              <button
                className="danger-button"
                type="button"
                disabled={!workspace.annotations.length}
                onClick={() => {
                  if (
                    window.confirm(
                      `Delete all ${workspace.annotations.length} annotations?`,
                    )
                  ) {
                    workspace.setAnnotations([]);
                    workspace.discardDraft();
                  }
                }}
              >
                Clear all
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".geojson,.json,application/geo+json,application/json"
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
          </div>
        </CollapsibleSection>
      </div>
    </CollapsibleSection>
  );
}

export function ExplorerAnnotationList(props: {
  workspace: ExplorerAnnotationWorkspace;
}) {
  const { workspace } = props;
  const selectedRowRef = useRef<HTMLLIElement | null>(null);
  const summaryRows = useMemo(() => {
    const rows = new Map<
      string,
      {
        color: string;
        count: number;
        name: string;
        type: string;
      }
    >();

    for (const item of workspace.classes) {
      rows.set(classKey(item), {
        color: item.color,
        count: 0,
        name: item.name,
        type: item.annotationType,
      });
    }

    for (const item of workspace.annotations) {
      const key = annotationClassKey(item);
      const row =
        rows.get(key) ??
        {
          color: item.classColor,
          count: 0,
          name: item.className,
          type: item.annotationType,
        };
      row.count += 1;
      rows.set(key, row);
    }

    return Array.from(rows.values());
  }, [workspace.annotations, workspace.classes]);

  useEffect(() => {
    if (!workspace.selectedAnnotationId) {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      selectedRowRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [workspace.selectedAnnotationId]);

  return (
    <CollapsibleSection
      sectionClassName="annotation-list-section"
      title="Annotations history"
      summary={
        workspace.annotations.length
          ? String(workspace.annotations.length)
          : undefined
      }
      defaultOpen={workspace.annotations.length <= 10}
      forceOpen={Boolean(workspace.selectedAnnotationId)}
    >
      <CollapsibleSection
        title="Summary"
        sectionClassName="collapsible-subsection annotation-history-summary"
        defaultOpen
      >
        <div className="annotation-history-total">
          <span>Total annotations</span>
          <strong>{workspace.annotations.length}</strong>
        </div>
        {summaryRows.length ? (
          <ul className="annotation-summary-list">
            {summaryRows.map((item) => (
              <li key={`${item.name}::${item.type}`}>
                <span
                  className="color-swatch"
                  style={{ background: item.color }}
                />
                <span>
                  {item.name}
                  <small>{item.type}</small>
                </span>
                <strong>{item.count}</strong>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No classes yet.</p>
        )}
      </CollapsibleSection>
      {workspace.annotations.length ? (
        <ul className="annotation-list annotation-list-scroll">
          {workspace.annotations.map((item) => {
            const coordinates = annotationCenter(item);
            return (
              <li
                key={item.id}
                ref={
                  workspace.selectedAnnotationId === item.id
                    ? selectedRowRef
                    : undefined
                }
                className={`annotation-row${
                  workspace.selectedAnnotationId === item.id ? " is-selected" : ""
                }`}
              >
                <span
                  className="color-swatch"
                  style={{ background: item.classColor }}
                />
                <span className="annotation-meta">
                  <strong>{item.className}</strong>
                  {item.prompt ? <span>{item.prompt}</span> : null}
                  <small>
                    {item.annotationType}
                    {coordinates
                      ? ` | ${coordinates.latitude.toFixed(7)}, ${coordinates.longitude.toFixed(7)}`
                      : ""}
                    {item.source ? ` | depth ${item.source.depthM.toFixed(2)} m` : ""}
                    {item.level !== null ? ` | level ${item.level}` : ""}
                    {` | ±${item.accuracyM.toFixed(1)} m`}
                  </small>
                </span>
                <button
                  className="link-button"
                  type="button"
                  onClick={() => workspace.focusAnnotation(item)}
                >
                  Focus
                </button>
                <button
                  className="link-button danger"
                  type="button"
                  onClick={() => workspace.deleteAnnotation(item.id)}
                >
                  Delete
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted">No annotations yet.</p>
      )}
    </CollapsibleSection>
  );
}

function classKey(item: AnnotationClass): string {
  return `${item.name}::${item.annotationType}`;
}

function annotationClassKey(item: AnnotationFeature): string {
  return `${item.className}::${item.annotationType}`;
}

function isGeoJsonFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return (
    name.endsWith(".geojson") ||
    name.endsWith(".json") ||
    file.type === "application/geo+json" ||
    file.type === "application/json" ||
    file.type === ""
  );
}

function parseOptionalNumber(value: string): number | null {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function clampAccuracy(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

export type RoiClassCount = { name: string; color: string; count: number };
export type RoiCounts = { perClass: RoiClassCount[]; total: number };

/**
 * Normalize a level value to the same canonical form the livemap uses for its
 * indoor-level feature filter (numeric levels collapse to their number string),
 * so annotation levels compare equal to the map's reported current level.
 */
function canonicalLevel(value: string | null | undefined): string | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const asNumber = Number(value);
  return Number.isNaN(asNumber) ? String(value) : String(asNumber);
}

/**
 * Ray-casting point-in-polygon test against an open ring of vertices (the first
 * vertex is treated as reconnected to the last).
 */
function pointInPolygon(
  longitude: number,
  latitude: number,
  ring: RoiPolygon,
): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].longitude;
    const yi = ring[i].latitude;
    const xj = ring[j].longitude;
    const yj = ring[j].latitude;
    const intersects =
      yi > latitude !== yj > latitude &&
      longitude < ((xj - xi) * (latitude - yi)) / (yj - yi) + xi;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Count point annotations whose location falls within `polygon`, grouped by
 * class. Polygon annotations are excluded, classes hidden on the map (in
 * `hiddenClassKeys`) are skipped, and — when `currentLevel` is provided —
 * annotations not on the selected livemap level are excluded, so counts match
 * what is visible on the map.
 */
export function countAnnotationsInPolygon(
  annotations: AnnotationFeature[],
  polygon: RoiPolygon | null,
  hiddenClassKeys: Set<string>,
  currentLevel: string | null = null,
): RoiCounts | null {
  if (!polygon || polygon.length < 3) {
    return null;
  }
  const activeLevel = canonicalLevel(currentLevel);
  const byClass = new Map<string, RoiClassCount>();
  let total = 0;
  for (const item of annotations) {
    if (item.annotationType !== "point") {
      continue;
    }
    if (hiddenClassKeys.has(annotationClassKey(item))) {
      continue;
    }
    if (canonicalLevel(item.level) !== activeLevel) {
      continue;
    }
    const coordinates = item.coordinates as number[];
    if (coordinates.length < 2) {
      continue;
    }
    const [longitude, latitude] = coordinates;
    if (!pointInPolygon(longitude, latitude, polygon)) {
      continue;
    }
    total += 1;
    const existing = byClass.get(item.className);
    if (existing) {
      existing.count += 1;
    } else {
      byClass.set(item.className, {
        name: item.className,
        color: item.classColor,
        count: 1,
      });
    }
  }
  const perClass = [...byClass.values()].sort(
    (a, b) => b.count - a.count || a.name.localeCompare(b.name),
  );
  return { perClass, total };
}

function annotationCenter(annotation: AnnotationFeature): FocusTarget | null {
  if (annotation.annotationType === "point") {
    const coordinates = annotation.coordinates as number[];
    if (coordinates.length < 2) {
      return null;
    }
    return {
      latitude: coordinates[1],
      longitude: coordinates[0],
      level: annotation.level,
    };
  }
  const ring = (annotation.coordinates as number[][][])[0];
  if (!ring?.length) {
    return null;
  }
  const hasDuplicateEnd =
    ring.length > 1 &&
    ring[0][0] === ring[ring.length - 1][0] &&
    ring[0][1] === ring[ring.length - 1][1];
  const points = hasDuplicateEnd ? ring.slice(0, -1) : ring;
  return {
    longitude: points.reduce((sum, point) => sum + point[0], 0) / points.length,
    latitude: points.reduce((sum, point) => sum + point[1], 0) / points.length,
    level: annotation.level,
  };
}

function focusTargetFor(annotation: AnnotationFeature): FocusTarget | null {
  return annotationCenter(annotation);
}
