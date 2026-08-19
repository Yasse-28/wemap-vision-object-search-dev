import { useEffect, useMemo, useRef, useState } from "react";

export type ObjectIdOption = {
  objectId: string;
  className: string | null;
  /** How many saved points already carry this id — is this really the same object? */
  count: number;
};

/**
 * The object id field: free text, with the ids already in use one click away.
 *
 * It was a `<datalist>`, which only ever shows what matches what is already typed and
 * hides the list behind a browser-specific affordance. Reusing an id is the whole
 * point of the field — it is what makes two views the same physical object — so the
 * list has to be openable in full, and it has to say which class each id belongs to,
 * because `chair-017` and `door-017` are told apart by that alone.
 */
function ObjectIdCombobox(props: {
  value: string;
  options: ObjectIdOption[];
  placeholder?: string;
  describedBy?: string;
  onChange: (value: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  /**
   * What to filter on, or `null` for the whole list.
   *
   * Deliberately not the field's value: the field arrives pre-filled with the next
   * free id (`maglock-002`), which matches nothing, so filtering on it hid every
   * saved object behind a message saying there were none. Only what the annotator
   * has actually typed since opening narrows the list.
   */
  const [typedQuery, setTypedQuery] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const query = typedQuery?.trim().toLowerCase() ?? "";
  const matches = useMemo(() => {
    if (!query) {
      return props.options;
    }
    return props.options.filter(
      (option) =>
        option.objectId.toLowerCase().includes(query) ||
        (option.className ?? "").toLowerCase().includes(query),
    );
  }, [props.options, query]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const close = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [isOpen]);

  useEffect(() => {
    setActiveIndex(-1);
  }, [query, isOpen]);

  function choose(objectId: string) {
    props.onChange(objectId);
    setTypedQuery(null);
    setIsOpen(false);
    inputRef.current?.focus();
  }

  function openFullList() {
    setTypedQuery(null);
    setIsOpen(true);
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      setIsOpen(false);
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!props.options.length) {
        return;
      }
      event.preventDefault();
      if (!isOpen) {
        // Opening resets the highlight, so the first press opens and the next moves.
        openFullList();
        return;
      }
      if (!matches.length) {
        return;
      }
      setActiveIndex((current) => {
        const step = event.key === "ArrowDown" ? 1 : -1;
        const next = current + step;
        if (next < 0) {
          return matches.length - 1;
        }
        return next >= matches.length ? 0 : next;
      });
      return;
    }
    if (event.key === "Enter" && isOpen && activeIndex >= 0 && matches[activeIndex]) {
      event.preventDefault();
      choose(matches[activeIndex].objectId);
    }
  }

  return (
    <div className="object-id-combobox" ref={containerRef}>
      <input
        ref={inputRef}
        value={props.value}
        role="combobox"
        aria-expanded={isOpen}
        aria-autocomplete="list"
        aria-describedby={props.describedBy}
        placeholder={props.placeholder}
        onChange={(event) => {
          props.onChange(event.target.value);
          setTypedQuery(event.target.value);
          setIsOpen(true);
        }}
        onKeyDown={onKeyDown}
      />
      <button
        type="button"
        className="object-id-combobox-toggle"
        tabIndex={-1}
        aria-label={isOpen ? "Hide existing object IDs" : "Show existing object IDs"}
        title="Existing object IDs"
        disabled={!props.options.length}
        onClick={() => (isOpen ? setIsOpen(false) : openFullList())}
      >
        ▾
      </button>
      {isOpen ? (
        <ul className="object-id-combobox-menu" role="listbox">
          {matches.length ? (
            matches.map((option, index) => (
              <li key={option.objectId}>
                <button
                  type="button"
                  role="option"
                  aria-selected={option.objectId === props.value}
                  className={`object-id-combobox-option${
                    index === activeIndex ? " is-active" : ""
                  }`}
                  onClick={() => choose(option.objectId)}
                >
                  <span className="object-id-combobox-id">{option.objectId}</span>
                  {option.className ? (
                    <span className="object-id-combobox-class">{option.className}</span>
                  ) : null}
                  <span className="object-id-combobox-count">
                    {option.count} view{option.count === 1 ? "" : "s"}
                  </span>
                </button>
              </li>
            ))
          ) : (
            <li className="object-id-combobox-empty">
              {props.options.length
                ? `No saved object ID contains “${typedQuery?.trim()}”.`
                : "No object has been saved yet — this will be the first."}
            </li>
          )}
        </ul>
      ) : null}
    </div>
  );
}

export default ObjectIdCombobox;
