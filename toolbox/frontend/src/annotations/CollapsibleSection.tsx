import { ReactNode, useEffect, useId, useState } from "react";

type Props = {
  title: string;
  summary?: string;
  defaultOpen?: boolean;
  forceOpen?: boolean;
  sectionClassName?: string;
  /** Shows a small dismiss control next to the collapse toggle when provided. */
  onDismiss?: () => void;
  dismissLabel?: string;
  children: ReactNode;
};

function CollapsibleSection(props: Props) {
  const [open, setOpen] = useState(props.defaultOpen ?? true);
  const bodyId = useId();

  useEffect(() => {
    if (props.forceOpen) {
      setOpen(true);
    }
  }, [props.forceOpen]);

  return (
    <section
      className={`${props.sectionClassName ?? "annotation-block"} collapsible-section${
        open ? " is-open" : ""
      }`}
    >
      <div className="collapsible-header-row">
        <button
          type="button"
          className="collapsible-header"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen((value) => !value)}
        >
          <h3>{props.title}</h3>
          {props.summary ? (
            <span className="collapsible-summary">{props.summary}</span>
          ) : null}
          <span className="collapsible-chevron" aria-hidden="true">
            {open ? "-" : "+"}
          </span>
        </button>
        {props.onDismiss ? (
          <button
            type="button"
            className="collapsible-dismiss"
            aria-label={props.dismissLabel ?? `Discard ${props.title}`}
            title={props.dismissLabel ?? `Discard ${props.title}`}
            onClick={props.onDismiss}
          >
            ×
          </button>
        ) : null}
      </div>
      {open ? (
        <div id={bodyId} className="collapsible-body">
          {props.children}
        </div>
      ) : null}
    </section>
  );
}

export default CollapsibleSection;
