import { ReactNode } from "react";

type Props = {
  variant: "info" | "error";
  onDismiss: () => void;
  children: ReactNode;
};

function DismissibleAlert(props: Props) {
  const className = props.variant === "info" ? "info-box" : "error-box";
  const role = props.variant === "error" ? "alert" : "status";

  return (
    <div className={`${className} dismissible-alert`} role={role}>
      <p className="dismissible-alert-message">{props.children}</p>
      <button
        type="button"
        className="dismissible-alert-close"
        onClick={props.onDismiss}
        aria-label="Dismiss message"
      >
        ×
      </button>
    </div>
  );
}

export default DismissibleAlert;
